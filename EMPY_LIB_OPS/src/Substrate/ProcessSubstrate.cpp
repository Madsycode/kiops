#include <eOps/Substrate/Substrate.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/Process.h>
#include "Detail.h"
#include <map>
#include <mutex>
#include <cstdlib>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#include <sys/stat.h>
#endif

namespace eOps::Substrate
{
    namespace
    {
        constexpr size_t MAX_CHILDREN = 1024;
        constexpr auto RETIRE_WAIT = std::chrono::milliseconds(5000);

        struct Child
        {
            std::unique_ptr<Process> m_Process;
            InstanceStatus m_Status;
            std::filesystem::path m_Descriptor;
            std::optional<int32_t> m_Exit;

            // collects the exit code once the child has ended
            bool Running()
            {
                if (!m_Exit)
                {
                    m_Exit = m_Process->Wait(std::chrono::milliseconds(0));
                }
                return !m_Exit.has_value();
            }
        };

        std::filesystem::path Environment(const char *name)
        {
            const char *value = std::getenv(name);
            return value && *value ? std::filesystem::path(value) : std::filesystem::path();
        }

        // per-user private state: %LOCALAPPDATA% on Windows, $XDG_STATE_HOME or ~/.local/state elsewhere; the
        // shared system temp directory (/tmp) is never the default, since other local users could pre-create it
        std::filesystem::path DefaultStateDirectory()
        {
#ifdef _WIN32
            std::filesystem::path base = Environment("LOCALAPPDATA");
            if (base.empty())
            {
                base = std::filesystem::temp_directory_path(); // per-user on Windows
            }
#else
            std::filesystem::path base = Environment("XDG_STATE_HOME");
            if (base.empty())
            {
                std::filesystem::path home = Environment("HOME");
                if (home.empty())
                {
                    throw std::runtime_error("process substrate: set state_directory (no $XDG_STATE_HOME or $HOME)");
                }
                base = home / ".local" / "state";
            }
#endif
            return base / "EmpyMLOps" / "process";
        }

        // creates `directory` and, on POSIX, refuses one that another user owns or others may write to (descriptors
        // and logs written there must not be replaceable by other local users)
        void PrepareDirectory(const std::filesystem::path &directory)
        {
            std::filesystem::create_directories(directory);
#ifndef _WIN32
            struct stat info{};
            if (::lstat(directory.c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
            {
                throw std::runtime_error("process substrate: " + directory.string() + " is not a directory");
            }
            if (info.st_uid != ::geteuid())
            {
                throw std::runtime_error("process substrate: " + directory.string() +
                                         " is owned by another user; choose a private state_directory/log_directory");
            }
            if (info.st_mode & (S_IWGRP | S_IWOTH))
            {
                if (::chmod(directory.c_str(), info.st_mode & ~(S_IWGRP | S_IWOTH)) != 0)
                {
                    throw std::runtime_error("process substrate: " + directory.string() +
                                             " is writable by other users and cannot be restricted");
                }
            }
#endif
        }
    } // namespace

    struct ProcessSubstrate::Impl
    {
        ProcessSubstrateOptions m_Options;
        // guards the child table only: killing and reaping a retired child happens outside it, so List (the
        // harness status) never waits for a child to stop
        std::mutex m_Mutex;
        std::map<std::string, Child> m_Children;
        std::map<std::string, InstanceStatus> m_Stopping; // instances a Retire is stopping
    };

    ProcessSubstrate::ProcessSubstrate(ProcessSubstrateOptions options) : m_Impl(std::make_unique<Impl>())
    {
        if (options.m_Executable.empty())
        {
            options.m_Executable = DefaultWorkloadExecutable();
        }
        if (options.m_StateDirectory.empty())
        {
            options.m_StateDirectory = DefaultStateDirectory();
        }
        if (options.m_LogDirectory.empty())
        {
            options.m_LogDirectory = options.m_StateDirectory;
        }
        options.m_StateDirectory = std::filesystem::absolute(options.m_StateDirectory).lexically_normal();
        options.m_LogDirectory = std::filesystem::absolute(options.m_LogDirectory).lexically_normal();
        m_Impl->m_Options = std::move(options);
    }

    ProcessSubstrate::~ProcessSubstrate()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        for (auto &[instance, child] : m_Impl->m_Children)
        {
            child.m_Process->Kill();
            child.m_Process->Wait(RETIRE_WAIT);
        }
        m_Impl->m_Children.clear();
    }

    std::string ProcessSubstrate::Name() const
    {
        return "process";
    }

    EnactResult ProcessSubstrate::Deploy(const DeploymentSpec &spec)
    {
        DeploymentSpec resolved;
        if (std::optional<std::string> error = Detail::Resolve(spec, resolved))
        {
            return Detail::Failure(resolved.m_Instance, *error);
        }
        const std::string &instance = resolved.m_Instance;
        const ProcessSubstrateOptions &options = m_Impl->m_Options;
        std::lock_guard lock(m_Impl->m_Mutex);
        if (m_Impl->m_Stopping.count(instance))
        {
            return Detail::Failure(instance, "instance '" + instance + "' is still stopping");
        }
        auto existing = m_Impl->m_Children.find(instance);
        if (existing != m_Impl->m_Children.end())
        {
            if (existing->second.Running())
            {
                return Detail::Failure(instance, "instance '" + instance + "' is already running");
            }
            m_Impl->m_Children.erase(existing); // an exited instance is replaced
        }
        if (m_Impl->m_Children.size() >= MAX_CHILDREN)
        {
            return Detail::Failure(instance, "process substrate instance limit reached");
        }

        std::filesystem::path descriptor = options.m_StateDirectory / (instance + ".descriptor.yaml");
        std::filesystem::path log = options.m_LogDirectory / (instance + ".log");
        ProcessOptions launch;
        try
        {
            PrepareDirectory(options.m_StateDirectory);
            PrepareDirectory(options.m_LogDirectory);
            WriteFileAtomic(descriptor, Descriptor::EmitDescriptor(resolved.m_Descriptor));
        }
        catch (const std::exception &error)
        {
            return Detail::Failure(instance, error.what());
        }
        launch.m_Program = options.m_Executable;
        launch.m_Args = WorkloadArgs(resolved, descriptor.string());
        if (!options.m_ClientConfig.empty())
        {
            launch.m_Args.push_back("--config");
            launch.m_Args.push_back(options.m_ClientConfig.string());
        }
        launch.m_Environment = options.m_Environment;
        launch.m_StdoutFile = log;
        launch.m_StderrFile = log;

        auto process = std::make_unique<Process>();
        std::string error;
        if (!process->Start(launch, &error))
        {
            std::error_code ignored;
            std::filesystem::remove(descriptor, ignored);
            return Detail::Failure(instance, "cannot launch " + options.m_Executable.string() + ": " + error);
        }
        EnactResult result;
        result.m_Ok = true;
        result.m_Instance = instance;
        result.m_Detail = "pid " + std::to_string(process->Pid());
        result.m_Artifacts = {descriptor.string(), log.string()};
        Child child;
        child.m_Process = std::move(process);
        child.m_Status = InstanceStatus{instance, resolved.m_Service, resolved.m_Role, resolved.m_Node, "running"};
        child.m_Descriptor = descriptor;
        m_Impl->m_Children.emplace(instance, std::move(child));
        return result;
    }

    EnactResult ProcessSubstrate::Retire(const std::string &instance)
    {
        // take the child out of the table, stop it without the lock, then commit
        std::map<std::string, Child>::node_type node;
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            auto found = m_Impl->m_Children.find(instance);
            if (found == m_Impl->m_Children.end())
            {
                return Detail::Failure(instance, "unknown instance '" + instance + "'");
            }
            InstanceStatus stopping = found->second.m_Status;
            stopping.m_State = "stopping";
            m_Impl->m_Stopping[instance] = std::move(stopping);
            node = m_Impl->m_Children.extract(found);
        }
        Child &child = node.mapped();
        bool running = child.Running();
        if (running)
        {
            child.m_Process->Kill();
            child.m_Exit = child.m_Process->Wait(RETIRE_WAIT);
        }
        EnactResult result;
        result.m_Instance = instance;
        std::lock_guard lock(m_Impl->m_Mutex);
        m_Impl->m_Stopping.erase(instance);
        if (!child.m_Exit)
        {
            result.m_Detail = "instance '" + instance + "' did not stop";
            m_Impl->m_Children.insert(std::move(node)); // still tracked, so a later Retire can try again
            return result;
        }
        result.m_Ok = true;
        result.m_Detail =
            std::string(running ? "killed" : "already exited") + " (exit code " + std::to_string(*child.m_Exit) + ")";
        std::error_code ignored;
        std::filesystem::remove(child.m_Descriptor, ignored);
        return result;
    }

    std::vector<InstanceStatus> ProcessSubstrate::List()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        std::vector<InstanceStatus> statuses;
        for (auto &[instance, child] : m_Impl->m_Children)
        {
            InstanceStatus status = child.m_Status;
            status.m_State = child.Running() ? "running" : "exited";
            statuses.push_back(std::move(status));
        }
        for (const auto &[instance, status] : m_Impl->m_Stopping)
        {
            statuses.push_back(status);
        }
        return statuses;
    }
} // namespace eOps::Substrate
