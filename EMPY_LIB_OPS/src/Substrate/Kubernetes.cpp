#include <eOps/Substrate/Substrate.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/Process.h>
#include "Detail.h"
#include <set>
#include <mutex>
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace eOps::Substrate
{
    namespace
    {
        struct CommandResult
        {
            bool m_Ok = false;
            std::string m_Detail;
        };

        // run `command` + `file` without a shell, capturing stdout/stderr through a scratch log
        CommandResult Run(const std::vector<std::string> &command, const std::filesystem::path &file,
                          const std::filesystem::path &log, uint32_t timeoutMs)
        {
            std::error_code ignored;
            std::filesystem::remove(log, ignored);
            ProcessOptions options;
            options.m_Program = command.front();
            options.m_Args.assign(command.begin() + 1, command.end());
            options.m_Args.push_back(file.string());
            options.m_StdoutFile = log;
            options.m_StderrFile = log;
            Process process;
            std::string error;
            CommandResult result;
            if (!process.Start(options, &error))
            {
                result.m_Detail = "cannot run '" + command.front() + "': " + error;
                return result;
            }
            std::optional<int32_t> code = process.Wait(std::chrono::milliseconds(timeoutMs));
            if (!code)
            {
                process.Kill();
                process.Wait(std::chrono::milliseconds(5000));
                result.m_Detail = "'" + command.front() + "' timed out after " + std::to_string(timeoutMs) + " ms";
            }
            else
            {
                std::string output = Detail::Tail(log);
                result.m_Ok = *code == 0;
                result.m_Detail = "'" + command.front() + "' exited with code " + std::to_string(*code) +
                                  (output.empty() ? std::string() : ": " + output);
            }
            std::filesystem::remove(log, ignored);
            return result;
        }

        std::optional<InstanceStatus> ReadMarker(const std::filesystem::path &file)
        {
            std::ifstream stream(file, std::ios::binary);
            std::string line;
            if (!stream || !std::getline(stream, line) || !line.starts_with(Detail::MANIFEST_MARKER))
            {
                return std::nullopt;
            }
            try
            {
                Json marker = Json::Parse(Trim(std::string_view(line).substr(Detail::MANIFEST_MARKER.size())));
                return InstanceStatus{marker.GetString("instance"),
                                      marker.GetString("service"),
                                      marker.GetString("role"),
                                      marker.GetString("node"),
                                      {}};
            }
            catch (const JsonError &)
            {
                return std::nullopt;
            }
        }
    } // namespace

    struct KubernetesSubstrate::Impl
    {
        KubernetesOptions m_Options;
        // guards the manifest directory bookkeeping only: apply/delete commands run outside it, so List (the
        // harness status) never waits for a slow API server
        std::mutex m_Mutex;
        std::set<std::string> m_Applying; // instances whose apply command runs
        std::set<std::string> m_Deleting; // instances whose delete command runs

        std::filesystem::path Manifest(const std::string &instance) const
        {
            return m_Options.m_OutputDirectory / (instance + ".yaml");
        }

        std::filesystem::path Log(const std::string &instance, std::string_view verb) const
        {
            return m_Options.m_OutputDirectory / ("." + instance + "." + std::string(verb) + ".log");
        }

        std::string State() const
        {
            return m_Options.m_ApplyCommand.empty() ? "rendered" : "applied";
        }
    };

    KubernetesSubstrate::KubernetesSubstrate(KubernetesOptions options) : m_Impl(std::make_unique<Impl>())
    {
        if (options.m_OutputDirectory.empty())
        {
            throw std::invalid_argument("kubernetes substrate needs an output directory");
        }
        if (!IsDnsLabel(options.m_Namespace))
        {
            throw std::invalid_argument("kubernetes namespace '" + options.m_Namespace + "' is not a DNS label");
        }
        if (options.m_CommandTimeoutMs == 0)
        {
            throw std::invalid_argument("kubernetes command timeout must be positive");
        }
        options.m_OutputDirectory = std::filesystem::absolute(options.m_OutputDirectory).lexically_normal();
        m_Impl->m_Options = std::move(options);
    }

    KubernetesSubstrate::~KubernetesSubstrate() = default;

    std::string KubernetesSubstrate::Name() const
    {
        return "kubernetes";
    }

    EnactResult KubernetesSubstrate::Deploy(const DeploymentSpec &spec)
    {
        DeploymentSpec resolved;
        if (std::optional<std::string> error = Detail::Resolve(spec, resolved))
        {
            return Detail::Failure(resolved.m_Instance, *error);
        }
        std::string manifests;
        try
        {
            manifests = RenderManifests(resolved, m_Impl->m_Options.m_Namespace, m_Impl->m_Options.m_Image);
        }
        catch (const std::exception &error)
        {
            return Detail::Failure(resolved.m_Instance, error.what());
        }
        const std::string &instance = resolved.m_Instance;
        std::filesystem::path file = m_Impl->Manifest(instance);
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            std::error_code exists;
            if (m_Impl->m_Applying.count(instance) || m_Impl->m_Deleting.count(instance) ||
                std::filesystem::exists(file, exists))
            {
                return Detail::Failure(instance, "instance '" + instance + "' is already deployed");
            }
            try
            {
                WriteFileAtomic(file, manifests);
            }
            catch (const std::exception &error)
            {
                return Detail::Failure(instance, error.what());
            }
            if (!m_Impl->m_Options.m_ApplyCommand.empty())
            {
                m_Impl->m_Applying.insert(instance);
            }
        }
        EnactResult result;
        result.m_Instance = instance;
        result.m_Artifacts.push_back(file.string());
        if (m_Impl->m_Options.m_ApplyCommand.empty())
        {
            result.m_Ok = true;
            result.m_Detail = "rendered " + file.string();
            return result;
        }
        CommandResult applied;
        try
        {
            applied = Run(m_Impl->m_Options.m_ApplyCommand, file, m_Impl->Log(instance, "apply"),
                          m_Impl->m_Options.m_CommandTimeoutMs);
        }
        catch (const std::exception &error)
        {
            applied.m_Detail = error.what();
        }
        std::lock_guard lock(m_Impl->m_Mutex);
        m_Impl->m_Applying.erase(instance);
        result.m_Ok = applied.m_Ok;
        result.m_Detail = applied.m_Detail;
        if (!applied.m_Ok)
        {
            // a failed apply leaves nothing deployed, so the manifest must not be listed
            std::error_code ignored;
            std::filesystem::remove(file, ignored);
            result.m_Artifacts.clear();
        }
        return result;
    }

    EnactResult KubernetesSubstrate::Retire(const std::string &instance)
    {
        if (!IsDnsLabel(instance))
        {
            return Detail::Failure(instance, "instance '" + instance + "' is not a DNS label");
        }
        std::filesystem::path file = m_Impl->Manifest(instance);
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            std::error_code exists;
            if (m_Impl->m_Applying.count(instance) || m_Impl->m_Deleting.count(instance))
            {
                return Detail::Failure(instance, "instance '" + instance + "' has a command in progress");
            }
            if (!std::filesystem::exists(file, exists))
            {
                return Detail::Failure(instance, "unknown instance '" + instance + "'");
            }
            m_Impl->m_Deleting.insert(instance);
        }
        EnactResult result;
        result.m_Instance = instance;
        result.m_Artifacts.push_back(file.string());
        CommandResult deleted;
        deleted.m_Ok = true;
        if (!m_Impl->m_Options.m_DeleteCommand.empty())
        {
            try
            {
                deleted = Run(m_Impl->m_Options.m_DeleteCommand, file, m_Impl->Log(instance, "delete"),
                              m_Impl->m_Options.m_CommandTimeoutMs);
            }
            catch (const std::exception &error)
            {
                deleted.m_Ok = false;
                deleted.m_Detail = error.what();
            }
        }
        std::lock_guard lock(m_Impl->m_Mutex);
        m_Impl->m_Deleting.erase(instance);
        result.m_Detail = deleted.m_Detail;
        if (!deleted.m_Ok)
        {
            return result; // the manifest stays so the retirement can be retried
        }
        std::error_code removed;
        if (!std::filesystem::remove(file, removed) && removed)
        {
            result.m_Detail = "cannot remove " + file.string() + ": " + removed.message();
            return result;
        }
        result.m_Ok = true;
        if (result.m_Detail.empty())
        {
            result.m_Detail = "removed " + file.string();
        }
        return result;
    }

    std::vector<InstanceStatus> KubernetesSubstrate::List()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        std::vector<InstanceStatus> statuses;
        std::error_code error;
        std::filesystem::directory_iterator it(m_Impl->m_Options.m_OutputDirectory, error);
        if (error)
        {
            return statuses; // nothing rendered yet
        }
        for (const std::filesystem::directory_entry &entry : it)
        {
            if (!entry.is_regular_file(error) || entry.path().extension() != ".yaml")
            {
                continue;
            }
            std::optional<InstanceStatus> status = ReadMarker(entry.path());
            if (status && status->m_Instance + ".yaml" == entry.path().filename().string())
            {
                status->m_State = m_Impl->m_Applying.count(status->m_Instance)   ? "applying"
                                  : m_Impl->m_Deleting.count(status->m_Instance) ? "deleting"
                                                                                 : m_Impl->State();
                statuses.push_back(std::move(*status));
            }
        }
        std::sort(statuses.begin(), statuses.end(),
                  [](const InstanceStatus &a, const InstanceStatus &b) { return a.m_Instance < b.m_Instance; });
        return statuses;
    }
} // namespace eOps::Substrate
