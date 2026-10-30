#pragma once
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <eCore/Common/Globals.h>
#ifndef _WIN32
#include <spawn.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/wait.h>
extern char **environ;
#endif

// child-process helper for executable tests: launches a program with an explicit, restricted
// environment (so DLL/so resolution cannot fall back to developer directories), redirects its
// output to a file, and waits with a timeout

namespace eTest
{
    class ChildProcess
    {
    public:
        ChildProcess() = default;
        ChildProcess(const ChildProcess &) = delete;
        ChildProcess &operator=(const ChildProcess &) = delete;

        ~ChildProcess()
        {
            Kill();
        }

        // `searchPath` replaces PATH (Windows) / LD_LIBRARY_PATH+PATH (POSIX) for the child
        bool Start(const std::filesystem::path &program, const std::vector<std::string> &args,
                   const std::filesystem::path &workingDirectory, const std::string &searchPath,
                   const std::filesystem::path &outputFile)
        {
            m_Output = outputFile;
#ifdef _WIN32
            std::wstring command = L"\"" + program.wstring() + L"\"";
            for (const std::string &arg : args)
            {
                command += L" \"" + std::filesystem::path(arg).wstring() + L"\"";
            }
            // environment block: only what a clean system offers plus the restricted PATH
            std::wstring environment;
            auto append = [&environment](const std::wstring &entry)
            {
                environment += entry;
                environment.push_back(L'\0');
            };
            append(L"PATH=" + std::filesystem::path(searchPath).wstring());
            for (const wchar_t *name : {L"SystemRoot", L"SystemDrive", L"TEMP", L"TMP", L"USERPROFILE", L"APPDATA",
                                        L"LOCALAPPDATA", L"COMPUTERNAME"})
            {
                wchar_t value[32768];
                DWORD size = GetEnvironmentVariableW(name, value, 32768);
                if (size > 0 && size < 32768)
                {
                    append(std::wstring(name) + L"=" + value);
                }
            }
            environment.push_back(L'\0');

            SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            HANDLE output = CreateFileW(outputFile.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                        &inherit, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (output == INVALID_HANDLE_VALUE)
            {
                return false;
            }
            STARTUPINFOW startup{};
            startup.cb = sizeof(startup);
            startup.dwFlags = STARTF_USESTDHANDLES;
            startup.hStdOutput = output;
            startup.hStdError = output;
            startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            PROCESS_INFORMATION info{};
            std::wstring mutableCommand = command;
            BOOL created = CreateProcessW(program.wstring().c_str(), mutableCommand.data(), nullptr, nullptr, TRUE,
                                          CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, environment.data(),
                                          workingDirectory.wstring().c_str(), &startup, &info);
            CloseHandle(output);
            if (!created)
            {
                return false;
            }
            CloseHandle(info.hThread);
            m_Process = info.hProcess;
            return true;
#else
            std::vector<std::string> storage;
            storage.push_back(program.string());
            for (const std::string &arg : args)
            {
                storage.push_back(arg);
            }
            std::vector<char *> argv;
            for (std::string &s : storage)
            {
                argv.push_back(s.data());
            }
            argv.push_back(nullptr);
            std::string pathEntry = "PATH=" + searchPath;
            std::string libraryEntry = "LD_LIBRARY_PATH=" + searchPath;
            std::vector<char *> envp = {pathEntry.data(), libraryEntry.data(), nullptr};
            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            posix_spawn_file_actions_addopen(&actions, 1, outputFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            posix_spawn_file_actions_adddup2(&actions, 1, 2);
            std::filesystem::path previous = std::filesystem::current_path();
            std::filesystem::current_path(workingDirectory);
            int status = posix_spawn(&m_Pid, program.c_str(), &actions, nullptr, argv.data(), envp.data());
            std::filesystem::current_path(previous);
            posix_spawn_file_actions_destroy(&actions);
            return status == 0;
#endif
        }

        // wait for exit; returns the exit code, or -1 on timeout
        int32_t Wait(std::chrono::milliseconds timeout)
        {
#ifdef _WIN32
            if (!m_Process)
            {
                return -1;
            }
            if (WaitForSingleObject(m_Process, static_cast<DWORD>(timeout.count())) != WAIT_OBJECT_0)
            {
                return -1;
            }
            DWORD code = 0;
            GetExitCodeProcess(m_Process, &code);
            CloseHandle(m_Process);
            m_Process = nullptr;
            return static_cast<int32_t>(code);
#else
            if (m_Pid <= 0)
            {
                return -1;
            }
            auto deadline = std::chrono::steady_clock::now() + timeout;
            int status = 0;
            while (std::chrono::steady_clock::now() < deadline)
            {
                pid_t done = waitpid(m_Pid, &status, WNOHANG);
                if (done == m_Pid)
                {
                    m_Pid = -1;
                    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            return -1;
#endif
        }

        bool Running() const
        {
#ifdef _WIN32
            return m_Process && WaitForSingleObject(m_Process, 0) == WAIT_TIMEOUT;
#else
            return m_Pid > 0 && waitpid(m_Pid, nullptr, WNOHANG) == 0;
#endif
        }

        void Kill()
        {
#ifdef _WIN32
            if (m_Process)
            {
                TerminateProcess(m_Process, 1);
                WaitForSingleObject(m_Process, 5000);
                CloseHandle(m_Process);
                m_Process = nullptr;
            }
#else
            if (m_Pid > 0)
            {
                kill(m_Pid, SIGKILL);
                waitpid(m_Pid, nullptr, 0);
                m_Pid = -1;
            }
#endif
        }

        std::string Output() const
        {
            std::ifstream file(m_Output, std::ios::binary);
            std::ostringstream text;
            text << file.rdbuf();
            return text.str();
        }

    private:
#ifdef _WIN32
        HANDLE m_Process = nullptr;
#else
        pid_t m_Pid = -1;
#endif
        std::filesystem::path m_Output;
    };

    // the directory holding the running test executable (also holds the staged binaries)
    inline std::filesystem::path BinaryDirectory()
    {
#ifdef _WIN32
        wchar_t buffer[MAX_PATH];
        DWORD size = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
        return std::filesystem::path(std::wstring(buffer, size)).parent_path();
#else
        return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
    }

    // search path containing only the binary directory and the operating system
    inline std::string RestrictedSearchPath(const std::filesystem::path &directory = BinaryDirectory())
    {
        std::string path = directory.string();
#ifdef _WIN32
        wchar_t system[MAX_PATH];
        UINT size = GetSystemDirectoryW(system, MAX_PATH);
        path += ";" + std::filesystem::path(std::wstring(system, size)).string();
#else
        path += ":/usr/bin:/bin";
#endif
        return path;
    }
} // namespace eTest
