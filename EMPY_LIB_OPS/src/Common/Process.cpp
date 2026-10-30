#include <eOps/Common/Process.h>
#include <atomic>
#include <format>
#include <thread>
#include <algorithm>
#ifdef _WIN32
#include <map>
#else
#include <spawn.h>
#include <poll.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <sys/wait.h>
extern char **environ;
#endif

namespace eOps
{
    namespace
    {
        // take the next complete line out of `buffer`
        std::optional<std::string> TakeLine(std::string &buffer, size_t maxLine, bool eof)
        {
            size_t end = buffer.find('\n');
            if (end == std::string::npos)
            {
                if (buffer.size() >= maxLine)
                {
                    std::string line = buffer.substr(0, maxLine);
                    buffer.erase(0, maxLine);
                    return line;
                }
                if (eof && !buffer.empty())
                {
                    std::string line = std::move(buffer);
                    buffer.clear();
                    return line;
                }
                return std::nullopt;
            }
            std::string line = buffer.substr(0, std::min(end, maxLine));
            buffer.erase(0, end + 1);
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            return line;
        }
    } // namespace

#ifdef _WIN32
    namespace
    {
        std::wstring Widen(std::string_view text)
        {
            if (text.empty())
            {
                return {};
            }
            int size = MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
            std::wstring out(static_cast<size_t>(size), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), size);
            return out;
        }

        // quote one argument so CommandLineToArgvW / the C runtime recover it exactly
        std::wstring QuoteArgument(const std::wstring &argument)
        {
            if (!argument.empty() && argument.find_first_of(L" \t\n\v\"") == std::wstring::npos)
            {
                return argument;
            }
            std::wstring out = L"\"";
            for (auto it = argument.begin();; ++it)
            {
                size_t backslashes = 0;
                while (it != argument.end() && *it == L'\\')
                {
                    ++it;
                    ++backslashes;
                }
                if (it == argument.end())
                {
                    out.append(backslashes * 2, L'\\');
                    break;
                }
                if (*it == L'"')
                {
                    out.append(backslashes * 2 + 1, L'\\');
                }
                else
                {
                    out.append(backslashes, L'\\');
                }
                out.push_back(*it);
            }
            out.push_back(L'"');
            return out;
        }

        struct CaseLess
        {
            bool operator()(const std::wstring &a, const std::wstring &b) const
            {
                return CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(),
                                            static_cast<int>(b.size()), TRUE) == CSTR_LESS_THAN;
            }
        };

        // inherited environment with overrides, as a sorted double-NUL-terminated block
        std::wstring EnvironmentBlock(const std::vector<std::pair<std::string, std::string>> &overrides)
        {
            std::map<std::wstring, std::wstring, CaseLess> variables;
            if (wchar_t *strings = GetEnvironmentStringsW())
            {
                for (const wchar_t *entry = strings; *entry; entry += wcslen(entry) + 1)
                {
                    std::wstring text(entry);
                    // hidden per-drive variables ("=C:=C:\\") start with '='
                    size_t equals = text.find(L'=', 1);
                    if (equals != std::wstring::npos)
                    {
                        variables[text.substr(0, equals)] = text.substr(equals + 1);
                    }
                }
                FreeEnvironmentStringsW(strings);
            }
            for (const auto &[name, value] : overrides)
            {
                variables[Widen(name)] = Widen(value);
            }
            std::wstring block;
            for (const auto &[name, value] : variables)
            {
                block += name;
                block.push_back(L'=');
                block += value;
                block.push_back(L'\0');
            }
            block.push_back(L'\0');
            return block;
        }

        std::string LastError(const char *what)
        {
            return std::format("{} failed (error {})", what, GetLastError());
        }

        struct Handle
        {
            HANDLE m_Value = nullptr;
            Handle() = default;
            explicit Handle(HANDLE value) : m_Value(value == INVALID_HANDLE_VALUE ? nullptr : value)
            {
            }
            Handle(const Handle &) = delete;
            Handle &operator=(const Handle &) = delete;
            Handle(Handle &&other) noexcept : m_Value(other.Release())
            {
            }
            Handle &operator=(Handle &&other) noexcept
            {
                if (this != &other)
                {
                    Reset(other.Release());
                }
                return *this;
            }
            ~Handle()
            {
                Reset();
            }
            void Reset(HANDLE value = nullptr)
            {
                if (m_Value)
                {
                    CloseHandle(m_Value);
                }
                m_Value = value == INVALID_HANDLE_VALUE ? nullptr : value;
            }
            HANDLE Release()
            {
                return std::exchange(m_Value, nullptr);
            }
            explicit operator bool() const
            {
                return m_Value != nullptr;
            }
        };

        std::atomic<uint64_t> s_PipeCounter{0};
    } // namespace

    struct Process::Impl
    {
        HANDLE m_Process = nullptr;
        HANDLE m_Job = nullptr;
        HANDLE m_Stdin = nullptr;
        HANDLE m_Stdout = nullptr;
        HANDLE m_Event = nullptr;
        HANDLE m_WriteEvent = nullptr;
        OVERLAPPED m_Overlapped{};
        std::vector<char> m_Chunk = std::vector<char>(64 * 1024);
        std::string m_Buffer;
        bool m_Pending = false;
        bool m_Eof = false;
        bool m_Started = false;
        uint64_t m_Pid = 0;
        size_t m_MaxLine = 4 * 1024 * 1024;
        std::optional<int32_t> m_Exit;

        ~Impl()
        {
            Kill();
            if (m_Pending)
            {
                DWORD ignored = 0;
                CancelIoEx(m_Stdout, &m_Overlapped);
                GetOverlappedResult(m_Stdout, &m_Overlapped, &ignored, TRUE);
            }
            for (HANDLE handle : {m_Process, m_Job, m_Stdin, m_Stdout, m_Event, m_WriteEvent})
            {
                if (handle)
                {
                    CloseHandle(handle);
                }
            }
        }

        void Kill()
        {
            if (!m_Process || m_Exit)
            {
                return;
            }
            if (WaitForSingleObject(m_Process, 0) == WAIT_TIMEOUT)
            {
                if (!m_Job || !TerminateJobObject(m_Job, 1))
                {
                    TerminateProcess(m_Process, 1);
                }
                WaitForSingleObject(m_Process, 5000);
            }
            DWORD code = 0;
            if (GetExitCodeProcess(m_Process, &code) && code != STILL_ACTIVE)
            {
                m_Exit = static_cast<int32_t>(code);
            }
        }

        bool Start(const ProcessOptions &options, std::string &error)
        {
            SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
            auto nullDevice = [&](DWORD access)
            {
                return Handle(CreateFileW(L"NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING,
                                          FILE_ATTRIBUTE_NORMAL, nullptr));
            };
            auto appendFile = [&](const std::filesystem::path &path)
            {
                return Handle(CreateFileW(path.wstring().c_str(), FILE_APPEND_DATA | SYNCHRONIZE,
                                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &inherit, OPEN_ALWAYS,
                                          FILE_ATTRIBUTE_NORMAL, nullptr));
            };

            // stdin: overlapped named pipe (parent end not inheritable) so Write can honour a timeout, or null
            Handle childIn, parentIn, writeEvent;
            if (options.m_PipeStdin)
            {
                std::wstring name = Widen(std::format("\\\\.\\pipe\\empy-ops-in-{}-{}-{}", GetCurrentProcessId(),
                                                      s_PipeCounter++, GetTickCount64()));
                parentIn.Reset(CreateNamedPipeW(
                    name.c_str(), PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                    PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 1 << 16, 1 << 16, 0, nullptr));
                if (!parentIn)
                {
                    error = LastError("CreateNamedPipe");
                    return false;
                }
                childIn.Reset(CreateFileW(name.c_str(), GENERIC_READ, 0, &inherit, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                                          nullptr));
                writeEvent.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
                if (!childIn || !writeEvent)
                {
                    error = LastError("stdin pipe client");
                    return false;
                }
            }
            else
            {
                childIn = nullDevice(GENERIC_READ);
            }

            // stdout: overlapped named pipe so ReadLine can honour its timeout, a file, or null
            Handle childOut, parentOut, event;
            if (options.m_PipeStdout)
            {
                std::wstring name = Widen(std::format("\\\\.\\pipe\\empy-ops-{}-{}-{}", GetCurrentProcessId(),
                                                      s_PipeCounter++, GetTickCount64()));
                parentOut.Reset(CreateNamedPipeW(
                    name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                    PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 1 << 16, 1 << 16,
                    0, nullptr));
                if (!parentOut)
                {
                    error = LastError("CreateNamedPipe");
                    return false;
                }
                childOut.Reset(CreateFileW(name.c_str(), GENERIC_WRITE, 0, &inherit, OPEN_EXISTING,
                                           FILE_ATTRIBUTE_NORMAL, nullptr));
                event.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
                if (!childOut || !event)
                {
                    error = LastError("pipe client");
                    return false;
                }
            }
            else if (!options.m_StdoutFile.empty())
            {
                childOut = appendFile(options.m_StdoutFile);
            }
            else
            {
                childOut = nullDevice(GENERIC_WRITE);
            }
            if (!childIn || !childOut)
            {
                error = LastError("opening child stdio");
                return false;
            }

            // stderr: shares the stdout file when both name the same path
            Handle childErr;
            bool shared =
                !options.m_PipeStdout && !options.m_StderrFile.empty() && options.m_StderrFile == options.m_StdoutFile;
            if (!shared)
            {
                childErr = options.m_StderrFile.empty() ? nullDevice(GENERIC_WRITE) : appendFile(options.m_StderrFile);
                if (!childErr)
                {
                    error = LastError("opening child stderr");
                    return false;
                }
            }
            HANDLE errorHandle = shared ? childOut.m_Value : childErr.m_Value;

            // only the child ends are inherited, even when other processes start concurrently
            HANDLE inherited[3] = {childIn.m_Value, childOut.m_Value, errorHandle};
            size_t inheritedCount = shared ? 2 : 3;
            SIZE_T attributeSize = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeSize);
            std::vector<uint8_t> attributeStorage(attributeSize);
            auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeStorage.data());
            if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attributeSize))
            {
                error = LastError("InitializeProcThreadAttributeList");
                return false;
            }
            struct AttributeGuard
            {
                LPPROC_THREAD_ATTRIBUTE_LIST m_List;
                ~AttributeGuard()
                {
                    DeleteProcThreadAttributeList(m_List);
                }
            } attributeGuard{attributes};
            if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                           inheritedCount * sizeof(HANDLE), nullptr, nullptr))
            {
                error = LastError("UpdateProcThreadAttribute");
                return false;
            }

            STARTUPINFOEXW startup{};
            startup.StartupInfo.cb = sizeof(startup);
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = childIn.m_Value;
            startup.StartupInfo.hStdOutput = childOut.m_Value;
            startup.StartupInfo.hStdError = errorHandle;
            startup.lpAttributeList = attributes;

            std::wstring program = options.m_Program.wstring();
            std::wstring command = program.find_first_of(L" \t") == std::wstring::npos && !program.empty()
                                       ? program
                                       : L"\"" + program + L"\"";
            for (const std::string &argument : options.m_Args)
            {
                command += L" " + QuoteArgument(Widen(argument));
            }
            std::wstring environment;
            if (!options.m_Environment.empty())
            {
                environment = EnvironmentBlock(options.m_Environment);
            }
            std::wstring directory = options.m_WorkingDirectory.wstring();

            // a kill-on-close job also reaps grandchildren when the owner exits or kills
            Handle job(CreateJobObjectW(nullptr, nullptr));
            if (job)
            {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(job.m_Value, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
            }

            PROCESS_INFORMATION info{};
            DWORD flags =
                EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW | CREATE_SUSPENDED;
            if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, flags,
                                environment.empty() ? nullptr : environment.data(),
                                directory.empty() ? nullptr : directory.c_str(), &startup.StartupInfo, &info))
            {
                error = LastError(("CreateProcess '" + options.m_Program.string() + "'").c_str());
                return false;
            }
            if (job && !AssignProcessToJobObject(job.m_Value, info.hProcess))
            {
                job.Reset();
            }
            ResumeThread(info.hThread);
            CloseHandle(info.hThread);
            m_Process = info.hProcess;
            m_Pid = info.dwProcessId;
            m_Job = job.Release();
            m_Stdin = parentIn.Release();
            m_Stdout = parentOut.Release();
            m_Event = event.Release();
            m_WriteEvent = writeEvent.Release();
            m_MaxLine = std::max<size_t>(options.m_MaxLineBytes, 1);
            return true;
        }

        bool Running()
        {
            return m_Process && !m_Exit && WaitForSingleObject(m_Process, 0) == WAIT_TIMEOUT;
        }

        std::optional<int32_t> Wait(std::chrono::milliseconds timeout)
        {
            if (!m_Process || m_Exit)
            {
                return m_Exit;
            }
            auto milliseconds = static_cast<DWORD>(std::clamp<int64_t>(timeout.count(), 0, INFINITE - 1));
            if (WaitForSingleObject(m_Process, milliseconds) != WAIT_OBJECT_0)
            {
                return std::nullopt;
            }
            DWORD code = 0;
            GetExitCodeProcess(m_Process, &code);
            m_Exit = static_cast<int32_t>(code);
            return m_Exit;
        }

        // overlapped writes, each awaited (or cancelled at the deadline) before the next one starts
        bool Write(std::string_view bytes, std::optional<std::chrono::milliseconds> timeout)
        {
            if (!m_Stdin)
            {
                return false;
            }
            auto deadline = std::chrono::steady_clock::now() + timeout.value_or(std::chrono::milliseconds(0));
            while (!bytes.empty())
            {
                DWORD written = 0;
                DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes.size(), 1 << 20));
                OVERLAPPED overlapped{};
                overlapped.hEvent = m_WriteEvent;
                ResetEvent(m_WriteEvent);
                if (!::WriteFile(m_Stdin, bytes.data(), chunk, nullptr, &overlapped))
                {
                    if (GetLastError() != ERROR_IO_PENDING)
                    {
                        return false;
                    }
                    DWORD wait = INFINITE;
                    if (timeout)
                    {
                        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now());
                        wait = static_cast<DWORD>(std::clamp<int64_t>(remaining.count(), 0, INFINITE - 1));
                    }
                    if (WaitForSingleObject(m_WriteEvent, wait) != WAIT_OBJECT_0)
                    {
                        // the child stopped reading: cancel and wait until the cancellation completed
                        CancelIoEx(m_Stdin, &overlapped);
                        GetOverlappedResult(m_Stdin, &overlapped, &written, TRUE);
                        return false;
                    }
                }
                if (!GetOverlappedResult(m_Stdin, &overlapped, &written, FALSE) || written == 0)
                {
                    return false;
                }
                bytes.remove_prefix(written);
            }
            return true;
        }

        void CloseStdin()
        {
            if (m_Stdin)
            {
                CloseHandle(m_Stdin);
                m_Stdin = nullptr;
            }
        }

        // completes the outstanding read; a failed read (broken pipe) is the end of the stream
        void Collect(DWORD bytes, bool success)
        {
            m_Pending = false;
            if (!success)
            {
                m_Eof = true;
                return;
            }
            m_Buffer.append(m_Chunk.data(), bytes);
        }

        std::optional<std::string> ReadLine(std::chrono::milliseconds timeout)
        {
            auto deadline = std::chrono::steady_clock::now() + timeout;
            while (true)
            {
                if (auto line = TakeLine(m_Buffer, m_MaxLine, m_Eof || !m_Stdout))
                {
                    return line;
                }
                if (m_Eof || !m_Stdout)
                {
                    return std::nullopt;
                }
                if (!m_Pending)
                {
                    ResetEvent(m_Event);
                    m_Overlapped = OVERLAPPED{};
                    m_Overlapped.hEvent = m_Event;
                    if (::ReadFile(m_Stdout, m_Chunk.data(), static_cast<DWORD>(m_Chunk.size()), nullptr,
                                   &m_Overlapped))
                    {
                        // sequenced: GetOverlappedResult fills `bytes` before Collect reads it
                        DWORD bytes = 0;
                        bool success = GetOverlappedResult(m_Stdout, &m_Overlapped, &bytes, FALSE) != 0;
                        Collect(bytes, success);
                        continue;
                    }
                    if (GetLastError() != ERROR_IO_PENDING)
                    {
                        m_Eof = true;
                        continue;
                    }
                    m_Pending = true;
                }
                auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                auto wait = static_cast<DWORD>(std::clamp<int64_t>(remaining.count(), 0, INFINITE - 1));
                if (WaitForSingleObject(m_Event, wait) != WAIT_OBJECT_0)
                {
                    return std::nullopt;
                }
                DWORD bytes = 0;
                bool success = GetOverlappedResult(m_Stdout, &m_Overlapped, &bytes, FALSE) != 0;
                Collect(bytes, success);
            }
        }
    };
#else
    namespace
    {
        std::mutex s_DirectoryMutex;

        void CloseFd(int &fd)
        {
            if (fd >= 0)
            {
                ::close(fd);
                fd = -1;
            }
        }

        bool MakePipe(int fds[2])
        {
            if (::pipe(fds) != 0)
            {
                return false;
            }
            for (int i = 0; i < 2; ++i)
            {
                ::fcntl(fds[i], F_SETFD, FD_CLOEXEC);
            }
            return true;
        }
    } // namespace

    struct Process::Impl
    {
        pid_t m_Pid = -1;
        int m_Stdin = -1;
        int m_Stdout = -1;
        std::string m_Buffer;
        bool m_Eof = false;
        bool m_Started = false;
        size_t m_MaxLine = 4 * 1024 * 1024;
        std::optional<int32_t> m_Exit;

        ~Impl()
        {
            Kill();
            CloseFd(m_Stdin);
            CloseFd(m_Stdout);
        }

        static int32_t Decode(int status)
        {
            if (WIFEXITED(status))
            {
                return WEXITSTATUS(status);
            }
            if (WIFSIGNALED(status))
            {
                return 128 + WTERMSIG(status);
            }
            return -1;
        }

        void Kill()
        {
            if (m_Pid <= 0 || m_Exit)
            {
                return;
            }
            ::kill(m_Pid, SIGKILL);
            int status = 0;
            if (::waitpid(m_Pid, &status, 0) == m_Pid)
            {
                m_Exit = Decode(status);
            }
        }

        bool Start(const ProcessOptions &options, std::string &error)
        {
            int input[2] = {-1, -1};
            int output[2] = {-1, -1};
            auto cleanup = [&]
            {
                for (int *fd : {&input[0], &input[1], &output[0], &output[1]})
                {
                    CloseFd(*fd);
                }
            };
            if ((options.m_PipeStdin && !MakePipe(input)) || (options.m_PipeStdout && !MakePipe(output)))
            {
                error = std::string("pipe failed: ") + std::strerror(errno);
                cleanup();
                return false;
            }
            posix_spawn_file_actions_t actions;
            posix_spawn_file_actions_init(&actions);
            if (options.m_PipeStdin)
            {
                posix_spawn_file_actions_adddup2(&actions, input[0], 0);
            }
            else
            {
                posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
            }
            std::string stdoutPath = options.m_StdoutFile.string();
            std::string stderrPath = options.m_StderrFile.string();
            // output files are never followed through a symbolic link (a planted link must not redirect the
            // child's output into another file of this user)
            constexpr int OUTPUT_FLAGS = O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW;
            if (options.m_PipeStdout)
            {
                posix_spawn_file_actions_adddup2(&actions, output[1], 1);
            }
            else if (stdoutPath.empty())
            {
                posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
            }
            else
            {
                posix_spawn_file_actions_addopen(&actions, 1, stdoutPath.c_str(), OUTPUT_FLAGS, 0600);
            }
            if (!options.m_PipeStdout && !stderrPath.empty() && stderrPath == stdoutPath)
            {
                posix_spawn_file_actions_adddup2(&actions, 1, 2);
            }
            else if (stderrPath.empty())
            {
                posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
            }
            else
            {
                posix_spawn_file_actions_addopen(&actions, 2, stderrPath.c_str(), OUTPUT_FLAGS, 0600);
            }
            std::string directory = options.m_WorkingDirectory.string();
#if (defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))) || defined(__APPLE__)
            bool chdirAction = !directory.empty();
            if (chdirAction)
            {
                posix_spawn_file_actions_addchdir_np(&actions, directory.c_str());
            }
#else
            bool chdirAction = false;
#endif
            std::vector<std::string> storage;
            storage.push_back(options.m_Program.string());
            storage.insert(storage.end(), options.m_Args.begin(), options.m_Args.end());
            std::vector<char *> argv;
            for (std::string &value : storage)
            {
                argv.push_back(value.data());
            }
            argv.push_back(nullptr);
            std::vector<std::string> variables;
            for (char **entry = environ; entry && *entry; ++entry)
            {
                std::string_view text(*entry);
                bool replaced = std::any_of(options.m_Environment.begin(), options.m_Environment.end(),
                                            [&](const auto &item) { return text.starts_with(item.first + "="); });
                if (!replaced)
                {
                    variables.emplace_back(text);
                }
            }
            for (const auto &[name, value] : options.m_Environment)
            {
                variables.push_back(name + "=" + value);
            }
            std::vector<char *> envp;
            for (std::string &value : variables)
            {
                envp.push_back(value.data());
            }
            envp.push_back(nullptr);
            bool search = storage.front().find('/') == std::string::npos;
            int status = 0;
            {
                std::unique_lock lock(s_DirectoryMutex, std::defer_lock);
                std::filesystem::path previous;
                if (!directory.empty() && !chdirAction)
                {
                    lock.lock();
                    previous = std::filesystem::current_path();
                    std::filesystem::current_path(directory);
                }
                status = search ? posix_spawnp(&m_Pid, argv[0], &actions, nullptr, argv.data(), envp.data())
                                : posix_spawn(&m_Pid, argv[0], &actions, nullptr, argv.data(), envp.data());
                if (lock.owns_lock())
                {
                    std::filesystem::current_path(previous);
                }
            }
            posix_spawn_file_actions_destroy(&actions);
            if (status != 0)
            {
                error = "spawn '" + storage.front() + "' failed: " + std::strerror(status);
                m_Pid = -1;
                cleanup();
                return false;
            }
            CloseFd(input[0]);
            CloseFd(output[1]);
            m_Stdin = input[1];
            m_Stdout = output[0];
            if (m_Stdin >= 0)
            {
                // Write polls for room so it can honour a timeout
                ::fcntl(m_Stdin, F_SETFL, ::fcntl(m_Stdin, F_GETFL) | O_NONBLOCK);
            }
#ifdef __APPLE__
            if (m_Stdin >= 0)
            {
                ::fcntl(m_Stdin, F_SETNOSIGPIPE, 1);
            }
#endif
            m_MaxLine = std::max<size_t>(options.m_MaxLineBytes, 1);
            return true;
        }

        bool Running()
        {
            if (m_Pid <= 0 || m_Exit)
            {
                return false;
            }
            int status = 0;
            pid_t done = ::waitpid(m_Pid, &status, WNOHANG);
            if (done == m_Pid)
            {
                m_Exit = Decode(status);
                return false;
            }
            return done == 0;
        }

        std::optional<int32_t> Wait(std::chrono::milliseconds timeout)
        {
            auto deadline = std::chrono::steady_clock::now() + timeout;
            while (Running())
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    return std::nullopt;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            return m_Exit;
        }

        // non-blocking writes; a full pipe is awaited with poll until the deadline (none = forever)
        bool Write(std::string_view bytes, std::optional<std::chrono::milliseconds> timeout)
        {
            if (m_Stdin < 0)
            {
                return false;
            }
#ifdef __linux__
            // a broken pipe must fail the write, not raise SIGPIPE in the owner process
            sigset_t block, previous;
            sigemptyset(&block);
            sigaddset(&block, SIGPIPE);
            pthread_sigmask(SIG_BLOCK, &block, &previous);
#endif
            auto deadline = std::chrono::steady_clock::now() + timeout.value_or(std::chrono::milliseconds(0));
            bool okay = true;
            int failure = 0;
            while (okay && !bytes.empty())
            {
                auto written = ::write(m_Stdin, bytes.data(), bytes.size());
                if (written < 0 && errno == EINTR)
                {
                    continue;
                }
                if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                {
                    int wait = -1;
                    if (timeout)
                    {
                        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now());
                        wait = static_cast<int>(std::clamp<int64_t>(remaining.count(), 0, INT32_MAX));
                    }
                    pollfd descriptor{m_Stdin, POLLOUT, 0};
                    int ready = ::poll(&descriptor, 1, wait);
                    if (ready < 0 && errno == EINTR)
                    {
                        continue;
                    }
                    okay = ready > 0; // 0: the child stopped reading before the deadline
                    continue;         // writable or an error condition: the next write reports it
                }
                okay = written > 0;
                failure = okay ? 0 : errno;
                if (okay)
                {
                    bytes.remove_prefix(static_cast<size_t>(written));
                }
            }
            (void)failure;
#ifdef __linux__
            if (!okay && failure == EPIPE)
            {
                timespec zero{0, 0};
                sigtimedwait(&block, nullptr, &zero);
            }
            pthread_sigmask(SIG_SETMASK, &previous, nullptr);
#endif
            return okay;
        }

        void CloseStdin()
        {
            CloseFd(m_Stdin);
        }

        std::optional<std::string> ReadLine(std::chrono::milliseconds timeout)
        {
            auto deadline = std::chrono::steady_clock::now() + timeout;
            char chunk[64 * 1024];
            while (true)
            {
                if (auto line = TakeLine(m_Buffer, m_MaxLine, m_Eof || m_Stdout < 0))
                {
                    return line;
                }
                if (m_Eof || m_Stdout < 0)
                {
                    return std::nullopt;
                }
                auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                pollfd descriptor{m_Stdout, POLLIN, 0};
                int ready =
                    ::poll(&descriptor, 1, static_cast<int>(std::clamp<int64_t>(remaining.count(), 0, INT32_MAX)));
                if (ready < 0 && errno == EINTR)
                {
                    continue;
                }
                if (ready <= 0)
                {
                    return std::nullopt;
                }
                auto count = ::read(m_Stdout, chunk, sizeof(chunk));
                if (count < 0 && errno == EINTR)
                {
                    continue;
                }
                if (count <= 0)
                {
                    m_Eof = true;
                    continue;
                }
                m_Buffer.append(chunk, static_cast<size_t>(count));
            }
        }
    };
#endif

    Process::Process() : m_Impl(std::make_unique<Impl>())
    {
    }

    Process::~Process() = default;

    bool Process::Start(const ProcessOptions &options, std::string *error)
    {
        std::string message;
        bool started = false;
        if (m_Impl->m_Started)
        {
            message = "process already started";
        }
        else if (options.m_Program.empty())
        {
            message = "no program given";
        }
        else
        {
            started = m_Impl->Start(options, message);
            m_Impl->m_Started = started;
        }
        if (!started && error)
        {
            *error = message;
        }
        return started;
    }

    bool Process::Running()
    {
        return m_Impl->Running();
    }

    std::optional<int32_t> Process::Wait(std::chrono::milliseconds timeout)
    {
        return m_Impl->Wait(timeout);
    }

    void Process::Kill()
    {
        m_Impl->Kill();
    }

    bool Process::Write(std::string_view bytes)
    {
        return m_Impl->Write(bytes, std::nullopt);
    }

    bool Process::Write(std::string_view bytes, std::chrono::milliseconds timeout)
    {
        return m_Impl->Write(bytes, timeout);
    }

    void Process::CloseStdin()
    {
        m_Impl->CloseStdin();
    }

    std::optional<std::string> Process::ReadLine(std::chrono::milliseconds timeout)
    {
        return m_Impl->ReadLine(timeout);
    }

    uint64_t Process::Pid() const noexcept
    {
        return m_Impl->m_Started ? static_cast<uint64_t>(m_Impl->m_Pid) : 0;
    }
} // namespace eOps
