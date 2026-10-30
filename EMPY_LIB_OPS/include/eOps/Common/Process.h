#pragma once
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <optional>
#include <filesystem>
#include <string_view>
#include "../API.h"

// cross-platform child process without a shell: explicit argv, optional stdin/stdout pipes,
// output files, environment overrides, timed line reads, waits and kill. One owner thread.

namespace eOps
{
    struct ProcessOptions
    {
        std::filesystem::path m_Program; // a bare name is searched on PATH
        std::vector<std::string> m_Args;
        std::filesystem::path m_WorkingDirectory;                       // empty → inherit
        std::vector<std::pair<std::string, std::string>> m_Environment; // added/overridden; rest inherited
        bool m_PipeStdin = false;
        bool m_PipeStdout = false;
        std::filesystem::path m_StdoutFile;      // appended when not piped; empty → null device
        std::filesystem::path m_StderrFile;      // appended; empty → null device; equal to m_StdoutFile → shared
        size_t m_MaxLineBytes = 4 * 1024 * 1024; // longer stdout lines are split at this size
    };

    class EMPY_OPS_API Process
    {
    public:
        Process();
        ~Process(); // kills a still-running child
        Process(const Process &) = delete;
        Process &operator=(const Process &) = delete;

        // false with `error` filled when the program cannot be launched; a Process starts once
        bool Start(const ProcessOptions &options, std::string *error = nullptr);

        bool Running();
        // exit code, or nullopt while still running after `timeout`; a killed child reports 1
        // (Windows) or 128 + signal (POSIX)
        std::optional<int32_t> Wait(std::chrono::milliseconds timeout);
        void Kill();

        // stdin pipe; false when not piped, closed or broken
        bool Write(std::string_view bytes);
        // as Write, but also false once `timeout` passes before every byte was accepted (a child that stops
        // reading its input); the stream is then unusable and the caller kills the child
        bool Write(std::string_view bytes, std::chrono::milliseconds timeout);
        void CloseStdin();

        // next stdout line without '\n' (and a trailing '\r'); nullopt on timeout or after EOF
        // once every buffered line was returned (a final unterminated line is returned first)
        std::optional<std::string> ReadLine(std::chrono::milliseconds timeout);

        uint64_t Pid() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps
