#pragma once
#include <mutex>
#include <atomic>
#include <chrono>
#include <format>
#include <thread>
#include <string>
#include <cstdint>
#include <sstream>
#include <iostream>
#include "Globals.h"

// Rang's Windows API lookup uses an upstream function-pointer cast.
#if defined(__GNUC__) && defined(_WIN32)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#include "../Vendors/Rang.h"
#if defined(__GNUC__) && defined(_WIN32)
#pragma GCC diagnostic pop
#endif

namespace eCore
{
    enum class LogLevel : uint8_t
    {
        None,
        Trace,
        Debug,
        Info,
        Warn,
        Error,
        Fatal
    };

    // process-wide logging state lives in EmpyCore so every module shares one threshold and lock
    // (header-level inline variables would be duplicated per DLL on Windows)
    EMPY_CORE_API std::atomic<LogLevel> &LogThreshold() noexcept;
    EMPY_CORE_API std::mutex &LogMutex() noexcept;

    // -------------------------
    // internal log function
    // -------------------------

    template <typename... _Args>
    EMPY_INLINE void LogWrite(LogLevel level, rang::fg color, const char *tag, std::format_string<_Args...> fmt,
                              _Args &&...args)
    {
        if (level < LogThreshold().load(std::memory_order_relaxed))
        {
            return;
        }

        auto sec = std::chrono::time_point_cast<std::chrono::seconds>(std::chrono::system_clock::now());

        std::ostringstream tid;
        tid << std::this_thread::get_id();
        std::string msg = std::format(fmt, std::forward<_Args>(args)...);

        std::scoped_lock lock(LogMutex());
        std::cout << color << std::format("[{:%Y-%m-%d %H:%M:%S}] [T:{}] [{:^5}] {}", sec, tid.str(), tag, msg)
                  << rang::style::reset << "\n";
    }

    // -------------------------
    // public API
    // -------------------------

    template <typename... _Args> EMPY_INLINE void Trace(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Trace, rang::fg::reset, "TRACE", fmt, std::forward<_Args>(args)...);
    }

    template <typename... _Args> EMPY_INLINE void Debug(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Debug, rang::fg::cyan, "DEBUG", fmt, std::forward<_Args>(args)...);
    }

    template <typename... _Args> EMPY_INLINE void Info(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Info, rang::fg::green, "INFO", fmt, std::forward<_Args>(args)...);
    }

    template <typename... _Args> EMPY_INLINE void Warn(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Warn, rang::fg::yellow, "WARN", fmt, std::forward<_Args>(args)...);
    }

    template <typename... _Args> EMPY_INLINE void Error(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Error, rang::fg::red, "ERROR", fmt, std::forward<_Args>(args)...);
    }

    template <typename... _Args> EMPY_INLINE void Fatal(std::format_string<_Args...> fmt, _Args &&...args)
    {
        LogWrite(LogLevel::Fatal, rang::fg::magenta, "FATAL", fmt, std::forward<_Args>(args)...);
    }

    // -------------------------
    // configuration
    // -------------------------

    EMPY_INLINE void SetLevel(LogLevel lvl)
    {
        LogThreshold().store(lvl, std::memory_order_relaxed);
    }
} // namespace eCore

// macros logging: Trace and Debug are compiled out of Release builds so hot paths pay nothing
#ifdef EMPY_ENABLE_LOGGING
#ifdef EMPY_DEBUG_BUILD
#define EMPY_TRACE(...) eCore::Trace(__VA_ARGS__)
#define EMPY_DEBUG(...) eCore::Debug(__VA_ARGS__)
#else
#define EMPY_TRACE(...) ((void)0)
#define EMPY_DEBUG(...) ((void)0)
#endif
#define EMPY_INFO(...) eCore::Info(__VA_ARGS__)
#define EMPY_WARN(...) eCore::Warn(__VA_ARGS__)
#define EMPY_ERROR(...) eCore::Error(__VA_ARGS__)
#define EMPY_FATAL(...) eCore::Fatal(__VA_ARGS__)
#else
#define EMPY_TRACE(...) ((void)0)
#define EMPY_DEBUG(...) ((void)0)
#define EMPY_INFO(...) ((void)0)
#define EMPY_WARN(...) ((void)0)
#define EMPY_ERROR(...) ((void)0)
#define EMPY_FATAL(...) ((void)0)
#endif
