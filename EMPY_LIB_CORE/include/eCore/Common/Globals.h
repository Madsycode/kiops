#pragma once

// platform detection
#ifdef _WIN32
// optimize win32 headers
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
// desable min/max macros
#ifndef NOMINMAX
#define NOMINMAX
#endif
// include win32 headers
#include <windows.h>
// windows.h maps these names to A/W variants; they would silently rename framework methods
#undef SendMessage
#undef PostMessage
#undef GetMessage
#undef CreateEvent
#elif defined(__linux__)
#include <unistd.h>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>

// export macros: each shared library owns its exported declarations
#ifdef _WIN32
#define EMPY_EXPORT __declspec(dllexport)
#define EMPY_IMPORT __declspec(dllimport)
#else
#define EMPY_EXPORT __attribute__((visibility("default")))
#define EMPY_IMPORT
#endif

// shared library macros: each shared library defines its own macro to control symbol visibility
#ifdef EMPY_CORE_SHARED
#define EMPY_CORE_API EMPY_EXPORT
#else
#define EMPY_CORE_API EMPY_IMPORT
#endif

// shared library macros: each shared library defines its own macro to control symbol visibility
#ifdef EMPY_MQ_SHARED
#define EMPY_MQ_API EMPY_EXPORT
#else
#define EMPY_MQ_API EMPY_IMPORT
#endif

// compile time assertion
#define EMPY_STATIC static_assert

// runtime assertion
#define EMPY_ASSERT assert

// discard attribute
#if __cplusplus >= 201703L
#define EMPY_NODISCARD [[nodiscard]]
#else
#define EMPY_NODISCARD
#endif

// free allocated memory
#define EMPY_FREE(ptr)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        delete ptr;                                                                                                    \
        ptr = nullptr;                                                                                                 \
    } while (0)

// function inline macros
#if defined(__clang__) || defined(__GNUC__)
#define EMPY_INLINE __attribute__((always_inline)) inline
#define EMPY_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define EMPY_INLINE __forceinline
#define EMPY_NOINLINE __declspec(noinline)
#else
#define EMPY_INLINE inline
#define EMPY_NOINLINE
#endif

// branch prediction hints for hot protocol paths
#define EMPY_LIKELY [[likely]]
#define EMPY_UNLIKELY [[unlikely]]

namespace eCore
{
    // stable type key shared by every module: derived from the compiler's type name, never
    // from per-module static counters, so DLLs and executables agree on event/component keys
    template <typename _Type> constexpr uint64_t TypeKey() noexcept
    {
#if defined(_MSC_VER) && !defined(__clang__)
        const char *name = __FUNCSIG__;
#else
        const char *name = __PRETTY_FUNCTION__;
#endif
        uint64_t hash = 14695981039346656037ull;
        for (const char *it = name; *it != '\0'; ++it)
        {
            hash = (hash ^ static_cast<uint8_t>(*it)) * 1099511628211ull;
        }
        return hash;
    }
} // namespace eCore
