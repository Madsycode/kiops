#pragma once
#include <chrono>
#include <format>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <eCore/Common/Globals.h>

// minimal self-contained test harness: EMPY_TEST registers a case, EMPY_CHECK* record
// failures and continue, EMPY_REQUIRE* abort the current case; the runner (Main.cpp)
// selects cases by suite so CTest can run each suite as an independent process

namespace eTest
{
    struct Failure
    {
        std::string File;
        int32_t Line = 0;
        std::string Message;
    };

    // thrown by EMPY_REQUIRE to abandon the current case
    struct AbortCase
    {
    };

    struct Case
    {
        const char *Suite = nullptr;
        const char *Name = nullptr;
        void (*Body)() = nullptr;
    };

    inline std::vector<Case> &Registry()
    {
        static std::vector<Case> s_Cases;
        return s_Cases;
    }

    inline std::vector<Failure> &Failures()
    {
        static std::vector<Failure> s_Failures;
        return s_Failures;
    }

    struct Registrar
    {
        Registrar(const char *suite, const char *name, void (*body)())
        {
            Registry().push_back(Case{suite, name, body});
        }
    };

    inline void Fail(const char *file, int32_t line, std::string message)
    {
        Failures().push_back(Failure{file, line, std::move(message)});
    }

    template <typename _Type> std::string Describe(const _Type &value)
    {
        if constexpr (std::is_same_v<_Type, bool>)
        {
            return value ? "true" : "false";
        }
        else if constexpr (std::is_enum_v<_Type>)
        {
            return std::to_string(static_cast<int64_t>(value));
        }
        else if constexpr (std::is_integral_v<_Type> && sizeof(_Type) == 1)
        {
            return std::to_string(static_cast<int32_t>(value));
        }
        else if constexpr (requires(std::ostream &os, const _Type &v) { os << v; })
        {
            std::ostringstream stream;
            stream << value;
            return stream.str();
        }
        else
        {
            return "<value>";
        }
    }

    // poll `condition` until true or `timeout` elapses (for asynchronous integration tests)
    template <typename _Condition>
    bool WaitUntil(_Condition &&condition, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000))
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!condition())
        {
            if (std::chrono::steady_clock::now() >= deadline)
            {
                return condition();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    }
} // namespace eTest

#define EMPY_TEST_CONCAT_INNER(a, b) a##b
#define EMPY_TEST_CONCAT(a, b) EMPY_TEST_CONCAT_INNER(a, b)

// declare a test case: EMPY_TEST(Suite, Name) { ... }
#define EMPY_TEST(Suite, Name)                                                                                         \
    static void EmpyTest_##Suite##_##Name();                                                                           \
    static ::eTest::Registrar EMPY_TEST_CONCAT(s_EmpyRegistrar_##Suite##_##Name,                                       \
                                               __LINE__)(#Suite, #Name, &EmpyTest_##Suite##_##Name);                   \
    static void EmpyTest_##Suite##_##Name()

#define EMPY_CHECK(...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(__VA_ARGS__))                                                                                            \
        {                                                                                                              \
            ::eTest::Fail(__FILE__, __LINE__, "CHECK(" #__VA_ARGS__ ")");                                              \
        }                                                                                                              \
    } while (0)

#define EMPY_REQUIRE(...)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(__VA_ARGS__))                                                                                            \
        {                                                                                                              \
            ::eTest::Fail(__FILE__, __LINE__, "REQUIRE(" #__VA_ARGS__ ")");                                            \
            throw ::eTest::AbortCase{};                                                                                \
        }                                                                                                              \
    } while (0)

#define EMPY_CHECK_EQ(actual, expected)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto &empyActual_ = (actual);                                                                            \
        const auto &empyExpected_ = (expected);                                                                        \
        if (!(empyActual_ == empyExpected_))                                                                           \
        {                                                                                                              \
            ::eTest::Fail(__FILE__, __LINE__,                                                                          \
                          std::string("CHECK_EQ(" #actual ", " #expected "): ") + ::eTest::Describe(empyActual_) +     \
                              " != " + ::eTest::Describe(empyExpected_));                                              \
        }                                                                                                              \
    } while (0)

#define EMPY_REQUIRE_EQ(actual, expected)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto &empyActual_ = (actual);                                                                            \
        const auto &empyExpected_ = (expected);                                                                        \
        if (!(empyActual_ == empyExpected_))                                                                           \
        {                                                                                                              \
            ::eTest::Fail(__FILE__, __LINE__,                                                                          \
                          std::string("REQUIRE_EQ(" #actual ", " #expected "): ") + ::eTest::Describe(empyActual_) +   \
                              " != " + ::eTest::Describe(empyExpected_));                                              \
            throw ::eTest::AbortCase{};                                                                                \
        }                                                                                                              \
    } while (0)

#define EMPY_CHECK_THROWS(expression)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        bool empyThrew_ = false;                                                                                       \
        try                                                                                                            \
        {                                                                                                              \
            (void)(expression);                                                                                        \
        }                                                                                                              \
        catch (...)                                                                                                    \
        {                                                                                                              \
            empyThrew_ = true;                                                                                         \
        }                                                                                                              \
        if (!empyThrew_)                                                                                               \
        {                                                                                                              \
            ::eTest::Fail(__FILE__, __LINE__, "CHECK_THROWS(" #expression ")");                                        \
        }                                                                                                              \
    } while (0)
