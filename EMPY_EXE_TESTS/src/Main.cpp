#include <thread>
#include <cstring>
#include "Checks.h"

// test runner: EmpyTests [--suite <Suite>] [--case <Suite.Name>] [--list]

int32_t main(int32_t argc, char **argv)
{
    std::string suite;
    std::string single;
    bool list = false;
    for (int32_t i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--suite") == 0 && i + 1 < argc)
        {
            suite = argv[++i];
        }
        else if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc)
        {
            single = argv[++i];
        }
        else if (std::strcmp(argv[i], "--list") == 0)
        {
            list = true;
        }
        else
        {
            std::cerr << "usage: EmpyTests [--suite <Suite>] [--case <Suite.Name>] [--list]\n";
            return 2;
        }
    }

    int32_t executed = 0;
    int32_t failed = 0;
    for (const eTest::Case &testCase : eTest::Registry())
    {
        std::string fullName = std::string(testCase.Suite) + "." + testCase.Name;
        if (!suite.empty() && suite != testCase.Suite)
        {
            continue;
        }
        if (!single.empty() && single != fullName)
        {
            continue;
        }
        if (list)
        {
            std::cout << fullName << "\n";
            continue;
        }

        size_t before = eTest::Failures().size();
        auto start = std::chrono::steady_clock::now();
        try
        {
            testCase.Body();
        }
        catch (const eTest::AbortCase &)
        {
        }
        catch (const std::exception &error)
        {
            eTest::Fail("<exception>", 0, std::string("unexpected exception: ") + error.what());
        }
        catch (...)
        {
            eTest::Fail("<exception>", 0, "unexpected non-standard exception");
        }
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

        ++executed;
        bool passed = eTest::Failures().size() == before;
        if (!passed)
        {
            ++failed;
        }
        std::cout << (passed ? "[ PASS ] " : "[ FAIL ] ") << fullName << " (" << elapsed << " ms)\n";
        for (size_t i = before; i < eTest::Failures().size(); ++i)
        {
            const eTest::Failure &failure = eTest::Failures()[i];
            std::cout << "         " << failure.File << ":" << failure.Line << ": " << failure.Message << "\n";
        }
    }

    if (list)
    {
        return 0;
    }
    if (executed == 0)
    {
        std::cout << "no test cases matched\n";
        return 1;
    }
    std::cout << "\n" << executed - failed << "/" << executed << " test cases passed\n";
    return failed == 0 ? 0 : 1;
}
