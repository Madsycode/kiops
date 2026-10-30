#include "Checks.h"
#include "Support/Process.h"
#include <eCore/Network/Authentication.h>
#include <charconv>
#include <iterator>

namespace
{
    struct PackageDirectory
    {
        std::filesystem::path m_Path =
            std::filesystem::temp_directory_path() / ("Empy package " + eCore::ToHex(eCore::MMQP::RandomBytes(8)));
        PackageDirectory()
        {
            std::filesystem::create_directory(m_Path);
        }
        ~PackageDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(m_Path, error);
        }
    };
} // namespace

EMPY_TEST(Packaging, IsolatedExecutablesCommunicate)
{
    PackageDirectory package;
    const auto original = eTest::BinaryDirectory();
#ifdef _WIN32
    const std::string suffix = ".exe";
#else
    const std::string suffix;
#endif
    for (const auto &name : {"EmpyBroker", "EmpyClient"})
    {
        std::filesystem::copy_file(original / (name + suffix), package.m_Path / (name + suffix));
#ifdef _WIN32
        std::ifstream manifest(original / (name + suffix + ".runtime.txt"));
        EMPY_REQUIRE(manifest.good());
        std::string line;
        std::getline(manifest, line);
        size_t verified = 0;
        while (std::getline(manifest, line))
        {
            EMPY_REQUIRE(line.size() > 66 && line.substr(64, 2) == "  ");
            auto filename = line.substr(66);
            EMPY_REQUIRE(std::filesystem::path(filename).filename() == filename);
            std::ifstream input(original / filename, std::ios::binary);
            EMPY_REQUIRE(input.good());
            std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
            EMPY_REQUIRE(eCore::ToHex(eCore::Sha256(bytes)) == line.substr(0, 64));
            std::filesystem::copy_file(original / filename, package.m_Path / filename,
                                       std::filesystem::copy_options::overwrite_existing);
            ++verified;
        }
        EMPY_REQUIRE(verified >= 3);
#endif
    }
#ifndef _WIN32
    for (const auto &entry : std::filesystem::directory_iterator(original))
    {
        if (entry.path().filename().string().find(".so") != std::string::npos)
        {
            std::filesystem::copy_file(entry.path(), package.m_Path / entry.path().filename());
        }
    }
#endif
    const auto path = eTest::RestrictedSearchPath(package.m_Path);
    const auto brokerProgram = package.m_Path / ("EmpyBroker" + suffix);
    const auto clientProgram = package.m_Path / ("EmpyClient" + suffix);
    eTest::ChildProcess broker;
    EMPY_REQUIRE(broker.Start(brokerProgram, {"--port", "0", "--seconds", "8"}, package.m_Path, path,
                              package.m_Path / "broker.log"));
    uint16_t port = 0;
    EMPY_REQUIRE(eTest::WaitUntil(
        [&]
        {
            auto output = broker.Output();
            const std::string marker = "MMQP broker listening on port ";
            const auto position = output.find(marker);
            if (position == std::string::npos)
            {
                return false;
            }
            auto first = output.data() + position + marker.size();
            const auto parsed = std::from_chars(first, output.data() + output.size(), port);
            return parsed.ec == std::errc{} && port;
        }));
    eTest::ChildProcess subscriber;
    EMPY_REQUIRE(subscriber.Start(
        clientProgram,
        {"subscribe", "package/value", "--port", std::to_string(port), "--count", "1", "--seconds", "6", "--verbose"},
        package.m_Path, path, package.m_Path / "subscriber.log"));
    EMPY_REQUIRE(eTest::WaitUntil([&] { return subscriber.Output().find("Profile ready:") != std::string::npos; }));
    eTest::ChildProcess publisher;
    EMPY_REQUIRE(publisher.Start(clientProgram,
                                 {"publish", "package/value", "isolated runtime", "--port", std::to_string(port)},
                                 package.m_Path, path, package.m_Path / "publisher.log"));
    EMPY_REQUIRE_EQ(publisher.Wait(std::chrono::seconds(5)), 0);
    EMPY_REQUIRE_EQ(subscriber.Wait(std::chrono::seconds(5)), 0);
    EMPY_REQUIRE(subscriber.Output().find("package/value isolated runtime") != std::string::npos);
    eTest::ChildProcess requester;
    EMPY_REQUIRE(requester.Start(clientProgram,
                                 {"request", "package/missing", "Q", "--port", std::to_string(port), "--seconds", "3"},
                                 package.m_Path, path, package.m_Path / "requester.log"));
    EMPY_REQUIRE_EQ(requester.Wait(std::chrono::seconds(4)), 1);
    EMPY_REQUIRE(requester.Output().find("No eligible responder") != std::string::npos);
    eTest::ChildProcess watcher;
    EMPY_REQUIRE(watcher.Start(
        clientProgram, {"subscribe", "package/loss", "--port", std::to_string(port), "--seconds", "20", "--verbose"},
        package.m_Path, path, package.m_Path / "watcher.log"));
    EMPY_REQUIRE(eTest::WaitUntil([&] { return watcher.Output().find("Profile ready:") != std::string::npos; }));
    EMPY_REQUIRE_EQ(broker.Wait(std::chrono::seconds(10)), 0);
    EMPY_REQUIRE_EQ(watcher.Wait(std::chrono::seconds(5)), 1);
    EMPY_REQUIRE(watcher.Output().find("MMQP connection failed:") != std::string::npos);
}
