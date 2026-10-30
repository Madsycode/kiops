#include "Modules/ClusterNode.h"
#include "Modules/Monitoring.h"
#include <charconv>
#include <csignal>
#include <iostream>

namespace
{
    // Signal handlers may only write sig_atomic_t; shutdown happens on the owner thread.
    volatile std::sig_atomic_t g_Interrupted = 0;

    uint32_t Number(std::string_view text, uint32_t maximum)
    {
        uint32_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > maximum)
        {
            throw std::invalid_argument("Invalid numeric argument");
        }
        return value;
    }
} // namespace

int32_t main(int32_t argc, char **argv)
{
    try
    {
        eMQ::Server::Configs options;
        uint32_t duration = 0;
        bool monitor = false;
        for (int32_t i = 1; i < argc; ++i)
        {
            const std::string_view argument(argv[i]);
            if (argument == "--help")
            {
                std::cout << "EmpyBroker [--config FILE] [--address IP] [--port PORT] [--seconds N]\n"
                          << "  [--monitor] [--telemetry-port PORT]\n";
                return EXIT_SUCCESS;
            }
            if (argument == "--monitor")
            {
                monitor = true;
                continue;
            }
            if (++i == argc)
            {
                throw std::invalid_argument("Missing option value");
            }
            if (argument == "--config")
            {
                options = eMQ::Server::LoadConfigs(argv[i]);
            }
            else if (argument == "--address")
            {
                options.m_Address = argv[i];
            }
            else if (argument == "--port")
            {
                options.m_Port = static_cast<uint16_t>(Number(argv[i], UINT16_MAX));
            }
            else if (argument == "--seconds")
            {
                duration = Number(argv[i], 86400);
            }
            else if (argument == "--telemetry-port")
            {
                if (!options.m_Telemetry)
                {
                    options.m_Telemetry.emplace();
                }
                options.m_Telemetry->m_Port = static_cast<uint16_t>(Number(argv[i], UINT16_MAX));
            }
            else
            {
                throw std::invalid_argument("Unknown broker option");
            }
        }
        eCore::SetLevel(eCore::LogLevel::Warn);
        std::signal(SIGINT, [](int) { g_Interrupted = 1; });
        std::signal(SIGTERM, [](int) { g_Interrupted = 1; });
        eMQ::Broker broker(std::move(options));
        broker.AttachModule<eMQ::ClusterNode>();
        auto telemetry = broker.AttachModule<eMQ::Monitoring>(monitor);
        const auto port = broker.Listen();
        broker.Poll();
        if (telemetry->Port())
        {
            std::cout << "Telemetry listening on port " << telemetry->Port() << std::endl;
        }
        std::cout << "MMQP broker listening on port " << port << std::endl;
        const auto deadline =
            duration ? eCore::Clock::now() + std::chrono::seconds(duration) : eCore::Clock::time_point::max();
        while (!g_Interrupted && !broker.IsTerminated() && eCore::Clock::now() < deadline)
        {
            broker.Poll(50);
        }
        broker.Shutdown();
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
