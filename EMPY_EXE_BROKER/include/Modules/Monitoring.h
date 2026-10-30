#pragma once
#include "../Runtime/Runtime.h"
#include "../Monitoring/Telemetry.inl"

namespace eMQ::Server
{
    struct Monitoring : eCore::ImplModule<Service, Context>
    {
        explicit Monitoring(bool console = false) : m_Console(console)
        {
        }
        uint16_t Port() const noexcept
        {
            return m_Port;
        }
        EMPY_INLINE void OnAttach(Service *service, Context *) override
        {
            if (service->Configuration().m_Telemetry)
            {
                m_Telemetry = std::make_unique<TelemetryServer>(*service, *service->Configuration().m_Telemetry);
                m_Port = m_Telemetry->Start();
            }
        }

        EMPY_INLINE void OnUpdate(Service *service, Context *) override
        {
            if (m_Console && eCore::Clock::now() >= m_NextReport)
            {
                auto metrics = service->Metrics();
                std::cout << "sessions=" << metrics.m_Online << '/' << metrics.m_Sessions
                          << " publications=" << metrics.m_Publications << " pending=" << metrics.m_Pending
                          << std::endl;
                m_NextReport = eCore::Clock::now() + std::chrono::seconds(1);
            }
        }

        EMPY_INLINE void OnDetach(Service *, Context *) override
        {
            m_Telemetry.reset();
        }

    private:
        std::unique_ptr<TelemetryServer> m_Telemetry;
        bool m_Console = false;
        uint16_t m_Port = 0;
        eCore::Clock::time_point m_NextReport{};
    };
} // namespace eMQ::Server

namespace eMQ
{
    using Server::Monitoring;
}
