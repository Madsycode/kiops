#pragma once
#include "Snapshot.h"

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    struct Service;

    struct TelemetryOptions
    {
        std::string m_Address = "127.0.0.1";
        uint16_t m_Port = 5601;
        std::string m_Token;
        uint32_t m_IntervalMs = 1000;
        size_t m_MaxSessions = 1024;
    };

    // A loopback-only, read-only HTTP endpoint with a bounded cached snapshot.
    // Start after Service::Listen and stop before Service::Stop. The broker must
    // outlive this object; sampling never runs on its protocol executor.
    struct TelemetryServer
    {
        explicit TelemetryServer(Service &broker, TelemetryOptions options = {});
        ~TelemetryServer();
        TelemetryServer(const TelemetryServer &) = delete;
        TelemetryServer &operator=(const TelemetryServer &) = delete;

        uint16_t Start();
        void Stop();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eMQ::Server
