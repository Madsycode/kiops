#pragma once
#include <eCore/API.h>
#include <eCore/Network/Events.h>
#include "Configs.h"

// All protocol state is confined to the serial I/O executor. Application events
// cross into the runtime owner thread through the bounded Core dispatch queue.
namespace eMQ::Server
{
    using namespace eCore::MMQP;
    struct Service : eCore::ImplService<eCore::SERVER>
    {
        explicit Service(Configs options = {});
        explicit Service(const char *filename);
        ~Service() override;

        uint16_t Listen();
        uint16_t Port() const;
        BrokerMetrics Metrics();
        BrokerSnapshot Inspect(size_t maxSessions = 1024);
        void Stop();
        const Configs &Configuration() const;
        void StartCluster();
        void StopCluster();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eMQ::Server

#include "Service.inl"
