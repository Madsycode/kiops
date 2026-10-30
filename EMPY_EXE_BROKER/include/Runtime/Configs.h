#pragma once
#include <eCore/Common/YAML.h>
#include "../Cluster/Node.h"
#include "../Monitoring/Telemetry.h"

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    struct TenantOptions
    {
        std::string m_Identity;
        std::string m_Root;
        Buffer m_AuthKey;
        std::set<uint32_t> m_Shards;
    };

    struct Configs
    {
        std::string m_Address = "127.0.0.1";
        uint16_t m_Port = 5600;
        uint32_t m_Workers = 1;
        TransportOptions m_Transport;
        size_t m_MaxSessions = 1024;
        size_t m_MaxProfiles = 256;
        size_t m_MaxInflight = 256;
        size_t m_MaxSessionBytes = 4 * 1024 * 1024;
        size_t m_MaxRetainedBytes = 16 * 1024 * 1024;
        size_t m_MaxRetainedTopics = 4096;
        size_t m_MaxRequests = 256;
        size_t m_MaxResponders = 64;
        uint32_t m_RetryInterval = 500;
        uint32_t m_MaxRetries = 5;
        uint32_t m_RequestTimeout = 5000;
        uint8_t m_SessionExpiry = 0x45;
        uint8_t m_Keepalive = 30;
        std::string m_TenantRoot;
        Buffer m_AuthKey;
        bool m_RequireAuth = false;
        bool m_AllowRemotePlaintext = false;
        bool m_AllowInsecureAuth = false;
        std::string m_StateDirectory;
        size_t m_MaxStateBytes = 256 * 1024 * 1024;
        size_t m_MaxReplayBytes = 16 * 1024 * 1024;
        std::optional<ClusterOptions> m_Cluster;
        std::vector<TenantOptions> m_Tenants;
        std::optional<TelemetryOptions> m_Telemetry;
    };

    inline Configs LoadConfigs(const char *filename);

} // namespace eMQ::Server

#include "Configs.inl"
