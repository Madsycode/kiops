#pragma once
#include <eCore/Network/Connection.h>
#include <eCore/Common/YAML.h>
#include "../Security/Crypto.h"

namespace eMQ
{
    using namespace eCore::MMQP;
    struct ClientEndpoint
    {
        std::string m_Host = "127.0.0.1";
        uint16_t m_Port = 5600;
    };

    struct Configs
    {
        TransportOptions m_Transport;
        SessionProperties m_Properties;
        Buffer m_AuthKey;
        std::string m_AuthIdentity;
        size_t m_MaxInflight = 256;
        size_t m_MaxProfiles = 256;
        size_t m_MaxRequests = 256;
        size_t m_MaxPendingBytes = 4 * 1024 * 1024;
        uint32_t m_RetryInterval = 500;
        uint32_t m_MaxRetries = 5;
        uint32_t m_OperationTimeout = 5000;
        uint32_t m_ResponseTimeout = 10000;
        bool m_AllowInsecureAuth = false;
        std::shared_ptr<PayloadCipher> m_Cipher;
        uint32_t m_KeyEpoch = 1;
        bool m_RequireEncryption = true;
        std::vector<ClientEndpoint> m_FailoverEndpoints;
        bool m_AutoReconnect = false;
        uint32_t m_ReconnectInterval = 250;
    };

    EMPY_MQ_API Configs LoadConfigs(const char *filename);

} // namespace eMQ
