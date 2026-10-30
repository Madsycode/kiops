#include <eMQ/Runtime/Service.h>
#include <eCore/Common/YAML.h>
#include <cstdlib>

namespace eMQ
{
    Configs LoadConfigs(const char *filename)
    {
        auto root = eCore::YamlNode::Load(filename);
        Configs options;
        options.m_Transport.m_MaxPacket = root.Get("max_packet_bytes", options.m_Transport.m_MaxPacket);
        options.m_Transport.m_MaxQueue = root.Get("max_send_bytes", options.m_Transport.m_MaxQueue);
        options.m_Transport.m_Timeout = root.Get("handshake_timeout_ms", options.m_Transport.m_Timeout);
        options.m_Transport.m_Tls = root.Get("tls", false);
        options.m_Transport.m_Quic = root.Get("quic", false);
        if (options.m_Transport.m_Quic)
        {
            QuicPolicy quic;
            quic.m_Strategy = root.Get<uint8_t>("quic_strategy", MMQP_QUIC_STRATEGY_SEMANTIC);
            if (*quic.m_Strategy == MMQP_QUIC_STRATEGY_BULK)
            {
                quic.m_Bulk =
                    BulkPolicy{root.Get<uint8_t>("bulk_streams", 2), root.Get<uint32_t>("bulk_threshold_bytes", 65536)};
            }
            auto datagram = root.Get<uint32_t>("datagram_threshold_bytes", 0);
            if (datagram)
            {
                quic.m_Datagram = datagram;
            }
            options.m_Properties.m_Quic = QuicPlan(quic).Policy();
        }
        options.m_Transport.m_Certificate = eCore::ConfigPath(filename, root.Get<std::string>("certificate", {}));
        options.m_Transport.m_CertificateHash = root.Get<std::string>("certificate_hash", {});
        options.m_Transport.m_PrivateKey = eCore::ConfigPath(filename, root.Get<std::string>("private_key", {}));
        options.m_Transport.m_Authority = eCore::ConfigPath(filename, root.Get<std::string>("authority", {}));
        options.m_Transport.m_ServerName = root.Get<std::string>("server_name", {});
        options.m_MaxInflight = root.Get("max_inflight", options.m_MaxInflight);
        options.m_MaxProfiles = root.Get("max_profiles", options.m_MaxProfiles);
        options.m_MaxRequests = root.Get("max_requests", options.m_MaxRequests);
        options.m_MaxPendingBytes = root.Get("max_pending_bytes", options.m_MaxPendingBytes);
        options.m_RetryInterval = root.Get("retry_interval_ms", options.m_RetryInterval);
        options.m_MaxRetries = root.Get("max_retries", options.m_MaxRetries);
        options.m_OperationTimeout = root.Get("operation_timeout_ms", options.m_OperationTimeout);
        options.m_ResponseTimeout = root.Get("response_timeout_ms", options.m_ResponseTimeout);
        options.m_AutoReconnect = root.Get("auto_reconnect", options.m_AutoReconnect);
        options.m_AuthIdentity = root.Get<std::string>("auth_identity", {});
        options.m_AllowInsecureAuth = root.Get("allow_insecure_auth", options.m_AllowInsecureAuth);
        options.m_ReconnectInterval = root.Get("reconnect_interval_ms", options.m_ReconnectInterval);
        if (root.Has("failover_endpoints"))
        {
            if (!root["failover_endpoints"].IsList())
            {
                throw std::invalid_argument("failover_endpoints must be a sequence");
            }
            for (const auto &node : root["failover_endpoints"].Items())
            {
                if (!node.IsMap())
                {
                    throw std::invalid_argument("Failover endpoint must be a mapping");
                }
                options.m_FailoverEndpoints.push_back(
                    {node.Get<std::string>("host", "127.0.0.1"), node.Get<uint16_t>("port", 5600)});
            }
        }
        SessionPolicy policy;
        policy.m_Reasons = root.Get("reason_strings", true);
        policy.m_Expiry = EncodeInterval(root.Get<uint32_t>("session_expiry_seconds", 300));
        policy.m_Keepalive = EncodeInterval(root.Get<uint32_t>("keepalive_seconds", 30));
        policy.m_MaxPacket = options.m_Transport.m_MaxPacket;
        options.m_Properties.m_Policy = policy;
        auto variable = root.Get<std::string>("auth_key_env", {});
        if (!variable.empty())
        {
            auto secret = std::getenv(variable.c_str());
            if (!secret)
            {
                throw std::invalid_argument("Authentication key environment variable is missing");
            }
            options.m_AuthKey = FromHex(secret);
        }
        auto payloadVariable = root.Get<std::string>("payload_key_env", {});
        if (!payloadVariable.empty())
        {
            if (payloadVariable == variable)
            {
                throw std::invalid_argument("Authentication and payload keys must use separate provisioning");
            }
            auto secret = std::getenv(payloadVariable.c_str());
            if (!secret)
            {
                throw std::invalid_argument("Payload key environment variable is missing");
            }
            auto key = FromHex(secret);
            if (!options.m_AuthKey.empty() && SecureEqual(options.m_AuthKey, key))
            {
                SecureErase(key);
                throw std::invalid_argument("Authentication and payload keys must be different");
            }
            options.m_KeyEpoch = root.Get<uint32_t>("key_epoch", 1);
            options.m_RequireEncryption = root.Get("require_encryption", options.m_RequireEncryption);
            auto minimum = root.Get("minimum_key_epoch", options.m_KeyEpoch);
            auto maximum = root.Get("maximum_key_epoch", options.m_KeyEpoch);
            auto topic = root.Get<std::string>("key_subtree", {});
            auto epoch = topic.empty() ? std::optional<uint32_t>{} : std::optional(options.m_KeyEpoch);
            auto directory = eCore::ConfigPath(filename, root.Get<std::string>("payload_state_directory", {}));
            auto keys = root.Get<size_t>("payload_key_cache", 128);
            auto replays = root.Get<size_t>("payload_replay_limit", 4096);
            auto reservation = root.Get<uint64_t>("payload_nonce_reservation", 1024);
            options.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{std::move(key), topic, epoch}, minimum, maximum,
                                                               keys, replays, directory, reservation);
        }
        return options;
    }
} // namespace eMQ
