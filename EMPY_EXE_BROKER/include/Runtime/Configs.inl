#pragma once
#include "Configs.h"
#include <eCore/Network/Authentication.h>
#include <eCore/Common/YAML.h>
#include <cstdlib>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    inline Configs LoadConfigs(const char *filename)
    {
        auto root = eCore::YamlNode::Load(filename);
        Configs options;
        options.m_Address = root.Get("address", options.m_Address);
        options.m_Port = root.Get("port", options.m_Port);
        options.m_Workers = root.Get("workers", options.m_Workers);
        options.m_MaxSessions = root.Get("max_sessions", options.m_MaxSessions);
        options.m_MaxProfiles = root.Get("max_profiles", options.m_MaxProfiles);
        options.m_MaxInflight = root.Get("max_inflight", options.m_MaxInflight);
        options.m_MaxSessionBytes = root.Get("max_session_bytes", options.m_MaxSessionBytes);
        options.m_MaxRetainedBytes = root.Get("max_retained_bytes", options.m_MaxRetainedBytes);
        options.m_MaxRetainedTopics = root.Get("max_retained_topics", options.m_MaxRetainedTopics);
        options.m_MaxRequests = root.Get("max_requests", options.m_MaxRequests);
        options.m_MaxResponders = root.Get("max_responders", options.m_MaxResponders);
        options.m_RetryInterval = root.Get("retry_interval_ms", options.m_RetryInterval);
        options.m_MaxRetries = root.Get("max_retries", options.m_MaxRetries);
        options.m_RequestTimeout = root.Get("request_timeout_ms", options.m_RequestTimeout);
        options.m_SessionExpiry = EncodeInterval(root.Get<uint32_t>("session_expiry_seconds", 300));
        options.m_Keepalive = EncodeInterval(root.Get<uint32_t>("keepalive_seconds", 30));
        options.m_TenantRoot = root.Get("tenant_root", options.m_TenantRoot);
        options.m_RequireAuth = root.Get("require_auth", options.m_RequireAuth);
        options.m_AllowRemotePlaintext = root.Get("allow_remote_plaintext", options.m_AllowRemotePlaintext);
        options.m_StateDirectory = eCore::ConfigPath(filename, root.Get<std::string>("state_directory", {}));
        options.m_MaxStateBytes = root.Get("max_state_bytes", options.m_MaxStateBytes);
        options.m_MaxReplayBytes = root.Get("max_replay_bytes", options.m_MaxReplayBytes);
        options.m_Transport.m_MaxPacket = root.Get("max_packet_bytes", options.m_Transport.m_MaxPacket);
        options.m_Transport.m_MaxQueue = root.Get("max_send_bytes", options.m_Transport.m_MaxQueue);
        options.m_Transport.m_Timeout = root.Get("handshake_timeout_ms", options.m_Transport.m_Timeout);
        options.m_Transport.m_Tls = root.Get("tls", false);
        options.m_Transport.m_Quic = root.Get("quic", false);
        if (options.m_Transport.m_Quic)
        {
            options.m_Transport.m_QuicPolicy.m_Strategy = root.Get<uint8_t>("quic_strategy", MMQP_QUIC_STRATEGY_BULK);
            if (*options.m_Transport.m_QuicPolicy.m_Strategy == MMQP_QUIC_STRATEGY_BULK)
            {
                options.m_Transport.m_QuicPolicy.m_Bulk =
                    BulkPolicy{root.Get<uint8_t>("bulk_streams", 2), root.Get<uint32_t>("bulk_threshold_bytes", 65536)};
            }
            else
            {
                options.m_Transport.m_QuicPolicy.m_Bulk.reset();
            }
            auto datagram = root.Get<uint32_t>("datagram_threshold_bytes", 1200);
            options.m_Transport.m_QuicPolicy.m_Datagram = datagram ? std::optional(datagram) : std::nullopt;
            QuicPlan(options.m_Transport.m_QuicPolicy);
        }
        options.m_Transport.m_Mutual = root.Get("mutual_tls", false);
        options.m_Transport.m_Certificate = eCore::ConfigPath(filename, root.Get<std::string>("certificate", {}));
        options.m_Transport.m_CertificateHash = root.Get<std::string>("certificate_hash", {});
        options.m_Transport.m_PrivateKey = eCore::ConfigPath(filename, root.Get<std::string>("private_key", {}));
        options.m_Transport.m_Authority = eCore::ConfigPath(filename, root.Get<std::string>("authority", {}));
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
        if (root.Has("telemetry"))
        {
            const auto &node = root["telemetry"];
            if (!node.IsMap())
            {
                throw std::invalid_argument("telemetry must be a mapping");
            }
            TelemetryOptions telemetry;
            telemetry.m_Address = node.Get("address", telemetry.m_Address);
            telemetry.m_Port = node.Get("port", telemetry.m_Port);
            telemetry.m_IntervalMs = node.Get("interval_ms", telemetry.m_IntervalMs);
            telemetry.m_MaxSessions = node.Get("max_sessions", telemetry.m_MaxSessions);
            auto variable = node.Get<std::string>("token_env", {});
            if (!variable.empty())
            {
                auto token = std::getenv(variable.c_str());
                if (!token || !*token)
                {
                    throw std::invalid_argument("Telemetry token environment variable is missing or empty");
                }
                telemetry.m_Token = token;
            }
            options.m_Telemetry = std::move(telemetry);
        }
        if (root.Has("cluster"))
        {
            const auto &node = root["cluster"];
            if (!node.IsMap())
            {
                throw std::invalid_argument("cluster must be a mapping");
            }
            ClusterOptions cluster;
            cluster.m_Identifier = node.Get<std::string>("identifier", {});
            cluster.m_Address = node.Get("address", cluster.m_Address);
            cluster.m_Port = node.Get("port", cluster.m_Port);
            cluster.m_Directory = eCore::ConfigPath(filename, node.Get<std::string>("state_directory", {}));
            cluster.m_Fabric.m_Epoch = node.Get("epoch", cluster.m_Fabric.m_Epoch);
            cluster.m_Fabric.m_Shards = node.Get("shards", cluster.m_Fabric.m_Shards);
            cluster.m_Fabric.m_Replicas = node.Get("replicas", cluster.m_Fabric.m_Replicas);
            cluster.m_Fabric.m_Fanout = node.Get("fanout", cluster.m_Fabric.m_Fanout);
            cluster.m_LeaseMilliseconds = node.Get("lease_ms", cluster.m_LeaseMilliseconds);
            cluster.m_TimeoutMilliseconds = node.Get("rpc_timeout_ms", cluster.m_TimeoutMilliseconds);
            cluster.m_MaxBytes = node.Get("max_state_bytes", cluster.m_MaxBytes);
            cluster.m_MaxRecords = node.Get("max_records", cluster.m_MaxRecords);
            cluster.m_MaxOperations = node.Get("max_operations", cluster.m_MaxOperations);
            cluster.m_MaxInboxBytes = node.Get("max_inbox_bytes", cluster.m_MaxInboxBytes);
            cluster.m_MaxInboxRecords = node.Get("max_inbox_records", cluster.m_MaxInboxRecords);
            cluster.m_Transport.m_Quic = node.Get("quic", true);
            cluster.m_Transport.m_Tls = true;
            cluster.m_Transport.m_Mutual = true;
            cluster.m_Transport.m_MaxPacket = node.Get<uint32_t>("max_packet_bytes", 16 * 1024 * 1024);
            cluster.m_Transport.m_MaxQueue = node.Get<size_t>("max_send_bytes", 64 * 1024 * 1024);
            cluster.m_Transport.m_Certificate = eCore::ConfigPath(filename, node.Get<std::string>("certificate", {}));
            cluster.m_Transport.m_PrivateKey = eCore::ConfigPath(filename, node.Get<std::string>("private_key", {}));
            cluster.m_Transport.m_Authority = eCore::ConfigPath(filename, node.Get<std::string>("authority", {}));
            cluster.m_Transport.m_CertificateHash = node.Get<std::string>("certificate_hash", {});
            auto variable = node.Get<std::string>("epoch_key_env", {});
            auto secret = variable.empty() ? nullptr : std::getenv(variable.c_str());
            if (!secret)
            {
                throw std::invalid_argument("Cluster epoch key environment variable is missing");
            }
            cluster.m_EpochKey = FromHex(secret);
            if (!node.Has("peers") || !node["peers"].IsList())
            {
                throw std::invalid_argument("Cluster peers must be a sequence");
            }
            for (const auto &peer : node["peers"].Items())
            {
                if (!peer.IsMap())
                {
                    throw std::invalid_argument("Cluster peer must be a mapping");
                }
                ClusterPeer member;
                member.m_Identifier = peer.Get<std::string>("identifier", {});
                member.m_Host = peer.Get("host", member.m_Host);
                member.m_Port = peer.Get<uint16_t>("port", 0);
                member.m_ServerName = peer.Get<std::string>("server_name", {});
                member.m_Fingerprint = ToHex(FromHex(peer.Get<std::string>("fingerprint", {})));
                cluster.m_Fabric.m_Brokers.push_back(member.m_Identifier);
                cluster.m_Peers.push_back(std::move(member));
            }
            options.m_Cluster = std::move(cluster);
        }
        if (root.Has("tenants"))
        {
            if (!root["tenants"].IsList())
            {
                throw std::invalid_argument("tenants must be a sequence");
            }
            for (const auto &node : root["tenants"].Items())
            {
                if (!node.IsMap())
                {
                    throw std::invalid_argument("Tenant must be a mapping");
                }
                TenantOptions tenant;
                tenant.m_Identity = node.Get<std::string>("identity", {});
                tenant.m_Root = node.Get<std::string>("root", {});
                auto variable = node.Get<std::string>("auth_key_env", {});
                auto secret = variable.empty() ? nullptr : std::getenv(variable.c_str());
                if (!secret)
                {
                    throw std::invalid_argument("Tenant authentication key environment variable is missing");
                }
                tenant.m_AuthKey = FromHex(secret);
                if (node.Has("shards"))
                {
                    if (!node["shards"].IsList())
                    {
                        throw std::invalid_argument("Tenant shards must be a sequence");
                    }
                    for (const auto &shard : node["shards"].Items())
                    {
                        auto value = shard.As<int64_t>();
                        if (value < 0 || value > UINT32_MAX)
                        {
                            throw std::invalid_argument("Invalid tenant shard number");
                        }
                        tenant.m_Shards.insert(static_cast<uint32_t>(value));
                    }
                }
                options.m_Tenants.push_back(std::move(tenant));
            }
        }
        return options;
    }
} // namespace eMQ::Server
