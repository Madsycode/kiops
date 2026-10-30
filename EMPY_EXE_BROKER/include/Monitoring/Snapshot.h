#pragma once
#include <eCore/Network/State.h>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    // Process-local observations. These counters do not change persisted MMQP state.
    struct TrafficMetrics
    {
        uint64_t m_ReceivedPackets = 0;
        uint64_t m_ReceivedBytes = 0;
        uint64_t m_SentPackets = 0;
        uint64_t m_SentBytes = 0;
    };

    struct BrokerMetrics : TrafficMetrics
    {
        uint64_t m_Accepted = 0;
        uint64_t m_Publications = 0;
        uint64_t m_Duplicates = 0;
        uint64_t m_Rejected = 0;
        uint64_t m_Forwarded = 0;
        uint64_t m_Retransmitted = 0;
        size_t m_Sessions = 0;
        size_t m_Online = 0;
        size_t m_RetainedBytes = 0;
        size_t m_Pending = 0;
    };

    struct PeerSnapshot
    {
        std::string m_Identifier;
        std::string m_Host;
        uint16_t m_Port = 0;
        bool m_Connected = false;
        size_t m_QueuedBytes = 0;
    };

    struct LeaseSnapshot
    {
        uint64_t m_Fence = 0;
        uint64_t m_RemainingMs = 0;
    };

    struct ClusterSnapshot
    {
        std::string m_Identifier;
        uint64_t m_Epoch = 0;
        uint32_t m_Shards = 0;
        uint16_t m_Replicas = 0;
        uint16_t m_Fanout = 0;
        size_t m_PendingOperations = 0;
        size_t m_ActiveLeases = 0;
        size_t m_StoredBytes = 0;
        uint64_t m_QuarantineMs = 0;
        std::vector<PeerSnapshot> m_Peers;
        // Only the keys requested by Inspect; never serialized as a lease directory.
        std::map<std::string, LeaseSnapshot> m_Ownership;
    };

    struct SessionSnapshot : TrafficMetrics
    {
        std::string m_Identifier;
        std::string m_Realm;
        std::string m_Transport;
        bool m_Online = false;
        bool m_Authenticated = false;
        bool m_Resumable = false;
        bool m_Owner = false;
        uint64_t m_ObservedAt = 0;
        uint64_t m_Publications = 0;
        uint64_t m_Deliveries = 0;
        std::optional<uint64_t> m_IdleMs;
        std::optional<uint64_t> m_ExpiresInMs;
        std::optional<LeaseSnapshot> m_Lease;
        uint32_t m_KeepaliveSeconds = 0;
        size_t m_Subscriptions = 0;
        size_t m_Registrations = 0;
        size_t m_Responders = 0;
        size_t m_Requesters = 0;
        size_t m_Pending = 0;
        size_t m_PendingBytes = 0;
        size_t m_QueuedBytes = 0;
        size_t m_Replay = 0;
        size_t m_ReplayBytes = 0;
        size_t m_Requests = 0;
        size_t m_ResponseRoutes = 0;
    };

    struct BrokerSnapshot
    {
        std::string m_Identifier;
        std::string m_Instance;
        std::string m_Address;
        std::string m_Transport;
        uint16_t m_Port = 0;
        uint64_t m_CollectedAt = 0;
        uint64_t m_UptimeMs = 0;
        bool m_Durable = false;
        bool m_AuthenticationRequired = false;
        BrokerMetrics m_Metrics;
        size_t m_MaxSessions = 0;
        size_t m_MaxInflight = 0;
        size_t m_MaxRetainedBytes = 0;
        size_t m_MaxStateBytes = 0;
        std::optional<uint64_t> m_StoredBytes;
        size_t m_OmittedSessions = 0;
        std::vector<SessionSnapshot> m_Sessions;
        std::optional<ClusterSnapshot> m_Cluster;
    };

    std::string SnapshotJson(const BrokerSnapshot &snapshot);
} // namespace eMQ::Server

#include "Snapshot.inl"
