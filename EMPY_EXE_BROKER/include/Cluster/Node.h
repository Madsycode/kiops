#pragma once
#include "Storage.h"
#include <eCore/Network/IOContext.h>
#include "../Monitoring/Snapshot.h"

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    struct ClusterPeer
    {
        std::string m_Identifier;
        std::string m_Host = "127.0.0.1";
        uint16_t m_Port = 0;
        std::string m_ServerName;
        std::string m_Fingerprint;
    };

    struct ClusterOptions
    {
        std::string m_Identifier;
        std::string m_Address = "127.0.0.1";
        uint16_t m_Port = 0;
        std::string m_Directory;
        FabricOptions m_Fabric;
        std::vector<ClusterPeer> m_Peers;
        Buffer m_EpochKey;
        TransportOptions m_Transport;
        uint32_t m_LeaseMilliseconds = 3000;
        uint32_t m_TimeoutMilliseconds = 700;
        size_t m_MaxBytes = 256 * 1024 * 1024;
        size_t m_MaxRecords = 65536;
        size_t m_MaxOperations = 1024;
        // Per-session inbox admission budget on each replica (paper section IV-J).
        // Zero derives max(maxBytes / 16, packet budget) bytes and maxRecords / 16 records.
        size_t m_MaxInboxBytes = 0;
        size_t m_MaxInboxRecords = 0;
    };

    struct ClusterProfile
    {
        std::string m_Broker;
        std::string m_Session;
        ProfileEntry m_Profile;
        uint64_t m_Expiry = 0;
        bool m_Online = false;
    };

    struct ClusterDelivery
    {
        Digest m_Identity{};
        std::string m_Sender;
        std::string m_Session;
        PublishRequest m_Publication;
        uint64_t m_Expiry = 0;
    };

    // Calls below block the caller, never the node's own I/O executor.
    // Service protocol state and backplane state therefore use distinct executors.
    struct FabricNode
    {
        explicit FabricNode(ClusterOptions options);
        ~FabricNode();
        FabricNode(const FabricNode &) = delete;
        FabricNode &operator=(const FabricNode &) = delete;

        uint16_t Start();
        void Stop();
        size_t Connected();
        ClusterSnapshot Inspect(const std::vector<std::string> &sessions = {});
        Buffer Acquire(std::string key, uint32_t timeout = 5000);
        bool Owns(std::string_view key);
        void Commit(std::string key, Buffer snapshot, uint64_t expiry = UINT64_MAX);
        Buffer Read(std::string key);
        void Release(std::string key);

        void Profiles(std::string session, const std::vector<ProfileEntry> &profiles, uint64_t expiry, bool online);
        std::vector<ClusterProfile> Resolve(std::string topic, uint8_t qos, bool responders = false);
        void Publish(const std::string &sender, uint64_t sequence, PublishRequest publication,
                     const std::vector<ClusterProfile> &targets, uint64_t expiry);
        std::vector<ClusterDelivery> Inbox(std::string session);
        void Acknowledge(const ClusterDelivery &delivery);
        // Consumes several inbox records with concurrent quorum rounds; failures are ignored
        // because an unconsumed record is only redelivered and deduplicated by its receipt.
        void Acknowledge(const std::vector<ClusterDelivery> &deliveries);
        void Retain(const PublishRequest &publication, const ProfileEntry &profile, const Digest &identity);
        std::vector<ClusterDelivery> Retained(const std::vector<ProfileEntry> &profiles);
        const Fabric &Placement() const;
        const std::string &Identifier() const;
        std::function<void(ClusterDelivery)> m_OnDelivery;
        std::function<void(ClusterDelivery)> m_OnTransient;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eMQ::Server

#include "Node.inl"
