#pragma once
#include <eCore/Network/State.h>
#include <eCore/Network/Authentication.h>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    struct FabricOptions
    {
        uint64_t m_Epoch = 1;
        uint32_t m_Shards = 64;
        uint16_t m_Replicas = 3;
        uint16_t m_Fanout = 2;
        std::vector<std::string> m_Brokers;
        size_t m_MaxDestinations = 4096;
    };

    struct DisseminationTree
    {
        std::vector<std::string> m_Brokers;
        uint16_t m_Fanout = 2;

        size_t Rank(std::string_view broker) const;
        std::optional<size_t> Parent(size_t rank) const;
        std::vector<size_t> Children(size_t rank) const;
        std::vector<size_t> Subtree(size_t rank) const;
        bool Edge(std::string_view sender, std::string_view receiver) const;
    };

    struct Fabric
    {
        explicit Fabric(FabricOptions options);
        uint32_t Shard(std::string_view topic) const;
        std::vector<uint32_t> ProfileShards(std::string_view filter) const;
        const std::vector<std::string> &Replicas(uint32_t shard) const;
        std::vector<std::string> SessionReplicas(std::string_view identifier) const;
        DisseminationTree Tree(std::string_view topic, std::string_view ingress,
                               const std::vector<std::string> &destinations) const;
        const FabricOptions &Options() const;

    private:
        std::vector<std::string> Select(std::span<const uint8_t> key) const;
        FabricOptions m_Options;
        std::vector<std::vector<std::string>> m_Shards;
    };

    Digest MessageIdentity(std::string_view session, uint64_t sequence, std::string_view topic, uint64_t acceptedEpoch,
                           std::span<const uint8_t> payload);

    struct ForwardRecord
    {
        Digest m_Identity{};
        uint64_t m_AcceptedEpoch = 0;
        uint64_t m_RoutingEpoch = 0;
        uint64_t m_Sequence = 0;
        uint64_t m_Expires = 0;
        std::string m_Session;
        std::string m_Ingress;
        uint32_t m_SenderRank = 0;
        std::vector<std::string> m_Destinations;
        PublishRequest m_Publication;
    };

    // Private MHF/1 envelope. This is separate from the nine client packet types.
    struct FabricCodec
    {
        static Buffer Encode(const ForwardRecord &record, std::span<const uint8_t> epochKey,
                             size_t limit = 2 * 1024 * 1024);
        static ForwardRecord Decode(std::span<const uint8_t> bytes, std::span<const uint8_t> epochKey,
                                    size_t limit = 2 * 1024 * 1024, size_t maxDestinations = 4096);
        // Expects a record produced by Decode, which already bound its identity to the payload.
        static DisseminationTree Validate(const ForwardRecord &record, const Fabric &fabric,
                                          std::string_view authenticatedPeer, std::string_view receiver, uint64_t now);
        // Rewrites the sender rank of an encoded record in place and re-authenticates it.
        static void Rebind(std::span<uint8_t> bytes, const ForwardRecord &record, uint32_t senderRank,
                           std::span<const uint8_t> epochKey);
    };

    struct EpochFilter
    {
        explicit EpochFilter(uint64_t epoch, size_t capacity = 65536);
        void Advance(uint64_t epoch);
        bool Accept(uint64_t epoch, const Digest &identity);
        size_t Size() const;

    private:
        uint64_t m_Current;
        size_t m_Capacity;
        std::map<uint64_t, std::set<Digest>> m_Epochs;
    };

    struct TenantPolicy
    {
        std::string m_Root;
        std::set<uint32_t> m_Shards;

        // Mirrors broker admission: an empty allowlist is unrestricted, and a
        // wildcard must be allowed in every shard that it can match.
        bool Allows(const Fabric &fabric, std::string_view topic) const
        {
            auto pattern = ParseTopic(topic, true).m_Filter;
            for (std::string_view prefix : {MMQP_BIRTH_STATUS_TOPIC_PREFIX, MMQP_LAST_STATUS_TOPIC_PREFIX})
            {
                if (pattern.starts_with(prefix))
                {
                    pattern.erase(0, prefix.size());
                    break;
                }
            }
            auto name = m_Root.empty() ? std::string() : ParseTopic(m_Root).m_Filter;
            if (!name.empty() && pattern != name && !pattern.starts_with(name + '/'))
            {
                return false;
            }
            if (m_Shards.empty())
            {
                return true;
            }
            for (auto shard : fabric.ProfileShards(pattern))
            {
                if (!m_Shards.contains(shard))
                {
                    return false;
                }
            }
            return true;
        }
    };

    enum class Admission : uint8_t
    {
        Accept,
        Drop,
        Defer,
        Reject
    };

    struct AdmissionBudget
    {
        size_t m_MaxBytes = 4 * 1024 * 1024;
        size_t m_MaxRecords = 256;

        Admission Reserve(uint8_t qos, size_t bytes, bool unexpired, bool defer = false)
        {
            if (qos > 1 || !unexpired || bytes > m_MaxBytes)
            {
                return qos ? Admission::Reject : Admission::Drop;
            }
            if (m_Records >= m_MaxRecords || m_Bytes > m_MaxBytes || bytes > m_MaxBytes - m_Bytes)
            {
                return !qos ? Admission::Drop : defer ? Admission::Defer : Admission::Reject;
            }
            m_Bytes += bytes;
            ++m_Records;
            return Admission::Accept;
        }

        void Release(size_t bytes)
        {
            if (!m_Records || bytes > m_Bytes)
            {
                throw std::logic_error("Admission budget release mismatch");
            }
            m_Bytes -= bytes;
            --m_Records;
        }

        EMPY_INLINE size_t Bytes() const
        {
            return m_Bytes;
        }
        EMPY_INLINE size_t Records() const
        {
            return m_Records;
        }

    private:
        size_t m_Bytes = 0;
        size_t m_Records = 0;
    };
} // namespace eMQ::Server

#include "Fabric.inl"
