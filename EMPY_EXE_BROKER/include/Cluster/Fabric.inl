#pragma once
#include <Cluster/Fabric.h>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    namespace FabricDetail
    {
        inline std::span<const uint8_t> Bytes(std::string_view value)
        {
            return {reinterpret_cast<const uint8_t *>(value.data()), value.size()};
        }

        inline void BrokerName(std::string_view name)
        {
            if (name.empty() || name.size() > 255 || !ValidUtf8(name))
            {
                throw std::invalid_argument("Service identity must be 1..255 UTF-8 bytes");
            }
        }
    } // namespace FabricDetail
    using namespace FabricDetail;

    inline size_t DisseminationTree::Rank(std::string_view broker) const
    {
        auto found = std::find(m_Brokers.begin(), m_Brokers.end(), broker);
        if (found == m_Brokers.end())
        {
            throw ProtocolError("Service is not in the forwarding tree", MMQP_REASON_UNAUTHORIZED);
        }
        return static_cast<size_t>(found - m_Brokers.begin());
    }

    inline std::optional<size_t> DisseminationTree::Parent(size_t rank) const
    {
        if (rank >= m_Brokers.size() || !m_Fanout)
        {
            throw std::out_of_range("Invalid tree rank or fanout");
        }
        return rank ? std::optional((rank - 1) / m_Fanout) : std::nullopt;
    }

    inline std::vector<size_t> DisseminationTree::Children(size_t rank) const
    {
        if (rank >= m_Brokers.size() || !m_Fanout)
        {
            throw std::out_of_range("Invalid tree rank or fanout");
        }
        std::vector<size_t> children;
        for (size_t child = rank * m_Fanout + 1; child < m_Brokers.size() && children.size() < m_Fanout; ++child)
        {
            children.push_back(child);
        }
        return children;
    }

    inline std::vector<size_t> DisseminationTree::Subtree(size_t rank) const
    {
        if (rank >= m_Brokers.size())
        {
            throw std::out_of_range("Invalid tree rank");
        }
        std::vector<size_t> nodes{rank};
        for (size_t offset = 0; offset < nodes.size(); ++offset)
        {
            auto children = this->Children(nodes[offset]);
            nodes.insert(nodes.end(), children.begin(), children.end());
        }
        return nodes;
    }

    inline bool DisseminationTree::Edge(std::string_view sender, std::string_view receiver) const
    {
        auto parent = this->Parent(this->Rank(receiver));
        return parent && *parent == this->Rank(sender);
    }

    inline Fabric::Fabric(FabricOptions options) : m_Options(std::move(options))
    {
        if (!m_Options.m_Epoch || !m_Options.m_Shards || m_Options.m_Shards > 65536 || !m_Options.m_Replicas ||
            m_Options.m_Replicas > m_Options.m_Brokers.size() || !m_Options.m_Fanout || m_Options.m_Fanout > 64 ||
            !m_Options.m_MaxDestinations || m_Options.m_MaxDestinations > UINT16_MAX)
        {
            throw std::invalid_argument("Invalid fabric epoch, shards, replica count or fanout");
        }
        for (const auto &broker : m_Options.m_Brokers)
        {
            BrokerName(broker);
        }
        std::sort(m_Options.m_Brokers.begin(), m_Options.m_Brokers.end());
        if (std::adjacent_find(m_Options.m_Brokers.begin(), m_Options.m_Brokers.end()) != m_Options.m_Brokers.end())
        {
            throw std::invalid_argument("Duplicate broker identity");
        }
        // Epoch and membership are static, so shard placement is computed once.
        m_Shards.reserve(m_Options.m_Shards);
        for (uint32_t shard = 0; shard < m_Options.m_Shards; ++shard)
        {
            Writer writer(8);
            writer.U8(1);
            writer.U32(shard);
            m_Shards.push_back(this->Select(writer.Data()));
        }
    }

    inline uint32_t Fabric::Shard(std::string_view topic) const
    {
        auto canonical = ParseTopic(topic).m_Filter;
        return static_cast<uint32_t>(Hash64(Bytes(canonical)) % m_Options.m_Shards);
    }

    inline std::vector<uint32_t> Fabric::ProfileShards(std::string_view filter) const
    {
        auto canonical = ParseTopic(filter, true).m_Filter;
        if (!HasWildcard(canonical))
        {
            return {this->Shard(canonical)};
        }
        std::vector<uint32_t> shards;
        shards.reserve(m_Options.m_Shards);
        for (uint32_t shard = 0; shard < m_Options.m_Shards; ++shard)
        {
            shards.push_back(shard);
        }
        return shards;
    }

    inline std::vector<std::string> Fabric::Select(std::span<const uint8_t> key) const
    {
        std::vector<std::pair<uint64_t, std::string>> ranked;
        for (const auto &broker : m_Options.m_Brokers)
        {
            Writer writer(1024);
            writer.Bytes(key);
            writer.U64(m_Options.m_Epoch);
            writer.String(broker);
            ranked.emplace_back(Hash64(writer.Data()), broker);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto &left, const auto &right)
                  { return left.first != right.first ? left.first > right.first : left.second < right.second; });
        std::vector<std::string> selected;
        for (uint16_t i = 0; i < m_Options.m_Replicas; ++i)
        {
            selected.push_back(ranked[i].second);
        }
        return selected;
    }

    inline const std::vector<std::string> &Fabric::Replicas(uint32_t shard) const
    {
        if (shard >= m_Options.m_Shards)
        {
            throw std::out_of_range("Unknown profile shard");
        }
        return m_Shards[shard];
    }

    inline std::vector<std::string> Fabric::SessionReplicas(std::string_view identifier) const
    {
        if (identifier.empty() || !ValidUtf8(identifier))
        {
            throw std::invalid_argument("Invalid session identity");
        }
        Writer writer(257);
        writer.U8(2);
        writer.String(identifier);
        return this->Select(writer.Data());
    }

    inline DisseminationTree Fabric::Tree(std::string_view topic, std::string_view ingress,
                                          const std::vector<std::string> &destinations) const
    {
        auto canonical = ParseTopic(topic).m_Filter;
        if (destinations.size() > m_Options.m_MaxDestinations)
        {
            throw ProtocolError("Tree exceeds destination budget", MMQP_REASON_POLICY_VIOLATION);
        }
        if (!std::binary_search(m_Options.m_Brokers.begin(), m_Options.m_Brokers.end(), std::string(ingress)))
        {
            throw ProtocolError("Ingress is outside this epoch's membership", MMQP_REASON_UNAUTHORIZED);
        }
        std::set<std::string> distinct(destinations.begin(), destinations.end());
        distinct.erase(std::string(ingress));
        std::vector<std::pair<uint64_t, std::string>> ranked;
        for (const auto &broker : distinct)
        {
            if (!std::binary_search(m_Options.m_Brokers.begin(), m_Options.m_Brokers.end(), broker))
            {
                throw ProtocolError("Destination is outside this epoch's membership", MMQP_REASON_UNAUTHORIZED);
            }
            Writer writer(1024);
            writer.String(canonical);
            writer.U64(m_Options.m_Epoch);
            writer.String(broker);
            ranked.emplace_back(Hash64(writer.Data()), broker);
        }
        std::sort(ranked.begin(), ranked.end());
        DisseminationTree tree{{std::string(ingress)}, m_Options.m_Fanout};
        for (const auto &[hash, broker] : ranked)
        {
            tree.m_Brokers.push_back(broker);
        }
        return tree;
    }

    inline const FabricOptions &Fabric::Options() const
    {
        return m_Options;
    }

    inline Digest MessageIdentity(std::string_view session, uint64_t sequence, std::string_view topic,
                                  uint64_t acceptedEpoch, std::span<const uint8_t> payload)
    {
        if (!sequence || !acceptedEpoch || session.empty() || !ValidUtf8(session))
        {
            throw std::invalid_argument("An origin sequence, accepted epoch and session are required");
        }
        Writer writer(1024);
        writer.String("MHF/1/message");
        writer.String(session);
        writer.U64(sequence);
        writer.Bytes(Sha256(payload));
        writer.String(ParseTopic(topic).m_Filter);
        writer.U64(acceptedEpoch);
        return Sha256(writer.Data());
    }

    inline Buffer FabricCodec::Encode(const ForwardRecord &record, std::span<const uint8_t> epochKey, size_t limit)
    {
        if (epochKey.size() < 32 || record.m_Destinations.size() > UINT16_MAX || !record.m_RoutingEpoch ||
            !record.m_Expires)
        {
            throw std::invalid_argument("Invalid forwarding key, epoch or destination count");
        }
        BrokerName(record.m_Ingress);
        if (std::set<std::string>(record.m_Destinations.begin(), record.m_Destinations.end()).size() !=
            record.m_Destinations.size())
        {
            throw ProtocolError("Duplicate forwarding destination");
        }
        if (!SecureEqual(record.m_Identity,
                         MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                         record.m_AcceptedEpoch, record.m_Publication.m_Payload)))
        {
            throw ProtocolError("Forward identity does not match the accepted publication");
        }
        Writer writer(limit);
        writer.String("MHF");
        writer.U8(1);
        writer.U8(1);
        writer.Bytes(record.m_Identity);
        writer.U64(record.m_AcceptedEpoch);
        writer.U64(record.m_RoutingEpoch);
        writer.U64(record.m_Sequence);
        writer.U64(record.m_Expires);
        writer.String(record.m_Session);
        writer.String(record.m_Ingress);
        writer.U32(record.m_SenderRank);
        writer.U16(static_cast<uint16_t>(record.m_Destinations.size()));
        for (const auto &broker : record.m_Destinations)
        {
            BrokerName(broker);
            writer.String(broker);
        }
        auto message = Codec::Encode(record.m_Publication, {true});
        writer.U32(static_cast<uint32_t>(message.size()));
        writer.Bytes(message);
        auto mac = Hmac256(epochKey, writer.Data());
        writer.Bytes(mac);
        return writer.Take();
    }

    inline ForwardRecord FabricCodec::Decode(std::span<const uint8_t> bytes, std::span<const uint8_t> epochKey,
                                             size_t limit, size_t maxDestinations)
    {
        if (epochKey.size() < 32 || bytes.size() > limit || bytes.size() < 32)
        {
            throw ProtocolError("Invalid authenticated forwarding envelope", MMQP_REASON_AUTH_FAILED);
        }
        auto body = bytes.first(bytes.size() - 32);
        if (!SecureEqual(Hmac256(epochKey, body), bytes.last(32)))
        {
            throw ProtocolError("Forwarding record MAC failed", MMQP_REASON_AUTH_FAILED);
        }
        Reader reader(body);
        if (reader.String() != "MHF" || reader.U8() != 1 || reader.U8() != 1)
        {
            throw ProtocolError("Unsupported fabric envelope version", MMQP_REASON_UNSUPPORTED_VERSION);
        }
        ForwardRecord record;
        auto identity = reader.Bytes(32);
        std::copy(identity.begin(), identity.end(), record.m_Identity.begin());
        record.m_AcceptedEpoch = reader.U64();
        record.m_RoutingEpoch = reader.U64();
        record.m_Sequence = reader.U64();
        record.m_Expires = reader.U64();
        record.m_Session = reader.String();
        record.m_Ingress = reader.String();
        BrokerName(record.m_Ingress);
        record.m_SenderRank = reader.U32();
        auto count = reader.U16();
        if (count > maxDestinations)
        {
            throw ProtocolError("Forwarding destination budget exceeded");
        }
        std::set<std::string> unique;
        for (uint16_t i = 0; i < count; ++i)
        {
            auto broker = reader.String();
            BrokerName(broker);
            if (!unique.insert(broker).second)
            {
                throw ProtocolError("Duplicate forwarding destination");
            }
            record.m_Destinations.emplace_back(std::move(broker));
        }
        auto length = reader.U32();
        auto message = Codec::Decode(reader.Bytes(length), {true});
        auto publication = std::get_if<PublishRequest>(&message);
        if (!publication || publication->m_Topic.empty())
        {
            throw ProtocolError("Forwarding requires a concrete topic");
        }
        record.m_Publication = std::move(*publication);
        reader.Finish();
        if (!SecureEqual(record.m_Identity,
                         MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                         record.m_AcceptedEpoch, record.m_Publication.m_Payload)))
        {
            throw ProtocolError("Forwarding identity mismatch", MMQP_REASON_AUTH_FAILED);
        }
        return record;
    }

    inline DisseminationTree FabricCodec::Validate(const ForwardRecord &record, const Fabric &fabric,
                                                   std::string_view authenticatedPeer, std::string_view receiver,
                                                   uint64_t now)
    {
        if (record.m_RoutingEpoch != fabric.Options().m_Epoch || record.m_AcceptedEpoch > record.m_RoutingEpoch ||
            record.m_Expires <= now)
        {
            throw ProtocolError("Stale, future or expired forwarding record", MMQP_REASON_UNAUTHORIZED);
        }
        auto tree = fabric.Tree(record.m_Publication.m_Topic, record.m_Ingress, record.m_Destinations);
        if (record.m_SenderRank >= tree.m_Brokers.size() || tree.m_Brokers[record.m_SenderRank] != authenticatedPeer ||
            !tree.Edge(authenticatedPeer, receiver))
        {
            throw ProtocolError("Forwarding edge does not match the authenticated tree parent",
                                MMQP_REASON_UNAUTHORIZED);
        }
        return tree;
    }

    inline void FabricCodec::Rebind(std::span<uint8_t> bytes, const ForwardRecord &record, uint32_t senderRank,
                                    std::span<const uint8_t> epochKey)
    {
        // The rank follows the fixed header, identity, four integers, session and ingress.
        Writer header(1024);
        header.String("MHF");
        header.U8(1);
        header.U8(1);
        header.Bytes(record.m_Identity);
        header.U64(record.m_AcceptedEpoch);
        header.U64(record.m_RoutingEpoch);
        header.U64(record.m_Sequence);
        header.U64(record.m_Expires);
        header.String(record.m_Session);
        header.String(record.m_Ingress);
        auto offset = header.Data().size();
        if (epochKey.size() < 32 || bytes.size() < offset + 4 + 32 ||
            !std::equal(header.Data().begin(), header.Data().end(), bytes.begin()))
        {
            throw ProtocolError("Forwarding envelope does not match its record");
        }
        uint32_t stored = 0;
        for (size_t i = 0; i < 4; ++i)
        {
            stored = (stored << 8) | bytes[offset + i];
        }
        if (stored != record.m_SenderRank)
        {
            throw ProtocolError("Forwarding envelope does not match its record");
        }
        for (size_t i = 0; i < 4; ++i)
        {
            bytes[offset + i] = static_cast<uint8_t>(senderRank >> (24 - 8 * i));
        }
        auto mac = Hmac256(epochKey, bytes.first(bytes.size() - 32));
        std::copy(mac.begin(), mac.end(), bytes.end() - 32);
    }

    inline EpochFilter::EpochFilter(uint64_t epoch, size_t capacity) : m_Current(epoch), m_Capacity(capacity)
    {
        if (!epoch || !capacity)
        {
            throw std::invalid_argument("Invalid epoch filter bounds");
        }
        m_Epochs.emplace(epoch, std::set<Digest>{});
    }

    inline void EpochFilter::Advance(uint64_t epoch)
    {
        if (epoch <= m_Current)
        {
            throw std::invalid_argument("Cluster epoch must advance");
        }
        m_Current = epoch;
        m_Epochs.emplace(epoch, std::set<Digest>{});
        while (m_Epochs.size() > 2)
        {
            m_Epochs.erase(m_Epochs.begin());
        }
    }

    inline bool EpochFilter::Accept(uint64_t epoch, const Digest &identity)
    {
        auto found = m_Epochs.find(epoch);
        if (found == m_Epochs.end())
        {
            throw ProtocolError("Epoch is outside the deduplication grace window");
        }
        // Identity remains stable when the routing epoch changes.
        for (const auto &[period, identities] : m_Epochs)
        {
            if (identities.contains(identity))
            {
                return false;
            }
        }
        if (found->second.size() >= m_Capacity)
        {
            throw ProtocolError("Epoch duplicate filter is full", MMQP_REASON_BUSY);
        }
        found->second.insert(identity);
        return true;
    }

    inline size_t EpochFilter::Size() const
    {
        size_t size = 0;
        for (const auto &[epoch, identities] : m_Epochs)
        {
            size += identities.size();
        }
        return size;
    }
} // namespace eMQ::Server
