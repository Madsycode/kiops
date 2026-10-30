#pragma once
#include <Cluster/Node.h>
#include <Cluster/Clock.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    namespace NodeDetail
    {
        enum class Operation : uint8_t
        {
            Read = 1,
            Grant,
            Write,
            Release,
            Put,
            Scan,
            Inbox,
            Consume
        };

        // Room for the MHF/2 envelope and PUBREQ framing around a reply page.
        constexpr size_t EMPY_CLUSTER_PACKET_RESERVE = 4096;
        constexpr size_t EMPY_CLUSTER_PLACEMENT_CACHE = 4096;
        constexpr size_t EMPY_CLUSTER_TOMBSTONE_SIZE = 8;

        using Targets = std::vector<std::pair<std::string, std::string>>;

        struct Value
        {
            uint64_t m_Fence = 0;
            uint64_t m_Revision = 0;
            Buffer m_Bytes;
            uint64_t m_Expiry = UINT64_MAX;

            auto Stamp() const
            {
                return std::pair(m_Fence, m_Revision);
            }
            void Save(Writer &writer) const
            {
                writer.U64(m_Fence);
                writer.U64(m_Revision);
                writer.U64(m_Expiry);
                WriteBuffer(writer, m_Bytes);
            }
            static Value Load(Reader &reader)
            {
                Value value;
                value.m_Fence = reader.U64();
                value.m_Revision = reader.U64();
                value.m_Expiry = reader.U64();
                value.m_Bytes = ReadBuffer(reader);
                return value;
            }
        };

        struct Lease
        {
            uint64_t m_Fence = 0;
            std::string m_Owner;
            LeaseTimePoint m_Expiry{};
            Value m_Value;
        };

        inline Buffer EncodeDelivery(const Digest &identity, std::string_view sender, std::string_view session,
                                     uint64_t expiry, std::span<const uint8_t> publication)
        {
            Writer writer(32 * 1024 * 1024);
            writer.Bytes(identity);
            writer.String(sender);
            writer.String(session);
            writer.U64(expiry);
            WriteBuffer(writer, publication);
            return writer.Take();
        }

        inline Buffer EncodeDelivery(const ClusterDelivery &delivery)
        {
            return EncodeDelivery(delivery.m_Identity, delivery.m_Sender, delivery.m_Session, delivery.m_Expiry,
                                  Codec::Encode(delivery.m_Publication, {true, UINT32_MAX}));
        }

        // Reads only the record prefix; the publication is not decoded.
        inline uint64_t InboxExpiry(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            if (bytes.size() == EMPY_CLUSTER_TOMBSTONE_SIZE)
            {
                return reader.U64();
            }
            reader.Bytes(32);
            reader.String();
            reader.String();
            return reader.U64();
        }

        inline ClusterDelivery DecodeDelivery(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            ClusterDelivery result;
            auto identity = reader.Bytes(32);
            std::copy(identity.begin(), identity.end(), result.m_Identity.begin());
            result.m_Sender = reader.String();
            result.m_Session = reader.String();
            result.m_Expiry = reader.U64();
            result.m_Publication = std::get<PublishRequest>(Codec::Decode(ReadBuffer(reader), {true, UINT32_MAX}));
            reader.Finish();
            return result;
        }

        inline std::string InboxPrefix(std::string_view session)
        {
            return "in/" + std::string(session) + '/';
        }

        // Inbox keys are the session prefix followed by a 64-character hexadecimal identity.
        inline std::string InboxSession(std::string_view key)
        {
            if (!key.starts_with("in/") || key.size() < 3 + 1 + 1 + 64 || key[key.size() - 65] != '/')
            {
                throw ProtocolError("Malformed inbox record key");
            }
            return std::string(key.substr(3, key.size() - 3 - 65));
        }
    } // namespace NodeDetail
    using namespace NodeDetail;

    struct FabricNode::Impl
    {
        struct Link
        {
            Connection::Pointer m_Connection;
            std::string m_Expected;
            std::string m_Peer;
            bool m_Outgoing = false;
            bool m_Ready = false;
        };
        struct Pending
        {
            std::string m_Peer;
            Buffer m_Packet;
            LeaseTimePoint m_Deadline;
            bool m_Sent = false;
            std::function<void(bool, Buffer)> m_Done;
        };
        struct Owned
        {
            uint64_t m_Fence = 0;
            uint64_t m_Revision = 0;
            LeaseTimePoint m_Expiry{};
            Buffer m_Bytes;
            bool m_Renewing = false;
        };
        // Replies of one quorum operation; callers wake as soon as its outcome is decided.
        struct Round
        {
            std::mutex m_Mutex;
            std::condition_variable m_Ready;
            std::vector<Buffer> m_Replies;
            size_t m_Failures = 0;
            size_t m_Total = 0;
            std::chrono::steady_clock::time_point m_Deadline{};
        };
        struct Usage
        {
            size_t m_Bytes = 0;
            size_t m_Records = 0;
        };
        using Placement = std::shared_ptr<const std::vector<std::string>>;

        FabricNode &m_Owner;
        ClusterOptions m_Options;
        Fabric m_Fabric;
        FileStore m_Store;
        IoContext m_Io;
        IoTimer m_Timer;
        std::map<std::string, ClusterPeer> m_Members;
        std::map<std::string, std::shared_ptr<Link>> m_Links;
        std::set<std::shared_ptr<Link>> m_Connections;
        std::map<std::string, LeaseTimePoint> m_Dialing;
        std::map<uint64_t, Pending> m_Pending;
        std::map<std::string, Lease> m_Leases;
        std::map<std::string, Owned> m_Owned;
        std::map<std::string, std::set<uint32_t>> m_ProfileShards;
        std::unordered_map<std::string, Usage> m_Inboxes;
        std::mutex m_PlacementMutex;
        std::unordered_map<std::string, Placement> m_Placements;
        std::string m_Incarnation;
        Digest m_Manifest{};
        LeaseTimePoint m_BootGuard{};
        uint64_t m_Counter = 0;
        uint64_t m_ProfileVersion = 0;
        bool m_Started = false;
        bool m_Stopping = false;
        uint64_t m_NextSweep = 0;

        Impl(FabricNode &node, ClusterOptions settings)
            : m_Owner(node), m_Options(std::move(settings)), m_Fabric(m_Options.m_Fabric),
              m_Store(m_Options.m_Directory, m_Options.m_MaxBytes, m_Options.m_MaxRecords, true), m_Timer(m_Io)
        {
            if (m_Options.m_Identifier.empty() || m_Options.m_Identifier.size() > 100 ||
                m_Options.m_EpochKey.size() < 32 || m_Options.m_LeaseMilliseconds < 300 ||
                !m_Options.m_TimeoutMilliseconds ||
                m_Options.m_TimeoutMilliseconds >= m_Options.m_LeaseMilliseconds / 2 || !m_Options.m_MaxOperations ||
                !m_Options.m_Transport.m_Mutual || (!m_Options.m_Transport.m_Tls && !m_Options.m_Transport.m_Quic))
            {
                throw std::invalid_argument(
                    "Cluster needs mutual TLS/QUIC, epoch credentials and valid lease/resource bounds");
            }
            m_Options.m_Transport.m_Backplane = true;
            m_Options.m_Transport.m_QuicPolicy = QuicPolicy{MMQP_QUIC_STRATEGY_BULK, BulkPolicy{4, 65536}, 1200};
            if (m_Options.m_Transport.m_MaxPacket < 1024 * 1024)
            {
                throw std::invalid_argument("Cluster packet budget is too small");
            }
            std::set<std::string> certificates;
            for (auto peer : m_Options.m_Peers)
            {
                if (!peer.m_Port || peer.m_Host.empty() || FromHex(peer.m_Fingerprint).size() != 32 ||
                    !certificates.insert(peer.m_Fingerprint).second ||
                    !m_Members.emplace(peer.m_Identifier, peer).second)
                {
                    throw std::invalid_argument(
                        "Cluster identities require distinct pinned certificates and valid endpoints");
                }
            }
            if (m_Members.size() != m_Fabric.Options().m_Brokers.size() || !m_Members.contains(m_Options.m_Identifier))
            {
                throw std::invalid_argument("Cluster endpoints must describe the complete configured membership");
            }
            Writer writer(65536);
            writer.U32(6); // Persistent backplane schema, with reusable request and responder identifiers.
            writer.U64(m_Fabric.Options().m_Epoch);
            writer.U32(m_Fabric.Options().m_Shards);
            writer.U16(m_Fabric.Options().m_Replicas);
            writer.U16(m_Fabric.Options().m_Fanout);
            for (const auto &broker : m_Fabric.Options().m_Brokers)
            {
                if (!m_Members.contains(broker))
                {
                    throw std::invalid_argument("Missing cluster member endpoint");
                }
                writer.String(broker);
                writer.String(m_Members.at(broker).m_Fingerprint);
            }
            m_Manifest = Sha256(writer.Data());
            if (auto previous = m_Store.Get("manifest"))
            {
                if (!SecureEqual(*previous, m_Manifest))
                {
                    throw std::invalid_argument(
                        "State directory belongs to another schema/membership/epoch; migrate it explicitly");
                }
                // A restarted voter cannot remember a monotonic lease deadline.
                m_BootGuard = LeaseClock::now() + std::chrono::milliseconds(m_Options.m_LeaseMilliseconds);
            }
            m_Store.Put("manifest", Buffer(m_Manifest.begin(), m_Manifest.end()));
            m_Incarnation = m_Options.m_Identifier + ':' + ToHex(RandomBytes(16));
            for (const auto &key : m_Store.Keys("lease/"))
            {
                auto bytes = m_Store.Find(key);
                Reader reader(*bytes);
                Lease lease;
                lease.m_Fence = reader.U64();
                lease.m_Value = Value::Load(reader);
                reader.Finish();
                m_Leases.emplace(key.substr(6), std::move(lease));
            }
            if (!m_Options.m_MaxInboxBytes)
            {
                m_Options.m_MaxInboxBytes =
                    std::min(m_Options.m_MaxBytes,
                             std::max<size_t>(m_Options.m_MaxBytes / 16, m_Options.m_Transport.m_MaxPacket));
            }
            if (!m_Options.m_MaxInboxRecords)
            {
                m_Options.m_MaxInboxRecords = std::max<size_t>(m_Options.m_MaxRecords / 16, 1);
            }
            for (const auto &key : m_Store.Keys("in/"))
            {
                if (auto bytes = m_Store.Find(key); bytes->size() != EMPY_CLUSTER_TOMBSTONE_SIZE)
                {
                    this->Track(key, bytes->size(), true);
                }
            }
        }

        ~Impl()
        {
            SecureErase(m_Options.m_EpochKey);
        }

        size_t Majority() const
        {
            return m_Fabric.Options().m_Replicas / 2 + 1;
        }

        // Charges or releases a live inbox record against its session budget.
        void Track(std::string_view key, size_t bytes, bool add)
        {
            auto session = InboxSession(key);
            auto &usage = m_Inboxes[session];
            if (add)
            {
                usage.m_Bytes += bytes;
                ++usage.m_Records;
            }
            else
            {
                usage.m_Bytes -= std::min(usage.m_Bytes, bytes);
                if (usage.m_Records)
                {
                    --usage.m_Records;
                }
            }
            if (!usage.m_Records)
            {
                m_Inboxes.erase(session);
            }
        }

        // Session placement hashes every broker; a bounded cache avoids repeating it per operation.
        Placement SessionReplicas(const std::string &key)
        {
            {
                std::lock_guard lock(m_PlacementMutex);
                if (auto found = m_Placements.find(key); found != m_Placements.end())
                {
                    return found->second;
                }
            }
            auto replicas = std::make_shared<const std::vector<std::string>>(m_Fabric.SessionReplicas(key));
            std::lock_guard lock(m_PlacementMutex);
            if (m_Placements.size() >= EMPY_CLUSTER_PLACEMENT_CACHE)
            {
                m_Placements.erase(m_Placements.begin());
            }
            m_Placements.emplace(key, replicas);
            return replicas;
        }

        void Persist(const std::string &key, const Lease &lease)
        {
            Writer writer(m_Options.m_MaxBytes);
            writer.U64(lease.m_Fence);
            lease.m_Value.Save(writer);
            m_Store.Put("lease/" + key, writer.Take());
        }

        Buffer Packet(uint8_t kind, uint64_t id, std::span<const uint8_t> body, bool data = false, uint8_t qos = 1)
        {
            Writer writer(m_Options.m_Transport.m_MaxPacket);
            writer.U32(0x4D484632);
            writer.U64(m_Fabric.Options().m_Epoch);
            writer.U8(kind);
            writer.U64(id);
            writer.String(m_Options.m_Identifier);
            WriteBuffer(writer, body);
            auto tag = Hmac256(m_Options.m_EpochKey, writer.Data());
            writer.Bytes(tag);
            PublishRequest publication;
            publication.m_Topic = data ? "$mhf/data" : "$mhf/control";
            publication.m_Alias = 1;
            publication.m_Qos = qos;
            publication.m_PacketId = qos ? 1 : 0;
            publication.m_Payload = writer.Take();
            return Codec::Encode(publication, {false, m_Options.m_Transport.m_MaxPacket});
        }

        void Bind(Connection::Pointer connection, std::string expected = {})
        {
            auto link = std::make_shared<Link>();
            link->m_Connection = std::move(connection);
            link->m_Expected = std::move(expected);
            link->m_Outgoing = !link->m_Expected.empty();
            m_Connections.insert(link);
            if (link->m_Connection->Quic())
            {
                link->m_Connection->Configure(m_Options.m_Transport.m_QuicPolicy, false);
            }
            std::weak_ptr<Link> weak = link;
            link->m_Connection->m_OnReceive = [this, weak](auto bytes)
            {
                if (auto link = weak.lock())
                {
                    try
                    {
                        this->Receive(link, bytes);
                    }
                    catch (const std::exception &error)
                    {
                        EMPY_WARN("Backplane record rejected: {}", error.what());
                        link->m_Connection->Close(false);
                    }
                }
            };
            link->m_Connection->m_OnClose = [this, weak](const auto &)
            {
                auto link = weak.lock();
                if (!link)
                {
                    return;
                }
                if (auto found = m_Links.find(link->m_Peer); found != m_Links.end() && found->second == link)
                {
                    m_Links.erase(found);
                }
                m_Connections.erase(link);
                for (auto &[id, request] : m_Pending)
                {
                    if (request.m_Peer == link->m_Peer)
                    {
                        request.m_Sent = false;
                    }
                }
            };
            if (link->m_Outgoing)
            {
                link->m_Connection->Send(this->Packet(0, 0, m_Manifest));
            }
        }

        void Dial(const std::string &peer)
        {
            if (peer == m_Options.m_Identifier || m_Links.contains(peer) || m_Stopping)
            {
                return;
            }
            auto now = LeaseClock::now();
            if (m_Dialing.contains(peer) && m_Dialing.at(peer) > now)
            {
                return;
            }
            m_Dialing[peer] = now + std::chrono::milliseconds(500);
            auto endpoint = m_Members.at(peer);
            auto transport = m_Options.m_Transport;
            transport.m_ServerName = endpoint.m_ServerName;
            try
            {
                m_Io.Connect(
                    endpoint.m_Host, endpoint.m_Port, transport,
                    [this, peer](auto connection) { this->Bind(std::move(connection), peer); },
                    []([[maybe_unused]] const auto &error) { EMPY_DEBUG("Backplane connection closed: {}", error); });
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("Backplane connect failed: {}", error.what());
            }
        }

        void Receive(const std::shared_ptr<Link> &link, std::span<const uint8_t> frame)
        {
            auto publication =
                std::get<PublishRequest>(Codec::Decode(frame, {false, m_Options.m_Transport.m_MaxPacket}));
            auto bytes = publication.m_Payload.Span();
            if (bytes.size() < 32)
            {
                throw ProtocolError("Short backplane record");
            }
            auto body = std::span<const uint8_t>(bytes).first(bytes.size() - 32);
            if (!SecureEqual(Hmac256(m_Options.m_EpochKey, body), std::span<const uint8_t>(bytes).last(32)))
            {
                throw ProtocolError("Backplane MAC failed");
            }
            Reader reader(body);
            if (reader.U32() != 0x4D484632 || reader.U64() != m_Fabric.Options().m_Epoch)
            {
                throw ProtocolError("Backplane version/epoch mismatch");
            }
            auto kind = reader.U8();
            auto id = reader.U64();
            auto peer = reader.String();
            auto payload = ReadBuffer(reader);
            reader.Finish();
            if (!m_Members.contains(peer) || peer == m_Options.m_Identifier ||
                (!link->m_Expected.empty() && link->m_Expected != peer) ||
                link->m_Connection->PeerFingerprint() != m_Members.at(peer).m_Fingerprint)
            {
                throw ProtocolError("Backplane certificate does not bind the claimed broker");
            }
            if (!link->m_Ready)
            {
                if (kind != 0 || !SecureEqual(payload, m_Manifest))
                {
                    throw ProtocolError("Backplane membership handshake failed");
                }
                link->m_Peer = peer;
                link->m_Ready = true;
                auto old = m_Links.find(peer);
                bool preferred = link->m_Outgoing == (m_Options.m_Identifier < peer);
                if (old != m_Links.end() && old->second != link)
                {
                    if (!preferred)
                    {
                        link->m_Connection->Close(false);
                        return;
                    }
                    auto previous = old->second;
                    m_Links.erase(old);
                    previous->m_Connection->Close(false);
                }
                m_Links[peer] = link;
                if (!link->m_Outgoing)
                {
                    link->m_Connection->Send(this->Packet(0, 0, m_Manifest));
                }
                this->Flush(peer);
                return;
            }
            if (peer != link->m_Peer)
            {
                throw ProtocolError("Backplane identity changed");
            }
            if (kind == 2 || kind == 3)
            {
                auto found = m_Pending.find(id);
                if (found == m_Pending.end() || found->second.m_Peer != peer)
                {
                    return;
                }
                auto done = std::move(found->second.m_Done);
                m_Pending.erase(found);
                done(kind == 2, std::move(payload));
            }
            else if (kind == 1)
            {
                try
                {
                    link->m_Connection->Send(this->Packet(2, id, this->Handle(peer, payload)));
                }
                catch (const std::exception &error)
                {
                    EMPY_DEBUG("Backplane operation rejected: {}", error.what());
                    link->m_Connection->Send(this->Packet(3, id, {}));
                }
            }
            else if (kind == 4)
            {
                this->Forward(peer, std::move(payload));
            }
            else
            {
                throw ProtocolError("Unknown backplane record kind");
            }
        }

        void Forward(const std::string &peer, Buffer bytes)
        {
            Reader reader(bytes);
            auto length = reader.U32();
            auto envelope = reader.Bytes(length);
            // Decode verifies the MAC before parsing and binds the identity to the payload.
            auto record = FabricCodec::Decode(envelope, m_Options.m_EpochKey, m_Options.m_Transport.m_MaxPacket,
                                              m_Fabric.Options().m_MaxDestinations);
            auto tree = FabricCodec::Validate(record, m_Fabric, peer, m_Options.m_Identifier, UnixMilliseconds());
            auto rank = tree.Rank(m_Options.m_Identifier);
            auto count = reader.U32();
            if (count > m_Options.m_MaxRecords)
            {
                throw ProtocolError("Forward target budget exceeded");
            }
            std::unordered_set<std::string_view> destinations(record.m_Destinations.begin(),
                                                              record.m_Destinations.end());
            Targets targets;
            targets.reserve(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                auto broker = reader.String();
                auto session = reader.String();
                if (!destinations.contains(broker))
                {
                    throw ProtocolError("Forward target is absent from destination tree");
                }
                targets.emplace_back(std::move(broker), std::move(session));
            }
            reader.Finish();
            this->Deliver(record, targets);
            auto children = tree.Children(rank);
            if (children.empty())
            {
                return; // Leaves neither re-encode nor re-authenticate the record.
            }
            FabricCodec::Rebind(std::span<uint8_t>(bytes).subspan(4, length), record, static_cast<uint32_t>(rank),
                                m_Options.m_EpochKey);
            this->Send(tree, children, bytes, record.m_Publication.m_Qos);
        }

        void Deliver(const ForwardRecord &record, const Targets &targets)
        {
            auto now = LeaseClock::now();
            for (const auto &[broker, session] : targets)
            {
                if (broker != m_Options.m_Identifier)
                {
                    continue;
                }
                auto found = m_Owned.find(session);
                if (found == m_Owned.end() || found->second.m_Expiry <= now)
                {
                    continue;
                }
                ClusterDelivery delivery{record.m_Identity, record.m_Session, session, record.m_Publication,
                                         record.m_Expires};
                if (!record.m_Publication.m_Qos && m_Owner.m_OnTransient)
                {
                    m_Owner.m_OnTransient(std::move(delivery));
                }
                else if (m_Owner.m_OnDelivery)
                {
                    m_Owner.m_OnDelivery(std::move(delivery));
                }
            }
        }

        void Send(const DisseminationTree &tree, const std::vector<size_t> &children, std::span<const uint8_t> bytes,
                  uint8_t qos)
        {
            auto packet = this->Packet(4, 0, bytes, true, qos);
            for (auto child : children)
            {
                const auto &destination = tree.m_Brokers[child];
                this->Dial(destination);
                if (auto link = m_Links.find(destination); link != m_Links.end())
                {
                    link->second->m_Connection->Send(packet);
                }
                // Reliable destinations recover from their replicated inbox if this branch is unavailable.
            }
        }

        void Flush(const std::string &peer)
        {
            if (!m_Links.contains(peer))
            {
                this->Dial(peer);
                return;
            }
            for (auto &[id, request] : m_Pending)
            {
                if (request.m_Peer == peer && !request.m_Sent)
                {
                    request.m_Sent = m_Links.at(peer)->m_Connection->Send(request.m_Packet);
                }
            }
        }

        void Call(const std::string &peer, Operation operation, std::span<const uint8_t> bytes,
                  std::function<void(bool, Buffer)> done)
        {
            Writer writer(m_Options.m_Transport.m_MaxPacket);
            writer.U8(static_cast<uint8_t>(operation));
            writer.Bytes(bytes);
            if (peer == m_Options.m_Identifier)
            {
                try
                {
                    auto result = this->Handle(peer, writer.Data());
                    done(true, std::move(result));
                }
                catch (...)
                {
                    done(false, {});
                }
                return;
            }
            if (m_Stopping || m_Pending.size() >= m_Options.m_MaxOperations || m_Counter == UINT64_MAX)
            {
                done(false, {});
                return;
            }
            auto id = ++m_Counter;
            m_Pending.emplace(id,
                              Pending{peer, this->Packet(1, id, writer.Data(), operation == Operation::Inbox),
                                      LeaseClock::now() + std::chrono::milliseconds(m_Options.m_TimeoutMilliseconds),
                                      false, std::move(done)});
            this->Flush(peer);
        }

        // Starts one operation on every replica from a caller thread; Await collects the outcome.
        std::shared_ptr<Round> Begin(const std::vector<std::string> &replicas, Operation operation, Buffer bytes)
        {
            auto round = std::make_shared<Round>();
            round->m_Total = replicas.size();
            round->m_Deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(m_Options.m_TimeoutMilliseconds + 100);
            m_Io.Post(
                [this, replicas, operation, bytes = std::move(bytes), round]
                {
                    for (const auto &replica : replicas)
                    {
                        this->Call(replica, operation, bytes,
                                   [round](bool success, Buffer result)
                                   {
                                       {
                                           std::lock_guard lock(round->m_Mutex);
                                           if (success)
                                           {
                                               round->m_Replies.push_back(std::move(result));
                                           }
                                           else
                                           {
                                               ++round->m_Failures;
                                           }
                                       }
                                       round->m_Ready.notify_all();
                                   });
                    }
                });
            return round;
        }

        std::vector<Buffer> Await(const std::shared_ptr<Round> &round)
        {
            auto majority = this->Majority();
            std::unique_lock lock(round->m_Mutex);
            round->m_Ready.wait_until(
                lock, round->m_Deadline,
                [&] { return round->m_Replies.size() >= majority || round->m_Total - round->m_Failures < majority; });
            if (round->m_Replies.size() < majority)
            {
                throw ProtocolError("Cluster majority unavailable", MMQP_REASON_BUSY);
            }
            return std::move(round->m_Replies);
        }

        std::vector<Buffer> Quorum(const std::vector<std::string> &replicas, Operation operation, Buffer bytes)
        {
            return this->Await(this->Begin(replicas, operation, std::move(bytes)));
        }

        // Issues independent quorum writes concurrently, bounded by the pending-operation budget.
        // Every request must reach its majority; the first failure is reported after issuing its window.
        void Replicate(std::vector<std::pair<std::vector<std::string>, Buffer>> requests, Operation operation)
        {
            auto window = std::max<size_t>(1, m_Options.m_MaxOperations / (2 * m_Fabric.Options().m_Replicas));
            for (size_t first = 0; first < requests.size(); first += window)
            {
                std::vector<std::shared_ptr<Round>> rounds;
                for (size_t i = first; i < std::min(requests.size(), first + window); ++i)
                {
                    rounds.push_back(this->Begin(requests[i].first, operation, std::move(requests[i].second)));
                }
                for (const auto &round : rounds)
                {
                    this->Await(round);
                }
            }
        }

        bool Replica(const std::string &key)
        {
            auto replicas = this->SessionReplicas(key);
            return std::find(replicas->begin(), replicas->end(), m_Options.m_Identifier) != replicas->end();
        }

        Buffer Handle(const std::string &peer, std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            auto operation = static_cast<Operation>(reader.U8());
            auto key = reader.String();
            Writer reply(m_Options.m_Transport.m_MaxPacket);
            if (operation == Operation::Read || operation == Operation::Grant || operation == Operation::Write ||
                operation == Operation::Release)
            {
                if (!this->Replica(key) || key.size() > 240)
                {
                    throw ProtocolError("Wrong session replica");
                }
                if (!m_Leases.contains(key) && m_Leases.size() >= m_Options.m_MaxRecords / 2)
                {
                    throw ProtocolError("Lease table budget exhausted");
                }
                auto &lease = m_Leases[key];
                if (operation == Operation::Read)
                {
                    reader.Finish();
                    reply.U64(lease.m_Fence);
                    lease.m_Value.Save(reply);
                    return reply.Take();
                }
                auto fence = reader.U64();
                auto claimant = reader.String();
                if (!claimant.starts_with(peer + ':'))
                {
                    throw ProtocolError("Lease claimant identity mismatch");
                }
                auto now = LeaseClock::now();
                if (now < m_BootGuard)
                {
                    throw ProtocolError("Restart lease quarantine is active");
                }
                if (operation == Operation::Grant)
                {
                    reader.Finish();
                    if (!fence || fence < lease.m_Fence || (lease.m_Expiry > now && lease.m_Owner != claimant) ||
                        (fence == lease.m_Fence && !lease.m_Owner.empty() && lease.m_Owner != claimant))
                    {
                        throw ProtocolError("Lease is fenced");
                    }
                    if (fence != lease.m_Fence)
                    {
                        lease.m_Fence = fence;
                        this->Persist(key, lease);
                    }
                    lease.m_Owner = claimant;
                    lease.m_Expiry = now + std::chrono::milliseconds(m_Options.m_LeaseMilliseconds);
                    lease.m_Value.Save(reply);
                    return reply.Take();
                }
                if (lease.m_Fence != fence || lease.m_Owner != claimant || lease.m_Expiry <= now)
                {
                    throw ProtocolError("Lease is no longer valid");
                }
                if (operation == Operation::Release)
                {
                    reader.Finish();
                    lease.m_Owner.clear();
                    lease.m_Expiry = now;
                    return {};
                }
                auto value = Value::Load(reader);
                reader.Finish();
                if (value.m_Expiry <= UnixMilliseconds() || value.m_Fence != fence || !value.m_Revision ||
                    value.Stamp() < lease.m_Value.Stamp() ||
                    (value.Stamp() == lease.m_Value.Stamp() && value.m_Bytes != lease.m_Value.m_Bytes))
                {
                    throw ProtocolError("Conflicting snapshot revision");
                }
                auto updated = lease;
                updated.m_Value = std::move(value);
                this->Persist(key, updated);
                lease = std::move(updated);
                return {};
            }
            if (operation == Operation::Scan)
            {
                // A page resumes after the last key of the previous page and stays within the packet budget.
                auto after = reader.String();
                reader.Finish();
                if (!key.starts_with("profile/") && !key.starts_with("in/") && !key.starts_with("ret/"))
                {
                    throw ProtocolError("Invalid cluster scan");
                }
                if (!after.empty() && !after.starts_with(key))
                {
                    throw ProtocolError("Invalid cluster scan resume key");
                }
                auto keys = m_Store.Keys(key);
                auto item = after.empty() ? keys.begin() : std::upper_bound(keys.begin(), keys.end(), after);
                auto budget = m_Options.m_Transport.m_MaxPacket - EMPY_CLUSTER_PACKET_RESERVE;
                size_t size = 4 + 1;
                std::vector<std::pair<std::string, Buffer>> page;
                for (; item != keys.end(); ++item)
                {
                    auto value = m_Store.Get(*item);
                    auto length = 1 + item->size() + 4 + value->size();
                    if (!page.empty() && (size + length > budget || page.size() >= m_Options.m_MaxRecords))
                    {
                        break;
                    }
                    size += length;
                    page.emplace_back(*item, std::move(*value));
                }
                reply.U32(static_cast<uint32_t>(page.size()));
                for (const auto &[name, value] : page)
                {
                    reply.String(name);
                    WriteBuffer(reply, value);
                }
                reply.U8(item != keys.end());
                return reply.Take();
            }
            if (operation == Operation::Put)
            {
                if (!key.starts_with("profile/") && !key.starts_with("ret/"))
                {
                    throw ProtocolError("Invalid replica record kind");
                }
                auto value = Value::Load(reader);
                reader.Finish();
                if (value.m_Expiry <= UnixMilliseconds())
                {
                    return {}; // Delayed expired index updates cannot resurrect compacted state.
                }
                if (auto previous = m_Store.Find(key))
                {
                    Reader input(*previous);
                    auto old = Value::Load(input);
                    if (old.Stamp() > value.Stamp())
                    {
                        return {};
                    }
                    if (old.Stamp() == value.Stamp())
                    {
                        if (old.m_Bytes != value.m_Bytes)
                        {
                            throw ProtocolError("Conflicting replica record");
                        }
                        return {};
                    }
                }
                value.Save(reply);
                m_Store.Put(key, reply.Take());
                return {};
            }
            if (operation == Operation::Inbox)
            {
                auto value = ReadBuffer(reader);
                reader.Finish();
                auto delivery = DecodeDelivery(value);
                if (delivery.m_Expiry <= UnixMilliseconds() ||
                    key != InboxPrefix(delivery.m_Session) + ToHex(delivery.m_Identity) ||
                    !this->Replica(delivery.m_Session))
                {
                    throw ProtocolError("Invalid inbox destination or expiry");
                }
                if (auto previous = m_Store.Find(key))
                {
                    if (previous->size() == 8)
                    {
                        return {}; // Durable consumed tombstone.
                    }
                    auto old = DecodeDelivery(*previous);
                    if (old.m_Sender != delivery.m_Sender || old.m_Session != delivery.m_Session ||
                        Codec::Encode(old.m_Publication, {true, UINT32_MAX}) !=
                            Codec::Encode(delivery.m_Publication, {true, UINT32_MAX}))
                    {
                        throw ProtocolError("Inbox identity conflict");
                    }
                }
                else
                {
                    // Q_bytes + b(m) <= B and Q_records + 1 <= N, in addition to the global store budget.
                    auto found = m_Inboxes.find(delivery.m_Session);
                    auto usage = found == m_Inboxes.end() ? Usage{} : found->second;
                    if (usage.m_Records >= m_Options.m_MaxInboxRecords || usage.m_Bytes > m_Options.m_MaxInboxBytes ||
                        value.size() > m_Options.m_MaxInboxBytes - usage.m_Bytes)
                    {
                        throw ProtocolError("Session inbox budget exhausted", MMQP_REASON_BUSY);
                    }
                    auto size = value.size();
                    m_Store.Put(key, std::move(value));
                    this->Track(key, size, true);
                }
                if (m_Owned.contains(delivery.m_Session) &&
                    m_Owned.at(delivery.m_Session).m_Expiry > LeaseClock::now() && m_Owner.m_OnDelivery)
                {
                    m_Owner.m_OnDelivery(std::move(delivery));
                }
                return {};
            }
            if (operation == Operation::Consume)
            {
                auto session = reader.String();
                auto fence = reader.U64();
                auto claimant = reader.String();
                auto expiry = reader.U64();
                reader.Finish();
                if (!this->Replica(session) || !key.starts_with(InboxPrefix(session)) ||
                    key.size() != InboxPrefix(session).size() + 64)
                {
                    throw ProtocolError("Wrong session replica");
                }
                auto now = LeaseClock::now();
                if (now < m_BootGuard)
                {
                    throw ProtocolError("Restart lease quarantine is active");
                }
                auto lease = m_Leases.find(session);
                if (lease == m_Leases.end() || lease->second.m_Fence != fence || lease->second.m_Owner != claimant ||
                    lease->second.m_Expiry <= now || !claimant.starts_with(peer + ':'))
                {
                    throw ProtocolError("Consumed inbox record is not owned");
                }
                auto previous = m_Store.Find(key);
                auto size = previous ? previous->size() : EMPY_CLUSTER_TOMBSTONE_SIZE;
                reply.U64(expiry);
                m_Store.Put(key, reply.Take());
                if (size != EMPY_CLUSTER_TOMBSTONE_SIZE)
                {
                    this->Track(key, size, false);
                }
                return {};
            }
            throw ProtocolError("Unknown cluster operation");
        }

        void Tick()
        {
            if (m_Stopping)
            {
                return;
            }
            m_Timer.Schedule(std::chrono::milliseconds(25), [this] { this->Tick(); });
            auto now = LeaseClock::now();
            std::vector<std::function<void()>> completed;
            std::set<std::string> retry;
            for (auto it = m_Pending.begin(); it != m_Pending.end();)
            {
                if (it->second.m_Deadline <= now)
                {
                    completed.push_back([done = std::move(it->second.m_Done)] { done(false, {}); });
                    it = m_Pending.erase(it);
                }
                else
                {
                    retry.insert(it->second.m_Peer);
                    ++it;
                }
            }
            for (auto &done : completed)
            {
                done();
            }
            for (const auto &peer : retry)
            {
                this->Flush(peer);
            }
            auto wall = UnixMilliseconds();
            if (wall >= m_NextSweep)
            {
                m_NextSweep = wall + 1000;
                for (const auto &key : m_Store.Keys("in/"))
                {
                    auto bytes = m_Store.Find(key);
                    if (InboxExpiry(*bytes) > wall)
                    {
                        continue;
                    }
                    auto size = bytes->size();
                    m_Store.Erase(key);
                    if (size != EMPY_CLUSTER_TOMBSTONE_SIZE)
                    {
                        this->Track(key, size, false);
                    }
                }
                for (auto it = m_Leases.begin(); it != m_Leases.end();)
                {
                    bool active = it->second.m_Expiry > now ||
                                  (m_Owned.contains(it->first) && m_Owned.at(it->first).m_Expiry > now);
                    if (!active && it->second.m_Value.m_Expiry < wall &&
                        wall - it->second.m_Value.m_Expiry > m_Options.m_LeaseMilliseconds * 2ULL)
                    {
                        m_Store.Erase("lease/" + it->first);
                        m_Owned.erase(it->first);
                        it = m_Leases.erase(it);
                    }
                    else
                    {
                        ++it;
                    }
                }
                for (const auto &prefix : {"profile/", "ret/"})
                {
                    for (const auto &key : m_Store.Keys(prefix))
                    {
                        // Values start with fence, revision and expiry; the snapshot itself is not copied.
                        auto bytes = m_Store.Find(key);
                        Reader reader(*bytes);
                        reader.U64();
                        reader.U64();
                        auto expiry = reader.U64();
                        if (expiry < wall && wall - expiry > m_Options.m_LeaseMilliseconds * 2ULL)
                        {
                            m_Store.Erase(key);
                        }
                    }
                }
            }
            for (auto &[key, lease] : m_Owned)
            {
                if (lease.m_Renewing || lease.m_Expiry <= now ||
                    lease.m_Expiry > now + std::chrono::milliseconds(m_Options.m_LeaseMilliseconds / 6))
                {
                    continue;
                }
                lease.m_Renewing = true;
                auto fence = lease.m_Fence;
                auto deadline = now + std::chrono::milliseconds(m_Options.m_LeaseMilliseconds / 3);
                Writer writer;
                writer.String(key);
                writer.U64(fence);
                writer.String(m_Incarnation);
                auto acknowledgements = std::make_shared<size_t>(0);
                auto remaining = std::make_shared<size_t>(m_Fabric.Options().m_Replicas);
                for (const auto &replica : *this->SessionReplicas(key))
                {
                    this->Call(replica, Operation::Grant, writer.Data(),
                               [this, key, fence, deadline, acknowledgements, remaining](bool success, Buffer)
                               {
                                   if (!m_Owned.contains(key) || m_Owned.at(key).m_Fence != fence)
                                   {
                                       return;
                                   }
                                   auto &value = m_Owned.at(key);
                                   if (success && ++*acknowledgements >= this->Majority() &&
                                       value.m_Expiry > LeaseClock::now())
                                   {
                                       value.m_Expiry = deadline;
                                   }
                                   if (--*remaining == 0)
                                   {
                                       value.m_Renewing = false;
                                   }
                               });
                }
            }
        }

        // Each page is a quorum read. Every reply covers its keys up to its last key, so keys up to
        // the smallest truncated boundary are covered by a full majority before the next page.
        std::map<std::string, Buffer> Scan(const std::vector<std::string> &replicas, const std::string &prefix,
                                           bool versioned)
        {
            std::map<std::string, Buffer> records;
            std::string after;
            while (true)
            {
                Writer request;
                request.String(prefix);
                request.String(after);
                auto replies = this->Quorum(replicas, Operation::Scan, request.Take());
                std::vector<std::vector<std::pair<std::string, Buffer>>> pages;
                std::optional<std::string> boundary;
                for (const auto &bytes : replies)
                {
                    Reader reader(bytes);
                    auto count = reader.U32();
                    if (count > m_Options.m_MaxRecords)
                    {
                        throw ProtocolError("Cluster scan budget exceeded");
                    }
                    auto &page = pages.emplace_back();
                    page.reserve(count);
                    for (uint32_t i = 0; i < count; ++i)
                    {
                        auto key = reader.String();
                        if (!key.starts_with(prefix) || key <= (page.empty() ? after : page.back().first))
                        {
                            throw ProtocolError("Cluster scan prefix or order mismatch");
                        }
                        page.emplace_back(std::move(key), ReadBuffer(reader));
                    }
                    auto more = reader.U8() != 0;
                    reader.Finish();
                    if (!more)
                    {
                        continue;
                    }
                    if (page.empty())
                    {
                        throw ProtocolError("Cluster scan page made no progress");
                    }
                    if (!boundary || page.back().first < *boundary)
                    {
                        boundary = page.back().first;
                    }
                }
                for (auto &page : pages)
                {
                    for (auto &[key, value] : page)
                    {
                        if (boundary && key > *boundary)
                        {
                            break;
                        }
                        this->Merge(records, std::move(key), std::move(value), versioned);
                    }
                }
                if (records.size() > m_Options.m_MaxRecords)
                {
                    throw ProtocolError("Cluster scan budget exceeded");
                }
                if (!boundary)
                {
                    return records;
                }
                after = std::move(*boundary);
            }
        }

        void Merge(std::map<std::string, Buffer> &records, std::string key, Buffer value, bool versioned)
        {
            auto found = records.find(key);
            if (found == records.end())
            {
                records.emplace(std::move(key), std::move(value));
            }
            else if (versioned)
            {
                Reader before(found->second), after(value);
                auto old = Value::Load(before), incoming = Value::Load(after);
                if (incoming.Stamp() > old.Stamp())
                {
                    found->second = std::move(value);
                }
                else if (incoming.Stamp() == old.Stamp() && incoming.m_Bytes != old.m_Bytes)
                {
                    throw ProtocolError("Replica equivocation");
                }
            }
            else if (value.size() == EMPY_CLUSTER_TOMBSTONE_SIZE)
            {
                found->second = std::move(value);
            }
            else if (found->second.size() != EMPY_CLUSTER_TOMBSTONE_SIZE && found->second != value)
            {
                throw ProtocolError("Inbox replica conflict");
            }
        }
    };

    inline FabricNode::FabricNode(ClusterOptions options) : m_Impl(std::make_unique<Impl>(*this, std::move(options)))
    {
    }
    inline FabricNode::~FabricNode()
    {
        this->Stop();
    }

    inline uint16_t FabricNode::Start()
    {
        if (m_Impl->m_Started || m_Impl->m_Stopping)
        {
            throw std::logic_error("Cluster node already started or stopped");
        }
        auto port = m_Impl->m_Io.Listen(
            m_Impl->m_Options.m_Address, m_Impl->m_Options.m_Port, m_Impl->m_Options.m_Transport,
            [this](auto connection) { m_Impl->Bind(std::move(connection)); }, m_Impl->m_Members.size() * 4);
        m_Impl->m_Started = true;
        m_Impl->m_Io.Post([this] { m_Impl->Tick(); });
        m_Impl->m_Io.Init();
        return port;
    }

    inline void FabricNode::Stop()
    {
        if (!m_Impl || m_Impl->m_Stopping)
        {
            return;
        }
        m_Impl->m_Io.Stop(
            [this]
            {
                m_Impl->m_Stopping = true;
                m_Impl->m_Timer.Cancel();
                auto pending = std::move(m_Impl->m_Pending);
                for (auto &[id, request] : pending)
                {
                    request.m_Done(false, {});
                }
            });
    }

    inline size_t FabricNode::Connected()
    {
        return m_Impl->m_Io.Submit([this] { return m_Impl->m_Links.size(); }).get();
    }

    inline ClusterSnapshot FabricNode::Inspect(const std::vector<std::string> &sessions)
    {
        auto collect = [this, &sessions]
        {
            ClusterSnapshot result;
            result.m_Identifier = ToHex(m_Impl->m_Manifest);
            const auto &options = m_Impl->m_Options;
            auto now = LeaseClock::now();
            auto remaining = [now](auto deadline) -> uint64_t
            {
                return deadline > now ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count()
                                      : 0;
            };
            result.m_Epoch = options.m_Fabric.m_Epoch;
            result.m_Shards = options.m_Fabric.m_Shards;
            result.m_Replicas = options.m_Fabric.m_Replicas;
            result.m_Fanout = options.m_Fabric.m_Fanout;
            result.m_PendingOperations = m_Impl->m_Pending.size();
            result.m_StoredBytes = m_Impl->m_Store.Bytes();
            result.m_QuarantineMs = remaining(m_Impl->m_BootGuard);
            for (const auto &[key, lease] : m_Impl->m_Owned)
            {
                if (lease.m_Expiry > now)
                {
                    ++result.m_ActiveLeases;
                }
            }
            for (const auto &id : sessions)
            {
                if (auto found = m_Impl->m_Owned.find(id);
                    found != m_Impl->m_Owned.end() && found->second.m_Expiry > now)
                {
                    result.m_Ownership.emplace(id,
                                               LeaseSnapshot{found->second.m_Fence, remaining(found->second.m_Expiry)});
                }
            }
            for (const auto &[id, member] : m_Impl->m_Members)
            {
                if (id == options.m_Identifier)
                {
                    continue;
                }
                auto link = m_Impl->m_Links.find(id);
                bool connected = link != m_Impl->m_Links.end() && link->second->m_Ready;
                result.m_Peers.push_back({id, member.m_Host, member.m_Port, connected,
                                          connected ? link->second->m_Connection->Queued() : 0});
            }
            return result;
        };
        return m_Impl->m_Started && !m_Impl->m_Stopping ? m_Impl->m_Io.Submit(collect).get() : collect();
    }

    inline const Fabric &FabricNode::Placement() const
    {
        return m_Impl->m_Fabric;
    }
    inline const std::string &FabricNode::Identifier() const
    {
        return m_Impl->m_Options.m_Identifier;
    }

    inline bool FabricNode::Owns(std::string_view key)
    {
        return m_Impl->m_Io
            .Submit([this, key = std::string(key)]
                    { return m_Impl->m_Owned.contains(key) && m_Impl->m_Owned.at(key).m_Expiry > LeaseClock::now(); })
            .get();
    }

    inline Buffer FabricNode::Acquire(std::string key, uint32_t timeout)
    {
        if (this->Owns(key))
        {
            return m_Impl->m_Io.Submit([this, key] { return m_Impl->m_Owned.at(key).m_Bytes; }).get();
        }
        auto deadline = LeaseClock::now() + std::chrono::milliseconds(timeout);
        auto replicas = m_Impl->SessionReplicas(key);
        while (LeaseClock::now() < deadline)
        {
            try
            {
                Writer read;
                read.String(key);
                auto responses = m_Impl->Quorum(*replicas, Operation::Read, read.Take());
                uint64_t maximum = 0;
                for (const auto &bytes : responses)
                {
                    Reader reader(bytes);
                    maximum = std::max(maximum, reader.U64());
                }
                auto count = m_Impl->m_Members.size();
                auto rank = static_cast<size_t>(
                    std::distance(m_Impl->m_Members.begin(), m_Impl->m_Members.find(m_Impl->m_Options.m_Identifier)));
                if (maximum > UINT64_MAX - 2 * count)
                {
                    throw std::overflow_error("Lease fence exhausted");
                }
                auto fence = (maximum / count + 1) * count + rank;
                Writer claim;
                claim.String(key);
                claim.U64(fence);
                claim.String(m_Impl->m_Incarnation);
                auto started = LeaseClock::now();
                auto votes = m_Impl->Quorum(*replicas, Operation::Grant, claim.Take());
                Value newest;
                for (const auto &bytes : votes)
                {
                    Reader reader(bytes);
                    auto value = Value::Load(reader);
                    reader.Finish();
                    if (value.Stamp() > newest.Stamp())
                    {
                        newest = std::move(value);
                    }
                    else if (value.Stamp() == newest.Stamp() && value.m_Bytes != newest.m_Bytes)
                    {
                        throw ProtocolError("Snapshot replicas disagree");
                    }
                }
                auto expiry = started + std::chrono::milliseconds(m_Impl->m_Options.m_LeaseMilliseconds / 3);
                if (expiry <= LeaseClock::now())
                {
                    continue;
                }
                m_Impl->m_Io
                    .Submit([this, key, fence, expiry, bytes = newest.m_Bytes]
                            { m_Impl->m_Owned[key] = Impl::Owned{fence, 0, expiry, bytes, false}; })
                    .get();
                if (newest.m_Expiry <= UnixMilliseconds())
                {
                    newest.m_Bytes.clear();
                }
                auto retention = newest.m_Bytes.empty() ? UnixMilliseconds() + m_Impl->m_Options.m_LeaseMilliseconds
                                                        : newest.m_Expiry;
                this->Commit(key, newest.m_Bytes, retention); // Adopt an accepted successor before serving.
                return newest.m_Bytes;
            }
            catch (const ProtocolError &)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        }
        throw ProtocolError("No majority could grant exclusive session ownership", MMQP_REASON_BUSY);
    }

    inline void FabricNode::Commit(std::string key, Buffer snapshot, uint64_t expiry)
    {
        auto owned = m_Impl->m_Io
                         .Submit(
                             [this, key]
                             {
                                 auto found = m_Impl->m_Owned.find(key);
                                 if (found == m_Impl->m_Owned.end() || found->second.m_Expiry <= LeaseClock::now() ||
                                     found->second.m_Revision == UINT64_MAX)
                                 {
                                     throw ProtocolError("Session primary is fenced", MMQP_REASON_BUSY);
                                 }
                                 ++found->second.m_Revision;
                                 return found->second;
                             })
                         .get();
        Writer writer(m_Impl->m_Options.m_Transport.m_MaxPacket);
        writer.String(key);
        writer.U64(owned.m_Fence);
        writer.String(m_Impl->m_Incarnation);
        Value{owned.m_Fence, owned.m_Revision, snapshot, expiry}.Save(writer);
        try
        {
            m_Impl->Quorum(*m_Impl->SessionReplicas(key), Operation::Write, writer.Take());
            m_Impl->m_Io
                .Submit(
                    [this, key, fence = owned.m_Fence, bytes = std::move(snapshot)]() mutable
                    {
                        auto &lease = m_Impl->m_Owned.at(key);
                        if (lease.m_Fence != fence || lease.m_Expiry <= LeaseClock::now())
                        {
                            throw ProtocolError("Session lease expired during commit", MMQP_REASON_BUSY);
                        }
                        lease.m_Bytes = std::move(bytes);
                    })
                .get();
        }
        catch (...)
        {
            m_Impl->m_Io.Submit([this, key] { m_Impl->m_Owned.erase(key); }).get();
            throw;
        }
    }

    inline Buffer FabricNode::Read(std::string key)
    {
        Writer writer;
        writer.String(key);
        Value newest;
        for (const auto &bytes : m_Impl->Quorum(*m_Impl->SessionReplicas(key), Operation::Read, writer.Take()))
        {
            Reader reader(bytes);
            reader.U64();
            auto value = Value::Load(reader);
            reader.Finish();
            if (value.Stamp() > newest.Stamp())
            {
                newest = std::move(value);
            }
        }
        return newest.m_Expiry > UnixMilliseconds() ? newest.m_Bytes : Buffer{};
    }

    inline void FabricNode::Release(std::string key)
    {
        auto lease = m_Impl->m_Io
                         .Submit(
                             [this, key]
                             {
                                 auto found = m_Impl->m_Owned.find(key);
                                 auto result = found == m_Impl->m_Owned.end() ? Impl::Owned{} : found->second;
                                 m_Impl->m_Owned.erase(key);
                                 return result;
                             })
                         .get();
        if (!lease.m_Fence)
        {
            return;
        }
        Writer writer;
        writer.String(key);
        writer.U64(lease.m_Fence);
        writer.String(m_Impl->m_Incarnation);
        try
        {
            m_Impl->Quorum(*m_Impl->SessionReplicas(key), Operation::Release, writer.Take());
        }
        catch (const ProtocolError &)
        {
        } // Expiry still fences the previous owner.
    }

    inline void FabricNode::Profiles(std::string session, const std::vector<ProfileEntry> &profiles, uint64_t expiry,
                                     bool online)
    {
        auto lease = m_Impl->m_Io
                         .Submit(
                             [this, session]
                             {
                                 if (!m_Impl->m_Owned.contains(session) ||
                                     m_Impl->m_Owned.at(session).m_Expiry <= LeaseClock::now())
                                 {
                                     throw ProtocolError("Profile owner is fenced");
                                 }
                                 return m_Impl->m_Owned.at(session);
                             })
                         .get();
        // Placements are computed once per profile; shards sharing a profile set share one encoding.
        std::map<uint32_t, std::vector<size_t>> placement;
        for (size_t i = 0; i < profiles.size(); ++i)
        {
            if (profiles[i].m_Type == MMQP_PROREQ_TYPE_SUB)
            {
                for (auto shard : m_Impl->m_Fabric.ProfileShards(profiles[i].m_Topic))
                {
                    placement[shard].push_back(i);
                }
            }
        }
        std::set<uint32_t> shards;
        for (const auto &[shard, members] : placement)
        {
            shards.insert(shards.end(), shard);
        }
        auto previous = m_Impl->m_Io
                            .Submit(
                                [this, session, shards]
                                {
                                    auto old = m_Impl->m_ProfileShards[session];
                                    m_Impl->m_ProfileShards[session] = shards;
                                    return old;
                                })
                            .get();
        shards.insert(previous.begin(), previous.end());
        std::map<std::vector<size_t>, Buffer> snapshots;
        std::vector<std::pair<std::vector<std::string>, Buffer>> requests;
        requests.reserve(shards.size());
        for (auto shard : shards)
        {
            Writer body(m_Impl->m_Options.m_Transport.m_MaxPacket);
            body.String(m_Impl->m_Options.m_Identifier);
            body.String(session);
            body.U64(expiry);
            body.U8(online);
            auto found = placement.find(shard);
            body.U8(found != placement.end());
            if (found != placement.end())
            {
                auto &snapshot = snapshots[found->second];
                if (snapshot.empty())
                {
                    std::vector<ProfileEntry> entries;
                    entries.reserve(found->second.size());
                    for (auto index : found->second)
                    {
                        entries.push_back(profiles[index]);
                    }
                    snapshot = Codec::Encode(ProfileRequest{1, std::move(entries)}, {false, UINT32_MAX, UINT32_MAX});
                }
                WriteBuffer(body, snapshot);
            }
            Writer request(m_Impl->m_Options.m_Transport.m_MaxPacket);
            request.String("profile/" + std::to_string(shard) + '/' + session);
            Value{lease.m_Fence, lease.m_Revision, body.Take(), expiry}.Save(request);
            requests.emplace_back(m_Impl->m_Fabric.Replicas(shard), request.Take());
        }
        m_Impl->Replicate(std::move(requests), Operation::Put);
    }

    inline std::vector<ClusterProfile> FabricNode::Resolve(std::string topic, uint8_t qos, bool responders)
    {
        topic = ParseTopic(topic).m_Filter;
        auto shard = m_Impl->m_Fabric.Shard(topic);
        auto records = m_Impl->Scan(m_Impl->m_Fabric.Replicas(shard), "profile/" + std::to_string(shard) + '/', true);
        std::vector<ClusterProfile> result;
        for (const auto &[key, bytes] : records)
        {
            Reader record(bytes);
            auto value = Value::Load(record);
            Reader reader(value.m_Bytes);
            auto broker = reader.String();
            auto session = reader.String();
            auto expiry = reader.U64();
            auto online = reader.U8() != 0;
            if (expiry <= UnixMilliseconds() || (!online && (!qos || responders)) || !reader.U8())
            {
                continue;
            }
            auto profiles =
                std::get<ProfileRequest>(Codec::Decode(ReadBuffer(reader), {false, UINT32_MAX, UINT32_MAX}));
            reader.Finish();
            for (const auto &profile : profiles.m_Entries)
            {
                if (profile.m_Type == MMQP_PROREQ_TYPE_SUB && qos <= profile.m_Qos &&
                    (!responders || profile.m_Request) && MatchTopic(profile.m_Topic, topic))
                {
                    result.push_back({broker, session, profile, expiry, online});
                }
            }
        }
        return result;
    }

    inline void FabricNode::Publish(const std::string &sender, uint64_t sequence, PublishRequest publication,
                                    const std::vector<ClusterProfile> &targets, uint64_t expiry)
    {
        publication.m_Alias = 1;
        publication.m_Duplicate = false;
        auto identity = MessageIdentity(sender, sequence, publication.m_Topic, m_Impl->m_Fabric.Options().m_Epoch,
                                        publication.m_Payload);
        std::map<std::string, ClusterProfile> distinct;
        for (const auto &target : targets)
        {
            distinct.emplace(target.m_Session, target);
        }
        if (publication.m_Qos)
        {
            // All recipient inbox writes run concurrently; acceptance still needs every majority.
            auto encoded = Codec::Encode(publication, {true, UINT32_MAX});
            std::vector<std::pair<std::vector<std::string>, Buffer>> requests;
            requests.reserve(distinct.size());
            for (const auto &[session, target] : distinct)
            {
                Writer request(m_Impl->m_Options.m_Transport.m_MaxPacket);
                request.String(InboxPrefix(session) + ToHex(identity));
                WriteBuffer(request, EncodeDelivery(identity, sender, session, expiry, encoded));
                requests.emplace_back(*m_Impl->SessionReplicas(session), request.Take());
            }
            m_Impl->Replicate(std::move(requests), Operation::Inbox);
        }
        ForwardRecord record;
        record.m_Identity = identity;
        record.m_AcceptedEpoch = record.m_RoutingEpoch = m_Impl->m_Fabric.Options().m_Epoch;
        record.m_Sequence = sequence;
        record.m_Expires = expiry;
        record.m_Session = sender;
        record.m_Ingress = m_Impl->m_Options.m_Identifier;
        record.m_Publication = std::move(publication);
        std::set<std::string> brokers;
        Targets forwarded;
        for (const auto &[session, target] : distinct)
        {
            if (!target.m_Broker.empty())
            {
                brokers.insert(target.m_Broker);
                forwarded.emplace_back(target.m_Broker, session);
            }
        }
        if (brokers.empty())
        {
            return;
        }
        record.m_Destinations.assign(brokers.begin(), brokers.end());
        // The ingress already holds the record: it is encoded once, and only when it has children.
        auto tree = m_Impl->m_Fabric.Tree(record.m_Publication.m_Topic, record.m_Ingress, record.m_Destinations);
        auto children = tree.Children(0);
        Buffer bytes;
        if (!children.empty())
        {
            Writer writer(m_Impl->m_Options.m_Transport.m_MaxPacket);
            WriteBuffer(writer, FabricCodec::Encode(record, m_Impl->m_Options.m_EpochKey,
                                                    m_Impl->m_Options.m_Transport.m_MaxPacket));
            writer.U32(static_cast<uint32_t>(forwarded.size()));
            for (const auto &[broker, session] : forwarded)
            {
                writer.String(broker);
                writer.String(session);
            }
            bytes = writer.Take();
        }
        m_Impl->m_Io
            .Submit(
                [this, &record, &forwarded, &tree, &children, &bytes]
                {
                    m_Impl->Deliver(record, forwarded);
                    if (!children.empty())
                    {
                        m_Impl->Send(tree, children, bytes, record.m_Publication.m_Qos);
                    }
                })
            .get();
    }

    inline std::vector<ClusterDelivery> FabricNode::Inbox(std::string session)
    {
        auto records = m_Impl->Scan(*m_Impl->SessionReplicas(session), InboxPrefix(session), false);
        std::vector<ClusterDelivery> result;
        for (const auto &[key, bytes] : records)
        {
            if (bytes.size() == 8)
            {
                continue;
            }
            auto delivery = DecodeDelivery(bytes);
            if (delivery.m_Expiry > UnixMilliseconds())
            {
                result.push_back(std::move(delivery));
            }
        }
        std::sort(result.begin(), result.end(),
                  [](const auto &left, const auto &right)
                  {
                      return std::tie(left.m_Sender, left.m_Publication.m_PacketId) <
                             std::tie(right.m_Sender, right.m_Publication.m_PacketId);
                  });
        return result;
    }

    inline void FabricNode::Acknowledge(const std::vector<ClusterDelivery> &deliveries)
    {
        std::vector<std::shared_ptr<Impl::Round>> rounds;
        for (const auto &delivery : deliveries)
        {
            try
            {
                auto lease = m_Impl->m_Io
                                 .Submit(
                                     [this, session = delivery.m_Session]
                                     {
                                         if (!m_Impl->m_Owned.contains(session))
                                         {
                                             throw ProtocolError("Inbox owner is fenced");
                                         }
                                         return m_Impl->m_Owned.at(session);
                                     })
                                 .get();
                Writer request;
                request.String(InboxPrefix(delivery.m_Session) + ToHex(delivery.m_Identity));
                request.String(delivery.m_Session);
                request.U64(lease.m_Fence);
                request.String(m_Impl->m_Incarnation);
                request.U64(delivery.m_Expiry);
                rounds.push_back(
                    m_Impl->Begin(*m_Impl->SessionReplicas(delivery.m_Session), Operation::Consume, request.Take()));
            }
            catch (const ProtocolError &)
            {
            }
            if (rounds.size() >=
                std::max<size_t>(1, m_Impl->m_Options.m_MaxOperations / (2 * m_Impl->m_Fabric.Options().m_Replicas)))
            {
                for (const auto &round : rounds)
                {
                    try
                    {
                        m_Impl->Await(round);
                    }
                    catch (const ProtocolError &)
                    {
                    }
                }
                rounds.clear();
            }
        }
        for (const auto &round : rounds)
        {
            try
            {
                m_Impl->Await(round);
            }
            catch (const ProtocolError &)
            {
            }
        }
    }

    inline void FabricNode::Acknowledge(const ClusterDelivery &delivery)
    {
        auto lease = m_Impl->m_Io
                         .Submit(
                             [this, session = delivery.m_Session]
                             {
                                 if (!m_Impl->m_Owned.contains(session))
                                 {
                                     throw ProtocolError("Inbox owner is fenced");
                                 }
                                 return m_Impl->m_Owned.at(session);
                             })
                         .get();
        Writer request;
        request.String(InboxPrefix(delivery.m_Session) + ToHex(delivery.m_Identity));
        request.String(delivery.m_Session);
        request.U64(lease.m_Fence);
        request.String(m_Impl->m_Incarnation);
        request.U64(delivery.m_Expiry);
        m_Impl->Quorum(*m_Impl->SessionReplicas(delivery.m_Session), Operation::Consume, request.Take());
    }

    inline void FabricNode::Retain(const PublishRequest &publication, const ProfileEntry &profile,
                                   const Digest &identity)
    {
        auto shard = m_Impl->m_Fabric.Shard(publication.m_Topic);
        auto key =
            "ret/" + std::to_string(shard) + '/' +
            ToHex(Sha256({reinterpret_cast<const uint8_t *>(publication.m_Topic.data()), publication.m_Topic.size()}));
        auto lock =
            "retain:" +
            ToHex(Sha256({reinterpret_cast<const uint8_t *>(publication.m_Topic.data()), publication.m_Topic.size()}));
        auto previous = this->Acquire(lock);
        std::vector<ClusterDelivery> records;
        bool duplicate = false;
        if (!previous.empty())
        {
            Reader reader(previous);
            auto count = reader.U32();
            if (count > 255)
            {
                throw ProtocolError("Retained queue exceeds profile limit");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto item = DecodeDelivery(ReadBuffer(reader));
                if (item.m_Identity == identity)
                {
                    duplicate = true;
                }
                if (item.m_Expiry > UnixMilliseconds())
                {
                    records.push_back(std::move(item));
                }
            }
            reader.Finish();
        }
        auto limit = profile.m_Qos ? profile.m_Queue : 1;
        if (!duplicate)
        {
            while (records.size() >= static_cast<size_t>(limit))
            {
                records.erase(records.begin());
            }
            records.push_back(
                {identity, {}, {}, publication, UnixMilliseconds() + DecodeInterval(profile.m_Expiry) * 1000ULL});
        }
        Writer body(m_Impl->m_Options.m_Transport.m_MaxPacket);
        body.U32(static_cast<uint32_t>(records.size()));
        for (const auto &item : records)
        {
            WriteBuffer(body, EncodeDelivery(item));
        }
        uint64_t expiry = 0;
        for (const auto &record : records)
        {
            expiry = std::max(expiry, record.m_Expiry);
        }
        // The lock record and the shard index carry the same fenced revision, so both quorum
        // writes run concurrently; the release then proceeds without being awaited.
        auto lease = m_Impl->m_Io
                         .Submit(
                             [this, lock]
                             {
                                 auto found = m_Impl->m_Owned.find(lock);
                                 if (found == m_Impl->m_Owned.end() || found->second.m_Expiry <= LeaseClock::now() ||
                                     found->second.m_Revision == UINT64_MAX)
                                 {
                                     throw ProtocolError("Retained queue lock is fenced", MMQP_REASON_BUSY);
                                 }
                                 ++found->second.m_Revision;
                                 return found->second;
                             })
                         .get();
        Value value{lease.m_Fence, lease.m_Revision, body.Take(), expiry};
        Writer commit(m_Impl->m_Options.m_Transport.m_MaxPacket);
        commit.String(lock);
        commit.U64(lease.m_Fence);
        commit.String(m_Impl->m_Incarnation);
        value.Save(commit);
        Writer index(m_Impl->m_Options.m_Transport.m_MaxPacket);
        index.String(key);
        value.Save(index);
        try
        {
            auto first = m_Impl->Begin(*m_Impl->SessionReplicas(lock), Operation::Write, commit.Take());
            auto second = m_Impl->Begin(m_Impl->m_Fabric.Replicas(shard), Operation::Put, index.Take());
            m_Impl->Await(first);
            m_Impl->Await(second);
        }
        catch (...)
        {
            m_Impl->m_Io.Submit([this, lock] { m_Impl->m_Owned.erase(lock); }).get();
            throw;
        }
        m_Impl->m_Io.Submit([this, lock] { m_Impl->m_Owned.erase(lock); }).get();
        Writer release;
        release.String(lock);
        release.U64(lease.m_Fence);
        release.String(m_Impl->m_Incarnation);
        // Lease expiry still fences this owner if the release is lost.
        try
        {
            m_Impl->Begin(*m_Impl->SessionReplicas(lock), Operation::Release, release.Take());
        }
        catch (const ProtocolError &)
        {
        }
    }

    inline std::vector<ClusterDelivery> FabricNode::Retained(const std::vector<ProfileEntry> &profiles)
    {
        std::set<uint32_t> shards;
        for (const auto &profile : profiles)
        {
            if (profile.m_Type == MMQP_PROREQ_TYPE_SUB && profile.m_Qos && !profile.m_Request)
            {
                for (auto shard : m_Impl->m_Fabric.ProfileShards(profile.m_Topic))
                {
                    shards.insert(shard);
                }
            }
        }
        std::set<Digest> seen;
        std::vector<ClusterDelivery> result;
        for (auto shard : shards)
        {
            auto records = m_Impl->Scan(m_Impl->m_Fabric.Replicas(shard), "ret/" + std::to_string(shard) + '/', true);
            for (const auto &[key, bytes] : records)
            {
                Reader record(bytes);
                auto value = Value::Load(record);
                Reader reader(value.m_Bytes);
                auto count = reader.U32();
                if (count > 255)
                {
                    throw ProtocolError("Retained queue exceeds profile limit");
                }
                for (uint32_t i = 0; i < count; ++i)
                {
                    auto delivery = DecodeDelivery(ReadBuffer(reader));
                    if (delivery.m_Expiry <= UnixMilliseconds())
                    {
                        continue;
                    }
                    for (const auto &profile : profiles)
                    {
                        if (profile.m_Type == MMQP_PROREQ_TYPE_SUB && profile.m_Qos && !profile.m_Request &&
                            MatchTopic(profile.m_Topic, delivery.m_Publication.m_Topic))
                        {
                            if (seen.insert(delivery.m_Identity).second)
                            {
                                result.push_back(std::move(delivery));
                            }
                            break;
                        }
                    }
                }
                reader.Finish();
            }
        }
        return result;
    }
} // namespace eMQ::Server
