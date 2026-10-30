#pragma once
#include <Runtime/Service.h>
#include <eCore/Network/State.h>
#include <eCore/Network/IOContext.h>
#include <eCore/Network/Authentication.h>
#include <deque>
#include <mutex>
#include <map>
#include <set>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    struct Service::Impl
    {
        enum class Phase
        {
            Request,
            Challenge,
            Active,
            Closing
        };

        struct Peer
        {
            uint64_t m_Id = 0;
            Connection::Pointer m_Connection;
            Phase m_Phase = Phase::Request;
            CodecOptions m_Codec;
            ConnectionRequest m_Request;
            Buffer m_Expected;
            std::string m_Session;
            std::string m_Realm;
            TimePoint m_Deadline;
        };

        struct ResponseRoute
        {
            std::string m_Requester;
            uint16_t m_CorrelationId = 0;
            std::string m_Topic;
            TimePoint m_Expiry;
        };

        struct Request : ResponseRoute
        {
            std::set<std::string> m_Responders;
            uint8_t m_Qos = 0;
            std::string m_Broker;
        };

        struct Session
        {
            Session() = default;
            static constexpr bool in_place_delete = true;
            std::string m_Identifier;
            SessionProperties m_Properties;
            ProfileTable m_Profiles;
            ReceiveWindow m_Received;
            PublicationSequence m_Sequence;
            PendingQueue m_Pending;
            std::map<uint64_t, std::pair<Digest, Reason>> m_Receipts; // keyed by incoming 64-bit sequence
            std::map<uint16_t, std::pair<Digest, Reason>> m_ProfileReceipts;
            uint16_t m_LastProfile = 0;
            std::weak_ptr<Peer> m_Peer;
            TimePoint m_Expiry;
            TimePoint m_Activity;
            TimePoint m_PingDeadline;
            bool m_Ping = false;
            bool m_Authenticated = false;
            bool m_Resumable = true;
            uint64_t m_OriginSequence = 0;
            std::deque<ClusterDelivery> m_Replay;
            size_t m_ReplayBytes = 0;
            std::map<Digest, uint64_t> m_FabricReceipts;
            TimePoint m_NextInbox{};
            uint32_t m_InboxInterval = 150;
            TimePoint m_NextCheckpoint{};
            CorrelationIds m_RequestIds;
            CorrelationIds m_ResponseIds;
            std::map<uint16_t, Request> m_Requests;
            std::map<uint16_t, ResponseRoute> m_ResponseRoutes;
            std::map<uint16_t, PublishRequest> m_RpcAdmissions;
            std::string m_Realm;
            TrafficMetrics m_Traffic;
            uint64_t m_ObservedAt = UnixMilliseconds();
            uint64_t m_Publications = 0;
            uint64_t m_Deliveries = 0;
        };

        struct Retained
        {
            PublishRequest m_Message;
            TimePoint m_Expiry;
            size_t m_Bytes = 0;
        };

        struct Dispatch
        {
            Session *m_Session;
            PublishRequest m_Message;
            WireFrame m_Bytes;
            TimePoint m_Expiry;
            std::string m_Topic;
            std::optional<ResponseRoute> m_Response;
        };

        Service &m_Owner;
        Configs m_Options;
        IoContext m_Io;
        IoTimer m_Timer;
        eCore::EntityRegistry m_Registry;
        std::map<std::string, eCore::Entity> m_Sessions;
        // Local subscription index: candidates are re-checked against the session profile.
        TopicTree<eCore::Entity> m_Routes;
        std::map<eCore::Entity, std::vector<std::string>> m_Indexed;
        std::map<uint64_t, std::shared_ptr<Peer>> m_Peers;
        std::map<std::string, std::deque<Retained>> m_Retained;
        std::map<std::pair<std::string, std::string>, size_t> m_SharedCursor;
        std::map<std::string, TimePoint> m_AuthTokens;
        BrokerMetrics m_Metrics;
        TimePoint m_StartedAt{};
        std::string m_Instance = ToHex(RandomBytes(16));
        uint64_t m_NextPeer = 0;
        uint16_t m_Port = 0;
        bool m_ClusterStarted = false;
        bool m_Started = false;
        bool m_Stopping = false;
        std::unique_ptr<FileStore> m_Storage;
        std::unique_ptr<FabricNode> m_Cluster;
        std::set<std::string> m_Dirty;
        std::set<std::string> m_DirtyProfiles;
        std::set<std::string> m_Removed;
        std::set<std::string> m_DirtyRetained;
        std::vector<std::pair<std::string, TimePoint>> m_NewTokens;
        uint64_t m_Generation = 0;
        size_t m_BaseBytes = 0;
        std::vector<std::function<void()>> m_Effects;
        size_t m_EffectBytes = 0;
        bool m_Transaction = false;
        bool m_StateDirty = false;
        std::atomic<size_t> m_NotificationBytes = 0;
        std::mutex m_TransientLock;
        std::vector<std::pair<size_t, ClusterDelivery>> m_Transients;

        Impl(Service &service, Configs settings) : m_Owner(service), m_Options(std::move(settings)), m_Timer(m_Io)
        {
            DecodeInterval(m_Options.m_SessionExpiry);
            DecodeInterval(m_Options.m_Keepalive);
            if (!m_Options.m_MaxSessions || !m_Options.m_MaxProfiles || !m_Options.m_MaxInflight ||
                m_Options.m_MaxInflight > 256 || !m_Options.m_MaxSessionBytes || !m_Options.m_MaxRetainedBytes ||
                !m_Options.m_MaxRetainedTopics || !m_Options.m_MaxRequests || !m_Options.m_MaxResponders ||
                !m_Options.m_Workers || !m_Options.m_RetryInterval || !m_Options.m_RequestTimeout ||
                !m_Options.m_MaxRetries || m_Options.m_Transport.m_MaxPacket < 64 ||
                m_Options.m_Transport.m_MaxQueue < m_Options.m_Transport.m_MaxPacket)
            {
                throw std::invalid_argument("Invalid broker resource bounds");
            }
            if (m_Options.m_RequireAuth && m_Options.m_AuthKey.size() < 32 && m_Options.m_Tenants.empty())
            {
                throw std::invalid_argument("Required authentication needs a separate key of at least 32 bytes");
            }
            if (!m_Options.m_AuthKey.empty() && m_Options.m_AuthKey.size() < 32)
            {
                throw std::invalid_argument("Authentication key must contain at least 32 bytes");
            }
            if (!m_Options.m_TenantRoot.empty())
            {
                m_Options.m_TenantRoot = ParseTopic(m_Options.m_TenantRoot).m_Filter;
            }
            if (!m_Options.m_Tenants.empty())
            {
                if (!m_Options.m_AuthKey.empty() || !m_Options.m_TenantRoot.empty())
                {
                    throw std::invalid_argument("Use tenant credentials or the default authentication realm, not both");
                }
                m_Options.m_RequireAuth = true;
                std::set<std::array<uint8_t, 8>> identities;
                std::vector<std::string> roots;
                for (auto &tenant : m_Options.m_Tenants)
                {
                    tenant.m_Root = ParseTopic(tenant.m_Root).m_Filter;
                    if (tenant.m_AuthKey.size() < 32 ||
                        !identities.insert(AuthenticationIdentity(tenant.m_Identity)).second)
                    {
                        throw std::invalid_argument("Invalid tenant credentials or duplicate authentication identity");
                    }
                    for (const auto &root : roots)
                    {
                        if (root == tenant.m_Root || root.starts_with(tenant.m_Root + '/') ||
                            tenant.m_Root.starts_with(root + '/'))
                        {
                            throw std::invalid_argument("Tenant namespace roots must be disjoint");
                        }
                    }
                    roots.push_back(tenant.m_Root);
                    for (auto shard : tenant.m_Shards)
                    {
                        if (!m_Options.m_Cluster || shard >= m_Options.m_Cluster->m_Fabric.m_Shards)
                        {
                            throw std::invalid_argument("Tenant shard is outside the configured fabric");
                        }
                    }
                }
            }
            if (!m_Options.m_Transport.m_Tls && !m_Options.m_Transport.m_Quic && !m_Options.m_AllowRemotePlaintext &&
                !IsLoopback(m_Options.m_Address))
            {
                throw std::invalid_argument("Remote plaintext listeners require explicit opt-in");
            }
            if (!m_Options.m_MaxStateBytes || !m_Options.m_MaxReplayBytes)
            {
                throw std::invalid_argument("Invalid state/replay bounds");
            }
            if (m_Options.m_Cluster)
            {
                m_Cluster = std::make_unique<FabricNode>(*m_Options.m_Cluster);
                m_Cluster->m_OnDelivery = [this](ClusterDelivery delivery)
                {
                    // The replicated inbox owns the payload; notifications only wake its consumer.
                    if (m_NotificationBytes.fetch_add(64) >= m_Options.m_MaxSessionBytes)
                    {
                        m_NotificationBytes.fetch_sub(64);
                        return;
                    }
                    m_Io.Post(
                        [this, session = std::move(delivery.m_Session)]
                        {
                            m_NotificationBytes.fetch_sub(64);
                            if (auto value = this->Find(session))
                            {
                                value->m_NextInbox = TimePoint{};
                            }
                        });
                };
                m_Cluster->m_OnTransient = [this](ClusterDelivery delivery)
                {
                    auto bytes = delivery.m_Publication.m_Payload.Size() + 512;
                    auto previous = m_NotificationBytes.fetch_add(bytes);
                    if (previous >= m_Options.m_MaxSessionBytes || bytes > m_Options.m_MaxSessionBytes - previous)
                    {
                        m_NotificationBytes.fetch_sub(bytes);
                        return;
                    }
                    std::scoped_lock lock(m_TransientLock);
                    m_Transients.emplace_back(bytes, std::move(delivery));
                    // One drain task commits every queued transient delivery as a group.
                    if (m_Transients.size() == 1)
                    {
                        m_Io.Post([this] { this->DrainTransients(); });
                    }
                };
            }
            else if (!m_Options.m_StateDirectory.empty())
            {
                m_Storage = std::make_unique<FileStore>(m_Options.m_StateDirectory, m_Options.m_MaxStateBytes);
                if (auto saved = m_Storage->Find("broker"))
                {
                    m_BaseBytes = saved->size();
                    this->Restore(*saved);
                }
                for (const auto &entry : m_Storage->Journal("broker"))
                {
                    this->ApplyJournal(entry);
                }
            }
        }

        ~Impl()
        {
            SecureErase(m_Options.m_AuthKey);
            for (auto &tenant : m_Options.m_Tenants)
            {
                SecureErase(tenant.m_AuthKey);
            }
        }

        Buffer SaveSession(const Session &session)
        {
            Writer writer(m_Options.m_MaxStateBytes);
            writer.U32(0x45535336);
            writer.String(session.m_Identifier);
            writer.String(session.m_Realm);
            ConnectionRequest request;
            request.m_Properties = session.m_Properties;
            WriteBuffer(writer, Codec::Encode(request, {true, UINT32_MAX}));
            writer.U8(!session.m_Profiles.m_Entries.empty());
            if (!session.m_Profiles.m_Entries.empty())
            {
                WriteBuffer(writer, Codec::Encode(ProfileRequest{1, session.m_Profiles.m_Entries},
                                                  {true, UINT32_MAX, UINT32_MAX}));
            }
            session.m_Profiles.m_Publishers.Save(writer);
            session.m_Profiles.m_Subscribers.Save(writer);
            session.m_Received.Save(writer);
            writer.U64(session.m_Sequence.Issued());
            writer.U16(session.m_LastProfile);
            writer.U8(session.m_Authenticated);
            writer.U8(session.m_Resumable);
            auto expiry =
                session.m_Peer.expired()
                    ? session.m_Expiry
                    : Clock::now() + std::chrono::seconds(DecodeInterval(*session.m_Properties.m_Policy->m_Expiry)) +
                          std::chrono::milliseconds(m_Options.m_Cluster ? m_Options.m_Cluster->m_LeaseMilliseconds : 0);
            writer.U64(SaveExpiry(expiry));
            writer.U64(session.m_OriginSequence);
            auto receipts = [&](const auto &values)
            {
                writer.U32(static_cast<uint32_t>(values.size()));
                for (const auto &[id, value] : values)
                {
                    // Publication receipts store their 64-bit sequence; profile receipts their PID.
                    if constexpr (sizeof(id) == sizeof(uint64_t))
                    {
                        writer.U64(id);
                    }
                    else
                    {
                        writer.U16(id);
                    }
                    writer.Bytes(value.first);
                    WriteBuffer(writer, Codec::Encode(PublishAck{WirePid(id), value.second}, {true, UINT32_MAX}));
                }
            };
            receipts(session.m_Receipts);
            receipts(session.m_ProfileReceipts);
            writer.U32(static_cast<uint32_t>(session.m_Pending.m_Entries.size()));
            for (const auto &[id, pending] : session.m_Pending.m_Entries)
            {
                writer.U64(SaveExpiry(pending.m_Expiry));
                WriteBuffer(writer, Codec::Encode(pending.m_Message, {true, UINT32_MAX}));
            }
            writer.U32(static_cast<uint32_t>(session.m_Replay.size()));
            for (const auto &record : session.m_Replay)
            {
                writer.Bytes(record.m_Identity);
                writer.U64(record.m_Expiry);
                WriteBuffer(writer, Codec::Encode(record.m_Publication, {true, UINT32_MAX}));
            }
            writer.U32(static_cast<uint32_t>(session.m_FabricReceipts.size()));
            for (const auto &[identity, expiry] : session.m_FabricReceipts)
            {
                writer.Bytes(identity);
                writer.U64(expiry);
            }
            session.m_RequestIds.Save(writer);
            session.m_ResponseIds.Save(writer);
            writer.U32(static_cast<uint32_t>(session.m_Requests.size()));
            for (const auto &[id, request] : session.m_Requests)
            {
                WriteBuffer(writer, this->SaveRequest(request));
            }
            writer.U32(static_cast<uint32_t>(session.m_ResponseRoutes.size()));
            for (const auto &[id, route] : session.m_ResponseRoutes)
            {
                writer.U16(id);
                writer.String(route.m_Requester);
                writer.U16(route.m_CorrelationId);
                writer.String(route.m_Topic);
                writer.U64(SaveExpiry(route.m_Expiry));
            }
            writer.U32(static_cast<uint32_t>(session.m_RpcAdmissions.size()));
            for (const auto &[id, message] : session.m_RpcAdmissions)
            {
                WriteBuffer(writer, Codec::Encode(message, {true, UINT32_MAX}));
            }
            return writer.Take();
        }

        Session LoadSession(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            auto version = reader.U32();
            if (version != 0x45535336)
            {
                throw ProtocolError("Unknown session snapshot version; reusable correlation IDs require ESS6");
            }
            constexpr bool legacy = false;
            Session session;
            session.m_Identifier = reader.String();
            session.m_Realm = reader.String();
            session.m_Properties =
                std::get<ConnectionRequest>(Codec::Decode(ReadBuffer(reader), {true, UINT32_MAX})).m_Properties;
            if (!session.m_Properties.m_Policy || !session.m_Properties.m_Policy->m_Expiry ||
                !session.m_Properties.m_Policy->m_Keepalive || !session.m_Properties.m_Policy->m_MaxPacket ||
                !session.m_Properties.m_Auth || session.m_Properties.m_Auth->m_Identifier != session.m_Identifier)
            {
                throw ProtocolError("Incomplete session snapshot");
            }
            if (reader.U8())
            {
                session.m_Profiles.m_Entries =
                    std::get<ProfileRequest>(
                        Codec::Decode(ReadBuffer(reader),
                                      {true, UINT32_MAX, static_cast<uint32_t>(m_Options.m_MaxProfiles)}))
                        .m_Entries;
            }
            session.m_Profiles.m_Publishers = AliasMap::Load(reader);
            session.m_Profiles.m_Subscribers = AliasMap::Load(reader);
            session.m_Received = ReceiveWindow::Load(reader, legacy);
            session.m_Sequence = PublicationSequence(legacy ? reader.U16() : reader.U64());
            session.m_LastProfile = reader.U16();
            session.m_Authenticated = reader.U8() != 0;
            session.m_Resumable = reader.U8() != 0;
            session.m_Expiry = LoadExpiry(reader.U64());
            session.m_OriginSequence = reader.U64();
            auto receipts = [&](auto &values)
            {
                auto count = reader.U32();
                if (count > MMQP_PID_WINDOW)
                {
                    throw ProtocolError("Stored receipt budget exceeded");
                }
                for (uint32_t i = 0; i < count; ++i)
                {
                    using Key = typename std::decay_t<decltype(values)>::key_type;
                    Key id = sizeof(Key) == sizeof(uint64_t) && !legacy ? static_cast<Key>(reader.U64())
                                                                        : static_cast<Key>(reader.U16());
                    Digest digest{};
                    auto bytes = reader.Bytes(digest.size());
                    std::copy(bytes.begin(), bytes.end(), digest.begin());
                    auto ack = std::get<PublishAck>(Codec::Decode(ReadBuffer(reader), {true, UINT32_MAX}));
                    if (ack.m_PacketId != WirePid(id) ||
                        !values.emplace(id, std::make_pair(digest, ack.m_Reason)).second)
                    {
                        throw ProtocolError("Invalid stored receipt");
                    }
                }
            };
            receipts(session.m_Receipts);
            receipts(session.m_ProfileReceipts);
            session.m_Pending.m_MaxBytes = m_Options.m_MaxSessionBytes;
            session.m_Pending.m_MaxCount = m_Options.m_MaxInflight;
            auto count = reader.U32();
            if (count > m_Options.m_MaxInflight)
            {
                throw ProtocolError("Stored pending budget exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto expiry = LoadExpiry(reader.U64());
                auto message = std::get<PublishRequest>(Codec::Decode(ReadBuffer(reader), {true, UINT32_MAX}));
                if (!session.m_Sequence.Sent(message.m_PacketId))
                {
                    throw ProtocolError("Stored outgoing PID exceeds sequence");
                }
                if (expiry <= Clock::now() && session.m_Properties.m_Quic)
                {
                    session.m_Resumable = false;
                }
                if (expiry > Clock::now())
                {
                    auto size =
                        Codec::EncodeFrame(message, {session.m_Properties.m_Policy->m_Reasons, UINT32_MAX}).Size();
                    auto sequence = session.m_Sequence.Sequence(message.m_PacketId);
                    session.m_Pending.Add(std::move(message), sequence, size, Clock::now(), expiry);
                }
            }
            count = reader.U32();
            if (count > m_Options.m_MaxReplayBytes / 32)
            {
                throw ProtocolError("Stored replay count exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                ClusterDelivery record;
                auto identity = reader.Bytes(32);
                std::copy(identity.begin(), identity.end(), record.m_Identity.begin());
                record.m_Expiry = reader.U64();
                auto bytes = ReadBuffer(reader);
                record.m_Publication = std::get<PublishRequest>(Codec::Decode(bytes, {true, UINT32_MAX}));
                if (record.m_Expiry > UnixMilliseconds())
                {
                    session.m_ReplayBytes += bytes.size();
                    if (session.m_ReplayBytes > m_Options.m_MaxReplayBytes)
                    {
                        throw ProtocolError("Stored replay byte budget exceeded");
                    }
                    session.m_Replay.emplace_back(std::move(record));
                }
            }
            count = reader.U32();
            if (count > 8192)
            {
                throw ProtocolError("Stored fabric receipt budget exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                Digest identity{};
                auto bytes = reader.Bytes(32);
                std::copy(bytes.begin(), bytes.end(), identity.begin());
                auto expiry = reader.U64();
                if (expiry > UnixMilliseconds())
                {
                    session.m_FabricReceipts.emplace(identity, expiry);
                }
            }
            session.m_RequestIds = CorrelationIds::Load(reader);
            session.m_ResponseIds = CorrelationIds::Load(reader);
            count = reader.U32();
            if (count > m_Options.m_MaxRequests)
            {
                throw ProtocolError("Stored request budget exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto request = this->ReadRequest(ReadBuffer(reader));
                if (request.m_Requester != session.m_Identifier ||
                    !session.m_RequestIds.Contains(request.m_CorrelationId) ||
                    !session.m_Requests.emplace(request.m_CorrelationId, std::move(request)).second)
                {
                    throw ProtocolError("Invalid stored request identity");
                }
            }
            count = reader.U32();
            if (count > m_Options.m_MaxRequests)
            {
                throw ProtocolError("Stored response-route budget exceeded");
            }
            size_t routeBytes = 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto id = reader.U16();
                ResponseRoute route;
                route.m_Requester = reader.String();
                route.m_CorrelationId = reader.U16();
                route.m_Topic = ParseTopic(reader.String()).m_Filter;
                route.m_Expiry = LoadExpiry(reader.U64());
                routeBytes += this->RouteSize(route);
                if (!id || !session.m_ResponseIds.Contains(id) || !route.m_CorrelationId || route.m_Requester.empty() ||
                    routeBytes > m_Options.m_MaxSessionBytes ||
                    !session.m_ResponseRoutes.emplace(id, std::move(route)).second)
                {
                    throw ProtocolError("Invalid stored response route");
                }
            }
            count = reader.U32();
            if (count > m_Options.m_MaxInflight)
            {
                throw ProtocolError("Stored RPC-admission count exceeded");
            }
            size_t admissionBytes = 0;
            for (uint32_t i = 0; i < count; ++i)
            {
                auto bytes = ReadBuffer(reader);
                admissionBytes += bytes.size();
                auto message = std::get<PublishRequest>(Codec::Decode(bytes, {true, UINT32_MAX}));
                const ResponseRoute *route = nullptr;
                if (message.m_Type == MMQP_PUBREQ_TYPE_REQ)
                {
                    auto request = session.m_Requests.find(message.m_CorrelationId);
                    if (request != session.m_Requests.end())
                    {
                        route = &request->second;
                    }
                }
                else if (message.m_Type == MMQP_PUBREQ_TYPE_RES)
                {
                    auto response = session.m_ResponseRoutes.find(message.m_CorrelationId);
                    if (response != session.m_ResponseRoutes.end())
                    {
                        route = &response->second;
                    }
                }
                if (admissionBytes > m_Options.m_MaxSessionBytes || !message.m_Qos || !route ||
                    session.m_Received.Inspect(message.m_PacketId) != Reception::New ||
                    route->m_Topic != message.m_Topic ||
                    !session.m_RpcAdmissions.emplace(message.m_PacketId, std::move(message)).second)
                {
                    throw ProtocolError("Invalid stored RPC admission");
                }
            }
            reader.Finish();
            return session;
        }

        // Standalone persistence: an EBR5 base snapshot plus an EBJ1 journal of the
        // sessions, retained topics and tokens changed by each commit. The base
        // carries a generation; journal entries from an older base are ignored.
        void SaveRetained(Writer &writer, const std::string &topic) const
        {
            writer.String(topic);
            auto found = m_Retained.find(topic);
            writer.U32(found == m_Retained.end() ? 0 : static_cast<uint32_t>(found->second.size()));
            if (found == m_Retained.end())
            {
                return;
            }
            for (const auto &record : found->second)
            {
                writer.U64(SaveExpiry(record.m_Expiry));
                WriteBuffer(writer, Codec::Encode(record.m_Message, {true, UINT32_MAX}));
            }
        }

        void LoadRetained(Reader &reader)
        {
            auto topic = reader.String();
            auto size = reader.U32();
            if (size > 255)
            {
                throw ProtocolError("Stored retention queue exceeds profile limit");
            }
            if (auto previous = m_Retained.find(topic); previous != m_Retained.end())
            {
                for (const auto &record : previous->second)
                {
                    m_Metrics.m_RetainedBytes -= record.m_Bytes;
                }
                m_Retained.erase(previous);
            }
            for (uint32_t j = 0; j < size; ++j)
            {
                auto expiry = LoadExpiry(reader.U64());
                auto record = ReadBuffer(reader);
                auto publication = std::get<PublishRequest>(Codec::Decode(record, {true, UINT32_MAX}));
                if (publication.m_Topic != topic)
                {
                    throw ProtocolError("Stored retained topic mismatch");
                }
                if (expiry <= Clock::now())
                {
                    continue;
                }
                auto bytes = Codec::EncodeFrame(publication, {false, UINT32_MAX}).Size();
                m_Metrics.m_RetainedBytes += bytes;
                if (m_Metrics.m_RetainedBytes > m_Options.m_MaxRetainedBytes)
                {
                    throw ProtocolError("Stored retention exceeds budget");
                }
                m_Retained[topic].push_back({std::move(publication), expiry, bytes});
            }
            if (m_Retained.size() > m_Options.m_MaxRetainedTopics)
            {
                throw ProtocolError("Stored topics exceed configured budget");
            }
        }

        void LoadSessionRecord(std::span<const uint8_t> bytes)
        {
            auto session = this->LoadSession(bytes);
            auto found = m_Sessions.find(session.m_Identifier);
            if (session.m_Expiry <= Clock::now())
            {
                if (found != m_Sessions.end())
                {
                    this->Destroy(found);
                }
                return;
            }
            if (found != m_Sessions.end())
            {
                auto &existing = m_Registry.Get<Session>(found->second);
                existing = std::move(session);
                this->Index(existing);
                return;
            }
            if (m_Sessions.size() >= m_Options.m_MaxSessions)
            {
                throw ProtocolError("Stored sessions exceed configured budget");
            }
            auto entity = m_Registry.Create();
            m_Sessions.emplace(session.m_Identifier, entity);
            this->Index(m_Registry.Attach<Session>(entity, std::move(session)));
        }

        Buffer Snapshot()
        {
            Writer writer(m_Options.m_MaxStateBytes);
            writer.U32(0x45425236);
            writer.U64(m_Generation);
            writer.U32(static_cast<uint32_t>(m_Sessions.size()));
            for (const auto &[id, entity] : m_Sessions)
            {
                WriteBuffer(writer, this->SaveSession(m_Registry.Get<Session>(entity)));
            }
            writer.U32(static_cast<uint32_t>(m_Retained.size()));
            for (const auto &[topic, records] : m_Retained)
            {
                this->SaveRetained(writer, topic);
            }
            writer.U32(static_cast<uint32_t>(m_AuthTokens.size()));
            for (const auto &[token, expiry] : m_AuthTokens)
            {
                writer.String(token);
                writer.U64(SaveExpiry(expiry));
            }
            return writer.Take();
        }

        void Restore(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            auto version = reader.U32();
            // State from incompatible schema versions requires an explicit offline migration.
            if (version != 0x45425236)
            {
                throw ProtocolError("Unknown broker snapshot version; reusable correlation IDs require EBR6");
            }
            m_Generation = reader.U64();
            auto count = reader.U32();
            if (count > m_Options.m_MaxSessions)
            {
                throw ProtocolError("Stored sessions exceed configured budget");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto record = ReadBuffer(reader);
                Reader identity(record);
                identity.U32();
                if (m_Sessions.contains(identity.String()))
                {
                    throw ProtocolError("Duplicate stored session");
                }
                this->LoadSessionRecord(record);
            }
            count = reader.U32();
            if (count > m_Options.m_MaxRetainedTopics)
            {
                throw ProtocolError("Stored topics exceed configured budget");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                this->LoadRetained(reader);
            }
            if (this->RequestCount() > m_Options.m_MaxRequests)
            {
                throw ProtocolError("Stored request budget exceeded");
            }
            count = reader.U32();
            if (count > 4096)
            {
                throw ProtocolError("Stored authentication budget exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto token = reader.String();
                auto expiry = LoadExpiry(reader.U64());
                if (expiry > Clock::now())
                {
                    m_AuthTokens.emplace(std::move(token), expiry);
                }
            }
            reader.Finish();
        }

        void ApplyJournal(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            if (reader.U32() != 0x45424A31)
            {
                throw ProtocolError("Unknown broker journal version");
            }
            if (reader.U64() != m_Generation)
            {
                return;
            }
            for (auto count = reader.U32(); count; --count)
            {
                this->LoadSessionRecord(ReadBuffer(reader));
            }
            for (auto count = reader.U32(); count; --count)
            {
                if (auto found = m_Sessions.find(reader.String()); found != m_Sessions.end())
                {
                    this->Destroy(found);
                }
            }
            for (auto count = reader.U32(); count; --count)
            {
                this->LoadRetained(reader);
            }
            for (auto count = reader.U32(); count; --count)
            {
                auto token = reader.String();
                auto expiry = LoadExpiry(reader.U64());
                if (expiry > Clock::now() && m_AuthTokens.size() < 4096)
                {
                    m_AuthTokens.emplace(std::move(token), expiry);
                }
            }
            reader.Finish();
            if (this->RequestCount() > m_Options.m_MaxRequests)
            {
                throw ProtocolError("Stored request budget exceeded");
            }
        }

        void Persist()
        {
            if (m_Storage->JournalBytes("broker") >= std::max<size_t>(1024 * 1024, m_BaseBytes))
            {
                // Compaction: the new base supersedes every journal entry before it.
                ++m_Generation;
                auto base = this->Snapshot();
                m_BaseBytes = base.size();
                m_Storage->Put("broker", std::move(base));
                m_Storage->Reset("broker");
                return;
            }
            Writer writer(m_Options.m_MaxStateBytes);
            writer.U32(0x45424A31);
            writer.U64(m_Generation);
            std::vector<const Session *> changed;
            for (const auto &id : m_Dirty)
            {
                if (auto session = this->Find(id))
                {
                    changed.push_back(session);
                }
            }
            writer.U32(static_cast<uint32_t>(changed.size()));
            for (auto session : changed)
            {
                WriteBuffer(writer, this->SaveSession(*session));
            }
            writer.U32(static_cast<uint32_t>(m_Removed.size()));
            for (const auto &id : m_Removed)
            {
                writer.String(id);
            }
            writer.U32(static_cast<uint32_t>(m_DirtyRetained.size()));
            for (const auto &topic : m_DirtyRetained)
            {
                this->SaveRetained(writer, topic);
            }
            writer.U32(static_cast<uint32_t>(m_NewTokens.size()));
            for (const auto &[token, expiry] : m_NewTokens)
            {
                writer.String(token);
                writer.U64(SaveExpiry(expiry));
            }
            if (changed.empty() && m_Removed.empty() && m_DirtyRetained.empty() && m_NewTokens.empty())
            {
                return;
            }
            m_Storage->Append("broker", writer.Data());
        }

        void Unindex(eCore::Entity entity)
        {
            auto found = m_Indexed.find(entity);
            if (found == m_Indexed.end())
            {
                return;
            }
            for (const auto &filter : found->second)
            {
                m_Routes.Remove(filter, entity);
            }
            m_Indexed.erase(found);
        }

        void Index(const Session &session)
        {
            auto found = m_Sessions.find(session.m_Identifier);
            if (found == m_Sessions.end())
            {
                return;
            }
            this->Unindex(found->second);
            std::vector<std::string> filters;
            for (const auto &entry : session.m_Profiles.m_Entries)
            {
                if (entry.m_Type != MMQP_PROREQ_TYPE_SUB)
                {
                    continue;
                }
                auto filter = std::string(SplitShared(entry.m_Topic).second);
                if (std::find(filters.begin(), filters.end(), filter) != filters.end())
                {
                    continue;
                }
                m_Routes.Insert(filter, found->second);
                filters.push_back(std::move(filter));
            }
            if (!filters.empty())
            {
                m_Indexed.emplace(found->second, std::move(filters));
            }
        }

        void Destroy(std::map<std::string, eCore::Entity>::iterator session)
        {
            this->Unindex(session->second);
            m_Registry.Destroy(session->second);
            m_Sessions.erase(session);
        }

        void Changed(Session &session, bool profile = false)
        {
            if (profile)
            {
                this->Index(session);
            }
            m_Dirty.insert(session.m_Identifier);
            if (profile)
            {
                m_DirtyProfiles.insert(session.m_Identifier);
            }
            m_StateDirty = true;
        }

        void Checkpoint()
        {
            if (!m_StateDirty)
            {
                return;
            }
            if (m_Storage)
            {
                this->Persist();
            }
            if (m_Cluster)
            {
                for (const auto &id : m_Dirty)
                {
                    if (auto session = this->Find(id))
                    {
                        if (m_Cluster->Owns(id))
                        {
                            auto expiry =
                                session->m_Peer.expired()
                                    ? SaveExpiry(session->m_Expiry)
                                    : UnixMilliseconds() +
                                          DecodeInterval(*session->m_Properties.m_Policy->m_Expiry) * 1000ULL +
                                          m_Options.m_Cluster->m_LeaseMilliseconds;
                            m_Cluster->Commit(id, this->SaveSession(*session), expiry);
                        }
                        else if (!session->m_Peer.expired())
                        {
                            throw ProtocolError("Session ownership was lost", MMQP_REASON_BUSY);
                        }
                    }
                }
                for (const auto &id : m_DirtyProfiles)
                {
                    if (auto session = this->Find(id); session && m_Cluster->Owns(id))
                    {
                        auto expiry = session->m_Peer.expired()
                                          ? SaveExpiry(session->m_Expiry)
                                          : UnixMilliseconds() +
                                                DecodeInterval(*session->m_Properties.m_Policy->m_Expiry) * 1000ULL;
                        m_Cluster->Profiles(id, session->m_Profiles.m_Entries, expiry, !session->m_Peer.expired());
                    }
                }
            }
            m_Dirty.clear();
            m_DirtyProfiles.clear();
            m_Removed.clear();
            m_DirtyRetained.clear();
            m_NewTokens.clear();
            m_StateDirty = false;
        }

        bool Transmit(const std::shared_ptr<Peer> &peer, WireFrame bytes)
        {
            auto size = bytes.Size();
            if (!peer->m_Connection->Send(std::move(bytes)))
            {
                return false;
            }
            ++m_Metrics.m_SentPackets;
            m_Metrics.m_SentBytes += size;
            if (auto session = this->Find(peer->m_Session))
            {
                ++session->m_Traffic.m_SentPackets;
                session->m_Traffic.m_SentBytes += size;
            }
            return true;
        }

        bool Queue(const std::shared_ptr<Peer> &peer, WireFrame bytes)
        {
            if (!m_Transaction)
            {
                return this->Transmit(peer, std::move(bytes));
            }
            if (m_EffectBytes > m_Options.m_MaxStateBytes || bytes.Size() > m_Options.m_MaxStateBytes - m_EffectBytes)
            {
                throw ProtocolError("Pending effect budget exhausted", MMQP_REASON_BUSY);
            }
            m_EffectBytes += bytes.Size();
            m_Effects.push_back(
                [this, peer, bytes = std::move(bytes)]() mutable
                {
                    if (m_Cluster && !peer->m_Session.empty() && !m_Cluster->Owns(peer->m_Session))
                    {
                        peer->m_Connection->Close(false);
                        return;
                    }
                    if (!this->Transmit(peer, std::move(bytes)))
                    {
                        peer->m_Connection->Close(false);
                    }
                });
            return true;
        }

        void Finish()
        {
            this->Checkpoint();
            m_Transaction = false;
            m_EffectBytes = 0;
            auto ready = std::move(m_Effects);
            m_Effects.clear();
            for (auto &effect : ready)
            {
                effect();
            }
        }

        void Close(const std::shared_ptr<Peer> &peer, bool drain)
        {
            if (m_Transaction)
            {
                m_Effects.push_back([peer, drain] { peer->m_Connection->Close(drain); });
            }
            else
            {
                peer->m_Connection->Close(drain);
            }
        }

        void DrainTransients()
        {
            std::vector<std::pair<size_t, ClusterDelivery>> batch;
            {
                std::scoped_lock lock(m_TransientLock);
                batch.swap(m_Transients);
            }
            m_Transaction = true;
            for (auto &[bytes, delivery] : batch)
            {
                m_NotificationBytes.fetch_sub(bytes);
                auto session = this->Find(delivery.m_Session);
                if (!session || session->m_Peer.expired() || !m_Cluster->Owns(delivery.m_Session) ||
                    delivery.m_Expiry <= UnixMilliseconds() || m_Stopping)
                {
                    continue;
                }
                try
                {
                    this->AdmitTopic(delivery.m_Publication.m_Topic, false, session);
                    bool response = delivery.m_Publication.m_Type == MMQP_PUBREQ_TYPE_RES;
                    if ((response && this->AcceptResponse(*session, delivery.m_Publication, delivery.m_Sender)) ||
                        (!response &&
                         session->m_Profiles.Subscribed(delivery.m_Publication.m_Topic, 0,
                                                        delivery.m_Publication.m_Type == MMQP_PUBREQ_TYPE_REQ)))
                    {
                        this->Deliver(this->Prepare(*session, delivery.m_Publication, LoadExpiry(delivery.m_Expiry),
                                                    delivery.m_Sender));
                        if (response)
                        {
                            this->CompleteResponse(*session, delivery.m_Publication.m_CorrelationId, delivery.m_Sender);
                        }
                    }
                }
                catch (const ProtocolError &)
                {
                } // Transient traffic is best effort; one rejection must not drop the batch.
            }
            try
            {
                this->Finish();
            }
            catch (...)
            {
                m_Transaction = false;
                m_Effects.clear();
                m_EffectBytes = 0;
            }
        }

        Session *Find(const std::string &identifier)
        {
            auto found = m_Sessions.find(identifier);
            return found == m_Sessions.end() ? nullptr : &m_Registry.Get<Session>(found->second);
        }

        const TenantOptions *Realm(const std::string &name) const
        {
            if (name.empty())
            {
                return nullptr;
            }
            for (const auto &tenant : m_Options.m_Tenants)
            {
                if (tenant.m_Identity == name)
                {
                    return &tenant;
                }
            }
            throw ProtocolError("Unknown authentication realm", MMQP_REASON_UNAUTHORIZED);
        }

        std::string TenantRoot(const Session *session) const
        {
            auto realm = this->Realm(session ? session->m_Realm : "");
            return realm ? realm->m_Root : m_Options.m_TenantRoot;
        }

        void AdmitTopic(std::string_view topic, bool filter, const Session *session = nullptr)
        {
            auto pattern = ParseTopic(topic, filter).m_Filter;
            if (pattern.starts_with(MMQP_BIRTH_STATUS_TOPIC_PREFIX))
            {
                pattern.erase(0, std::string_view(MMQP_BIRTH_STATUS_TOPIC_PREFIX).size());
            }
            else if (pattern.starts_with(MMQP_LAST_STATUS_TOPIC_PREFIX))
            {
                pattern.erase(0, std::string_view(MMQP_LAST_STATUS_TOPIC_PREFIX).size());
            }
            auto root = this->TenantRoot(session);
            if (!root.empty() && pattern != root && !pattern.starts_with(root + '/'))
            {
                throw ProtocolError("Topic is outside the configured tenant", MMQP_REASON_UNAUTHORIZED);
            }
            auto realm = this->Realm(session ? session->m_Realm : "");
            if (realm && !realm->m_Shards.empty() && m_Cluster)
            {
                for (auto shard : m_Cluster->Placement().ProfileShards(pattern))
                {
                    if (!realm->m_Shards.contains(shard))
                    {
                        throw ProtocolError("Topic exceeds the tenant shard budget", MMQP_REASON_UNAUTHORIZED);
                    }
                }
            }
        }

        void AdmitProfiles(const ProfileRequest &profile, const Session *session = nullptr)
        {
            for (const auto &entry : profile.m_Entries)
            {
                if (entry.m_Type == MMQP_PROREQ_TYPE_REG && !entry.m_Topic.empty() &&
                    ParseTopic(entry.m_Topic, true).m_Filter.starts_with('$'))
                {
                    throw ProtocolError("Reserved topics cannot be registered by publishers", MMQP_REASON_UNAUTHORIZED);
                }
                if (entry.m_Type == MMQP_PROREQ_TYPE_SUB && entry.m_Alias >= 0x8000)
                {
                    if (!session)
                    {
                        throw ProtocolError("New subscriber aliases must be below 0x8000", MMQP_REASON_INVALID_ALIAS);
                    }
                    const auto &mapped = session->m_Profiles.m_Subscribers.Resolve(entry.m_Alias);
                    if (!entry.m_Topic.empty() && mapped != CanonicalTopic(entry.m_Topic))
                    {
                        throw ProtocolError("Service-assigned alias cannot be rebound", MMQP_REASON_INVALID_ALIAS);
                    }
                }
                if (!entry.m_Topic.empty())
                {
                    this->AdmitTopic(entry.m_Topic, true, session);
                }
            }
        }

        bool Send(const std::shared_ptr<Peer> &peer, const Message &message)
        {
            if (peer->m_Phase == Phase::Closing)
            {
                return false;
            }
            return this->Queue(peer, Codec::EncodeFrame(message, peer->m_Codec));
        }

        void Fail(const std::shared_ptr<Peer> &peer, Reason reason, bool connecting = false)
        {
            ++m_Metrics.m_Rejected;
            if (peer->m_Phase == Phase::Closing)
            {
                return;
            }
            if (m_Cluster && connecting && peer->m_Request.m_Resume && peer->m_Request.m_Properties.m_Auth)
            {
                auto id = peer->m_Request.m_Properties.m_Auth->m_Identifier;
                auto session = this->Find(id);
                if (!session || session->m_Peer.expired())
                {
                    m_Cluster->Release(id);
                }
            }
            try
            {
                if (connecting)
                {
                    this->Send(peer, ConnectionAck{reason, {}});
                }
                else
                {
                    this->Send(peer, ConnectionEnd{reason});
                }
            }
            catch (...)
            {
            }
            peer->m_Phase = Phase::Closing;
            this->Close(peer, true);
        }

        void Accepted(Connection::Pointer connection)
        {
            auto peer = std::make_shared<Peer>();
            peer->m_Id = ++m_NextPeer;
            peer->m_Connection = std::move(connection);
            peer->m_Codec.m_MaxPacket = m_Options.m_Transport.m_MaxPacket;
            peer->m_Codec.m_MaxProfiles = static_cast<uint32_t>(m_Options.m_MaxProfiles);
            peer->m_Deadline = Clock::now() + std::chrono::milliseconds(m_Options.m_Transport.m_Timeout);
            m_Peers.emplace(peer->m_Id, peer);
            peer->m_Connection->m_OnReceive = [this, id = peer->m_Id](auto bytes) { this->Receive(id, bytes); };
            peer->m_Connection->m_OnFlush = [this, id = peer->m_Id] { this->Flush(id); };
            peer->m_Connection->m_OnClose = [this, id = peer->m_Id](const auto &reason) { this->Closed(id, reason); };
        }

        void Establish(const std::shared_ptr<Peer> &peer, bool authenticated)
        {
            auto &request = peer->m_Request;
            auto now = Clock::now();
            Session *session = nullptr;
            bool created = false;
            if (m_Cluster && request.m_Resume)
            {
                auto identifier = request.m_Properties.m_Auth->m_Identifier;
                auto existing = this->Find(identifier);
                if (existing && !existing->m_Peer.expired())
                {
                    throw ProtocolError("Session is already connected", MMQP_REASON_BUSY);
                }
                auto snapshot = m_Cluster->Acquire(identifier);
                if (snapshot.empty())
                {
                    m_Cluster->Release(identifier);
                    throw ProtocolError("Unknown replicated session", MMQP_REASON_SESSION_EXPIRED);
                }
                auto restored = this->LoadSession(snapshot);
                if (restored.m_Identifier != identifier)
                {
                    throw ProtocolError("Snapshot identity differs from requested SID");
                }
                if (existing)
                {
                    *existing = std::move(restored);
                    this->Index(*existing);
                }
                else
                {
                    if (m_Sessions.size() >= m_Options.m_MaxSessions)
                    {
                        throw ProtocolError("Session budget exhausted", MMQP_REASON_BUSY);
                    }
                    auto entity = m_Registry.Create();
                    m_Sessions.emplace(identifier, entity);
                    this->Index(m_Registry.Attach<Session>(entity, std::move(restored)));
                }
            }
            if (request.m_Resume)
            {
                session = this->Find(request.m_Properties.m_Auth->m_Identifier);
                if (!session || !session->m_Resumable || session->m_Expiry <= now)
                {
                    throw ProtocolError("Stored session has expired", MMQP_REASON_SESSION_EXPIRED);
                }
                if (!session->m_Peer.expired())
                {
                    throw ProtocolError("Session is already connected", MMQP_REASON_BUSY);
                }
                if (session->m_Authenticated != authenticated)
                {
                    throw ProtocolError("Session authentication cannot change on resume", MMQP_REASON_AUTH_FAILED);
                }
                if (session->m_Realm != peer->m_Realm)
                {
                    throw ProtocolError("Session cannot change authentication realm", MMQP_REASON_AUTH_FAILED);
                }
                if (request.m_Properties.m_Profile || request.m_Properties.m_Will)
                {
                    throw ProtocolError("Resume preserves stored profiles and wills; update profiles after CONACK",
                                        MMQP_REASON_POLICY_VIOLATION);
                }
            }
            else
            {
                if (m_Sessions.size() >= m_Options.m_MaxSessions)
                {
                    throw ProtocolError("Session budget exhausted", MMQP_REASON_BUSY);
                }
                auto entity = m_Registry.Create();
                session = &m_Registry.Attach<Session>(entity);
                session->m_Realm = peer->m_Realm;
                session->m_Identifier = ToHex(RandomBytes(16));
                while (m_Sessions.contains(session->m_Identifier))
                {
                    session->m_Identifier = ToHex(RandomBytes(16));
                }
                m_Sessions.emplace(session->m_Identifier, entity);
                created = true;
            }
            try
            {
                if (m_Cluster && created)
                {
                    m_Cluster->Acquire(session->m_Identifier);
                }
                if (request.m_Properties.m_Quic && !peer->m_Connection->Quic())
                {
                    throw ProtocolError("QUIC policy is not valid on the TCP listener", MMQP_REASON_POLICY_VIOLATION);
                }
                std::optional<QuicPolicy> quic;
                if (peer->m_Connection->Quic())
                {
                    quic = QuicPlan::Negotiate(request.m_Properties.m_Quic.value_or(QuicPolicy{}),
                                               m_Options.m_Transport.m_QuicPolicy);
                }
                if (request.m_Resume && quic != session->m_Properties.m_Quic)
                {
                    throw ProtocolError("Resume must preserve the negotiated transport and QUIC policy",
                                        MMQP_REASON_POLICY_VIOLATION);
                }
                if (created)
                {
                    session->m_Received = ReceiveWindow(!peer->m_Connection->Quic());
                }
                session->m_Properties.m_Quic = quic;
                auto policy = request.m_Properties.m_Policy.value_or(SessionPolicy{});
                if (!policy.m_Expiry)
                {
                    policy.m_Expiry = m_Options.m_SessionExpiry;
                }
                if (!policy.m_Keepalive)
                {
                    policy.m_Keepalive = m_Options.m_Keepalive;
                }
                policy.m_MaxPacket = std::min(policy.m_MaxPacket.value_or(m_Options.m_Transport.m_MaxPacket),
                                              m_Options.m_Transport.m_MaxPacket);
                if (*policy.m_MaxPacket < 64)
                {
                    throw ProtocolError("Negotiated packet limit must be at least 64", MMQP_REASON_POLICY_VIOLATION);
                }
                peer->m_Codec.m_Reasons = policy.m_Reasons;
                peer->m_Codec.m_MaxPacket = *policy.m_MaxPacket;
                peer->m_Connection->SetLimit(*policy.m_MaxPacket);
                if (request.m_Properties.m_Profile)
                {
                    this->AdmitProfiles(*request.m_Properties.m_Profile, session);
                    session->m_Profiles.Apply(*request.m_Properties.m_Profile, m_Options.m_MaxProfiles);
                    session->m_LastProfile = request.m_Properties.m_Profile->m_PacketId;
                }
                if (request.m_Properties.m_Will)
                {
                    auto topic = request.m_Properties.m_Will->m_Topic;
                    if (!topic.empty() && ParseTopic(topic).m_Filter.starts_with('$'))
                    {
                        throw ProtocolError("Will topic must be an ordinary topic suffix", MMQP_REASON_UNAUTHORIZED);
                    }
                    if (!topic.empty())
                    {
                        this->AdmitTopic(topic, false, session);
                    }
                    session->m_Properties.m_Will = request.m_Properties.m_Will;
                    auto root = this->TenantRoot(session);
                    if (topic.empty() && !root.empty())
                    {
                        session->m_Properties.m_Will->m_Topic = root + '/' + session->m_Identifier;
                    }
                    auto suffix = session->m_Properties.m_Will->m_Topic.empty() ? session->m_Identifier
                                                                                : session->m_Properties.m_Will->m_Topic;
                    ParseTopic(std::string(MMQP_BIRTH_STATUS_TOPIC_PREFIX) + suffix);
                }
                session->m_Properties.m_Policy = policy;
                session->m_Properties.m_Auth = SessionAuth{session->m_Identifier, {}};
                session->m_Pending.m_MaxBytes = m_Options.m_MaxSessionBytes;
                session->m_Pending.m_MaxCount = m_Options.m_MaxInflight;
                session->m_Authenticated = authenticated;
                session->m_Activity = now;
                session->m_Expiry = TimePoint::max();
                session->m_Ping = false;
                session->m_Peer = peer;
                peer->m_Session = session->m_Identifier;
                peer->m_Phase = Phase::Active;
                this->Changed(*session, true);
                if (m_Cluster)
                {
                    for (auto &[id, pending] : session->m_Requests)
                    {
                        if (pending.m_Expiry <= now)
                        {
                            continue;
                        }
                        auto key = this->RequestKey(session->m_Identifier, id);
                        m_Cluster->Acquire(key);
                        pending.m_Broker = m_Cluster->Identifier();
                        // The session snapshot includes responders already consumed.
                        // Do not replace it with a potentially older directory copy.
                        m_Cluster->Commit(key, this->SaveRequest(pending), SaveExpiry(pending.m_Expiry));
                    }
                }
                this->RecoverExchanges(*session);
                ConnectionAck ack;
                ack.m_Properties.m_Auth = session->m_Properties.m_Auth;
                ack.m_Properties.m_Policy = session->m_Properties.m_Policy;
                ack.m_Properties.m_Quic = session->m_Properties.m_Quic;
                if (!this->Send(peer, ack))
                {
                    throw ProtocolError("Transport send budget exhausted", MMQP_REASON_BUSY);
                }
                if (quic)
                {
                    peer->m_Connection->Configure(*quic, policy.m_Reasons);
                }
                ++m_Metrics.m_Accepted;
                m_Owner.TryPost<SessionConnected>(session->m_Identifier.size(), session->m_Identifier,
                                                  request.m_Resume);
                this->Will(*session, true);
                if (created && request.m_Properties.m_Profile)
                {
                    this->Replay(*session, *request.m_Properties.m_Profile);
                }
                for (auto &[id, publication] : session->m_Pending.m_Entries)
                {
                    publication.m_Retry = now;
                    publication.m_Attempts = 0;
                }
            }
            catch (...)
            {
                if (created)
                {
                    auto identifier = session->m_Identifier;
                    this->Destroy(m_Sessions.find(identifier));
                    m_Removed.insert(identifier);
                    if (m_Cluster)
                    {
                        m_Cluster->Release(identifier);
                    }
                }
                throw;
            }
        }

        void Connect(const std::shared_ptr<Peer> &peer, ConnectionRequest request)
        {
            peer->m_Request = std::move(request);
            if (peer->m_Request.m_Properties.m_Policy)
            {
                peer->m_Codec.m_Reasons = peer->m_Request.m_Properties.m_Policy->m_Reasons;
            }
            auto auth = peer->m_Request.m_Properties.m_Auth;
            if (auth && !auth->m_Token.empty())
            {
                const Buffer *key = &m_Options.m_AuthKey;
                if (auth->m_Token.size() == 32)
                {
                    for (const auto &tenant : m_Options.m_Tenants)
                    {
                        if (SecureEqual(std::span(auth->m_Token).first(8), AuthenticationIdentity(tenant.m_Identity)))
                        {
                            key = &tenant.m_AuthKey;
                            peer->m_Realm = tenant.m_Identity;
                            break;
                        }
                    }
                }
                if (key->empty() || auth->m_Token.size() != 32 ||
                    (!peer->m_Connection->Secure() && !m_Options.m_AllowInsecureAuth))
                {
                    throw ProtocolError("Authentication requires configured credentials and secure transport",
                                        MMQP_REASON_AUTH_FAILED);
                }
                auto token = ToHex(auth->m_Token);
                if (m_AuthTokens.contains(token) || m_AuthTokens.size() >= 4096)
                {
                    throw ProtocolError("Repeated token or authentication budget exhausted", MMQP_REASON_AUTH_FAILED);
                }
                auto expiry = Clock::now() + std::chrono::hours(1);
                m_AuthTokens.emplace(token, expiry);
                m_NewTokens.emplace_back(token, expiry);
                m_StateDirty = true;
                if (m_Cluster)
                {
                    auto key = "auth:" + token;
                    auto previous = m_Cluster->Acquire(key);
                    if (!previous.empty())
                    {
                        Reader reader(previous);
                        if (reader.U64() > UnixMilliseconds())
                        {
                            m_Cluster->Release(key);
                            throw ProtocolError("Repeated cluster authentication token", MMQP_REASON_AUTH_FAILED);
                        }
                    }
                    Writer record;
                    record.U64(UnixMilliseconds() + 3600000);
                    m_Cluster->Commit(key, record.Take(), UnixMilliseconds() + 3600000);
                    m_Cluster->Release(key);
                }
                auto response = Hmac256(*key, auth->m_Token);
                auto expected = Hmac256(*key, response);
                peer->m_Expected.assign(expected.begin(), expected.end());
                SecureErase(expected);
                peer->m_Phase = Phase::Challenge;
                if (!this->Send(peer, ConnectionSync{Buffer(response.begin(), response.end())}))
                {
                    throw ProtocolError("Authentication send failed", MMQP_REASON_BUSY);
                }
            }
            else
            {
                if (m_Options.m_RequireAuth)
                {
                    throw ProtocolError("Authentication is required", MMQP_REASON_AUTH_FAILED);
                }
                this->Establish(peer, false);
            }
        }

        std::vector<Session *> Destinations(const PublishRequest &publication, bool responders)
        {
            std::vector<eCore::Entity> candidates;
            m_Routes.Match(publication.m_Topic, [&](eCore::Entity entity) { candidates.push_back(entity); });
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
            auto now = Clock::now();
            std::vector<Session *> result;
            std::map<std::pair<std::string_view, std::string_view>, std::vector<Session *>> shared;
            for (auto entity : candidates)
            {
                auto *session = m_Registry.TryGet<Session>(entity);
                if (!session || (session->m_Peer.expired() && (!publication.m_Qos || responders)) ||
                    session->m_Expiry <= now)
                {
                    continue;
                }
                bool direct = false;
                session->m_Profiles.ForEachSubscription(publication.m_Topic, publication.m_Qos, responders,
                                                        [&](const ProfileEntry &entry)
                                                        {
                                                            auto [group, filter] = SplitShared(entry.m_Topic);
                                                            if (group.empty())
                                                            {
                                                                direct = true;
                                                            }
                                                            else if (!session->m_Peer.expired())
                                                            {
                                                                shared[{group, filter}].push_back(session);
                                                            }
                                                        });
                if (direct)
                {
                    result.push_back(session);
                }
            }
            auto order = [](const Session *left, const Session *right)
            { return left->m_Identifier < right->m_Identifier; };
            for (auto &[group, members] : shared)
            {
                // SID order keeps the round-robin rotation stable between lookups.
                std::sort(members.begin(), members.end(), order);
                auto &cursor = m_SharedCursor[{std::string(group.first), std::string(group.second)}];
                result.push_back(members[cursor++ % members.size()]);
            }
            // Overlapping subscriptions are delivered once per recipient session.
            std::sort(result.begin(), result.end(), order);
            result.erase(std::unique(result.begin(), result.end()), result.end());
            return result;
        }

        void PruneShared()
        {
            std::set<std::pair<std::string, std::string>> groups;
            for (const auto &[id, entity] : m_Sessions)
            {
                for (const auto &profile : m_Registry.Get<Session>(entity).m_Profiles.m_Entries)
                {
                    if (profile.m_Type != MMQP_PROREQ_TYPE_SUB)
                    {
                        continue;
                    }
                    auto [group, filter] = SplitShared(profile.m_Topic);
                    if (!group.empty())
                    {
                        groups.emplace(group, filter);
                    }
                }
            }
            std::erase_if(m_SharedCursor, [&](const auto &cursor) { return !groups.contains(cursor.first); });
        }

        size_t RouteSize(const ResponseRoute &route) const
        {
            return sizeof(ResponseRoute) + 32 + route.m_Requester.size() + route.m_Topic.size();
        }

        Dispatch Prepare(Session &session, const PublishRequest &source, TimePoint expiry, std::string_view sender = {})
        {
            // QoS 0 deliveries are encoded from the source payload; only QoS 1 keeps a copy for retries.
            PublishRequest message;
            message.m_Type = source.m_Type;
            message.m_Qos = source.m_Qos;
            message.m_CorrelationId = source.m_CorrelationId;
            message.m_Responses = source.m_Responses;
            message.m_Reason = source.m_Reason;
            message.m_Encryption = source.m_Encryption;
            std::optional<ResponseRoute> response;
            if (message.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                if (sender.empty() || !message.m_CorrelationId)
                {
                    throw ProtocolError("Request is missing its origin");
                }
                if (session.m_ResponseRoutes.size() >= m_Options.m_MaxRequests)
                {
                    throw ProtocolError("Responder route budget exhausted", MMQP_REASON_BUSY);
                }
                response = ResponseRoute{std::string(sender), message.m_CorrelationId, source.m_Topic, expiry};
                auto size = this->RouteSize(*response);
                for (const auto &[id, route] : session.m_ResponseRoutes)
                {
                    size += this->RouteSize(route);
                }
                if (size > m_Options.m_MaxSessionBytes)
                {
                    throw ProtocolError("Responder route byte budget exhausted", MMQP_REASON_BUSY);
                }
                message.m_CorrelationId = session.m_ResponseIds.Peek();
            }
            message.m_Alias = session.m_Profiles.m_Subscribers.Candidate(source.m_Topic, 0x8000);
            message.m_Topic = session.m_Profiles.m_Subscribers.Confirmed(message.m_Alias) ? "" : source.m_Topic;
            message.m_Duplicate = false;
            message.m_PacketId = message.m_Qos ? session.m_Sequence.Peek() : 0;
            auto peer = session.m_Peer.lock();
            CodecOptions codec;
            codec.m_Reasons = session.m_Properties.m_Policy->m_Reasons;
            codec.m_MaxPacket = *session.m_Properties.m_Policy->m_MaxPacket;
            message.m_Payload = source.m_Payload;
            auto bytes = Codec::EncodeFrame(message, codec);
            if ((message.m_Qos && !session.m_Pending.CanAdd(bytes.Size(), session.m_Sequence.Issued() + 1)) ||
                (peer && bytes.Size() > peer->m_Connection->Capacity() - peer->m_Connection->Queued()))
            {
                throw ProtocolError("Subscriber send budget exhausted", MMQP_REASON_BUSY);
            }
            return {&session, std::move(message), std::move(bytes), expiry, source.m_Topic, std::move(response)};
        }

        void Deliver(Dispatch dispatch)
        {
            auto &session = *dispatch.m_Session;
            if (dispatch.m_Response)
            {
                if (session.m_ResponseIds.Next() != dispatch.m_Message.m_CorrelationId)
                {
                    throw std::logic_error("Response-route admission order changed");
                }
                session.m_ResponseRoutes.emplace(dispatch.m_Message.m_CorrelationId, std::move(*dispatch.m_Response));
            }
            session.m_Profiles.m_Subscribers.Map(dispatch.m_Message.m_Alias, dispatch.m_Topic);
            if (dispatch.m_Message.m_Qos)
            {
                if (session.m_Sequence.Next() != dispatch.m_Message.m_PacketId)
                {
                    throw std::logic_error("Publication admission order changed");
                }
                session.m_Pending.Add(dispatch.m_Message, session.m_Sequence.Issued(), dispatch.m_Bytes.Size(),
                                      Clock::now() + std::chrono::milliseconds(m_Options.m_RetryInterval),
                                      dispatch.m_Expiry);
            }
            this->Changed(session);
            if (auto peer = session.m_Peer.lock())
            {
                if (!this->Queue(peer, std::move(dispatch.m_Bytes)))
                {
                    this->Close(peer, false);
                }
            }
            ++m_Metrics.m_Forwarded;
            ++session.m_Deliveries;
        }

        void PurgeRetained(TimePoint now)
        {
            for (auto it = m_Retained.begin(); it != m_Retained.end();)
            {
                std::erase_if(it->second,
                              [&](const Retained &value)
                              {
                                  if (value.m_Expiry > now)
                                  {
                                      return false;
                                  }
                                  m_Metrics.m_RetainedBytes -= value.m_Bytes;
                                  return true;
                              });
                if (it->second.empty())
                {
                    it = m_Retained.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        void Retain(const PublishRequest &publication, const ProfileEntry &profile)
        {
            // Updates the topic queue in place: no copy of retained payloads and no
            // global expiry sweep unless a budget would otherwise reject the record.
            m_StateDirty = true;
            auto now = Clock::now();
            auto found = m_Retained.find(publication.m_Topic);
            if (found == m_Retained.end() && m_Retained.size() >= m_Options.m_MaxRetainedTopics)
            {
                this->PurgeRetained(now);
                if (m_Retained.size() >= m_Options.m_MaxRetainedTopics)
                {
                    throw ProtocolError("Retained topic budget exhausted", MMQP_REASON_BUSY);
                }
            }
            auto limit = static_cast<size_t>(profile.m_Qos ? profile.m_Queue : 1);
            auto bytes = Codec::EncodeFrame(publication, {false, m_Options.m_Transport.m_MaxPacket}).Size();
            auto evicted = [&]
            {
                if (found == m_Retained.end())
                {
                    return std::pair<size_t, size_t>{0, 0};
                }
                auto &queue = found->second;
                std::erase_if(queue,
                              [&](const Retained &value)
                              {
                                  if (value.m_Expiry > now)
                                  {
                                      return false;
                                  }
                                  m_Metrics.m_RetainedBytes -= value.m_Bytes;
                                  return true;
                              });
                size_t count = queue.size() >= limit ? queue.size() - limit + 1 : 0;
                size_t freed = 0;
                for (size_t i = 0; i < count; ++i)
                {
                    freed += queue[i].m_Bytes;
                }
                return std::pair{count, freed};
            };
            auto [count, freed] = evicted();
            if (bytes > m_Options.m_MaxRetainedBytes - (m_Metrics.m_RetainedBytes - freed))
            {
                this->PurgeRetained(now);
                found = m_Retained.find(publication.m_Topic);
                std::tie(count, freed) = evicted();
                if (bytes > m_Options.m_MaxRetainedBytes - (m_Metrics.m_RetainedBytes - freed))
                {
                    throw ProtocolError("Retained payload budget exhausted", MMQP_REASON_BUSY);
                }
            }
            auto &queue = found == m_Retained.end() ? m_Retained[publication.m_Topic] : found->second;
            queue.erase(queue.begin(), queue.begin() + static_cast<std::ptrdiff_t>(count));
            queue.push_back({publication, now + std::chrono::seconds(DecodeInterval(profile.m_Expiry)), bytes});
            m_Metrics.m_RetainedBytes = m_Metrics.m_RetainedBytes - freed + bytes;
            m_DirtyRetained.insert(publication.m_Topic);
        }

        void Replay(Session &session, const ProfileRequest &profile)
        {
            this->PurgeRetained(Clock::now());
            std::vector<ClusterDelivery> records;
            if (m_Cluster)
            {
                records = m_Cluster->Retained(profile.m_Entries);
            }
            for (const auto &[topic, queue] : m_Retained)
            {
                bool match = false;
                for (const auto &entry : profile.m_Entries)
                {
                    if (entry.m_Type != MMQP_PROREQ_TYPE_SUB || !entry.m_Qos || entry.m_Request)
                    {
                        continue;
                    }
                    const auto &filter =
                        entry.m_Topic.empty() ? session.m_Profiles.m_Subscribers.Resolve(entry.m_Alias) : entry.m_Topic;
                    if (MatchTopic(CanonicalTopic(filter), topic))
                    {
                        match = true;
                        break;
                    }
                }
                if (!match)
                {
                    continue;
                }
                for (const auto &record : queue)
                {
                    records.push_back({{}, {}, session.m_Identifier, record.m_Message, SaveExpiry(record.m_Expiry)});
                }
            }
            size_t bytes = session.m_ReplayBytes;
            for (const auto &record : records)
            {
                auto size = Codec::EncodeFrame(record.m_Publication, {true, UINT32_MAX}).Size();
                if (bytes > m_Options.m_MaxReplayBytes || size > m_Options.m_MaxReplayBytes - bytes)
                {
                    throw ProtocolError("Retained replay budget exceeded", MMQP_REASON_BUSY);
                }
                bytes += size;
            }
            session.m_ReplayBytes = bytes;
            for (auto &record : records)
            {
                session.m_Replay.push_back(std::move(record));
            }
            this->Changed(session);
            this->DrainReplay(session);
        }

        void DrainReplay(Session &session)
        {
            if (session.m_Peer.expired())
            {
                return;
            }
            while (!session.m_Replay.empty())
            {
                const auto &record = session.m_Replay.front();
                if (record.m_Expiry > UnixMilliseconds())
                {
                    try
                    {
                        this->Deliver(this->Prepare(session, record.m_Publication, LoadExpiry(record.m_Expiry)));
                    }
                    catch (const ProtocolError &error)
                    {
                        if (error.m_Reason == MMQP_REASON_BUSY)
                        {
                            return;
                        }
                        throw;
                    }
                }
                session.m_ReplayBytes -= Codec::EncodeFrame(record.m_Publication, {true, UINT32_MAX}).Size();
                session.m_Replay.pop_front();
                this->Changed(session);
            }
        }

        size_t RequestCount() const
        {
            size_t count = 0;
            for (const auto &[id, entity] : m_Sessions)
            {
                if (!m_Cluster || m_Cluster->Owns(id))
                {
                    count += m_Registry.Get<Session>(entity).m_Requests.size();
                }
            }
            return count;
        }

        std::string RequestKey(std::string_view requester, uint16_t correlationId)
        {
            Writer writer;
            writer.String(requester);
            writer.U16(correlationId);
            return "rpc:" + ToHex(Sha256(writer.Data()));
        }

        Buffer SaveRequest(const Request &request)
        {
            Writer writer(m_Options.m_MaxStateBytes);
            writer.U32(0x45525131);
            writer.String(request.m_Requester);
            writer.U16(request.m_CorrelationId);
            writer.String(request.m_Topic);
            writer.String(request.m_Broker);
            writer.U64(SaveExpiry(request.m_Expiry));
            writer.U8(request.m_Qos);
            writer.U32(static_cast<uint32_t>(request.m_Responders.size()));
            for (const auto &responder : request.m_Responders)
            {
                writer.String(responder);
            }
            return writer.Take();
        }

        Request ReadRequest(std::span<const uint8_t> bytes)
        {
            Reader reader(bytes);
            if (reader.U32() != 0x45525131)
            {
                throw ProtocolError("Unknown request snapshot version");
            }
            Request result;
            result.m_Requester = reader.String();
            result.m_CorrelationId = reader.U16();
            result.m_Topic = ParseTopic(reader.String()).m_Filter;
            result.m_Broker = reader.String();
            result.m_Expiry = LoadExpiry(reader.U64());
            result.m_Qos = reader.U8();
            if (result.m_Requester.empty() || !result.m_CorrelationId || result.m_Qos > 1)
            {
                throw ProtocolError("Invalid stored request");
            }
            auto count = reader.U32();
            if (count > m_Options.m_MaxResponders)
            {
                throw ProtocolError("Cluster responder limit exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto responder = reader.String();
                if (responder.empty() || !result.m_Responders.insert(responder).second)
                {
                    throw ProtocolError("Invalid stored responder");
                }
            }
            reader.Finish();
            return result;
        }

        bool AcceptResponse(const Session &session, const PublishRequest &publication, const std::string &sender) const
        {
            auto found = session.m_Requests.find(publication.m_CorrelationId);
            return found != session.m_Requests.end() && found->second.m_Topic == publication.m_Topic &&
                   found->second.m_Expiry > Clock::now() && found->second.m_Responders.contains(sender);
        }

        void UpdateRequest(const Request &request)
        {
            if (!m_Cluster)
            {
                return;
            }
            m_Effects.push_back(
                [this, request]
                {
                    auto key = this->RequestKey(request.m_Requester, request.m_CorrelationId);
                    try
                    {
                        if (request.m_Expiry > Clock::now())
                        {
                            m_Cluster->Commit(key, this->SaveRequest(request), SaveExpiry(request.m_Expiry));
                        }
                        if (request.m_Responders.empty())
                        {
                            m_Cluster->Release(key);
                        }
                    }
                    catch (const ProtocolError &)
                    {
                    }
                });
        }

        void CompleteResponse(Session &session, uint16_t correlationId, const std::string &responder)
        {
            auto found = session.m_Requests.find(correlationId);
            if (found == session.m_Requests.end() || !found->second.m_Responders.erase(responder))
            {
                return;
            }
            this->UpdateRequest(found->second);
            if (found->second.m_Responders.empty())
            {
                session.m_RequestIds.Release(correlationId);
                session.m_Requests.erase(found);
            }
            this->Changed(session);
        }

        bool AdmissionPending(const Session &session, uint8_t type, uint16_t correlationId) const
        {
            return std::any_of(
                session.m_RpcAdmissions.begin(), session.m_RpcAdmissions.end(), [&](const auto &value)
                { return value.second.m_Type == type && value.second.m_CorrelationId == correlationId; });
        }

        void StageExchange(Session &session, const PublishRequest &publication)
        {
            auto bytes = Codec::EncodeFrame(publication, {true, UINT32_MAX}).Size();
            for (const auto &[id, message] : session.m_RpcAdmissions)
            {
                bytes += Codec::EncodeFrame(message, {true, UINT32_MAX}).Size();
            }
            if (session.m_RpcAdmissions.size() >= m_Options.m_MaxInflight || bytes > m_Options.m_MaxSessionBytes)
            {
                throw ProtocolError("RPC-admission recovery budget exhausted", MMQP_REASON_BUSY);
            }
            session.m_RpcAdmissions.emplace(publication.m_PacketId, publication);
            this->Changed(session);
        }

        void RecoverExchanges(Session &session)
        {
            if (!m_Cluster)
            {
                return;
            }
            for (auto it = session.m_RpcAdmissions.begin(); it != session.m_RpcAdmissions.end();)
            {
                const auto &message = it->second;
                auto publication = message;
                TimePoint expiry;
                std::vector<ClusterProfile> targets;
                if (message.m_Type == MMQP_PUBREQ_TYPE_REQ)
                {
                    const auto &request = session.m_Requests.at(message.m_CorrelationId);
                    expiry = request.m_Expiry;
                    for (const auto &responder : request.m_Responders)
                    {
                        targets.push_back({{}, responder, {}, SaveExpiry(expiry), true});
                    }
                }
                else
                {
                    const auto &route = session.m_ResponseRoutes.at(message.m_CorrelationId);
                    expiry = route.m_Expiry;
                    publication.m_CorrelationId = route.m_CorrelationId;
                    targets.push_back({{}, route.m_Requester, {}, SaveExpiry(expiry), true});
                }
                Reason reason;
                if (expiry <= Clock::now())
                {
                    reason = {MMQP_REASON_TIMEOUT, "Exchange expired during admission recovery"};
                }
                else
                {
                    // Repeat the original inbox writes with the same identity.
                    // Current SID owners discover them through quorum polling;
                    // stale broker locations need not be restored or guessed.
                    m_Cluster->Publish(session.m_Identifier, session.m_Received.Sequence(message.m_PacketId),
                                       publication, targets, SaveExpiry(expiry));
                }
                auto canonical = message;
                canonical.m_Alias = 1;
                canonical.m_Duplicate = false;
                auto sequence = session.m_Received.Sequence(message.m_PacketId);
                session.m_Received.Commit(message.m_PacketId);
                session.m_Receipts[sequence] = {FrameDigest(Codec::EncodeFrame(canonical, {true, UINT32_MAX})), reason};
                while (session.m_Receipts.size() > MMQP_PID_WINDOW)
                {
                    session.m_Receipts.erase(session.m_Receipts.begin());
                }
                if (message.m_Type == MMQP_PUBREQ_TYPE_RES)
                {
                    session.m_ResponseRoutes.erase(message.m_CorrelationId);
                    session.m_ResponseIds.Release(message.m_CorrelationId);
                }
                it = session.m_RpcAdmissions.erase(it);
                this->Changed(session);
            }
        }

        Reason RouteCluster(Session *publisher, PublishRequest publication, std::optional<ProfileEntry> registration,
                            const ResponseRoute *response, uint16_t correlationId)
        {
            auto sender = publisher ? publisher->m_Identifier : "status:" + m_Cluster->Identifier();
            uint64_t sequence =
                publisher && publication.m_Qos ? publisher->m_Received.Sequence(publication.m_PacketId) : 0;
            if (!publisher)
            {
                auto random = RandomBytes(8);
                Reader reader(random);
                sequence = reader.U64() | (1ULL << 63);
            }
            else if (!publication.m_Qos)
            {
                if (publisher->m_OriginSequence >= (1ULL << 63) - 1)
                {
                    throw ProtocolError("Publication origin sequence exhausted");
                }
                sequence = ++publisher->m_OriginSequence | (1ULL << 63);
            }
            auto expiry = Clock::now() + std::chrono::milliseconds(m_Options.m_RequestTimeout);
            if (registration && publication.m_Type == MMQP_PUBREQ_TYPE_MSG)
            {
                expiry = Clock::now() + std::chrono::seconds(DecodeInterval(registration->m_Expiry));
            }
            std::vector<ClusterProfile> targets;
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                auto bytes = m_Cluster->Read(this->RequestKey(response->m_Requester, publication.m_CorrelationId));
                if (bytes.empty())
                {
                    throw ProtocolError("Response has no pending request", MMQP_REASON_INVALID_CORRELATION);
                }
                auto request = this->ReadRequest(bytes);
                if (request.m_Requester != response->m_Requester ||
                    request.m_CorrelationId != publication.m_CorrelationId || request.m_Topic != publication.m_Topic ||
                    request.m_Expiry <= Clock::now() || !request.m_Responders.contains(sender))
                {
                    throw ProtocolError("Response is not authorized by the pending request", MMQP_REASON_UNAUTHORIZED);
                }
                expiry = request.m_Expiry;
                targets.push_back({request.m_Broker, request.m_Requester, {}, SaveExpiry(expiry), true});
            }
            else
            {
                auto candidates = m_Cluster->Resolve(publication.m_Topic, publication.m_Qos,
                                                     publication.m_Type == MMQP_PUBREQ_TYPE_REQ);
                std::map<std::string, ClusterProfile> selected;
                std::map<std::pair<std::string, std::string>, std::vector<ClusterProfile>> groups;
                for (const auto &candidate : candidates)
                {
                    auto [group, filter] = SplitShared(candidate.m_Profile.m_Topic);
                    if (group.empty())
                    {
                        selected.emplace(candidate.m_Session, candidate);
                    }
                    else if (candidate.m_Online)
                    {
                        groups[{std::string(group), std::string(filter)}].push_back(candidate);
                    }
                }
                for (auto &[group, members] : groups)
                {
                    auto &cursor = m_SharedCursor[group];
                    auto member = members[cursor++ % members.size()];
                    selected.emplace(member.m_Session, std::move(member));
                }
                for (auto &[id, candidate] : selected)
                {
                    targets.push_back(std::move(candidate));
                }
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                auto maximum = publication.m_Responses
                                   ? std::min<size_t>(publication.m_Responses, m_Options.m_MaxResponders)
                                   : m_Options.m_MaxResponders;
                if (targets.size() > maximum)
                {
                    targets.resize(maximum);
                }
                if (targets.empty())
                {
                    auto response = publication;
                    response.m_Type = MMQP_PUBREQ_TYPE_RES;
                    response.m_Payload.Reset();
                    response.m_Encryption.reset();
                    response.m_Reason = {MMQP_REASON_NO_RESPONDER, "No eligible responder"};
                    this->Deliver(this->Prepare(*publisher, response, expiry));
                    return {}; // Hop admission succeeded; the correlated response reports the application outcome.
                }
                if (publication.m_Qos)
                {
                    this->StageExchange(*publisher, publication);
                }
                Request request;
                request.m_Requester = sender;
                request.m_CorrelationId = publication.m_CorrelationId;
                request.m_Topic = publication.m_Topic;
                request.m_Expiry = expiry;
                request.m_Qos = publication.m_Qos;
                request.m_Broker = m_Cluster->Identifier();
                for (const auto &target : targets)
                {
                    request.m_Responders.insert(target.m_Session);
                }
                auto key = this->RequestKey(sender, publication.m_CorrelationId);
                if (auto previous = m_Cluster->Acquire(key);
                    !previous.empty() && !this->ReadRequest(previous).m_Responders.empty())
                {
                    m_Cluster->Release(key);
                    throw ProtocolError("Request correlation ID already exists", MMQP_REASON_INVALID_CORRELATION);
                }
                m_Cluster->Commit(key, this->SaveRequest(request), SaveExpiry(request.m_Expiry));
                publisher->m_Requests.emplace(publication.m_CorrelationId, std::move(request));
                this->Changed(*publisher);
                // Recovery must know the request before any responder can see it.
                this->Checkpoint();
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES && publication.m_Qos)
            {
                auto original = publication;
                original.m_CorrelationId = correlationId;
                this->StageExchange(*publisher, original);
                this->Checkpoint();
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_MSG && publication.m_Qos && registration)
            {
                auto identity = MessageIdentity(sender, sequence, publication.m_Topic,
                                                m_Cluster->Placement().Options().m_Epoch, publication.m_Payload);
                m_Cluster->Retain(publication, *registration, identity);
            }
            m_Cluster->Publish(sender, sequence, publication, targets, SaveExpiry(expiry));
            ++m_Metrics.m_Publications;
            if (publisher)
            {
                this->Changed(*publisher);
            }
            return {targets.empty() ? MMQP_REASON_NOSUBSCRIBER : MMQP_REASON_SUCCESS, {}};
        }

        void DrainInbox(Session &session)
        {
            if (!m_Cluster || session.m_Peer.expired() || session.m_NextInbox > Clock::now() ||
                !m_Cluster->Owns(session.m_Identifier))
            {
                return;
            }
            auto records = m_Cluster->Inbox(session.m_Identifier);
            // Notifications wake the drain immediately; polling only recovers lost notifications,
            // so an empty inbox backs off from 150 ms to 2 s.
            session.m_InboxInterval =
                records.empty() ? std::min<uint32_t>(2000, std::max<uint32_t>(150, session.m_InboxInterval * 2)) : 150;
            session.m_NextInbox = Clock::now() + std::chrono::milliseconds(session.m_InboxInterval);
            std::vector<ClusterDelivery> consumed;
            std::erase_if(session.m_FabricReceipts,
                          [](const auto &entry) { return entry.second <= UnixMilliseconds(); });
            for (const auto &record : records)
            {
                if (!session.m_FabricReceipts.contains(record.m_Identity))
                {
                    if (session.m_FabricReceipts.size() >= 8192)
                    {
                        break;
                    }
                    const auto &message = record.m_Publication;
                    this->AdmitTopic(message.m_Topic, false, &session);
                    bool allowed = message.m_Type == MMQP_PUBREQ_TYPE_RES
                                       ? this->AcceptResponse(session, message, record.m_Sender)
                                       : session.m_Profiles.Subscribed(message.m_Topic, message.m_Qos,
                                                                       message.m_Type == MMQP_PUBREQ_TYPE_REQ);
                    if (allowed)
                    {
                        try
                        {
                            this->Deliver(
                                this->Prepare(session, message, LoadExpiry(record.m_Expiry), record.m_Sender));
                        }
                        catch (const ProtocolError &error)
                        {
                            if (error.m_Reason == MMQP_REASON_BUSY)
                            {
                                break;
                            }
                            throw;
                        }
                        if (message.m_Type == MMQP_PUBREQ_TYPE_RES)
                        {
                            this->CompleteResponse(session, message.m_CorrelationId, record.m_Sender);
                        }
                    }
                    session.m_FabricReceipts[record.m_Identity] = record.m_Expiry;
                    this->Changed(session);
                }
                consumed.push_back(record);
            }
            if (!consumed.empty())
            {
                m_Effects.push_back([this, consumed = std::move(consumed)] { m_Cluster->Acknowledge(consumed); });
            }
        }

        Reason Route(Session *publisher, PublishRequest publication, std::optional<ProfileEntry> registration = {})
        {
            const ResponseRoute *response = nullptr;
            auto correlationId = publication.m_CorrelationId;
            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                if (!publisher)
                {
                    throw ProtocolError("Internal messages cannot issue requests");
                }
                publisher->m_RequestIds.Claim(publication.m_CorrelationId);
                this->Changed(*publisher);
                if (this->RequestCount() >= m_Options.m_MaxRequests)
                {
                    throw ProtocolError("Outstanding request budget exhausted", MMQP_REASON_BUSY);
                }
            }
            else if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                if (!publisher)
                {
                    throw ProtocolError("Response is missing its responder");
                }
                auto found = publisher->m_ResponseRoutes.find(publication.m_CorrelationId);
                if (found == publisher->m_ResponseRoutes.end() || found->second.m_Expiry <= Clock::now() ||
                    found->second.m_Topic != publication.m_Topic)
                {
                    throw ProtocolError("Response has no matching correlation route", MMQP_REASON_INVALID_CORRELATION);
                }
                response = &found->second;
                publication.m_CorrelationId = response->m_CorrelationId;
            }
            if (m_Cluster)
            {
                return this->RouteCluster(publisher, std::move(publication), registration, response, correlationId);
            }
            auto now = Clock::now();
            auto expiry = now + std::chrono::milliseconds(m_Options.m_RequestTimeout);
            if (publication.m_Type == MMQP_PUBREQ_TYPE_MSG && registration)
            {
                expiry = now + std::chrono::seconds(DecodeInterval(registration->m_Expiry));
            }
            std::vector<Session *> targets;
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                auto target = this->Find(response->m_Requester);
                if (!target || !this->AcceptResponse(*target, publication, publisher->m_Identifier))
                {
                    throw ProtocolError("Response has no matching outstanding request",
                                        MMQP_REASON_INVALID_CORRELATION);
                }
                targets.push_back(target);
                expiry = target->m_Requests.at(publication.m_CorrelationId).m_Expiry;
            }
            else
            {
                targets = this->Destinations(publication, publication.m_Type == MMQP_PUBREQ_TYPE_REQ);
            }

            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                auto count = publication.m_Responses
                                 ? std::min<size_t>(publication.m_Responses, m_Options.m_MaxResponders)
                                 : m_Options.m_MaxResponders;
                if (targets.size() > count)
                {
                    targets.resize(count);
                }
                if (targets.empty())
                {
                    auto response = publication;
                    response.m_Type = MMQP_PUBREQ_TYPE_RES;
                    response.m_Reason = {MMQP_REASON_NO_RESPONDER, "No eligible responder"};
                    response.m_Payload.Reset();
                    response.m_Encryption.reset();
                    this->Deliver(this->Prepare(*publisher, response, expiry));
                    return {}; // The no-responder result is carried by the correlated response.
                }
            }
            std::vector<Dispatch> deliveries;
            for (auto target : targets)
            {
                try
                {
                    deliveries.emplace_back(
                        this->Prepare(*target, publication, expiry, publisher ? publisher->m_Identifier : ""));
                }
                catch (const ProtocolError &)
                {
                    if (publication.m_Qos)
                    {
                        throw;
                    }
                    ++m_Metrics.m_Rejected;
                }
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_MSG && publication.m_Qos && registration)
            {
                this->Retain(publication, *registration);
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                Request request;
                request.m_Requester = publisher->m_Identifier;
                request.m_CorrelationId = publication.m_CorrelationId;
                request.m_Topic = publication.m_Topic;
                request.m_Expiry = expiry;
                request.m_Qos = publication.m_Qos;
                for (const auto &dispatch : deliveries)
                {
                    request.m_Responders.insert(dispatch.m_Session->m_Identifier);
                }
                publisher->m_Requests.emplace(publication.m_CorrelationId, std::move(request));
                this->Changed(*publisher);
            }
            for (auto &dispatch : deliveries)
            {
                this->Deliver(std::move(dispatch));
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                this->CompleteResponse(*targets.front(), publication.m_CorrelationId, publisher->m_Identifier);
            }
            ++m_Metrics.m_Publications;
            m_Owner.TryPost<PublicationRouted>(publication.m_Topic.size(), publication.m_Topic, deliveries.size(),
                                               publication.m_Payload.Size());
            return {deliveries.empty() ? MMQP_REASON_NOSUBSCRIBER : MMQP_REASON_SUCCESS, {}};
        }

        void Will(Session &session, bool birth)
        {
            if (!session.m_Properties.m_Will || m_Stopping)
            {
                return;
            }
            const auto &will = *session.m_Properties.m_Will;
            const auto &payload = birth ? will.m_Birth : will.m_Last;
            if (!payload)
            {
                return;
            }
            PublishRequest message;
            message.m_Topic = std::string(birth ? MMQP_BIRTH_STATUS_TOPIC_PREFIX : MMQP_LAST_STATUS_TOPIC_PREFIX) +
                              (will.m_Topic.empty() ? session.m_Identifier : will.m_Topic);
            message.m_Alias = 1;
            message.m_Qos = will.m_Qos;
            message.m_PacketId = will.m_Qos ? 1 : 0;
            message.m_Payload = *payload;
            try
            {
                this->Route(nullptr, std::move(message));
            }
            catch (const ProtocolError &)
            {
                ++m_Metrics.m_Rejected;
            }
        }

        void Publish(const std::shared_ptr<Peer> &peer, Session &session, PublishRequest publication)
        {
            Digest digest{};
            if (publication.m_Qos)
            {
                auto canonical = publication;
                if (canonical.m_Topic.empty())
                {
                    canonical.m_Topic = session.m_Profiles.m_Publishers.Resolve(canonical.m_Alias);
                }
                canonical.m_Alias = 1;
                canonical.m_Duplicate = false;
                digest = FrameDigest(Codec::EncodeFrame(canonical, {true, UINT32_MAX}));
            }
            auto state = publication.m_Qos ? session.m_Received.Inspect(publication.m_PacketId) : Reception::New;
            if (state == Reception::OutsideWindow)
            {
                throw ProtocolError("Publication exceeds the receive window", MMQP_REASON_INVALID_PID);
            }
            if (state == Reception::Duplicate)
            {
                auto found = session.m_Receipts.find(session.m_Received.Sequence(publication.m_PacketId));
                if (found == session.m_Receipts.end())
                {
                    throw ProtocolError("PID is older than the receipt window", MMQP_REASON_INVALID_PID);
                }
                if (!SecureEqual(found->second.first, digest))
                {
                    throw ProtocolError("Publication PID was reused for different content", MMQP_REASON_INVALID_PID);
                }
                ++m_Metrics.m_Duplicates;
                this->Send(peer, PublishAck{publication.m_PacketId, found->second.second});
                return;
            }
            Reason reason;
            auto packetId = publication.m_PacketId;
            auto qos = publication.m_Qos;
            auto type = publication.m_Type;
            auto correlationId = publication.m_CorrelationId;
            try
            {
                bool mapping = !publication.m_Topic.empty();
                if (!mapping)
                {
                    publication.m_Topic = session.m_Profiles.m_Publishers.Resolve(publication.m_Alias);
                }
                this->AdmitTopic(publication.m_Topic, false, &session);
                if (publication.m_Topic.front() == '$')
                {
                    throw ProtocolError("Reserved topics cannot be published by clients", MMQP_REASON_UNAUTHORIZED);
                }
                auto registration = session.m_Profiles.Registration(publication.m_Topic, publication.m_Type);
                if (!registration)
                {
                    throw ProtocolError("Publication has no authorized profile", MMQP_REASON_UNAUTHORIZED);
                }
                session.m_Profiles.m_Publishers.Map(publication.m_Alias, publication.m_Topic, true);
                if (mapping)
                {
                    this->Send(peer, ProfileSync{publication.m_Alias, publication.m_Topic, {}});
                }
                reason = this->Route(&session, std::move(publication), registration);
                if (reason.m_Code < 0x80)
                {
                    ++session.m_Publications;
                }
                if (type == MMQP_PUBREQ_TYPE_RES && reason.m_Code < 0x80)
                {
                    session.m_ResponseRoutes.erase(correlationId);
                    session.m_ResponseIds.Release(correlationId);
                }
            }
            catch (const ProtocolError &error)
            {
                ++m_Metrics.m_Rejected;
                if (!qos)
                {
                    if (type == MMQP_PUBREQ_TYPE_REQ && !session.m_Requests.contains(correlationId))
                    {
                        session.m_RequestIds.Release(correlationId);
                    }
                    throw;
                }
                reason = {error.m_Reason, error.what()};
            }
            if (type == MMQP_PUBREQ_TYPE_REQ && !session.m_Requests.contains(correlationId))
            {
                session.m_RequestIds.Release(correlationId);
                this->Changed(session);
            }
            if (qos)
            {
                session.m_RpcAdmissions.erase(packetId);
                auto sequence = session.m_Received.Sequence(packetId);
                session.m_Received.Commit(packetId);
                session.m_Receipts[sequence] = std::make_pair(digest, reason);
                while (session.m_Receipts.size() > MMQP_PID_WINDOW)
                {
                    session.m_Receipts.erase(session.m_Receipts.begin());
                }
                this->Send(peer, PublishAck{packetId, reason});
            }
        }

        void Profile(const std::shared_ptr<Peer> &peer, Session &session, const ProfileRequest &request)
        {
            auto digest = Sha256(Codec::Encode(request, peer->m_Codec));
            if (auto previous = session.m_ProfileReceipts.find(request.m_PacketId);
                previous != session.m_ProfileReceipts.end())
            {
                if (!SecureEqual(previous->second.first, digest))
                {
                    throw ProtocolError("Profile PID was reused for different content", MMQP_REASON_INVALID_PID);
                }
                this->Send(peer, ProfileAck{request.m_PacketId, previous->second.second});
                return;
            }
            if (request.m_PacketId <= session.m_LastProfile)
            {
                throw ProtocolError("Profile PID is older than the receipt window", MMQP_REASON_INVALID_PID);
            }
            Reason reason;
            try
            {
                this->AdmitProfiles(request, &session);
                session.m_Profiles.Apply(request, m_Options.m_MaxProfiles);
            }
            catch (const ProtocolError &error)
            {
                reason = {error.m_Reason, error.what()};
            }
            session.m_ProfileReceipts.emplace(request.m_PacketId, std::make_pair(digest, reason));
            session.m_LastProfile = request.m_PacketId;
            this->Changed(session, true);
            this->PruneShared();
            while (session.m_ProfileReceipts.size() > m_Options.m_MaxInflight)
            {
                session.m_ProfileReceipts.erase(session.m_ProfileReceipts.begin());
            }
            this->Send(peer, ProfileAck{request.m_PacketId, reason});
            if (reason.m_Code < 0x80)
            {
                this->Replay(session, request);
            }
        }

        // Frames from one transport read form one transaction. Their single checkpoint
        // (group commit) precedes every acknowledgement and delivery they produced.
        void Receive(uint64_t id, std::span<const uint8_t> bytes)
        {
            m_Transaction = true;
            try
            {
                this->ReceivePacket(id, bytes);
            }
            catch (const std::exception &error)
            {
                this->Abort(id, error);
            }
        }

        void Flush(uint64_t id)
        {
            if (!m_Transaction)
            {
                return;
            }
            try
            {
                this->Finish();
            }
            catch (const std::exception &error)
            {
                this->Abort(id, error);
            }
        }

        void Abort(uint64_t id, [[maybe_unused]] const std::exception &error)
        {
            m_Transaction = false;
            m_Effects.clear();
            m_EffectBytes = 0;
            EMPY_ERROR("Service checkpoint failed: {}", error.what());
            if (auto peer = m_Peers.find(id); peer != m_Peers.end())
            {
                this->Fail(peer->second, {MMQP_REASON_BUSY, "Durable majority unavailable"});
            }
        }

        void ReceivePacket(uint64_t id, std::span<const uint8_t> bytes)
        {
            auto found = m_Peers.find(id);
            if (found == m_Peers.end())
            {
                return;
            }
            auto peer = found->second;
            ++m_Metrics.m_ReceivedPackets;
            m_Metrics.m_ReceivedBytes += bytes.size();
            if (auto session = this->Find(peer->m_Session))
            {
                ++session->m_Traffic.m_ReceivedPackets;
                session->m_Traffic.m_ReceivedBytes += bytes.size();
            }
            try
            {
                auto message = Codec::Decode(bytes, peer->m_Codec);
                if (peer->m_Phase == Phase::Request)
                {
                    auto request = std::get_if<ConnectionRequest>(&message);
                    if (!request)
                    {
                        throw ProtocolError("CONREQ must be the first packet");
                    }
                    this->Connect(peer, std::move(*request));
                    return;
                }
                if (peer->m_Phase == Phase::Challenge)
                {
                    auto sync = std::get_if<ConnectionSync>(&message);
                    if (!sync || !sync->m_Token || !SecureEqual(*sync->m_Token, peer->m_Expected))
                    {
                        throw ProtocolError("Authentication proof failed", MMQP_REASON_AUTH_FAILED);
                    }
                    SecureErase(peer->m_Expected);
                    peer->m_Expected.clear();
                    this->Establish(peer, true);
                    return;
                }
                if (peer->m_Phase != Phase::Active)
                {
                    return;
                }
                auto session = this->Find(peer->m_Session);
                if (!session)
                {
                    throw ProtocolError("Missing active session", MMQP_REASON_SESSION_EXPIRED);
                }
                if (m_Cluster && !m_Cluster->Owns(session->m_Identifier))
                {
                    throw ProtocolError("Session primary lost its majority", MMQP_REASON_BUSY);
                }
                this->Changed(*session);
                session->m_Activity = Clock::now();
                std::visit(
                    [&](auto &value)
                    {
                        using Type = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<Type, PublishRequest>)
                        {
                            this->Publish(peer, *session, std::move(value));
                        }
                        else if constexpr (std::is_same_v<Type, ProfileRequest>)
                        {
                            this->Profile(peer, *session, value);
                        }
                        else if constexpr (std::is_same_v<Type, ProfileSync>)
                        {
                            if (value.m_Reason.m_Code >= 0x80)
                            {
                                throw ProtocolError("Peer rejected alias mapping", value.m_Reason.m_Code);
                            }
                            session->m_Profiles.m_Subscribers.Confirm(value.m_Alias, value.m_Topic);
                        }
                        else if constexpr (std::is_same_v<Type, PublishAck>)
                        {
                            if (!session->m_Pending.Remove(value.m_PacketId) &&
                                !session->m_Sequence.Sent(value.m_PacketId))
                            {
                                throw ProtocolError("Acknowledgement references an unsent PID",
                                                    MMQP_REASON_INVALID_PID);
                            }
                            this->DrainReplay(*session);
                        }
                        else if constexpr (std::is_same_v<Type, ConnectionSync>)
                        {
                            if (value.m_Token || !session->m_Ping)
                            {
                                throw ProtocolError("Unexpected session synchronization");
                            }
                            session->m_Ping = false;
                            m_DirtyProfiles.insert(session->m_Identifier);
                        }
                        else if constexpr (std::is_same_v<Type, ConnectionEnd>)
                        {
                            peer->m_Phase = Phase::Closing;
                            this->Close(peer, true);
                        }
                        else
                        {
                            throw ProtocolError("Packet is invalid in the active broker state");
                        }
                    },
                    message);
            }
            catch (const ProtocolError &error)
            {
                this->Fail(peer, {error.m_Reason, error.what()}, peer->m_Session.empty());
            }
            catch (const std::exception &error)
            {
                EMPY_ERROR("Service processing failed: {}", error.what());
                this->Fail(peer, {MMQP_REASON_INTERNAL_ERROR, "Internal processing failure"}, peer->m_Session.empty());
            }
        }

        void Closed(uint64_t id, const std::string &reason)
        {
            auto found = m_Peers.find(id);
            if (found == m_Peers.end())
            {
                return;
            }
            auto peer = found->second;
            m_Transaction = true;
            peer->m_Phase = Phase::Closing;
            if (auto session = this->Find(peer->m_Session))
            {
                session->m_Peer.reset();
                session->m_Ping = false;
                session->m_Expiry =
                    session->m_Resumable
                        ? Clock::now() + std::chrono::seconds(DecodeInterval(*session->m_Properties.m_Policy->m_Expiry))
                        : Clock::now();
                if (!m_Cluster || m_Cluster->Owns(session->m_Identifier))
                {
                    this->Changed(*session, true);
                    this->Will(*session, false);
                    if (m_Cluster)
                    {
                        m_Effects.push_back([this, id = session->m_Identifier] { m_Cluster->Release(id); });
                    }
                }
                m_Owner.TryPost<SessionDisconnected>(session->m_Identifier.size() + reason.size(),
                                                     session->m_Identifier, reason);
            }
            SecureErase(peer->m_Expected);
            m_Peers.erase(found);
            try
            {
                this->Finish();
            }
            catch (...)
            {
                m_Transaction = false;
                m_Effects.clear();
                m_EffectBytes = 0;
            }
        }

        void Tick()
        {
            if (m_Stopping)
            {
                return;
            }
            m_Timer.Schedule(std::chrono::milliseconds(50),
                             [this]
                             {
                                 {
                                     try
                                     {
                                         this->Tick();
                                     }
                                     catch (const std::exception &failure)
                                     {
                                         m_Transaction = false;
                                         m_Effects.clear();
                                         m_EffectBytes = 0;
                                         EMPY_ERROR("Service timer failed: {}", failure.what());
                                     }
                                 }
                             });
            m_Transaction = true;
            auto now = Clock::now();
            std::vector<std::shared_ptr<Peer>> timedOut;
            for (const auto &[id, peer] : m_Peers)
            {
                if (peer->m_Phase != Phase::Active && peer->m_Phase != Phase::Closing && peer->m_Deadline <= now)
                {
                    timedOut.push_back(peer);
                }
            }
            for (const auto &peer : timedOut)
            {
                this->Fail(peer, {MMQP_REASON_TIMEOUT, "Handshake timeout"}, true);
            }
            for (auto it = m_Sessions.begin(); it != m_Sessions.end();)
            {
                auto &session = m_Registry.Get<Session>(it->second);
                if (m_Cluster && !session.m_Peer.expired() && !m_Cluster->Owns(session.m_Identifier))
                {
                    this->Fail(session.m_Peer.lock(), {MMQP_REASON_BUSY, "Session primary is fenced"});
                    ++it;
                    continue;
                }
                if (session.m_Expiry <= now)
                {
                    for (auto &[id, request] : session.m_Requests)
                    {
                        request.m_Responders.clear();
                        this->UpdateRequest(request);
                    }
                    m_Removed.insert(it->first);
                    this->Destroy(it++);
                    this->PruneShared();
                    m_StateDirty = true;
                    continue;
                }
                ++it;
                if (std::erase_if(session.m_ResponseRoutes,
                                  [&](const auto &value)
                                  {
                                      if (value.second.m_Expiry > now ||
                                          this->AdmissionPending(session, MMQP_PUBREQ_TYPE_RES, value.first))
                                      {
                                          return false;
                                      }
                                      session.m_ResponseIds.Release(value.first);
                                      return true;
                                  }))
                {
                    this->Changed(session);
                }
                if ((m_Storage || m_Cluster) && !session.m_Peer.expired() && session.m_NextCheckpoint <= now)
                {
                    auto interval =
                        std::min<uint32_t>(30000, DecodeInterval(*session.m_Properties.m_Policy->m_Expiry) * 1000U / 3);
                    session.m_NextCheckpoint = now + std::chrono::milliseconds(interval);
                    this->Changed(session, true);
                }
                this->DrainReplay(session);
                this->DrainInbox(session);
                std::vector<uint16_t> expired;
                for (const auto &[packetId, pending] : session.m_Pending.m_Entries)
                {
                    if (pending.m_Expiry <= now)
                    {
                        expired.push_back(packetId);
                    }
                }
                for (auto packetId : expired)
                {
                    session.m_Pending.Remove(packetId);
                }
                if (!expired.empty())
                {
                    this->Changed(session);
                }
                if (!expired.empty() && session.m_Properties.m_Quic)
                {
                    // An expired hole cannot be skipped by the unordered checkpoint on the wire.
                    session.m_Resumable = false;
                    if (auto peer = session.m_Peer.lock())
                    {
                        this->Fail(peer, {MMQP_REASON_TIMEOUT, "Reliable publication expired"});
                    }
                    else
                    {
                        session.m_Expiry = now;
                    }
                    continue;
                }
                if (auto peer = session.m_Peer.lock())
                {
                    auto keepalive = std::chrono::seconds(DecodeInterval(*session.m_Properties.m_Policy->m_Keepalive));
                    if (session.m_Ping && session.m_PingDeadline <= now)
                    {
                        this->Fail(peer, {MMQP_REASON_KEEPALIVE_TIMEOUT, "Keepalive response timed out"});
                        continue;
                    }
                    if (!session.m_Ping && session.m_Activity + keepalive <= now)
                    {
                        session.m_Ping = true;
                        session.m_PingDeadline = now + keepalive;
                        this->Send(peer, ConnectionSync{});
                    }
                    bool failed = false;
                    for (auto &[packetId, pending] : session.m_Pending.m_Entries)
                    {
                        if (pending.m_Retry > now)
                        {
                            continue;
                        }
                        if (pending.m_Attempts >= m_Options.m_MaxRetries)
                        {
                            failed = true;
                            break;
                        }
                        auto message = pending.m_Message;
                        message.m_Duplicate = true;
                        if (!this->Send(peer, message))
                        {
                            failed = true;
                            break;
                        }
                        pending.m_Retry = now + std::chrono::milliseconds(m_Options.m_RetryInterval);
                        ++pending.m_Attempts;
                        ++m_Metrics.m_Retransmitted;
                    }
                    if (failed)
                    {
                        this->Fail(peer, {MMQP_REASON_TIMEOUT, "Reliable delivery timed out"});
                    }
                }
            }
            std::erase_if(m_AuthTokens, [&](const auto &value) { return value.second <= now; });
            this->PurgeRetained(now);
            for (const auto &[id, entity] : m_Sessions)
            {
                if (m_Cluster && !m_Cluster->Owns(id))
                {
                    continue;
                }
                auto &session = m_Registry.Get<Session>(entity);
                for (auto it = session.m_Requests.begin(); it != session.m_Requests.end();)
                {
                    if (it->second.m_Expiry > now || this->AdmissionPending(session, MMQP_PUBREQ_TYPE_REQ, it->first))
                    {
                        ++it;
                        continue;
                    }
                    if (!it->second.m_Responders.empty())
                    {
                        it->second.m_Responders.clear();
                        this->UpdateRequest(it->second);
                        this->Changed(session);
                    }
                    PublishRequest response;
                    response.m_Topic = it->second.m_Topic;
                    response.m_Type = MMQP_PUBREQ_TYPE_RES;
                    response.m_CorrelationId = it->first;
                    response.m_Qos = it->second.m_Qos;
                    response.m_Reason = {MMQP_REASON_TIMEOUT, "Response timeout"};
                    try
                    {
                        this->Deliver(this->Prepare(session, response, now + std::chrono::seconds(5)));
                    }
                    catch (const ProtocolError &error)
                    {
                        ++m_Metrics.m_Rejected;
                        // Keep the timed-out exchange until its result can be
                        // queued. Another request must not consume its identity.
                        if (error.m_Reason == MMQP_REASON_BUSY)
                        {
                            ++it;
                            continue;
                        }
                    }
                    session.m_RequestIds.Release(it->first);
                    it = session.m_Requests.erase(it);
                    this->Changed(session);
                }
            }
            this->Finish();
        }
    };

    inline Service::Service(Configs options) : m_Impl(std::make_unique<Impl>(*this, std::move(options)))
    {
    }
    inline Service::Service(const char *filename) : Service(LoadConfigs(filename))
    {
    }
    inline Service::~Service()
    {
        m_Workers.Stop();
        this->Stop();
    }

    inline uint16_t Service::Listen()
    {
        if (m_Impl->m_Started || m_Impl->m_Stopping)
        {
            throw std::logic_error("Service already started or stopped");
        }
        this->StartCluster();
        m_Impl->m_Port = m_Impl->m_Io.Listen(
            m_Impl->m_Options.m_Address, m_Impl->m_Options.m_Port, m_Impl->m_Options.m_Transport,
            [this](auto connection) { m_Impl->Accepted(std::move(connection)); }, m_Impl->m_Options.m_MaxSessions);
        m_Impl->m_Started = true;
        m_Impl->m_StartedAt = Clock::now();
        m_Impl->m_Io.Post([this] { m_Impl->Tick(); });
        m_Impl->m_Io.Init(m_Impl->m_Options.m_Workers);
        return m_Impl->m_Port;
    }

    inline uint16_t Service::Port() const
    {
        return m_Impl->m_Port;
    }

    inline BrokerMetrics Service::Metrics()
    {
        auto collect = [this]
        {
            auto metrics = m_Impl->m_Metrics;
            metrics.m_Sessions = m_Impl->m_Sessions.size();
            metrics.m_Online = 0;
            metrics.m_Pending = 0;
            for (const auto &[id, entity] : m_Impl->m_Sessions)
            {
                const auto &session = m_Impl->m_Registry.Get<Impl::Session>(entity);
                if (!session.m_Peer.expired())
                {
                    ++metrics.m_Online;
                }
                metrics.m_Pending += session.m_Pending.m_Entries.size();
            }
            return metrics;
        };
        return m_Impl->m_Started && !m_Impl->m_Stopping ? m_Impl->m_Io.Submit(collect).get() : collect();
    }

    inline BrokerSnapshot Service::Inspect(size_t maxSessions)
    {
        if (!maxSessions || maxSessions > 4096)
        {
            throw std::invalid_argument("Snapshot session limit must be in 1..4096");
        }
        auto collect = [this, maxSessions]
        {
            BrokerSnapshot result;
            const auto &options = m_Impl->m_Options;
            auto now = Clock::now();
            auto milliseconds = [](auto duration) -> uint64_t
            { return std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()); };
            result.m_Identifier = m_Impl->m_Cluster ? m_Impl->m_Cluster->Identifier()
                                                    : options.m_Address + ':' + std::to_string(m_Impl->m_Port);
            result.m_Instance = m_Impl->m_Instance;
            result.m_Address = options.m_Address;
            result.m_Port = m_Impl->m_Port;
            result.m_Transport = options.m_Transport.m_Quic ? "QUIC" : options.m_Transport.m_Tls ? "TLS" : "TCP";
            result.m_CollectedAt = UnixMilliseconds();
            result.m_UptimeMs = m_Impl->m_StartedAt == TimePoint{} ? 0 : milliseconds(now - m_Impl->m_StartedAt);
            result.m_Durable = m_Impl->m_Cluster || m_Impl->m_Storage;
            result.m_AuthenticationRequired = options.m_RequireAuth;
            result.m_Metrics = m_Impl->m_Metrics;
            result.m_Metrics.m_Sessions = m_Impl->m_Sessions.size();
            result.m_MaxSessions = options.m_MaxSessions;
            result.m_MaxInflight = options.m_MaxInflight;
            result.m_MaxRetainedBytes = options.m_MaxRetainedBytes;
            result.m_MaxStateBytes = options.m_Cluster ? options.m_Cluster->m_MaxBytes : options.m_MaxStateBytes;
            if (m_Impl->m_Storage)
            {
                result.m_StoredBytes = m_Impl->m_Storage->Bytes();
            }
            std::vector<std::string> identifiers;
            for (const auto &[id, entity] : m_Impl->m_Sessions)
            {
                const auto &session = m_Impl->m_Registry.Get<Impl::Session>(entity);
                auto peer = session.m_Peer.lock();
                if (peer)
                {
                    ++result.m_Metrics.m_Online;
                }
                result.m_Metrics.m_Pending += session.m_Pending.m_Entries.size();
                if (result.m_Sessions.size() == maxSessions)
                {
                    ++result.m_OmittedSessions;
                    continue;
                }
                SessionSnapshot value;
                static_cast<TrafficMetrics &>(value) = session.m_Traffic;
                value.m_Identifier = id;
                value.m_Realm = session.m_Realm;
                value.m_Transport = peer ? (peer->m_Connection->Quic()     ? "QUIC"
                                            : peer->m_Connection->Secure() ? "TLS"
                                                                           : "TCP")
                                         : "";
                value.m_Online = peer != nullptr;
                value.m_Authenticated = session.m_Authenticated;
                value.m_Resumable = session.m_Resumable;
                value.m_Owner = !m_Impl->m_Cluster;
                value.m_ObservedAt = session.m_ObservedAt;
                value.m_Publications = session.m_Publications;
                value.m_Deliveries = session.m_Deliveries;
                if (session.m_Activity != TimePoint{})
                {
                    value.m_IdleMs = milliseconds(now - session.m_Activity);
                }
                if (!peer)
                {
                    value.m_ExpiresInMs = milliseconds(session.m_Expiry - now);
                }
                if (session.m_Properties.m_Policy && session.m_Properties.m_Policy->m_Keepalive)
                {
                    value.m_KeepaliveSeconds = DecodeInterval(*session.m_Properties.m_Policy->m_Keepalive);
                }
                for (const auto &profile : session.m_Profiles.m_Entries)
                {
                    if (profile.m_Type == MMQP_PROREQ_TYPE_SUB)
                    {
                        ++value.m_Subscriptions;
                        if (profile.m_Request)
                        {
                            ++value.m_Responders;
                        }
                    }
                    else if (profile.m_Type == MMQP_PROREQ_TYPE_REG)
                    {
                        ++value.m_Registrations;
                        if (profile.m_Request)
                        {
                            ++value.m_Requesters;
                        }
                    }
                }
                value.m_Pending = session.m_Pending.m_Entries.size();
                value.m_PendingBytes = session.m_Pending.Bytes();
                value.m_QueuedBytes = peer ? peer->m_Connection->Queued() : 0;
                value.m_Replay = session.m_Replay.size();
                value.m_ReplayBytes = session.m_ReplayBytes;
                value.m_Requests = session.m_Requests.size();
                value.m_ResponseRoutes = session.m_ResponseRoutes.size();
                identifiers.push_back(id);
                result.m_Sessions.push_back(std::move(value));
            }
            if (m_Impl->m_Cluster)
            {
                result.m_Cluster = m_Impl->m_Cluster->Inspect(identifiers);
                result.m_StoredBytes = result.m_Cluster->m_StoredBytes;
                for (auto &session : result.m_Sessions)
                {
                    if (auto found = result.m_Cluster->m_Ownership.find(session.m_Identifier);
                        found != result.m_Cluster->m_Ownership.end())
                    {
                        session.m_Owner = true;
                        session.m_Lease = found->second;
                    }
                }
            }
            return result;
        };
        return m_Impl->m_Started && !m_Impl->m_Stopping ? m_Impl->m_Io.Submit(collect).get() : collect();
    }

    inline void Service::Stop()
    {
        if (m_Impl->m_Stopping)
        {
            return;
        }
        m_Impl->m_Io.Stop(
            [this]
            {
                m_Impl->m_Stopping = true;
                m_Impl->m_Timer.Cancel();
            });
        this->StopCluster();
        this->Terminate();
    }

    inline const Configs &Service::Configuration() const
    {
        return m_Impl->m_Options;
    }
    inline void Service::StartCluster()
    {
        if (m_Impl->m_Cluster && !m_Impl->m_ClusterStarted)
        {
            m_Impl->m_Cluster->Start();
            m_Impl->m_ClusterStarted = true;
        }
    }
    inline void Service::StopCluster()
    {
        if (m_Impl->m_Cluster && m_Impl->m_ClusterStarted)
        {
            m_Impl->m_Cluster->Stop();
            m_Impl->m_ClusterStarted = false;
        }
    }
} // namespace eMQ::Server
