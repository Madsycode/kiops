#include <eMQ/Runtime/Service.h>
#include <eCore/Network/State.h>
#include <eCore/Network/IOContext.h>
#include <algorithm>
#include <deque>

namespace eMQ
{
    inline constexpr size_t MMQP_CLIENT_RECEIPTS = 256;

    struct Service::Impl
    {
        enum class Phase
        {
            Offline,
            Transport,
            Challenge,
            Ack,
            Active
        };
        struct ProfileTask
        {
            ProfileRequest m_Request;
            ProfileTable m_Accepted;
            std::shared_ptr<std::promise<Reason>> m_Result;
            TimePoint m_Deadline;
            TimePoint m_Retry;
            uint32_t m_Attempts = 0;
        };

        struct Exchange
        {
            std::string m_Topic;
            uint8_t m_Remaining = 0;
            uint16_t m_PacketId = 0;
            TimePoint m_Expiry;
        };

        Service &m_Owner;
        Configs m_Options;
        IoContext m_Io;
        IoTimer m_Timer;
        Connection::Pointer m_Connection;
        Phase m_Phase = Phase::Offline;
        CodecOptions m_Codec;
        ProfileTable m_Profiles;
        PublicationSequence m_Publications;
        PacketSequence m_ProfileIds;
        CorrelationIds m_RequestIds;
        std::map<uint16_t, Exchange> m_Requests;
        ReceiveWindow m_Received;
        std::map<uint64_t, Digest> m_Receipts; // keyed by incoming 64-bit sequence
        PendingQueue m_Pending;
        std::map<uint16_t, std::shared_ptr<std::promise<Reason>>> m_Results;
        std::deque<ProfileTask> m_ProfileTasks;
        std::shared_ptr<std::promise<std::string>> m_Connecting;
        std::shared_ptr<std::promise<void>> m_Disconnecting;
        std::string m_Identifier;
        Buffer m_Expected;
        std::optional<QuicPolicy> m_Negotiated;
        TimePoint m_Deadline;
        Reason m_CloseReason{MMQP_REASON_UNSPECIFIED, "Transport disconnected"};
        std::atomic_bool m_Active = false;
        std::atomic_bool m_Stopping = false;
        bool m_Resume = false;
        bool m_Resumable = true;
        bool m_ManualDisconnect = false;
        TimePoint m_ReconnectAt{};
        TimePoint m_LastReceive{};
        TimePoint m_LastSend{};
        std::chrono::milliseconds m_Keepalive{0};
        size_t m_ReconnectIndex = 0;

        Impl(Service &service, Configs settings)
            : m_Owner(service), m_Options(std::move(settings)), m_Io(4096, m_Options.m_MaxPendingBytes), m_Timer(m_Io)
        {
            if (!m_Options.m_MaxInflight || m_Options.m_MaxInflight > 256 || !m_Options.m_MaxProfiles ||
                !m_Options.m_MaxPendingBytes || !m_Options.m_MaxRequests || m_Options.m_MaxRequests > UINT16_MAX ||
                !m_Options.m_ResponseTimeout || !m_Options.m_RetryInterval || !m_Options.m_OperationTimeout ||
                !m_Options.m_MaxRetries || m_Options.m_Transport.m_MaxPacket < 64 ||
                m_Options.m_Transport.m_MaxQueue < m_Options.m_Transport.m_MaxPacket)
            {
                throw std::invalid_argument("Invalid client resource bounds");
            }
            if (!m_Options.m_AuthKey.empty() && m_Options.m_AuthKey.size() < 32)
            {
                throw std::invalid_argument("Authentication key must contain at least 32 bytes");
            }
            if (m_Options.m_AutoReconnect && (m_Options.m_FailoverEndpoints.empty() || !m_Options.m_ReconnectInterval))
            {
                throw std::invalid_argument("Automatic reconnect needs endpoints and a positive retry interval");
            }
            for (const auto &endpoint : m_Options.m_FailoverEndpoints)
            {
                if (endpoint.m_Host.empty() || !endpoint.m_Port)
                {
                    throw std::invalid_argument("Invalid failover endpoint");
                }
            }
            if (!m_Options.m_Properties.m_Policy)
            {
                m_Options.m_Properties.m_Policy =
                    SessionPolicy{true, EncodeInterval(300), EncodeInterval(30), m_Options.m_Transport.m_MaxPacket};
            }
            if (m_Options.m_Transport.m_Quic)
            {
                if (!m_Options.m_Properties.m_Quic)
                {
                    m_Options.m_Properties.m_Quic = QuicPolicy{MMQP_QUIC_STRATEGY_SEMANTIC, {}, {}};
                }
                m_Options.m_Transport.m_QuicPolicy = QuicPlan(*m_Options.m_Properties.m_Quic).Policy();
                m_Options.m_Properties.m_Quic = m_Options.m_Transport.m_QuicPolicy;
            }
            else if (m_Options.m_Properties.m_Quic)
            {
                throw std::invalid_argument("QUIC properties require a QUIC transport");
            }
            m_Options.m_Properties.m_Policy->m_MaxPacket =
                std::min(m_Options.m_Transport.m_MaxPacket,
                         m_Options.m_Properties.m_Policy->m_MaxPacket.value_or(m_Options.m_Transport.m_MaxPacket));
            m_Codec = {m_Options.m_Properties.m_Policy->m_Reasons, *m_Options.m_Properties.m_Policy->m_MaxPacket,
                       static_cast<uint32_t>(m_Options.m_MaxProfiles)};
            m_Pending.m_MaxCount = m_Options.m_MaxInflight;
            m_Pending.m_MaxBytes = m_Options.m_MaxPendingBytes;
        }

        ~Impl()
        {
            SecureErase(m_Options.m_AuthKey);
            SecureErase(m_Expected);
        }

        bool Write(const Message &message)
        {
            if (!m_Connection || !m_Connection->Send(Codec::EncodeFrame(message, m_Codec)))
            {
                return false;
            }
            m_LastSend = Clock::now();
            return true;
        }

        static Digest Receipt(PublishRequest &publication)
        {
            // Mirror the broker receipt: resolved topic, alias 1 and DUP clear.
            auto alias = publication.m_Alias;
            auto duplicate = publication.m_Duplicate;
            publication.m_Alias = 1;
            publication.m_Duplicate = false;
            auto digest = FrameDigest(Codec::EncodeFrame(publication, {true, UINT32_MAX}));
            publication.m_Alias = alias;
            publication.m_Duplicate = duplicate;
            return digest;
        }

        void Acknowledge(uint16_t packetId, const Digest &digest)
        {
            auto sequence = m_Received.Sequence(packetId);
            m_Received.Commit(packetId);
            m_Receipts[sequence] = digest;
            while (m_Receipts.size() > MMQP_CLIENT_RECEIPTS)
            {
                m_Receipts.erase(m_Receipts.begin());
            }
            this->Write(PublishAck{packetId, {}});
        }

        void Closed(const std::string &text)
        {
            m_Active = false;
            m_Phase = Phase::Offline;
            m_Connection.reset();
            auto failure = std::make_exception_ptr(std::runtime_error(text));
            if (m_Connecting)
            {
                m_Connecting->set_exception(failure);
                m_Connecting.reset();
            }
            if (m_Disconnecting)
            {
                m_Disconnecting->set_value();
                m_Disconnecting.reset();
            }
            bool recover =
                m_Options.m_AutoReconnect && !m_ManualDisconnect && !m_Stopping && m_Resumable && !m_Identifier.empty();
            if (!recover)
            {
                for (auto &[id, result] : m_Results)
                {
                    result->set_exception(failure);
                }
                m_Results.clear();
                if (!m_ProfileTasks.empty())
                {
                    m_Resumable = false;
                }
                for (auto &task : m_ProfileTasks)
                {
                    task.m_Result->set_exception(failure);
                }
                m_ProfileTasks.clear();
            }
            m_ReconnectAt = Clock::now() + std::chrono::milliseconds(m_Options.m_ReconnectInterval);
            SecureErase(m_Expected);
            m_Expected.clear();
            if (m_CloseReason.m_Code == MMQP_REASON_UNSPECIFIED)
            {
                m_CloseReason.m_Text = text;
            }
            m_Owner.TryPost<eMQ::Disconnect>(m_CloseReason.m_Text.size(), m_CloseReason, recover);
            m_Owner.TryPost<eMQ::Session>(m_Identifier.size() + m_CloseReason.m_Text.size(), m_Identifier, false, false,
                                          m_CloseReason);
        }

        void Fail(const ProtocolError &error)
        {
            m_CloseReason = {error.m_Reason, error.what()};
            // Broker silence leaves client state consistent, so the session may still resume.
            if (error.m_Reason != MMQP_REASON_KEEPALIVE_TIMEOUT &&
                !(m_Options.m_AutoReconnect && m_Phase != Phase::Active &&
                  (error.m_Reason == MMQP_REASON_BUSY || error.m_Reason == MMQP_REASON_TIMEOUT)))
            {
                m_Resumable = false;
            }
            if (m_Connection)
            {
                try
                {
                    this->Write(ConnectionEnd{m_CloseReason});
                }
                catch (...)
                {
                }
                m_Connection->Close(true);
            }
            else
            {
                this->Closed(error.what());
            }
        }

        void Ready(Connection::Pointer transport)
        {
            m_Connection = std::move(transport);
            m_Connection->m_OnReceive = [this](auto bytes) { this->Receive(bytes); };
            ConnectionRequest request;
            request.m_Properties = m_Options.m_Properties;
            request.m_Properties.m_Auth.reset();
            request.m_Resume = m_Resume;
            if (m_Resume)
            {
                request.m_Properties.m_Profile.reset();
                request.m_Properties.m_Will.reset();
                request.m_Properties.m_Auth = SessionAuth{m_Identifier, {}};
            }
            if (!m_Options.m_AuthKey.empty())
            {
                if (!m_Connection->Secure() && !m_Options.m_AllowInsecureAuth)
                {
                    throw ProtocolError("Authentication requires secure transport", MMQP_REASON_AUTH_FAILED);
                }
                if (!request.m_Properties.m_Auth)
                {
                    request.m_Properties.m_Auth = SessionAuth{};
                }
                request.m_Properties.m_Auth->m_Token = RandomBytes(32);
                if (!m_Options.m_AuthIdentity.empty())
                {
                    auto prefix = AuthenticationIdentity(m_Options.m_AuthIdentity);
                    std::copy(prefix.begin(), prefix.end(), request.m_Properties.m_Auth->m_Token.begin());
                }
                auto proof = Hmac256(m_Options.m_AuthKey, request.m_Properties.m_Auth->m_Token);
                m_Expected.assign(proof.begin(), proof.end());
                SecureErase(proof);
                m_Phase = Phase::Challenge;
            }
            else
            {
                m_Phase = Phase::Ack;
            }
            if (!this->Write(request))
            {
                throw ProtocolError("Connection request exceeds send budget", MMQP_REASON_BUSY);
            }
        }

        void Begin(std::string host, uint16_t port, bool restore, std::shared_ptr<std::promise<std::string>> result)
        {
            if (m_Phase != Phase::Offline || m_Stopping)
            {
                throw std::logic_error("Client is already connecting, connected or stopped");
            }
            if (restore && (m_Identifier.empty() || !m_Resumable))
            {
                throw std::logic_error("No safely resumable session is available");
            }
            if (restore)
            {
                for (const auto &[id, item] : m_Pending.m_Entries)
                {
                    if (item.m_Expiry <= Clock::now())
                    {
                        throw std::logic_error("A reliable publication expired; establish a fresh session");
                    }
                }
            }
            m_Resume = restore;
            m_ManualDisconnect = false;
            if (!m_Resume)
            {
                m_Identifier.clear();
                m_Profiles = {};
                m_Publications = PublicationSequence{};
                m_RequestIds = CorrelationIds{};
                m_Requests.clear();
                m_ProfileIds =
                    PacketSequence(m_Options.m_Properties.m_Profile ? m_Options.m_Properties.m_Profile->m_PacketId : 0);
                m_Received = ReceiveWindow(!m_Options.m_Transport.m_Quic);
                m_Receipts.clear();
                m_Negotiated.reset();
                m_Pending = PendingQueue{};
                m_Pending.m_MaxCount = m_Options.m_MaxInflight;
                m_Pending.m_MaxBytes = m_Options.m_MaxPendingBytes;
                m_Resumable = true;
            }
            m_Codec = {m_Options.m_Properties.m_Policy->m_Reasons, *m_Options.m_Properties.m_Policy->m_MaxPacket,
                       static_cast<uint32_t>(m_Options.m_MaxProfiles)};
            m_CloseReason = {MMQP_REASON_UNSPECIFIED, "Transport disconnected"};
            m_Connecting = std::move(result);
            m_Deadline = Clock::now() + std::chrono::milliseconds(m_Options.m_Transport.m_Timeout);
            m_Phase = Phase::Transport;
            try
            {
                m_Connection = m_Io.Connect(
                    std::move(host), port, m_Options.m_Transport, [this](auto transport)
                    { this->Ready(std::move(transport)); }, [this](const auto &text) { this->Closed(text); });
            }
            catch (...)
            {
                m_Phase = Phase::Offline;
                m_Connecting.reset();
                throw;
            }
        }

        void Connected(const ConnectionAck &ack)
        {
            if (ack.m_Reason.m_Code >= 0x80)
            {
                throw ProtocolError(ack.m_Reason.m_Text.empty() ? "Connection rejected" : ack.m_Reason.m_Text.c_str(),
                                    ack.m_Reason.m_Code);
            }
            if (m_Phase != Phase::Ack)
            {
                throw ProtocolError("Missing broker authentication proof", MMQP_REASON_AUTH_FAILED);
            }
            if (!ack.m_Properties.m_Auth || ack.m_Properties.m_Auth->m_Identifier.empty() ||
                !ack.m_Properties.m_Policy || !ack.m_Properties.m_Policy->m_MaxPacket ||
                !ack.m_Properties.m_Policy->m_Keepalive || !ack.m_Properties.m_Policy->m_Expiry ||
                ack.m_Properties.m_Quic.has_value() != m_Connection->Quic())
            {
                throw ProtocolError("Incomplete or unsupported negotiated session properties");
            }
            if (m_Resume && m_Negotiated != ack.m_Properties.m_Quic)
            {
                throw ProtocolError("Broker changed the resumed QUIC policy");
            }
            if (ack.m_Properties.m_Quic)
            {
                auto allowed = QuicPlan::Negotiate(*ack.m_Properties.m_Quic, m_Options.m_Transport.m_QuicPolicy);
                if (allowed != *ack.m_Properties.m_Quic)
                {
                    throw ProtocolError("Broker exceeded the requested QUIC resources");
                }
            }
            if (m_Resume && ack.m_Properties.m_Auth->m_Identifier != m_Identifier)
            {
                throw ProtocolError("Broker changed the resumed session identity", MMQP_REASON_AUTH_FAILED);
            }
            m_Identifier = ack.m_Properties.m_Auth->m_Identifier;
            m_Codec.m_Reasons = ack.m_Properties.m_Policy->m_Reasons;
            m_Codec.m_MaxPacket = std::min(m_Codec.m_MaxPacket, *ack.m_Properties.m_Policy->m_MaxPacket);
            m_Connection->SetLimit(m_Codec.m_MaxPacket);
            if (!m_Resume && m_Options.m_Properties.m_Profile)
            {
                m_Profiles.Apply(*m_Options.m_Properties.m_Profile, m_Options.m_MaxProfiles);
            }
            m_Keepalive = std::chrono::seconds(DecodeInterval(*ack.m_Properties.m_Policy->m_Keepalive));
            m_LastReceive = m_LastSend = Clock::now();
            m_Phase = Phase::Active;
            m_Active = true;
            m_Negotiated = ack.m_Properties.m_Quic;
            if (m_Negotiated)
            {
                m_Connection->Configure(*m_Negotiated, m_Codec.m_Reasons);
            }
            for (auto &[id, publication] : m_Pending.m_Entries)
            {
                publication.m_Retry = Clock::now();
                publication.m_Attempts = 0;
            }
            auto result = std::move(m_Connecting);
            for (auto &task : m_ProfileTasks)
            {
                task.m_Deadline = Clock::now() + std::chrono::milliseconds(m_Options.m_OperationTimeout);
                task.m_Retry = Clock::now();
                task.m_Attempts = 0;
            }
            this->PumpProfile();
            result->set_value(m_Identifier);
            m_Owner.TryPost<eMQ::Connect>(m_Identifier.size(), m_Identifier, m_Resume);
            m_Owner.TryPost<eMQ::Session>(m_Identifier.size(), m_Identifier, true, m_Resume, Reason{});
        }

        void PumpProfile()
        {
            while (!m_ProfileTasks.empty() && m_Phase == Phase::Active && !m_ProfileTasks.front().m_Request.m_PacketId)
            {
                auto &task = m_ProfileTasks.front();
                try
                {
                    task.m_Accepted = m_Profiles;
                    for (auto &entry : task.m_Request.m_Entries)
                    {
                        if (!entry.m_Topic.empty())
                        {
                            entry.m_Topic = CanonicalTopic(entry.m_Topic);
                            if (!entry.m_Alias && !HasWildcard(entry.m_Topic))
                            {
                                auto &aliases = entry.m_Type == MMQP_PROREQ_TYPE_REG ? task.m_Accepted.m_Publishers
                                                                                     : task.m_Accepted.m_Subscribers;
                                if (entry.m_Type == MMQP_PROREQ_TYPE_UNS)
                                {
                                    auto alias = aliases.Find(entry.m_Topic);
                                    if (!alias)
                                    {
                                        throw ProtocolError("No subscription alias to remove",
                                                            MMQP_REASON_INVALID_ALIAS);
                                    }
                                    entry.m_Alias = *alias;
                                }
                                else
                                {
                                    entry.m_Alias = aliases.Assign(
                                        entry.m_Topic, 1, entry.m_Type == MMQP_PROREQ_TYPE_SUB ? 0x7FFF : UINT16_MAX);
                                }
                            }
                        }
                    }
                    task.m_Request.m_PacketId = m_ProfileIds.Next();
                    task.m_Accepted.Apply(task.m_Request, m_Options.m_MaxProfiles);
                    // Reserve outgoing choices while the profile is in flight; live alias maps
                    // must also retain mappings learned from interleaved publications.
                    for (const auto &entry : task.m_Request.m_Entries)
                    {
                        if (!entry.m_Alias || entry.m_Type == MMQP_PROREQ_TYPE_UNS || entry.m_Topic.empty())
                        {
                            continue;
                        }
                        auto &aliases =
                            entry.m_Type == MMQP_PROREQ_TYPE_REG ? m_Profiles.m_Publishers : m_Profiles.m_Subscribers;
                        aliases.Map(entry.m_Alias, entry.m_Topic);
                    }
                    task.m_Deadline = Clock::now() + std::chrono::milliseconds(m_Options.m_OperationTimeout);
                    task.m_Retry = Clock::now() + std::chrono::milliseconds(m_Options.m_RetryInterval);
                    if (!this->Write(task.m_Request))
                    {
                        throw ProtocolError("Profile send budget exhausted", MMQP_REASON_BUSY);
                    }
                }
                catch (...)
                {
                    task.m_Result->set_exception(std::current_exception());
                    m_ProfileTasks.pop_front();
                }
            }
        }

        void ProfileAcked(const ProfileAck &ack)
        {
            if (m_ProfileTasks.empty() || m_ProfileTasks.front().m_Request.m_PacketId != ack.m_PacketId)
            {
                if (ack.m_PacketId <= m_ProfileIds.Value())
                {
                    return;
                }
                throw ProtocolError("Unknown profile acknowledgement", MMQP_REASON_INVALID_PID);
            }
            auto task = std::move(m_ProfileTasks.front());
            m_ProfileTasks.pop_front();
            if (ack.m_Reason.m_Code < 0x80)
            {
                m_Profiles.Apply(task.m_Request, m_Options.m_MaxProfiles);
            }
            task.m_Result->set_value(ack.m_Reason);
            for (const auto &entry : task.m_Request.m_Entries)
            {
                m_Owner.TryPost<eMQ::Profile>(entry.m_Topic.size() + ack.m_Reason.m_Text.size(), entry.m_Topic,
                                              ack.m_PacketId, ack.m_Reason);
            }
            this->PumpProfile();
        }

        void Publish(PublishRequest publication, const std::shared_ptr<std::promise<Reason>> &result)
        {
            if (m_Phase != Phase::Active)
            {
                throw std::logic_error("Client is not connected");
            }
            publication.m_Topic = ParseTopic(publication.m_Topic).m_Filter;
            if (!m_Profiles.Registration(publication.m_Topic, publication.m_Type))
            {
                throw ProtocolError("Register the topic or responder profile before publishing",
                                    MMQP_REASON_UNAUTHORIZED);
            }
            publication.m_Alias = m_Profiles.m_Publishers.Candidate(publication.m_Topic);
            auto qos = publication.m_Qos;
            auto packetId = qos ? m_Publications.Peek() : uint16_t{0};
            publication.m_PacketId = packetId;
            publication.m_Duplicate = false;
            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                if (m_Requests.size() >= m_Options.m_MaxRequests)
                {
                    throw ProtocolError("Active request budget exhausted", MMQP_REASON_BUSY);
                }
                if (!publication.m_CorrelationId)
                {
                    publication.m_CorrelationId = m_RequestIds.Peek();
                }
                if (m_RequestIds.Contains(publication.m_CorrelationId))
                {
                    throw ProtocolError("Correlation ID is already active", MMQP_REASON_INVALID_CORRELATION);
                }
            }
            auto topic = publication.m_Topic;
            if (m_Options.m_Cipher)
            {
                m_Options.m_Cipher->Encrypt(publication, m_Options.m_KeyEpoch);
            }
            // A confirmed alias is already mapped, so its topic is omitted and needs no remapping.
            bool confirmed = m_Profiles.m_Publishers.Confirmed(publication.m_Alias);
            if (confirmed)
            {
                publication.m_Topic.clear();
            }
            auto bytes = Codec::EncodeFrame(publication, m_Codec);
            auto capacity = m_Connection->Capacity();
            auto queued = std::min(capacity, m_Connection->Queued());
            // Match the strictest transport admission: QUIC rejects bytes >= remaining space.
            if (bytes.Size() >= capacity - queued ||
                (qos && !m_Pending.CanAdd(bytes.Size(), m_Publications.Issued() + 1)))
            {
                throw ProtocolError("Client send budget exhausted", MMQP_REASON_BUSY);
            }
            auto size = bytes.Size();
            if (!m_Connection->Send(std::move(bytes)))
            {
                throw ProtocolError("Transport rejected publication", MMQP_REASON_BUSY);
            }
            auto now = Clock::now();
            if (publication.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                m_RequestIds.Claim(publication.m_CorrelationId);
                m_Requests.emplace(publication.m_CorrelationId,
                                   Exchange{std::move(topic), publication.m_Responses, packetId,
                                            now + std::chrono::milliseconds(m_Options.m_ResponseTimeout)});
            }
            m_LastSend = now;
            if (!confirmed)
            {
                m_Profiles.m_Publishers.Map(publication.m_Alias, publication.m_Topic);
            }
            if (!qos)
            {
                auto observation = publication;
                if (observation.m_Topic.empty())
                {
                    observation.m_Topic = m_Profiles.m_Publishers.Resolve(observation.m_Alias);
                }
                const auto cost = observation.m_Topic.size() + observation.m_Payload.Size();
                m_Owner.TryPost<eMQ::Publish>(cost, std::move(observation), Reason{});
                result->set_value({});
                return;
            }
            m_Publications.Next();
            m_Pending.Add(std::move(publication), m_Publications.Issued(), size,
                          now + std::chrono::milliseconds(m_Options.m_RetryInterval),
                          now + std::chrono::milliseconds(m_Options.m_OperationTimeout));
            m_Results.emplace(packetId, result);
        }

        void Publication(PublishRequest publication)
        {
            bool mapping = !publication.m_Topic.empty();
            if (!mapping)
            {
                publication.m_Topic = m_Profiles.m_Subscribers.Resolve(publication.m_Alias);
            }
            Digest digest{};
            if (publication.m_Qos)
            {
                auto state = m_Received.Inspect(publication.m_PacketId);
                if (state == Reception::OutsideWindow)
                {
                    throw ProtocolError("Publication exceeds the receive window", MMQP_REASON_INVALID_PID);
                }
                digest = Receipt(publication);
                if (state == Reception::Duplicate)
                {
                    // Only an identical retransmission is acknowledged without redelivery.
                    auto found = m_Receipts.find(m_Received.Sequence(publication.m_PacketId));
                    if (found == m_Receipts.end())
                    {
                        throw ProtocolError("Delivery PID is older than the receipt cache", MMQP_REASON_INVALID_PID);
                    }
                    if (!SecureEqual(found->second, digest))
                    {
                        throw ProtocolError("Delivery PID was reused for different content", MMQP_REASON_INVALID_PID);
                    }
                    this->Write(PublishAck{publication.m_PacketId, {}});
                    return;
                }
            }
            if (mapping)
            {
                // Clients allocate concrete subscriber aliases below 0x8000; new broker-assigned ones lie above.
                if (publication.m_Alias < 0x8000 &&
                    m_Profiles.m_Subscribers.Find(publication.m_Topic) != publication.m_Alias)
                {
                    throw ProtocolError("Broker-assigned subscriber alias is outside 32768..65535",
                                        MMQP_REASON_INVALID_ALIAS);
                }
                m_Profiles.m_Subscribers.Map(publication.m_Alias, publication.m_Topic, true);
            }
            bool authorized = publication.m_Type == MMQP_PUBREQ_TYPE_RES
                                  ? m_Profiles.Registration(publication.m_Topic, MMQP_PUBREQ_TYPE_REQ).has_value()
                                  : m_Profiles.Subscribed(publication.m_Topic, publication.m_Qos,
                                                          publication.m_Type == MMQP_PUBREQ_TYPE_REQ);
            if (!authorized)
            {
                throw ProtocolError("Received publication is outside the client profile", MMQP_REASON_UNAUTHORIZED);
            }
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                auto found = m_Requests.find(publication.m_CorrelationId);
                if (found == m_Requests.end() || found->second.m_Topic != publication.m_Topic)
                {
                    throw ProtocolError("Response names an unknown request or topic", MMQP_REASON_INVALID_CORRELATION);
                }
            }
            bool decrypted = false;
            if (publication.m_Encryption)
            {
                if (!m_Options.m_Cipher)
                {
                    throw ProtocolError("No key provider for encrypted publication", MMQP_REASON_AUTH_FAILED);
                }
                try
                {
                    publication.m_Payload = m_Options.m_Cipher->Decrypt(publication, publication.m_Topic);
                }
                catch (const PayloadReplay &)
                {
                    // Retention can redeliver an authenticated envelope under a new broker PID.
                    if (mapping)
                    {
                        this->Write(ProfileSync{publication.m_Alias, publication.m_Topic, {}});
                    }
                    if (publication.m_Qos)
                    {
                        this->Acknowledge(publication.m_PacketId, digest);
                    }
                    return;
                }
                decrypted = true;
            }
            else if (m_Options.m_Cipher && m_Options.m_RequireEncryption &&
                     !(publication.m_Type == MMQP_PUBREQ_TYPE_RES && publication.m_Reason.m_Code >= 0x80 &&
                       publication.m_Payload.Empty()))
            {
                throw ProtocolError("Unencrypted application publication rejected", MMQP_REASON_AUTH_FAILED);
            }
            if (mapping)
            {
                this->Write(ProfileSync{publication.m_Alias, publication.m_Topic, {}});
            }
            auto packetId = publication.m_PacketId;
            auto qos = publication.m_Qos;
            auto correlationId = publication.m_CorrelationId;
            bool complete = false;
            if (publication.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                auto &exchange = m_Requests.at(correlationId);
                // An application error is still one responder's outcome. Only the
                // broker's empty terminal outcome closes the entire exchange.
                const bool terminal =
                    publication.m_Payload.Empty() && (publication.m_Reason.m_Code == MMQP_REASON_NO_RESPONDER ||
                                                      publication.m_Reason.m_Code == MMQP_REASON_TIMEOUT);
                complete = terminal || (exchange.m_Remaining && --exchange.m_Remaining == 0);
            }
            auto bytes = publication.m_Topic.size() + publication.m_Payload.Size();
            if (!m_Owner.TryPost<eMQ::Receive>(bytes, std::move(publication), decrypted))
            {
                throw ProtocolError("Application receive queue is full; poll events before accepting more data",
                                    MMQP_REASON_BUSY);
            }
            if (complete)
            {
                m_RequestIds.Release(correlationId);
                m_Requests.erase(correlationId);
            }
            if (qos)
            {
                this->Acknowledge(packetId, digest);
            }
        }

        void Receive(std::span<const uint8_t> bytes)
        {
            m_LastReceive = Clock::now();
            try
            {
                auto message = Codec::Decode(bytes, m_Codec);
                if (auto ack = std::get_if<ConnectionAck>(&message); ack && m_Phase != Phase::Active)
                {
                    this->Connected(*ack);
                    return;
                }
                if (m_Phase == Phase::Challenge)
                {
                    auto sync = std::get_if<ConnectionSync>(&message);
                    if (!sync || !sync->m_Token || !SecureEqual(m_Expected, *sync->m_Token))
                    {
                        throw ProtocolError("Broker authentication failed", MMQP_REASON_AUTH_FAILED);
                    }
                    auto response = Hmac256(m_Options.m_AuthKey, *sync->m_Token);
                    SecureErase(m_Expected);
                    m_Expected.clear();
                    m_Phase = Phase::Ack;
                    if (!this->Write(ConnectionSync{Buffer(response.begin(), response.end())}))
                    {
                        throw ProtocolError("Authentication response send failed", MMQP_REASON_BUSY);
                    }
                    return;
                }
                if (m_Phase != Phase::Active)
                {
                    throw ProtocolError("Unexpected packet during connection establishment");
                }
                std::visit(
                    [&](auto &value)
                    {
                        using Type = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<Type, PublishRequest>)
                        {
                            this->Publication(std::move(value));
                        }
                        else if constexpr (std::is_same_v<Type, ProfileAck>)
                        {
                            this->ProfileAcked(value);
                        }
                        else if constexpr (std::is_same_v<Type, PublishAck>)
                        {
                            if (auto item = m_Pending.m_Entries.find(m_Publications.Sequence(value.m_PacketId));
                                item != m_Pending.m_Entries.end())
                            {
                                auto observation = item->second.m_Message;
                                if (observation.m_Topic.empty())
                                {
                                    observation.m_Topic = m_Profiles.m_Publishers.Resolve(observation.m_Alias);
                                }
                                const auto cost = observation.m_Topic.size() + observation.m_Payload.Size() +
                                                  value.m_Reason.m_Text.size();
                                m_Owner.TryPost<eMQ::Publish>(cost, std::move(observation), value.m_Reason);
                            }
                            if (value.m_Reason.m_Code >= 0x80)
                            {
                                for (auto it = m_Requests.begin(); it != m_Requests.end(); ++it)
                                {
                                    if (it->second.m_PacketId != value.m_PacketId)
                                    {
                                        continue;
                                    }
                                    m_RequestIds.Release(it->first);
                                    m_Requests.erase(it);
                                    break;
                                }
                            }
                            if (!m_Pending.Remove(value.m_PacketId) && !m_Publications.Sent(value.m_PacketId))
                            {
                                throw ProtocolError("Unknown publication acknowledgement", MMQP_REASON_INVALID_PID);
                            }
                            if (auto result = m_Results.find(value.m_PacketId); result != m_Results.end())
                            {
                                result->second->set_value(value.m_Reason);
                                m_Results.erase(result);
                            }
                            m_Owner.TryPost<eMQ::Delivery>(value.m_Reason.m_Text.size(), value.m_PacketId,
                                                           value.m_Reason);
                        }
                        else if constexpr (std::is_same_v<Type, ProfileSync>)
                        {
                            if (value.m_Reason.m_Code >= 0x80)
                            {
                                throw ProtocolError("Broker rejected alias mapping", value.m_Reason.m_Code);
                            }
                            m_Profiles.m_Publishers.Confirm(value.m_Alias, value.m_Topic);
                        }
                        else if constexpr (std::is_same_v<Type, ConnectionSync>)
                        {
                            // Tokens belong to the authentication challenge; active CONSYN is an empty keepalive.
                            if (value.m_Token)
                            {
                                throw ProtocolError("Token-bearing CONSYN outside authentication");
                            }
                            this->Write(ConnectionSync{});
                        }
                        else if constexpr (std::is_same_v<Type, ConnectionEnd>)
                        {
                            m_CloseReason = value.m_Reason;
                            m_Connection->Close(true);
                        }
                        else
                        {
                            throw ProtocolError("Packet is invalid in the active client state");
                        }
                    },
                    message);
            }
            catch (const ProtocolError &error)
            {
                this->Fail(error);
            }
            catch (const std::exception &)
            {
                this->Fail(ProtocolError("Client processing failed", MMQP_REASON_INTERNAL_ERROR));
            }
        }

        void Tick()
        {
            if (m_Stopping)
            {
                return;
            }
            auto now = Clock::now();
            if (m_Phase == Phase::Offline && m_Options.m_AutoReconnect && !m_ManualDisconnect && m_Resumable &&
                !m_Identifier.empty() && now >= m_ReconnectAt)
            {
                const auto &endpoint =
                    m_Options.m_FailoverEndpoints[m_ReconnectIndex++ % m_Options.m_FailoverEndpoints.size()];
                try
                {
                    this->Begin(endpoint.m_Host, endpoint.m_Port, true, std::make_shared<std::promise<std::string>>());
                }
                catch (const std::logic_error &)
                {
                    m_Resumable = false;
                    this->Closed("Session cannot safely resume after the outage");
                }
                catch (...)
                {
                    m_ReconnectAt = now + std::chrono::milliseconds(m_Options.m_ReconnectInterval);
                }
            }
            if (m_Phase != Phase::Offline && m_Phase != Phase::Active && now >= m_Deadline)
            {
                this->Fail(ProtocolError("Connection handshake timed out", MMQP_REASON_TIMEOUT));
            }
            if (m_Phase == Phase::Active)
            {
                try
                {
                    for (auto it = m_Requests.begin(); it != m_Requests.end();)
                    {
                        if (it->second.m_Expiry > now)
                        {
                            ++it;
                            continue;
                        }
                        PublishRequest response;
                        response.m_Type = MMQP_PUBREQ_TYPE_RES;
                        response.m_Topic = it->second.m_Topic;
                        response.m_CorrelationId = it->first;
                        response.m_Reason = {MMQP_REASON_TIMEOUT, "Local response deadline expired"};
                        if (!m_Owner.TryPost<eMQ::Receive>(response.m_Topic.size(), std::move(response), false))
                        {
                            throw ProtocolError("Application receive queue is full", MMQP_REASON_BUSY);
                        }
                        m_RequestIds.Release(it->first);
                        it = m_Requests.erase(it);
                    }
                    // The broker pings after one keepalive of client silence; allow one more for its arrival.
                    if (now - std::max(m_LastReceive, m_LastSend) >= 2 * m_Keepalive)
                    {
                        throw ProtocolError("Broker silent for twice the keepalive interval",
                                            MMQP_REASON_KEEPALIVE_TIMEOUT);
                    }
                    if (!m_ProfileTasks.empty())
                    {
                        auto &task = m_ProfileTasks.front();
                        if (task.m_Deadline <= now)
                        {
                            throw ProtocolError("Profile acknowledgement timed out", MMQP_REASON_TIMEOUT);
                        }
                        if (task.m_Retry <= now)
                        {
                            if (++task.m_Attempts > m_Options.m_MaxRetries || !this->Write(task.m_Request))
                            {
                                throw ProtocolError("Profile retry budget exhausted", MMQP_REASON_TIMEOUT);
                            }
                            task.m_Retry = now + std::chrono::milliseconds(m_Options.m_RetryInterval);
                        }
                    }
                    for (auto &[id, publication] : m_Pending.m_Entries)
                    {
                        if (publication.m_Expiry <= now)
                        {
                            throw ProtocolError("Publication expired before acknowledgement", MMQP_REASON_TIMEOUT);
                        }
                        if (publication.m_Retry > now)
                        {
                            continue;
                        }
                        if (++publication.m_Attempts > m_Options.m_MaxRetries)
                        {
                            throw ProtocolError("Publication retry budget exhausted", MMQP_REASON_TIMEOUT);
                        }
                        publication.m_Message.m_Duplicate = true;
                        if (!this->Write(publication.m_Message))
                        {
                            throw ProtocolError("Publication retry send budget exhausted", MMQP_REASON_BUSY);
                        }
                        publication.m_Retry = now + std::chrono::milliseconds(m_Options.m_RetryInterval);
                    }
                }
                catch (const ProtocolError &error)
                {
                    this->Fail(error);
                }
            }
            m_Timer.Schedule(std::chrono::milliseconds(50), [this] { this->Tick(); });
        }
    };

    Service::Service(Configs options) : m_Impl(std::make_unique<Impl>(*this, std::move(options)))
    {
        m_Impl->m_Io.Post([this] { m_Impl->Tick(); });
        m_Impl->m_Io.Init();
    }

    Service::Service(const char *filename) : Service(LoadConfigs(filename))
    {
    }
    Service::~Service()
    {
        m_Workers.Stop();
        this->Stop();
    }

    std::future<std::string> Service::AsyncConnect(std::string host, uint16_t port, bool resume)
    {
        auto result = std::make_shared<std::promise<std::string>>();
        auto future = result->get_future();
        m_Impl->m_Io.Post(
            [this, host = std::move(host), port, resume, result]
            {
                try
                {
                    m_Impl->Begin(host, port, resume, result);
                }
                catch (...)
                {
                    result->set_exception(std::current_exception());
                }
            });
        return future;
    }

    std::string Service::Connect(std::string host, uint16_t port, bool resume)
    {
        return this->AsyncConnect(std::move(host), port, resume).get();
    }

    void Service::Disconnect(bool graceful)
    {
        if (m_Impl->m_Stopping)
        {
            return;
        }
        auto result = std::make_shared<std::promise<void>>();
        auto future = result->get_future();
        m_Impl->m_Io.Post(
            [this, graceful, result]
            {
                try
                {
                    if (m_Impl->m_Disconnecting)
                    {
                        throw std::logic_error("Disconnect is already pending");
                    }
                    m_Impl->m_CloseReason = {};
                    m_Impl->m_ManualDisconnect = true;
                    if (m_Impl->m_Connection)
                    {
                        m_Impl->m_Disconnecting = result;
                        if (graceful && m_Impl->m_Phase == Impl::Phase::Active)
                        {
                            m_Impl->Write(ConnectionEnd{});
                        }
                        m_Impl->m_Connection->Close(graceful);
                    }
                    else
                    {
                        result->set_value();
                    }
                }
                catch (...)
                {
                    result->set_exception(std::current_exception());
                }
            });
        future.get();
    }

    void Service::Stop()
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
        this->Terminate();
    }

    bool Service::Connected() const
    {
        return m_Impl->m_Active.load();
    }
    std::string Service::Identifier()
    {
        if (m_Impl->m_Stopping)
        {
            return m_Impl->m_Identifier;
        }
        return m_Impl->m_Io.Submit([this] { return m_Impl->m_Identifier; }).get();
    }

    std::future<Reason> Service::UpdateProfile(std::vector<ProfileEntry> entries)
    {
        auto result = std::make_shared<std::promise<Reason>>();
        auto future = result->get_future();
        size_t bytes = entries.capacity() * sizeof(ProfileEntry);
        try
        {
            if (entries.size() > m_Impl->m_Options.m_MaxProfiles)
            {
                throw ProtocolError("Profile request exceeds its entry budget", MMQP_REASON_BUSY);
            }
            for (const auto &entry : entries)
            {
                if (entry.m_Topic.size() > UINT8_MAX || entry.m_Topic.capacity() > SIZE_MAX - bytes)
                {
                    throw ProtocolError("Profile topic exceeds its byte budget", MMQP_REASON_BUSY);
                }
                bytes += entry.m_Topic.capacity();
            }
            m_Impl->m_Io.Post(
                [this, entries = std::move(entries), result]() mutable
                {
                    try
                    {
                        if (m_Impl->m_Phase != Impl::Phase::Active)
                        {
                            throw std::logic_error("Client is not connected");
                        }
                        if (m_Impl->m_ProfileTasks.size() >= m_Impl->m_Options.m_MaxInflight)
                        {
                            throw ProtocolError("Profile request budget exhausted", MMQP_REASON_BUSY);
                        }
                        m_Impl->m_ProfileTasks.push_back(
                            {ProfileRequest{0, std::move(entries)}, {}, result, {}, {}, 0});
                        m_Impl->PumpProfile();
                    }
                    catch (...)
                    {
                        result->set_exception(std::current_exception());
                    }
                },
                bytes);
        }
        catch (...)
        {
            result->set_exception(std::current_exception());
        }
        return future;
    }

    std::future<Reason> Service::Register(std::string topic, uint8_t retain, uint32_t expiry, uint8_t queue,
                                          bool requests)
    {
        return this->UpdateProfile(
            {ProfileEntry{MMQP_PROREQ_TYPE_REG, 0, std::move(topic), retain, requests, EncodeInterval(expiry), queue}});
    }

    std::future<Reason> Service::Subscribe(std::string topic, uint8_t qos, bool responder)
    {
        return this->UpdateProfile({ProfileEntry{MMQP_PROREQ_TYPE_SUB, 0, std::move(topic), qos, responder}});
    }

    std::future<Reason> Service::Unsubscribe(std::string topic)
    {
        return this->UpdateProfile({ProfileEntry{MMQP_PROREQ_TYPE_UNS, 0, std::move(topic)}});
    }

    std::future<Reason> Service::Send(PublishRequest publication)
    {
        auto result = std::make_shared<std::promise<Reason>>();
        auto future = result->get_future();
        const auto bytes = publication.m_Payload.Size() + publication.m_Topic.size() + sizeof(PublishRequest) +
                           publication.m_Reason.m_Text.size();
        try
        {
            m_Impl->m_Io.Post(
                [this, publication = std::move(publication), result]() mutable
                {
                    try
                    {
                        m_Impl->Publish(std::move(publication), result);
                    }
                    catch (...)
                    {
                        result->set_exception(std::current_exception());
                    }
                },
                bytes);
        }
        catch (...)
        {
            result->set_exception(std::current_exception());
        }
        return future;
    }

    std::future<Reason> Service::Publish(std::string topic, eCore::SharedBytes payload, uint8_t qos)
    {
        PublishRequest publication;
        publication.m_Topic = std::move(topic);
        publication.m_Payload = std::move(payload);
        publication.m_Qos = qos;
        return this->Send(std::move(publication));
    }

    std::future<Reason> Service::Request(std::string topic, eCore::SharedBytes payload, uint8_t responses, uint8_t qos,
                                         uint16_t correlationId)
    {
        PublishRequest publication;
        publication.m_Type = MMQP_PUBREQ_TYPE_REQ;
        publication.m_Topic = std::move(topic);
        publication.m_Payload = std::move(payload);
        publication.m_Responses = responses;
        publication.m_CorrelationId = correlationId;
        publication.m_Qos = qos;
        return this->Send(std::move(publication));
    }

    std::future<Reason> Service::Respond(std::string topic, uint16_t correlationId, eCore::SharedBytes payload,
                                         Reason reason, uint8_t qos)
    {
        PublishRequest publication;
        publication.m_Type = MMQP_PUBREQ_TYPE_RES;
        publication.m_CorrelationId = correlationId;
        publication.m_Topic = std::move(topic);
        publication.m_Payload = std::move(payload);
        publication.m_Reason = std::move(reason);
        publication.m_Qos = qos;
        return this->Send(std::move(publication));
    }

    std::future<Reason> Service::Respond(const PublishRequest &request, eCore::SharedBytes payload, Reason reason)
    {
        if (request.m_Type != MMQP_PUBREQ_TYPE_REQ || !request.m_CorrelationId)
        {
            throw std::invalid_argument("Respond requires a correlated request");
        }
        return this->Respond(request.m_Topic, request.m_CorrelationId, std::move(payload), std::move(reason),
                             request.m_Qos);
    }
} // namespace eMQ
