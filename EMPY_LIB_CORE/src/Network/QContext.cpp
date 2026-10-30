#include "QContext.h"
#include <eCore/Common/Stdout.h>
#include <eCore/Network/Authentication.h>

#if EMPY_WITH_QUIC
#include <msquic.h>
#include <openssl/pem.h>
#include <openssl/pkcs7.h>
#include <openssl/x509v3.h>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <cstring>
#include <map>
#include <set>

namespace eCore::MMQP
{
    namespace
    {
        void CheckQuic(QUIC_STATUS status, const char *operation)
        {
            if (QUIC_FAILED(status))
            {
                throw std::runtime_error(std::string(operation) + " failed (MsQuic status " +
                                         std::to_string(static_cast<uint32_t>(status)) + ')');
            }
        }

        struct QuicConfiguration
        {
            const QUIC_API_TABLE *m_Api;
            HQUIC m_Handle = nullptr;
            std::unique_ptr<X509_STORE, decltype(&X509_STORE_free)> m_Trust{nullptr, X509_STORE_free};

            explicit QuicConfiguration(const QUIC_API_TABLE *table) : m_Api(table)
            {
            }
            ~QuicConfiguration()
            {
                if (m_Handle)
                {
                    m_Api->ConfigurationClose(m_Handle);
                }
            }

            bool Verify(const QUIC_CONNECTION_EVENT &event, const std::string &name, bool server)
            {
                if (!m_Trust)
                {
                    return QUIC_SUCCEEDED(event.PEER_CERTIFICATE_RECEIVED.DeferredStatus);
                }
                auto leaf = reinterpret_cast<const QUIC_BUFFER *>(event.PEER_CERTIFICATE_RECEIVED.Certificate);
                auto chain = reinterpret_cast<const QUIC_BUFFER *>(event.PEER_CERTIFICATE_RECEIVED.Chain);
                if (!leaf || !leaf->Buffer || !leaf->Length)
                {
                    return false;
                }
                const unsigned char *cursor = leaf->Buffer;
                std::unique_ptr<X509, decltype(&X509_free)> cert(d2i_X509(nullptr, &cursor, leaf->Length), X509_free);
                if (!cert || cursor != leaf->Buffer + leaf->Length)
                {
                    return false;
                }
                std::unique_ptr<PKCS7, decltype(&PKCS7_free)> bundle(nullptr, PKCS7_free);
                STACK_OF(X509) *intermediates = nullptr;
                if (chain && chain->Length)
                {
                    cursor = chain->Buffer;
                    bundle.reset(d2i_PKCS7(nullptr, &cursor, chain->Length));
                    if (!bundle || cursor != chain->Buffer + chain->Length || !PKCS7_type_is_signed(bundle.get()))
                    {
                        return false;
                    }
                    intermediates = bundle->d.sign->cert;
                }
                std::unique_ptr<X509_STORE_CTX, decltype(&X509_STORE_CTX_free)> context(X509_STORE_CTX_new(),
                                                                                        X509_STORE_CTX_free);
                if (!context || X509_STORE_CTX_init(context.get(), m_Trust.get(), cert.get(), intermediates) != 1)
                {
                    return false;
                }
                if (X509_STORE_CTX_set_purpose(context.get(),
                                               server ? X509_PURPOSE_SSL_CLIENT : X509_PURPOSE_SSL_SERVER) != 1)
                {
                    return false;
                }
                if (!server)
                {
                    auto parameters = X509_STORE_CTX_get0_param(context.get());
                    asio::error_code error;
                    asio::ip::make_address(name, error);
                    int success = error ? X509_VERIFY_PARAM_set1_host(parameters, name.c_str(), name.size())
                                        : X509_VERIFY_PARAM_set1_ip_asc(parameters, name.c_str());
                    if (success != 1)
                    {
                        return false;
                    }
                    X509_VERIFY_PARAM_set_hostflags(parameters, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
                }
                return X509_verify_cert(context.get()) == 1;
            }
        };
    } // namespace

    struct QcContext::Impl : std::enable_shared_from_this<Impl>
    {
        struct QuicConnection;
        struct Stream : std::enable_shared_from_this<Stream>
        {
            std::weak_ptr<QuicConnection> m_Connection;
            HQUIC m_Handle = nullptr;
            FrameStream m_Decoder;
            std::optional<QuicChannel> m_Channel;
            bool m_Outgoing = false;
            bool m_Prefixed = false;
            bool m_Fin = false;
            bool m_Sent = false;
            size_t m_Frames = 0;

            explicit Stream(uint32_t limit) : m_Decoder(limit)
            {
            }
        };

        struct Write
        {
            WireFrame m_Frame;
            std::array<QUIC_BUFFER, 3> m_Buffers{};
            uint32_t m_Count = 0;
            uint8_t m_Prefix = 0;
            size_t m_Size = 0;
            std::shared_ptr<QuicConnection> m_Connection;

            Write(WireFrame data, std::shared_ptr<QuicConnection> owner, uint8_t marker)
                : m_Frame(std::move(data)), m_Prefix(marker), m_Size(m_Frame.Size() + (marker ? 1 : 0)),
                  m_Connection(std::move(owner))
            {
                if (m_Prefix)
                {
                    m_Buffers[m_Count++] = {1, &m_Prefix};
                }
                m_Buffers[m_Count++] = {static_cast<uint32_t>(m_Frame.m_Head.Size()), m_Frame.m_Head.Data()};
                if (!m_Frame.m_Tail.Empty())
                {
                    m_Buffers[m_Count++] = {static_cast<uint32_t>(m_Frame.m_Tail.Size()),
                                            const_cast<uint8_t *>(m_Frame.m_Tail.Data())};
                }
            }
        };

        struct QuicConnection : Connection
        {
            std::shared_ptr<Impl> m_Context;
            std::shared_ptr<QuicConfiguration> m_Configuration;
            TransportOptions m_Options;
            HQUIC m_Handle = nullptr;
            asio::steady_timer m_Timeout;
            IoContext::Accept m_Ready;
            std::string m_Name;
            std::string m_Failure = "QUIC connection closed";
            std::map<HQUIC, std::shared_ptr<Stream>> m_Streams;
            std::map<QuicChannel, std::shared_ptr<Stream>> m_Persistent;
            std::set<QuicChannel> m_Incoming;
            std::deque<std::pair<QuicChannel, Buffer>> m_Deferred;
            std::optional<QuicPlan> m_Plan;
            std::atomic_size_t m_Posted = 0;
            size_t m_DeferredBytes = 0;
            size_t m_Queued = 0;
            size_t m_QueuedFrames = 0;
            size_t m_Bulk = 0;
            size_t m_IncomingBulk = 0;
            uint16_t m_DatagramLimit = 0;
            uint32_t m_Limit;
            bool m_Server;
            bool m_Reasons = false;
            bool m_Draining = false;
            bool m_Closing = false;
            bool m_Closed = false;

            QuicConnection(std::shared_ptr<Impl> owner, TransportOptions settings, bool listener)
                : m_Context(std::move(owner)), m_Options(std::move(settings)), m_Timeout(m_Context->m_Executor),
                  m_Limit(m_Options.m_MaxPacket), m_Server(listener)
            {
            }

            std::shared_ptr<QuicConnection> Self()
            {
                return std::static_pointer_cast<QuicConnection>(shared_from_this());
            }

            void Post(std::function<void()> task)
            {
                asio::post(m_Context->m_Executor,
                           [self = Self(), task = std::move(task)]
                           {
                               try
                               {
                                   task();
                               }
                               catch (const std::exception &error)
                               {
                                   self->Fail(error.what());
                               }
                           });
            }

            void Deadline()
            {
                m_Timeout.expires_after(std::chrono::milliseconds(m_Options.m_Timeout));
                m_Timeout.async_wait(
                    [self = Self()](auto error)
                    {
                        if (!error)
                        {
                            self->Fail("QUIC connection timed out");
                        }
                    });
            }

            void Ready()
            {
                if (m_Closed || m_Closing)
                {
                    return;
                }
                m_Timeout.cancel();
                if (!m_Server)
                {
                    m_Persistent[QuicChannel::Control] = this->Open(QuicChannel::Control);
                }
                m_Ready(Self());
            }

            std::shared_ptr<Stream> Open(QuicChannel channel)
            {
                auto stream = std::make_shared<Stream>(m_Limit);
                stream->m_Connection = Self();
                stream->m_Channel = channel;
                stream->m_Outgoing = true;
                CheckQuic(m_Context->m_Api->StreamOpen(m_Handle,
                                                       channel == QuicChannel::Control
                                                           ? QUIC_STREAM_OPEN_FLAG_NONE
                                                           : QUIC_STREAM_OPEN_FLAG_UNIDIRECTIONAL,
                                                       OnStream, stream.get(), &stream->m_Handle),
                          "StreamOpen");
                m_Streams.emplace(stream->m_Handle, stream);
                auto status = m_Context->m_Api->StreamStart(stream->m_Handle, QUIC_STREAM_START_FLAG_NONE);
                if (QUIC_FAILED(status))
                {
                    m_Streams.erase(stream->m_Handle);
                    m_Context->m_Api->StreamClose(stream->m_Handle);
                    stream->m_Handle = nullptr;
                    CheckQuic(status, "StreamStart");
                }
                if (channel == QuicChannel::Bulk)
                {
                    ++m_Bulk;
                }
                return stream;
            }

            void PeerStream(std::shared_ptr<Stream> stream, bool unidirectional)
            {
                if (m_Closed || m_Closing)
                {
                    m_Context->m_Api->StreamShutdown(stream->m_Handle, QUIC_STREAM_SHUTDOWN_FLAG_ABORT, 1);
                    m_Streams.emplace(stream->m_Handle, std::move(stream));
                    return;
                }
                m_Streams.emplace(stream->m_Handle, stream);
                stream->m_Decoder.SetLimit(m_Limit);
                if (!unidirectional)
                {
                    uint64_t id = 0;
                    uint32_t size = sizeof(id);
                    CheckQuic(m_Context->m_Api->GetParam(stream->m_Handle, QUIC_PARAM_STREAM_ID, &size, &id),
                              "Stream ID");
                    if (!m_Server || id != 0 || m_Persistent.contains(QuicChannel::Control))
                    {
                        throw ProtocolError("QUIC requires exactly one client-initiated bidirectional control stream");
                    }
                    stream->m_Channel = QuicChannel::Control;
                    m_Persistent.emplace(QuicChannel::Control, stream);
                }
                if (m_Streams.size() >
                    2 * (static_cast<size_t>(m_Options.m_QuicPolicy.m_Bulk ? m_Options.m_QuicPolicy.m_Bulk->m_Streams
                                                                           : 0) +
                         2) +
                        1)
                {
                    throw ProtocolError("QUIC stream budget exhausted", MMQP_REASON_BUSY);
                }
            }

            size_t Buffered() const
            {
                size_t total = m_DeferredBytes;
                for (const auto &[id, stream] : m_Streams)
                {
                    total += stream->m_Decoder.Buffered();
                }
                return total;
            }

            void Receive(const std::shared_ptr<Stream> &stream, std::span<const uint8_t> bytes)
            {
                if (m_Closed || m_Closing || m_Draining)
                {
                    return;
                }
                if (bytes.size() > m_Options.m_MaxQueue - std::min(m_Options.m_MaxQueue, this->Buffered()))
                {
                    throw ProtocolError("QUIC receive budget exhausted", MMQP_REASON_BUSY);
                }
                if (!stream->m_Channel && !bytes.empty())
                {
                    auto lane = static_cast<QuicChannel>(bytes.front());
                    bytes = bytes.subspan(1);
                    if (lane != QuicChannel::BestEffort && lane != QuicChannel::Reliable && lane != QuicChannel::Bulk)
                    {
                        throw ProtocolError("Unknown QUIC stream class");
                    }
                    if (lane == QuicChannel::Bulk)
                    {
                        const auto &policy = m_Plan ? m_Plan->Policy() : m_Options.m_QuicPolicy;
                        size_t maximum = policy.m_Bulk ? policy.m_Bulk->m_Streams : 0;
                        if (++m_IncomingBulk > maximum)
                        {
                            throw ProtocolError("QUIC bulk stream budget exhausted");
                        }
                    }
                    else if (!m_Incoming.insert(lane).second)
                    {
                        throw ProtocolError("Duplicate persistent QUIC stream");
                    }
                    stream->m_Channel = lane;
                }
                stream->m_Decoder.Feed(bytes,
                                       [&](auto frame)
                                       {
                                           if (!stream->m_Channel)
                                           {
                                               throw ProtocolError("QUIC stream is missing its class byte");
                                           }
                                           if (*stream->m_Channel == QuicChannel::Bulk && stream->m_Frames != 0)
                                           {
                                               throw ProtocolError("A bulk stream carries exactly one publication");
                                           }
                                           ++stream->m_Frames;
                                           this->Frame(*stream->m_Channel, frame);
                                       });
                this->Flush();
            }

            void Flush()
            {
                if (!m_Closed && m_OnFlush)
                {
                    m_OnFlush();
                }
            }

            void Frame(QuicChannel channel, std::span<const uint8_t> bytes)
            {
                if (m_Closed || m_Closing || m_Draining)
                {
                    return;
                }
                if (!m_Plan && channel != QuicChannel::Control)
                {
                    if (bytes.size() > m_Options.m_MaxQueue - std::min(m_Options.m_MaxQueue, this->Buffered()))
                    {
                        throw ProtocolError("Pre-negotiation QUIC receive budget exhausted");
                    }
                    m_Deferred.emplace_back(channel, Buffer(bytes.begin(), bytes.end()));
                    m_DeferredBytes += bytes.size();
                    return;
                }
                if (m_Plan)
                {
                    // The consumer decodes and validates the body; routing needs only the headers.
                    auto packet = Codec::Peek(bytes, {m_Reasons, m_Limit, UINT32_MAX});
                    if (m_Options.m_Backplane && packet.Publication() && packet.m_Topic == "$mhf/control")
                    {
                        if (channel != QuicChannel::Control)
                        {
                            throw ProtocolError("Backplane control record on data stream");
                        }
                    }
                    else
                    {
                        m_Plan->Validate(channel, packet, bytes.size());
                    }
                }
                if (m_OnReceive)
                {
                    m_OnReceive(bytes);
                }
            }

            bool Send(WireFrame bytes) override
            {
                if (m_Closed || m_Closing || m_Draining || m_QueuedFrames >= m_Options.m_MaxQueueFrames ||
                    m_Queued >= m_Options.m_MaxQueue || bytes.Size() >= m_Options.m_MaxQueue - m_Queued)
                {
                    return false;
                }
                uint8_t prefix = 0;
                QuicChannel channel = QuicChannel::Control;
                if (m_Plan)
                {
                    auto packet = Codec::PeekFrame(bytes, {m_Reasons, m_Limit, UINT32_MAX});
                    if (!(m_Options.m_Backplane && packet.Publication() && packet.m_Topic == "$mhf/control"))
                    {
                        channel = m_Plan->Select(packet, bytes.Size(), m_DatagramLimit, m_Bulk);
                    }
                }
                std::shared_ptr<Stream> stream;
                if (channel != QuicChannel::Datagram)
                {
                    if (channel == QuicChannel::Bulk)
                    {
                        stream = this->Open(channel);
                    }
                    else
                    {
                        if (!m_Persistent.contains(channel))
                        {
                            if (channel == QuicChannel::Control)
                            {
                                return false;
                            }
                            m_Persistent[channel] = this->Open(channel);
                        }
                        stream = m_Persistent.at(channel);
                    }
                    if (!stream->m_Handle)
                    {
                        return false;
                    }
                    if (channel != QuicChannel::Control && !stream->m_Prefixed)
                    {
                        prefix = static_cast<uint8_t>(channel);
                        stream->m_Prefixed = true;
                    }
                }
                if (bytes.Size() > m_Options.m_MaxQueue - m_Queued)
                {
                    return false;
                }
                auto write = std::make_unique<Write>(std::move(bytes), Self(), prefix);
                auto size = write->m_Size;
                m_Queued += size;
                ++m_QueuedFrames;
                QUIC_STATUS status;
                if (channel == QuicChannel::Datagram)
                {
                    status = m_Context->m_Api->DatagramSend(m_Handle, write->m_Buffers.data(), write->m_Count,
                                                            QUIC_SEND_FLAG_NONE, write.get());
                }
                else
                {
                    status = m_Context->m_Api->StreamSend(
                        stream->m_Handle, write->m_Buffers.data(), write->m_Count,
                        channel == QuicChannel::Bulk ? QUIC_SEND_FLAG_FIN : QUIC_SEND_FLAG_NONE, write.get());
                    if (channel == QuicChannel::Bulk)
                    {
                        stream->m_Fin = true;
                    }
                }
                if (QUIC_FAILED(status))
                {
                    m_Queued -= size;
                    --m_QueuedFrames;
                    this->Fail("QUIC send failed");
                    return false;
                }
                write.release(); // Released only by SEND_COMPLETE or the final datagram event.
                return true;
            }

            void Sent(std::unique_ptr<Write> write)
            {
                m_Queued -= write->m_Size;
                --m_QueuedFrames;
            }

            void PeerFinished(const std::shared_ptr<Stream> &stream)
            {
                if (m_Closed || m_Closing || m_Draining)
                {
                    return;
                }
                if (stream->m_Channel != QuicChannel::Bulk)
                {
                    this->Fail("Peer closed a persistent QUIC stream");
                    return;
                }
                if (stream->m_Frames != 1 || stream->m_Decoder.Buffered())
                {
                    throw ProtocolError("Truncated QUIC bulk publication");
                }
            }

            void StreamClosed(const std::shared_ptr<Stream> &stream)
            {
                if (!stream->m_Handle)
                {
                    return;
                }
                auto native = stream->m_Handle;
                stream->m_Handle = nullptr;
                m_Context->m_Api->StreamClose(native);
                m_Streams.erase(native);
                if (stream->m_Channel == QuicChannel::Bulk)
                {
                    if (stream->m_Outgoing)
                    {
                        --m_Bulk;
                    }
                    else
                    {
                        --m_IncomingBulk;
                    }
                }
            }

            void FinishedSending(const std::shared_ptr<Stream> &stream)
            {
                stream->m_Sent = true;
                if (m_Draining)
                {
                    bool complete = true;
                    for (const auto &[id, item] : m_Streams)
                    {
                        if ((item->m_Outgoing || item->m_Channel == QuicChannel::Control) && !item->m_Sent)
                        {
                            complete = false;
                        }
                    }
                    if (complete)
                    {
                        this->Abort();
                    }
                }
            }

            void Abort()
            {
                if (m_Closing || m_Closed)
                {
                    return;
                }
                m_Closing = true;
                m_Timeout.cancel();
                m_Context->m_Api->ConnectionShutdown(m_Handle, QUIC_CONNECTION_SHUTDOWN_FLAG_NONE, 0);
            }

            void Fail(const std::string &text)
            {
                if (m_Closed || m_Closing)
                {
                    return;
                }
                m_Failure = text;
                this->Abort();
            }

            void Close(bool drain) override
            {
                if (m_Closed || m_Closing || m_Draining)
                {
                    return;
                }
                if (!drain)
                {
                    this->Abort();
                    return;
                }
                m_Draining = true;
                bool waiting = false;
                for (auto &[id, stream] : m_Streams)
                {
                    if ((!stream->m_Outgoing && stream->m_Channel != QuicChannel::Control) || stream->m_Sent)
                    {
                        continue;
                    }
                    waiting = true;
                    if (!stream->m_Fin)
                    {
                        stream->m_Fin = true;
                        m_Context->m_Api->StreamShutdown(stream->m_Handle, QUIC_STREAM_SHUTDOWN_FLAG_GRACEFUL, 0);
                    }
                }
                if (!waiting)
                {
                    this->Abort();
                }
                else
                {
                    this->Deadline();
                }
            }

            void Finish()
            {
                if (m_Closed)
                {
                    return;
                }
                m_Closed = true;
                m_Timeout.cancel();
                auto remaining = m_Streams;
                for (const auto &[id, stream] : remaining)
                {
                    this->StreamClosed(stream);
                }
                m_Context->m_Api->ConnectionClose(m_Handle);
                m_Handle = nullptr;
                if (m_OnClose)
                {
                    try
                    {
                        m_OnClose(m_Failure);
                    }
                    catch (const std::exception &error)
                    {
                        EMPY_ERROR("QUIC close callback failed: {}", error.what());
                    }
                }
                std::lock_guard lock(m_Context->m_Mutex);
                m_Context->m_Connections.erase(this);
                m_Context->m_Complete.notify_all();
            }

            void SetLimit(uint32_t maximum) override
            {
                for (auto &[id, stream] : m_Streams)
                {
                    stream->m_Decoder.SetLimit(maximum);
                }
                m_Limit = maximum;
            }

            void Configure(const QuicPolicy &policy, bool strings) override
            {
                if (m_Plan)
                {
                    throw std::logic_error("QUIC policy is immutable within a transport connection");
                }
                m_Plan.emplace(policy);
                m_Reasons = strings;
                if (m_IncomingBulk > (m_Plan->Policy().m_Bulk ? m_Plan->Policy().m_Bulk->m_Streams : 0u))
                {
                    throw ProtocolError("Peer exceeded the negotiated bulk stream budget");
                }
                for (const auto &[channel, bytes] : m_Deferred)
                {
                    this->Frame(channel, bytes);
                }
                m_Deferred.clear();
                m_DeferredBytes = 0;
            }

            size_t Queued() const override
            {
                return m_Queued;
            }
            size_t Capacity() const override
            {
                return m_Options.m_MaxQueue;
            }
            bool Secure() const override
            {
                return true;
            }
            std::string m_Fingerprint;
            std::string PeerFingerprint() const override
            {
                return m_Fingerprint;
            }
            bool Quic() const override
            {
                return true;
            }
        };

        asio::any_io_executor m_Executor;
        const QUIC_API_TABLE *m_Api = nullptr;
        QUIC_TLS_PROVIDER m_Provider = QUIC_TLS_PROVIDER_SCHANNEL;
        HQUIC m_Registration = nullptr;
        HQUIC m_Listener = nullptr;
        QUIC_BUFFER m_Alpn{4, reinterpret_cast<uint8_t *>(const_cast<char *>("MMQP"))};
        std::shared_ptr<QuicConfiguration> m_ServerConfiguration;
        TransportOptions m_ServerOptions;
        IoContext::Accept m_Accepted;
        size_t m_MaxConnections = 1024;
        std::mutex m_Mutex;
        std::condition_variable m_Complete;
        std::map<QuicConnection *, std::shared_ptr<QuicConnection>> m_Connections;
        bool m_Stopping = false;

        explicit Impl(asio::any_io_executor actor) : m_Executor(std::move(actor))
        {
            CheckQuic(MsQuicOpen2(&m_Api), "MsQuicOpen2");
            QUIC_REGISTRATION_CONFIG config{"Empy/MMQP", QUIC_EXECUTION_PROFILE_LOW_LATENCY};
            auto status = m_Api->RegistrationOpen(&config, &m_Registration);
            if (QUIC_FAILED(status))
            {
                MsQuicClose(m_Api);
                m_Api = nullptr;
                CheckQuic(status, "RegistrationOpen");
            }
            uint32_t size = sizeof(m_Provider);
            auto providerStatus = m_Api->GetParam(nullptr, QUIC_PARAM_GLOBAL_TLS_PROVIDER, &size, &m_Provider);
            if (QUIC_FAILED(providerStatus))
            {
                m_Api->RegistrationClose(m_Registration);
                MsQuicClose(m_Api);
                m_Registration = nullptr;
                m_Api = nullptr;
                CheckQuic(providerStatus, "QUIC TLS provider");
            }
        }

        ~Impl()
        {
            m_ServerConfiguration.reset();
            if (m_Listener)
            {
                m_Api->ListenerClose(m_Listener);
            }
            if (m_Registration)
            {
                m_Api->RegistrationClose(m_Registration);
            }
            if (m_Api)
            {
                MsQuicClose(m_Api);
            }
        }

        std::shared_ptr<QuicConfiguration> Configuration(const TransportOptions &options, bool server)
        {
            if (!options.m_Timeout || options.m_MaxPacket < 64 || options.m_MaxQueue < options.m_MaxPacket ||
                options.m_MaxQueue > UINT32_MAX)
            {
                throw std::invalid_argument("Invalid QUIC packet, queue or timeout bounds");
            }
            auto plan = QuicPlan(options.m_QuicPolicy);
            auto config = std::make_shared<QuicConfiguration>(m_Api);
            QUIC_SETTINGS settings{};
            settings.IsSet.PeerBidiStreamCount = true;
            settings.PeerBidiStreamCount = server ? 1 : 0;
            settings.IsSet.PeerUnidiStreamCount = true;
            settings.PeerUnidiStreamCount =
                static_cast<uint16_t>(2 + (plan.Policy().m_Bulk ? plan.Policy().m_Bulk->m_Streams : 0));
            settings.IsSet.DatagramReceiveEnabled = true;
            settings.DatagramReceiveEnabled = plan.Policy().m_Datagram.has_value();
            settings.IsSet.HandshakeIdleTimeoutMs = true;
            settings.HandshakeIdleTimeoutMs = options.m_Timeout;
            settings.IsSet.IdleTimeoutMs = true;
            settings.IdleTimeoutMs = 0; // MMQP keepalive owns the negotiated idle deadline.
            settings.IsSet.ConnFlowControlWindow = true;
            settings.ConnFlowControlWindow = static_cast<uint32_t>(options.m_MaxQueue);
            CheckQuic(m_Api->ConfigurationOpen(m_Registration, &m_Alpn, 1, &settings, sizeof(settings), nullptr,
                                               &config->m_Handle),
                      "ConfigurationOpen");
            QUIC_CREDENTIAL_CONFIG credential{};
            credential.Type = QUIC_CREDENTIAL_TYPE_NONE;
            credential.Flags = server ? QUIC_CREDENTIAL_FLAG_NONE : QUIC_CREDENTIAL_FLAG_CLIENT;
            if (server && options.m_Mutual)
            {
                credential.Flags |= QUIC_CREDENTIAL_FLAG_REQUIRE_CLIENT_AUTHENTICATION;
            }
            QUIC_CERTIFICATE_FILE file{};
            QUIC_CERTIFICATE_HASH hash{};
            if (server || !options.m_Certificate.empty() || !options.m_PrivateKey.empty() ||
                !options.m_CertificateHash.empty())
            {
#ifdef _WIN32
                if (m_Provider == QUIC_TLS_PROVIDER_SCHANNEL)
                {
                    if (options.m_CertificateHash.empty())
                    {
                        throw std::invalid_argument("Schannel QUIC requires certificate_hash for an existing Windows "
                                                    "credential; use the MsQuic OpenSSL build for PEM files");
                    }
                    auto bytes = FromHex(options.m_CertificateHash);
                    if (bytes.size() != sizeof(hash.ShaHash))
                    {
                        throw std::invalid_argument("certificate_hash requires exactly 40 hexadecimal digits");
                    }
                    std::copy(bytes.begin(), bytes.end(), hash.ShaHash);
                    credential.Type = QUIC_CREDENTIAL_TYPE_CERTIFICATE_HASH;
                    credential.CertificateHash = &hash;
                }
                else
#endif
                {
                    if (options.m_Certificate.empty() || options.m_PrivateKey.empty() ||
                        !options.m_CertificateHash.empty())
                    {
                        throw std::invalid_argument("OpenSSL QUIC requires a PEM certificate and private key");
                    }
                    file.CertificateFile = options.m_Certificate.c_str();
                    file.PrivateKeyFile = options.m_PrivateKey.c_str();
                    credential.Type = QUIC_CREDENTIAL_TYPE_CERTIFICATE_FILE;
                    credential.CertificateFile = &file;
                }
            }
            if (m_Provider == QUIC_TLS_PROVIDER_OPENSSL)
            {
                credential.Flags |= QUIC_CREDENTIAL_FLAG_USE_TLS_BUILTIN_CERTIFICATE_VALIDATION;
                if (!options.m_Authority.empty())
                {
                    credential.Flags |= QUIC_CREDENTIAL_FLAG_SET_CA_CERTIFICATE_FILE;
                    credential.CaCertificateFile = options.m_Authority.c_str();
                }
            }
            else if (!options.m_Authority.empty() && (!server || options.m_Mutual))
            {
                config->m_Trust.reset(X509_STORE_new());
                if (!config->m_Trust || X509_STORE_load_file(config->m_Trust.get(), options.m_Authority.c_str()) != 1)
                {
                    throw std::runtime_error("Cannot load the QUIC certificate authority");
                }
                credential.Flags |= QUIC_CREDENTIAL_FLAG_INDICATE_CERTIFICATE_RECEIVED |
                                    QUIC_CREDENTIAL_FLAG_DEFER_CERTIFICATE_VALIDATION |
                                    QUIC_CREDENTIAL_FLAG_USE_PORTABLE_CERTIFICATES;
            }
            if (options.m_Backplane)
            {
                credential.Flags |=
                    QUIC_CREDENTIAL_FLAG_INDICATE_CERTIFICATE_RECEIVED | QUIC_CREDENTIAL_FLAG_USE_PORTABLE_CERTIFICATES;
            }
            CheckQuic(m_Api->ConfigurationLoadCredential(config->m_Handle, &credential), "ConfigurationLoadCredential");
            return config;
        }

        static QUIC_STATUS QUIC_API OnListen(HQUIC, void *raw, QUIC_LISTENER_EVENT *event)
        {
            auto self = static_cast<Impl *>(raw);
            if (event->Type != QUIC_LISTENER_EVENT_NEW_CONNECTION)
            {
                return QUIC_STATUS_SUCCESS;
            }
            try
            {
                std::lock_guard lock(self->m_Mutex);
                if (self->m_Stopping || self->m_Connections.size() >= self->m_MaxConnections)
                {
                    return QUIC_STATUS_CONNECTION_REFUSED;
                }
                auto connection =
                    std::make_shared<QuicConnection>(self->shared_from_this(), self->m_ServerOptions, true);
                connection->m_Configuration = self->m_ServerConfiguration;
                connection->m_Ready = self->m_Accepted;
                connection->m_Handle = event->NEW_CONNECTION.Connection;
                self->m_Connections.emplace(connection.get(), connection);
                self->m_Api->SetCallbackHandler(connection->m_Handle, reinterpret_cast<void *>(OnConnection),
                                                connection.get());
                auto status = self->m_Api->ConnectionSetConfiguration(connection->m_Handle,
                                                                      self->m_ServerConfiguration->m_Handle);
                if (QUIC_FAILED(status))
                {
                    self->m_Connections.erase(connection.get());
                }
                return status;
            }
            catch (...)
            {
                return QUIC_STATUS_INTERNAL_ERROR;
            }
        }

        static QUIC_STATUS QUIC_API OnConnection(HQUIC, void *raw, QUIC_CONNECTION_EVENT *event)
        {
            auto connection = static_cast<QuicConnection *>(raw)->Self();
            try
            {
                switch (event->Type)
                {
                case QUIC_CONNECTION_EVENT_CONNECTED:
                    connection->Post([connection] { connection->Ready(); });
                    break;
                case QUIC_CONNECTION_EVENT_PEER_CERTIFICATE_RECEIVED:
                    if (connection->m_Options.m_Backplane)
                    {
                        auto leaf = reinterpret_cast<const QUIC_BUFFER *>(event->PEER_CERTIFICATE_RECEIVED.Certificate);
                        if (!leaf || !leaf->Buffer || !leaf->Length)
                        {
                            return QUIC_STATUS_BAD_CERTIFICATE;
                        }
                        auto fingerprint = ToHex(Sha256({leaf->Buffer, leaf->Length}));
                        connection->Post([connection, fingerprint] { connection->m_Fingerprint = fingerprint; });
                    }
                    return connection->m_Configuration->Verify(*event, connection->m_Name, connection->m_Server)
                               ? QUIC_STATUS_SUCCESS
                               : QUIC_STATUS_BAD_CERTIFICATE;
                case QUIC_CONNECTION_EVENT_SHUTDOWN_INITIATED_BY_TRANSPORT:
                    connection->Post(
                        [connection, status = event->SHUTDOWN_INITIATED_BY_TRANSPORT.Status]
                        {
                            connection->m_Failure =
                                "QUIC transport closed (status " + std::to_string(static_cast<uint32_t>(status)) + ')';
                        });
                    break;
                case QUIC_CONNECTION_EVENT_SHUTDOWN_COMPLETE:
                    connection->Post([connection] { connection->Finish(); });
                    break;
                case QUIC_CONNECTION_EVENT_PEER_STREAM_STARTED:
                {
                    auto stream = std::make_shared<Stream>(connection->m_Options.m_MaxPacket);
                    stream->m_Handle = event->PEER_STREAM_STARTED.Stream;
                    stream->m_Connection = connection;
                    // Keep the object alive until the actor has installed it, before setting its native callback.
                    connection->Post(
                        [connection, stream, flags = event->PEER_STREAM_STARTED.Flags]
                        { connection->PeerStream(stream, (flags & QUIC_STREAM_OPEN_FLAG_UNIDIRECTIONAL) != 0); });
                    connection->m_Context->m_Api->SetCallbackHandler(stream->m_Handle,
                                                                     reinterpret_cast<void *>(OnStream), stream.get());
                    break;
                }
                case QUIC_CONNECTION_EVENT_DATAGRAM_STATE_CHANGED:
                    connection->Post([connection, enabled = event->DATAGRAM_STATE_CHANGED.SendEnabled,
                                      limit = event->DATAGRAM_STATE_CHANGED.MaxSendLength]
                                     { connection->m_DatagramLimit = enabled ? limit : 0; });
                    break;
                case QUIC_CONNECTION_EVENT_DATAGRAM_RECEIVED:
                {
                    auto &datagram = *event->DATAGRAM_RECEIVED.Buffer;
                    if (event->DATAGRAM_RECEIVED.Flags & QUIC_RECEIVE_FLAG_0_RTT)
                    {
                        return QUIC_STATUS_ABORTED;
                    }
                    if (connection->m_Posted.fetch_add(datagram.Length + 128) + datagram.Length + 128 >
                        connection->m_Options.m_MaxQueue)
                    {
                        connection->m_Posted.fetch_sub(datagram.Length + 128);
                        break; // QoS 0 overload is a drop.
                    }
                    Buffer bytes(datagram.Buffer, datagram.Buffer + datagram.Length);
                    connection->Post(
                        [connection, bytes = std::move(bytes)]
                        {
                            connection->m_Posted.fetch_sub(bytes.size() + 128);
                            connection->Frame(QuicChannel::Datagram, bytes);
                            connection->Flush();
                        });
                    break;
                }
                case QUIC_CONNECTION_EVENT_DATAGRAM_SEND_STATE_CHANGED:
                    if (QUIC_DATAGRAM_SEND_STATE_IS_FINAL(event->DATAGRAM_SEND_STATE_CHANGED.State))
                    {
                        auto write = static_cast<Write *>(event->DATAGRAM_SEND_STATE_CHANGED.ClientContext);
                        event->DATAGRAM_SEND_STATE_CHANGED.ClientContext = nullptr;
                        connection->Post([connection, write] { connection->Sent(std::unique_ptr<Write>(write)); });
                    }
                    break;
                default:
                    break;
                }
                return QUIC_STATUS_SUCCESS;
            }
            catch (...)
            {
                return QUIC_STATUS_INTERNAL_ERROR;
            }
        }

        static QUIC_STATUS QUIC_API OnStream(HQUIC, void *raw, QUIC_STREAM_EVENT *event)
        {
            auto stream = static_cast<Stream *>(raw)->shared_from_this();
            auto connection = stream->m_Connection.lock();
            if (!connection)
            {
                return QUIC_STATUS_ABORTED;
            }
            try
            {
                switch (event->Type)
                {
                case QUIC_STREAM_EVENT_RECEIVE:
                {
                    if (event->RECEIVE.Flags & QUIC_RECEIVE_FLAG_0_RTT)
                    {
                        return QUIC_STATUS_ABORTED;
                    }
                    auto size = event->RECEIVE.TotalBufferLength;
                    if (size > connection->m_Options.m_MaxQueue ||
                        connection->m_Posted.fetch_add(static_cast<size_t>(size) + 128) + size + 128 >
                            connection->m_Options.m_MaxQueue)
                    {
                        if (size <= connection->m_Options.m_MaxQueue)
                        {
                            connection->m_Posted.fetch_sub(static_cast<size_t>(size) + 128);
                        }
                        connection->Post([connection] { connection->Fail("QUIC callback queue exhausted"); });
                        return QUIC_STATUS_ABORTED;
                    }
                    Buffer bytes;
                    bytes.reserve(static_cast<size_t>(size));
                    for (uint32_t i = 0; i < event->RECEIVE.BufferCount; ++i)
                    {
                        const auto &buffer = event->RECEIVE.Buffers[i];
                        bytes.insert(bytes.end(), buffer.Buffer, buffer.Buffer + buffer.Length);
                    }
                    connection->Post(
                        [connection, stream, bytes = std::move(bytes)]
                        {
                            connection->m_Posted.fetch_sub(bytes.size() + 128);
                            connection->Receive(stream, bytes);
                        });
                    break;
                }
                case QUIC_STREAM_EVENT_SEND_COMPLETE:
                {
                    auto write = static_cast<Write *>(event->SEND_COMPLETE.ClientContext);
                    connection->Post([connection, write] { connection->Sent(std::unique_ptr<Write>(write)); });
                    break;
                }
                case QUIC_STREAM_EVENT_PEER_SEND_SHUTDOWN:
                    connection->Post([connection, stream] { connection->PeerFinished(stream); });
                    break;
                case QUIC_STREAM_EVENT_PEER_SEND_ABORTED:
                case QUIC_STREAM_EVENT_PEER_RECEIVE_ABORTED:
                    connection->Post([connection] { connection->Fail("Peer aborted a QUIC stream"); });
                    break;
                case QUIC_STREAM_EVENT_SEND_SHUTDOWN_COMPLETE:
                    connection->Post([connection, stream] { connection->FinishedSending(stream); });
                    break;
                case QUIC_STREAM_EVENT_SHUTDOWN_COMPLETE:
                    connection->Post([connection, stream] { connection->StreamClosed(stream); });
                    break;
                default:
                    break;
                }
                return QUIC_STATUS_SUCCESS;
            }
            catch (...)
            {
                return QUIC_STATUS_INTERNAL_ERROR;
            }
        }
    };

    QcContext::QcContext(asio::any_io_executor executor) : m_Impl(std::make_shared<Impl>(std::move(executor)))
    {
    }
    QcContext::~QcContext()
    {
        this->Stop();
    }
    bool QcContext::Available()
    {
        return true;
    }

    uint16_t QcContext::Listen(std::string_view host, uint16_t port, TransportOptions options, IoContext::Accept accept,
                               size_t maxConnections)
    {
        if (m_Impl->m_Listener || m_Impl->m_Stopping || !maxConnections)
        {
            throw std::logic_error("QUIC listener already configured or stopped");
        }
        m_Impl->m_ServerConfiguration = m_Impl->Configuration(options, true);
        m_Impl->m_ServerOptions = std::move(options);
        m_Impl->m_MaxConnections = maxConnections;
        m_Impl->m_Accepted = std::move(accept);
        CheckQuic(
            m_Impl->m_Api->ListenerOpen(m_Impl->m_Registration, Impl::OnListen, m_Impl.get(), &m_Impl->m_Listener),
            "ListenerOpen");
        auto endpoint = asio::ip::udp::endpoint(asio::ip::make_address(host), port);
        QUIC_ADDR address{};
        if (endpoint.size() > sizeof(address))
        {
            throw std::runtime_error("Unsupported QUIC address size");
        }
        std::memcpy(&address, endpoint.data(), endpoint.size());
        CheckQuic(m_Impl->m_Api->ListenerStart(m_Impl->m_Listener, &m_Impl->m_Alpn, 1, &address), "ListenerStart");
        uint32_t length = sizeof(address);
        CheckQuic(m_Impl->m_Api->GetParam(m_Impl->m_Listener, QUIC_PARAM_LISTENER_LOCAL_ADDRESS, &length, &address),
                  "Listener address");
        return QuicAddrGetPort(&address);
    }

    Connection::Pointer QcContext::Connect(std::string host, uint16_t port, TransportOptions options,
                                           IoContext::Accept ready, Connection::Closed closed)
    {
        if (m_Impl->m_Stopping)
        {
            throw std::logic_error("QUIC context is stopped");
        }
        auto connection = std::make_shared<Impl::QuicConnection>(m_Impl, options, false);
        connection->m_Configuration = m_Impl->Configuration(options, false);
        connection->m_Name = options.m_ServerName.empty() ? host : options.m_ServerName;
        connection->m_Ready = std::move(ready);
        connection->m_OnClose = std::move(closed);
        CheckQuic(m_Impl->m_Api->ConnectionOpen(m_Impl->m_Registration, Impl::OnConnection, connection.get(),
                                                &connection->m_Handle),
                  "ConnectionOpen");
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            m_Impl->m_Connections.emplace(connection.get(), connection);
        }
        try
        {
            if (connection->m_Name != host)
            {
                asio::ip::udp::resolver resolver(m_Impl->m_Executor);
                auto addresses = resolver.resolve(host, std::to_string(port));
                if (addresses.empty())
                {
                    throw std::runtime_error("No QUIC endpoint resolved");
                }
                auto endpoint = addresses.begin()->endpoint();
                QUIC_ADDR address{};
                std::memcpy(&address, endpoint.data(), endpoint.size());
                CheckQuic(m_Impl->m_Api->SetParam(connection->m_Handle, QUIC_PARAM_CONN_REMOTE_ADDRESS, sizeof(address),
                                                  &address),
                          "QUIC remote address");
            }
            connection->Deadline();
            CheckQuic(m_Impl->m_Api->ConnectionStart(connection->m_Handle, connection->m_Configuration->m_Handle,
                                                     QUIC_ADDRESS_FAMILY_UNSPEC, connection->m_Name.c_str(), port),
                      "ConnectionStart");
        }
        catch (...)
        {
            connection->m_Timeout.cancel();
            m_Impl->m_Api->ConnectionClose(connection->m_Handle);
            connection->m_Handle = nullptr;
            std::lock_guard lock(m_Impl->m_Mutex);
            m_Impl->m_Connections.erase(connection.get());
            throw;
        }
        return connection;
    }

    void QcContext::Stop()
    {
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            if (m_Impl->m_Stopping)
            {
                return;
            }
            m_Impl->m_Stopping = true;
        }
        if (m_Impl->m_Listener)
        {
            m_Impl->m_Api->ListenerClose(m_Impl->m_Listener);
            m_Impl->m_Listener = nullptr;
        }
        m_Impl->m_Api->RegistrationShutdown(m_Impl->m_Registration, QUIC_CONNECTION_SHUTDOWN_FLAG_SILENT, 0);
        std::unique_lock lock(m_Impl->m_Mutex);
        m_Impl->m_Complete.wait(lock, [this] { return m_Impl->m_Connections.empty(); });
    }
} // namespace eCore::MMQP

#else
namespace eCore::MMQP
{
    struct QcContext::Impl
    {
    };
    QcContext::QcContext(asio::any_io_executor) : m_Impl(std::make_shared<Impl>())
    {
    }
    QcContext::~QcContext() = default;
    bool QcContext::Available()
    {
        return false;
    }
    void QcContext::Stop()
    {
    }
    uint16_t QcContext::Listen(std::string_view, uint16_t, TransportOptions, IoContext::Accept, size_t)
    {
        throw std::runtime_error("QUIC requires a build with EMPY_ENABLE_QUIC=ON and MsQuic");
    }
    Connection::Pointer QcContext::Connect(std::string, uint16_t, TransportOptions, IoContext::Accept,
                                           Connection::Closed)
    {
        throw std::runtime_error("QUIC requires a build with EMPY_ENABLE_QUIC=ON and MsQuic");
    }
} // namespace eCore::MMQP
#endif
