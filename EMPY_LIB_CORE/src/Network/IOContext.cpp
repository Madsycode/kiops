#include <eCore/Network/IOContext.h>
#include "QContext.h"
#include <eCore/Common/Stdout.h>
#include <eCore/Network/Authentication.h>
#include <asio/ssl.hpp>
#include <deque>
#include <thread>

namespace eCore::MMQP
{
    namespace
    {
        using TlsContext = asio::ssl::context;

        std::shared_ptr<TlsContext> MakeTls(const TransportOptions &options, bool server)
        {
            auto context = std::make_shared<TlsContext>(server ? TlsContext::tls_server : TlsContext::tls_client);
            if (!options.m_Tls)
            {
                return context;
            }
            if (SSL_CTX_set_min_proto_version(context->native_handle(), TLS1_3_VERSION) != 1)
            {
                throw std::runtime_error("TLS 1.3 is unavailable");
            }
            if (!options.m_Authority.empty())
            {
                context->load_verify_file(options.m_Authority);
            }
            else
            {
                context->set_default_verify_paths();
            }
            if (!options.m_Certificate.empty())
            {
                context->use_certificate_chain_file(options.m_Certificate);
            }
            if (!options.m_PrivateKey.empty())
            {
                context->use_private_key_file(options.m_PrivateKey, TlsContext::pem);
            }
            if (server && options.m_Tls && (options.m_Certificate.empty() || options.m_PrivateKey.empty()))
            {
                throw std::invalid_argument("TLS listeners require a certificate and private key");
            }
            if (!server || options.m_Mutual)
            {
                context->set_verify_mode(asio::ssl::verify_peer |
                                         (options.m_Mutual ? asio::ssl::verify_fail_if_no_peer_cert : 0));
            }
            return context;
        }

        struct TcpConnection : Connection
        {
            std::shared_ptr<TlsContext> m_Context;
            asio::ssl::stream<asio::ip::tcp::socket> m_Stream;
            asio::ip::tcp::resolver m_Resolver;
            asio::steady_timer m_Timeout;
            TransportOptions m_Options;
            FrameStream m_Decoder;
            std::array<uint8_t, 8192> m_Input{};
            std::deque<WireFrame> m_Output;
            std::vector<asio::const_buffer> m_Gather;
            Buffer m_Staging;
            size_t m_Inflight = 0;
            size_t m_Queued = 0;
            bool m_Closed = false;
            bool m_Draining = false;

            TcpConnection(asio::ip::tcp::socket socket, std::shared_ptr<TlsContext> tls, TransportOptions settings)
                : m_Context(std::move(tls)), m_Stream(std::move(socket), *m_Context),
                  m_Resolver(m_Stream.get_executor()), m_Timeout(m_Stream.get_executor()),
                  m_Options(std::move(settings)), m_Decoder(m_Options.m_MaxPacket)
            {
                if (!m_Options.m_Timeout || !m_Options.m_MaxQueueFrames || m_Options.m_MaxPacket < 2 ||
                    m_Options.m_MaxQueue < m_Options.m_MaxPacket)
                {
                    throw std::invalid_argument("Invalid transport queue or timeout bounds");
                }
            }

            std::shared_ptr<TcpConnection> Self()
            {
                return std::static_pointer_cast<TcpConnection>(shared_from_this());
            }

            void Deadline()
            {
                m_Timeout.expires_after(std::chrono::milliseconds(m_Options.m_Timeout));
                m_Timeout.async_wait(
                    [self = Self()](auto error)
                    {
                        if (!error)
                        {
                            self->Fail("Transport connection timed out");
                        }
                    });
            }

            void Dial(std::string host, uint16_t port, IoContext::Accept ready)
            {
                this->Deadline();
                m_Resolver.async_resolve(
                    host, std::to_string(port),
                    [self = Self(), host, ready = std::move(ready)](auto error, auto endpoints) mutable
                    {
                        if (error)
                        {
                            self->Fail(error.message());
                            return;
                        }
                        asio::async_connect(self->m_Stream.next_layer(), endpoints,
                                            [self, host, ready = std::move(ready)](auto failure, auto) mutable
                                            {
                                                if (failure)
                                                {
                                                    self->Fail(failure.message());
                                                    return;
                                                }
                                                self->Handshake(false, host, std::move(ready));
                                            });
                    });
            }

            void Handshake(bool server, const std::string &host, IoContext::Accept ready)
            {
                if (m_Closed)
                {
                    return;
                }
                asio::error_code error;
                m_Stream.next_layer().set_option(asio::ip::tcp::no_delay(true), error);
                if (error)
                {
                    this->Fail(error.message());
                    return;
                }
                if (!m_Options.m_Tls)
                {
                    this->Ready(std::move(ready));
                    return;
                }
                if (!server)
                {
                    auto name = m_Options.m_ServerName.empty() ? host : m_Options.m_ServerName;
                    if (SSL_set_tlsext_host_name(m_Stream.native_handle(), name.c_str()) != 1)
                    {
                        this->Fail("TLS server name configuration failed");
                        return;
                    }
                    m_Stream.set_verify_callback(asio::ssl::host_name_verification(name));
                }
                m_Stream.async_handshake(server ? asio::ssl::stream_base::server : asio::ssl::stream_base::client,
                                         [self = Self(), ready = std::move(ready)](auto failure) mutable
                                         {
                                             if (failure)
                                             {
                                                 self->Fail(failure.message());
                                                 return;
                                             }
                                             self->Ready(std::move(ready));
                                         });
            }

            void Ready(IoContext::Accept ready)
            {
                if (m_Closed)
                {
                    return;
                }
                m_Timeout.cancel();
                try
                {
                    ready(Self());
                }
                catch (const std::exception &error)
                {
                    this->Fail(error.what());
                    return;
                }
                this->Read();
            }

            void Read()
            {
                if (m_Closed || m_Draining)
                {
                    return;
                }
                auto receive = [self = Self()](asio::error_code error, size_t count)
                {
                    if (error)
                    {
                        self->Fail(error.message());
                        return;
                    }
                    try
                    {
                        self->m_Decoder.Feed(std::span(self->m_Input).first(count),
                                             [self](auto frame)
                                             {
                                                 if (!self->m_Closed && !self->m_Draining && self->m_OnReceive)
                                                 {
                                                     self->m_OnReceive(frame);
                                                 }
                                             });
                        if (!self->m_Closed && self->m_OnFlush)
                        {
                            self->m_OnFlush();
                        }
                    }
                    catch (const std::exception &failure)
                    {
                        self->Fail(failure.what());
                        return;
                    }
                    self->Read();
                };
                if (m_Options.m_Tls)
                {
                    m_Stream.async_read_some(asio::buffer(m_Input), std::move(receive));
                }
                else
                {
                    m_Stream.next_layer().async_read_some(asio::buffer(m_Input), std::move(receive));
                }
            }

            bool Send(WireFrame bytes) override
            {
                if (m_Closed || m_Draining || m_Output.size() >= m_Options.m_MaxQueueFrames ||
                    bytes.Size() > m_Options.m_MaxQueue - m_Queued)
                {
                    return false;
                }
                bool idle = m_Output.empty();
                m_Queued += bytes.Size();
                m_Output.emplace_back(std::move(bytes));
                if (idle)
                {
                    this->Write();
                }
                return true;
            }

            void Write()
            {
                constexpr size_t batchLimit = 64;
                constexpr size_t byteLimit = 64 * 1024;
                constexpr size_t recordLimit = 16 * 1024;
                size_t bytes = 0;
                m_Inflight = 0;
                m_Gather.clear();
                for (const auto &item : m_Output)
                {
                    if (m_Inflight == batchLimit ||
                        (m_Inflight && bytes + item.Size() > (m_Options.m_Tls ? recordLimit : byteLimit)))
                    {
                        break;
                    }
                    ++m_Inflight;
                    bytes += item.Size();
                    m_Gather.emplace_back(item.m_Head.Data(), item.m_Head.Size());
                    if (!item.m_Tail.Empty())
                    {
                        m_Gather.emplace_back(item.m_Tail.Data(), item.m_Tail.Size());
                    }
                }
                auto complete = [self = Self()](asio::error_code error, size_t)
                {
                    if (self->m_Closed)
                    {
                        return;
                    }
                    if (error)
                    {
                        self->Fail(error.message());
                        return;
                    }
                    for (; self->m_Inflight; --self->m_Inflight)
                    {
                        self->m_Queued -= self->m_Output.front().Size();
                        self->m_Output.pop_front();
                    }
                    if (!self->m_Output.empty())
                    {
                        self->Write();
                    }
                    else if (self->m_Draining)
                    {
                        self->Fail("Connection closed");
                    }
                };
                // Small TLS packets share a record. Large TCP/TLS publications retain
                // gathered headers and shared tails until asynchronous completion.
                if (m_Options.m_Tls && bytes <= recordLimit)
                {
                    m_Staging.resize(bytes);
                    size_t offset = 0;
                    for (const auto &buffer : m_Gather)
                    {
                        std::memcpy(m_Staging.data() + offset, buffer.data(), buffer.size());
                        offset += buffer.size();
                    }
                    asio::async_write(m_Stream, asio::buffer(m_Staging), std::move(complete));
                }
                else if (m_Options.m_Tls)
                {
                    asio::async_write(m_Stream, m_Gather, std::move(complete));
                }
                else
                {
                    asio::async_write(m_Stream.next_layer(), m_Gather, std::move(complete));
                }
            }

            void Close(bool drain) override
            {
                if (m_Closed)
                {
                    return;
                }
                m_Draining = drain;
                if (!drain || m_Output.empty())
                {
                    this->Fail("Connection closed");
                }
                else
                {
                    m_Timeout.expires_after(std::chrono::milliseconds(m_Options.m_Timeout));
                    m_Timeout.async_wait(
                        [self = Self()](auto error)
                        {
                            if (!error)
                            {
                                self->Fail("Connection drain timed out");
                            }
                        });
                }
            }

            void Fail(const std::string &message)
            {
                if (m_Closed)
                {
                    return;
                }
                m_Closed = true;
                m_Resolver.cancel();
                m_Timeout.cancel();
                asio::error_code ignored;
                m_Stream.next_layer().cancel(ignored);
                m_Stream.next_layer().shutdown(asio::ip::tcp::socket::shutdown_both, ignored);
                m_Stream.next_layer().close(ignored);
                // Pending writes own their buffers until their completion handlers run.
                if (m_OnClose)
                {
                    try
                    {
                        m_OnClose(message);
                    }
                    catch (const std::exception &error)
                    {
                        EMPY_ERROR("Connection close callback failed: {}", error.what());
                    }
                }
            }

            void SetLimit(uint32_t limit) override
            {
                m_Decoder.SetLimit(limit);
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
                return m_Options.m_Tls;
            }
            std::string PeerFingerprint() const override
            {
                if (!m_Options.m_Tls)
                {
                    return {};
                }
                auto ssl = const_cast<TcpConnection *>(this)->m_Stream.native_handle();
                std::unique_ptr<X509, decltype(&X509_free)> certificate(SSL_get1_peer_certificate(ssl), X509_free);
                Digest digest{};
                unsigned int size = 0;
                if (!certificate || X509_digest(certificate.get(), EVP_sha256(), digest.data(), &size) != 1 ||
                    size != digest.size())
                {
                    return {};
                }
                return ToHex(digest);
            }
        };
    } // namespace

    struct IoContext::Impl
    {
        asio::io_context m_Context;
        asio::strand<asio::io_context::executor_type> m_Strand{m_Context.get_executor()};
        std::optional<asio::executor_work_guard<asio::io_context::executor_type>> m_Guard;
        std::optional<asio::ip::tcp::acceptor> m_Acceptor;
        std::unique_ptr<QcContext> m_Quic;
        std::vector<std::thread> m_Workers;
        std::vector<std::weak_ptr<TcpConnection>> m_Connections;
        std::shared_ptr<TlsContext> m_Tls;
        TransportOptions m_Options;
        Accept m_Accepted;
        size_t m_MaxConnections = 1024;
        std::atomic_bool m_Stopped = false;
        std::mutex m_Admission;
        bool m_Accepting = true;
        size_t m_TaskCount = 0;
        size_t m_TaskBytes = 0;
        size_t m_MaxTasks;
        size_t m_MaxBytes;

        Impl(size_t tasks, size_t bytes) : m_MaxTasks(tasks), m_MaxBytes(bytes)
        {
            if (!tasks || !bytes)
            {
                throw std::invalid_argument("I/O admission budgets must be positive");
            }
        }

        void Run(bool poll = false)
        {
            // An escaping callback must not terminate a worker or strand shutdown.
            for (;;)
            {
                try
                {
                    if (poll)
                    {
                        m_Context.poll();
                    }
                    else
                    {
                        m_Context.run();
                    }
                    return;
                }
                catch (const std::exception &failure)
                {
                    EMPY_ERROR("I/O callback failed: {}", failure.what());
                }
                catch (...)
                {
                    EMPY_ERROR("I/O callback failed with an unknown exception");
                }
            }
        }

        void AcceptNext()
        {
            if (!m_Acceptor || !m_Acceptor->is_open())
            {
                return;
            }
            m_Acceptor->async_accept(
                m_Strand,
                [this](auto error, auto socket)
                {
                    if (!error)
                    {
                        std::erase_if(m_Connections, [](const auto &weak) { return weak.expired(); });
                        if (m_Connections.size() < m_MaxConnections)
                        {
                            auto connection = std::make_shared<TcpConnection>(std::move(socket), m_Tls, m_Options);
                            m_Connections.emplace_back(connection);
                            connection->Deadline();
                            connection->Handshake(true, {}, m_Accepted);
                        }
                    }
                    if (error != asio::error::operation_aborted)
                    {
                        this->AcceptNext();
                    }
                });
        }
    };

    IoContext::IoContext(size_t tasks, size_t bytes) : m_Impl(std::make_unique<Impl>(tasks, bytes))
    {
    }
    IoContext::~IoContext()
    {
        this->Stop();
    }

    uint16_t IoContext::Listen(std::string_view host, uint16_t port, TransportOptions options, Accept accept,
                               size_t maxConnections)
    {
        if (!m_Impl->m_Workers.empty() || m_Impl->m_Acceptor || m_Impl->m_Stopped || !maxConnections)
        {
            throw std::logic_error("Configure a listener before starting I/O workers");
        }
        if (options.m_Quic)
        {
            if (!m_Impl->m_Quic)
            {
                m_Impl->m_Quic = std::make_unique<QcContext>(m_Impl->m_Strand);
            }
            return m_Impl->m_Quic->Listen(host, port, std::move(options), std::move(accept), maxConnections);
        }
        auto address = asio::ip::make_address(host);
        m_Impl->m_Tls = MakeTls(options, true);
        m_Impl->m_Options = std::move(options);
        m_Impl->m_Accepted = std::move(accept);
        m_Impl->m_MaxConnections = maxConnections;
        auto &acceptor = m_Impl->m_Acceptor.emplace(m_Impl->m_Strand);
        asio::ip::tcp::endpoint endpoint(address, port);
        acceptor.open(endpoint.protocol());
        acceptor.set_option(asio::socket_base::reuse_address(true));
        if (address.is_v6())
        {
            acceptor.set_option(asio::ip::v6_only(false));
        }
        acceptor.bind(endpoint);
        acceptor.listen(asio::socket_base::max_listen_connections);
        m_Impl->AcceptNext();
        return acceptor.local_endpoint().port();
    }

    Connection::Pointer IoContext::Connect(std::string host, uint16_t port, TransportOptions options, Accept connected,
                                           Connection::Closed closed)
    {
        if (m_Impl->m_Stopped)
        {
            throw std::logic_error("I/O context is stopped");
        }
        if (options.m_Quic)
        {
            if (!m_Impl->m_Quic)
            {
                m_Impl->m_Quic = std::make_unique<QcContext>(m_Impl->m_Strand);
            }
            return m_Impl->m_Quic->Connect(std::move(host), port, std::move(options), std::move(connected),
                                           std::move(closed));
        }
        auto tls = MakeTls(options, false);
        auto connection = std::make_shared<TcpConnection>(asio::ip::tcp::socket(m_Impl->m_Strand), tls, options);
        connection->m_OnClose = std::move(closed);
        std::erase_if(m_Impl->m_Connections, [](const auto &weak) { return weak.expired(); });
        m_Impl->m_Connections.emplace_back(connection);
        connection->Dial(std::move(host), port, std::move(connected));
        return connection;
    }

    bool IoContext::IsCurrent() const noexcept
    {
        return m_Impl->m_Strand.running_in_this_thread();
    }

    struct IoTimer::Impl
    {
        asio::steady_timer m_Timer;
        explicit Impl(asio::any_io_executor executor) : m_Timer(executor)
        {
        }
    };

    IoTimer::IoTimer(IoContext &context) : m_Impl(std::make_unique<Impl>(context.m_Impl->m_Strand))
    {
    }
    IoTimer::~IoTimer() = default;
    void IoTimer::Cancel()
    {
        m_Impl->m_Timer.cancel();
    }
    void IoTimer::Schedule(std::chrono::milliseconds delay, std::function<void()> callback)
    {
        m_Impl->m_Timer.expires_after(delay);
        m_Impl->m_Timer.async_wait(
            [callback = std::move(callback)](asio::error_code error)
            {
                if (!error)
                {
                    callback();
                }
            });
    }

    bool QuicAvailable()
    {
        return QcContext::Available();
    }
    bool IsLoopback(std::string_view address)
    {
        asio::error_code error;
        auto parsed = asio::ip::make_address(address, error);
        return !error && parsed.is_loopback();
    }

    void IoContext::Init(uint32_t workers)
    {
        if (!workers || workers > 256 || !m_Impl->m_Workers.empty() || m_Impl->m_Stopped)
        {
            throw std::logic_error("I/O workers already started or invalid worker count");
        }
        m_Impl->m_Context.restart();
        m_Impl->m_Guard.emplace(m_Impl->m_Context.get_executor());
        try
        {
            for (uint32_t i = 0; i < workers; ++i)
            {
                m_Impl->m_Workers.emplace_back([this] { m_Impl->Run(); });
            }
        }
        catch (...)
        {
            m_Impl->m_Guard.reset();
            m_Impl->m_Context.stop();
            for (auto &worker : m_Impl->m_Workers)
            {
                worker.join();
            }
            m_Impl->m_Workers.clear();
            m_Impl->m_Context.restart();
            throw;
        }
    }

    void IoContext::Post(std::function<void()> task, size_t bytes)
    {
        std::lock_guard lock(m_Impl->m_Admission);
        if (!m_Impl->m_Accepting)
        {
            throw std::logic_error("I/O context is stopped");
        }
        if (m_Impl->m_TaskCount >= m_Impl->m_MaxTasks || bytes > m_Impl->m_MaxBytes - m_Impl->m_TaskBytes)
        {
            throw std::length_error("I/O submission budget exhausted");
        }
        ++m_Impl->m_TaskCount;
        m_Impl->m_TaskBytes += bytes;
        try
        {
            asio::post(m_Impl->m_Strand,
                       [this, task = std::move(task), bytes]
                       {
                           {
                               std::lock_guard lock(m_Impl->m_Admission);
                               --m_Impl->m_TaskCount;
                               m_Impl->m_TaskBytes -= bytes;
                           }
                           task();
                       });
        }
        catch (...)
        {
            --m_Impl->m_TaskCount;
            m_Impl->m_TaskBytes -= bytes;
            throw;
        }
    }

    void IoContext::Stop(std::function<void()> beforeClose)
    {
        if (m_Impl->m_Stopped)
        {
            return;
        }
        for (const auto &worker : m_Impl->m_Workers)
        {
            if (worker.get_id() == std::this_thread::get_id())
            {
                throw std::logic_error("Stop I/O from the owner thread, not an I/O callback");
            }
        }
        {
            std::lock_guard lock(m_Impl->m_Admission);
            m_Impl->m_Accepting = false;
        }
        // Stop control must remain admissible even when application submissions filled the queue.
        auto invoke = [this](std::function<void()> callback)
        {
            if (m_Impl->m_Workers.empty())
            {
                m_Impl->Run(true);
                callback();
                return;
            }
            std::packaged_task<void()> task(std::move(callback));
            auto complete = task.get_future();
            asio::post(m_Impl->m_Strand, [task = std::move(task)]() mutable { task(); });
            complete.get();
        };
        if (beforeClose)
        {
            invoke(std::move(beforeClose));
        }
        if (m_Impl->m_Quic)
        {
            // Native completion callbacks must drain on the actor before its executor stops.
            if (m_Impl->m_Workers.empty())
            {
                this->Init();
            }
            m_Impl->m_Quic->Stop();
        }
        auto close = [this]
        {
            if (m_Impl->m_Acceptor)
            {
                asio::error_code ignored;
                m_Impl->m_Acceptor->close(ignored);
            }
            auto connections = m_Impl->m_Connections;
            for (const auto &weak : connections)
            {
                if (auto connection = weak.lock())
                {
                    connection->Close(false);
                }
            }
        };
        invoke(close);
        m_Impl->m_Stopped = true;
        m_Impl->m_Guard.reset();
        m_Impl->m_Context.stop();
        for (auto &worker : m_Impl->m_Workers)
        {
            worker.join();
        }
        m_Impl->m_Workers.clear();
    }
} // namespace eCore::MMQP
