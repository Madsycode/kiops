#include <asio.hpp>
#pragma once
#include <Monitoring/Telemetry.h>
#include <Runtime/Service.h>
#include <eCore/Network/Authentication.h>
#include <condition_variable>
#include <sstream>
#include <thread>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    namespace TelemetryDetail
    {
        inline std::shared_ptr<const std::string> Response(std::string_view status, std::string_view body)
        {
            return std::make_shared<const std::string>(
                "HTTP/1.1 " + std::string(status) +
                "\r\nContent-Type: application/json; charset=utf-8\r\nCache-Control: no-store\r\n"
                "X-Content-Type-Options: nosniff\r\nConnection: close\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
        }

        inline std::string Lower(std::string text)
        {
            for (auto &c : text)
            {
                if (c >= 'A' && c <= 'Z')
                {
                    c += 'a' - 'A';
                }
            }
            return text;
        }
    } // namespace TelemetryDetail
    using namespace TelemetryDetail;

    struct TelemetryServer::Impl
    {
        struct Request : std::enable_shared_from_this<Request>
        {
            Impl &m_Owner;
            asio::ip::tcp::socket m_Socket;
            asio::steady_timer m_Deadline;
            std::string m_Input;
            std::shared_ptr<const std::string> m_Response;
            bool m_Closed = false;

            Request(Impl &server, asio::ip::tcp::socket stream)
                : m_Owner(server), m_Socket(std::move(stream)), m_Deadline(m_Socket.get_executor())
            {
            }

            void Start()
            {
                m_Deadline.expires_after(std::chrono::seconds(3));
                m_Deadline.async_wait(
                    [self = shared_from_this()](auto error)
                    {
                        if (!error)
                        {
                            self->Close();
                        }
                    });
                asio::async_read_until(m_Socket, asio::dynamic_buffer(m_Input, 8192), "\r\n\r\n",
                                       [self = shared_from_this()](auto error, size_t size)
                                       {
                                           if (error)
                                           {
                                               self->Close();
                                               return;
                                           }
                                           self->Reply(
                                               self->m_Owner.Handle(std::string_view(self->m_Input).substr(0, size)));
                                       });
            }

            void Reply(std::shared_ptr<const std::string> bytes)
            {
                if (m_Closed)
                {
                    return;
                }
                m_Response = std::move(bytes);
                asio::async_write(m_Socket, asio::buffer(*m_Response),
                                  [self = shared_from_this()](auto, size_t) { self->Close(); });
            }

            void Close()
            {
                if (m_Closed)
                {
                    return;
                }
                m_Closed = true;
                asio::error_code ignored;
                m_Deadline.cancel();
                m_Socket.close(ignored);
                m_Owner.m_Requests.erase(shared_from_this());
            }
        };

        Service &m_Broker;
        TelemetryOptions m_Options;
        asio::io_context m_Io;
        asio::executor_work_guard<asio::io_context::executor_type> m_Work{m_Io.get_executor()};
        asio::ip::tcp::acceptor m_Acceptor{m_Io};
        std::set<std::shared_ptr<Request>> m_Requests;
        std::set<std::string> m_Hosts;
        std::thread m_Worker;
        std::thread m_Sampler;
        std::mutex m_Mutex;
        std::condition_variable m_Wake;
        std::atomic<bool> m_Stopping = false;
        std::shared_ptr<const std::string> m_Cached =
            Response("503 Service Unavailable", "{\"error\":\"Awaiting first snapshot\"}");

        Impl(Service &service, TelemetryOptions settings) : m_Broker(service), m_Options(std::move(settings))
        {
            if (!IsLoopback(m_Options.m_Address))
            {
                throw std::invalid_argument("Telemetry must bind to loopback; use an authenticated TLS proxy or SSH "
                                            "tunnel for remote collection");
            }
            if (m_Options.m_IntervalMs < 250 || m_Options.m_IntervalMs > 10000 || !m_Options.m_MaxSessions ||
                m_Options.m_MaxSessions > 4096)
            {
                throw std::invalid_argument("Invalid telemetry sampling bounds");
            }
            if (!m_Options.m_Token.empty() && (m_Options.m_Token.size() < 32 || m_Options.m_Token.size() > 256 ||
                                               std::any_of(m_Options.m_Token.begin(), m_Options.m_Token.end(),
                                                           [](auto c) { return c < 33 || c > 126; })))
            {
                throw std::invalid_argument(
                    "Telemetry token must contain 32..256 printable non-space ASCII characters");
            }
        }

        ~Impl()
        {
            std::fill(m_Options.m_Token.begin(), m_Options.m_Token.end(), '\0');
        }

        std::shared_ptr<const std::string> Handle(std::string_view input)
        {
            std::istringstream stream{std::string(input)};
            std::string line, method, path, version, extra;
            std::getline(stream, line);
            std::istringstream first(line);
            if (!(first >> method >> path >> version) || (first >> extra) ||
                (version != "HTTP/1.1" && version != "HTTP/1.0"))
            {
                return Response("400 Bad Request", "{\"error\":\"Invalid request\"}");
            }
            std::map<std::string, std::string> headers;
            while (std::getline(stream, line) && line != "\r")
            {
                if (line.empty() || line.back() != '\r')
                {
                    return Response("400 Bad Request", "{}");
                }
                line.pop_back();
                auto colon = line.find(':');
                if (colon == std::string::npos || !colon)
                {
                    return Response("400 Bad Request", "{}");
                }
                auto key = Lower(line.substr(0, colon));
                if (std::any_of(key.begin(), key.end(), [](auto c) { return !(c >= 'a' && c <= 'z') && c != '-'; }))
                {
                    return Response("400 Bad Request", "{}");
                }
                auto begin = line.find_first_not_of(" \t", colon + 1);
                auto end = line.find_last_not_of(" \t");
                auto value = begin == std::string::npos ? std::string{} : line.substr(begin, end - begin + 1);
                if (!headers.emplace(key, value).second)
                {
                    return Response("400 Bad Request", "{}");
                }
            }
            if (!m_Hosts.contains(Lower(headers["host"])))
            {
                return Response("403 Forbidden", "{\"error\":\"Invalid host\"}");
            }
            if (auto origin = headers.find("origin");
                origin != headers.end() &&
                !m_Hosts.contains(Lower(origin->second).starts_with("http://") ? Lower(origin->second).substr(7) : ""))
            {
                return Response("403 Forbidden", "{\"error\":\"Cross-origin requests are disabled\"}");
            }
            if (!m_Options.m_Token.empty())
            {
                auto expected = "Bearer " + m_Options.m_Token;
                const auto &provided = headers["authorization"];
                if (!SecureEqual(std::span(reinterpret_cast<const uint8_t *>(provided.data()), provided.size()),
                                 std::span(reinterpret_cast<const uint8_t *>(expected.data()), expected.size())))
                {
                    return Response("401 Unauthorized", "{\"error\":\"Telemetry token required\"}");
                }
            }
            if (method != "GET")
            {
                return Response("405 Method Not Allowed", "{\"error\":\"Read-only endpoint\"}");
            }
            if (headers.contains("transfer-encoding") ||
                (headers.contains("content-length") && headers["content-length"] != "0"))
            {
                return Response("400 Bad Request", "{}");
            }
            if (path == "/health")
            {
                return Response("200 OK", "{\"status\":\"ok\",\"schema\":1}");
            }
            if (path != "/api/v1/snapshot")
            {
                return Response("404 Not Found", "{\"error\":\"Unknown endpoint\"}");
            }
            std::lock_guard lock(m_Mutex);
            return m_Cached;
        }

        void Accept()
        {
            m_Acceptor.async_accept(
                [this](auto error, auto socket)
                {
                    if (!error && !m_Stopping && m_Requests.size() < 32)
                    {
                        auto request = std::make_shared<Request>(*this, std::move(socket));
                        m_Requests.insert(request);
                        request->Start();
                    }
                    if (!m_Stopping)
                    {
                        this->Accept();
                    }
                });
        }

        void Sample()
        {
            while (!m_Stopping)
            {
                try
                {
                    auto result = Response("200 OK", SnapshotJson(m_Broker.Inspect(m_Options.m_MaxSessions)));
                    std::lock_guard lock(m_Mutex);
                    m_Cached = std::move(result);
                }
                catch (const std::exception &)
                {
                    // Preserve the last observation. Its unchanged uptime lets
                    // collectors detect stale state without exposing internals.
                }
                std::unique_lock lock(m_Mutex);
                m_Wake.wait_for(lock, std::chrono::milliseconds(m_Options.m_IntervalMs),
                                [this] { return m_Stopping.load(); });
            }
        }
    };

    inline TelemetryServer::TelemetryServer(Service &broker, TelemetryOptions options)
        : m_Impl(std::make_unique<Impl>(broker, std::move(options)))
    {
    }
    inline TelemetryServer::~TelemetryServer()
    {
        this->Stop();
    }

    inline uint16_t TelemetryServer::Start()
    {
        if (m_Impl->m_Worker.joinable() || m_Impl->m_Stopping)
        {
            throw std::logic_error("Telemetry already started or stopped");
        }
        asio::ip::tcp::endpoint endpoint(asio::ip::make_address(m_Impl->m_Options.m_Address), m_Impl->m_Options.m_Port);
        m_Impl->m_Acceptor.open(endpoint.protocol());
        m_Impl->m_Acceptor.bind(endpoint);
        m_Impl->m_Acceptor.listen(32);
        auto port = m_Impl->m_Acceptor.local_endpoint().port();
        auto suffix = ':' + std::to_string(port);
        auto address =
            endpoint.address().is_v6() ? '[' + m_Impl->m_Options.m_Address + ']' : m_Impl->m_Options.m_Address;
        m_Impl->m_Hosts = {Lower(address) + suffix, "localhost" + suffix};
        m_Impl->Accept();
        m_Impl->m_Worker = std::thread([this] { m_Impl->m_Io.run(); });
        m_Impl->m_Sampler = std::thread([this] { m_Impl->Sample(); });
        return port;
    }

    inline void TelemetryServer::Stop()
    {
        if (m_Impl->m_Stopping.exchange(true))
        {
            return;
        }
        m_Impl->m_Wake.notify_all();
        if (m_Impl->m_Sampler.joinable())
        {
            m_Impl->m_Sampler.join();
        }
        if (m_Impl->m_Worker.joinable())
        {
            asio::post(m_Impl->m_Io,
                       [this]
                       {
                           asio::error_code ignored;
                           m_Impl->m_Acceptor.close(ignored);
                           auto requests = m_Impl->m_Requests;
                           for (const auto &request : requests)
                           {
                               request->Close();
                           }
                       });
            m_Impl->m_Work.reset();
            m_Impl->m_Worker.join();
        }
    }
} // namespace eMQ::Server
