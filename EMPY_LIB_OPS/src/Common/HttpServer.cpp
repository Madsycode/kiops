#include <eOps/Common/HttpServer.h>
#include "HttpWire.h"
#include <atomic>
#include <mutex>
#include <thread>

namespace eOps
{
    namespace
    {
        using namespace Detail;
        using asio::ip::tcp;

        constexpr auto READ_TIMEOUT = std::chrono::seconds(5);

        std::string Serialise(const HttpResponse &response, bool head)
        {
            int32_t status = response.m_Status >= 100 && response.m_Status <= 999 ? response.m_Status : 500;
            std::string text = "HTTP/1.1 " + std::to_string(status) + " " + ReasonPhrase(status) + "\r\n";
            bool typed = false;
            for (const auto &[name, value] : response.m_Headers)
            {
                if (!IsToken(name) || !SafeValue(value) || SameText(name, "Content-Length") ||
                    SameText(name, "Connection") || SameText(name, "Transfer-Encoding"))
                {
                    continue;
                }
                typed = typed || SameText(name, "Content-Type");
                text += name + ": " + value + "\r\n";
            }
            if (!typed && !response.m_Body.empty())
            {
                text += "Content-Type: text/plain; charset=utf-8\r\n";
            }
            text += "Content-Length: " + std::to_string(response.m_Body.size()) + "\r\nConnection: close\r\n\r\n";
            if (!head)
            {
                text += response.m_Body;
            }
            return text;
        }

        HttpResponse Plain(int32_t status, std::string body)
        {
            HttpResponse response;
            response.m_Status = status;
            response.m_Body = std::move(body);
            return response;
        }
    } // namespace

    std::string HttpServerRequest::Header(std::string_view name) const
    {
        return FindField(m_Headers, name);
    }

    struct HttpServer::Impl
    {
        std::string m_Address;
        uint16_t m_Port = 0;
        HttpHandler m_Handler;
        size_t m_MaxRequestBytes = 1 << 20;
        asio::io_context m_Io;
        tcp::acceptor m_Acceptor{m_Io};
        tcp::socket *m_Active = nullptr; // server thread only
        std::thread m_Thread;
        std::mutex m_Mutex;
        std::atomic<bool> m_Stopping{false};
        std::atomic<uint16_t> m_Bound{0};
        bool m_Used = false;

        void Run()
        {
            while (!m_Stopping.load())
            {
                tcp::socket socket(m_Io);
                std::error_code code;
                bool done = false;
                m_Acceptor.async_accept(socket,
                                        [&](std::error_code error)
                                        {
                                            code = error;
                                            done = true;
                                        });
                m_Io.restart();
                while (!done && m_Io.run_one() > 0)
                {
                }
                if (!done || m_Stopping.load() || code == asio::error::operation_aborted)
                {
                    break;
                }
                if (code)
                {
                    // transient accept failure (descriptor exhaustion): back off briefly
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
                m_Active = &socket;
                Serve(socket);
                m_Active = nullptr;
            }
            // complete anything Stop posted
            m_Io.restart();
            m_Io.poll();
        }

        void Serve(tcp::socket &socket)
        {
            auto cancel = [&socket]
            {
                std::error_code ignored;
                socket.close(ignored);
            };
            auto deadline = Clock::now() + READ_TIMEOUT;
            HttpResponse response;
            bool head = false;
            try
            {
                Reader<tcp::socket> reader{m_Io, socket, deadline, cancel};
                std::string text = reader.Head(std::min<size_t>(m_MaxRequestBytes, 64 * 1024));
                HttpServerRequest request;
                std::string start = ParseHead(text, request.m_Headers);
                // "METHOD SP target SP HTTP/1.x"
                size_t first = start.find(' ');
                size_t second = first == std::string::npos ? std::string::npos : start.find(' ', first + 1);
                if (second == std::string::npos || !start.substr(second + 1).starts_with("HTTP/1."))
                {
                    throw HttpError("malformed HTTP request line");
                }
                request.m_Method = start.substr(0, first);
                request.m_Target = start.substr(first + 1, second - first - 1);
                if (!IsToken(request.m_Method) || request.m_Target.empty())
                {
                    throw HttpError("malformed HTTP request line");
                }
                head = SameText(request.m_Method, "HEAD");
                if (SameText(request.Header("Expect"), "100-continue"))
                {
                    static const std::string s_Continue = "HTTP/1.1 100 Continue\r\n\r\n";
                    Await(
                        m_Io, deadline,
                        [&](auto done)
                        {
                            asio::async_write(socket, asio::buffer(s_Continue),
                                              [done](std::error_code error, size_t) { done(error); });
                        },
                        cancel);
                }
                request.m_Body = reader.Body(request.m_Headers, m_MaxRequestBytes - text.size(), false);
                try
                {
                    response = m_Handler(request);
                }
                catch (const std::exception &error)
                {
                    response = Plain(500, std::string("handler failed: ") + error.what());
                }
                catch (...)
                {
                    response = Plain(500, "handler failed");
                }
            }
            catch (const TooLarge &)
            {
                response = Plain(413, "request too large");
            }
            catch (const Expired &)
            {
                return; // the connection was closed by the deadline
            }
            catch (const HttpError &error)
            {
                if (!socket.is_open())
                {
                    return;
                }
                response = Plain(400, error.what());
            }
            if (!socket.is_open())
            {
                return;
            }
            std::string text = Serialise(response, head);
            try
            {
                Await(
                    m_Io, Clock::now() + READ_TIMEOUT,
                    [&](auto done)
                    {
                        asio::async_write(socket, asio::buffer(text),
                                          [done](std::error_code error, size_t) { done(error); });
                    },
                    cancel);
            }
            catch (const HttpError &)
            {
                return;
            }
            std::error_code ignored;
            socket.shutdown(tcp::socket::shutdown_send, ignored);
            // lingering close: unread request bytes must not reset the connection before the peer
            // has read the response (413 replies leave the body unread)
            auto linger = Clock::now() + std::chrono::seconds(1);
            size_t drained = 0;
            try
            {
                while (drained < 4 * 1024 * 1024)
                {
                    char sink[4096];
                    size_t count = 0;
                    std::error_code code = Await(
                        m_Io, linger,
                        [&](auto done)
                        {
                            socket.async_read_some(asio::buffer(sink),
                                                   [&count, done](std::error_code error, size_t bytes)
                                                   {
                                                       count = bytes;
                                                       done(error);
                                                   });
                        },
                        cancel);
                    if (code)
                    {
                        break;
                    }
                    drained += count;
                }
            }
            catch (const HttpError &)
            {
            }
            socket.close(ignored);
        }
    };

    HttpServer::HttpServer(std::string address, uint16_t port, HttpHandler handler, size_t maxRequestBytes)
        : m_Impl(std::make_unique<Impl>())
    {
        if (!handler)
        {
            throw HttpError("HTTP server requires a handler");
        }
        m_Impl->m_Address = std::move(address);
        m_Impl->m_Port = port;
        m_Impl->m_Handler = std::move(handler);
        m_Impl->m_MaxRequestBytes = std::max<size_t>(maxRequestBytes, 1024);
    }

    HttpServer::~HttpServer()
    {
        Stop();
    }

    uint16_t HttpServer::Start()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        if (m_Impl->m_Used)
        {
            throw HttpError("HTTP server was already started");
        }
        std::error_code code;
        auto address = asio::ip::make_address(m_Impl->m_Address, code);
        if (code)
        {
            throw HttpError("invalid HTTP server address '" + m_Impl->m_Address + "'");
        }
        tcp::endpoint endpoint(address, m_Impl->m_Port);
        tcp::acceptor &acceptor = m_Impl->m_Acceptor;
        acceptor.open(endpoint.protocol(), code);
#ifndef _WIN32
        // Windows SO_REUSEADDR would allow port theft; POSIX needs it to rebind after TIME_WAIT
        if (!code)
        {
            acceptor.set_option(tcp::acceptor::reuse_address(true), code);
        }
#endif
        if (!code)
        {
            acceptor.bind(endpoint, code);
        }
        if (!code)
        {
            acceptor.listen(asio::socket_base::max_listen_connections, code);
        }
        if (code)
        {
            std::error_code ignored;
            acceptor.close(ignored);
            throw HttpError("cannot listen on " + m_Impl->m_Address + ":" + std::to_string(m_Impl->m_Port) + ": " +
                            code.message());
        }
        m_Impl->m_Bound = acceptor.local_endpoint().port();
        m_Impl->m_Used = true;
        m_Impl->m_Thread = std::thread([impl = m_Impl.get()] { impl->Run(); });
        return m_Impl->m_Bound;
    }

    void HttpServer::Stop()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        if (!m_Impl->m_Thread.joinable())
        {
            return;
        }
        m_Impl->m_Stopping = true;
        asio::post(m_Impl->m_Io,
                   [impl = m_Impl.get()]
                   {
                       std::error_code ignored;
                       impl->m_Acceptor.close(ignored);
                       if (impl->m_Active)
                       {
                           impl->m_Active->close(ignored);
                       }
                   });
        m_Impl->m_Thread.join();
        std::error_code ignored;
        m_Impl->m_Acceptor.close(ignored);
    }

    uint16_t HttpServer::Port() const noexcept
    {
        return m_Impl->m_Bound;
    }
} // namespace eOps
