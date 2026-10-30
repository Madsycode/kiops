#include <eOps/Common/Http.h>
#ifdef _WIN32
#include <wincrypt.h>
#endif
#include "HttpWire.h"
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>
#include <mutex>

namespace eOps
{
    namespace
    {
        using namespace Detail;
        using asio::ip::tcp;

#ifdef _WIN32
        // DER copies of the system ROOT store, read once per process
        const std::vector<std::vector<uint8_t>> &WindowsRoots()
        {
            static std::once_flag s_Once;
            static std::vector<std::vector<uint8_t>> s_Roots;
            std::call_once(s_Once,
                           []
                           {
                               HCERTSTORE store = CertOpenSystemStoreW(0, L"ROOT");
                               if (!store)
                               {
                                   return;
                               }
                               PCCERT_CONTEXT context = nullptr;
                               while ((context = CertEnumCertificatesInStore(store, context)) != nullptr)
                               {
                                   s_Roots.emplace_back(context->pbCertEncoded,
                                                        context->pbCertEncoded + context->cbCertEncoded);
                               }
                               CertCloseStore(store, 0);
                           });
            return s_Roots;
        }

        void ImportWindowsRoots(SSL_CTX *context)
        {
            X509_STORE *store = SSL_CTX_get_cert_store(context);
            for (const auto &der : WindowsRoots())
            {
                const unsigned char *cursor = der.data();
                X509 *certificate = d2i_X509(nullptr, &cursor, static_cast<long>(der.size()));
                if (certificate)
                {
                    // duplicates of default-path anchors are rejected harmlessly
                    X509_STORE_add_cert(store, certificate);
                    X509_free(certificate);
                }
            }
            ERR_clear_error();
        }
#endif

        bool IsAddress(const std::string &host)
        {
            std::error_code code;
            asio::ip::make_address(host, code);
            return !code;
        }

        std::string BuildRequest(const HttpRequest &request, const Url &url)
        {
            if (!IsToken(request.m_Method))
            {
                throw HttpError("invalid HTTP method");
            }
            std::string host = url.m_Host.find(':') != std::string::npos ? "[" + url.m_Host + "]" : url.m_Host;
            bool defaultPort =
                (url.m_Scheme == "http" && url.m_Port == 80) || (url.m_Scheme == "https" && url.m_Port == 443);
            if (!defaultPort)
            {
                host += ":" + std::to_string(url.m_Port);
            }
            std::string text = request.m_Method + " " + url.m_Target + " HTTP/1.1\r\nHost: " + host + "\r\n";
            for (const auto &[name, value] : request.m_Headers)
            {
                if (!IsToken(name) || !SafeValue(value))
                {
                    // never echo the value: headers carry credentials
                    throw HttpError("invalid HTTP header field '" + (IsToken(name) ? name : std::string("?")) + "'");
                }
                if (SameText(name, "Host") || SameText(name, "Content-Length") || SameText(name, "Connection") ||
                    SameText(name, "Transfer-Encoding"))
                {
                    continue;
                }
                text += name + ": " + value + "\r\n";
            }
            bool bodyless = (SameText(request.m_Method, "GET") || SameText(request.m_Method, "HEAD") ||
                             SameText(request.m_Method, "DELETE") || SameText(request.m_Method, "OPTIONS")) &&
                            request.m_Body.empty();
            if (!bodyless)
            {
                text += "Content-Length: " + std::to_string(request.m_Body.size()) + "\r\n";
            }
            text += "Connection: close\r\n\r\n";
            text += request.m_Body;
            return text;
        }

        template <typename _Stream>
        HttpResponse Exchange(asio::io_context &io, _Stream &stream, tcp::socket &socket,
                              const tcp::resolver::results_type &endpoints, SSL *tls, const Url &url,
                              const HttpRequest &request, const std::string &text, Clock::time_point deadline)
        {
            auto cancel = [&socket]
            {
                std::error_code ignored;
                socket.close(ignored);
            };
            std::error_code code = Await(
                io, deadline,
                [&](auto done)
                {
                    asio::async_connect(socket, endpoints,
                                        [done](std::error_code error, const tcp::endpoint &) { done(error); });
                },
                cancel);
            if (code)
            {
                throw HttpError("cannot connect to " + url.m_Host + ":" + std::to_string(url.m_Port) + ": " +
                                code.message());
            }
            if constexpr (!std::is_same_v<_Stream, tcp::socket>)
            {
                code = Await(
                    io, deadline,
                    [&](auto done)
                    {
                        stream.async_handshake(asio::ssl::stream_base::client,
                                               [done](std::error_code error) { done(error); });
                    },
                    cancel);
                if (code)
                {
                    long verify = SSL_get_verify_result(tls);
                    if (request.m_VerifyPeer && verify != X509_V_OK)
                    {
                        throw HttpError("TLS certificate verification failed for " + url.m_Host + ": " +
                                        X509_verify_cert_error_string(verify));
                    }
                    throw HttpError("TLS handshake with " + url.m_Host + " failed: " + code.message());
                }
            }
            code = Await(
                io, deadline,
                [&](auto done)
                {
                    asio::async_write(stream, asio::buffer(text),
                                      [done](std::error_code error, size_t) { done(error); });
                },
                cancel);
            if (code)
            {
                throw HttpError("HTTP write to " + url.m_Host + " failed: " + code.message());
            }

            Reader<_Stream> reader{io, stream, deadline, cancel};
            size_t limit = request.m_MaxResponseBytes;
            while (true)
            {
                std::string head = reader.Head(std::min<size_t>(limit, 64 * 1024));
                Fields fields;
                std::string status = ParseHead(head, fields);
                // "HTTP/1.x SSS reason"
                if (status.size() < 12 || !status.starts_with("HTTP/1.") || status[8] != ' ')
                {
                    throw HttpError("malformed HTTP status line");
                }
                int32_t code = 0;
                auto result = std::from_chars(status.data() + 9, status.data() + 12, code);
                if (result.ec != std::errc{} || result.ptr != status.data() + 12 || code < 100 || code > 999)
                {
                    throw HttpError("malformed HTTP status code");
                }
                if (code >= 100 && code < 200 && code != 101)
                {
                    continue; // interim response (100 Continue, 103 Early Hints)
                }
                HttpResponse response;
                response.m_Status = code;
                response.m_Headers = std::move(fields);
                bool bodyless = SameText(request.m_Method, "HEAD") || code < 200 || code == 204 || code == 304;
                if (!bodyless)
                {
                    response.m_Body = reader.Body(response.m_Headers, limit - std::min(limit, head.size()), true);
                }
                return response;
            }
        }
    } // namespace

    std::string HttpResponse::Header(std::string_view name) const
    {
        return FindField(m_Headers, name);
    }

    Url ParseUrl(std::string_view text)
    {
        Url url;
        size_t separator = text.find("://");
        if (separator == std::string_view::npos)
        {
            throw HttpError("URL has no scheme");
        }
        for (char c : text.substr(0, separator))
        {
            url.m_Scheme.push_back(LowerChar(c));
        }
        if (url.m_Scheme != "http" && url.m_Scheme != "https")
        {
            throw HttpError("unsupported URL scheme '" + url.m_Scheme + "'");
        }
        std::string_view rest = text.substr(separator + 3);
        size_t authorityEnd = rest.find_first_of("/?#");
        std::string_view authority = rest.substr(0, authorityEnd);
        std::string_view target =
            authorityEnd == std::string_view::npos ? std::string_view{} : rest.substr(authorityEnd);
        target = target.substr(0, target.find('#'));
        url.m_Target = target.empty() || target.front() != '/' ? "/" + std::string(target) : std::string(target);
        // RFC 3986 path/query characters only: CR/LF/space would split or extend the request line
        static constexpr std::string_view TARGET_SYMBOLS = "-._~!$&'()*+,;=:@/?%";
        for (char c : url.m_Target)
        {
            bool alphanumeric = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
            if (!alphanumeric && TARGET_SYMBOLS.find(c) == std::string_view::npos)
            {
                throw HttpError("invalid character in URL path or query (percent-encode it)");
            }
        }
        if (authority.find('@') != std::string_view::npos)
        {
            throw HttpError("URL user information is not supported");
        }
        std::string_view port;
        if (!authority.empty() && authority.front() == '[')
        {
            size_t close = authority.find(']');
            if (close == std::string_view::npos)
            {
                throw HttpError("unterminated IPv6 address in URL");
            }
            url.m_Host = std::string(authority.substr(1, close - 1));
            std::string_view after = authority.substr(close + 1);
            if (!after.empty())
            {
                if (after.front() != ':')
                {
                    throw HttpError("invalid URL authority");
                }
                port = after.substr(1);
                if (port.empty())
                {
                    throw HttpError("empty URL port");
                }
            }
        }
        else
        {
            size_t colon = authority.find(':');
            url.m_Host = std::string(authority.substr(0, colon));
            if (colon != std::string_view::npos)
            {
                port = authority.substr(colon + 1);
                if (port.empty() || port.find(':') != std::string_view::npos)
                {
                    throw HttpError("invalid URL port");
                }
            }
        }
        if (url.m_Host.empty())
        {
            throw HttpError("URL has no host");
        }
        for (char c : url.m_Host)
        {
            if (static_cast<uint8_t>(c) <= 0x20 || c == '/' || c == '\\' || c == '"' || c == '<' || c == '>')
            {
                throw HttpError("invalid character in URL host");
            }
        }
        url.m_Port = url.m_Scheme == "https" ? 443 : 80;
        if (!port.empty())
        {
            uint32_t value = 0;
            auto result = std::from_chars(port.data(), port.data() + port.size(), value);
            if (result.ec != std::errc{} || result.ptr != port.data() + port.size() || value == 0 || value > 65535)
            {
                throw HttpError("invalid URL port");
            }
            url.m_Port = static_cast<uint16_t>(value);
        }
        return url;
    }

    bool IsLoopbackHost(std::string_view host)
    {
        std::string lower;
        for (char c : host)
        {
            lower.push_back(LowerChar(c));
        }
        if (lower == "localhost" || lower == "::1" || lower == "0:0:0:0:0:0:0:1")
        {
            return true;
        }
        // dotted IPv4 127.a.b.c
        std::vector<std::string_view> parts;
        std::string_view rest = lower;
        while (true)
        {
            size_t dot = rest.find('.');
            parts.push_back(rest.substr(0, dot));
            if (dot == std::string_view::npos)
            {
                break;
            }
            rest.remove_prefix(dot + 1);
        }
        if (parts.size() != 4 || parts.front() != "127")
        {
            return false;
        }
        for (std::string_view part : parts)
        {
            uint32_t value = 0;
            auto result = std::from_chars(part.data(), part.data() + part.size(), value);
            if (part.empty() || result.ec != std::errc{} || result.ptr != part.data() + part.size() || value > 255)
            {
                return false;
            }
        }
        return true;
    }

    HttpResponse SendHttp(const HttpRequest &request)
    {
        Url url = ParseUrl(request.m_Url);
        std::string text = BuildRequest(request, url);
        auto deadline = Clock::now() + std::chrono::milliseconds(request.m_TimeoutMs);
        try
        {
            asio::io_context io;
            tcp::resolver resolver(io);
            tcp::resolver::results_type endpoints;
            std::error_code code = Await(
                io, deadline,
                [&](auto done)
                {
                    resolver.async_resolve(
                        url.m_Host, std::to_string(url.m_Port),
                        [&endpoints, done](std::error_code error, tcp::resolver::results_type results)
                        {
                            endpoints = std::move(results);
                            done(error);
                        });
                },
                [&resolver] { resolver.cancel(); });
            if (code)
            {
                throw HttpError("cannot resolve '" + url.m_Host + "': " + code.message());
            }
            if (url.m_Scheme == "http")
            {
                tcp::socket socket(io);
                return Exchange(io, socket, socket, endpoints, nullptr, url, request, text, deadline);
            }

            asio::ssl::context context(asio::ssl::context::tls_client);
            SSL_CTX *native = context.native_handle();
            SSL_CTX_set_min_proto_version(native, TLS1_2_VERSION);
            SSL_CTX_set_options(native, SSL_OP_NO_COMPRESSION | SSL_OP_NO_RENEGOTIATION);
            if (request.m_VerifyPeer)
            {
                context.set_verify_mode(asio::ssl::verify_peer);
                if (!request.m_CaFile.empty())
                {
                    if (SSL_CTX_load_verify_locations(native, request.m_CaFile.c_str(), nullptr) != 1)
                    {
                        ERR_clear_error();
                        throw HttpError("cannot load CA file '" + request.m_CaFile + "'");
                    }
                }
                else
                {
                    SSL_CTX_set_default_verify_paths(native);
                    ERR_clear_error();
#ifdef _WIN32
                    ImportWindowsRoots(native);
#endif
                }
            }
            else
            {
                context.set_verify_mode(asio::ssl::verify_none);
            }
            asio::ssl::stream<tcp::socket> stream(io, context);
            SSL *tls = stream.native_handle();
            bool address = IsAddress(url.m_Host);
            if (!address && SSL_set_tlsext_host_name(tls, url.m_Host.c_str()) != 1)
            {
                throw HttpError("cannot set TLS server name");
            }
            if (request.m_VerifyPeer)
            {
                SSL_set_hostflags(tls, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
                int bound = address ? X509_VERIFY_PARAM_set1_ip_asc(SSL_get0_param(tls), url.m_Host.c_str())
                                    : SSL_set1_host(tls, url.m_Host.c_str());
                if (bound != 1)
                {
                    throw HttpError("cannot bind TLS hostname verification");
                }
            }
            return Exchange(io, stream, stream.next_layer(), endpoints, tls, url, request, text, deadline);
        }
        catch (const HttpError &)
        {
            throw;
        }
        catch (const std::exception &error)
        {
            throw HttpError(std::string("HTTP request failed: ") + error.what());
        }
    }
} // namespace eOps
