#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string_view>
#include "../API.h"

// blocking HTTP/1.1 client (asio + OpenSSL inside EmpyOps): one request per connection,
// Content-Length / chunked / close-delimited bodies, an overall deadline and verified TLS

namespace eOps
{
    struct HttpRequest
    {
        std::string m_Method = "POST";
        std::string m_Url;
        std::vector<std::pair<std::string, std::string>> m_Headers; // Host/Content-Length/Connection are managed
        std::string m_Body;
        uint32_t m_TimeoutMs = 60000; // whole exchange: resolve, connect, handshake, write and read
        std::string m_CaFile;         // PEM trust anchors; empty → system trust
        bool m_VerifyPeer = true;
        size_t m_MaxResponseBytes = 8 * 1024 * 1024; // headers plus decoded body
    };

    struct HttpResponse
    {
        int32_t m_Status = 0;
        std::vector<std::pair<std::string, std::string>> m_Headers;
        std::string m_Body;
        // first header value with a case-insensitive name match ("" when absent)
        EMPY_OPS_API std::string Header(std::string_view name) const;
    };

    // transport, protocol, TLS verification, limit and timeout failures (never contains header values)
    struct HttpError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct Url
    {
        std::string m_Scheme; // "http" | "https" (lowercase)
        std::string m_Host;   // without IPv6 brackets
        uint16_t m_Port = 0;  // explicit or scheme default
        std::string m_Target; // path + query, at least "/"
    };

    // http/https only, no userinfo; the fragment is dropped; the path and query may only use RFC 3986 characters
    // (no spaces, control bytes or raw non-ASCII, so nothing can split the request line); throws HttpError
    EMPY_OPS_API Url ParseUrl(std::string_view url);

    // "localhost", 127.0.0.0/8 or ::1 (credentials may travel over plain http only to these)
    EMPY_OPS_API bool IsLoopbackHost(std::string_view host);

    // returns any HTTP status; throws HttpError on transport failure
    EMPY_OPS_API HttpResponse SendHttp(const HttpRequest &request);
} // namespace eOps
