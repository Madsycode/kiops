#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <functional>
#include "Http.h"

// tiny loopback HTTP/1.1 server: one background thread, sequential connections, a request size
// limit, a 5 s read timeout and `Connection: close`. Used by the harness status endpoint and by
// tests as a mock upstream. The handler runs on the server thread; exceptions become 500.

namespace eOps
{
    struct HttpServerRequest
    {
        std::string m_Method;
        std::string m_Target;
        std::string m_Body;
        std::vector<std::pair<std::string, std::string>> m_Headers;
        // first header value with a case-insensitive name match ("" when absent)
        EMPY_OPS_API std::string Header(std::string_view name) const;
    };

    using HttpHandler = std::function<HttpResponse(const HttpServerRequest &)>;

    class EMPY_OPS_API HttpServer
    {
    public:
        HttpServer(std::string address, uint16_t port, HttpHandler handler, size_t maxRequestBytes = 1 << 20);
        ~HttpServer(); // stops
        HttpServer(const HttpServer &) = delete;
        HttpServer &operator=(const HttpServer &) = delete;

        uint16_t Start(); // returns the bound port (0 → ephemeral); throws HttpError on bind failure
        void Stop();      // idempotent; aborts an active connection and joins the thread
        uint16_t Port() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps
