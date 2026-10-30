#include <eOps/Harness/Status.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/HttpServer.h>
#include <eOps/Common/Text.h>
#include <mutex>

namespace eOps::Harness
{
    namespace
    {
        HttpResponse Reply(int32_t status, const Json &body)
        {
            HttpResponse response;
            response.m_Status = status;
            response.m_Headers = {{"Content-Type", "application/json"}, {"Cache-Control", "no-store"}};
            response.m_Body = body.Dump();
            return response;
        }

        HttpResponse Error(int32_t status, std::string text)
        {
            return Reply(status, Json::MakeObject({{"error", std::move(text)}}));
        }
    } // namespace

    struct StatusServer::Impl
    {
        std::string m_Address;
        uint16_t m_Port = 0;
        mutable std::mutex m_Mutex;
        Json m_Snapshot = Json::MakeObject();
        uint64_t m_UpdatedAt = 0;
        std::unique_ptr<HttpServer> m_Server;

        HttpResponse Handle(const HttpServerRequest &request)
        {
            std::string path = request.m_Target.substr(0, request.m_Target.find('?'));
            if (path != "/api/mlops/status" && path != "/api/mlops/healthz")
            {
                return Error(404, "not found");
            }
            if (request.m_Method != "GET")
            {
                HttpResponse response = Error(405, "read-only endpoint");
                response.m_Headers.emplace_back("Allow", "GET");
                return response;
            }
            uint64_t now = NowMillis();
            std::scoped_lock lock(m_Mutex);
            if (path == "/api/mlops/status")
            {
                return Reply(200, m_Snapshot);
            }
            uint64_t age = m_UpdatedAt ? now - std::min(now, m_UpdatedAt) : 0;
            return Reply(200, Json::MakeObject({{"status", "ok"}, {"time", now}, {"snapshotAge", age}}));
        }
    };

    StatusServer::StatusServer(std::string address, uint16_t port) : m_Impl(std::make_unique<Impl>())
    {
        m_Impl->m_Address = std::move(address);
        m_Impl->m_Port = port;
    }

    StatusServer::~StatusServer()
    {
        Stop();
    }

    uint16_t StatusServer::Start()
    {
        if (m_Impl->m_Server)
        {
            return m_Impl->m_Server->Port();
        }
        Impl *impl = m_Impl.get();
        auto server =
            std::make_unique<HttpServer>(m_Impl->m_Address, m_Impl->m_Port,
                                         [impl](const HttpServerRequest &request) { return impl->Handle(request); });
        uint16_t port = server->Start();
        m_Impl->m_Server = std::move(server);
        return port;
    }

    void StatusServer::Stop()
    {
        if (m_Impl->m_Server)
        {
            m_Impl->m_Server->Stop();
            m_Impl->m_Server.reset();
        }
    }

    void StatusServer::Update(Json snapshot)
    {
        uint64_t now = NowMillis();
        std::scoped_lock lock(m_Impl->m_Mutex);
        m_Impl->m_Snapshot = std::move(snapshot);
        m_Impl->m_UpdatedAt = now;
    }

    uint16_t StatusServer::Port() const noexcept
    {
        return m_Impl->m_Server ? m_Impl->m_Server->Port() : 0;
    }
} // namespace eOps::Harness
