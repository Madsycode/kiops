#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <eMQ/Client.h>
#include <Runtime/Runtime.h>
#include <Monitoring/Telemetry.inl>
#include <thread>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    std::string Http(uint16_t port, std::string path, std::string headers = {}, std::string method = "GET",
                     std::string host = {})
    {
        asio::io_context io;
        asio::ip::tcp::socket socket(io);
        socket.connect({asio::ip::make_address("127.0.0.1"), port});
        if (host.empty())
        {
            host = "127.0.0.1:" + std::to_string(port);
        }
        auto request = method + ' ' + path + " HTTP/1.1\r\nHost: " + host + "\r\n" + headers + "\r\n";
        asio::write(socket, asio::buffer(request));
        std::string reply;
        asio::error_code error;
        asio::read(socket, asio::dynamic_buffer(reply, 1024 * 1024), error);
        EMPY_REQUIRE(error == asio::error::eof || error == asio::error::connection_reset);
        return reply;
    }
} // namespace

EMPY_TEST(Monitoring, ReferenceCases)
{
    eMQ::Server::Configs settings;
    settings.m_Port = 0;
    settings.m_Transport.m_MaxPacket = 65536;
    Broker broker(settings);
    auto port = broker.Listen();
    Client publisher, subscriber;
    publisher.Connect("127.0.0.1", port);
    auto sid = subscriber.Connect("127.0.0.1", port);
    EMPY_REQUIRE(subscriber.Subscribe("monitor/#").get().m_Code < 0x80);
    EMPY_REQUIRE(publisher.Register("monitor/value").get().m_Code < 0x80);
    auto initial = broker.Inspect();
    EMPY_REQUIRE(initial.m_Metrics.m_Online == 2 && initial.m_Sessions.size() == 2 && !initial.m_Cluster);
    auto one = broker.Inspect(1);
    EMPY_REQUIRE(one.m_Sessions.size() == 1 && one.m_OmittedSessions == 1 && one.m_Metrics.m_Sessions == 2);
    Throws([&] { broker.Inspect(0); });
    Throws([&] { broker.Inspect(4097); });
    EMPY_REQUIRE(publisher.Publish("monitor/value", {'S', 'E', 'C', 'R', 'E', 'T'}, 1).get().m_Code < 0x80);
    auto snapshot = broker.Inspect();
    EMPY_REQUIRE(snapshot.m_Metrics.m_Publications == 1 && snapshot.m_Metrics.m_Forwarded == 1);
    EMPY_REQUIRE(snapshot.m_Metrics.m_ReceivedBytes > initial.m_Metrics.m_ReceivedBytes &&
                 snapshot.m_Metrics.m_SentBytes > initial.m_Metrics.m_SentBytes);
    auto session = std::find_if(snapshot.m_Sessions.begin(), snapshot.m_Sessions.end(),
                                [&](const auto &item) { return item.m_Identifier == sid; });
    EMPY_REQUIRE(session != snapshot.m_Sessions.end() && session->m_Online && session->m_Owner &&
                 session->m_Subscriptions == 1);
    EMPY_REQUIRE(session->m_Deliveries == 1 && session->m_SentPackets > 0 && session->m_Transport == "TCP");
    auto json = SnapshotJson(snapshot);
    EMPY_REQUIRE(json.find("SECRET") == std::string::npos && json.find("monitor/value") == std::string::npos);
    snapshot.m_Identifier = "quoted\"\\\n\t";
    EMPY_REQUIRE(SnapshotJson(snapshot).find("quoted\\\"\\\\\\u000a\\u0009") != std::string::npos);

    TelemetryOptions options;
    options.m_Port = 0;
    options.m_IntervalMs = 250;
    options.m_Token = std::string(40, 't');
    auto invalid = options;
    invalid.m_Address = "0.0.0.0";
    Throws([&] { TelemetryServer server(broker, invalid); });
    invalid = options;
    invalid.m_Token = "short";
    Throws([&] { TelemetryServer server(broker, invalid); });
    invalid = options;
    invalid.m_IntervalMs = 0;
    Throws([&] { TelemetryServer server(broker, invalid); });

    TelemetryServer server(broker, options);
    auto httpPort = server.Start();
    auto authorization = "Authorization: Bearer " + options.m_Token + "\r\n";
    EMPY_REQUIRE(Http(httpPort, "/api/v1/snapshot").starts_with("HTTP/1.1 401"));
    EMPY_REQUIRE(Http(httpPort, "/health", authorization).starts_with("HTTP/1.1 200"));
    EMPY_REQUIRE(Http(httpPort, "/health", authorization, "POST").starts_with("HTTP/1.1 405"));
    EMPY_REQUIRE(Http(httpPort, "/unknown", authorization).starts_with("HTTP/1.1 404"));
    EMPY_REQUIRE(Http(httpPort, "/health", authorization, "GET", "attacker.example").starts_with("HTTP/1.1 403"));
    EMPY_REQUIRE(
        Http(httpPort, "/health", authorization + "Origin: https://attacker.example\r\n").starts_with("HTTP/1.1 403"));
    EMPY_REQUIRE(Http(httpPort, "/health", authorization + authorization).starts_with("HTTP/1.1 400"));
    EMPY_REQUIRE(
        Http(httpPort, "/health", authorization + "Transfer-Encoding: chunked\r\n").starts_with("HTTP/1.1 400"));
    EMPY_REQUIRE(Http(httpPort, "/health", "X-Large: " + std::string(9000, 'a') + "\r\n").empty());
    auto deadline = Clock::now() + std::chrono::seconds(3);
    do
    {
        json = Http(httpPort, "/api/v1/snapshot", authorization);
        if (json.starts_with("HTTP/1.1 200"))
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (Clock::now() < deadline);
    EMPY_REQUIRE(json.starts_with("HTTP/1.1 200") && json.find("\"schema\":1") != std::string::npos);
    EMPY_REQUIRE(json.find(options.m_Token) == std::string::npos && json.find("SECRET") == std::string::npos);
    subscriber.Disconnect();
    deadline = Clock::now() + std::chrono::seconds(3);
    do
    {
        snapshot = broker.Inspect();
        if (snapshot.m_Metrics.m_Online == 1)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (Clock::now() < deadline);
    EMPY_REQUIRE(snapshot.m_Metrics.m_Online == 1 && snapshot.m_Metrics.m_Sessions == 2);
    session = std::find_if(snapshot.m_Sessions.begin(), snapshot.m_Sessions.end(),
                           [&](const auto &item) { return item.m_Identifier == sid; });
    EMPY_REQUIRE(session != snapshot.m_Sessions.end() && !session->m_Online && session->m_ExpiresInMs.has_value());
    // Stop must cancel slow HTTP readers immediately, before the broker shuts down.
    asio::io_context io;
    asio::ip::tcp::socket slow(io);
    slow.connect({asio::ip::make_address("127.0.0.1"), httpPort});
    asio::write(slow, asio::buffer(std::string("GET /")));
    auto started = Clock::now();
    server.Stop();
    EMPY_REQUIRE(Clock::now() - started < std::chrono::seconds(2));
    publisher.Disconnect();
}
