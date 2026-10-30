#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <eMQ/Client.h>
#include <eCore/Network/IOContext.h>
#include <Runtime/Runtime.h>
#include <deque>
#include <thread>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    struct RawPeer
    {
        asio::io_context m_Io;
        asio::ip::tcp::socket m_Socket{m_Io};
        FrameStream m_Decoder;
        std::deque<Message> m_Messages;

        explicit RawPeer(uint16_t port)
        {
            m_Socket.connect({asio::ip::make_address("127.0.0.1"), port});
            m_Socket.non_blocking(true);
        }

        explicit RawPeer(asio::ip::tcp::acceptor &acceptor)
        {
            acceptor.accept(m_Socket);
            m_Socket.non_blocking(true);
        }

        void Bytes(const Buffer &bytes)
        {
            asio::write(m_Socket, asio::buffer(bytes));
        }
        void Send(const Message &message)
        {
            this->Bytes(Codec::Encode(message));
        }
        void Close()
        {
            m_Socket.close();
        }

        std::optional<Message> Next(uint32_t timeout = 3000)
        {
            auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
            std::array<uint8_t, 8192> bytes;
            while (m_Messages.empty())
            {
                asio::error_code error;
                auto length = m_Socket.read_some(asio::buffer(bytes), error);
                if (!error)
                {
                    m_Decoder.Feed(std::span(bytes).first(length),
                                   [&](auto frame) { m_Messages.push_back(Codec::Decode(frame)); });
                }
                else if (error != asio::error::would_block && error != asio::error::try_again)
                {
                    return {};
                }
                if (Clock::now() >= deadline)
                {
                    throw std::runtime_error("Raw peer receive timeout");
                }
                if (m_Messages.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                }
            }
            auto message = std::move(m_Messages.front());
            m_Messages.pop_front();
            return message;
        }

        template <typename _Message> _Message Read()
        {
            for (size_t i = 0; i < 16; ++i)
            {
                auto message = this->Next();
                if (!message)
                {
                    throw std::runtime_error("Raw peer closed before expected packet");
                }
                if (auto result = std::get_if<_Message>(&*message))
                {
                    return *result;
                }
            }
            throw std::runtime_error("Unexpected raw peer packet sequence");
        }

        ConnectionAck Connect(SessionPolicy policy = {})
        {
            ConnectionRequest request;
            request.m_Properties.m_Policy = policy;
            this->Send(request);
            return this->Read<ConnectionAck>();
        }
    };

    template <typename _Predicate> void Until(_Predicate &&predicate, Client *client = nullptr, uint32_t timeout = 3000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        do
        {
            if (client)
            {
                client->Poll();
            }
            if (predicate())
            {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (Clock::now() < deadline);
        throw std::runtime_error("Adversarial condition timed out");
    }
} // namespace

EMPY_TEST(Adversarial, ReferenceCases)
{
    eMQ::Server::Configs options;
    options.m_Port = 0;
    options.m_RetryInterval = 50;
    Broker broker(options);
    auto port = broker.Listen();
    {
        RawPeer wrongOrder(port);
        wrongOrder.Send(PublishAck{1, {}});
        EMPY_REQUIRE(wrongOrder.Read<ConnectionAck>().m_Reason.m_Code == MMQP_REASON_MALFORMED_PACKET);
        EMPY_REQUIRE(!wrongOrder.Next());
    }
    {
        RawPeer oversized(port);
        oversized.Bytes({0x89, 0x20, 0x00, 0x00}); // Advertise 2 MiB without sending its body.
        EMPY_REQUIRE(!oversized.Next());
    }
    Client subscriber;
    subscriber.Connect("127.0.0.1", port);
    EMPY_REQUIRE(subscriber.Subscribe("raw/value").get().m_Code < 0x80);
    size_t received = 0;
    subscriber.On<Receive>("raw/value", [&](const Receive &) { ++received; });
    {
        RawPeer publisher(port);
        EMPY_REQUIRE(publisher.Connect().m_Reason.m_Code == 0);
        publisher.Send(ProfileRequest{1, {{MMQP_PROREQ_TYPE_REG, 1, "raw/value", 0}}});
        EMPY_REQUIRE(publisher.Read<ProfileAck>().m_Reason.m_Code == 0);
        PublishRequest publication;
        publication.m_Alias = 1;
        publication.m_Qos = 1;
        publication.m_PacketId = 1;
        publication.m_Payload = {1, 2, 3};
        publisher.Send(publication);
        EMPY_REQUIRE(publisher.Read<PublishAck>().m_Reason.m_Code == 0);
        publication.m_Duplicate = true;
        publisher.Send(publication); // Simulate an acknowledgement lost to the sender.
        EMPY_REQUIRE(publisher.Read<PublishAck>().m_Reason.m_Code == 0);
        Until([&] { return received == 1; }, &subscriber);
        publication.m_Payload.MutableData()[0] ^= 1;
        publisher.Send(publication);
        EMPY_REQUIRE(publisher.Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_INVALID_PID);
    }
    {
        RawPeer silentSubscriber(port);
        silentSubscriber.Connect();
        silentSubscriber.Send(ProfileRequest{1, {{MMQP_PROREQ_TYPE_SUB, 1, "raw/retry", 1}}});
        EMPY_REQUIRE(silentSubscriber.Read<ProfileAck>().m_Reason.m_Code == 0);
        Client publisher;
        publisher.Connect("127.0.0.1", port);
        EMPY_REQUIRE(publisher.Register("raw/retry").get().m_Code < 0x80);
        EMPY_REQUIRE(publisher.Publish("raw/retry", {7}, 1).get().m_Code < 0x80);
        auto first = silentSubscriber.Read<PublishRequest>();
        auto retry = silentSubscriber.Read<PublishRequest>();
        EMPY_REQUIRE(first.m_PacketId == retry.m_PacketId && retry.m_Duplicate && first.m_Payload == retry.m_Payload);
        silentSubscriber.Send(PublishAck{retry.m_PacketId, {}});
        Until([&] { return broker.Metrics().m_Pending == 0; });
    }
    {
        Client responder;
        responder.Connect("127.0.0.1", port);
        EMPY_REQUIRE(responder.Subscribe("raw/rpc", 1, true).get().m_Code == 0);
        std::vector<PublishRequest> requests;
        responder.On<Receive>("raw/rpc", [&](const auto &event) { requests.push_back(event.m_Message); });
        RawPeer requester(port);
        requester.Connect();
        requester.Send(ProfileRequest{1, {{MMQP_PROREQ_TYPE_REG, 1, "raw/rpc", 0, true}}});
        EMPY_REQUIRE(requester.Read<ProfileAck>().m_Reason.m_Code == 0);
        PublishRequest request;
        request.m_Type = MMQP_PUBREQ_TYPE_REQ;
        request.m_Alias = 1;
        request.m_Qos = 1;
        request.m_PacketId = 1;
        request.m_CorrelationId = 100;
        request.m_Payload = {'Q'};
        requester.Send(request);
        EMPY_REQUIRE(requester.Read<PublishAck>().m_Reason.m_Code == 0);
        request.m_Duplicate = true;
        requester.Send(request);
        EMPY_REQUIRE(requester.Read<PublishAck>().m_Reason.m_Code == 0);
        Until([&] { return requests.size() == 1; }, &responder);
        request.m_PacketId = 2;
        request.m_Duplicate = false;
        requester.Send(request); // A new hop PID must not permit CID reuse.
        EMPY_REQUIRE(requester.Read<PublishAck>().m_Reason.m_Code == MMQP_REASON_INVALID_CORRELATION);
        request.m_PacketId = 3;
        request.m_CorrelationId = 101;
        requester.Send(request);
        EMPY_REQUIRE(requester.Read<PublishAck>().m_Reason.m_Code == 0);
        Until([&] { return requests.size() == 2; }, &responder);
        EMPY_REQUIRE(responder.Respond(requests[1], {'B'}).get().m_Code == 0);
        auto response = requester.Read<PublishRequest>();
        EMPY_REQUIRE(response.m_CorrelationId == 101 && response.m_Payload == Buffer{'B'});
        requester.Send(PublishAck{response.m_PacketId, {}});
        EMPY_REQUIRE(responder.Respond(requests[0], {'A'}).get().m_Code == 0);
        response = requester.Read<PublishRequest>();
        EMPY_REQUIRE(response.m_CorrelationId == 100 && response.m_Payload == Buffer{'A'});
        requester.Send(PublishAck{response.m_PacketId, {}});
        request.m_Duplicate = true;
        request.m_CorrelationId = 102;
        requester.Send(request); // The reliable receipt digest binds the CID.
        EMPY_REQUIRE(requester.Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_INVALID_PID);
    }
    {
        eMQ::Server::Configs bounded = options;
        bounded.m_MaxSessionBytes = 64;
        Broker limited(bounded);
        auto limitedPort = limited.Listen();
        Client publisher, receiver;
        publisher.Connect("127.0.0.1", limitedPort);
        receiver.Connect("127.0.0.1", limitedPort);
        publisher.Register("budget/value").get();
        receiver.Subscribe("budget/value").get();
        uint16_t delivered = 0;
        receiver.On<Receive>("budget/value", [&](const Receive &event) { delivered = event.m_Message.m_PacketId; });
        for (size_t i = 0; i < 3; ++i)
        {
            EMPY_REQUIRE(publisher.Publish("budget/value", Buffer(128), 1).get().m_Code == MMQP_REASON_BUSY);
        }
        EMPY_REQUIRE(publisher.Publish("budget/value", {1}, 1).get().m_Code < 0x80);
        Until([&] { return delivered != 0; }, &receiver);
        EMPY_REQUIRE(delivered == 1); // Failed fan-out admission must not consume the subscriber's PID.
    }
    {
        RawPeer responder(port);
        responder.Connect();
        responder.Send(ProfileRequest{1, {{MMQP_PROREQ_TYPE_SUB, 1, "raw/rpc-retry", 1, true}}});
        EMPY_REQUIRE(responder.Read<ProfileAck>().m_Reason.m_Code == 0);
        Client requester;
        requester.Connect("127.0.0.1", port);
        EMPY_REQUIRE(requester.Register("raw/rpc-retry", 0, 60, 1, true).get().m_Code == 0);
        size_t responses = 0;
        requester.On<Receive>("raw/rpc-retry",
                              [&](const auto &event)
                              {
                                  EMPY_REQUIRE(event.m_Message.m_CorrelationId == 0x2345);
                                  ++responses;
                              });
        EMPY_REQUIRE(requester.Request("raw/rpc-retry", {'Q'}, 1, 1, 0x2345).get().m_Code == 0);
        auto request = responder.Read<PublishRequest>();
        auto retry = responder.Read<PublishRequest>();
        EMPY_REQUIRE(request.m_CorrelationId == retry.m_CorrelationId && request.m_PacketId == retry.m_PacketId &&
                     retry.m_Duplicate);
        responder.Send(PublishAck{retry.m_PacketId, {}});
        PublishRequest response;
        response.m_Type = MMQP_PUBREQ_TYPE_RES;
        response.m_Alias = 1;
        response.m_Topic = "raw/rpc-retry";
        response.m_Qos = 1;
        response.m_PacketId = 1;
        response.m_CorrelationId = request.m_CorrelationId;
        response.m_Payload = {'R'};
        responder.Send(response);
        EMPY_REQUIRE(responder.Read<PublishAck>().m_Reason.m_Code == 0);
        response.m_Duplicate = true;
        responder.Send(response); // Retry after the response mapping was consumed.
        EMPY_REQUIRE(responder.Read<PublishAck>().m_Reason.m_Code == 0);
        Until([&] { return responses == 1 && broker.Metrics().m_Pending == 0; }, &requester);
    }
    {
        RawPeer idle(port);
        auto ack = idle.Connect(SessionPolicy{false, EncodeInterval(1), EncodeInterval(1), {}});
        EMPY_REQUIRE(ack.m_Reason.m_Code == 0);
        EMPY_REQUIRE(!idle.Read<ConnectionSync>().m_Token);
        EMPY_REQUIRE(idle.Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_KEEPALIVE_TIMEOUT);
        idle.Close();
        std::this_thread::sleep_for(std::chrono::milliseconds(1100));
        RawPeer expired(port);
        ConnectionRequest resume;
        resume.m_Resume = true;
        resume.m_Properties.m_Auth = SessionAuth{ack.m_Properties.m_Auth->m_Identifier, {}};
        expired.Send(resume);
        EMPY_REQUIRE(expired.Read<ConnectionAck>().m_Reason.m_Code == MMQP_REASON_SESSION_EXPIRED);
    }
    {
        eMQ::Configs crypto;
        crypto.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{Buffer(16, 0x11)}, 1, 1);
        Client publisher(crypto);
        crypto.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{Buffer(16, 0x11)}, 1, 1);
        Client receiver(crypto);
        publisher.Connect("127.0.0.1", port);
        receiver.Connect("127.0.0.1", port);
        publisher.Register("encrypted/retained").get();
        receiver.Subscribe("encrypted/retained").get();
        size_t count = 0;
        receiver.On<Receive>("encrypted/#", [&](const Receive &) { ++count; });
        publisher.Publish("encrypted/retained", {1, 2, 3}, 1).get();
        Until([&] { return count == 1; }, &receiver);
        receiver.Unsubscribe("encrypted/retained").get();
        receiver.Subscribe("encrypted/retained").get();
        Until([&] { return broker.Metrics().m_Pending == 0; }, &receiver);
        EMPY_REQUIRE(count == 1 && receiver.Connected());
    }
    EMPY_REQUIRE(broker.Metrics().m_Duplicates >= 1 && broker.Metrics().m_Retransmitted >= 1);
    {
        Client publisher, wildcard;
        publisher.Connect("127.0.0.1", port);
        wildcard.Connect("127.0.0.1", port);
        publisher.Register("aliases/value").get();
        wildcard.Subscribe("aliases/#").get();
        uint16_t alias = 0;
        wildcard.On<Receive>("aliases/#", [&](const Receive &event) { alias = event.m_Message.m_Alias; });
        publisher.Publish("aliases/value", {1}, 1).get();
        Until([&] { return alias != 0; }, &wildcard);
        EMPY_REQUIRE(alias >= 0x8000);
        EMPY_REQUIRE(wildcard.Subscribe("aliases/value").get().m_Code == 0);
    }
    {
        eMQ::Server::Configs tenant = options;
        tenant.m_TenantRoot = "tenant";
        Broker isolated(tenant);
        auto isolatedPort = isolated.Listen();
        Client monitor;
        monitor.Connect("127.0.0.1", isolatedPort);
        EMPY_REQUIRE(monitor.Subscribe("other/#").get().m_Code == MMQP_REASON_UNAUTHORIZED);
        EMPY_REQUIRE(monitor.Subscribe("$bwm/other/#").get().m_Code == MMQP_REASON_UNAUTHORIZED);
        EMPY_REQUIRE(monitor.Subscribe("$bwm/tenant/#").get().m_Code == 0);
        std::string birth;
        monitor.On<Receive>("$bwm/tenant/#", [&](const Receive &event) { birth = event.m_Message.m_Topic; });
        eMQ::Configs status;
        status.m_Properties.m_Will = SessionWill{Buffer{1}, {}, {}, 0};
        Client device(status);
        auto sid = device.Connect("127.0.0.1", isolatedPort);
        Until([&] { return !birth.empty(); }, &monitor);
        EMPY_REQUIRE(birth == "$bwm/tenant/" + sid);
    }
    {
        // A scripted broker exercises client-side checks the real broker never violates.
        asio::io_context context;
        asio::ip::tcp::acceptor acceptor(context, {asio::ip::make_address("127.0.0.1"), 0});
        auto fakePort = acceptor.local_endpoint().port();
        auto serve = [&](Client &client, uint32_t keepalive = 30)
        {
            auto connected = client.AsyncConnect("127.0.0.1", fakePort);
            auto peer = std::make_unique<RawPeer>(acceptor);
            peer->Read<ConnectionRequest>();
            ConnectionAck ack;
            ack.m_Properties.m_Auth = SessionAuth{"scripted", {}};
            ack.m_Properties.m_Policy = SessionPolicy{false, EncodeInterval(300), EncodeInterval(keepalive), 65536};
            peer->Send(ack);
            EMPY_REQUIRE(connected.get() == "scripted");
            return peer;
        };
        auto subscribe = [](Client &client, RawPeer &peer, std::string topic)
        {
            auto result = client.Subscribe(std::move(topic));
            auto request = peer.Read<ProfileRequest>();
            peer.Send(ProfileAck{request.m_PacketId, {}});
            EMPY_REQUIRE(result.get().m_Code == 0);
            return request.m_Entries.front().m_Alias;
        };
        auto delivery = [](uint16_t alias, std::string topic, uint16_t packetId, Buffer payload)
        {
            PublishRequest publication;
            publication.m_Alias = alias;
            publication.m_Topic = std::move(topic);
            publication.m_Qos = 1;
            publication.m_PacketId = packetId;
            publication.m_Payload = std::move(payload);
            return publication;
        };
        {
            Client client;
            auto peer = serve(client);
            auto alias = subscribe(client, *peer, "scripted/value");
            size_t count = 0;
            client.On<Receive>("scripted/#", [&](const Receive &) { ++count; });
            auto publication = delivery(alias, "scripted/value", 1, {1});
            peer->Send(publication);
            EMPY_REQUIRE(peer->Read<PublishAck>().m_PacketId == 1);
            publication.m_Topic.clear();
            publication.m_Duplicate = true;
            peer->Send(publication); // Identical content under the same PID is acknowledged only.
            EMPY_REQUIRE(peer->Read<PublishAck>().m_PacketId == 1);
            peer->Send(ConnectionSync{});
            EMPY_REQUIRE(!peer->Read<ConnectionSync>().m_Token);
            Until([&] { return count == 1; }, &client);
            publication.m_Payload.MutableData()[0] ^= 1;
            peer->Send(publication);
            EMPY_REQUIRE(peer->Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_INVALID_PID);
            Until([&] { return !client.Connected(); });
            EMPY_REQUIRE(count == 1);
        }
        {
            Client client;
            auto peer = serve(client);
            auto alias = subscribe(client, *peer, "scripted/value");
            peer->Send(delivery(alias, "scripted/value", 2, {2}));
            EMPY_REQUIRE(peer->Read<PublishAck>().m_PacketId == 2);
            peer->Send(delivery(alias, {}, 1, {1})); // At or below the checkpoint without a cached receipt.
            EMPY_REQUIRE(peer->Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_INVALID_PID);
        }
        {
            Client client;
            auto peer = serve(client);
            peer->Send(ConnectionSync{Buffer(32, 0x42)}); // Tokens are valid only during the challenge.
            EMPY_REQUIRE(peer->Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_MALFORMED_PACKET);
        }
        {
            Client client;
            auto peer = serve(client);
            subscribe(client, *peer, "scripted/#");
            peer->Send(delivery(0x8001, "scripted/wild", 1, {1}));
            EMPY_REQUIRE(peer->Read<PublishAck>().m_PacketId == 1);
            peer->Send(delivery(5, "scripted/other", 2, {2})); // New broker aliases must be >= 32768.
            EMPY_REQUIRE(peer->Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_INVALID_ALIAS);
        }
        {
            eMQ::Configs recovering;
            recovering.m_AutoReconnect = true;
            recovering.m_FailoverEndpoints = {{"127.0.0.1", fakePort}};
            Client client(recovering);
            std::vector<Disconnect> disconnects;
            client.On<Disconnect>([&](const Disconnect &event) { disconnects.push_back(event); });
            auto start = Clock::now();
            auto peer = serve(client, 1);
            // Stay silent: no ping arrives, so the client must give up after twice the keepalive.
            EMPY_REQUIRE(peer->Read<ConnectionEnd>().m_Reason.m_Code == MMQP_REASON_KEEPALIVE_TIMEOUT);
            EMPY_REQUIRE(Clock::now() - start >= std::chrono::milliseconds(1900));
            Until([&] { return !disconnects.empty(); }, &client);
            EMPY_REQUIRE(disconnects[0].m_Reason.m_Code == MMQP_REASON_KEEPALIVE_TIMEOUT &&
                         disconnects[0].m_Reconnecting);
            RawPeer successor(acceptor);
            EMPY_REQUIRE(successor.Read<ConnectionRequest>().m_Resume);
        }
    }
}
