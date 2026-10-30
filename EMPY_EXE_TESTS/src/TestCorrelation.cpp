#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <eMQ/Client.h>
#include <Runtime/Runtime.h>
#include <filesystem>
#include <thread>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    template <typename _Predicate>
    void Await(_Predicate &&predicate, std::initializer_list<Client *> clients = {}, uint32_t timeout = 4000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        do
        {
            for (auto client : clients)
            {
                client->Poll();
            }
            if (predicate())
            {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (Clock::now() < deadline);
        throw std::runtime_error("Correlation exchange timed out");
    }

    void Accepted(std::future<Reason> result)
    {
        EMPY_REQUIRE(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        auto reason = result.get();
        if (reason.m_Code >= 0x80)
        {
            throw std::runtime_error("Correlation exchange rejected: " + reason.m_Text);
        }
    }

    void Concurrent(bool encrypted)
    {
        eMQ::Server::Configs settings;
        settings.m_Port = 0;
        Broker broker(settings);
        auto port = broker.Listen();
        auto options = [encrypted]
        {
            eMQ::Configs result;
            if (encrypted)
            {
                result.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{Buffer(16, 0x51)}, 1, 1);
            }
            return result;
        };
        Client first(options()), second(options()), responder(options()), intruder(options());
        first.Connect("127.0.0.1", port);
        second.Connect("127.0.0.1", port);
        responder.Connect("127.0.0.1", port);
        intruder.Connect("127.0.0.1", port);
        Accepted(first.Register("rpc/#", 0, 60, 1, true));
        Accepted(second.Register("rpc/#", 0, 60, 1, true));
        Accepted(responder.Subscribe("rpc/#", 1, true));
        std::vector<PublishRequest> incoming, firstReplies, secondReplies;
        responder.On<Receive>("rpc/#",
                              [&](const Receive &event)
                              {
                                  EMPY_REQUIRE(event.m_Decrypted == encrypted);
                                  incoming.push_back(event.m_Message);
                              });
        first.On<Receive>("rpc/#",
                          [&](const Receive &event)
                          {
                              EMPY_REQUIRE(event.m_Decrypted == encrypted);
                              firstReplies.push_back(event.m_Message);
                          });
        second.On<Receive>("rpc/#",
                           [&](const Receive &event)
                           {
                               EMPY_REQUIRE(event.m_Decrypted == encrypted);
                               secondReplies.push_back(event.m_Message);
                           });

        for (uint8_t qos : {uint8_t{1}, uint8_t{0}})
        {
            incoming.clear();
            firstReplies.clear();
            secondReplies.clear();
            auto id = static_cast<uint16_t>(qos ? 4001 : 2001);
            // Concurrent callers, nonmonotonic explicit IDs, and equal IDs in
            // different requester sessions must remain independent.
            auto a =
                std::async(std::launch::async, [&] { return first.Request("rpc/value", {'A'}, 1, qos, id + 1).get(); });
            auto b =
                std::async(std::launch::async, [&] { return first.Request("rpc/value", {'B'}, 1, qos, id).get(); });
            auto c =
                std::async(std::launch::async, [&] { return second.Request("rpc/value", {'C'}, 1, qos, id).get(); });
            Accepted(std::move(a));
            Accepted(std::move(b));
            Accepted(std::move(c));
            Await([&] { return incoming.size() == 3; }, {&responder});
            std::set<uint16_t> routes;
            for (const auto &request : incoming)
            {
                EMPY_REQUIRE(routes.insert(request.m_CorrelationId).second);
            }

            auto wrong = incoming.front();
            wrong.m_Topic = "rpc/wrong";
            EMPY_REQUIRE(responder.Respond(wrong.m_Topic, wrong.m_CorrelationId, {'X'}).get().m_Code ==
                         MMQP_REASON_INVALID_CORRELATION);
            EMPY_REQUIRE(responder.Respond("rpc/value", UINT16_MAX, {'X'}).get().m_Code ==
                         MMQP_REASON_INVALID_CORRELATION);
            Throws([&] { responder.Respond("rpc/value", 0, {'X'}).get(); });
            Accepted(intruder.Subscribe("rpc/#", 1, true));
            EMPY_REQUIRE(intruder.Respond("rpc/value", incoming.front().m_CorrelationId, {'X'}).get().m_Code ==
                         MMQP_REASON_INVALID_CORRELATION);
            Accepted(intruder.Unsubscribe("rpc/#"));
            for (auto it = incoming.rbegin(); it != incoming.rend(); ++it)
            {
                Accepted(responder.Respond(*it, it->m_Payload));
            }
            Await([&] { return firstReplies.size() == 2 && secondReplies.size() == 1; }, {&first, &second});
            for (const auto &reply : firstReplies)
            {
                EMPY_REQUIRE(reply.m_CorrelationId == (reply.m_Payload == Buffer{'A'} ? id + 1 : id));
                EMPY_REQUIRE(reply.m_Payload == Buffer{'A'} || reply.m_Payload == Buffer{'B'});
            }
            EMPY_REQUIRE(secondReplies.front().m_CorrelationId == id && secondReplies.front().m_Payload == Buffer{'C'});
            EMPY_REQUIRE(responder.Respond("rpc/value", incoming.front().m_CorrelationId, {'X'}).get().m_Code ==
                         MMQP_REASON_INVALID_CORRELATION);
            // Completion releases the caller's ID; the same SID may start a new exchange.
            incoming.clear();
            auto count = firstReplies.size();
            Accepted(first.Request("rpc/value", {'R'}, 1, 1, id));
            Await([&] { return incoming.size() == 1; }, {&responder});
            Throws([&] { first.Request("rpc/value", {}, 1, 1, id).get(); });
            Accepted(responder.Respond(incoming.front(), {'R'}));
            Await([&] { return firstReplies.size() == count + 1; }, {&first});
            EMPY_REQUIRE(firstReplies.back().m_CorrelationId == id && firstReplies.back().m_Payload == Buffer{'R'});
        }

        incoming.clear();
        firstReplies.clear();
        Accepted(first.Request("rpc/value", {'D'}));
        Accepted(first.Request("rpc/value", {'E'}));
        Await([&] { return incoming.size() == 2; }, {&responder});
        Accepted(responder.Respond(incoming[1], incoming[1].m_Payload));
        Accepted(responder.Respond(incoming[0], incoming[0].m_Payload));
        Await([&] { return firstReplies.size() == 2; }, {&first});
        EMPY_REQUIRE(firstReplies[0].m_Payload == Buffer{'E'} && firstReplies[1].m_Payload == Buffer{'D'});
        EMPY_REQUIRE(firstReplies[0].m_CorrelationId == 2 && firstReplies[1].m_CorrelationId == 1);
    }

    void Deadlines()
    {
        eMQ::Server::Configs settings;
        settings.m_Port = 0;
        settings.m_RequestTimeout = 350;
        settings.m_MaxRequests = 2;
        Broker broker(settings);
        auto port = broker.Listen();
        Client requester, responder;
        requester.Connect("127.0.0.1", port);
        responder.Connect("127.0.0.1", port);
        Accepted(requester.Register("rpc/#", 0, 60, 1, true));
        Accepted(responder.Subscribe("rpc/slow", 1, true));
        std::vector<PublishRequest> incoming, replies;
        responder.On<Receive>("rpc/#", [&](const auto &event) { incoming.push_back(event.m_Message); });
        requester.On<Receive>("rpc/#", [&](const auto &event) { replies.push_back(event.m_Message); });
        EMPY_REQUIRE(requester.Request("rpc/missing", {}, 1, 1, 40).get().m_Code == MMQP_REASON_SUCCESS);
        Await([&] { return replies.size() == 1; }, {&requester});
        EMPY_REQUIRE(replies.back().m_CorrelationId == 40 &&
                     replies.back().m_Reason.m_Code == MMQP_REASON_NO_RESPONDER);
        Accepted(requester.Request("rpc/slow", {}, 1, 1, 41));
        Accepted(requester.Request("rpc/slow", {}, 1, 1, 42));
        EMPY_REQUIRE(requester.Request("rpc/slow", {}, 1, 1, 43).get().m_Code == MMQP_REASON_BUSY);
        Await([&] { return incoming.size() == 2 && replies.size() == 3; }, {&requester, &responder});
        std::set<uint16_t> timedOut;
        for (size_t i = 1; i < replies.size(); ++i)
        {
            EMPY_REQUIRE(replies[i].m_Reason.m_Code == MMQP_REASON_TIMEOUT);
            timedOut.insert(replies[i].m_CorrelationId);
        }
        EMPY_REQUIRE(timedOut == std::set<uint16_t>({41, 42}));
        Accepted(requester.Request("rpc/slow", {'N'}, 1, 1, 44));
        Await([&] { return incoming.size() == 3; }, {&responder});
        EMPY_REQUIRE(responder.Respond(incoming[0], {'L'}).get().m_Code == MMQP_REASON_INVALID_CORRELATION);
        Accepted(responder.Respond(incoming[2], {'N'}));
        Await([&] { return replies.size() == 4; }, {&requester});
        EMPY_REQUIRE(replies.back().m_CorrelationId == 44 && replies.back().m_Payload == Buffer{'N'});
        for (uint16_t id : {uint16_t{40}, uint16_t{41}, uint16_t{43}})
        {
            incoming.clear();
            auto count = replies.size();
            Accepted(requester.Request("rpc/slow", {'U'}, 1, 1, id));
            Await([&] { return incoming.size() == 1; }, {&responder});
            Accepted(responder.Respond(incoming.front(), {'U'}));
            Await([&] { return replies.size() == count + 1; }, {&requester});
            EMPY_REQUIRE(replies.back().m_CorrelationId == id);
        }
    }

    struct Directory
    {
        std::filesystem::path m_Path =
            std::filesystem::temp_directory_path() / ("EmpyCorrelation-" + ToHex(RandomBytes(8)));
        ~Directory()
        {
            std::error_code error;
            std::filesystem::remove_all(m_Path, error);
        }
    };

    void Recovery()
    {
        Directory directory;
        eMQ::Server::Configs settings;
        settings.m_Port = 0;
        settings.m_RequestTimeout = 10000;
        settings.m_StateDirectory = directory.m_Path.string();
        auto broker = std::make_unique<Broker>(settings);
        auto port = broker->Listen();
        Client requester, responder, other;
        auto requesterId = requester.Connect("127.0.0.1", port);
        auto responderId = responder.Connect("127.0.0.1", port);
        auto otherId = other.Connect("127.0.0.1", port);
        Accepted(requester.Register("rpc/restart", 0, 60, 1, true));
        Accepted(responder.Subscribe("rpc/restart", 1, true));
        Accepted(other.Subscribe("rpc/restart", 1, true));
        std::vector<PublishRequest> first, second, replies;
        responder.On<Receive>("rpc/restart", [&](const auto &event) { first.push_back(event.m_Message); });
        other.On<Receive>("rpc/restart", [&](const auto &event) { second.push_back(event.m_Message); });
        requester.On<Receive>("rpc/restart", [&](const auto &event) { replies.push_back(event.m_Message); });
        Accepted(requester.Request("rpc/restart", {'A'}, 0, 1, 51));
        Accepted(requester.Request("rpc/restart", {'B'}, 2, 1, 52));
        Await([&] { return first.size() == 2 && second.size() == 2; }, {&responder, &other});
        Accepted(responder.Respond(first[0], {'A', '1'}));
        Await([&] { return replies.size() == 1 && broker->Metrics().m_Pending == 0; }, {&requester});
        requester.Disconnect();
        responder.Disconnect();
        other.Disconnect();
        Await([&] { return broker->Metrics().m_Online == 0; });
        broker.reset();
        broker = std::make_unique<Broker>(settings);
        port = broker->Listen();
        EMPY_REQUIRE(requester.Connect("127.0.0.1", port, true) == requesterId);
        EMPY_REQUIRE(responder.Connect("127.0.0.1", port, true) == responderId);
        EMPY_REQUIRE(other.Connect("127.0.0.1", port, true) == otherId);
        EMPY_REQUIRE(responder.Respond(first[0], {'X'}).get().m_Code == MMQP_REASON_INVALID_CORRELATION);
        Accepted(other.Respond(second[1], {'B', '2'}));
        Accepted(responder.Respond(first[1], {'B', '1'}));
        Accepted(other.Respond(second[0], {'A', '2'}));
        Await([&] { return replies.size() == 4; }, {&requester});
        std::map<uint16_t, size_t> counts;
        for (const auto &reply : replies)
        {
            ++counts[reply.m_CorrelationId];
            EMPY_REQUIRE(reply.m_CorrelationId == (reply.m_Payload.Span().front() == 'A' ? 51 : 52));
        }
        EMPY_REQUIRE(counts[51] == 2 && counts[52] == 2);
        Throws([&] { requester.Request("rpc/restart", {}, 1, 1, 51).get(); });
        Accepted(requester.Request("rpc/restart", {'C'}, 0));
        Await([&] { return first.size() == 3 && second.size() == 3; }, {&responder, &other});
        EMPY_REQUIRE(first[2].m_CorrelationId != first[0].m_CorrelationId &&
                     first[2].m_CorrelationId != first[1].m_CorrelationId);
        Accepted(responder.Respond(first[2], {'C', '1'}));
        Accepted(other.Respond(second[2], {'C', '2'}));
        Await([&] { return replies.size() == 6; }, {&requester});
        EMPY_REQUIRE(replies[4].m_CorrelationId == 1 && replies[5].m_CorrelationId == 1);
    }
} // namespace

EMPY_TEST(Correlation, ReferenceCases)
{
    Concurrent(false);
    Concurrent(true);
    Deadlines();
    Recovery();
}

EMPY_TEST(Correlation, ResponderErrorIsOneOutcome)
{
    eMQ::Server::Configs settings;
    settings.m_Port = 0;
    Broker broker(settings);
    const auto port = broker.Listen();
    Client requester, first, second;
    requester.Connect("127.0.0.1", port);
    first.Connect("127.0.0.1", port);
    second.Connect("127.0.0.1", port);
    Accepted(requester.Register("rpc/outcomes", 0, 60, 1, true));
    Accepted(first.Subscribe("rpc/outcomes", 1, true));
    Accepted(second.Subscribe("rpc/outcomes", 1, true));
    std::vector<PublishRequest> firstRequests, secondRequests, responses;
    first.On<Receive>("rpc/outcomes", [&](const auto &event) { firstRequests.push_back(event.m_Message); });
    second.On<Receive>("rpc/outcomes", [&](const auto &event) { secondRequests.push_back(event.m_Message); });
    requester.On<Receive>("rpc/outcomes", [&](const auto &event) { responses.push_back(event.m_Message); });
    Accepted(requester.Request("rpc/outcomes", {'Q'}, 2, 1, 9));
    Await([&] { return firstRequests.size() == 1 && secondRequests.size() == 1; }, {&first, &second});
    Accepted(first.Respond(firstRequests.front(), {'E'}, {MMQP_REASON_BUSY, "Application busy"}));
    Await([&] { return responses.size() == 1; }, {&requester});
    Accepted(second.Respond(secondRequests.front(), {'S'}));
    Await([&] { return responses.size() == 2; }, {&requester});
    EMPY_REQUIRE(responses[0].m_Reason.m_Code == MMQP_REASON_BUSY &&
                 responses[1].m_Reason.m_Code == MMQP_REASON_SUCCESS);
    EMPY_REQUIRE(requester.Connected());
    Accepted(requester.Request("rpc/outcomes", {'R'}, 2, 1, 9));
}
