#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Fabric/Topics.h>
#include <eOps/Fabric/Envelope.h>
#include <eOps/Fabric/Bus.h>
#include <eOps/Common/Text.h>
#include <Runtime/Runtime.h>
#include <cmath>
#include <memory>
#include <set>
#include <thread>

using namespace eOps;
using namespace eOps::Fabric;

namespace
{
    using eTest::Ops::LocalBroker;
    using eTest::Ops::Peer;

    using Clock = std::chrono::steady_clock;

    int64_t ElapsedMs(Clock::time_point start)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }

    eCore::SharedBytes Bytes(std::string_view text)
    {
        return eCore::SharedBytes::Copy(text);
    }

    // poll every peer until `done` holds or `timeoutMs` elapses
    template <typename _Predicate>
    bool Pump(std::initializer_list<Peer *> peers, _Predicate &&done, uint32_t timeoutMs = 5000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true)
        {
            for (auto *peer : peers)
            {
                peer->m_Client->Poll(1);
            }
            if (done())
            {
                return true;
            }
            if (Clock::now() >= deadline)
            {
                return false;
            }
        }
    }

    void Accepted(std::future<eMQ::Reason> result)
    {
        EMPY_REQUIRE(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        auto reason = result.get();
        EMPY_REQUIRE(reason.m_Code < 0x80);
    }

    // records a module's attach/detach and the messages its owned listener received
    struct Probe : eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
        std::string m_Filter;
        std::vector<std::string> *m_Seen = nullptr;

        Probe(std::string filter, std::vector<std::string> *seen) : m_Filter(std::move(filter)), m_Seen(seen)
        {
        }

        void OnAttach(eMQ::Service *service, eMQ::Context *) override
        {
            auto reason = BusOf(*service).Listen(
                m_Filter, [this](const std::string &topic, const Envelope &) { m_Seen->push_back(topic); }, 1, this);
            if (reason.m_Code >= 0x80)
            {
                throw std::runtime_error("Probe listen failed: " + reason.m_Text);
            }
        }

        void OnUpdate(eMQ::Service *, eMQ::Context *) override
        {
        }

        void OnDetach(eMQ::Service *service, eMQ::Context *) override
        {
            BusOf(*service).Remove(this);
        }
    };
} // namespace

EMPY_TEST(OpsFabric, Topics)
{
    Topics topics;
    EMPY_CHECK_EQ(topics.Intent(), std::string("mlops/intent"));
    EMPY_CHECK_EQ(topics.Deploy(), std::string("mlops/deploy"));
    EMPY_CHECK_EQ(topics.Workflow(), std::string("mlops/harness/workflow"));
    EMPY_CHECK_EQ(topics.Status(), std::string("mlops/harness/status"));
    EMPY_CHECK_EQ(topics.Agent("executive", "interpret"), std::string("mlops/agent/executive/interpret"));
    EMPY_CHECK_EQ(topics.AgentFilter("placement"), std::string("mlops/agent/placement/+"));
    EMPY_CHECK_EQ(topics.Trigger("home-control", "training"), std::string("mlops/trigger/time/home-control/training"));
    EMPY_CHECK_EQ(topics.TriggerFilter("training"), std::string("mlops/trigger/time/+/training"));
    EMPY_CHECK_EQ(topics.Event("home-control", "drift"), std::string("mlops/event/home-control/drift"));
    EMPY_CHECK_EQ(topics.EventFilter("drift"), std::string("mlops/event/+/drift"));
    EMPY_CHECK_EQ(topics.Heartbeat("home-control", "inf-0"), std::string("mlops/status/home-control/inf-0"));
    EMPY_CHECK_EQ(topics.HeartbeatFilter(), std::string("mlops/status/+/+"));
    EMPY_CHECK_EQ(topics.Workload("home-control", "train"), std::string("mlops/workload/home-control/train"));
    EMPY_CHECK_EQ(topics.WorkloadFilter("home-control"), std::string("mlops/workload/home-control/+"));
    EMPY_CHECK_EQ(topics.Candidate("home-control"), std::string("mlops/model/home-control/candidate"));
    EMPY_CHECK_EQ(topics.CandidateFilter(), std::string("mlops/model/+/candidate"));
    EMPY_CHECK_EQ(topics.Hitl(), std::string("mlops/hitl/approval"));
    EMPY_CHECK_EQ(topics.State("services"), std::string("mlops/state/services"));
    EMPY_CHECK_EQ(topics.Locality("ran"), std::string("mlops/telemetry/locality/ran"));
    EMPY_CHECK_EQ(topics.LocalityFilter(), std::string("mlops/telemetry/locality/+"));
    EMPY_CHECK_EQ(topics.All(), std::string("mlops/#"));

    Topics custom("lab/ml//");
    EMPY_CHECK_EQ(custom.m_Root, std::string("lab/ml"));
    EMPY_CHECK_EQ(custom.Intent(), std::string("lab/ml/intent"));
    auto relative = custom.Relative("lab/ml/event/home-control/drift");
    EMPY_REQUIRE(relative.has_value());
    EMPY_CHECK(*relative == std::vector<std::string>({"event", "home-control", "drift"}));
    EMPY_CHECK(custom.Relative("lab/ml")->empty());
    EMPY_CHECK(!custom.Relative("lab/mlx/event").has_value());
    EMPY_CHECK(!custom.Relative("lab").has_value());
    EMPY_CHECK(!custom.Relative("other/intent").has_value());

    for (const char *root : {"", "/", "a/+", "a/#", "$sys", "a//b", "+"})
    {
        EMPY_CHECK_THROWS(Topics(root));
    }
    for (std::string_view level : {std::string_view(""), std::string_view("a/b"), std::string_view("+"),
                                   std::string_view("#"), std::string_view("x#")})
    {
        EMPY_CHECK_THROWS(topics.Event(level, "drift"));
        EMPY_CHECK_THROWS(topics.Agent("executive", level));
    }
    EMPY_CHECK_THROWS(topics.State(std::string(300, 'x')));
}

EMPY_TEST(OpsFabric, EnvelopeRoundTrip)
{
    auto made =
        Envelope::Make("intent", Json::Object{{"text", "keep the home at 21 degrees"}, {"priority", 2}}, "operator");
    EMPY_CHECK(made.m_Id.starts_with("intent-"));
    EMPY_CHECK(made.m_Time > 0);
    auto decoded = Envelope::Decode(made.Encode());
    EMPY_CHECK_EQ(decoded.m_Kind, std::string("intent"));
    EMPY_CHECK_EQ(decoded.m_Id, made.m_Id);
    EMPY_CHECK_EQ(decoded.m_Time, made.m_Time);
    EMPY_CHECK_EQ(decoded.m_Source, std::string("operator"));
    EMPY_CHECK(decoded.m_Body == made.m_Body);

    // wire field names and defaults applied when encoding
    Envelope bare;
    bare.m_Kind = "status-request";
    auto json = bare.ToJson();
    EMPY_CHECK(json.IsObject() && json.Size() == 5);
    EMPY_CHECK(json.GetString("id").starts_with("status-request-"));
    EMPY_CHECK(json["time"].AsUint() > 0);
    EMPY_CHECK(json["body"].IsNull());
    EMPY_CHECK_EQ(json.GetString("source"), std::string());

    // optional members may be absent
    auto minimal = Envelope::FromJson(Json::Parse(R"({"kind":"trigger"})"));
    EMPY_CHECK_EQ(minimal.m_Kind, std::string("trigger"));
    EMPY_CHECK(minimal.m_Id.empty() && minimal.m_Time == 0 && minimal.m_Source.empty() && minimal.m_Body.IsNull());
    auto nested = Envelope::FromJson(Json::Parse(R"({"kind":"k","id":null,"body":{"a":[1,2,{"b":true}]}})"));
    EMPY_CHECK(nested.m_Body.At("a.2.b") && nested.m_Body.At("a.2.b")->AsBool());

    Envelope empty;
    EMPY_CHECK_THROWS(empty.ToJson());
    EMPY_CHECK_THROWS(empty.Encode());
}

EMPY_TEST(OpsFabric, EnvelopeMalformed)
{
    for (const char *text : {"", "not json", "[1,2]", "42", R"({"id":"x"})", R"({"kind":""})", R"({"kind":5})",
                             R"({"kind":"k","time":-1})", R"({"kind":"k","time":"now"})", R"({"kind":"k","time":1.5})",
                             R"({"kind":"k","id":7})", R"({"kind":"k","source":[]})", R"({"kind":"k",})",
                             "{\"kind\":\"k\",\"body\":\"\xff\"}"})
    {
        bool threw = false;
        try
        {
            Envelope::Decode(Bytes(text));
        }
        catch (const JsonError &)
        {
            threw = true;
        }
        EMPY_CHECK(threw);
    }
    // parser limits apply
    std::string deep = R"({"kind":"k","body":)" + std::string(40, '[') + std::string(40, ']') + "}";
    EMPY_CHECK_THROWS(Envelope::Decode(Bytes(deep), JsonLimits{16, 1024}));
    EMPY_CHECK_THROWS(Envelope::Decode(Bytes(R"({"kind":"k","body":"0123456789"})"), JsonLimits{64, 16}));
    EMPY_CHECK_EQ(Envelope::Decode(Bytes(deep)).m_Kind, std::string("k"));
}

EMPY_TEST(OpsFabric, ParseValue)
{
    auto value = [](std::string_view text) { return ParseValue(Bytes(text)); };
    EMPY_CHECK(value("21.5") == 21.5);
    EMPY_CHECK(value(" 21.5\n") == 21.5);
    EMPY_CHECK(value("-3") == -3.0);
    EMPY_CHECK(value("1e2") == 100.0);
    EMPY_CHECK(value("0") == 0.0);
    EMPY_CHECK(value(R"({"value":4.25,"unit":"celsius"})") == 4.25);
    EMPY_CHECK(value(R"({"value":-7})") == -7.0);
    EMPY_CHECK(value("  {\"value\": 3}  ") == 3.0);
    for (std::string_view bad :
         {std::string_view(""), std::string_view("abc"), std::string_view("nan"), std::string_view("inf"),
          std::string_view("1e999"), std::string_view("+1"), std::string_view("[1]"),
          std::string_view(R"({"value":"4"})"), std::string_view(R"({"value":null})"), std::string_view(R"({"v":1})"),
          std::string_view("true"), std::string_view("12 13")})
    {
        EMPY_CHECK(!value(bad).has_value());
    }

    auto action = EncodeAction(21.5, "celsius", "home-control", "inf-0", 3, 1700000000000ull);
    auto json = FromBytes(action);
    EMPY_CHECK_EQ(json.Dump(), std::string(R"({"value":21.5,"unit":"celsius","service":"home-control",)"
                                           R"("instance":"inf-0","model":3,"time":1700000000000})"));
    EMPY_CHECK(ParseValue(action) == 21.5);
    auto stamped = FromBytes(EncodeAction(1, "", "s", "i", 0));
    EMPY_CHECK(stamped["time"].AsUint() > 0);
    EMPY_CHECK_THROWS(EncodeAction(std::nan(""), "u", "s", "i", 1));
}

EMPY_TEST(OpsFabric, ListenWildcard)
{
    LocalBroker fabric;
    Peer publisher(fabric.m_Port, "publisher"), listener(fabric.m_Port, "listener");
    Topics topics;
    std::vector<std::pair<std::string, Envelope>> seen;
    std::vector<double> values;
    const auto owner = std::this_thread::get_id();
    bool threads = true;
    EMPY_REQUIRE(listener.Get()
                     .Listen(topics.EventFilter("drift"),
                             [&](const std::string &topic, const Envelope &envelope)
                             {
                                 threads = threads && std::this_thread::get_id() == owner;
                                 seen.emplace_back(topic, envelope);
                             })
                     .m_Code < 0x80);
    EMPY_REQUIRE(listener.Get()
                     .ListenRaw("home/sense/+",
                                [&](const eMQ::PublishRequest &message)
                                {
                                    if (auto parsed = ParseValue(message.m_Payload))
                                    {
                                        values.push_back(*parsed);
                                    }
                                })
                     .m_Code < 0x80);
    // a second listener on the same filter shares the subscription
    size_t second = 0;
    EMPY_REQUIRE(listener.Get()
                     .Listen(topics.EventFilter("drift"), [&](const std::string &, const Envelope &) { ++second; })
                     .m_Code < 0x80);

    auto &bus = publisher.Get();
    Accepted(bus.Publish(topics.Event("svc-a", "drift"), Envelope::Make("event", Json::Object{{"value", 0.4}})));
    Accepted(bus.Publish(topics.Event("svc-b", "drift"), Envelope::Make("event", Json::Object{{"value", 0.3}})));
    Accepted(bus.Publish(topics.Event("svc-a", "metrics"), Envelope::Make("event", Json::Object{})));
    Accepted(bus.PublishRaw("home/sense/temp", Bytes("21.5")));
    Accepted(bus.PublishRaw("home/sense/humidity", Bytes(R"({"value":48})")));
    Accepted(bus.PublishRaw("home/other/temp", Bytes("1")));
    // a malformed envelope is dropped without disturbing later deliveries
    Accepted(bus.PublishRaw(topics.Event("svc-c", "drift"), Bytes("{broken"), 1));
    Accepted(bus.Publish(topics.Event("svc-d", "drift"), Envelope::Make("event", Json::Object{{"value", 0.1}})));
    // publishing to a filter is rejected locally
    EMPY_CHECK_EQ(bus.Publish(topics.EventFilter("drift"), Envelope::Make("event", {})).get().m_Code,
                  eMQ::MMQP_REASON_INVALID_TOPIC);

    EMPY_REQUIRE(Pump({&listener}, [&] { return seen.size() == 3 && values.size() == 2; }));
    Pump({&listener}, [] { return false; }, 100);
    EMPY_CHECK_EQ(seen.size(), size_t(3));
    EMPY_CHECK_EQ(second, size_t(3));
    EMPY_CHECK(threads);
    EMPY_CHECK_EQ(seen[0].first, std::string("mlops/event/svc-a/drift"));
    EMPY_CHECK_EQ(seen[1].first, std::string("mlops/event/svc-b/drift"));
    EMPY_CHECK_EQ(seen[2].first, std::string("mlops/event/svc-d/drift"));
    EMPY_CHECK_EQ(seen[0].second.m_Source, std::string("publisher"));
    EMPY_CHECK_EQ(seen[0].second.m_Body.GetNumber("value"), 0.4);
    EMPY_CHECK(values == std::vector<double>({21.5, 48.0}));

    EMPY_CHECK_EQ(listener.Get().Listen("a/#/b", [](const std::string &, const Envelope &) {}).m_Code,
                  eMQ::MMQP_REASON_INVALID_TOPIC);
}

EMPY_TEST(OpsFabric, ServeCallRoundTrip)
{
    LocalBroker fabric;
    Peer caller(fabric.m_Port, "harness"), agent(fabric.m_Port, "executive");
    Topics topics;
    std::vector<Inbound> requests;
    std::vector<Bus::Responder> deferred;
    EMPY_REQUIRE(agent.Get()
                     .Serve(topics.AgentFilter("executive"),
                            [&](const Inbound &request, Bus::Responder respond)
                            {
                                requests.push_back(request);
                                const auto &task = request.m_Topic;
                                if (task.ends_with("/interpret"))
                                {
                                    respond(Envelope::Make("agent-response",
                                                           Json::Object{{"result", request.m_Envelope.m_Body}}),
                                            {});
                                }
                                else if (task.ends_with("/reject"))
                                {
                                    respond({}, {0x80, "intent is ambiguous"});
                                }
                                else
                                {
                                    deferred.push_back(std::move(respond)); // answered after more polling
                                }
                            })
                     .m_Code < 0x80);

    auto &bus = caller.Get();
    std::optional<std::pair<eMQ::Reason, std::optional<Envelope>>> reply;
    bus.Call(topics.Agent("executive", "interpret"), Envelope::Make("agent-request", Json::Object{{"intent", "x"}}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_CHECK_EQ(bus.Pending(), size_t(1));
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE(reply->second.has_value());
    EMPY_CHECK_EQ(reply->second->m_Kind, std::string("agent-response"));
    EMPY_CHECK_EQ(reply->second->m_Source, std::string("executive"));
    EMPY_CHECK_EQ(reply->second->m_Body["result"].GetString("intent"), std::string("x"));
    EMPY_CHECK_EQ(bus.Pending(), size_t(0));
    EMPY_REQUIRE_EQ(requests.size(), size_t(1));
    EMPY_CHECK_EQ(requests[0].m_Topic, std::string("mlops/agent/executive/interpret"));
    EMPY_CHECK_EQ(requests[0].m_Envelope.m_Source, std::string("harness"));
    EMPY_CHECK(requests[0].m_CorrelationId != 0);

    // an application error reaches the caller with its reason and error envelope
    reply.reset();
    bus.Call(topics.Agent("executive", "reject"), Envelope::Make("agent-request", {}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, uint8_t(0x80));
    EMPY_CHECK_EQ(reply->first.m_Text, std::string("intent is ambiguous"));
    EMPY_REQUIRE(reply->second.has_value());
    EMPY_CHECK_EQ(reply->second->m_Kind, std::string("error"));
    EMPY_CHECK_EQ(reply->second->m_Body.GetString("text"), std::string("intent is ambiguous"));

    // a deferred answer (e.g. after Async work) through the stored Responder
    reply.reset();
    bus.Call(topics.Agent("executive", "decompose"), Envelope::Make("agent-request", {}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return deferred.size() == 1; }));
    Pump({&caller, &agent}, [] { return false; }, 50);
    EMPY_CHECK(!reply.has_value());
    deferred.front()(Envelope::Make("agent-response", Json::Object{{"result", "later"}}), {});
    deferred.front()(Envelope::Make("agent-response", {}), {}); // a second answer is ignored
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_CHECK_EQ(reply->second->m_Body.GetString("result"), std::string("later"));

    // a Responder dropped without answering answers with an error
    reply.reset();
    deferred.clear();
    bus.Call(topics.Agent("executive", "plan"), Envelope::Make("agent-request", {}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return deferred.size() == 1; }));
    deferred.clear();
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, eMQ::MMQP_REASON_INTERNAL_ERROR);

    // the CLI helper polls the runtimes while it waits
    auto [reason, envelope] =
        bus.CallSync(topics.Agent("executive", "interpret"), Envelope::Make("agent-request", Json::Object{{"n", 1}}),
                     [&]
                     {
                         caller.m_Client->Poll(1);
                         agent.m_Client->Poll(1);
                     });
    EMPY_CHECK_EQ(reason.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE(envelope.has_value());
    EMPY_CHECK_EQ(envelope->m_Body["result"].GetInt("n"), int64_t(1));
}

EMPY_TEST(OpsFabric, SelfRequest)
{
    // one runtime hosting both the requester and the responder (EmpyHarness run with agent roles)
    LocalBroker fabric;
    Peer host(fabric.m_Port, "harness");
    Topics topics;
    auto &bus = host.Get();
    EMPY_REQUIRE(bus.Serve(topics.AgentFilter("executive"), [](const Inbound &request, Bus::Responder respond)
                           { respond(Envelope::Make("agent-response", request.m_Envelope.m_Body), {}); })
                     .m_Code < 0x80);
    std::optional<std::pair<eMQ::Reason, std::optional<Envelope>>> reply;
    bus.Call(topics.Agent("executive", "interpret"), Envelope::Make("agent-request", Json::Object{{"n", 7}}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&host}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE(reply->second.has_value());
    EMPY_CHECK_EQ(reply->second->m_Body.GetInt("n"), int64_t(7));
    EMPY_CHECK_EQ(reply->second->m_Source, std::string("harness"));

    // a non-canonical request topic is matched with its canonical response topic
    reply.reset();
    bus.Call("mlops//agent/executive/plan/", Envelope::Make("agent-request", Json::Object{{"n", 8}}),
             [&](eMQ::Reason reason, std::optional<Envelope> envelope) { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&host}, [&] { return reply.has_value(); }));
    EMPY_CHECK_EQ(reply->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE(reply->second.has_value());
    EMPY_CHECK_EQ(reply->second->m_Body.GetInt("n"), int64_t(8));
}

EMPY_TEST(OpsFabric, ServeRemovalKeepsListener)
{
    LocalBroker fabric;
    Peer caller(fabric.m_Port, "harness"), agent(fabric.m_Port, "placement");
    Topics topics;
    int serveTag = 0, listenTag = 0;
    size_t served = 0;
    std::vector<std::string> seen;
    auto &bus = agent.Get();
    EMPY_REQUIRE(bus.Listen(
                        topics.AgentFilter("placement"),
                        [&](const std::string &topic, const Envelope &) { seen.push_back(topic); }, 1, &listenTag)
                     .m_Code < 0x80);
    EMPY_REQUIRE(bus.Serve(
                        topics.AgentFilter("placement"),
                        [&](const Inbound &, Bus::Responder respond)
                        {
                            ++served;
                            respond(Envelope::Make("agent-response", {}), {});
                        },
                        &serveTag)
                     .m_Code < 0x80);
    std::optional<eMQ::Reason> outcome;
    bool answered = false; // an envelope came back (a broker terminal outcome has none)
    auto call = [&]
    {
        outcome.reset();
        caller.Get().Call(topics.Agent("placement", "place"), Envelope::Make("agent-request", {}),
                          [&](eMQ::Reason reason, std::optional<Envelope> envelope)
                          {
                              outcome = reason;
                              answered = envelope.has_value();
                          });
        return Pump({&caller, &agent}, [&] { return outcome.has_value(); });
    };
    EMPY_REQUIRE(call());
    EMPY_CHECK_EQ(outcome->m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_CHECK_EQ(served, size_t(1));
    EMPY_CHECK(answered);

    // without a server the shared subscription stops being a responder: the broker reports
    // NO_RESPONDER instead of routing the request to a session that cannot answer it
    bus.Remove(&serveTag);
    Pump({&caller, &agent}, [] { return false; }, 200);
    EMPY_REQUIRE(call());
    EMPY_CHECK_EQ(outcome->m_Code, eMQ::MMQP_REASON_NO_RESPONDER);
    EMPY_CHECK(!answered);
    EMPY_CHECK_EQ(served, size_t(1));

    // the listener still receives plain publications on the filter
    Accepted(caller.Get().Publish(topics.Agent("placement", "audit"), Envelope::Make("trigger", {})));
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return seen.size() == 1; }));
    EMPY_CHECK_EQ(seen[0], std::string("mlops/agent/placement/audit"));
    bus.Remove(&listenTag);
}

EMPY_TEST(OpsFabric, GatherTwoResponders)
{
    LocalBroker fabric;
    Peer harness(fabric.m_Port, "harness"), first(fabric.m_Port, "inf-0"), second(fabric.m_Port, "inf-1");
    Topics topics;
    for (auto *peer : {&first, &second})
    {
        auto name = peer->Get().Options().m_Source;
        auto served = peer->Get().Serve(topics.WorkloadFilter("home-control"),
                                        [name](const Inbound &request, Bus::Responder respond)
                                        {
                                            auto action = request.m_Envelope.m_Body.GetString("action");
                                            Json body = Json::Object{{"instance", name}, {"action", action}};
                                            respond(Envelope::Make("model", std::move(body)), {});
                                        });
        EMPY_REQUIRE(served.m_Code < 0x80);
    }
    auto &bus = harness.Get();
    std::optional<std::pair<eMQ::Reason, std::vector<Envelope>>> result;
    auto start = Clock::now();
    bus.Gather(
        topics.Workload("home-control", "model"), Envelope::Make("model", Json::Object{{"action", "load"}}), 2,
        [&](eMQ::Reason reason, std::vector<Envelope> replies) { result.emplace(reason, std::move(replies)); }, 4000);
    EMPY_REQUIRE(Pump({&harness, &first, &second}, [&] { return result.has_value(); }));
    EMPY_CHECK(ElapsedMs(start) < 3000); // completes on the last expected reply, not at the timeout
    EMPY_CHECK_EQ(result->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE_EQ(result->second.size(), size_t(2));
    std::set<std::string> instances;
    for (const auto &reply : result->second)
    {
        instances.insert(reply.m_Body.GetString("instance"));
        EMPY_CHECK_EQ(reply.m_Body.GetString("action"), std::string("load"));
    }
    EMPY_CHECK(instances == std::set<std::string>({"inf-0", "inf-1"}));

    // 0 = every responder until the timeout
    result.reset();
    start = Clock::now();
    bus.Gather(
        topics.Workload("home-control", "model"), Envelope::Make("model", Json::Object{{"action", "status"}}), 0,
        [&](eMQ::Reason reason, std::vector<Envelope> replies) { result.emplace(reason, std::move(replies)); }, 600);
    EMPY_REQUIRE(Pump({&harness, &first, &second}, [&] { return result.has_value(); }));
    EMPY_CHECK(ElapsedMs(start) >= 550);
    EMPY_CHECK_EQ(result->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_CHECK_EQ(result->second.size(), size_t(2));

    // more expected replies than responders: the timeout reports what arrived
    result.reset();
    bus.Gather(
        topics.Workload("home-control", "model"), Envelope::Make("model", {}), 3,
        [&](eMQ::Reason reason, std::vector<Envelope> replies) { result.emplace(reason, std::move(replies)); }, 500);
    EMPY_REQUIRE(Pump({&harness, &first, &second}, [&] { return result.has_value(); }));
    EMPY_CHECK_EQ(result->first.m_Code, eMQ::MMQP_REASON_TIMEOUT);
    EMPY_CHECK_EQ(result->second.size(), size_t(2));
    EMPY_CHECK_EQ(bus.Pending(), size_t(0));
}

// a responder's application error is that responder's reply, not the end of the whole gather (eMQ::Service
// semantics): the other responders still count, in whatever order the answers arrive
EMPY_TEST(OpsFabric, GatherResponderError)
{
    LocalBroker fabric;
    Peer harness(fabric.m_Port, "harness"), failing(fabric.m_Port, "inf-old"), healthy(fabric.m_Port, "inf-new");
    Topics topics;
    std::vector<Bus::Responder> parked;
    EMPY_REQUIRE(failing.Get()
                     .Serve(topics.WorkloadFilter("home-control"), [](const Inbound &, Bus::Responder respond)
                            { respond({}, {eMQ::MMQP_REASON_POLICY_VIOLATION, "Artifact rejected: features"}); })
                     .m_Code < 0x80);
    EMPY_REQUIRE(healthy.Get()
                     .Serve(topics.WorkloadFilter("home-control"),
                            [&](const Inbound &, Bus::Responder respond) { parked.push_back(std::move(respond)); })
                     .m_Code < 0x80);
    auto &bus = harness.Get();
    std::optional<std::pair<eMQ::Reason, std::vector<Envelope>>> result;
    bus.Gather(
        topics.Workload("home-control", "model"), Envelope::Make("model", Json::Object{{"action", "load"}}), 2,
        [&](eMQ::Reason reason, std::vector<Envelope> replies) { result.emplace(reason, std::move(replies)); }, 4000);
    // the error arrives first; the gather waits for the healthy responder
    EMPY_REQUIRE(Pump({&harness, &failing, &healthy}, [&] { return parked.size() == 1; }));
    Pump({&harness, &failing, &healthy}, [] { return false; }, 150);
    EMPY_CHECK(!result.has_value());
    parked.front()(Envelope::Make("model", Json::Object{{"loaded", 2}}), {});
    EMPY_REQUIRE(Pump({&harness, &failing, &healthy}, [&] { return result.has_value(); }));
    EMPY_CHECK_EQ(result->first.m_Code, eMQ::MMQP_REASON_SUCCESS);
    EMPY_REQUIRE_EQ(result->second.size(), size_t(2));
    size_t errors = 0, loaded = 0;
    for (const Envelope &reply : result->second)
    {
        if (reply.m_Kind == "error")
        {
            ++errors;
            EMPY_CHECK_EQ(reply.m_Source, std::string("inf-old"));
            EMPY_CHECK_EQ(reply.m_Body.GetInt("code"), int64_t(eMQ::MMQP_REASON_POLICY_VIOLATION));
            EMPY_CHECK(reply.m_Body.GetString("text").find("features") != std::string::npos);
        }
        else
        {
            loaded += reply.m_Body.GetInt("loaded") == 2 ? 1 : 0;
        }
    }
    EMPY_CHECK_EQ(errors, size_t(1));
    EMPY_CHECK_EQ(loaded, size_t(1));

    // every responder failing: the gather reports the responder error with every error reply
    result.reset();
    parked.clear();
    bus.Gather(
        topics.Workload("home-control", "model"), Envelope::Make("model", Json::Object{{"action", "load"}}), 2,
        [&](eMQ::Reason reason, std::vector<Envelope> replies) { result.emplace(reason, std::move(replies)); }, 4000);
    EMPY_REQUIRE(Pump({&harness, &failing, &healthy}, [&] { return parked.size() == 1; }));
    parked.front()({}, {eMQ::MMQP_REASON_UNSPECIFIED, "no model"});
    EMPY_REQUIRE(Pump({&harness, &failing, &healthy}, [&] { return result.has_value(); }));
    EMPY_CHECK(result->first.m_Code >= 0x80);
    EMPY_CHECK_EQ(result->second.size(), size_t(2));
    EMPY_CHECK_EQ(bus.Pending(), size_t(0));
}

EMPY_TEST(OpsFabric, CallTimeout)
{
    LocalBroker fabric;
    Peer caller(fabric.m_Port, "harness"), silent(fabric.m_Port, "lifecycle");
    Topics topics;
    std::vector<Bus::Responder> held;
    EMPY_REQUIRE(silent.Get()
                     .Serve(topics.AgentFilter("lifecycle"),
                            [&](const Inbound &, Bus::Responder respond) { held.push_back(std::move(respond)); })
                     .m_Code < 0x80);
    auto &bus = caller.Get();
    size_t calls = 0;
    eMQ::Reason outcome;
    auto start = Clock::now();
    int64_t elapsed = 0;
    bus.Call(
        topics.Agent("lifecycle", "evaluate"), Envelope::Make("agent-request", {}),
        [&](eMQ::Reason reason, std::optional<Envelope> envelope)
        {
            ++calls;
            outcome = reason;
            elapsed = ElapsedMs(start);
            EMPY_CHECK(!envelope.has_value());
        },
        300);
    EMPY_REQUIRE(Pump({&caller, &silent}, [&] { return calls == 1; }));
    EMPY_CHECK_EQ(outcome.m_Code, eMQ::MMQP_REASON_TIMEOUT);
    EMPY_CHECK(elapsed >= 290 && elapsed < 2000);
    EMPY_CHECK_EQ(bus.Pending(), size_t(0));

    // the late answer is ignored and the next exchange is unaffected
    EMPY_REQUIRE_EQ(held.size(), size_t(1));
    held.front()(Envelope::Make("agent-response", {}), {});
    Pump({&caller, &silent}, [] { return false; }, 200);
    EMPY_CHECK_EQ(calls, size_t(1));
    std::optional<eMQ::Reason> next;
    bus.Call(
        topics.Agent("lifecycle", "evaluate"), Envelope::Make("agent-request", {}),
        [&](eMQ::Reason reason, std::optional<Envelope>) { next = reason; }, 3000);
    EMPY_REQUIRE(Pump({&caller, &silent}, [&] { return held.size() == 2; }));
    held.back()(Envelope::Make("agent-response", {}), {});
    EMPY_REQUIRE(Pump({&caller, &silent}, [&] { return next.has_value(); }));
    EMPY_CHECK_EQ(next->m_Code, eMQ::MMQP_REASON_SUCCESS);
}

EMPY_TEST(OpsFabric, NoResponder)
{
    LocalBroker fabric;
    Peer caller(fabric.m_Port, "harness");
    Topics topics;
    std::optional<eMQ::Reason> outcome;
    auto start = Clock::now();
    caller.Get().Call(topics.Hitl(), Envelope::Make("hitl", Json::Object{{"question", "approve?"}}),
                      [&](eMQ::Reason reason, std::optional<Envelope> envelope)
                      {
                          outcome = reason;
                          EMPY_CHECK(!envelope.has_value());
                      });
    EMPY_REQUIRE(Pump({&caller}, [&] { return outcome.has_value(); }));
    EMPY_CHECK_EQ(outcome->m_Code, eMQ::MMQP_REASON_NO_RESPONDER);
    EMPY_CHECK(ElapsedMs(start) < 2000);

    // Gather reports it the same way; a filter is not a request topic
    std::optional<eMQ::Reason> gathered;
    caller.Get().Gather(topics.Workload("nobody", "model"), Envelope::Make("model", {}), 2,
                        [&](eMQ::Reason reason, std::vector<Envelope> replies)
                        {
                            gathered = reason;
                            EMPY_CHECK(replies.empty());
                        });
    EMPY_REQUIRE(Pump({&caller}, [&] { return gathered.has_value(); }));
    EMPY_CHECK_EQ(gathered->m_Code, eMQ::MMQP_REASON_NO_RESPONDER);
    std::optional<eMQ::Reason> invalid;
    caller.Get().Call(topics.AgentFilter("executive"), Envelope::Make("agent-request", {}),
                      [&](eMQ::Reason reason, std::optional<Envelope>) { invalid = reason; });
    EMPY_REQUIRE(invalid.has_value()); // rejected before anything is sent
    EMPY_CHECK_EQ(invalid->m_Code, eMQ::MMQP_REASON_INVALID_TOPIC);
}

EMPY_TEST(OpsFabric, MalformedRequest)
{
    LocalBroker fabric;
    Peer agent(fabric.m_Port, "compliance");
    Topics topics;
    size_t handled = 0;
    EMPY_REQUIRE(agent.Get()
                     .Serve(topics.AgentFilter("compliance"),
                            [&](const Inbound &, Bus::Responder respond)
                            {
                                ++handled;
                                respond(Envelope::Make("agent-response", {}), {});
                            })
                     .m_Code < 0x80);
    // a plain client sends a payload that is not an envelope
    eMQ::Client raw;
    raw.Connect("127.0.0.1", fabric.m_Port);
    std::vector<eMQ::PublishRequest> replies;
    raw.On<eMQ::Receive>([&](const eMQ::Receive &event) { replies.push_back(event.m_Message); });
    Accepted(raw.Register(topics.AgentFilter("compliance"), 0, 60, 1, true));
    Accepted(raw.Request(topics.Agent("compliance", "review"), Bytes("{not an envelope"), 1, 1, 7));
    auto pumpRaw = [&](auto &&done)
    {
        auto deadline = Clock::now() + std::chrono::seconds(5);
        while (!done() && Clock::now() < deadline)
        {
            agent.m_Client->Poll(1);
            raw.Poll(1);
        }
        return done();
    };
    EMPY_REQUIRE(pumpRaw([&] { return replies.size() == 1; }));
    EMPY_CHECK_EQ(handled, size_t(0));
    EMPY_CHECK_EQ(replies[0].m_Type, eMQ::MMQP_PUBREQ_TYPE_RES);
    EMPY_CHECK_EQ(replies[0].m_CorrelationId, uint16_t(7));
    EMPY_CHECK_EQ(replies[0].m_Reason.m_Code, eMQ::MMQP_REASON_MALFORMED_PACKET);
    auto error = Envelope::Decode(replies[0].m_Payload);
    EMPY_CHECK_EQ(error.m_Kind, std::string("error"));
    EMPY_CHECK_EQ(error.m_Source, std::string("compliance"));
    EMPY_CHECK(error.m_Body.GetString("text").find("Malformed request envelope") != std::string::npos);

    // a well-formed request still reaches the handler afterwards
    Accepted(raw.Request(topics.Agent("compliance", "review"), Envelope::Make("agent-request", {}).Encode(), 1, 1, 8));
    EMPY_REQUIRE(pumpRaw([&] { return replies.size() == 2; }));
    EMPY_CHECK_EQ(handled, size_t(1));
    EMPY_CHECK_EQ(replies[1].m_Reason.m_Code, eMQ::MMQP_REASON_SUCCESS);
}

EMPY_TEST(OpsFabric, CloseCompletesPending)
{
    LocalBroker fabric;
    Peer caller(fabric.m_Port, "harness"), agent(fabric.m_Port, "placement"), other(fabric.m_Port, "operator");
    Topics topics;
    std::vector<Bus::Responder> held;
    EMPY_REQUIRE(agent.Get()
                     .Serve(topics.AgentFilter("placement"),
                            [&](const Inbound &, Bus::Responder respond) { held.push_back(std::move(respond)); })
                     .m_Code < 0x80);

    // closing the caller completes its pending exchanges at once
    auto &bus = caller.Get();
    std::vector<eMQ::Reason> outcomes;
    for (int i = 0; i < 2; ++i)
    {
        bus.Call(
            topics.Agent("placement", "place"), Envelope::Make("agent-request", {}),
            [&](eMQ::Reason reason, std::optional<Envelope>) { outcomes.push_back(reason); }, 10000);
    }
    EMPY_REQUIRE(Pump({&caller, &agent}, [&] { return held.size() == 2; }));
    EMPY_CHECK_EQ(bus.Pending(), size_t(2));
    bus.Close();
    EMPY_REQUIRE_EQ(outcomes.size(), size_t(2));
    for (const auto &reason : outcomes)
    {
        EMPY_CHECK(reason.m_Code >= 0x80);
        EMPY_CHECK_EQ(reason.m_Text, std::string("Bus closed"));
    }
    EMPY_CHECK_EQ(bus.Pending(), size_t(0));
    std::optional<eMQ::Reason> after;
    bus.Call(topics.Agent("placement", "place"), Envelope::Make("agent-request", {}),
             [&](eMQ::Reason reason, std::optional<Envelope>) { after = reason; });
    EMPY_REQUIRE(after.has_value());
    EMPY_CHECK(after->m_Code >= 0x80);
    EMPY_CHECK(bus.Listen("x/y", [](const std::string &, const Envelope &) {}).m_Code >= 0x80);
    held.clear(); // answers to the closed caller's IDs are dropped by its client

    // closing a responder answers the requests it still holds
    std::optional<std::pair<eMQ::Reason, std::optional<Envelope>>> reply;
    other.Get().Call(topics.Agent("placement", "place"), Envelope::Make("agent-request", {}),
                     [&](eMQ::Reason reason, std::optional<Envelope> envelope)
                     { reply.emplace(reason, std::move(envelope)); });
    EMPY_REQUIRE(Pump({&other, &agent}, [&] { return held.size() == 1; }));
    agent.Get().Close();
    EMPY_REQUIRE(Pump({&other, &agent}, [&] { return reply.has_value(); }));
    EMPY_CHECK(reply->first.m_Code >= 0x80);
    EMPY_CHECK_EQ(reply->first.m_Text, std::string("Responder bus closed"));
    held.clear(); // no further answer once closed
}

EMPY_TEST(OpsFabric, RetainedLateSubscriber)
{
    LocalBroker fabric;
    Peer harness(fabric.m_Port, "harness");
    Topics topics;
    auto &bus = harness.Get();
    Accepted(bus.Publish(topics.State("services"), Envelope::Make("state", Json::Object{{"version", 1}}), 1, 1));
    Accepted(bus.Publish(topics.State("services"), Envelope::Make("state", Json::Object{{"version", 2}}), 1, 1));
    Accepted(bus.Publish(topics.Heartbeat("home-control", "inf-0"),
                         Envelope::Make("status", Json::Object{{"state", "ready"}}), 1, 1));
    EMPY_CHECK_EQ(bus.Publish(topics.State("x"), Envelope::Make("state", {}), 0, 1).get().m_Code,
                  eMQ::MMQP_REASON_POLICY_VIOLATION);

    // a subscriber that arrives later receives the latest retained record only
    Peer agent(fabric.m_Port, "lifecycle");
    std::vector<int64_t> versions;
    std::vector<std::string> heartbeats;
    EMPY_REQUIRE(agent.Get()
                     .Listen(topics.State("services"), [&](const std::string &, const Envelope &envelope)
                             { versions.push_back(envelope.m_Body.GetInt("version")); })
                     .m_Code < 0x80);
    EMPY_REQUIRE(agent.Get()
                     .Listen(topics.HeartbeatFilter(),
                             [&](const std::string &topic, const Envelope &) { heartbeats.push_back(topic); })
                     .m_Code < 0x80);
    EMPY_REQUIRE(Pump({&agent}, [&] { return versions.size() == 1 && heartbeats.size() == 1; }));
    Pump({&agent}, [] { return false; }, 150);
    EMPY_CHECK(versions == std::vector<int64_t>({2}));
    EMPY_CHECK(heartbeats == std::vector<std::string>({"mlops/status/home-control/inf-0"}));

    // live publications keep flowing to the subscriber
    Accepted(bus.Publish(topics.State("services"), Envelope::Make("state", Json::Object{{"version", 3}}), 1, 1));
    EMPY_REQUIRE(Pump({&agent}, [&] { return versions.size() == 2; }));
    EMPY_CHECK_EQ(versions.back(), int64_t(3));
}

EMPY_TEST(OpsFabric, ModuleAttachDetach)
{
    LocalBroker fabric;
    Peer publisher(fabric.m_Port, "publisher");
    eMQ::Client client;
    client.Connect("127.0.0.1", fabric.m_Port);
    EMPY_CHECK_THROWS(BusOf(client));
    auto *module = client.AttachModule<BusModule>(BusOptions{"probe", 2000, 16, {}});
    EMPY_CHECK_THROWS(module->Get()); // OnAttach runs when the runtime starts
    client.Poll();
    EMPY_CHECK(&BusOf(client) == &module->Get());
    EMPY_CHECK_EQ(BusOf(client).Options().m_Source, std::string("probe"));
    const size_t base = client.ListenerCount<eMQ::Receive>();

    std::vector<std::string> seen;
    client.AttachModule<Probe>(Topics().EventFilter("drift"), &seen);
    auto pumpBoth = [&](auto &&done, uint32_t ms = 5000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(ms);
        while (!done() && Clock::now() < deadline)
        {
            client.Poll(1);
            publisher.m_Client->Poll(1);
        }
        return done();
    };
    Accepted(publisher.Get().Publish(Topics().Event("svc", "drift"), Envelope::Make("event", {})));
    EMPY_REQUIRE(pumpBoth([&] { return seen.size() == 1; }));

    // an owner-tagged pending exchange is dropped silently with its module
    bool invoked = false;
    BusOf(client).Call(
        Topics().Agent("nobody", "task"), Envelope::Make("agent-request", {}),
        [&](eMQ::Reason, std::optional<Envelope>) { invoked = true; }, 5000, client.GetModule<Probe>());
    EMPY_CHECK(client.DetachModule<Probe>());
    pumpBoth([] { return false; }, 300);
    EMPY_CHECK(!invoked);
    EMPY_CHECK_EQ(BusOf(client).Pending(), size_t(0));
    Accepted(publisher.Get().Publish(Topics().Event("svc", "drift"), Envelope::Make("event", {})));
    pumpBoth([] { return false; }, 300);
    EMPY_CHECK_EQ(seen.size(), size_t(1)); // the detached module's listener is gone

    // detaching the BusModule closes its Bus and removes the dispatch listener
    EMPY_CHECK(client.DetachModule<BusModule>());
    client.Poll(); // detachment is deferred while the runtime runs
    EMPY_CHECK_THROWS(BusOf(client));
    EMPY_CHECK(client.ListenerCount<eMQ::Receive>() < base);
    // reattaching yields a fresh, working Bus
    client.AttachModule<BusModule>(BusOptions{"again", 2000, 16, {}});
    std::vector<std::string> topics;
    EMPY_REQUIRE(BusOf(client)
                     .Listen(Topics().EventFilter("drift"),
                             [&](const std::string &topic, const Envelope &) { topics.push_back(topic); })
                     .m_Code < 0x80);
    Accepted(publisher.Get().Publish(Topics().Event("svc2", "drift"), Envelope::Make("event", {})));
    // (a retained "svc" record from the earlier publications may be replayed first)
    EMPY_REQUIRE(pumpBoth([&] { return !topics.empty() && topics.back() == "mlops/event/svc2/drift"; }));
}
