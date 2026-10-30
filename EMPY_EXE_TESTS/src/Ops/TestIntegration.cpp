#include "Checks.h"
#include "Support/Process.h"
#include "Support/Ops.h"
#include <eOps/Harness/State.h>
#include <eOps/Harness/Harness.h>
#include <eOps/Harness/AgentHost.h>
#include <eOps/Workload/Runtime.h>
#include <eOps/Fabric/Bus.h>
#include <eOps/Fabric/Topics.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Context/Comas.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/Process.h>
#include <eOps/Common/Text.h>
#include <Runtime/Runtime.h>
#include <charconv>
#include <filesystem>
#include <iostream>
#include <list>
#include <memory>
#include <thread>

// OpsIntegration: the Agentic-MLOps framework (thesis §2.1) end to end. In process: an operator intent over MMQP
// is interpreted into the Listing 2.1 BoM, reviewed, placed and enacted; the workloads, the sensor gateway and
// the operators are MMQP peers, so bootstrap training, promotion, closed-loop actions, concept drift, retraining,
// temporal triggers and governance conflicts (with a human in the loop) all travel through the broker. Process
// level: EmpyBroker, EmpyHarness run (process substrate launching EmpyWorkload), EmpyWorkload simulate and the
// EmpyHarness operator commands.

using namespace eOps;
using namespace eOps::Harness;
using Fabric::Bus;
using Fabric::Envelope;

namespace
{
    using Clock = std::chrono::steady_clock;
    using eTest::Ops::ClientConfigs;
    using eTest::Ops::ConfigFile;

    // request/response exchanges outlive the agent, workflow and HITL round trips of this suite
    constexpr uint32_t EXCHANGE_MS = 30000;

    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsIntegration")
        {
        }
    };

    struct LocalBroker : eTest::Ops::LocalBroker
    {
        LocalBroker() : eTest::Ops::LocalBroker(EXCHANGE_MS)
        {
        }
    };

    struct Peer : eTest::Ops::Peer
    {
        Peer(uint16_t port, std::string source) : eTest::Ops::Peer(port, std::move(source), ClientConfigs(EXCHANGE_MS))
        {
        }
    };

    const char *LISTING_INTENT =
        "Deploy a home temperature controller that keeps the temperature between 16 and 24 degrees using temperature "
        "and humidity sensors, with a p95 latency under 5 ms, retrained daily on a GPU";
    const char *SERVICE = "home-control";

#ifdef _WIN32
    const std::string SUFFIX = ".exe";
#else
    const std::string SUFFIX;
#endif

    Descriptor::RichMLService HomeControl()
    {
        return Descriptor::LoadDescriptor(ConfigFile("HomeControl.yaml"));
    }

    Descriptor::RichMLService EnergySaver()
    {
        return Descriptor::LoadDescriptor(ConfigFile("EnergySaver.yaml"));
    }

    // the MLOps control plane on one MMQP client (BusModule + AgentHost with the four agents + ExecutionHarness over
    // a simulated substrate) and the peers around it; every client is polled from the test thread (owner thread)
    struct ControlPlane
    {
        LocalBroker m_Broker;
        Semantics::Ontology m_Ontology = Semantics::Ontology::Load(ConfigFile("Ontology.yaml"));
        Context::KnowledgePlane m_Knowledge;
        Intelligence::ProviderSet m_Providers; // deterministic agents
        std::shared_ptr<Context::Comas> m_Comas;
        std::shared_ptr<Substrate::SimulatedSubstrate> m_Substrate = std::make_shared<Substrate::SimulatedSubstrate>();
        Fabric::Topics m_Topics;
        std::unique_ptr<eMQ::Client> m_Client;
        ExecutionHarness *m_Harness = nullptr;
        AgentHost *m_Agents = nullptr;
        std::list<std::unique_ptr<Peer>> m_Peers;

        ControlPlane()
        {
            m_Knowledge.Seed(eCore::YamlNode::Load(ConfigFile("Knowledge.yaml").string()));
            m_Comas = std::make_shared<Context::Comas>(m_Knowledge, &m_Providers);
            m_Client = std::make_unique<eMQ::Client>(ClientConfigs(EXCHANGE_MS));
            m_Client->Connect("127.0.0.1", m_Broker.m_Port);
            Fabric::BusOptions bus;
            bus.m_Source = "harness";
            m_Client->AttachModule<Fabric::BusModule>(bus);
            Descriptor::ValidationOptions validation;
            validation.m_Regions = {"ran", "edge", "cloud"};
            AgentHostOptions agents;
            agents.m_Policies =
                Agents::PolicySet::FromYaml(eCore::YamlNode::Load(ConfigFile("Policies.yaml").string()));
            agents.m_Validation = validation;
            m_Agents = m_Client->AttachModule<AgentHost>(
                agents, Agents::AgentEnvironment{m_Comas.get(), &m_Providers, &m_Ontology});
            HarnessOptions harness;
            harness.m_HitlTimeoutMs = 10000;
            harness.m_Validation = validation;
            harness.m_BrokerPort = m_Broker.m_Port;
            m_Harness = m_Client->AttachModule<ExecutionHarness>(harness, m_Substrate, m_Comas);
            m_Client->Poll();
        }

        ~ControlPlane()
        {
            m_Peers.clear();
            m_Client->Shutdown();
            m_Client.reset();
        }

        Peer &Join(std::string source)
        {
            m_Peers.push_back(std::make_unique<Peer>(m_Broker.m_Port, std::move(source)));
            return *m_Peers.back();
        }

        void Leave(Peer *peer)
        {
            m_Peers.remove_if([peer](const std::unique_ptr<Peer> &item) { return item.get() == peer; });
        }

        void PollOnce()
        {
            m_Client->Poll(1);
            for (const std::unique_ptr<Peer> &peer : m_Peers)
            {
                peer->m_Client->Poll(1);
            }
        }

        template <typename _Predicate> bool Pump(_Predicate &&done, uint32_t timeoutMs)
        {
            auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
            while (true)
            {
                PollOnce();
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

        // MMQP request from a peer (operator, agent proposal, ...) while the whole system keeps running
        std::pair<eMQ::Reason, std::optional<Envelope>> Call(Peer &from, const std::string &topic,
                                                             const Envelope &request, uint32_t timeoutMs = 20000)
        {
            return from.Get().CallSync(topic, request, [this] { PollOnce(); }, timeoutMs);
        }

        // operator request answered with a finished workflow record
        WorkflowRecord Workflow(Peer &from, const std::string &topic, const Envelope &request)
        {
            auto [reason, reply] = Call(from, topic, request);
            EMPY_REQUIRE(!reason);
            EMPY_REQUIRE(reply.has_value());
            EMPY_REQUIRE_EQ(reply->m_Kind, std::string("workflow"));
            return WorkflowRecord::FromJson(reply->m_Body);
        }

        const StateStore &State() const
        {
            return m_Harness->State();
        }

        const Deployment *Find(std::string_view service) const
        {
            return State().FindDeployment(service);
        }

        uint32_t Active(std::string_view service) const
        {
            const Deployment *deployment = Find(service);
            return deployment && deployment->m_Lifecycle.m_Active ? *deployment->m_Lifecycle.m_Active : 0;
        }
    };

    Envelope DeployRequest(const Descriptor::RichMLService &service)
    {
        return Envelope::Make("deploy", Json::MakeObject({{"descriptor", Descriptor::EmitDescriptor(service)}}));
    }

    Envelope IntentRequest()
    {
        return Envelope::Make("intent", Json::MakeObject({{"intent", LISTING_INTENT}, {"operator", "alice"}}));
    }

    const StepRecord *Step(const WorkflowRecord &record, std::string_view name)
    {
        for (const StepRecord &step : record.m_Steps)
        {
            if (step.m_Name == name)
            {
                return &step;
            }
        }
        return nullptr;
    }

    bool Succeeded(const WorkflowRecord &record, std::initializer_list<const char *> steps)
    {
        for (const char *name : steps)
        {
            const StepRecord *step = Step(record, name);
            if (!step || step->m_Status != "succeeded")
            {
                return false;
            }
        }
        return true;
    }

    // workflows of `kind` for `service`, newest first
    std::vector<const WorkflowRecord *> Workflows(const ControlPlane &plane, std::string_view kind,
                                                  std::string_view service = SERVICE)
    {
        std::vector<const WorkflowRecord *> found;
        for (const WorkflowRecord *record : plane.State().Workflows())
        {
            if (record->m_Kind == kind && record->m_Service == service)
            {
                found.push_back(record);
            }
        }
        return found;
    }

    std::vector<std::string> NodesOf(const Deployment &deployment, std::string_view role)
    {
        std::vector<std::string> nodes;
        for (const Instance &instance : deployment.m_Instances)
        {
            if (instance.m_Role == role)
            {
                nodes.push_back(instance.m_Node);
            }
        }
        return nodes;
    }

    const Instance *InstanceOf(const Deployment &deployment, std::string_view role)
    {
        for (const Instance &instance : deployment.m_Instances)
        {
            if (instance.m_Role == role)
            {
                return &instance;
            }
        }
        return nullptr;
    }

    // the test plays the execution substrate: a workload for the instance the harness enacted for `role`
    Workload::WorkloadOptions Enacted(const ControlPlane &plane, std::string_view role)
    {
        const Deployment *deployment = plane.Find(SERVICE);
        EMPY_REQUIRE(deployment != nullptr);
        const Instance *instance = InstanceOf(*deployment, role);
        EMPY_REQUIRE(instance != nullptr);
        Workload::WorkloadOptions options;
        options.m_Service = deployment->m_Descriptor;
        options.m_Instance = instance->m_Id;
        options.m_Node = instance->m_Node;
        options.m_Topics = plane.m_Topics;
        for (const Substrate::DeploymentSpec &spec : plane.m_Substrate->Deployed())
        {
            if (spec.m_Instance == instance->m_Id)
            {
                options.m_Region = spec.m_Region;
            }
        }
        EMPY_REQUIRE(!options.m_Region.empty());
        options.m_HeartbeatMs = 500;
        options.m_MetricsMs = 0; // drift events (not periodic metrics) drive the lifecycle agent here
        options.m_DriftDebounceMs = 1000;
        return options;
    }

    // synthetic home gateway: home/temp, home/hum and the ground truth home/label/temp every 20 ms
    Workload::SimulatorOptions Gateway(const Fabric::Topics &topics, uint32_t driftAfterMs = 0)
    {
        Workload::SimulatorOptions options;
        options.m_Service = HomeControl();
        options.m_Topics = topics;
        options.m_PeriodMs = 20;
        options.m_DriftAfterMs = driftAfterMs;
        return options;
    }

    struct Action
    {
        std::string m_Topic;
        Json m_Body;
    };

    void ListenActions(Peer &peer, std::vector<Action> &seen)
    {
        EMPY_REQUIRE(!peer.Get().ListenRaw(
            "home/set/#", [&seen](const eMQ::PublishRequest &message)
            { seen.push_back(Action{message.m_Topic, FromBytes(message.m_Payload)}); }, 1));
    }

    void ListenEnvelopes(Peer &peer, const std::string &filter, std::vector<Envelope> &seen)
    {
        EMPY_REQUIRE(!peer.Get().Listen(filter, [&seen](const std::string &, const Envelope &envelope)
                                        { seen.push_back(envelope); }));
    }

    size_t ActionsOf(const std::vector<Action> &actions, int64_t model)
    {
        size_t count = 0;
        for (const Action &action : actions)
        {
            count += action.m_Body.GetInt("model") == model ? 1 : 0;
        }
        return count;
    }

    // every action stays inside the declared execution boundary of home/set/temp (16..24 celsius)
    bool Bounded(const std::vector<Action> &actions)
    {
        for (const Action &action : actions)
        {
            double value = action.m_Body.GetNumber("value", -1);
            if (action.m_Topic != "home/set/temp" || action.m_Body.GetString("service") != SERVICE || value < 16 ||
                value > 24 || action.m_Body.GetInt("model") < 1)
            {
                return false;
            }
        }
        return !actions.empty();
    }

    // agents memory episode of `agent`/`kind` ("" = any kind) whose data matches
    bool Remembered(const ControlPlane &plane, std::string_view agent, std::string_view kind,
                    const std::function<bool(const Json &data)> &match = {})
    {
        for (const Context::Entry &entry : plane.m_Knowledge.Scan(Context::MEMORY))
        {
            if (entry.m_Data.GetString("agent") == agent && (kind.empty() || entry.m_Data.GetString("kind") == kind) &&
                (!match || match(entry.m_Data["data"])))
            {
                return true;
            }
        }
        return false;
    }

    // lifecycle status of model `version` of the service ("" when unknown)
    std::string ModelStatus(const ControlPlane &plane, uint32_t version)
    {
        if (const Deployment *deployment = plane.Find(SERVICE))
        {
            for (const Agents::ModelRecord &record : deployment->m_Lifecycle.m_History)
            {
                if (record.m_Version == version)
                {
                    return record.m_Status;
                }
            }
        }
        return {};
    }
} // namespace

EMPY_TEST(OpsIntegration, IntentToClosedLoop)
{
    ControlPlane plane;
    const Fabric::Topics &topics = plane.m_Topics;
    Peer &operatorPeer = plane.Join("operator");
    Peer &observer = plane.Join("observer");
    std::vector<Action> actions;
    std::vector<Envelope> drifts, candidates;
    ListenActions(observer, actions);
    ListenEnvelopes(observer, topics.EventFilter("drift"), drifts);
    ListenEnvelopes(observer, topics.CandidateFilter(), candidates);
    Peer *gateway = &plane.Join("gateway");
    gateway->Attach<Workload::Simulator>(Gateway(topics));

    // 1. operator intent over MMQP: Executive (ontology-grounded BoM) → Compliance → Placement → enactment
    WorkflowRecord deploy = plane.Workflow(operatorPeer, topics.Intent(), IntentRequest());
    EMPY_REQUIRE_EQ(deploy.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(deploy.m_Kind, std::string("deploy"));
    EMPY_CHECK_EQ(deploy.m_Trigger, std::string("intent"));
    EMPY_CHECK_EQ(deploy.m_Service, std::string(SERVICE));
    EMPY_CHECK(Succeeded(
        deploy, {"interpret", "validate", "review", "place-inference", "place-training", "enact", "schedule"}));
    const StepRecord *interpret = Step(deploy, "interpret");
    EMPY_REQUIRE(interpret != nullptr);
    EMPY_CHECK(interpret->m_Detail["valid"].AsBool());
    EMPY_CHECK_EQ(interpret->m_Detail["frame"].GetString("name"), std::string(SERVICE));
    const Deployment *deployment = plane.Find(SERVICE);
    EMPY_REQUIRE(deployment != nullptr);
    // the generated BoM is Listing 2.1 (Configs/MLOps/HomeControl.yaml)
    EMPY_CHECK(Descriptor::ToJson(deployment->m_Descriptor) == Descriptor::ToJson(HomeControl()));
    EMPY_CHECK_EQ(deploy.m_Result.GetString("compliance"), std::string("approved"));
    // inference near the dense home telemetry (RAN), training on accelerator capacity
    EMPY_CHECK(NodesOf(*deployment, "inference") == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(NodesOf(*deployment, "training") == std::vector<std::string>({"cloud-1"}));
    EMPY_CHECK_EQ(plane.m_Substrate->Deployed().size(), size_t(2));
    EMPY_CHECK(!deployment->m_Lifecycle.m_Active.has_value());
    EMPY_CHECK_EQ(plane.m_Harness->Schedules().Entries().size(), size_t(1));

    // 2. the enacted workloads join the fabric; the trainer collects the minimum dataset from the gateway
    Workload::WorkloadOptions trainingAt = Enacted(plane, "training");
    Workload::WorkloadOptions inferenceAt = Enacted(plane, "inference");
    EMPY_CHECK_EQ(inferenceAt.m_Region, std::string("ran"));
    Peer *trainer = &plane.Join(trainingAt.m_Instance);
    auto *training = trainer->Attach<Workload::TrainingRuntime>(trainingAt);
    EMPY_REQUIRE(plane.Pump([&] { return training->Samples() >= 32; }, 10000));
    Peer *predictor = &plane.Join(inferenceAt.m_Instance);
    auto *inference = predictor->Attach<Workload::InferenceRuntime>(inferenceAt);

    // event-driven activation: waiting-model heartbeat → Lifecycle bootstrap proposal → retrain workflow → train →
    // lifecycle/evaluate → Guard → promotion → model load
    EMPY_REQUIRE(plane.Pump([&] { return plane.Active(SERVICE) == 1 && inference->Active().has_value(); }, 20000));
    EMPY_CHECK_EQ(uint32_t(inference->Active()->m_Version), uint32_t(1));
    std::vector<const WorkflowRecord *> retrains = Workflows(plane, "retrain");
    EMPY_REQUIRE_EQ(retrains.size(), size_t(1));
    const WorkflowRecord *bootstrap = retrains.front();
    EMPY_CHECK_EQ(bootstrap->m_Trigger, std::string("bootstrap"));
    EMPY_CHECK_EQ(bootstrap->m_Status, std::string("succeeded"));
    EMPY_CHECK(Succeeded(*bootstrap, {"train", "evaluate", "load"}));
    EMPY_CHECK_EQ(bootstrap->m_Result["decision"].GetString("action"), std::string("promote"));
    EMPY_CHECK_EQ(bootstrap->m_Result.GetInt("version"), int64_t(1));
    EMPY_CHECK(bootstrap->m_Result.GetNumber("accuracy") >= 0.9);
    EMPY_CHECK_EQ(ModelStatus(plane, 1), std::string("active"));
    EMPY_CHECK(
        Remembered(plane, "lifecycle", "decision", [](const Json &data)
                   { return data.GetString("trigger") == "bootstrap" && data.GetString("action") == "retrain"; }));
    // the candidate was published on the train/model observable
    EMPY_REQUIRE(plane.Pump([&] { return !candidates.empty(); }, 5000));
    EMPY_CHECK_EQ(candidates.front().m_Kind, std::string("artifact"));
    EMPY_CHECK_EQ(candidates.front().m_Body["artifact"].GetInt("version"), int64_t(1));

    // closed loop: model v1 drives home/set/temp within the declared bounds; heartbeats reach the harness
    EMPY_REQUIRE(plane.Pump([&] { return ActionsOf(actions, 1) >= 10; }, 10000));
    EMPY_CHECK(Bounded(actions));
    EMPY_CHECK_EQ(actions.front().m_Body.GetString("instance"), inferenceAt.m_Instance);
    EMPY_REQUIRE(plane.Pump(
        [&]
        {
            const Deployment *current = plane.Find(SERVICE);
            const Instance *predicting = InstanceOf(*current, "inference");
            const Instance *learning = InstanceOf(*current, "training");
            return predicting->m_State == "ready" && predicting->m_LastSeen > 0 && learning->m_LastSeen > 0;
        },
        5000));
    // every agent and the harness left their decisions in the CoMaS agents memory
    for (const char *agent : {"executive", "compliance", "placement", "lifecycle", "harness"})
    {
        EMPY_CHECK(Remembered(plane, agent, ""));
    }

    // 3. the operator tightens the lifecycle policy (version 1.1.0, 1 s retraining cooldown) with a descriptor over
    // MMQP: the harness replaces the instances and carries the promoted model over
    Descriptor::RichMLService upgraded = plane.Find(SERVICE)->m_Descriptor;
    upgraded.m_Metadata.m_Version = "1.1.0";
    upgraded.m_Lifecycle.m_CooldownSeconds = 1;
    WorkflowRecord upgrade = plane.Workflow(operatorPeer, topics.Deploy(), DeployRequest(upgraded));
    EMPY_REQUIRE_EQ(upgrade.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(upgrade.m_Trigger, std::string("descriptor"));
    EMPY_CHECK_EQ(upgrade.m_Result["replaced"].Size(), size_t(2));
    deployment = plane.Find(SERVICE);
    EMPY_CHECK_EQ(deployment->m_Version, std::string("1.1.0"));
    EMPY_CHECK_EQ(plane.Active(SERVICE), uint32_t(1));
    EMPY_CHECK_EQ(deployment->m_Artifacts.count(1), size_t(1));
    EMPY_CHECK_EQ(plane.m_Substrate->Deployed().size(), size_t(2));

    // the replaced workloads and the gateway leave; a gateway with drifted coefficients takes over (concept drift)
    plane.Leave(gateway);
    plane.Leave(predictor);
    plane.Leave(trainer);
    gateway = &plane.Join("gateway-drift");
    auto *drifting = gateway->Attach<Workload::Simulator>(Gateway(topics, 1));
    EMPY_REQUIRE(plane.Pump([&] { return drifting->Drifted(); }, 5000));
    size_t drifted = drifting->Published();
    EMPY_REQUIRE(plane.Pump([&] { return drifting->Published() >= drifted + 6; }, 5000));

    // the new instances: the trainer sees only drifted data; the predictor is re-synchronised with model v1
    Workload::WorkloadOptions trainingNext = Enacted(plane, "training");
    Workload::WorkloadOptions inferenceNext = Enacted(plane, "inference");
    EMPY_CHECK(trainingNext.m_Instance != trainingAt.m_Instance);
    EMPY_CHECK(inferenceNext.m_Instance != inferenceAt.m_Instance);
    trainer = &plane.Join(trainingNext.m_Instance);
    training = trainer->Attach<Workload::TrainingRuntime>(trainingNext);
    EMPY_REQUIRE(plane.Pump([&] { return training->Samples() >= 32; }, 10000));
    predictor = &plane.Join(inferenceNext.m_Instance);
    inference = predictor->Attach<Workload::InferenceRuntime>(inferenceNext);
    uint32_t resynced = 0;
    EMPY_REQUIRE(plane.Pump(
        [&]
        {
            resynced = inference->Active() ? inference->Active()->m_Version : 0;
            return resynced != 0;
        },
        10000));
    EMPY_CHECK_EQ(resynced, uint32_t(1));

    // 4. drift: v1 mispredicts the drifted relation → accuracy drift event → Lifecycle proposal → retrain on the
    // drifted data → evaluated against the observed accuracy → promoted and loaded
    EMPY_REQUIRE(plane.Pump(
        [&]
        {
            uint32_t active = plane.Active(SERVICE);
            return active > 1 && inference->Active() && inference->Active()->m_Version == active;
        },
        25000));
    uint32_t promoted = plane.Active(SERVICE);
    bool accuracyDrift = false;
    for (const Envelope &event : drifts)
    {
        if (event.m_Body.GetString("kind") == "accuracy" && event.m_Body.GetInt("modelVersion") == 1 &&
            event.m_Body.GetString("instance") == inferenceNext.m_Instance)
        {
            accuracyDrift = true;
            EMPY_CHECK(event.m_Body.GetNumber("value") < 0.9);
            EMPY_CHECK_EQ(event.m_Body.GetNumber("threshold"), 0.9);
        }
    }
    EMPY_CHECK(accuracyDrift);
    // every retrain since the upgrade follows the lifecycle policy: the drift-triggered one is promoted because it
    // meets the threshold and improves on the observed accuracy; any other candidate is rejected with its reason
    const WorkflowRecord *driftRetrain = nullptr;
    for (const WorkflowRecord *record : Workflows(plane, "retrain"))
    {
        if (record == bootstrap || record->m_Created < upgrade.m_Created)
        {
            continue;
        }
        if (record->m_Status == "succeeded")
        {
            EMPY_CHECK(record->m_Result.GetInt("version") <= int64_t(promoted));
            if (record->m_Result.GetInt("version") == int64_t(promoted))
            {
                driftRetrain = record;
            }
        }
        else
        {
            EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
            EMPY_CHECK_EQ(record->m_Result["decision"].GetString("action"), std::string("reject"));
            EMPY_CHECK(!record->m_Error.empty());
        }
    }
    EMPY_REQUIRE(driftRetrain != nullptr);
    EMPY_CHECK_EQ(driftRetrain->m_Trigger, std::string("drift"));
    EMPY_CHECK(driftRetrain->m_Input.GetNumber("observed", 1) < 0.9);
    EMPY_CHECK(Succeeded(*driftRetrain, {"train", "evaluate", "load"}));
    EMPY_CHECK_EQ(driftRetrain->m_Result["decision"].GetString("action"), std::string("promote"));
    double accuracy = driftRetrain->m_Result.GetNumber("accuracy");
    EMPY_CHECK(accuracy >= 0.9);
    EMPY_CHECK(accuracy - driftRetrain->m_Input.GetNumber("observed") >= 0.02);
    EMPY_CHECK_EQ(driftRetrain->m_Result.GetInt("previous"), int64_t(1));
    const StepRecord *evaluate = Step(*driftRetrain, "evaluate");
    EMPY_REQUIRE(evaluate != nullptr);
    EMPY_CHECK(evaluate->m_Detail["observed"].IsNumber());
    EMPY_CHECK_EQ(ModelStatus(plane, 1), std::string("retired"));
    EMPY_CHECK_EQ(ModelStatus(plane, promoted), std::string("active"));
    EMPY_CHECK(Remembered(plane, "lifecycle", "decision", [](const Json &data)
                          { return data.GetString("trigger") == "drift" && data.GetString("action") == "retrain"; }));
    bool observed = false;
    for (const Context::Entry &entry : plane.m_Knowledge.Scan(Context::OBSERVATIONS))
    {
        observed = observed || entry.m_Data.GetString("kind") == "drift";
    }
    EMPY_CHECK(observed);
    // the promoted model closes the loop again, within the same execution boundary
    EMPY_REQUIRE(plane.Pump([&] { return ActionsOf(actions, promoted) >= 5; }, 10000));
    EMPY_CHECK(Bounded(actions));
}

EMPY_TEST(OpsIntegration, TemporalTriggerReachesLifecycle)
{
    ControlPlane plane;
    const Fabric::Topics &topics = plane.m_Topics;
    Peer &observer = plane.Join("observer");
    std::vector<Envelope> triggers;
    ListenEnvelopes(observer, topics.TriggerFilter("training"), triggers);
    plane.Join("gateway").Attach<Workload::Simulator>(Gateway(topics));

    // the workloads run before the deployment, under the instance names the substrate will assign; the predictor
    // reports only on attach and on model changes, so the temporal trigger alone starts the training
    Descriptor::RichMLService service = HomeControl();
    service.m_Training->m_Schedule = "@every 1s";
    service.m_Lifecycle.m_CooldownSeconds = 1;
    Workload::WorkloadOptions trainingAt;
    trainingAt.m_Service = service;
    trainingAt.m_Instance = Substrate::InstanceName(SERVICE, "training", "cloud-1", 0);
    trainingAt.m_Node = "cloud-1";
    trainingAt.m_Region = "cloud";
    trainingAt.m_Topics = topics;
    Workload::WorkloadOptions inferenceAt = trainingAt;
    inferenceAt.m_Instance = Substrate::InstanceName(SERVICE, "inference", "ran-1", 0);
    inferenceAt.m_Node = "ran-1";
    inferenceAt.m_Region = "ran";
    inferenceAt.m_HeartbeatMs = 0;
    inferenceAt.m_MetricsMs = 0;
    auto *training = plane.Join(trainingAt.m_Instance).Attach<Workload::TrainingRuntime>(trainingAt);
    auto *inference = plane.Join(inferenceAt.m_Instance).Attach<Workload::InferenceRuntime>(inferenceAt);
    EMPY_REQUIRE(plane.Pump([&] { return training->Samples() >= 32; }, 10000));

    WorkflowRecord deploy = plane.Workflow(observer, topics.Deploy(), DeployRequest(service));
    EMPY_REQUIRE_EQ(deploy.m_Status, std::string("succeeded"));
    const Deployment *deployment = plane.Find(SERVICE);
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK_EQ(InstanceOf(*deployment, "training")->m_Id, trainingAt.m_Instance);
    EMPY_CHECK_EQ(InstanceOf(*deployment, "inference")->m_Id, inferenceAt.m_Instance);
    auto entries = plane.m_Harness->Schedules().Entries();
    EMPY_REQUIRE_EQ(entries.size(), size_t(1));
    EMPY_CHECK_EQ(entries[0].m_Expression, std::string("@every 1s"));

    // trigger on <root>/trigger/time/<service>/training → Lifecycle agent (topic-profile routing) → retrain
    // proposal → harness retrain → promotion of v1 into the running predictor
    auto finished = [&](std::string_view status)
    {
        for (const WorkflowRecord *record : Workflows(plane, "retrain"))
        {
            if (record->m_Trigger == "schedule" && record->m_Status == status)
            {
                return record;
            }
        }
        return static_cast<const WorkflowRecord *>(nullptr);
    };
    EMPY_REQUIRE(plane.Pump([&] { return finished("succeeded") && inference->Active().has_value(); }, 15000));
    const WorkflowRecord *scheduled = finished("succeeded");
    EMPY_CHECK_EQ(scheduled->m_Result.GetInt("version"), int64_t(1));
    EMPY_CHECK(Succeeded(*scheduled, {"train", "evaluate", "load"}));
    EMPY_CHECK_EQ(uint32_t(inference->Active()->m_Version), uint32_t(1));
    EMPY_REQUIRE(!triggers.empty());
    EMPY_CHECK_EQ(triggers.front().m_Kind, std::string("trigger"));
    EMPY_CHECK_EQ(triggers.front().m_Body.GetString("service"), std::string(SERVICE));
    EMPY_CHECK_EQ(triggers.front().m_Body.GetString("job"), std::string("training"));
    EMPY_CHECK_EQ(triggers.front().m_Body.GetString("schedule"), std::string("@every 1s"));
    EMPY_CHECK(
        Remembered(plane, "lifecycle", "decision", [](const Json &data)
                   { return data.GetString("trigger") == "schedule" && data.GetString("action") == "retrain"; }));
    for (const WorkflowRecord *record : Workflows(plane, "retrain"))
    {
        EMPY_CHECK_EQ(record->m_Trigger, std::string("schedule")); // no bootstrap, drift or metrics activation
    }

    // a later trigger retrains on the same (undrifted) relation: the candidate is as accurate as v1, so the
    // lifecycle policy (minImprovement 0.02) rejects it and v1 stays active
    EMPY_REQUIRE(plane.Pump([&] { return finished("rejected") != nullptr; }, 15000));
    const WorkflowRecord *rejected = finished("rejected");
    EMPY_CHECK_EQ(rejected->m_Result["decision"].GetString("action"), std::string("reject"));
    EMPY_CHECK(rejected->m_Error.find("not promoted") != std::string::npos);
    EMPY_CHECK_EQ(plane.Active(SERVICE), uint32_t(1));
    EMPY_CHECK_EQ(ModelStatus(plane, static_cast<uint32_t>(rejected->m_Result.GetInt("candidate"))),
                  std::string("rejected"));

    // retiring the service removes its temporal trigger
    WorkflowRecord retire = plane.Workflow(
        observer, topics.Workflow(),
        Envelope::Make("workflow-request", Json::MakeObject({{"kind", "retire"}, {"service", SERVICE}})));
    EMPY_REQUIRE_EQ(retire.m_Status, std::string("succeeded"));
    EMPY_CHECK(plane.m_Harness->Schedules().Entries().empty());
    plane.Pump([] { return false; }, 300);
    size_t fired = triggers.size();
    plane.Pump([] { return false; }, 1500);
    EMPY_CHECK_EQ(triggers.size(), fired);
}

EMPY_TEST(OpsIntegration, ConflictGovernanceWithHitl)
{
    ControlPlane plane;
    const Fabric::Topics &topics = plane.m_Topics;
    // the operator answers HITL questions over MMQP request/response
    Peer &operatorPeer = plane.Join("operator");
    std::vector<Json> questions;
    std::string decision = "reject";
    EMPY_REQUIRE(!operatorPeer.Get().Serve(
        topics.Hitl(),
        [&](const Fabric::Inbound &request, Bus::Responder respond)
        {
            questions.push_back(request.m_Envelope.m_Body);
            respond(Envelope::Make("hitl-decision",
                                   Json::MakeObject({{"decision", decision}, {"operator", "alice"}, {"note", "test"}})),
                    {});
        }));

    WorkflowRecord home = plane.Workflow(operatorPeer, topics.Intent(), IntentRequest());
    EMPY_REQUIRE_EQ(home.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(home.m_Result.GetString("compliance"), std::string("approved"));

    // EnergySaver (priority -1) claims home/set/temp, which HomeControl (priority 0) already drives: dropping the
    // lower-priority actionable leaves nothing, so Policies.yaml-governed review rejects it without a human
    WorkflowRecord saver = plane.Workflow(operatorPeer, topics.Deploy(), DeployRequest(EnergySaver()));
    EMPY_CHECK_EQ(saver.m_Status, std::string("rejected"));
    EMPY_CHECK(saver.m_Error.find("conflict") != std::string::npos);
    const StepRecord *review = Step(saver, "review");
    EMPY_REQUIRE(review != nullptr);
    EMPY_CHECK_EQ(review->m_Detail.GetString("verdict"), std::string("rejected"));
    EMPY_CHECK(plane.Find("energy-saver") == nullptr);
    EMPY_CHECK(questions.empty());

    // with an uncontested lighting actionable the conflict is resolved automatically: amended (home/set/temp dropped)
    Descriptor::RichMLService lighting = EnergySaver();
    Descriptor::Actionable light;
    light.m_Name = "home/set/light";
    light.m_Unit = "percent";
    light.m_Min = 0;
    light.m_Max = 100;
    lighting.m_Actionables.push_back(light);
    lighting.m_Metadata.m_Intent += " and dim the lighting";
    WorkflowRecord amended = plane.Workflow(operatorPeer, topics.Deploy(), DeployRequest(lighting));
    EMPY_REQUIRE_EQ(amended.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(amended.m_Result.GetString("compliance"), std::string("amended"));
    const Deployment *deployment = plane.Find("energy-saver");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_REQUIRE_EQ(deployment->m_Descriptor.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(deployment->m_Descriptor.m_Actionables[0].m_Name, std::string("home/set/light"));
    bool dropped = false;
    for (const Json &resolution : Step(amended, "review")->m_Detail["resolutions"].AsArray())
    {
        dropped =
            dropped || (resolution.GetString("type") == "drop" && resolution.GetString("topic") == "home/set/temp");
    }
    EMPY_CHECK(dropped);
    EMPY_CHECK(questions.empty());

    // a higher-priority EnergySaver 1.1.0 claiming home/set/temp again cannot be resolved automatically (the two
    // controllers would oscillate): escalated to the human in the loop, who first rejects, then approves
    Descriptor::RichMLService contested = lighting;
    contested.m_Metadata.m_Version = "1.1.0";
    contested.m_Constraints.m_Priority = 1;
    WorkflowRecord refused = plane.Workflow(operatorPeer, topics.Deploy(), DeployRequest(contested));
    EMPY_CHECK_EQ(refused.m_Status, std::string("rejected"));
    EMPY_CHECK(refused.m_Error.find("operator 'alice'") != std::string::npos);
    EMPY_REQUIRE_EQ(questions.size(), size_t(1));
    EMPY_CHECK_EQ(questions[0].GetString("service"), std::string("energy-saver"));
    EMPY_CHECK_EQ(questions[0].GetString("workflowId"), refused.m_Id);
    EMPY_CHECK(questions[0].GetString("question").find(SERVICE) != std::string::npos);
    EMPY_CHECK(questions[0]["findings"].Size() > 0);
    EMPY_CHECK_EQ(Step(refused, "review")->m_Detail.GetString("verdict"), std::string("escalate"));
    EMPY_CHECK_EQ(plane.Find("energy-saver")->m_Version, std::string("1.0.0"));

    decision = "approve";
    WorkflowRecord approved = plane.Workflow(operatorPeer, topics.Deploy(), DeployRequest(contested));
    EMPY_REQUIRE_EQ(approved.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(questions.size(), size_t(2));
    const StepRecord *hitl = Step(approved, "hitl");
    EMPY_REQUIRE(hitl != nullptr);
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("decision"), std::string("approve"));
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("operator"), std::string("alice"));
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("source"), std::string("operator"));
    deployment = plane.Find("energy-saver");
    EMPY_CHECK_EQ(deployment->m_Version, std::string("1.1.0"));
    EMPY_CHECK(deployment->m_Descriptor.FindActionable("home/set/temp") != nullptr);
    EMPY_CHECK(plane.Find(SERVICE) != nullptr);
    EMPY_CHECK(Remembered(plane, "compliance", ""));
    EMPY_CHECK_EQ(plane.State().Deployments().size(), size_t(2));
}

EMPY_TEST(OpsIntegration, ExecutablesEndToEnd)
{
    const std::filesystem::path bin = eTest::BinaryDirectory();
    const std::filesystem::path brokerProgram = bin / ("EmpyBroker" + SUFFIX);
    const std::filesystem::path harnessProgram = bin / ("EmpyHarness" + SUFFIX);
    const std::filesystem::path workloadProgram = bin / ("EmpyWorkload" + SUFFIX);
    for (const std::filesystem::path &program : {brokerProgram, harnessProgram, workloadProgram})
    {
        EMPY_REQUIRE(std::filesystem::exists(program));
    }
    TempDir temp;
    const std::string path = eTest::RestrictedSearchPath();
    // shutdown is decoupled from wall-clock time: the harness and the gateway stop gracefully once the stop file
    // exists (after the retirement), the broker is stopped last. The --seconds ceiling only bounds a test that dies
    // half-way. Every wait below uses the same step budget.
    constexpr auto STEP = std::chrono::seconds(45);
    const std::string ceiling = "600";
    const std::filesystem::path stopFile = temp.m_Path / "stop";

    eTest::ChildProcess broker, harness, simulator;
    auto diagnose = [&]
    {
        std::cout << "--- broker\n"
                  << broker.Output() << "\n--- harness\n"
                  << harness.Output() << "\n--- simulator\n"
                  << simulator.Output() << "\n";
        std::error_code error;
        for (const auto &entry : std::filesystem::directory_iterator(temp.m_Path / "logs", error))
        {
            std::cout << "--- " << entry.path().filename().string() << "\n" << ReadFile(entry.path()) << "\n";
        }
    };

    // EmpyBroker with the MLOps profile (request_timeout_ms 120000) on an ephemeral port
    EMPY_REQUIRE(broker.Start(brokerProgram,
                              {"--config", ConfigFile("Broker.yaml").string(), "--port", "0", "--seconds", ceiling},
                              temp.m_Path, path, temp.m_Path / "broker.log"));
    uint16_t port = 0;
    EMPY_REQUIRE(eTest::WaitUntil(
        [&]
        {
            std::string output = broker.Output();
            const std::string marker = "MMQP broker listening on port ";
            size_t position = output.find(marker);
            if (position == std::string::npos)
            {
                return false;
            }
            const char *first = output.data() + position + marker.size();
            auto parsed = std::from_chars(first, output.data() + output.size(), port);
            return parsed.ec == std::errc{} && port != 0;
        },
        STEP));

    // HarnessLocal-style configuration: process substrate (EmpyWorkload beside EmpyHarness), ready heartbeats
    auto quoted = [](const std::filesystem::path &file) { return "'" + file.generic_string() + "'"; };
    const std::filesystem::path config = temp.m_Path / "Harness.yaml";
    WriteFileAtomic(config,
                    "broker: {host: 127.0.0.1, port: " + std::to_string(port) +
                        "}\n"
                        "topic_root: mlops\n"
                        "identity: harness\n"
                        "roles: [harness, executive, placement, lifecycle, compliance]\n"
                        "ontology: " +
                        quoted(ConfigFile("Ontology.yaml")) + "\nknowledge: " + quoted(ConfigFile("Knowledge.yaml")) +
                        "\npolicies: " + quoted(ConfigFile("Policies.yaml")) +
                        "\n"
                        "status: {address: 127.0.0.1, port: 0}\n"
                        "timeouts: {agent_ms: 20000, hitl_ms: 5000, ready_ms: 20000}\n"
                        "hitl: {default: reject}\n"
                        "wait_for_ready: true\n"
                        "substrate:\n"
                        "  type: process\n"
                        "  executable: \"\"\n"
                        "  state_directory: " +
                        quoted(temp.m_Path / "process") + "\n  log_directory: " + quoted(temp.m_Path / "logs") + "\n");
    const std::string portText = std::to_string(port);
    EMPY_REQUIRE(harness.Start(
        harnessProgram, {"run", "--config", config.string(), "--seconds", ceiling, "--stop-file", stopFile.string()},
        temp.m_Path, path, temp.m_Path / "harness.log"));
    bool running =
        eTest::WaitUntil([&] { return harness.Output().find("EmpyHarness running") != std::string::npos; }, STEP);
    if (!running)
    {
        diagnose();
    }
    EMPY_REQUIRE(running);
    EMPY_CHECK(harness.Output().find("substrate process") != std::string::npos);
    EMPY_REQUIRE(simulator.Start(workloadProgram,
                                 {"simulate", "--descriptor", ConfigFile("HomeControl.yaml").string(), "--host",
                                  "127.0.0.1", "--port", portText, "--period-ms", "20", "--seconds", ceiling,
                                  "--stop-file", stopFile.string()},
                                 temp.m_Path, path, temp.m_Path / "simulator.log"));
    EMPY_REQUIRE(eTest::WaitUntil([&] { return simulator.Output().find("connected") != std::string::npos; }, STEP));

    // an MMQP observer on the same broker sees what the executables publish
    Fabric::Topics topics;
    auto observer = std::make_unique<Peer>(port, "observer");
    std::vector<Envelope> beats;
    std::vector<Action> actions;
    ListenEnvelopes(*observer, topics.HeartbeatFilter(), beats);
    ListenActions(*observer, actions);
    auto settle = [&](const std::function<bool()> &done, std::chrono::milliseconds timeout)
    {
        auto deadline = Clock::now() + timeout;
        while (!done())
        {
            if (Clock::now() >= deadline)
            {
                return false;
            }
            if (observer)
            {
                observer->m_Client->Poll(10);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        return true;
    };

    // operator commands are separate EmpyHarness processes; stdout carries only the JSON result
    struct Outcome
    {
        std::optional<int32_t> m_Code;
        std::string m_Stdout;
        std::string m_Stderr;
    };
    auto command = [&](std::vector<std::string> args, const std::string &name)
    {
        args.insert(args.end(), {"--config", config.string()});
        ProcessOptions options;
        options.m_Program = harnessProgram;
        options.m_Args = std::move(args);
        options.m_WorkingDirectory = temp.m_Path;
        options.m_StdoutFile = temp.m_Path / (name + ".out");
        options.m_StderrFile = temp.m_Path / (name + ".err");
        Process process;
        std::string error;
        EMPY_REQUIRE(process.Start(options, &error));
        Outcome outcome;
        settle([&] { return !process.Running(); }, STEP);
        outcome.m_Code = process.Wait(std::chrono::seconds(1));
        outcome.m_Stdout = ReadFile(options.m_StdoutFile);
        outcome.m_Stderr = ReadFile(options.m_StderrFile);
        if (outcome.m_Code != std::optional<int32_t>(0))
        {
            std::cout << "--- EmpyHarness " << name << " (exit " << outcome.m_Code.value_or(-1) << ")\n"
                      << outcome.m_Stdout << "\n"
                      << outcome.m_Stderr << "\n";
            diagnose();
        }
        return outcome;
    };

    // EmpyHarness intent "<Listing 2.1 sentence>": generated BoM deployed as EmpyWorkload processes, ready
    Outcome intent = command({"intent", LISTING_INTENT, "--operator", "alice"}, "intent");
    EMPY_REQUIRE(intent.m_Code == std::optional<int32_t>(0));
    Json record = Json::Parse(intent.m_Stdout);
    EMPY_CHECK_EQ(record.GetString("status"), std::string("succeeded"));
    EMPY_CHECK_EQ(record.GetString("kind"), std::string("deploy"));
    EMPY_CHECK_EQ(record.GetString("service"), std::string(SERVICE));
    bool ready = false;
    for (const Json &step : record["steps"].AsArray())
    {
        ready = ready || (step.GetString("name") == "ready" && step.GetString("status") == "succeeded");
    }
    EMPY_CHECK(ready);
    EMPY_CHECK_EQ(record["result"]["placement"]["inference"].AsArray().front().AsString(), std::string("ran-1"));

    // the workload processes report heartbeats; bootstrap training promotes a model; actions flow
    auto beat = [&](std::string_view role, std::string_view state, int64_t model)
    {
        for (const Envelope &envelope : beats)
        {
            if (envelope.m_Body.GetString("role") == role && envelope.m_Body.GetString("state") == state &&
                envelope.m_Body.GetInt("modelVersion", 0) >= model)
            {
                return true;
            }
        }
        return false;
    };
    bool promoted = settle(
        [&]
        {
            return beat("inference", "ready", 1) &&
                   (beat("training", "ready", 0) || beat("training", "collecting", 0)) && actions.size() >= 5;
        },
        STEP);
    if (!promoted)
    {
        diagnose();
    }
    EMPY_REQUIRE(promoted);
    EMPY_CHECK(Bounded(actions));

    // EmpyHarness status: the deployment, its instances seen through heartbeats and the promoted model
    Outcome status = command({"status"}, "status");
    EMPY_REQUIRE(status.m_Code == std::optional<int32_t>(0));
    Json snapshot = Json::Parse(status.m_Stdout);
    EMPY_CHECK_EQ(snapshot["substrate"].GetString("type"), std::string("process"));
    EMPY_REQUIRE_EQ(snapshot["deployments"].Size(), size_t(1));
    const Json &deployed = snapshot["deployments"].AsArray().front();
    EMPY_CHECK_EQ(deployed.GetString("service"), std::string(SERVICE));
    EMPY_CHECK(deployed["lifecycle"].GetInt("active") >= 1);
    EMPY_REQUIRE_EQ(deployed["instances"].Size(), size_t(2));
    for (const Json &instance : deployed["instances"].AsArray())
    {
        EMPY_CHECK(instance.GetInt("lastSeen") > 0);
        if (instance.GetString("role") == "inference")
        {
            EMPY_CHECK_EQ(instance.GetString("state"), std::string("ready"));
        }
    }
    bool bootstrapped = false;
    for (const Json &workflow : snapshot["workflows"].AsArray())
    {
        bootstrapped =
            bootstrapped || (workflow.GetString("kind") == "retrain" && workflow.GetString("trigger") == "bootstrap" &&
                             workflow.GetString("status") == "succeeded");
    }
    EMPY_CHECK(bootstrapped);

    // the read-only HTTP status endpoint of `run`
    std::string output = harness.Output();
    const std::string marker = "status http://127.0.0.1:";
    size_t position = output.find(marker);
    EMPY_REQUIRE(position != std::string::npos);
    uint16_t statusPort = 0;
    std::from_chars(output.data() + position + marker.size(), output.data() + output.size(), statusPort);
    EMPY_REQUIRE(statusPort != 0);
    HttpRequest health;
    health.m_Method = "GET";
    health.m_Url = "http://127.0.0.1:" + std::to_string(statusPort) + "/api/mlops/healthz";
    health.m_TimeoutMs = 5000;
    HttpResponse healthz = SendHttp(health);
    EMPY_CHECK_EQ(healthz.m_Status, int32_t(200));
    EMPY_CHECK_EQ(Json::Parse(healthz.m_Body).GetString("status"), std::string("ok"));

    // clean shutdown: the operator retires the service (the substrate stops the workload processes), then the
    // harness and the gateway stop gracefully on the stop file and exit with status 0
    auto [reason, reply] = observer->Get().CallSync(
        topics.Workflow(),
        Envelope::Make("workflow-request", Json::MakeObject({{"kind", "retire"}, {"service", SERVICE}})),
        [&] { observer->m_Client->Poll(5); },
        static_cast<uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(STEP).count()));
    EMPY_REQUIRE(!reason);
    EMPY_REQUIRE(reply.has_value());
    EMPY_CHECK_EQ(reply->m_Body.GetString("status"), std::string("succeeded"));
    EMPY_CHECK_EQ(reply->m_Body["result"]["retired"].Size(), size_t(2));
    observer.reset();
    Outcome after = command({"status"}, "status-after");
    EMPY_REQUIRE(after.m_Code == std::optional<int32_t>(0));
    EMPY_CHECK_EQ(Json::Parse(after.m_Stdout)["deployments"].Size(), size_t(0));

    WriteFileAtomic(stopFile, "stop\n");
    int32_t harnessExit = harness.Wait(STEP);
    int32_t simulatorExit = simulator.Wait(STEP);
    if (harnessExit != 0 || simulatorExit != 0)
    {
        diagnose();
    }
    EMPY_CHECK_EQ(harnessExit, 0);
    EMPY_CHECK_EQ(simulatorExit, 0);
    EMPY_CHECK(harness.Output().find("EmpyHarness stopped") != std::string::npos);
    EMPY_CHECK(broker.Running()); // the clients left first; the broker (no stop file of its own) is stopped now
    broker.Kill();
}
