#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Harness/State.h>
#include <eOps/Harness/Guard.h>
#include <eOps/Harness/Scheduler.h>
#include <eOps/Harness/Harness.h>
#include <eOps/Harness/AgentHost.h>
#include <eOps/Harness/Config.h>
#include <eOps/Harness/Status.h>
#include <eOps/Fabric/Bus.h>
#include <eOps/Fabric/Topics.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Context/Comas.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/Text.h>
#include <Runtime/Runtime.h>
#include <map>
#include <set>
#include <mutex>
#include <atomic>
#include <limits>
#include <memory>
#include <filesystem>
#include <functional>

using namespace eOps;
using namespace eOps::Harness;
using Fabric::Bus;
using Fabric::Envelope;

namespace
{
    using Clock = std::chrono::steady_clock;
    using SystemClock = std::chrono::system_clock;
    using eTest::Ops::ClientConfigs;
    using eTest::Ops::ConfigFile;

    // exchanges of this suite (agents, workflows, HITL) are bounded by 20 s on both sides
    constexpr uint32_t EXCHANGE_MS = 20000;

    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsHarness")
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

    Descriptor::RichMLService HomeControl()
    {
        return Descriptor::LoadDescriptor(ConfigFile("HomeControl.yaml"));
    }

    Descriptor::RichMLService EnergySaver()
    {
        return Descriptor::LoadDescriptor(ConfigFile("EnergySaver.yaml"));
    }

    struct StackOptions
    {
        std::vector<std::string> m_Roles = {"executive", "placement", "lifecycle", "compliance"};
        uint32_t m_HitlTimeoutMs = 10000;
        bool m_HitlDefaultApprove = false;
        uint32_t m_PlacementAuditSeconds = 0;
        std::filesystem::path m_StateDirectory; // "" = in-memory harness state
        bool m_WaitForReady = false;
        uint32_t m_ReadyTimeoutMs = 30000;
        std::optional<uint32_t> m_LivenessMs; // unset = the harness default
        bool m_SecureWorkloads = false;
        std::shared_ptr<Substrate::SimulatedSubstrate> m_Substrate; // null = a plain simulated substrate
        // adjusts the knowledge plane after the shipped seed (resource registry, network knowledge)
        std::function<void(Context::KnowledgePlane &)> m_Knowledge;
        std::shared_ptr<Intelligence::Provider> m_Reasoning; // reasoning provider of the agents (null = none)
        std::filesystem::path m_OntologyState;               // AgentHost evolved-ontology file
    };

    // in-process harness runtime: BusModule + AgentHost + ExecutionHarness over a simulated substrate
    struct Stack
    {
        TempDir m_Temp;
        LocalBroker m_Broker;
        Semantics::Ontology m_Ontology = Semantics::Ontology::Load(ConfigFile("Ontology.yaml"));
        Context::KnowledgePlane m_Plane;
        Intelligence::ProviderSet m_Providers;
        std::shared_ptr<Context::Comas> m_Comas;
        std::shared_ptr<Substrate::SimulatedSubstrate> m_Substrate = std::make_shared<Substrate::SimulatedSubstrate>();
        Fabric::Topics m_Topics;
        std::unique_ptr<eMQ::Client> m_Client;
        ExecutionHarness *m_Harness = nullptr;
        AgentHost *m_Agents = nullptr;

        explicit Stack(StackOptions options = {})
        {
            m_Plane.Seed(eCore::YamlNode::Load(ConfigFile("Knowledge.yaml").string()));
            if (options.m_Knowledge)
            {
                options.m_Knowledge(m_Plane);
            }
            if (options.m_Substrate)
            {
                m_Substrate = options.m_Substrate;
            }
            if (options.m_Reasoning)
            {
                m_Providers.Set(Intelligence::Capability::Reasoning, options.m_Reasoning);
            }
            m_Comas = std::make_shared<Context::Comas>(m_Plane, &m_Providers);
            m_Client = std::make_unique<eMQ::Client>(ClientConfigs(EXCHANGE_MS));
            m_Client->Connect("127.0.0.1", m_Broker.m_Port);
            Fabric::BusOptions bus;
            bus.m_Source = "harness";
            m_Client->AttachModule<Fabric::BusModule>(bus);
            Descriptor::ValidationOptions validation;
            validation.m_Regions = {"ran", "edge", "cloud"};
            AgentHostOptions agents;
            agents.m_Roles = options.m_Roles;
            agents.m_Policies =
                Agents::PolicySet::FromYaml(eCore::YamlNode::Load(ConfigFile("Policies.yaml").string()));
            agents.m_Validation = validation;
            agents.m_OntologyState = options.m_OntologyState;
            m_Agents = m_Client->AttachModule<AgentHost>(
                agents, Agents::AgentEnvironment{m_Comas.get(), &m_Providers, &m_Ontology});
            HarnessOptions harness;
            harness.m_HitlTimeoutMs = options.m_HitlTimeoutMs;
            harness.m_HitlDefaultApprove = options.m_HitlDefaultApprove;
            harness.m_PlacementAuditSeconds = options.m_PlacementAuditSeconds;
            harness.m_Validation = validation;
            harness.m_BrokerPort = m_Broker.m_Port;
            harness.m_StateDirectory = options.m_StateDirectory;
            harness.m_WaitForReady = options.m_WaitForReady;
            harness.m_ReadyTimeoutMs = options.m_ReadyTimeoutMs;
            harness.m_LivenessMs = options.m_LivenessMs.value_or(harness.m_LivenessMs);
            harness.m_SecureWorkloads = options.m_SecureWorkloads;
            m_Harness = m_Client->AttachModule<ExecutionHarness>(harness, m_Substrate, m_Comas);
            m_Client->Poll();
        }

        ~Stack()
        {
            m_Client->Shutdown();
            m_Client.reset();
        }
    };

    template <typename _Predicate>
    bool Pump(Stack &stack, std::initializer_list<Peer *> peers, _Predicate &&done, uint32_t timeoutMs = 15000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true)
        {
            stack.m_Client->Poll(1);
            for (Peer *peer : peers)
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

    // submit a workflow and pump until it finishes
    std::optional<WorkflowRecord> RunWorkflow(Stack &stack, Json request, std::initializer_list<Peer *> peers = {},
                                              uint32_t timeoutMs = 15000)
    {
        auto done = std::make_shared<std::optional<WorkflowRecord>>();
        stack.m_Harness->Submit(std::move(request), "test", [done](const WorkflowRecord &record) { *done = record; });
        Pump(stack, peers, [&] { return done->has_value(); }, timeoutMs);
        return *done;
    }

    Json DeployDescriptor(const Descriptor::RichMLService &service)
    {
        return Json::MakeObject({{"kind", "deploy"}, {"descriptor", Descriptor::EmitDescriptor(service)}});
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

    // the first instance of `role` ("" when none)
    std::string InstanceOf(const Deployment &deployment, std::string_view role)
    {
        for (const Instance &instance : deployment.m_Instances)
        {
            if (instance.m_Role == role)
            {
                return instance.m_Id;
            }
        }
        return {};
    }

    // the newest workflow record of `kind` (nullptr when none)
    const WorkflowRecord *Newest(const StateStore &state, std::string_view kind)
    {
        for (const WorkflowRecord *record : state.Workflows())
        {
            if (record->m_Kind == kind)
            {
                return record;
            }
        }
        return nullptr;
    }

    // a workload heartbeat on Heartbeat(service, instance) as the runtimes publish it
    void Beat(Peer &peer, const Fabric::Topics &topics, const std::string &service, const std::string &instance,
              const std::string &role, int64_t model = 0, int64_t samples = 0)
    {
        Json body = Json::MakeObject({{"service", service},
                                      {"instance", instance},
                                      {"role", role},
                                      {"state", role == "inference" && model == 0 ? "waiting-model" : "running"},
                                      {"modelVersion", model},
                                      {"samples", samples}});
        EMPY_REQUIRE(!peer.Get().Publish(topics.Heartbeat(service, instance), Envelope::Make("status", body)).get());
    }

    void ListenEnvelopes(Peer &peer, const std::string &filter, std::vector<Envelope> &seen)
    {
        EMPY_REQUIRE(!peer.Get().Listen(filter, [&seen](const std::string &, const Envelope &envelope)
                                        { seen.push_back(envelope); }));
    }

    // a memory entry of `agent`/`kind` whose data matches `match`
    bool Remembered(Context::KnowledgePlane &plane, std::string_view agent, std::string_view kind,
                    const std::function<bool(const Json &)> &match = {})
    {
        for (const Context::Entry &entry : plane.Scan(Context::MEMORY))
        {
            if (entry.m_Data.GetString("agent") == agent && entry.m_Data.GetString("kind") == kind &&
                (!match || match(entry.m_Data["data"])))
            {
                return true;
            }
        }
        return false;
    }

    // a simulated substrate whose List reports chosen instances as exited (a crashed EmpyWorkload, a deleted pod)
    struct CrashingSubstrate : Substrate::SimulatedSubstrate
    {
        std::mutex m_Mutex; // List runs on a harness worker
        std::set<std::string> m_Exited;

        std::vector<Substrate::InstanceStatus> List() override
        {
            std::vector<Substrate::InstanceStatus> statuses = SimulatedSubstrate::List();
            std::scoped_lock lock(m_Mutex);
            for (Substrate::InstanceStatus &status : statuses)
            {
                if (m_Exited.count(status.m_Instance))
                {
                    status.m_State = "exited";
                }
            }
            return statuses;
        }

        void Crash(const std::string &instance)
        {
            std::scoped_lock lock(m_Mutex);
            m_Exited.insert(instance);
        }
    };

    // a reasoning provider answering every prompt with a fixed structured document
    struct ScriptedProvider : Intelligence::Provider
    {
        Json m_Answer;
        std::atomic<size_t> m_Calls{0};

        explicit ScriptedProvider(Json answer) : m_Answer(std::move(answer))
        {
        }
        std::string Name() const override
        {
            return "scripted";
        }
        bool Available() const override
        {
            return true;
        }
        Intelligence::Completion Complete(const Intelligence::Prompt &) override
        {
            ++m_Calls;
            Intelligence::Completion completion;
            completion.m_Ok = true;
            completion.m_Provider = "scripted";
            completion.m_Json = m_Answer;
            completion.m_Text = m_Answer.Dump();
            return completion;
        }
    };

    Workload::ModelArtifact Artifact(uint32_t version, double accuracy,
                                     std::vector<std::string> features = {"home/temp", "home/hum"})
    {
        Workload::ModelArtifact artifact;
        artifact.m_Service = "home-control";
        artifact.m_Version = version;
        artifact.m_Id = "home-control-v" + std::to_string(version);
        artifact.m_Weights.assign(features.size() + 1, 0.25);
        artifact.m_Features = std::move(features);
        artifact.m_Accuracy = accuracy;
        artifact.m_Samples = 64;
        artifact.m_TrainedAt = NowMillis();
        artifact.m_Dataset = "home-telemetry";
        return artifact;
    }

    // in-test stand-in for the workload contract (spec §12): Workload(service, train|model) responders
    struct FakeWorkload
    {
        Peer m_Peer;
        std::vector<Json> m_TrainRequests;
        std::vector<Json> m_Loads;
        std::function<Workload::ModelArtifact(uint32_t version)> m_Train;

        FakeWorkload(uint16_t port, const Fabric::Topics &topics, const std::string &service)
            : m_Peer(port, "fake-workload")
        {
            EMPY_REQUIRE(!m_Peer.Get().Serve(
                topics.Workload(service, "train"),
                [this](const Fabric::Inbound &request, Bus::Responder respond)
                {
                    m_TrainRequests.push_back(request.m_Envelope.m_Body);
                    uint32_t version = static_cast<uint32_t>(request.m_Envelope.m_Body.GetInt("version"));
                    Workload::ModelArtifact artifact = m_Train(version);
                    respond(Envelope::Make("artifact", Json::MakeObject({{"artifact", artifact.ToJson()}})), {});
                }));
            EMPY_REQUIRE(!m_Peer.Get().Serve(
                topics.Workload(service, "model"),
                [this](const Fabric::Inbound &request, Bus::Responder respond)
                {
                    const Json &body = request.m_Envelope.m_Body;
                    m_Loads.push_back(body);
                    respond(Envelope::Make("model", Json::MakeObject({{"loaded", body.At("artifact.version")
                                                                                     ? *body.At("artifact.version")
                                                                                     : Json()}})),
                            {});
                }));
        }
    };

    Topology::Inventory SmallInventory()
    {
        Topology::Inventory inventory;
        Topology::Node ran;
        ran.m_Id = "ran-1";
        ran.m_Region = "ran";
        ran.m_Cpu = 4;
        ran.m_Memory = 8ull << 30;
        ran.m_Accelerators["npu"] = 1;
        Topology::Node edge;
        edge.m_Id = "edge-1";
        edge.m_Region = "edge";
        edge.m_Cpu = 16;
        edge.m_Memory = 32ull << 30;
        edge.m_Accelerators["gpu"] = 1;
        Topology::Node cloud;
        cloud.m_Id = "cloud-1";
        cloud.m_Region = "cloud";
        cloud.m_Cpu = 64;
        cloud.m_Memory = 256ull << 30;
        cloud.m_Accelerators["gpu"] = 8;
        inventory.m_Nodes = {ran, edge, cloud};
        return inventory;
    }

    Agents::PlacementPlan Plan(Agents::Role role, std::vector<std::string> nodes)
    {
        Agents::PlacementPlan plan;
        plan.m_Service = "home-control";
        plan.m_Role = role;
        plan.m_Nodes = std::move(nodes);
        return plan;
    }

    double Or(const std::optional<double> &value, double fallback = -1e300)
    {
        return value.value_or(fallback);
    }

    Agents::LifecycleDecision Decision(Agents::LifecycleAction action, std::optional<uint32_t> target)
    {
        Agents::LifecycleDecision decision;
        decision.m_Action = action;
        decision.m_Target = target;
        return decision;
    }

    bool Mentions(const GuardResult &result, std::string_view text)
    {
        for (const std::string &violation : result.m_Violations)
        {
            if (violation.find(text) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }
} // namespace

EMPY_TEST(OpsHarness, GuardMatrix)
{
    Guard guard;
    Descriptor::RichMLService service = HomeControl();
    Descriptor::ValidationOptions options;
    options.m_Regions = {"ran", "edge", "cloud"};

    // descriptor: validator errors are violations
    EMPY_CHECK(guard.CheckDescriptor(service, options).m_Ok);
    Descriptor::RichMLService broken = service;
    broken.m_Actionables[0].m_Min = 30;
    broken.m_Inference->m_Regions = {"moon"};
    GuardResult descriptor = guard.CheckDescriptor(broken, options);
    EMPY_CHECK(!descriptor.m_Ok);
    EMPY_CHECK(descriptor.m_Violations.size() >= 2);
    EMPY_CHECK(descriptor.ToJson()["ok"].AsBool() == false);

    // compliance: only approved/amended, findings re-checked, amended descriptor re-validated
    Agents::ComplianceReport report;
    EMPY_CHECK(guard.CheckCompliance(report, options).m_Ok);
    report.m_Verdict = Agents::Verdict::Escalate;
    EMPY_CHECK(!guard.CheckCompliance(report, options).m_Ok);
    report.m_Verdict = Agents::Verdict::Rejected;
    EMPY_CHECK(!guard.CheckCompliance(report, options).m_Ok);
    report.m_Verdict = Agents::Verdict::Approved;
    report.m_Findings.push_back(Agents::Finding{"policy", "error", "denied topic", "home/set/door", ""});
    EMPY_CHECK(Mentions(guard.CheckCompliance(report, options), "contradicts"));
    report.m_Findings.clear();
    report.m_Verdict = Agents::Verdict::Amended;
    EMPY_CHECK(Mentions(guard.CheckCompliance(report, options), "without an amended descriptor"));
    report.m_Amended = broken;
    EMPY_CHECK(Mentions(guard.CheckCompliance(report, options), "amended descriptor"));
    report.m_Amended = service;
    EMPY_CHECK(guard.CheckCompliance(report, options).m_Ok);

    // placement: existence, count, distinctness, regions, avoid, pin, capacity, role and service
    Topology::Inventory inventory = SmallInventory();
    EMPY_CHECK(
        guard.CheckPlacement(service, Agents::Role::Inference, Plan(Agents::Role::Inference, {"ran-1"}), inventory)
            .m_Ok);
    EMPY_CHECK(
        guard.CheckPlacement(service, Agents::Role::Training, Plan(Agents::Role::Training, {"cloud-1"}), inventory)
            .m_Ok);
    EMPY_CHECK(Mentions(
        guard.CheckPlacement(service, Agents::Role::Inference, Plan(Agents::Role::Inference, {"nowhere"}), inventory),
        "unknown node"));
    EMPY_CHECK(
        Mentions(guard.CheckPlacement(service, Agents::Role::Inference, Plan(Agents::Role::Inference, {}), inventory),
                 "required"));
    EMPY_CHECK(Mentions(
        guard.CheckPlacement(service, Agents::Role::Inference, Plan(Agents::Role::Inference, {"cloud-1"}), inventory),
        "outside the declared"));
    EMPY_CHECK(Mentions(
        guard.CheckPlacement(service, Agents::Role::Training, Plan(Agents::Role::Training, {"ran-1"}), inventory),
        "outside the declared"));
    Descriptor::RichMLService replicated = service;
    replicated.m_Inference->m_Replicas = 2;
    EMPY_CHECK(Mentions(guard.CheckPlacement(replicated, Agents::Role::Inference,
                                             Plan(Agents::Role::Inference, {"ran-1", "ran-1"}), inventory),
                        "twice"));
    EMPY_CHECK(guard
                   .CheckPlacement(replicated, Agents::Role::Inference,
                                   Plan(Agents::Role::Inference, {"ran-1", "edge-1"}), inventory)
                   .m_Ok);
    Descriptor::RichMLService avoiding = service;
    avoiding.m_Constraints.m_Placement.m_Avoid = {"ran"};
    EMPY_CHECK(Mentions(
        guard.CheckPlacement(avoiding, Agents::Role::Inference, Plan(Agents::Role::Inference, {"ran-1"}), inventory),
        "avoided"));
    Descriptor::RichMLService pinned = service;
    pinned.m_Constraints.m_Placement.m_Nodes = {"edge-1"};
    EMPY_CHECK(Mentions(
        guard.CheckPlacement(pinned, Agents::Role::Inference, Plan(Agents::Role::Inference, {"ran-1"}), inventory),
        "pinned"));
    // capacity counts reservations already in the inventory
    Topology::Inventory busy = inventory;
    busy.Find("ran-1")->m_UsedCpu = 3.5;
    EMPY_CHECK(
        Mentions(guard.CheckPlacement(service, Agents::Role::Inference, Plan(Agents::Role::Inference, {"ran-1"}), busy),
                 "cannot host"));
    busy = inventory;
    busy.Find("cloud-1")->m_UsedAccelerators["gpu"] = 8;
    EMPY_CHECK(
        Mentions(guard.CheckPlacement(service, Agents::Role::Training, Plan(Agents::Role::Training, {"cloud-1"}), busy),
                 "gpu 0 free"));
    EMPY_CHECK(
        !guard.CheckPlacement(service, Agents::Role::Training, Plan(Agents::Role::Inference, {"cloud-1"}), inventory)
             .m_Ok);
    Agents::PlacementPlan foreign = Plan(Agents::Role::Inference, {"ran-1"});
    foreign.m_Service = "energy-saver";
    EMPY_CHECK(
        Mentions(guard.CheckPlacement(service, Agents::Role::Inference, foreign, inventory), "not 'home-control'"));
    Descriptor::RichMLService inferenceOnly = service;
    inferenceOnly.m_Training.reset();
    EMPY_CHECK(Mentions(guard.CheckPlacement(inferenceOnly, Agents::Role::Training,
                                             Plan(Agents::Role::Training, {"cloud-1"}), inventory),
                        "declares no training"));

    // workflow requests: kind, service, duplicates
    StateStore state;
    EMPY_CHECK(!guard.CheckWorkflowRequest(Json(5), state).m_Ok);
    EMPY_CHECK(
        Mentions(guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "explode"}}), state), "unknown workflow"));
    EMPY_CHECK(guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "deploy"}, {"intent", "x"}}), state).m_Ok);
    EMPY_CHECK(!guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "deploy"}}), state).m_Ok);
    EMPY_CHECK(
        !guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "deploy"}, {"intent", "x"}, {"descriptor", "y"}}), state)
             .m_Ok);
    EMPY_CHECK(
        Mentions(guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "retrain"}}), state), "needs a 'service'"));
    EMPY_CHECK(Mentions(
        guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}}), state),
        "not deployed"));
    Deployment deployment;
    deployment.m_Service = "home-control";
    deployment.m_Version = "1.0.0";
    deployment.m_Descriptor = service;
    state.PutDeployment(deployment);
    Json retrain = Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}});
    EMPY_CHECK(guard.CheckWorkflowRequest(retrain, state).m_Ok);
    WorkflowRecord &running = state.CreateWorkflow("retrain", "home-control", "test", retrain);
    EMPY_CHECK(Mentions(guard.CheckWorkflowRequest(retrain, state), "already running"));
    EMPY_CHECK(
        !guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}}), state).m_Ok);
    EMPY_CHECK(
        !guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}}), state).m_Ok);
    EMPY_CHECK(
        guard.CheckWorkflowRequest(Json::MakeObject({{"kind", "relocate"}, {"service", "home-control"}}), state).m_Ok);
    EMPY_CHECK(!guard
                    .CheckWorkflowRequest(
                        Json::MakeObject({{"kind", "relocate"}, {"service", "home-control"}, {"role", "gpu"}}), state)
                    .m_Ok);
    running.m_Status = "succeeded";
    state.SaveWorkflow(running);
    EMPY_CHECK(guard.CheckWorkflowRequest(retrain, state).m_Ok);
    EMPY_CHECK(!guard
                    .CheckWorkflowRequest(
                        Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}, {"target", -1}}), state)
                    .m_Ok);
    deployment.m_Descriptor.m_Training.reset();
    state.PutDeployment(deployment);
    EMPY_CHECK(Mentions(guard.CheckWorkflowRequest(retrain, state), "declares no training"));

    // retrain 'observed' is an accuracy; relocate 'replace' names instances
    deployment.m_Descriptor = service;
    state.PutDeployment(deployment);
    Json observed = retrain;
    observed.Set("observed", 1.5);
    EMPY_CHECK(Mentions(guard.CheckWorkflowRequest(observed, state), "[0, 1]"));
    observed.Set("observed", "high");
    EMPY_CHECK(Mentions(guard.CheckWorkflowRequest(observed, state), "[0, 1]"));
    observed.Set("observed", 0.4);
    EMPY_CHECK(guard.CheckWorkflowRequest(observed, state).m_Ok);
    Json replace = Json::MakeObject({{"kind", "relocate"}, {"service", "home-control"}});
    replace.Set("replace", Json::MakeArray({Json(5)}));
    EMPY_CHECK(Mentions(guard.CheckWorkflowRequest(replace, state), "instance ids"));
    replace.Set("replace", Json::MakeArray({Json("home-control-inference-ran-1-0")}));
    EMPY_CHECK(guard.CheckWorkflowRequest(replace, state).m_Ok);

    // promotion: decision, artefact contract (Workload::CheckArtifact), version, accuracy
    deployment.m_Descriptor = service;
    Agents::LifecycleDecision promote = Decision(Agents::LifecycleAction::Promote, 2);
    deployment.m_Lifecycle.m_Active = 1;
    EMPY_CHECK(guard.CheckPromotion(deployment, Artifact(2, 0.95), promote).m_Ok);
    EMPY_CHECK(
        Mentions(guard.CheckPromotion(deployment, Artifact(2, 0.95), Decision(Agents::LifecycleAction::Reject, 2)),
                 "not promote"));
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, Artifact(2, 0.95, {"home/temp"}), promote), "do not match"));
    Workload::ModelArtifact weights = Artifact(2, 0.95);
    weights.m_Weights.pop_back();
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, weights, promote), "weights"));
    weights = Artifact(2, 0.95);
    weights.m_Weights[1] = std::nan("");
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, weights, promote), "finite"));
    EMPY_CHECK(
        Mentions(guard.CheckPromotion(deployment, Artifact(1, 0.95), Decision(Agents::LifecycleAction::Promote, 1)),
                 "not newer"));
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, Artifact(2, 0.5), promote), "below the threshold"));
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, Artifact(3, 0.95), promote), "targets v2"));
    Workload::ModelArtifact other = Artifact(2, 0.95);
    other.m_Service = "energy-saver";
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, other, promote), "belongs to"));
    // the improvement over the active model is re-derived from its record (never from a requester's claim): 0.95
    // does not beat an active 0.97 by minImprovement 0.02; only the lower harness-observed accuracy changes that
    deployment.m_Lifecycle.m_History = {Agents::ModelRecord{1, "home-control-v1", 0.97, 10, "active"}};
    EMPY_CHECK(Mentions(guard.CheckPromotion(deployment, Artifact(2, 0.95), promote), "minimum"));
    EMPY_CHECK(guard.CheckPromotion(deployment, Artifact(2, 0.95), promote, 0.5).m_Ok);
    EMPY_CHECK(!guard.CheckPromotion(deployment, Artifact(2, 0.95), promote, 0.99).m_Ok);
    EMPY_CHECK(guard.CheckPromotion(deployment, Artifact(2, 0.995), promote).m_Ok);

    // rollback: decision, target exists, not active, artefact contract, promoted before, last known accuracy
    deployment.m_Artifacts.clear();
    for (uint32_t version = 1; version <= 5; ++version)
    {
        deployment.m_Artifacts[version] = Artifact(version, 0.95);
    }
    deployment.m_Lifecycle.m_Active = 2;
    deployment.m_Lifecycle.m_History = {Agents::ModelRecord{1, "home-control-v1", 0.93, 10, "retired"},
                                        Agents::ModelRecord{2, "home-control-v2", 0.95, 20, "active"},
                                        Agents::ModelRecord{3, "home-control-v3", 0.4, 30, "rejected"},
                                        Agents::ModelRecord{4, "home-control-v4", 0.97, 40, "candidate"}};
    Agents::LifecycleDecision rollback = Decision(Agents::LifecycleAction::Rollback, 1);
    EMPY_CHECK(guard.CheckRollback(deployment, rollback).m_Ok);
    EMPY_CHECK(Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, std::nullopt)),
                        "without a target"));
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, 7)), "no artefact"));
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, 2)), "already active"));
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Retrain, 1)), "not rollback"));
    // a rejected or never promoted model is no rollback target, whatever the artefact claims
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, 3)), "promoted before"));
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, 4)), "promoted before"));
    EMPY_CHECK(
        Mentions(guard.CheckRollback(deployment, Decision(Agents::LifecycleAction::Rollback, 5)), "never promoted"));
    // a model replaced because it degraded keeps the degraded accuracy as its last known one
    deployment.m_Lifecycle.m_History[0].m_Accuracy = 0.3;
    EMPY_CHECK(Mentions(guard.CheckRollback(deployment, rollback), "below the threshold"));
    deployment.m_Lifecycle.m_History[0].m_Accuracy = 0.93;
    deployment.m_Artifacts[1] = Artifact(1, 0.93, {"home/temp"});
    EMPY_CHECK(Mentions(guard.CheckRollback(deployment, rollback), "features"));
    deployment.m_Artifacts[1] = Artifact(1, 0.93);
    deployment.m_Artifacts[1].m_Weights[0] = std::numeric_limits<double>::infinity();
    EMPY_CHECK(Mentions(guard.CheckRollback(deployment, rollback), "finite"));
}

EMPY_TEST(OpsHarness, GuardAmendmentAndLatency)
{
    Guard guard;
    Descriptor::RichMLService service = HomeControl();
    Descriptor::ValidationOptions options;
    options.m_Regions = {"ran", "edge", "cloud"};

    // an amendment may only restrict the reviewed candidate: identity and every other section unchanged, each
    // actionable covered by a declared one with bounds, rate and QoS inside the declared ones
    Descriptor::RichMLService narrowed = service;
    narrowed.m_Actionables[0].m_Min = 18;
    narrowed.m_Actionables[0].m_Max = 22;
    narrowed.m_Actionables[0].m_MaxRatePerMinute = 30;
    EMPY_CHECK(guard.CheckAmendment(service, narrowed).m_Ok);
    Agents::ComplianceReport report;
    report.m_Verdict = Agents::Verdict::Amended;
    report.m_Amended = narrowed;
    EMPY_CHECK(guard.CheckCompliance(service, report, options).m_Ok);
    Descriptor::RichMLService amended = narrowed;
    amended.m_Metadata.m_Name = "energy-saver";
    report.m_Amended = amended;
    EMPY_CHECK(guard.CheckCompliance(report, options).m_Ok); // the candidate-free check cannot see a rename
    EMPY_CHECK(Mentions(guard.CheckCompliance(service, report, options), "renames"));
    amended = narrowed;
    amended.m_Metadata.m_Version = "0.0.1";
    EMPY_CHECK(Mentions(guard.CheckAmendment(service, amended), "changes the version"));
    amended = narrowed;
    amended.m_Actionables[0].m_Max = 30;
    EMPY_CHECK(Mentions(guard.CheckAmendment(service, amended), "widens the declared maximum"));
    amended = narrowed;
    amended.m_Actionables[0].m_Min.reset();
    EMPY_CHECK(Mentions(guard.CheckAmendment(service, amended), "widens the declared minimum"));
    amended = narrowed;
    amended.m_Actionables.push_back(amended.m_Actionables[0]);
    amended.m_Actionables.back().m_Name = "home/set/door";
    EMPY_CHECK(Mentions(guard.CheckAmendment(service, amended), "not covered"));
    amended = narrowed;
    amended.m_Inference->m_Regions.push_back("cloud");
    EMPY_CHECK(Mentions(guard.CheckAmendment(service, amended), "changes more than its actionables"));
    amended = narrowed;
    amended.m_Observables.pop_back();
    EMPY_CHECK(!guard.CheckAmendment(service, amended).m_Ok);
    Descriptor::RichMLService limited = service;
    limited.m_Actionables[0].m_MaxRatePerMinute = 6;
    amended = limited;
    amended.m_Actionables[0].m_MaxRatePerMinute = 0; // unlimited
    EMPY_CHECK(Mentions(guard.CheckAmendment(limited, amended), "rate limit"));
    amended.m_Actionables[0].m_MaxRatePerMinute = 3;
    EMPY_CHECK(guard.CheckAmendment(limited, amended).m_Ok);

    // the inference latency objective (p95 under 5 ms) is re-derived from the network: every node within 2.5 ms
    // of the data region (densest observable telemetry), whatever the placement agent claims
    Topology::Inventory inventory = SmallInventory();
    Topology::NetworkModel network;
    network.m_Regions = {"ran", "edge", "cloud"};
    network.m_LatencyMs["ran"]["edge"] = 5;
    network.m_LatencyMs["ran"]["cloud"] = 25;
    network.m_TopicDensity["home/#"]["ran"] = 40;
    Descriptor::RichMLService wide = service;
    wide.m_Inference->m_Regions = {"ran", "edge", "cloud"};
    EMPY_CHECK(
        guard
            .CheckPlacement(wide, Agents::Role::Inference, Plan(Agents::Role::Inference, {"ran-1"}), inventory, network)
            .m_Ok);
    EMPY_CHECK(
        guard.CheckPlacement(wide, Agents::Role::Inference, Plan(Agents::Role::Inference, {"cloud-1"}), inventory)
            .m_Ok);
    EMPY_CHECK(Mentions(guard.CheckPlacement(wide, Agents::Role::Inference, Plan(Agents::Role::Inference, {"cloud-1"}),
                                             inventory, network),
                        "latency objective"));
    EMPY_CHECK(Mentions(guard.CheckPlacement(wide, Agents::Role::Inference, Plan(Agents::Role::Inference, {"edge-1"}),
                                             inventory, network),
                        "data region 'ran'"));
    // without telemetry density the densest actionable consumers define the data region
    network.m_TopicDensity.clear();
    network.m_ConsumerDensity["home/set/#"]["cloud"] = 3;
    EMPY_CHECK(guard
                   .CheckPlacement(wide, Agents::Role::Inference, Plan(Agents::Role::Inference, {"cloud-1"}), inventory,
                                   network)
                   .m_Ok);
    // training has no latency objective
    EMPY_CHECK(guard
                   .CheckPlacement(service, Agents::Role::Training, Plan(Agents::Role::Training, {"cloud-1"}),
                                   inventory, network)
                   .m_Ok);
}

EMPY_TEST(OpsHarness, StatePersistence)
{
    TempDir temp;
    StateOptions options{temp.m_Path / "state", 3};
    std::string running, finished;
    {
        StateStore state(options);
        Deployment deployment;
        deployment.m_Service = "home-control";
        deployment.m_Version = "1.0.0";
        deployment.m_Descriptor = HomeControl();
        deployment.m_Status = "deployed";
        deployment.m_Instances.push_back(Instance{"home-control-inference-ran-1-0", "inference", "ran-1", "ready", 5});
        deployment.m_Lifecycle.m_Service = "home-control";
        deployment.m_Lifecycle.m_HasTraining = true;
        deployment.m_Lifecycle.m_Training = true; // a retrain in flight when the harness stops
        deployment.m_Lifecycle.m_Active = 1;
        deployment.m_Lifecycle.m_History.push_back(Agents::ModelRecord{1, "home-control-v1", 0.93, 10, "active"});
        deployment.m_Artifacts[1] = Artifact(1, 0.93);
        state.PutDeployment(deployment);

        Deployment other;
        other.m_Service = "energy-saver";
        other.m_Version = "1.0.0";
        other.m_Descriptor = EnergySaver();
        state.PutDeployment(other);
        EMPY_CHECK(state.EraseDeployment("energy-saver"));
        EMPY_CHECK(!state.EraseDeployment("energy-saver"));

        Descriptor::Resources resources;
        resources.m_Cpu = 2;
        resources.m_Memory = 4ull << 30;
        resources.m_Accelerator = "gpu";
        resources.m_AcceleratorCount = 1;
        state.Reserve("home-control-training-cloud-1-0", "cloud-1", resources);
        state.Reserve("ghost", "ran-1", resources);
        EMPY_CHECK(state.Release("ghost"));
        EMPY_CHECK(!state.Release("ghost"));

        for (int32_t i = 0; i < 5; ++i)
        {
            WorkflowRecord &record = state.CreateWorkflow("deploy", "svc-" + std::to_string(i), "test", Json(i));
            record.m_Status = "succeeded";
            record.m_Steps.push_back(StepRecord{"validate", "succeeded", 1, 2, Json::MakeObject({{"ok", true}})});
            state.SaveWorkflow(record);
            finished = record.m_Id;
        }
        running = state.CreateWorkflow("retrain", "home-control", "drift", Json()).m_Id;
        // the bound keeps the 3 newest finished records plus the running one
        EMPY_CHECK_EQ(state.Workflows().size(), size_t(4));
        EMPY_CHECK_EQ(state.Workflows().front()->m_Id, running);
        EMPY_CHECK(state.HasRunning("retrain", "home-control"));
        EMPY_CHECK(state.HasRunning("", "home-control"));
        EMPY_CHECK(!state.HasRunning("deploy", "home-control"));
        // a second store on the same directory is refused (exclusive FileStore owner lock)
        EMPY_CHECK_THROWS(StateStore(options));
    }
    StateStore reloaded(options);
    const Deployment *deployment = reloaded.FindDeployment("home-control");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK(reloaded.FindDeployment("energy-saver") == nullptr);
    EMPY_CHECK_EQ(deployment->m_Version, std::string("1.0.0"));
    EMPY_CHECK_EQ(deployment->m_Descriptor.m_Metadata.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(deployment->m_Descriptor.m_Actionables.size(), size_t(1));
    EMPY_REQUIRE_EQ(deployment->m_Instances.size(), size_t(1));
    EMPY_CHECK_EQ(deployment->m_Instances[0].m_Node, std::string("ran-1"));
    EMPY_CHECK_EQ(deployment->m_Instances[0].m_LastSeen, uint64_t(5));
    EMPY_CHECK(deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(1));
    EMPY_CHECK(!deployment->m_Lifecycle.m_Training); // cannot resume after a restart
    EMPY_REQUIRE_EQ(deployment->m_Artifacts.count(1), size_t(1));
    EMPY_CHECK_EQ(deployment->m_Artifacts.at(1).m_Weights.size(), size_t(3));
    EMPY_CHECK(deployment->m_Updated > 0);

    Json reservations = reloaded.Reservations();
    EMPY_REQUIRE_EQ(reservations.Size(), size_t(1));
    EMPY_CHECK_EQ(reservations.AsArray().front().GetString("node"), std::string("cloud-1"));
    Topology::Inventory inventory = reloaded.ApplyReservations(SmallInventory());
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->m_UsedCpu, 2.0);
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->m_UsedAccelerators["gpu"], uint32_t(1));
    EMPY_CHECK_EQ(inventory.Find("ran-1")->m_UsedCpu, 0.0);
    inventory = reloaded.ApplyReservationsExcept(SmallInventory(), {"home-control-training-cloud-1-0"});
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->m_UsedCpu, 0.0);

    // the bound now holds: the failed record counts as finished and the oldest finished one is evicted
    auto workflows = reloaded.Workflows();
    EMPY_REQUIRE_EQ(workflows.size(), size_t(3));
    EMPY_CHECK_EQ(workflows.front()->m_Id, running);
    // the running record of the stopped harness is failed on reload
    EMPY_CHECK_EQ(workflows.front()->m_Status, std::string("failed"));
    EMPY_CHECK(workflows.front()->m_Error.find("restarted") != std::string::npos);
    EMPY_CHECK_EQ(workflows[1]->m_Id, finished);
    EMPY_REQUIRE_EQ(workflows[1]->m_Steps.size(), size_t(1));
    EMPY_CHECK(workflows[1]->m_Steps[0].m_Detail["ok"].AsBool());
    EMPY_CHECK(!reloaded.HasRunning("", "home-control"));
    Json json = reloaded.ToJson(2);
    EMPY_CHECK_EQ(json["deployments"].Size(), size_t(1));
    EMPY_CHECK_EQ(json["workflows"].Size(), size_t(2));
    EMPY_CHECK_EQ(json["reservations"].Size(), size_t(1));
    EMPY_CHECK(Deployment::FromJson(deployment->ToJson()).ToJson() == deployment->ToJson());
    EMPY_CHECK_THROWS(Deployment::FromJson(Json::MakeObject({{"service", ""}})));
    EMPY_CHECK_THROWS(WorkflowRecord::FromJson(Json::MakeObject({{"kind", "deploy"}})));
}

EMPY_TEST(OpsHarness, SchedulerTicks)
{
    std::vector<std::pair<std::string, uint64_t>> fired;
    Scheduler scheduler([&](const ScheduleEntry &entry, SystemClock::time_point)
                        { fired.emplace_back(entry.m_Service + "/" + entry.m_Job, entry.m_Fired); });
    auto t0 = SystemClock::time_point(std::chrono::seconds(1700000000));
    scheduler.AddCron("home-control", "training", "@every 2s", t0);
    scheduler.AddCron("energy-saver", "training", "0 3 * * *", t0);
    scheduler.AddInterval("home-control", "placement", std::chrono::milliseconds(1500), t0);
    EMPY_CHECK_THROWS(scheduler.AddCron("x", "training", "not cron", t0));
    EMPY_CHECK_THROWS(scheduler.AddInterval("x", "placement", std::chrono::milliseconds(0), t0));
    EMPY_CHECK_THROWS(scheduler.AddCron("", "training", "@hourly", t0));
    EMPY_CHECK_EQ(scheduler.Entries().size(), size_t(3));
    EMPY_CHECK(*scheduler.NextDue() == t0 + std::chrono::milliseconds(1500));
    EMPY_CHECK_EQ(scheduler.Tick(t0 + std::chrono::milliseconds(1000)), size_t(0));
    EMPY_CHECK_EQ(scheduler.Tick(t0 + std::chrono::milliseconds(2000)), size_t(2));
    // a long stall collapses the missed firings into one per job
    EMPY_CHECK_EQ(scheduler.Tick(t0 + std::chrono::seconds(30)), size_t(2));
    EMPY_CHECK_EQ(fired.size(), size_t(4));
    for (const ScheduleEntry &entry : scheduler.Entries())
    {
        EMPY_CHECK(entry.m_Next > t0 + std::chrono::seconds(30));
    }
    EMPY_CHECK_EQ(scheduler.Tick(t0 + std::chrono::seconds(30)), size_t(0));
    Json json = scheduler.ToJson();
    EMPY_REQUIRE_EQ(json.Size(), size_t(3));
    EMPY_CHECK_EQ(json.AsArray().front().GetString("service"),
                  std::string("energy-saver")); // ordered by (service, job)
    EMPY_CHECK_EQ(json[1].GetString("schedule"), std::string("@every 1500ms"));
    EMPY_CHECK_EQ(json[2].GetString("schedule"), std::string("@every 2s"));
    EMPY_CHECK_EQ(json[1].GetInt("fired"), int64_t(2));
    EMPY_CHECK_EQ(json[2].GetInt("fired"), int64_t(2));
    // a replaced entry restarts; removal by job and by service
    scheduler.AddCron("home-control", "training", "@hourly", t0);
    EMPY_CHECK_EQ(scheduler.Entries()[2].m_Fired, uint64_t(0));
    EMPY_CHECK(scheduler.Remove("home-control", "training"));
    EMPY_CHECK(!scheduler.Remove("home-control", "training"));
    EMPY_CHECK_EQ(scheduler.RemoveService("home-control"), size_t(1));
    EMPY_CHECK_EQ(scheduler.Entries().size(), size_t(1));
    // a callback may remove entries while firing
    Scheduler removing([&](const ScheduleEntry &, SystemClock::time_point) {});
    removing.AddInterval("a", "x", std::chrono::seconds(1), t0);
    EMPY_CHECK_EQ(removing.Tick(t0 + std::chrono::seconds(1)), size_t(1));
    EMPY_CHECK(!Scheduler([](const ScheduleEntry &, SystemClock::time_point) {}).NextDue().has_value());
}

EMPY_TEST(OpsHarness, DeployDescriptorEndToEnd)
{
    Stack stack;
    Peer operatorPeer(stack.m_Broker.m_Port, "operator");
    // the descriptor arrives as an MMQP request on Deploy() (EmpyHarness deploy)
    std::string yaml = ReadFile(ConfigFile("HomeControl.yaml"));
    auto [reason, reply] = operatorPeer.Get().CallSync(
        stack.m_Topics.Deploy(), Envelope::Make("deploy", Json::MakeObject({{"descriptor", yaml}})),
        [&]
        {
            stack.m_Client->Poll(1);
            operatorPeer.m_Client->Poll(1);
        },
        15000);
    EMPY_REQUIRE(!reason);
    EMPY_REQUIRE(reply.has_value());
    EMPY_CHECK_EQ(reply->m_Kind, std::string("workflow"));
    WorkflowRecord record = WorkflowRecord::FromJson(reply->m_Body);
    EMPY_REQUIRE_EQ(record.m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(record.m_Kind, std::string("deploy"));
    EMPY_CHECK_EQ(record.m_Trigger, std::string("descriptor"));
    EMPY_CHECK_EQ(record.m_Service, std::string("home-control"));
    for (const char *name : {"parse", "validate", "review", "place-inference", "place-training", "enact", "schedule"})
    {
        const StepRecord *step = Step(record, name);
        EMPY_CHECK(step != nullptr);
        EMPY_CHECK(step && step->m_Status == "succeeded");
    }
    EMPY_CHECK_EQ(record.m_Result.GetString("compliance"), std::string("approved"));

    const Deployment *deployment = stack.m_Harness->State().FindDeployment("home-control");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK_EQ(deployment->m_Status, std::string("deployed"));
    EMPY_CHECK_EQ(deployment->m_Version, std::string("1.0.0"));
    EMPY_CHECK(NodesOf(*deployment, "inference") == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(NodesOf(*deployment, "training") == std::vector<std::string>({"cloud-1"}));
    EMPY_CHECK(deployment->m_Lifecycle.m_HasTraining);
    EMPY_CHECK(!deployment->m_Lifecycle.m_Active.has_value());
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    for (const Substrate::DeploymentSpec &spec : stack.m_Substrate->Deployed())
    {
        EMPY_CHECK_EQ(spec.m_BrokerPort, stack.m_Broker.m_Port);
        EMPY_CHECK_EQ(spec.m_TopicRoot, std::string("mlops"));
        EMPY_CHECK(spec.m_Region == (spec.m_Role == "inference" ? "ran" : "cloud"));
    }
    // reservations against the CoMaS resource registry
    Json reservations = stack.m_Harness->State().Reservations();
    EMPY_CHECK_EQ(reservations.Size(), size_t(2));
    Topology::Inventory inventory = stack.m_Harness->State().ApplyReservations(stack.m_Plane.Inventory());
    EMPY_CHECK_EQ(inventory.Find("ran-1")->m_UsedCpu, 1.0);
    EMPY_CHECK_EQ(inventory.Find("ran-1")->m_UsedMemory, uint64_t(1) << 30);
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->m_UsedCpu, 2.0);
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->m_UsedAccelerators["gpu"], uint32_t(1));
    // the training schedule is registered
    auto entries = stack.m_Harness->Schedules().Entries();
    EMPY_REQUIRE_EQ(entries.size(), size_t(1));
    EMPY_CHECK_EQ(entries[0].m_Service, std::string("home-control"));
    EMPY_CHECK_EQ(entries[0].m_Job, std::string("training"));
    EMPY_CHECK_EQ(entries[0].m_Expression, std::string("0 0 * * *"));

    // a late subscriber receives the retained state publication
    Peer late(stack.m_Broker.m_Port, "late");
    std::optional<Envelope> state;
    EMPY_REQUIRE(!late.Get().Listen(stack.m_Topics.State("services"),
                                    [&](const std::string &, const Envelope &envelope) { state = envelope; }));
    EMPY_REQUIRE(Pump(stack, {&late}, [&] { return state.has_value(); }));
    EMPY_CHECK_EQ(state->m_Kind, std::string("state"));
    EMPY_REQUIRE_EQ(state->m_Body["deployments"].Size(), size_t(1));
    EMPY_CHECK_EQ(state->m_Body["deployments"].AsArray().front().GetString("service"), std::string("home-control"));
    EMPY_CHECK_EQ(state->m_Body["schedules"].Size(), size_t(1));
    EMPY_CHECK_EQ(state->m_Body.GetString("identity"), std::string("harness"));

    // status request/response
    auto [statusReason, snapshot] = operatorPeer.Get().CallSync(
        stack.m_Topics.Status(), Envelope::Make("status-request", Json()),
        [&]
        {
            stack.m_Client->Poll(1);
            operatorPeer.m_Client->Poll(1);
        },
        5000);
    EMPY_REQUIRE(!statusReason);
    EMPY_REQUIRE(snapshot.has_value());
    EMPY_CHECK_EQ(snapshot->m_Kind, std::string("snapshot"));
    EMPY_CHECK_EQ(snapshot->m_Body["deployments"].Size(), size_t(1));
    EMPY_CHECK(snapshot->m_Body["workflows"].Size() >= 1);
    EMPY_CHECK_EQ(snapshot->m_Body["substrate"].GetString("type"), std::string("simulated"));
    EMPY_CHECK_EQ(snapshot->m_Body["inventory"]["nodes"].Size(), stack.m_Plane.Inventory().m_Nodes.size());

    // the same version is not redeployed; a newer one replaces the instances
    auto again = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(again.has_value());
    EMPY_CHECK_EQ(again->m_Status, std::string("rejected"));
    EMPY_CHECK(again->m_Error.find("not newer") != std::string::npos);
    Descriptor::RichMLService newer = HomeControl();
    newer.m_Metadata.m_Version = "1.1.0";
    auto upgrade = RunWorkflow(stack, DeployDescriptor(newer));
    EMPY_REQUIRE(upgrade.has_value());
    EMPY_REQUIRE_EQ(upgrade->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(upgrade->m_Result["replaced"].Size(), size_t(2));
    EMPY_CHECK(Step(*upgrade, "retire-previous") != nullptr);
    deployment = stack.m_Harness->State().FindDeployment("home-control");
    EMPY_CHECK_EQ(deployment->m_Version, std::string("1.1.0"));
    EMPY_CHECK(NodesOf(*deployment, "inference") == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    EMPY_CHECK_EQ(stack.m_Harness->State().Reservations().Size(), size_t(2));
    std::set<std::string> instances;
    for (const Instance &instance : deployment->m_Instances)
    {
        instances.insert(instance.m_Id);
    }
    for (const Json &replaced : upgrade->m_Result["replaced"].AsArray())
    {
        EMPY_CHECK(!instances.count(replaced.AsString()));
    }
}

EMPY_TEST(OpsHarness, DeployIntentEndToEnd)
{
    Stack stack;
    Peer operatorPeer(stack.m_Broker.m_Port, "operator");
    // a dry run over MMQP stops after placement
    auto [reason, reply] = operatorPeer.Get().CallSync(
        stack.m_Topics.Intent(),
        Envelope::Make("intent", Json::MakeObject({{"intent", LISTING_INTENT}, {"dryRun", true}})),
        [&]
        {
            stack.m_Client->Poll(1);
            operatorPeer.m_Client->Poll(1);
        },
        15000);
    EMPY_REQUIRE(!reason);
    EMPY_REQUIRE(reply.has_value());
    WorkflowRecord dry = WorkflowRecord::FromJson(reply->m_Body);
    EMPY_REQUIRE_EQ(dry.m_Status, std::string("succeeded"));
    EMPY_CHECK(dry.m_Result.GetBool("dryRun"));
    EMPY_CHECK(Step(dry, "enact") == nullptr);
    EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") == nullptr);
    EMPY_CHECK(stack.m_Substrate->Deployed().empty());
    EMPY_CHECK_EQ(stack.m_Harness->State().Reservations().Size(), size_t(0));

    auto record = RunWorkflow(stack, Json::MakeObject({{"kind", "deploy"}, {"intent", LISTING_INTENT}}));
    EMPY_REQUIRE(record.has_value());
    EMPY_REQUIRE_EQ(record->m_Status, std::string("succeeded"));
    const StepRecord *interpret = Step(*record, "interpret");
    EMPY_REQUIRE(interpret != nullptr);
    EMPY_CHECK(interpret->m_Detail["valid"].AsBool());
    EMPY_CHECK_EQ(interpret->m_Detail["frame"].GetString("name"), std::string("home-control"));
    // the generated BoM equals Listing 2.1 (Configs/MLOps/HomeControl.yaml)
    const Deployment *deployment = stack.m_Harness->State().FindDeployment("home-control");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK(Descriptor::ToJson(deployment->m_Descriptor) == Descriptor::ToJson(HomeControl()));
    EMPY_CHECK(NodesOf(*deployment, "inference") == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(NodesOf(*deployment, "training") == std::vector<std::string>({"cloud-1"}));
    EMPY_CHECK_EQ(stack.m_Harness->State().Reservations().Size(), size_t(2));
    auto entries = stack.m_Harness->Schedules().Entries();
    EMPY_REQUIRE_EQ(entries.size(), size_t(1));
    EMPY_CHECK_EQ(entries[0].m_Expression, std::string("0 0 * * *"));
    EMPY_CHECK_EQ(record->m_Result["placement"]["inference"].AsArray().front().AsString(), std::string("ran-1"));
    // the Executive remembered the interpretation and the harness the workflow (CoMaS agents memory)
    std::set<std::string> agents;
    for (const Context::Entry &entry : stack.m_Plane.Scan(Context::MEMORY))
    {
        agents.insert(entry.m_Data.GetString("agent"));
    }
    for (const char *agent : {"executive", "compliance", "placement", "harness"})
    {
        EMPY_CHECK(agents.count(agent));
    }
    Json snapshot = stack.m_Harness->Snapshot();
    EMPY_CHECK_EQ(snapshot["deployments"].Size(), size_t(1));
    EMPY_CHECK_EQ(snapshot["workflows"].Size(), size_t(2));
    EMPY_CHECK_EQ(snapshot["workflows"].AsArray().front().GetString("id"), record->m_Id); // latest first
}

EMPY_TEST(OpsHarness, ScheduleTriggerPublished)
{
    // no lifecycle role here: the trigger must reach a plain subscriber, not start retraining
    StackOptions options;
    options.m_Roles = {"executive", "placement", "compliance"};
    options.m_PlacementAuditSeconds = 1;
    Stack stack(options);
    Peer subscriber(stack.m_Broker.m_Port, "subscriber");
    std::vector<std::pair<std::string, Envelope>> training, audits;
    EMPY_REQUIRE(!subscriber.Get().Listen(stack.m_Topics.TriggerFilter("training"),
                                          [&](const std::string &topic, const Envelope &envelope)
                                          { training.emplace_back(topic, envelope); }));
    EMPY_REQUIRE(!subscriber.Get().Listen(stack.m_Topics.TriggerFilter("placement"),
                                          [&](const std::string &topic, const Envelope &envelope)
                                          { audits.emplace_back(topic, envelope); }));
    Descriptor::RichMLService service = HomeControl();
    service.m_Training->m_Schedule = "@every 1s";
    auto record = RunWorkflow(stack, DeployDescriptor(service), {&subscriber});
    EMPY_REQUIRE(record.has_value());
    EMPY_REQUIRE_EQ(record->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(record->m_Result["schedules"].Size(), size_t(2));
    EMPY_REQUIRE(Pump(stack, {&subscriber}, [&] { return training.size() >= 2 && !audits.empty(); }, 8000));
    EMPY_CHECK_EQ(training[0].first, std::string("mlops/trigger/time/home-control/training"));
    EMPY_CHECK_EQ(training[0].second.m_Kind, std::string("trigger"));
    EMPY_CHECK_EQ(training[0].second.m_Body.GetString("service"), std::string("home-control"));
    EMPY_CHECK_EQ(training[0].second.m_Body.GetString("job"), std::string("training"));
    EMPY_CHECK_EQ(training[0].second.m_Body.GetString("schedule"), std::string("@every 1s"));
    EMPY_CHECK(training[0].second.m_Body.GetInt("firedAt") > 0);
    EMPY_CHECK_EQ(training[0].second.m_Source, std::string("harness"));
    EMPY_CHECK_EQ(audits[0].first, std::string("mlops/trigger/time/home-control/placement"));
    // the placement audit found nothing better: no relocation was proposed
    Pump(stack, {&subscriber}, [] { return false; }, 300);
    for (const WorkflowRecord *workflow : stack.m_Harness->State().Workflows())
    {
        EMPY_CHECK(workflow->m_Kind == "deploy");
    }
    for (const ScheduleEntry &entry : stack.m_Harness->Schedules().Entries())
    {
        EMPY_CHECK(entry.m_Fired >= 1);
    }
    // retiring the service removes its temporal triggers and releases its reservations
    auto retire =
        RunWorkflow(stack, Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}}), {&subscriber});
    EMPY_REQUIRE(retire.has_value());
    EMPY_CHECK_EQ(retire->m_Status, std::string("succeeded"));
    EMPY_CHECK(stack.m_Harness->Schedules().Entries().empty());
    EMPY_CHECK_EQ(stack.m_Harness->State().Reservations().Size(), size_t(0));
    EMPY_CHECK(stack.m_Substrate->Deployed().empty());
}

EMPY_TEST(OpsHarness, HitlApprove)
{
    Stack stack;
    Peer operatorPeer(stack.m_Broker.m_Port, "operator");
    std::vector<Json> questions;
    EMPY_REQUIRE(!operatorPeer.Get().Serve(
        stack.m_Topics.Hitl(),
        [&](const Fabric::Inbound &request, Bus::Responder respond)
        {
            questions.push_back(request.m_Envelope.m_Body);
            respond(Envelope::Make(
                        "hitl-decision",
                        Json::MakeObject({{"decision", "approve"}, {"operator", "alice"}, {"note", "comfort wins"}})),
                    {});
        }));
    auto first = RunWorkflow(stack, DeployDescriptor(EnergySaver()), {&operatorPeer});
    EMPY_REQUIRE(first.has_value());
    EMPY_REQUIRE_EQ(first->m_Status, std::string("succeeded"));
    EMPY_CHECK(questions.empty());
    // HomeControl (priority 0) after EnergySaver (priority -1) on home/set/temp: escalated to a human
    auto record = RunWorkflow(stack, DeployDescriptor(HomeControl()), {&operatorPeer});
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("succeeded"));
    EMPY_REQUIRE_EQ(questions.size(), size_t(1));
    EMPY_CHECK_EQ(questions[0].GetString("service"), std::string("home-control"));
    EMPY_CHECK_EQ(questions[0].GetString("workflowId"), record->m_Id);
    EMPY_CHECK(questions[0].GetString("question").find("energy-saver") != std::string::npos);
    EMPY_CHECK(questions[0]["findings"].Size() > 0);
    const StepRecord *review = Step(*record, "review");
    EMPY_REQUIRE(review != nullptr);
    EMPY_CHECK_EQ(review->m_Detail.GetString("verdict"), std::string("escalate"));
    const StepRecord *hitl = Step(*record, "hitl");
    EMPY_REQUIRE(hitl != nullptr);
    EMPY_CHECK_EQ(hitl->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("decision"), std::string("approve"));
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("operator"), std::string("alice"));
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("source"), std::string("operator"));
    EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") != nullptr);
    EMPY_CHECK_EQ(stack.m_Harness->State().Deployments().size(), size_t(2));
}

EMPY_TEST(OpsHarness, HitlReject)
{
    Stack stack;
    Peer operatorPeer(stack.m_Broker.m_Port, "operator");
    EMPY_REQUIRE(!operatorPeer.Get().Serve(
        stack.m_Topics.Hitl(),
        [&](const Fabric::Inbound &, Bus::Responder respond)
        {
            respond(Envelope::Make("hitl-decision", Json::MakeObject({{"decision", "reject"}, {"operator", "bob"}})),
                    {});
        }));
    auto first = RunWorkflow(stack, DeployDescriptor(EnergySaver()), {&operatorPeer});
    EMPY_REQUIRE(first && first->m_Status == "succeeded");
    auto record = RunWorkflow(stack, DeployDescriptor(HomeControl()), {&operatorPeer});
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
    EMPY_CHECK(record->m_Error.find("operator 'bob'") != std::string::npos);
    const StepRecord *hitl = Step(*record, "hitl");
    EMPY_REQUIRE(hitl != nullptr);
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("decision"), std::string("reject"));
    EMPY_CHECK(Step(*record, "enact") == nullptr);
    EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") == nullptr);
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2)); // energy-saver only
}

EMPY_TEST(OpsHarness, HitlTimeoutDefault)
{
    // an operator that never answers: the configured default (reject) applies at the HITL timeout
    {
        StackOptions options;
        options.m_HitlTimeoutMs = 600;
        Stack stack(options);
        Peer silent(stack.m_Broker.m_Port, "operator");
        std::vector<Bus::Responder> parked;
        EMPY_REQUIRE(!silent.Get().Serve(stack.m_Topics.Hitl(), [&](const Fabric::Inbound &, Bus::Responder respond)
                                         { parked.push_back(std::move(respond)); }));
        auto first = RunWorkflow(stack, DeployDescriptor(EnergySaver()), {&silent});
        EMPY_REQUIRE(first && first->m_Status == "succeeded");
        auto started = Clock::now();
        std::optional<WorkflowRecord> done;
        std::string id = stack.m_Harness->Submit(DeployDescriptor(HomeControl()), "test",
                                                 [&](const WorkflowRecord &record) { done = record; });
        // while the question is open the workflow waits for the human
        EMPY_REQUIRE(Pump(stack, {&silent}, [&] { return !parked.empty(); }));
        EMPY_CHECK_EQ(stack.m_Harness->State().FindWorkflow(id)->m_Status, std::string("waiting-hitl"));
        EMPY_REQUIRE(Pump(stack, {&silent}, [&] { return done.has_value(); }, 5000));
        EMPY_CHECK(Clock::now() - started >= std::chrono::milliseconds(500));
        EMPY_CHECK_EQ(done->m_Status, std::string("rejected"));
        EMPY_CHECK(done->m_Error.find("by default") != std::string::npos);
        const StepRecord *hitl = Step(*done, "hitl");
        EMPY_REQUIRE(hitl != nullptr);
        EMPY_CHECK_EQ(hitl->m_Detail.GetString("source"), std::string("default"));
        EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") == nullptr);
        // a late answer changes nothing
        parked.front()(Envelope::Make("hitl-decision", Json::MakeObject({{"decision", "approve"}})), {});
        Pump(stack, {&silent}, [] { return false; }, 200);
        EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") == nullptr);
    }
    // no responder at all with an approving default: NO_RESPONDER resolves at once to approve
    {
        StackOptions options;
        options.m_HitlTimeoutMs = 5000;
        options.m_HitlDefaultApprove = true;
        Stack stack(options);
        auto first = RunWorkflow(stack, DeployDescriptor(EnergySaver()));
        EMPY_REQUIRE(first && first->m_Status == "succeeded");
        auto started = Clock::now();
        auto record = RunWorkflow(stack, DeployDescriptor(HomeControl()));
        EMPY_REQUIRE(record.has_value());
        EMPY_CHECK(Clock::now() - started < std::chrono::milliseconds(4000));
        EMPY_CHECK_EQ(record->m_Status, std::string("succeeded"));
        const StepRecord *hitl = Step(*record, "hitl");
        EMPY_REQUIRE(hitl != nullptr);
        EMPY_CHECK_EQ(hitl->m_Detail.GetString("decision"), std::string("approve"));
        EMPY_CHECK_EQ(hitl->m_Detail.GetString("source"), std::string("default"));
        EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") != nullptr);
    }
}

EMPY_TEST(OpsHarness, ConflictingServicePolicy)
{
    Stack stack;
    auto first = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(first && first->m_Status == "succeeded");
    // EnergySaver (lower priority) on the actionable HomeControl already drives: nothing left after the drop
    auto record = RunWorkflow(stack, DeployDescriptor(EnergySaver()));
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
    EMPY_CHECK(record->m_Error.find("compliance rejected") != std::string::npos);
    EMPY_CHECK(record->m_Error.find("conflict") != std::string::npos);
    EMPY_CHECK(stack.m_Harness->State().FindDeployment("energy-saver") == nullptr);

    // with a second, uncontested actionable the conflicting one is dropped automatically (amended)
    Descriptor::RichMLService saver = EnergySaver();
    Descriptor::Actionable light;
    light.m_Name = "home/set/light";
    light.m_Unit = "percent";
    light.m_Min = 0;
    light.m_Max = 100;
    saver.m_Actionables.push_back(light);
    saver.m_Metadata.m_Intent += " and dim the lighting";
    auto amended = RunWorkflow(stack, DeployDescriptor(saver));
    EMPY_REQUIRE(amended.has_value());
    EMPY_REQUIRE_EQ(amended->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(amended->m_Result.GetString("compliance"), std::string("amended"));
    EMPY_CHECK(amended->m_Result.GetBool("amended"));
    const Deployment *deployment = stack.m_Harness->State().FindDeployment("energy-saver");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_REQUIRE_EQ(deployment->m_Descriptor.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(deployment->m_Descriptor.m_Actionables[0].m_Name, std::string("home/set/light"));
    const StepRecord *review = Step(*amended, "review");
    EMPY_REQUIRE(review != nullptr);
    bool dropped = false;
    for (const Json &resolution : review->m_Detail["resolutions"].AsArray())
    {
        dropped =
            dropped || (resolution.GetString("type") == "drop" && resolution.GetString("topic") == "home/set/temp");
    }
    EMPY_CHECK(dropped);
}

EMPY_TEST(OpsHarness, RetrainPromoteRejectRollbackRetire)
{
    Stack stack;
    Descriptor::RichMLService service = HomeControl();
    service.m_Lifecycle.m_CooldownSeconds = 0; // consecutive promotions in one test
    auto deployed = RunWorkflow(stack, DeployDescriptor(service));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    FakeWorkload workload(stack.m_Broker.m_Port, stack.m_Topics, "home-control");
    std::map<uint32_t, Workload::ModelArtifact> plan = {
        {1, Artifact(1, 0.92)}, {2, Artifact(2, 0.93)}, {3, Artifact(3, 0.97)}, {4, Artifact(4, 0.99, {"home/temp"})}};
    workload.m_Train = [&](uint32_t version) { return plan.at(version); };
    Json retrain = Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}, {"reason", "test"}});
    const StateStore &state = stack.m_Harness->State();

    // v1: first model meeting the threshold is promoted and loaded into the inference instance
    auto v1 = RunWorkflow(stack, retrain, {&workload.m_Peer});
    EMPY_REQUIRE(v1.has_value());
    EMPY_REQUIRE_EQ(v1->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(v1->m_Result.GetInt("version"), int64_t(1));
    EMPY_CHECK_EQ(v1->m_Result.GetInt("loaded"), int64_t(1));
    for (const char *name : {"train", "evaluate", "load"})
    {
        EMPY_CHECK(Step(*v1, name) && Step(*v1, name)->m_Status == "succeeded");
    }
    EMPY_REQUIRE_EQ(workload.m_TrainRequests.size(), size_t(1));
    EMPY_CHECK_EQ(workload.m_TrainRequests[0].GetInt("version"), int64_t(1));
    EMPY_CHECK_EQ(workload.m_TrainRequests[0].GetString("workflowId"), v1->m_Id);
    EMPY_REQUIRE_EQ(workload.m_Loads.size(), size_t(1));
    EMPY_CHECK_EQ(workload.m_Loads[0].GetString("action"), std::string("load"));
    EMPY_CHECK_EQ(workload.m_Loads[0]["artifact"].GetInt("version"), int64_t(1));
    const Deployment *deployment = state.FindDeployment("home-control");
    EMPY_CHECK(deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(1));
    EMPY_CHECK(!deployment->m_Lifecycle.m_Training);
    EMPY_CHECK(deployment->m_Lifecycle.m_LastPromotion > 0);
    EMPY_CHECK_EQ(deployment->m_Lifecycle.ActiveRecord()->m_Status, std::string("active"));
    EMPY_CHECK_EQ(deployment->m_Artifacts.count(1), size_t(1));

    // v2: improvement 0.01 < minImprovement 0.02 → rejected, nothing loaded
    auto v2 = RunWorkflow(stack, retrain, {&workload.m_Peer});
    EMPY_REQUIRE(v2.has_value());
    EMPY_CHECK_EQ(v2->m_Status, std::string("rejected"));
    EMPY_CHECK_EQ(v2->m_Result["decision"].GetString("action"), std::string("reject"));
    EMPY_CHECK(v2->m_Error.find("improves accuracy") != std::string::npos);
    EMPY_CHECK_EQ(workload.m_Loads.size(), size_t(1));
    deployment = state.FindDeployment("home-control");
    EMPY_CHECK(deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(1));
    std::map<uint32_t, std::string> statuses;
    for (const Agents::ModelRecord &item : deployment->m_Lifecycle.m_History)
    {
        statuses[item.m_Version] = item.m_Status;
    }
    EMPY_CHECK_EQ(statuses[2], std::string("rejected"));
    EMPY_CHECK(!deployment->m_Lifecycle.m_Training);

    // v3: promoted; v1 retired
    auto v3 = RunWorkflow(stack, retrain, {&workload.m_Peer});
    EMPY_REQUIRE(v3.has_value());
    EMPY_REQUIRE_EQ(v3->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(v3->m_Result.GetInt("previous"), int64_t(1));
    deployment = state.FindDeployment("home-control");
    EMPY_CHECK(deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(3));
    for (const Agents::ModelRecord &item : deployment->m_Lifecycle.m_History)
    {
        statuses[item.m_Version] = item.m_Status;
    }
    EMPY_CHECK_EQ(statuses[1], std::string("retired"));
    EMPY_CHECK_EQ(statuses[3], std::string("active"));
    EMPY_CHECK_EQ(workload.m_Loads.size(), size_t(2));

    // v4: the lifecycle agent would promote it, but the Guard re-derives the feature contract
    auto v4 = RunWorkflow(stack, retrain, {&workload.m_Peer});
    EMPY_REQUIRE(v4.has_value());
    EMPY_CHECK_EQ(v4->m_Status, std::string("rejected"));
    EMPY_CHECK(v4->m_Error.find("promotion guard") != std::string::npos);
    EMPY_CHECK_EQ(workload.m_Loads.size(), size_t(2));
    EMPY_CHECK(state.FindDeployment("home-control")->m_Lifecycle.m_Active == std::optional<uint32_t>(3));

    // rollback to the previous (retired) model
    auto rollback =
        RunWorkflow(stack, Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}}), {&workload.m_Peer});
    EMPY_REQUIRE(rollback.has_value());
    EMPY_REQUIRE_EQ(rollback->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(rollback->m_Result.GetInt("version"), int64_t(1));
    EMPY_CHECK_EQ(rollback->m_Result.GetInt("previous"), int64_t(3));
    EMPY_REQUIRE_EQ(workload.m_Loads.size(), size_t(3));
    EMPY_CHECK_EQ(workload.m_Loads[2]["artifact"].GetInt("version"), int64_t(1));
    deployment = state.FindDeployment("home-control");
    EMPY_CHECK(deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(1));
    for (const Agents::ModelRecord &item : deployment->m_Lifecycle.m_History)
    {
        statuses[item.m_Version] = item.m_Status;
    }
    EMPY_CHECK_EQ(statuses[3], std::string("rolledback"));
    EMPY_CHECK_EQ(statuses[1], std::string("active"));
    // rolling back onto the active version is refused by the Guard
    auto refused =
        RunWorkflow(stack, Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}, {"target", 1}}),
                    {&workload.m_Peer});
    EMPY_REQUIRE(refused.has_value());
    EMPY_CHECK_EQ(refused->m_Status, std::string("rejected"));
    EMPY_CHECK(refused->m_Error.find("already active") != std::string::npos);

    // a retrain without a reachable training workload fails cleanly and clears the training flag
    workload.m_Peer.m_Client->Shutdown();
    auto unreachable = RunWorkflow(stack, retrain);
    EMPY_REQUIRE(unreachable.has_value());
    EMPY_CHECK_EQ(unreachable->m_Status, std::string("failed"));
    EMPY_CHECK(unreachable->m_Error.find("training workload") != std::string::npos);
    EMPY_CHECK(!state.FindDeployment("home-control")->m_Lifecycle.m_Training);

    // retire releases every reservation
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
    auto retire = RunWorkflow(stack, Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}}));
    EMPY_REQUIRE(retire.has_value());
    EMPY_CHECK_EQ(retire->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(retire->m_Result["retired"].Size(), size_t(2));
    EMPY_CHECK(state.FindDeployment("home-control") == nullptr);
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(0));
    EMPY_CHECK(stack.m_Substrate->Deployed().empty());
    EMPY_CHECK(stack.m_Harness->Schedules().Entries().empty());
    Topology::Inventory inventory = state.ApplyReservations(stack.m_Plane.Inventory());
    for (const Topology::Node &node : inventory.m_Nodes)
    {
        EMPY_CHECK_EQ(node.m_UsedCpu, 0.0);
    }
    auto unknown = RunWorkflow(stack, Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}}));
    EMPY_REQUIRE(unknown.has_value());
    EMPY_CHECK_EQ(unknown->m_Status, std::string("rejected"));
}

EMPY_TEST(OpsHarness, DriftProposesRetrain)
{
    // event-driven activation: a drift event → lifecycle agent proposal → harness retrain → promotion
    Stack stack;
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    FakeWorkload workload(stack.m_Broker.m_Port, stack.m_Topics, "home-control");
    workload.m_Train = [](uint32_t version) { return Artifact(version, 0.95); };
    Peer sensor(stack.m_Broker.m_Port, "inference");
    // the agent knows the service from the retained state publication
    Pump(stack, {&workload.m_Peer, &sensor}, [] { return false; }, 300);
    Json drift = Json::MakeObject({{"instance", "home-control-inference-ran-1-0"},
                                   {"kind", "accuracy"},
                                   {"value", 0.7},
                                   {"threshold", 0.9},
                                   {"modelVersion", 0},
                                   {"samples", 40}});
    EMPY_REQUIRE(
        !sensor.Get().Publish(stack.m_Topics.Event("home-control", "drift"), Envelope::Make("event", drift)).get());
    const StateStore &state = stack.m_Harness->State();
    auto promoted = [&]
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && deployment->m_Lifecycle.m_Active.has_value();
    };
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &sensor}, promoted, 10000));
    const WorkflowRecord *retrain = nullptr;
    for (const WorkflowRecord *record : state.Workflows())
    {
        if (record->m_Kind == "retrain")
        {
            retrain = record;
        }
    }
    EMPY_REQUIRE(retrain != nullptr);
    EMPY_CHECK_EQ(retrain->m_Trigger, std::string("drift"));
    EMPY_CHECK_EQ(retrain->m_Status, std::string("succeeded"));
    EMPY_CHECK_EQ(retrain->m_Input.GetNumber("observed"), 0.7);
    EMPY_CHECK_EQ(workload.m_Loads.size(), size_t(1));
    EMPY_CHECK(stack.m_Agents->Status().GetInt("proposals") >= 1);
    // the decision is in the agents memory
    bool remembered = false;
    for (const Context::Entry &entry : stack.m_Plane.Scan(Context::MEMORY))
    {
        remembered = remembered ||
                     (entry.m_Data.GetString("agent") == "lifecycle" && entry.m_Data.GetString("kind") == "decision");
    }
    EMPY_CHECK(remembered);
}

EMPY_TEST(OpsHarness, PlacementAuditRelocates)
{
    // the harness audit trigger reaches the placement agent, which proposes a relocation once the current node
    // can no longer host the inference role; the harness re-places, deploys the new instance, retires the old one
    StackOptions options;
    options.m_PlacementAuditSeconds = 1;
    Stack stack(options);
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    EMPY_REQUIRE(NodesOf(*state.FindDeployment("home-control"), "inference") == std::vector<std::string>({"ran-1"}));
    std::string old = state.FindDeployment("home-control")->m_Instances.front().m_Id;
    // no better placement exists yet: audits keep the current nodes
    Pump(stack, {}, [] { return false; }, 1500);
    EMPY_CHECK(NodesOf(*state.FindDeployment("home-control"), "inference") == std::vector<std::string>({"ran-1"}));
    // the resource registry now reports ran-1 with too little capacity
    std::optional<Context::Entry> entry = stack.m_Plane.Get(Context::RESOURCES, "ran-1");
    EMPY_REQUIRE(entry.has_value());
    entry->m_Data.Set("cpu", 0.5);
    entry->m_Updated = 0;
    stack.m_Plane.Put(*entry);
    auto moved = [&]
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && NodesOf(*deployment, "inference") == std::vector<std::string>({"ran-2"});
    };
    EMPY_REQUIRE(Pump(stack, {}, moved, 10000));
    Pump(stack, {}, [&] { return !state.HasRunning("relocate", "home-control"); }, 5000);
    const WorkflowRecord *relocate = nullptr;
    for (const WorkflowRecord *record : state.Workflows())
    {
        if (record->m_Kind == "relocate" && record->m_Status == "succeeded")
        {
            relocate = record;
        }
    }
    EMPY_REQUIRE(relocate != nullptr);
    // the periodic audit or, usually first, the harness's own exhaustion event for the over-committed node
    EMPY_CHECK(relocate->m_Trigger == "audit" || relocate->m_Trigger == "exhaustion");
    EMPY_CHECK(relocate->m_Result.GetBool("relocated"));
    EMPY_REQUIRE_EQ(relocate->m_Result["moves"].Size(), size_t(1));
    const Json &move = relocate->m_Result["moves"].AsArray().front();
    EMPY_CHECK_EQ(move.GetString("role"), std::string("inference"));
    EMPY_CHECK_EQ(move["retired"].AsArray().front().AsString(), old);
    // the old instance and its reservation are gone; the training role stayed where it was
    EMPY_CHECK(NodesOf(*state.FindDeployment("home-control"), "training") == std::vector<std::string>({"cloud-1"}));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    for (const Substrate::DeploymentSpec &spec : stack.m_Substrate->Deployed())
    {
        EMPY_CHECK(spec.m_Instance != old);
    }
    Topology::Inventory inventory = state.ApplyReservations(stack.m_Plane.Inventory());
    EMPY_CHECK_EQ(inventory.Find("ran-1")->m_UsedCpu, 0.0);
    EMPY_CHECK_EQ(inventory.Find("ran-2")->m_UsedCpu, 1.0);
}

EMPY_TEST(OpsHarness, AgentTasksOverMmqp)
{
    Stack stack;
    Peer caller(stack.m_Broker.m_Port, "caller");
    auto call = [&](const std::string &role, const std::string &task, Json body)
    {
        return caller.Get().CallSync(
            stack.m_Topics.Agent(role, task), Envelope::Make("agent-request", std::move(body)),
            [&]
            {
                stack.m_Client->Poll(1);
                caller.m_Client->Poll(1);
            },
            10000);
    };
    auto [reason, reply] =
        call("placement", "place",
             Json::MakeObject({{"descriptor", Descriptor::ToJson(HomeControl())}, {"role", "inference"}}));
    EMPY_REQUIRE(!reason);
    EMPY_REQUIRE(reply.has_value());
    EMPY_CHECK_EQ(reply->m_Kind, std::string("agent-response"));
    Agents::PlacementPlan plan = Agents::PlacementPlan::FromJson(reply->m_Body["result"]);
    EMPY_CHECK(plan.m_Nodes == std::vector<std::string>({"ran-1"}));
    auto [unknownReason, unknown] = call("executive", "explode", Json::MakeObject());
    EMPY_CHECK(static_cast<bool>(unknownReason));
    EMPY_CHECK(unknownReason.m_Text.find("no task") != std::string::npos);
    auto [badReason, bad] = call("compliance", "review", Json::MakeObject({{"descriptor", 5}}));
    EMPY_CHECK_EQ(badReason.m_Code, eMQ::MMQP_REASON_MALFORMED_PACKET);
    auto [emptyReason, empty] = call("executive", "interpret", Json::MakeObject({{"intent", "  "}}));
    EMPY_CHECK_EQ(emptyReason.m_Code, eMQ::MMQP_REASON_MALFORMED_PACKET);
    Json status = stack.m_Agents->Status();
    EMPY_CHECK_EQ(status.GetInt("requests"), int64_t(4));
    EMPY_CHECK_EQ(status.GetInt("failures"), int64_t(3));
    EMPY_CHECK_EQ(status["roles"].Size(), size_t(4));
    EMPY_CHECK(!status["busy"].GetBool("executive"));
}

EMPY_TEST(OpsHarness, StatusEndpoint)
{
    Stack stack;
    StatusServer server("127.0.0.1", 0);
    uint16_t port = server.Start();
    EMPY_REQUIRE(port != 0);
    stack.m_Harness->SetSnapshotSink([&](const Json &snapshot) { server.Update(snapshot); });
    auto record = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(record && record->m_Status == "succeeded");
    auto get = [&](std::string path, std::string method = "GET")
    {
        HttpRequest request;
        request.m_Method = std::move(method);
        request.m_Url = "http://127.0.0.1:" + std::to_string(port) + path;
        request.m_TimeoutMs = 5000;
        return SendHttp(request);
    };
    HttpResponse status = get("/api/mlops/status");
    EMPY_CHECK_EQ(status.m_Status, int32_t(200));
    EMPY_CHECK(status.Header("content-type").find("application/json") != std::string::npos);
    Json snapshot = Json::Parse(status.m_Body);
    EMPY_CHECK_EQ(snapshot.GetString("identity"), std::string("harness"));
    EMPY_REQUIRE_EQ(snapshot["deployments"].Size(), size_t(1));
    EMPY_CHECK_EQ(snapshot["deployments"].AsArray().front().GetString("service"), std::string("home-control"));
    EMPY_CHECK_EQ(snapshot["deployments"].AsArray().front().GetString("status"), std::string("deployed"));
    EMPY_CHECK(snapshot["workflows"].Size() >= 1);
    EMPY_CHECK_EQ(snapshot["schedules"].Size(), size_t(1));
    EMPY_CHECK(snapshot["time"].AsUint() > 0);
    HttpResponse health = get("/api/mlops/healthz?verbose=1");
    EMPY_CHECK_EQ(health.m_Status, int32_t(200));
    Json healthz = Json::Parse(health.m_Body);
    EMPY_CHECK_EQ(healthz.GetString("status"), std::string("ok"));
    EMPY_CHECK(healthz["snapshotAge"].AsUint() < 60000);
    EMPY_CHECK_EQ(get("/api/mlops/other").m_Status, int32_t(404));
    EMPY_CHECK_EQ(get("/api/mlops/status", "POST").m_Status, int32_t(405));
    server.Stop();
    EMPY_CHECK_EQ(server.Port(), uint16_t(0));
}

EMPY_TEST(OpsHarness, ConfigFiles)
{
    HarnessConfig config = LoadHarnessConfig(ConfigFile("Harness.yaml"));
    EMPY_CHECK_EQ(config.m_TopicRoot, std::string("mlops"));
    EMPY_CHECK_EQ(config.m_Identity, std::string("harness"));
    EMPY_CHECK(config.HasRole("harness") && config.HasRole("compliance"));
    EMPY_CHECK(std::filesystem::path(config.m_Ontology).is_absolute());
    EMPY_CHECK(std::filesystem::exists(config.m_Ontology));
    EMPY_CHECK(std::filesystem::exists(config.m_Knowledge));
    EMPY_CHECK(std::filesystem::exists(config.m_Policies));
    EMPY_CHECK(!config.m_HitlDefaultApprove);
    EMPY_CHECK(!config.m_WaitForReady);
    HarnessComponents components = BuildComponents(config);
    EMPY_CHECK(components.m_Ontology->Size() > 0);
    EMPY_CHECK(components.m_Knowledge->Size(Context::RESOURCES) > 0);
    EMPY_CHECK(components.m_Mcp == nullptr);
    EMPY_CHECK_EQ(components.m_Substrate->Name(), std::string("simulated"));
    EMPY_CHECK(components.m_Regions == std::vector<std::string>({"ran", "edge", "cloud"}));
    EMPY_CHECK(!components.m_Policies.m_Ownership.empty());
    EMPY_CHECK(components.Environment().m_Comas == components.m_Comas.get());
    HarnessOptions harness = MakeHarnessOptions(config, components);
    EMPY_CHECK_EQ(harness.m_AgentTimeoutMs, config.m_AgentTimeoutMs);
    EMPY_CHECK(harness.m_Validation.m_Regions == components.m_Regions);
    AgentHostOptions agents = MakeAgentHostOptions(config, components);
    EMPY_CHECK(agents.m_Roles == std::vector<std::string>({"executive", "placement", "lifecycle", "compliance"}));
    EMPY_CHECK(agents.m_HarnessTimeoutMs > config.m_AgentTimeoutMs);

    // event-driven triggers and the periodic placement audit are on in the shipped configuration
    EMPY_CHECK_EQ(config.m_PlacementAuditSeconds, uint32_t(300));
    EMPY_CHECK_EQ(harness.m_PlacementAuditSeconds, uint32_t(300));
    EMPY_CHECK_EQ(harness.m_LivenessMs, uint32_t(15000));
    EMPY_CHECK(!harness.m_SecureWorkloads);
    EMPY_CHECK_EQ(agents.m_ViolationLimit, uint32_t(3));
    EMPY_CHECK_EQ(agents.m_ViolationWindowSeconds, uint32_t(300));
    EMPY_CHECK(agents.m_OntologyState.empty());

    HarnessConfig local = LoadHarnessConfig(ConfigFile("HarnessLocal.yaml"));
    EMPY_CHECK(local.m_WaitForReady);
    EMPY_CHECK_EQ(local.m_Substrate["type"].As<std::string>(), std::string("process"));
    EMPY_CHECK_EQ(local.m_PlacementAuditSeconds, uint32_t(300));
    EMPY_CHECK(!local.m_SecureWorkloads);

    // minimal configuration (as written by the MCP subprocess test) and rejected ones
    TempDir temp;
    HarnessConfig minimal =
        ParseHarnessConfig("identity: test\nsubstrate: {type: simulated}\n", (temp.m_Path / "Harness.yaml").string());
    EMPY_CHECK_EQ(minimal.m_Identity, std::string("test"));
    EMPY_CHECK(minimal.m_Ontology.empty());
    EMPY_CHECK_EQ(BuildComponents(minimal).m_Ontology->Size(), size_t(0));
    HarnessConfig relative =
        ParseHarnessConfig("knowledge: sub/K.yaml\nstate_directory: st\n", (temp.m_Path / "Harness.yaml").string());
    EMPY_CHECK(std::filesystem::path(relative.m_Knowledge) ==
               std::filesystem::absolute(temp.m_Path / "sub" / "K.yaml").lexically_normal());
    EMPY_CHECK_THROWS(ParseHarnessConfig("unknown: 1\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("roles: [harness, wizard]\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("hitl: {default: maybe}\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("timeouts: {agent_ms: 0}\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("status: {port: 70000}\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("topic_root: \"a/+\"\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("broker: {hots: x}\n", ""));
    EMPY_CHECK_THROWS(LoadHarnessConfig(temp.m_Path / "missing.yaml"));

    // liveness, compliance escalation, workload session security and the evolved ontology
    HarnessConfig tuned = ParseHarnessConfig("timeouts: {liveness_ms: 0}\ncompliance: {violation_limit: 0, "
                                             "violation_window_seconds: 60}\nbroker: {secure_workloads: true}\n"
                                             "ontology_state: evolved.yaml\n",
                                             (temp.m_Path / "Harness.yaml").string());
    EMPY_CHECK_EQ(tuned.m_LivenessMs, uint32_t(0));
    EMPY_CHECK_EQ(tuned.m_ViolationLimit, uint32_t(0));
    EMPY_CHECK_EQ(tuned.m_ViolationWindowSeconds, uint32_t(60));
    EMPY_CHECK(tuned.m_SecureWorkloads);
    EMPY_CHECK(std::filesystem::path(tuned.m_OntologyState) ==
               std::filesystem::absolute(temp.m_Path / "evolved.yaml").lexically_normal());
    EMPY_CHECK_THROWS(ParseHarnessConfig("compliance: {limit: 1}\n", ""));
    EMPY_CHECK_THROWS(ParseHarnessConfig("compliance: {violation_window_seconds: 0}\n", ""));
    // a process substrate handing the workloads a TLS client configuration makes their sessions secure
    WriteFileAtomic(temp.m_Path / "ClientTls.yaml", "tls: true\nserver_name: localhost\n");
    HarnessConfig secure = ParseHarnessConfig("substrate: {type: process, client_config: ClientTls.yaml}\n",
                                              (temp.m_Path / "Harness.yaml").string());
    EMPY_CHECK(secure.m_SecureWorkloads);
    EMPY_CHECK(!ParseHarnessConfig("substrate: {type: process, client_config: ClientTls.yaml}\nbroker: "
                                   "{secure_workloads: false}\n",
                                   (temp.m_Path / "Harness.yaml").string())
                    .m_SecureWorkloads);
    // the evolved ontology of an earlier run is loaded in preference to the seed
    Semantics::Ontology evolved = Semantics::Ontology::Load(ConfigFile("Ontology.yaml"));
    Semantics::Concept greenhouse;
    greenhouse.m_Id = "greenhouse";
    greenhouse.m_Kind = "domain";
    greenhouse.m_Synonyms = {"glasshouse"};
    EMPY_REQUIRE(evolved.Extend(greenhouse));
    evolved.Save(temp.m_Path / "evolved.yaml");
    HarnessConfig resumed = ParseHarnessConfig("ontology: '" + ConfigFile("Ontology.yaml").generic_string() +
                                                   "'\nontology_state: evolved.yaml\nsubstrate: {type: simulated}\n",
                                               (temp.m_Path / "Harness.yaml").string());
    EMPY_CHECK(BuildComponents(resumed).m_Ontology->Find("greenhouse") != nullptr);
    EMPY_CHECK_EQ(MakeAgentHostOptions(resumed, BuildComponents(resumed)).m_OntologyState.string(),
                  resumed.m_OntologyState);
    std::filesystem::remove(temp.m_Path / "evolved.yaml");
    EMPY_CHECK(BuildComponents(resumed).m_Ontology->Find("greenhouse") == nullptr); // first run: the seed
}

EMPY_TEST(OpsHarness, PersistentHarnessResumes)
{
    TempDir state;
    StackOptions options;
    options.m_StateDirectory = state.m_Path / "state";
    {
        Stack stack(options);
        auto record = RunWorkflow(stack, DeployDescriptor(HomeControl()));
        EMPY_REQUIRE(record && record->m_Status == "succeeded");
    }
    // a restarted harness resumes its deployments, reservations, records and temporal triggers
    HarnessOptions harness;
    harness.m_StateDirectory = options.m_StateDirectory;
    harness.m_Validation.m_Regions = {"ran", "edge", "cloud"};
    // a workflow cut short by the stop (enact or retire-previous) left reservations no deployment holds
    {
        StateStore stopped(StateOptions{options.m_StateDirectory, 256});
        Descriptor::Resources resources;
        resources.m_Cpu = 2;
        resources.m_Memory = 4ull << 30;
        stopped.Reserve("home-control-inference-ran-1-7", "ran-1", resources);
        EMPY_CHECK_EQ(stopped.Reservations().Size(), size_t(3));
    }
    ExecutionHarness resumed(harness, std::make_shared<Substrate::SimulatedSubstrate>(), nullptr);
    const Deployment *deployment = resumed.State().FindDeployment("home-control");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK_EQ(deployment->m_Instances.size(), size_t(2));
    // no workflow survives a restart: the orphaned reservation is released, the deployment's ones stay
    const Json reservations = resumed.State().Reservations();
    EMPY_CHECK_EQ(reservations.Size(), size_t(2));
    for (const Json &reservation : reservations.AsArray())
    {
        EMPY_CHECK(reservation.GetString("instance") != "home-control-inference-ran-1-7");
    }
    EMPY_CHECK_EQ(resumed.State().ApplyReservations(SmallInventory()).Find("ran-1")->m_UsedCpu, 1.0);
    auto entries = resumed.Schedules().Entries();
    EMPY_REQUIRE_EQ(entries.size(), size_t(1));
    EMPY_CHECK_EQ(entries[0].m_Expression, std::string("0 0 * * *"));
    EMPY_REQUIRE_EQ(resumed.State().Workflows().size(), size_t(1));
    EMPY_CHECK_EQ(resumed.State().Workflows().front()->m_Status, std::string("succeeded"));
    EMPY_CHECK_THROWS(resumed.Submit(Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}}), "test"));
}

EMPY_TEST(OpsHarness, ClaimedAccuracyNeverSetsTheBaseline)
{
    // a requester's 'observed' claim is provenance only; the harness judges candidates against the accuracy it
    // observed itself, and the model replaced because it degraded keeps that accuracy (no rollback onto it)
    Stack stack;
    Descriptor::RichMLService service = HomeControl();
    service.m_Lifecycle.m_CooldownSeconds = 0;
    auto deployed = RunWorkflow(stack, DeployDescriptor(service));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    FakeWorkload workload(stack.m_Broker.m_Port, stack.m_Topics, "home-control");
    std::map<uint32_t, Workload::ModelArtifact> plan = {
        {1, Artifact(1, 0.95)}, {2, Artifact(2, 0.96)}, {3, Artifact(3, 0.93)}};
    workload.m_Train = [&](uint32_t version) { return plan.at(version); };
    const StateStore &state = stack.m_Harness->State();
    auto v1 =
        RunWorkflow(stack, Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}}), {&workload.m_Peer});
    EMPY_REQUIRE(v1 && v1->m_Status == "succeeded");

    // claimed 0: 0.96 does not improve on the recorded 0.95 by minImprovement 0.02
    auto claimed =
        RunWorkflow(stack, Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}, {"observed", 0}}),
                    {&workload.m_Peer});
    EMPY_REQUIRE(claimed.has_value());
    EMPY_CHECK_EQ(claimed->m_Status, std::string("rejected"));
    const StepRecord *evaluate = Step(*claimed, "evaluate");
    EMPY_REQUIRE(evaluate != nullptr);
    EMPY_CHECK_EQ(evaluate->m_Detail.GetNumber("claimed", -1), 0.0);
    EMPY_CHECK(!evaluate->m_Detail.Has("observed"));
    EMPY_CHECK(state.FindDeployment("home-control")->m_Lifecycle.m_Active == std::optional<uint32_t>(1));
    // out-of-range claims are refused by the request guard
    auto absurd =
        RunWorkflow(stack, Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}, {"observed", -3}}));
    EMPY_REQUIRE(absurd.has_value());
    EMPY_CHECK_EQ(absurd->m_Status, std::string("rejected"));
    EMPY_CHECK(absurd->m_Error.find("[0, 1]") != std::string::npos);

    // the active v1 degrades on the fabric (metrics event of its inference instance): the lifecycle agent
    // proposes a retrain and the harness judges v3 against the observed 0.5, not the recorded 0.95
    Peer sensor(stack.m_Broker.m_Port, "inference");
    Pump(stack, {&workload.m_Peer, &sensor}, [] { return false; }, 300);
    Json metrics = Json::MakeObject({{"instance", InstanceOf(*state.FindDeployment("home-control"), "inference")},
                                     {"accuracy", Json::MakeObject({{"value", 0.5}})},
                                     {"modelVersion", 1},
                                     {"samples", 64}});
    EMPY_REQUIRE(
        !sensor.Get().Publish(stack.m_Topics.Event("home-control", "metrics"), Envelope::Make("event", metrics)).get());
    auto promoted = [&]
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && deployment->m_Lifecycle.m_Active == std::optional<uint32_t>(3) &&
               !state.HasRunning("retrain", "home-control");
    };
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &sensor}, promoted, 10000));
    const WorkflowRecord *retrain = Newest(state, "retrain");
    EMPY_REQUIRE(retrain != nullptr);
    EMPY_CHECK_EQ(retrain->m_Trigger, std::string("metrics"));
    EMPY_CHECK_EQ(retrain->m_Status, std::string("succeeded"));
    const StepRecord *judged = Step(*retrain, "evaluate");
    EMPY_REQUIRE(judged != nullptr);
    EMPY_CHECK_EQ(judged->m_Detail.GetNumber("observed"), 0.5);
    EMPY_CHECK_EQ(judged->m_Detail.GetNumber("recorded"), 0.95);
    // v1 keeps the degraded accuracy as its last known one: neither the lifecycle agent nor a request returns to it
    const Deployment *deployment = state.FindDeployment("home-control");
    const Agents::ModelRecord *v1Record = nullptr;
    for (const Agents::ModelRecord &record : deployment->m_Lifecycle.m_History)
    {
        v1Record = record.m_Version == 1 ? &record : v1Record;
    }
    EMPY_REQUIRE(v1Record != nullptr);
    EMPY_CHECK_EQ(v1Record->m_Status, std::string("retired"));
    EMPY_CHECK_EQ(v1Record->m_Accuracy, 0.5);
    auto rollback =
        RunWorkflow(stack, Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}}), {&workload.m_Peer});
    EMPY_REQUIRE(rollback.has_value());
    EMPY_CHECK_EQ(rollback->m_Status, std::string("rejected"));
    EMPY_CHECK(rollback->m_Error.find("below the threshold") != std::string::npos);
    // a rejected candidate is no rollback target either
    auto rejected =
        RunWorkflow(stack, Json::MakeObject({{"kind", "rollback"}, {"service", "home-control"}, {"target", 2}}),
                    {&workload.m_Peer});
    EMPY_REQUIRE(rejected.has_value());
    EMPY_CHECK_EQ(rejected->m_Status, std::string("rejected"));
    EMPY_CHECK(rejected->m_Error.find("promoted before") != std::string::npos);
    EMPY_CHECK(state.FindDeployment("home-control")->m_Lifecycle.m_Active == std::optional<uint32_t>(3));
}

EMPY_TEST(OpsHarness, LostInstanceRedeployed)
{
    // liveness: an inference instance falls silent → the harness raises Event(service, "failure") → the placement
    // agent proposes a relocation that replaces the lost instance (even on the same node) → the old one is retired
    StackOptions options;
    options.m_LivenessMs = 600;
    Stack stack(options);
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    const std::string lost = InstanceOf(*state.FindDeployment("home-control"), "inference");
    Peer workload(stack.m_Broker.m_Port, "workload");
    std::vector<Envelope> failures;
    ListenEnvelopes(workload, stack.m_Topics.EventFilter("failure"), failures);
    Beat(workload, stack.m_Topics, "home-control", lost, "inference");
    auto replaced = [&]
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && !InstanceOf(*deployment, "inference").empty() &&
               InstanceOf(*deployment, "inference") != lost && !state.HasRunning("relocate", "home-control");
    };
    EMPY_REQUIRE(Pump(stack, {&workload}, replaced, 10000));
    const WorkflowRecord *relocate = Newest(state, "relocate");
    EMPY_REQUIRE(relocate != nullptr);
    EMPY_CHECK_EQ(relocate->m_Trigger, std::string("failure"));
    EMPY_CHECK_EQ(relocate->m_Status, std::string("succeeded"));
    EMPY_REQUIRE_EQ(relocate->m_Input["replace"].Size(), size_t(1));
    EMPY_CHECK_EQ(relocate->m_Input["replace"][size_t(0)].AsString(), lost);
    EMPY_REQUIRE_EQ(relocate->m_Result["moves"].Size(), size_t(1));
    EMPY_CHECK_EQ(relocate->m_Result["moves"][size_t(0)]["retired"][size_t(0)].AsString(), lost);
    EMPY_REQUIRE(!failures.empty());
    EMPY_CHECK_EQ(failures.front().m_Body.GetString("instance"), lost);
    EMPY_CHECK_EQ(failures.front().m_Body.GetString("role"), std::string("inference"));
    EMPY_CHECK(failures.front().m_Body.GetString("reason").find("no heartbeat") != std::string::npos);
    for (const Substrate::DeploymentSpec &spec : stack.m_Substrate->Deployed())
    {
        EMPY_CHECK(spec.m_Instance != lost);
    }
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
    EMPY_CHECK(Remembered(stack.m_Plane, "placement", "failure",
                          [&](const Json &data) { return data.GetString("instance") == lost; }));
    bool observed = false;
    for (const Context::Entry &entry : stack.m_Plane.Scan(Context::OBSERVATIONS))
    {
        observed =
            observed || (entry.m_Data.GetString("source") == "harness" && entry.m_Data.GetString("kind") == "failure");
    }
    EMPY_CHECK(observed);
}

EMPY_TEST(OpsHarness, SubstrateExitRaisesFailure)
{
    // a workload the substrate reports exited (crashed EmpyWorkload, deleted pod) is redeployed
    auto substrate = std::make_shared<CrashingSubstrate>();
    StackOptions options;
    options.m_Substrate = substrate;
    Stack stack(options);
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    const std::string crashed = InstanceOf(*state.FindDeployment("home-control"), "training");
    Peer observer(stack.m_Broker.m_Port, "observer");
    std::vector<Envelope> failures;
    ListenEnvelopes(observer, stack.m_Topics.EventFilter("failure"), failures);
    Pump(stack, {&observer}, [] { return false; }, 300);
    substrate->Crash(crashed);
    auto replaced = [&]
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && !InstanceOf(*deployment, "training").empty() &&
               InstanceOf(*deployment, "training") != crashed && !state.HasRunning("relocate", "home-control");
    };
    EMPY_REQUIRE(Pump(stack, {&observer}, replaced, 10000));
    const WorkflowRecord *relocate = Newest(state, "relocate");
    EMPY_REQUIRE(relocate != nullptr);
    EMPY_CHECK_EQ(relocate->m_Trigger, std::string("failure"));
    EMPY_CHECK_EQ(relocate->m_Input.GetString("role"), std::string("training"));
    EMPY_CHECK_EQ(relocate->m_Status, std::string("succeeded"));
    EMPY_REQUIRE(!failures.empty());
    EMPY_CHECK_EQ(failures.front().m_Body.GetString("instance"), crashed);
    EMPY_CHECK(failures.front().m_Body.GetString("reason").find("exited") != std::string::npos);
    // the inference instance was not touched
    EMPY_CHECK(NodesOf(*state.FindDeployment("home-control"), "inference") == std::vector<std::string>({"ran-1"}));
}

EMPY_TEST(OpsHarness, ExhaustionAndAcceleratorEventsRelocate)
{
    // without any periodic audit, the harness's own capacity checks activate the placement agent
    Stack stack;
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    EMPY_REQUIRE(NodesOf(*state.FindDeployment("home-control"), "inference") == std::vector<std::string>({"ran-1"}));
    EMPY_REQUIRE(NodesOf(*state.FindDeployment("home-control"), "training") == std::vector<std::string>({"cloud-1"}));
    Peer observer(stack.m_Broker.m_Port, "observer");
    std::vector<Envelope> exhaustion, accelerator;
    ListenEnvelopes(observer, stack.m_Topics.EventFilter("exhaustion"), exhaustion);
    ListenEnvelopes(observer, stack.m_Topics.EventFilter("accelerator"), accelerator);
    Pump(stack, {&observer}, [] { return false; }, 300);
    // the resource registry shrinks ran-1 below the reserved inference cpu
    std::optional<Context::Entry> entry = stack.m_Plane.Get(Context::RESOURCES, "ran-1");
    EMPY_REQUIRE(entry.has_value());
    entry->m_Data.Set("cpu", 0.5);
    entry->m_Updated = 0;
    stack.m_Plane.Put(*entry);
    auto moved = [&](std::string_view role, std::string_view from)
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        std::vector<std::string> nodes = deployment ? NodesOf(*deployment, role) : std::vector<std::string>();
        return nodes.size() == 1 && nodes.front() != from && !state.HasRunning("relocate", "home-control");
    };
    EMPY_REQUIRE(Pump(stack, {&observer}, [&] { return moved("inference", "ran-1"); }, 10000));
    const WorkflowRecord *relocate = Newest(state, "relocate");
    EMPY_REQUIRE(relocate != nullptr);
    EMPY_CHECK_EQ(relocate->m_Trigger, std::string("exhaustion"));
    EMPY_CHECK_EQ(relocate->m_Status, std::string("succeeded"));
    EMPY_REQUIRE(!exhaustion.empty());
    EMPY_CHECK_EQ(exhaustion.front().m_Body.GetString("node"), std::string("ran-1"));
    // cloud-1 loses its GPUs: the training role moves to another accelerator node
    entry = stack.m_Plane.Get(Context::RESOURCES, "cloud-1");
    EMPY_REQUIRE(entry.has_value());
    entry->m_Data.Set("accelerators", Json::MakeObject({{"gpu", 0}}));
    entry->m_Updated = 0;
    stack.m_Plane.Put(*entry);
    EMPY_REQUIRE(Pump(stack, {&observer}, [&] { return moved("training", "cloud-1"); }, 10000));
    relocate = Newest(state, "relocate");
    EMPY_REQUIRE(relocate != nullptr);
    EMPY_CHECK_EQ(relocate->m_Trigger, std::string("accelerator"));
    EMPY_CHECK_EQ(relocate->m_Input.GetString("role"), std::string("training"));
    EMPY_REQUIRE(!accelerator.empty());
    EMPY_CHECK_EQ(accelerator.front().m_Body.GetString("accelerator"), std::string("gpu"));
    EMPY_CHECK_EQ(accelerator.front().m_Body.GetInt("available"), int64_t(0));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
}

EMPY_TEST(OpsHarness, ViolationsEscalateRetirement)
{
    // repeated policy violations at the execution boundary → the compliance agent proposes a retirement that the
    // harness puts to a human (HITL) first; a rate limit throttling the service is never escalated
    Stack stack;
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    Peer operatorPeer(stack.m_Broker.m_Port, "operator");
    std::vector<Json> questions;
    EMPY_REQUIRE(!operatorPeer.Get().Serve(
        stack.m_Topics.Hitl(),
        [&](const Fabric::Inbound &request, Bus::Responder respond)
        {
            questions.push_back(request.m_Envelope.m_Body);
            respond(Envelope::Make("hitl-decision", Json::MakeObject({{"decision", "approve"}, {"operator", "bob"}})),
                    {});
        }));
    Peer workload(stack.m_Broker.m_Port, "workload");
    Pump(stack, {&operatorPeer, &workload}, [] { return false; }, 300);
    auto violate = [&](const char *rule, const char *reason)
    {
        Json body = Json::MakeObject({{"service", "home-control"},
                                      {"instance", InstanceOf(*state.FindDeployment("home-control"), "inference")},
                                      {"topic", "home/set/temp"},
                                      {"rule", rule},
                                      {"reason", reason}});
        EMPY_REQUIRE(!workload.Get()
                          .Publish(stack.m_Topics.Event("home-control", "violation"), Envelope::Make("event", body))
                          .get());
    };
    for (int32_t i = 0; i < 4; ++i)
    {
        violate("rate", "rate limit of 6 actions per minute exceeded on 'home/set/temp'");
    }
    Pump(stack, {&operatorPeer, &workload}, [] { return false; }, 800);
    EMPY_CHECK(Newest(state, "retire") == nullptr);
    EMPY_CHECK(questions.empty());
    EMPY_CHECK(Remembered(stack.m_Plane, "compliance", "violation",
                          [](const Json &data) { return data.GetString("rule") == "rate"; }));
    for (int32_t i = 0; i < 3; ++i)
    {
        violate("domain", "'home/set/door' is not a declared actionable");
    }
    auto retired = [&]
    { return state.FindDeployment("home-control") == nullptr && !state.HasRunning("retire", "home-control"); };
    EMPY_REQUIRE(Pump(stack, {&operatorPeer, &workload}, retired, 10000));
    const WorkflowRecord *retire = Newest(state, "retire");
    EMPY_REQUIRE(retire != nullptr);
    EMPY_CHECK_EQ(retire->m_Trigger, std::string("violation"));
    EMPY_CHECK_EQ(retire->m_Status, std::string("succeeded"));
    EMPY_CHECK(retire->m_Input.GetBool("escalate"));
    const StepRecord *hitl = Step(*retire, "hitl");
    EMPY_REQUIRE(hitl != nullptr);
    EMPY_CHECK_EQ(hitl->m_Detail.GetString("operator"), std::string("bob"));
    EMPY_REQUIRE_EQ(questions.size(), size_t(1));
    EMPY_CHECK_EQ(questions[0].GetString("service"), std::string("home-control"));
    EMPY_CHECK(questions[0].GetString("question").find("policy violations") != std::string::npos);
    EMPY_CHECK(stack.m_Substrate->Deployed().empty());
}

EMPY_TEST(OpsHarness, EscalatedRetirementRejectedByDefault)
{
    // an escalated retirement nobody answers keeps the service (HITL default: reject)
    StackOptions options;
    options.m_HitlTimeoutMs = 300;
    Stack stack(options);
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    auto retire = RunWorkflow(
        stack,
        Json::MakeObject({{"kind", "retire"}, {"service", "home-control"}, {"escalate", true}, {"reason", "test"}}));
    EMPY_REQUIRE(retire.has_value());
    EMPY_CHECK_EQ(retire->m_Status, std::string("rejected"));
    EMPY_CHECK(retire->m_Error.find("by default") != std::string::npos);
    EMPY_CHECK(stack.m_Harness->State().FindDeployment("home-control") != nullptr);
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
}

EMPY_TEST(OpsHarness, LatencyDriftReassessesPlacement)
{
    // latency drift is a placement concern: no retrain, a placement reassessment of the inference role instead
    Stack stack;
    auto deployed = RunWorkflow(stack, DeployDescriptor(HomeControl()));
    EMPY_REQUIRE(deployed && deployed->m_Status == "succeeded");
    const StateStore &state = stack.m_Harness->State();
    Peer sensor(stack.m_Broker.m_Port, "inference");
    Pump(stack, {&sensor}, [] { return false; }, 300);
    Json drift = Json::MakeObject({{"instance", InstanceOf(*state.FindDeployment("home-control"), "inference")},
                                   {"kind", "latency"},
                                   {"value", 12.5},
                                   {"threshold", 5},
                                   {"modelVersion", 0},
                                   {"samples", 64}});
    EMPY_REQUIRE(
        !sensor.Get().Publish(stack.m_Topics.Event("home-control", "drift"), Envelope::Make("event", drift)).get());
    auto reassessed = [&]
    {
        return Remembered(stack.m_Plane, "placement", "reassessment", [](const Json &data)
                          { return data.GetString("trigger") == "latency" && data.GetString("role") == "inference"; });
    };
    EMPY_REQUIRE(Pump(stack, {&sensor}, reassessed, 10000));
    Pump(stack, {&sensor}, [] { return false; }, 500);
    EMPY_CHECK(Newest(state, "retrain") == nullptr);
    EMPY_CHECK(Remembered(stack.m_Plane, "lifecycle", "decision",
                          [](const Json &data) { return data.GetString("action") == "none"; }));
}

EMPY_TEST(OpsHarness, RedeployWaitsForReadiness)
{
    // a redeploy retires the replaced instances only once the new ones serve the active model (inference) and
    // hold the samples the replaced trainer had (training); new instances that never report are retired instead
    StackOptions options;
    options.m_WaitForReady = true;
    options.m_ReadyTimeoutMs = 3000;
    options.m_LivenessMs = 0;
    Stack stack(options);
    const StateStore &state = stack.m_Harness->State();
    FakeWorkload workload(stack.m_Broker.m_Port, stack.m_Topics, "home-control");
    workload.m_Train = [](uint32_t version) { return Artifact(version, 0.95); };
    Peer beats(stack.m_Broker.m_Port, "workloads");
    auto submit = [&](Json request)
    {
        auto done = std::make_shared<std::optional<WorkflowRecord>>();
        stack.m_Harness->Submit(std::move(request), "test", [done](const WorkflowRecord &record) { *done = record; });
        return done;
    };
    auto version = [&](const std::string &wanted)
    {
        const Deployment *deployment = state.FindDeployment("home-control");
        return deployment && deployment->m_Version == wanted;
    };

    // first deployment: ready once both instances report
    Descriptor::RichMLService service = HomeControl();
    auto first = submit(DeployDescriptor(service));
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return version("1.0.0"); }));
    const std::string inference1 = InstanceOf(*state.FindDeployment("home-control"), "inference");
    const std::string training1 = InstanceOf(*state.FindDeployment("home-control"), "training");
    Beat(beats, stack.m_Topics, "home-control", inference1, "inference");
    Beat(beats, stack.m_Topics, "home-control", training1, "training", 0, 50);
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return first->has_value(); }));
    EMPY_REQUIRE_EQ((*first)->m_Status, std::string("succeeded"));
    auto v1 = RunWorkflow(stack, Json::MakeObject({{"kind", "retrain"}, {"service", "home-control"}}),
                          {&workload.m_Peer, &beats});
    EMPY_REQUIRE(v1 && v1->m_Status == "succeeded");
    Beat(beats, stack.m_Topics, "home-control", inference1, "inference", 1);
    Beat(beats, stack.m_Topics, "home-control", training1, "training", 0, 50);

    // redeploy 1.1.0: the old instances keep running while the new ones come up
    service.m_Metadata.m_Version = "1.1.0";
    auto redeploy = submit(DeployDescriptor(service));
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return version("1.1.0"); }));
    const std::string inference2 = InstanceOf(*state.FindDeployment("home-control"), "inference");
    const std::string training2 = InstanceOf(*state.FindDeployment("home-control"), "training");
    EMPY_CHECK(inference2 != inference1 && training2 != training1);
    Pump(stack, {&workload.m_Peer, &beats}, [] { return false; }, 300);
    EMPY_CHECK(!redeploy->has_value());
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(4));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(4));
    // the new inference instance reports without a model: the harness loads the active v1 into it at once
    size_t loads = workload.m_Loads.size();
    Beat(beats, stack.m_Topics, "home-control", inference2, "inference");
    Beat(beats, stack.m_Topics, "home-control", training2, "training", 0, 10);
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return workload.m_Loads.size() > loads; }));
    EMPY_CHECK_EQ(workload.m_Loads.back()["artifact"].GetInt("version"), int64_t(1));
    Beat(beats, stack.m_Topics, "home-control", inference2, "inference", 1);
    Pump(stack, {&workload.m_Peer, &beats}, [] { return false; }, 300);
    EMPY_CHECK(!redeploy->has_value()); // the new trainer holds 10 of the 32 samples it needs
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(4));
    Beat(beats, stack.m_Topics, "home-control", training2, "training", 0, 40);
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return redeploy->has_value(); }));
    EMPY_REQUIRE_EQ((*redeploy)->m_Status, std::string("succeeded"));
    EMPY_CHECK((*redeploy)->m_Result["replaced"] == Json::MakeArray({Json(inference1), Json(training1)}));
    const StepRecord *ready = Step(**redeploy, "ready");
    EMPY_REQUIRE(ready != nullptr);
    EMPY_CHECK_EQ(ready->m_Status, std::string("succeeded"));
    EMPY_CHECK(Step(**redeploy, "retire-previous") != nullptr);
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
    EMPY_CHECK(state.FindDeployment("home-control")->m_Lifecycle.m_Active == std::optional<uint32_t>(1));

    // redeploy 1.2.0 whose instances never report: they are retired, 1.1.0 keeps running
    service.m_Metadata.m_Version = "1.2.0";
    auto late = submit(DeployDescriptor(service));
    EMPY_REQUIRE(Pump(stack, {&workload.m_Peer, &beats}, [&] { return late->has_value(); }, 10000));
    EMPY_CHECK_EQ((*late)->m_Status, std::string("failed"));
    EMPY_CHECK((*late)->m_Error.find("1.1.0 keeps running") != std::string::npos);
    EMPY_CHECK((*late)->m_Result["missing"].Size() == size_t(2));
    EMPY_REQUIRE(version("1.1.0"));
    EMPY_CHECK_EQ(InstanceOf(*state.FindDeployment("home-control"), "inference"), inference2);
    EMPY_CHECK_EQ(stack.m_Substrate->Deployed().size(), size_t(2));
    EMPY_CHECK_EQ(state.Reservations().Size(), size_t(2));
}

EMPY_TEST(OpsHarness, NodeBrokerAndSecurityRequirements)
{
    // an instance joins the broker of its node (session locality) and carries the declared identity; a
    // descriptor declaring encryption is refused while the workload sessions are not TLS/QUIC
    StackOptions options;
    options.m_Knowledge = [](Context::KnowledgePlane &plane)
    {
        std::optional<Context::Entry> entry = plane.Get(Context::RESOURCES, "ran-1");
        EMPY_REQUIRE(entry.has_value());
        entry->m_Data.Set("broker", "10.1.2.3:5611");
        entry->m_Updated = 0;
        plane.Put(*entry);
    };
    Stack stack(options);
    Descriptor::RichMLService service = HomeControl();
    service.m_Constraints.m_Security.m_Identity = "home-control-svc";
    auto deployed = RunWorkflow(stack, DeployDescriptor(service));
    EMPY_REQUIRE(deployed.has_value());
    EMPY_REQUIRE_EQ(deployed->m_Status, std::string("succeeded"));
    std::vector<Substrate::DeploymentSpec> specs = stack.m_Substrate->Deployed();
    EMPY_REQUIRE_EQ(specs.size(), size_t(2));
    for (const Substrate::DeploymentSpec &spec : specs)
    {
        if (spec.m_Node == "ran-1")
        {
            EMPY_CHECK_EQ(spec.m_BrokerHost, std::string("10.1.2.3"));
            EMPY_CHECK_EQ(spec.m_BrokerPort, uint16_t(5611));
        }
        else
        {
            EMPY_CHECK_EQ(spec.m_BrokerHost, std::string("127.0.0.1"));
            EMPY_CHECK_EQ(spec.m_BrokerPort, stack.m_Broker.m_Port);
        }
        EMPY_CHECK(spec.m_ExtraArgs == std::vector<std::string>({"--identity", "home-control-svc"}));
        std::vector<std::string> args = Substrate::WorkloadArgs(spec, "d.yaml");
        EMPY_CHECK(std::find(args.begin(), args.end(), "--identity") != args.end());
    }
    EMPY_CHECK(NodesOf(*stack.m_Harness->State().FindDeployment("home-control"), "inference") ==
               std::vector<std::string>({"ran-1"}));

    Descriptor::RichMLService secret = EnergySaver();
    secret.m_Constraints.m_Security.m_Encryption = true;
    auto refused = RunWorkflow(stack, DeployDescriptor(secret));
    EMPY_REQUIRE(refused.has_value());
    EMPY_CHECK_EQ(refused->m_Status, std::string("rejected"));
    EMPY_CHECK(refused->m_Error.find("security.encryption") != std::string::npos);
    StackOptions secure;
    secure.m_SecureWorkloads = true;
    Stack tls(secure);
    auto accepted = RunWorkflow(tls, DeployDescriptor(secret));
    EMPY_REQUIRE(accepted.has_value());
    EMPY_CHECK_EQ(accepted->m_Status, std::string("succeeded"));
}

EMPY_TEST(OpsHarness, ComplianceAmendmentIsGuarded)
{
    // the harness never trusts the compliance agent: an amendment may only restrict the reviewed candidate, and
    // an escalated amendment shows the operator exactly what is enacted
    StackOptions options;
    options.m_Roles = {"executive", "placement", "lifecycle"};
    Stack stack(options);
    Peer reviewer(stack.m_Broker.m_Port, "compliance");
    Agents::ComplianceReport verdict;
    EMPY_REQUIRE(!reviewer.Get().Serve(
        stack.m_Topics.Agent("compliance", "review"), [&](const Fabric::Inbound &, Bus::Responder respond)
        { respond(Envelope::Make("agent-response", Json::MakeObject({{"result", verdict.ToJson()}})), {}); }));
    std::vector<Json> questions;
    EMPY_REQUIRE(!reviewer.Get().Serve(
        stack.m_Topics.Hitl(),
        [&](const Fabric::Inbound &request, Bus::Responder respond)
        {
            questions.push_back(request.m_Envelope.m_Body);
            respond(Envelope::Make("hitl-decision", Json::MakeObject({{"decision", "approve"}, {"operator", "eve"}})),
                    {});
        }));
    const StateStore &state = stack.m_Harness->State();
    Descriptor::RichMLService service = HomeControl();

    // a renaming amendment (it would replace and downgrade another deployment) is refused
    verdict.m_Verdict = Agents::Verdict::Amended;
    Descriptor::RichMLService renamed = service;
    renamed.m_Metadata.m_Name = "energy-saver";
    renamed.m_Metadata.m_Version = "0.0.1";
    verdict.m_Amended = renamed;
    auto record = RunWorkflow(stack, DeployDescriptor(service), {&reviewer});
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
    EMPY_CHECK(record->m_Error.find("renames") != std::string::npos);
    EMPY_CHECK(state.Deployments().empty());
    // a widening amendment is refused
    Descriptor::RichMLService widened = service;
    widened.m_Actionables[0].m_Max = 40;
    verdict.m_Amended = widened;
    record = RunWorkflow(stack, DeployDescriptor(service), {&reviewer});
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
    EMPY_CHECK(record->m_Error.find("widens the declared maximum") != std::string::npos);
    // escalated: the operator sees the amended descriptor (and the original); the approved amendment is enacted
    verdict.m_Verdict = Agents::Verdict::Escalate;
    verdict.m_Question = "narrow home/set/temp?";
    Descriptor::RichMLService narrowed = service;
    narrowed.m_Actionables[0].m_Min = 18;
    narrowed.m_Actionables[0].m_Max = 21;
    verdict.m_Amended = narrowed;
    record = RunWorkflow(stack, DeployDescriptor(service), {&reviewer});
    EMPY_REQUIRE(record.has_value());
    EMPY_REQUIRE_EQ(record->m_Status, std::string("succeeded"));
    EMPY_REQUIRE_EQ(questions.size(), size_t(1));
    EMPY_CHECK(questions[0].GetBool("amended"));
    Descriptor::RichMLService shown = Descriptor::ParseDescriptor(questions[0].GetString("descriptor"), "<hitl>");
    EMPY_CHECK_EQ(Or(shown.m_Actionables[0].m_Max), 21.0);
    Descriptor::RichMLService original = Descriptor::ParseDescriptor(questions[0].GetString("original"), "<hitl>");
    EMPY_CHECK_EQ(Or(original.m_Actionables[0].m_Max), 24.0);
    const Deployment *deployment = state.FindDeployment("home-control");
    EMPY_REQUIRE(deployment != nullptr);
    EMPY_CHECK_EQ(Or(deployment->m_Descriptor.m_Actionables[0].m_Max), 21.0);
    EMPY_CHECK(record->m_Result.GetBool("amended"));
    // an approved escalation cannot launder a renaming amendment either
    Descriptor::RichMLService next = service;
    next.m_Metadata.m_Version = "1.1.0";
    renamed = next;
    renamed.m_Metadata.m_Name = "other-service";
    verdict.m_Amended = renamed;
    record = RunWorkflow(stack, DeployDescriptor(next), {&reviewer});
    EMPY_REQUIRE(record.has_value());
    EMPY_CHECK_EQ(record->m_Status, std::string("rejected"));
    EMPY_CHECK(record->m_Error.find("renames") != std::string::npos);
    EMPY_CHECK_EQ(questions.size(), size_t(2));
    EMPY_CHECK(state.FindDeployment("other-service") == nullptr);
    EMPY_CHECK_EQ(state.FindDeployment("home-control")->m_Version, std::string("1.0.0"));
}

EMPY_TEST(OpsHarness, EvolvedOntologyPersisted)
{
    // Fig. 25 map/evolve: a concept the executive accepts into the ontology is saved to the ontology state file
    TempDir temp;
    StackOptions options;
    options.m_Roles = {"executive"};
    options.m_OntologyState = temp.m_Path / "evolved.yaml";
    options.m_Reasoning = std::make_shared<ScriptedProvider>(Json::Parse(R"({
        "concepts": [{"id": "greenhouse", "kind": "domain", "term": "greenhouse", "unit": null, "topic": null,
                      "quantity": null, "min": null, "max": null, "cron": null}],
        "latencyMs": null, "statistic": null, "accuracy": null, "schedule": null, "accelerator": null,
        "inferenceRegions": [], "trainingRegions": [], "objectives": []})"));
    Stack stack(options);
    Peer client(stack.m_Broker.m_Port, "operator");
    std::optional<Envelope> reply;
    bool done = false;
    client.Get().Call(
        stack.m_Topics.Agent("executive", "interpret"),
        Envelope::Make("agent-request", Json::MakeObject({{"intent", "Deploy a greenhouse temperature predictor"}})),
        [&](eMQ::Reason reason, std::optional<Envelope> envelope)
        {
            EMPY_CHECK(!reason);
            reply = std::move(envelope);
            done = true;
        });
    EMPY_REQUIRE(Pump(stack, {&client}, [&] { return done; }));
    EMPY_REQUIRE(reply.has_value());
    EMPY_REQUIRE_EQ(reply->m_Body["result"]["evolved"].Size(), size_t(1));
    EMPY_REQUIRE(std::filesystem::exists(options.m_OntologyState));
    Semantics::Ontology saved = Semantics::Ontology::Load(options.m_OntologyState);
    EMPY_CHECK(saved.Find("greenhouse") != nullptr);
    EMPY_CHECK(saved.Size() == stack.m_Ontology.Size());
    EMPY_CHECK(Semantics::Ontology::Load(ConfigFile("Ontology.yaml")).Find("greenhouse") == nullptr); // seed intact
}
