#include <eOps/Harness/AgentHost.h>
#include <eOps/Harness/State.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include "Detail.h"
#include <set>
#include <map>
#include <array>
#include <deque>
#include <mutex>
#include <atomic>
#include <condition_variable>

namespace eOps::Harness
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        const char *const AGENT_ROLES[] = {"executive", "placement", "lifecycle", "compliance"};
        constexpr size_t ROLE_COUNT = 4;
        // locality observations are recorded at most this often per region (the network model always updates)
        constexpr auto OBSERVE_INTERVAL = std::chrono::seconds(10);

        // task failure with the MMQP reason the requester receives
        struct TaskError : std::runtime_error
        {
            uint8_t m_Code;
            TaskError(uint8_t code, const std::string &text) : std::runtime_error(text), m_Code(code)
            {
            }
        };

        size_t RoleIndex(std::string_view role)
        {
            for (size_t i = 0; i < ROLE_COUNT; ++i)
            {
                if (role == AGENT_ROLES[i])
                {
                    return i;
                }
            }
            return ROLE_COUNT;
        }

        const Json &Member(const Json &body, std::string_view key)
        {
            const Json *value = body.IsObject() ? body.Find(key) : nullptr;
            if (!value || value->IsNull())
            {
                throw TaskError(eMQ::MMQP_REASON_MALFORMED_PACKET, "request body lacks '" + std::string(key) + "'");
            }
            return *value;
        }

        // a descriptor given as JSON (Descriptor::ToJson form) or as YAML text
        Descriptor::RichMLService DescriptorOf(const Json &value)
        {
            try
            {
                if (value.IsString())
                {
                    return Descriptor::ParseDescriptor(value.AsString(), "<request>");
                }
                return Descriptor::FromJson(value);
            }
            catch (const std::exception &error)
            {
                throw TaskError(eMQ::MMQP_REASON_MALFORMED_PACKET, std::string("invalid descriptor: ") + error.what());
            }
        }

        template <typename _Callback> auto Decode(const char *what, _Callback &&decode)
        {
            try
            {
                return decode();
            }
            catch (const TaskError &)
            {
                throw;
            }
            catch (const std::exception &error)
            {
                throw TaskError(eMQ::MMQP_REASON_MALFORMED_PACKET,
                                std::string("invalid ") + what + ": " + error.what());
            }
        }

        // the harness view of one deployed service (from the retained State("services") snapshot)
        struct ServiceView
        {
            Deployment m_Deployment;
        };
    } // namespace

    struct AgentHost::Impl : std::enable_shared_from_this<Impl>
    {
        using Done = std::function<void(bool ok, Json result, uint8_t code, std::string error)>;

        AgentHostOptions m_Options;
        Agents::AgentEnvironment m_Environment;
        Agents::Executive m_Executive;
        Agents::Placement m_Placement;
        Agents::Lifecycle m_Lifecycle;
        Agents::Compliance m_Compliance;
        std::array<std::mutex, ROLE_COUNT> m_AgentMutex; // one per agent core
        std::mutex m_OntologyMutex;                      // executive + compliance read/extend the ontology
        std::array<std::atomic<uint32_t>, ROLE_COUNT> m_Busy{};
        std::atomic<uint64_t> m_Requests{0}, m_Failures{0}, m_Proposals{0};

        // in-flight worker tasks (OnDetach waits for them: they use the environment)
        std::mutex m_FlightMutex;
        std::condition_variable m_FlightDone;
        size_t m_InFlight = 0;

        // owner-thread state
        std::shared_ptr<bool> m_Alive = std::make_shared<bool>(false);
        eMQ::Service *m_Service = nullptr;
        Fabric::Bus *m_Bus = nullptr;
        const void *m_Owner = nullptr;
        uint64_t m_NextTask = 1;
        std::map<uint64_t, Fabric::Bus::Responder> m_Responders;
        std::map<std::string, ServiceView> m_Services;
        Topology::Inventory m_Inventory;
        bool m_HasInventory = false;
        std::set<std::string> m_Proposing;                   // services with a Workflow() proposal in flight
        std::map<std::string, Clock::time_point> m_Observed; // last observation per kind/key
        std::map<std::string, std::deque<Clock::time_point>> m_Violations; // compliance: recent violations per service

        Impl(AgentHostOptions options, Agents::AgentEnvironment environment)
            : m_Options(std::move(options)), m_Environment(environment), m_Executive(environment),
              m_Placement(environment, m_Options.m_Weights), m_Lifecycle(environment),
              m_Compliance(environment, m_Options.m_Policies)
        {
            m_Executive.SetValidation(m_Options.m_Validation);
        }

        bool HasRole(std::string_view role) const
        {
            return Detail::Contains(m_Options.m_Roles, role);
        }

        // ---- worker execution

        // run `work` on a worker under the role's mutex; `done` runs on the owner thread (unless detached)
        void Run(size_t role, std::function<Json()> work, Done done)
        {
            {
                std::scoped_lock lock(m_FlightMutex);
                ++m_InFlight;
            }
            ++m_Busy[role];
            std::shared_ptr<Impl> self = shared_from_this();
            std::shared_ptr<bool> alive = m_Alive;
            eMQ::Service *service = m_Service;
            auto task = [self, alive, service, role, work = std::move(work), done]() mutable
            {
                bool ok = false;
                Json result;
                uint8_t code = eMQ::MMQP_REASON_INTERNAL_ERROR;
                std::string error;
                try
                {
                    std::unique_lock<std::mutex> ontology;
                    if (role == RoleIndex("executive") || role == RoleIndex("compliance"))
                    {
                        ontology = std::unique_lock<std::mutex>(self->m_OntologyMutex);
                    }
                    std::scoped_lock lock(self->m_AgentMutex[role]);
                    result = work();
                    ok = true;
                }
                catch (const TaskError &failure)
                {
                    code = failure.m_Code;
                    error = failure.what();
                }
                catch (const std::exception &failure)
                {
                    error = failure.what();
                }
                catch (...)
                {
                    error = "agent task failed with a non-standard exception";
                }
                --self->m_Busy[role];
                {
                    std::scoped_lock lock(self->m_FlightMutex);
                    --self->m_InFlight;
                }
                self->m_FlightDone.notify_all();
                // the environment is no longer used; hand the outcome back to the owner thread
                service->Dispatch(eCore::Task(
                    [self, alive, done = std::move(done), ok, result = std::move(result), code,
                     error = std::move(error)]() mutable
                    {
                        if (*alive)
                        {
                            done(ok, std::move(result), code, std::move(error));
                        }
                    }));
            };
            try
            {
                (void)m_Service->Async(std::move(task));
            }
            catch (const std::exception &failure)
            {
                --m_Busy[role];
                {
                    std::scoped_lock lock(m_FlightMutex);
                    --m_InFlight;
                }
                done(false, Json(), eMQ::MMQP_REASON_BUSY, std::string("agent worker unavailable: ") + failure.what());
            }
        }

        // ---- served tasks (worker thread, under the role mutex)

        Json Execute(std::string_view role, std::string_view task, const Json &body)
        {
            if (role == "executive" && task == "interpret")
            {
                const Json &intent = Member(body, "intent");
                if (!intent.IsString() || Trim(intent.AsString()).empty())
                {
                    throw TaskError(eMQ::MMQP_REASON_MALFORMED_PACKET, "intent must be a non-empty string");
                }
                std::string operatorId = body.GetString("operator");
                Agents::Interpretation interpretation = m_Executive.Interpret(intent.AsString(), operatorId);
                if (!interpretation.m_Evolved.empty())
                {
                    PersistOntology(); // the map/evolve loop accumulates across restarts (ontology mutex held)
                }
                return interpretation.ToJson();
            }
            if (role == "compliance" && task == "review")
            {
                Descriptor::RichMLService candidate = DescriptorOf(Member(body, "descriptor"));
                std::vector<Descriptor::RichMLService> deployed;
                if (const Json *items = body.Find("deployed"); items && !items->IsNull())
                {
                    if (!items->IsArray())
                    {
                        throw TaskError(eMQ::MMQP_REASON_MALFORMED_PACKET, "deployed must be an array");
                    }
                    for (const Json &item : items->AsArray())
                    {
                        deployed.push_back(DescriptorOf(item));
                    }
                }
                Agents::ComplianceReport report = m_Compliance.Review(candidate, deployed);
                Remember("compliance", "review", candidate.m_Metadata.m_Name,
                         Json::MakeObject({{"verdict", Agents::VerdictName(report.m_Verdict)},
                                           {"findings", report.m_Findings.size()},
                                           {"resolutions", report.m_Resolutions.size()},
                                           {"deployed", deployed.size()},
                                           {"outcome", Agents::VerdictName(report.m_Verdict)}}));
                return report.ToJson();
            }
            if (role == "placement" && task == "place")
            {
                Descriptor::RichMLService service = DescriptorOf(Member(body, "descriptor"));
                Agents::Role placed = Decode("role", [&] { return Agents::ParseRole(body.GetString("role")); });
                Topology::Inventory inventory =
                    body.Has("inventory") && !body["inventory"].IsNull()
                        ? Decode("inventory", [&] { return Topology::Inventory::FromJson(body["inventory"]); })
                        : (m_Environment.m_Comas ? m_Environment.m_Comas->Plane().Inventory() : Topology::Inventory{});
                Topology::NetworkModel network =
                    body.Has("network") && !body["network"].IsNull()
                        ? Decode("network", [&] { return Topology::NetworkModel::FromJson(body["network"]); })
                        : (m_Environment.m_Comas ? m_Environment.m_Comas->Plane().Network() : Topology::NetworkModel{});
                std::vector<std::string> current =
                    Decode("current", [&] { return Detail::ReadStrings(body, "current"); });
                Agents::PlacementPlan plan = m_Placement.Place(service, placed, inventory, network, current);
                Remember("placement", "placement", service.m_Metadata.m_Name,
                         Json::MakeObject({{"role", Agents::RoleName(placed)},
                                           {"nodes", Detail::Strings(plan.m_Nodes)},
                                           {"current", Detail::Strings(current)},
                                           {"relocate", plan.m_Relocate},
                                           {"rationale", plan.m_Rationale},
                                           {"outcome", plan.m_Nodes.empty() ? "infeasible" : "placed"}}));
                return plan.ToJson();
            }
            if (role == "lifecycle" && task == "evaluate")
            {
                Agents::LifecycleState state =
                    Decode("state", [&] { return Agents::LifecycleState::FromJson(Member(body, "state")); });
                Descriptor::LifecyclePolicy policy =
                    Decode("policy", [&] { return Detail::PolicyFrom(body["policy"]); });
                Workload::ModelArtifact candidate =
                    Decode("candidate", [&] { return Workload::ModelArtifact::FromJson(Member(body, "candidate")); });
                uint64_t now = Decode("now", [&] { return Detail::ReadUnsigned(body, "now", NowMillis()); });
                Agents::LifecycleDecision decision = m_Lifecycle.Evaluate(state, policy, candidate, now);
                Json data = decision.ToJson();
                data.Set("candidate", candidate.m_Version);
                data.Set("accuracy", candidate.m_Accuracy);
                Remember("lifecycle", "evaluation", state.m_Service.empty() ? candidate.m_Service : state.m_Service,
                         std::move(data));
                return decision.ToJson();
            }
            throw TaskError(eMQ::MMQP_REASON_UNSPECIFIED,
                            "agent '" + std::string(role) + "' has no task '" + std::string(task) + "'");
        }

        // the evolved BoM ontology goes to the ontology state file (never over the seed), worker thread with the
        // ontology mutex held
        void PersistOntology()
        {
            if (m_Options.m_OntologyState.empty() || !m_Environment.m_Ontology)
            {
                return;
            }
            try
            {
                m_Environment.m_Ontology->Save(m_Options.m_OntologyState);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("AgentHost: evolved ontology not saved to {}: {}", m_Options.m_OntologyState.string(),
                          error.what());
            }
        }

        void Remember(std::string_view agent, std::string_view kind, std::string_view service, Json data)
        {
            if (!m_Environment.m_Comas)
            {
                return;
            }
            try
            {
                m_Environment.m_Comas->Remember(agent, kind, service, std::move(data));
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("AgentHost: CoMaS memory write failed: {}", error.what());
            }
        }

        // ---- owner thread

        void OnRequest(const std::string &role, const Fabric::Inbound &request, Fabric::Bus::Responder respond)
        {
            ++m_Requests;
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(request.m_Topic);
            std::string task = levels && levels->size() == 3 ? levels->back() : std::string();
            uint64_t id = m_NextTask++;
            m_Responders.emplace(id, std::move(respond));
            Json body = request.m_Envelope.m_Body;
            Impl *self = this;
            Run(
                RoleIndex(role), [self, role, task, body]() { return self->Execute(role, task, body); },
                [self, id, role, task](bool ok, Json result, uint8_t code, std::string error)
                {
                    auto found = self->m_Responders.find(id);
                    if (found == self->m_Responders.end())
                    {
                        return;
                    }
                    Fabric::Bus::Responder respond = std::move(found->second);
                    self->m_Responders.erase(found);
                    if (ok)
                    {
                        respond(Fabric::Envelope::Make("agent-response",
                                                       Json::MakeObject({{"result", std::move(result)}}),
                                                       self->m_Options.m_Identity),
                                {});
                        return;
                    }
                    ++self->m_Failures;
                    EMPY_WARN("AgentHost: {}/{} failed: {}", role, task, error);
                    respond(Fabric::Envelope::Make("agent-response",
                                                   Json::MakeObject({{"error", error}, {"role", role}, {"task", task}}),
                                                   self->m_Options.m_Identity),
                            {code, error});
                });
        }

        // retained harness snapshot → per-service lifecycle/descriptor view and the reserved inventory
        void OnState(const Fabric::Envelope &envelope)
        {
            const Json &body = envelope.m_Body;
            const Json *deployments = body.IsObject() ? body.Find("deployments") : nullptr;
            if (!deployments || !deployments->IsArray())
            {
                return;
            }
            std::map<std::string, ServiceView> services;
            for (const Json &item : deployments->AsArray())
            {
                try
                {
                    Deployment deployment = Deployment::FromJson(item);
                    std::string name = deployment.m_Service;
                    services[name] = ServiceView{std::move(deployment)};
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("AgentHost: ignoring a malformed deployment in the harness state: {}", error.what());
                }
            }
            m_Services = std::move(services);
            if (const Json *inventory = body.Find("inventory"); inventory && !inventory->IsNull())
            {
                try
                {
                    m_Inventory = Topology::Inventory::FromJson(*inventory);
                    m_HasInventory = true;
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("AgentHost: ignoring a malformed inventory in the harness state: {}", error.what());
                }
            }
        }

        // the agent proposes, the harness enacts (Workflow() request/response)
        void Propose(Json request)
        {
            std::string service = request.GetString("service");
            if (service.empty() || m_Proposing.count(service))
            {
                return;
            }
            m_Proposing.insert(service);
            ++m_Proposals;
            std::string kind = request.GetString("kind");
            std::string trigger = request.GetString("trigger");
            EMPY_INFO("AgentHost: proposing {} for {} ({})", kind, service, request.GetString("reason"));
            Impl *self = this;
            m_Bus->Call(
                m_Options.m_Topics.Workflow(),
                Fabric::Envelope::Make("workflow-request", request, m_Options.m_Identity),
                [self, service, kind, trigger](eMQ::Reason reason, std::optional<Fabric::Envelope> reply)
                {
                    self->m_Proposing.erase(service);
                    std::string status = reason ? "unanswered" : "unknown";
                    std::string error = reason ? reason.m_Text : std::string();
                    if (!reason && reply)
                    {
                        status = reply->m_Body.GetString("status", "unknown");
                        error = reply->m_Body.GetString("error");
                    }
                    std::string agent = kind == "relocate" ? "placement"
                                        : kind == "retire" ? "compliance"
                                                           : "lifecycle";
                    self->Remember(agent, "proposal", service,
                                   Json::MakeObject({{"kind", kind},
                                                     {"trigger", trigger},
                                                     {"status", status},
                                                     {"error", error},
                                                     {"outcome", status}}));
                },
                m_Options.m_HarnessTimeoutMs, m_Owner);
        }

        // lifecycle role: an event for `service` → decision on a worker → proposal
        void LifecycleEvent(const std::string &service, std::string trigger, Json body)
        {
            auto found = m_Services.find(service);
            if (found == m_Services.end() || m_Proposing.count(service))
            {
                return;
            }
            const Deployment &deployment = found->second.m_Deployment;
            Agents::LifecycleState state = deployment.m_Lifecycle;
            state.m_Service = service;
            state.m_HasTraining = deployment.m_Descriptor.m_Training.has_value();
            Descriptor::LifecyclePolicy policy = deployment.m_Descriptor.m_Lifecycle;
            Impl *self = this;
            Run(
                RoleIndex("lifecycle"),
                [self, state, policy, trigger, body]()
                {
                    uint64_t now = NowMillis();
                    Agents::LifecycleDecision decision;
                    if (trigger == "drift")
                    {
                        decision = self->m_Lifecycle.OnDrift(state, policy, body, now);
                    }
                    else if (trigger == "metrics")
                    {
                        decision = self->m_Lifecycle.OnMetrics(state, policy, body, now);
                    }
                    else if (trigger == "schedule")
                    {
                        decision = self->m_Lifecycle.OnSchedule(state, policy, now);
                    }
                    else
                    {
                        decision = self->m_Lifecycle.OnStatus(state, policy, body, now);
                    }
                    // routine heartbeat/metrics outcomes without action are not worth an episode
                    if (decision.m_Action != Agents::LifecycleAction::None ||
                        (trigger != "bootstrap" && trigger != "metrics"))
                    {
                        Json data = decision.ToJson();
                        data.Set("trigger", trigger);
                        data.Set("event", body);
                        self->Remember("lifecycle", "decision", state.m_Service, std::move(data));
                    }
                    return decision.ToJson();
                },
                [self, service, trigger](bool ok, Json result, uint8_t, std::string error)
                {
                    if (!ok)
                    {
                        EMPY_WARN("AgentHost: lifecycle decision for {} failed: {}", service, error);
                        return;
                    }
                    Agents::LifecycleDecision decision = Agents::LifecycleDecision::FromJson(result);
                    if (decision.m_Action != Agents::LifecycleAction::Retrain &&
                        decision.m_Action != Agents::LifecycleAction::Rollback)
                    {
                        return;
                    }
                    Json request = Json::MakeObject({{"kind", Agents::ActionName(decision.m_Action)},
                                                     {"service", service},
                                                     {"reason", decision.m_Reason},
                                                     {"trigger", trigger}});
                    if (decision.m_Target)
                    {
                        request.Set("target", *decision.m_Target);
                    }
                    // the accuracy the decision was based on (provenance; the harness judges candidates against
                    // the accuracy it observes itself)
                    if (decision.m_Observed && *decision.m_Observed >= 0 && *decision.m_Observed <= 1)
                    {
                        request.Set("observed", *decision.m_Observed);
                    }
                    self->Propose(std::move(request));
                });
        }

        void OnLifecycleMessage(const std::string &topic, const Fabric::Envelope &envelope)
        {
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
            if (!levels || levels->size() < 3)
            {
                return;
            }
            const std::vector<std::string> &path = *levels;
            if (path[0] == "trigger" && path.size() == 4 && path[3] == "training")
            {
                LifecycleEvent(path[2], "schedule", envelope.m_Body);
            }
            else if (path[0] == "event" && path.size() == 3 && (path[2] == "drift" || path[2] == "metrics"))
            {
                Record("lifecycle", path[1], path[2], envelope.m_Body);
                LifecycleEvent(path[1], path[2], envelope.m_Body);
            }
            else if (path[0] == "status" && path.size() == 3)
            {
                // bootstrap: an inference instance waiting for its first model
                const Json &body = envelope.m_Body;
                if (body.GetString("role") == "inference" && body.GetString("state") == "waiting-model")
                {
                    LifecycleEvent(path[1], "bootstrap", body);
                }
            }
        }

        // operational observations for CoMaS: every event, metrics at most every OBSERVE_INTERVAL per service
        void Record(const std::string &source, const std::string &service, const std::string &kind, const Json &body)
        {
            if (!m_Environment.m_Comas)
            {
                return;
            }
            auto now = Clock::now();
            std::string key = source + "/" + kind + "/" + service;
            auto last = m_Observed.find(key);
            if (kind == "metrics" && last != m_Observed.end() && now - last->second < OBSERVE_INTERVAL)
            {
                return;
            }
            m_Observed[key] = now;
            Json data = body.IsObject() ? body : Json::MakeObject({{"value", body}});
            data.Set("service", service);
            try
            {
                m_Environment.m_Comas->Observe(source, kind, std::move(data));
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("AgentHost: CoMaS observation failed: {}", error.what());
            }
        }

        // placement role: locality reports update the CoMaS network knowledge
        void OnLocality(const std::string &topic, const Fabric::Envelope &envelope)
        {
            if (!m_Environment.m_Comas)
            {
                return;
            }
            const Json &body = envelope.m_Body;
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
            std::string region = body.GetString("region", levels && levels->size() == 3 ? levels->back() : "");
            const Json &topics = body["topics"];
            try
            {
                m_Environment.m_Comas->Plane().UpdateNetwork([&](Topology::NetworkModel &model)
                                                             { model.ApplyLocality(region, topics); });
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("AgentHost: ignoring locality report from {}: {}", region, error.what());
                return;
            }
            auto now = Clock::now();
            auto last = m_Observed.find("locality/" + region);
            if (last == m_Observed.end() || now - last->second >= OBSERVE_INTERVAL)
            {
                m_Observed["locality/" + region] = now;
                try
                {
                    m_Environment.m_Comas->Observe("placement", "locality", body);
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("AgentHost: CoMaS observation failed: {}", error.what());
                }
            }
        }

        // placement role: audit trigger → re-place each role with its current nodes → relocate proposal
        void OnAudit(const std::string &topic, const Fabric::Envelope &)
        {
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
            if (levels && levels->size() == 4)
            {
                Reassess((*levels)[2], "audit", "");
            }
        }

        // placement role: re-place `role` ("" = every role) of `service` with its current nodes on a worker and
        // propose a relocate workflow when the placement core relocates (`trigger` names the activation)
        void Reassess(const std::string &service, const std::string &trigger, const std::string &onlyRole)
        {
            auto found = m_Services.find(service);
            if (!m_HasInventory || found == m_Services.end() || m_Proposing.count(service))
            {
                return;
            }
            const Deployment deployment = found->second.m_Deployment;
            Topology::Inventory inventory = m_Inventory;
            Impl *self = this;
            Run(
                RoleIndex("placement"),
                [self, deployment, inventory, trigger, onlyRole]()
                {
                    Topology::NetworkModel network = self->m_Environment.m_Comas
                                                         ? self->m_Environment.m_Comas->Plane().Network()
                                                         : Topology::NetworkModel{};
                    Json moves = Json::MakeArray();
                    for (Agents::Role role : {Agents::Role::Inference, Agents::Role::Training})
                    {
                        if (!onlyRole.empty() && onlyRole != Agents::RoleName(role))
                        {
                            continue;
                        }
                        std::vector<std::string> current;
                        for (const Instance &instance : deployment.m_Instances)
                        {
                            if (instance.m_Role == Agents::RoleName(role))
                            {
                                current.push_back(instance.m_Node);
                            }
                        }
                        if (current.empty())
                        {
                            continue;
                        }
                        Agents::PlacementPlan plan =
                            self->m_Placement.Place(deployment.m_Descriptor, role, inventory, network, current);
                        Json data = Json::MakeObject({{"role", Agents::RoleName(role)},
                                                      {"trigger", trigger},
                                                      {"current", Detail::Strings(current)},
                                                      {"nodes", Detail::Strings(plan.m_Nodes)},
                                                      {"relocate", plan.m_Relocate},
                                                      {"currentScore", plan.m_CurrentScore},
                                                      {"rationale", plan.m_Rationale},
                                                      {"outcome", plan.m_Relocate ? "relocate" : "keep"}});
                        self->Remember("placement", trigger == "audit" ? "audit" : "reassessment", deployment.m_Service,
                                       data);
                        if (plan.m_Relocate)
                        {
                            moves.Push(std::move(data));
                        }
                    }
                    return moves;
                },
                [self, service, trigger](bool ok, Json moves, uint8_t, std::string error)
                {
                    if (!ok)
                    {
                        EMPY_WARN("AgentHost: placement {} of {} failed: {}", trigger, service, error);
                        return;
                    }
                    if (moves.Size() == 0)
                    {
                        return;
                    }
                    Json request = Json::MakeObject({{"kind", "relocate"},
                                                     {"service", service},
                                                     {"reason", moves.AsArray().front().GetString("rationale")},
                                                     {"trigger", trigger}});
                    if (moves.Size() == 1)
                    {
                        request.Set("role", moves.AsArray().front().GetString("role"));
                    }
                    self->Propose(std::move(request));
                });
        }

        // placement role: event-driven triggers. A lost instance (failure) is redeployed (relocate with replace);
        // latency drift, resource exhaustion and accelerator loss re-evaluate the placement of the affected role
        void OnPlacementEvent(const std::string &topic, const Fabric::Envelope &envelope)
        {
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
            if (!levels || levels->size() != 3 || !envelope.m_Body.IsObject())
            {
                return;
            }
            const std::string &service = (*levels)[1];
            const std::string &kind = (*levels)[2];
            const Json &body = envelope.m_Body;
            if (kind == "drift" && body.GetString("kind") != "latency")
            {
                return; // accuracy drift is the lifecycle agent's
            }
            Record("placement", service, kind, body);
            auto found = m_Services.find(service);
            if (found == m_Services.end())
            {
                return;
            }
            if (kind == "failure")
            {
                std::string instance = body.GetString("instance");
                bool known = false;
                std::string role;
                for (const Instance &item : found->second.m_Deployment.m_Instances)
                {
                    if (item.m_Id == instance)
                    {
                        known = true;
                        role = item.m_Role;
                    }
                }
                if (!known || m_Proposing.count(service))
                {
                    return;
                }
                Json replace = Json::MakeArray();
                replace.Push(instance);
                Remember("placement", "failure", service,
                         Json::MakeObject(
                             {{"instance", instance}, {"reason", body.GetString("reason")}, {"outcome", "redeploy"}}));
                Propose(Json::MakeObject({{"kind", "relocate"},
                                          {"service", service},
                                          {"role", role},
                                          {"replace", std::move(replace)},
                                          {"reason", "instance " + instance + " lost: " + body.GetString("reason")},
                                          {"trigger", "failure"}}));
                return;
            }
            // a retrained model cannot shorten the observation-to-action path: the inference placement can
            Reassess(service, kind == "drift" ? "latency" : kind,
                     kind == "drift" ? "inference" : body.GetString("role"));
        }

        // compliance role: repeated policy violations at the execution boundary (actions outside the declared domain
        // or non-finite values; a rate limit throttling the service is recorded but not escalated) → a retirement
        // proposal that the harness puts to the operators (HITL) before enacting
        void OnViolation(const std::string &topic, const Fabric::Envelope &envelope)
        {
            std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
            if (!levels || levels->size() != 3)
            {
                return;
            }
            const std::string &service = (*levels)[1];
            const Json &body = envelope.m_Body;
            Record("compliance", service, "violation", body);
            std::string rule = body.GetString("rule");
            Remember("compliance", "violation", service,
                     Json::MakeObject({{"instance", body.GetString("instance")},
                                       {"topic", body.GetString("topic")},
                                       {"rule", rule},
                                       {"reason", body.GetString("reason")},
                                       {"outcome", "violation"}}));
            if (!m_Services.count(service) || m_Options.m_ViolationLimit == 0 || rule == "rate")
            {
                return;
            }
            auto now = Clock::now();
            std::deque<Clock::time_point> &seen = m_Violations[service];
            seen.push_back(now);
            while (!seen.empty() && now - seen.front() > std::chrono::seconds(m_Options.m_ViolationWindowSeconds))
            {
                seen.pop_front();
            }
            if (seen.size() < m_Options.m_ViolationLimit || m_Proposing.count(service))
            {
                return;
            }
            std::string reason = std::to_string(seen.size()) + " policy violations within " +
                                 std::to_string(m_Options.m_ViolationWindowSeconds) +
                                 " s (last: " + body.GetString("reason", "action rejected at the execution boundary") +
                                 ")";
            seen.clear();
            Propose(Json::MakeObject({{"kind", "retire"},
                                      {"service", service},
                                      {"escalate", true},
                                      {"reason", reason},
                                      {"trigger", "violation"}}));
        }

        void Fail(const std::string &what, const eMQ::Reason &reason)
        {
            if (reason)
            {
                throw std::runtime_error("AgentHost: " + what + " failed: " + reason.m_Text);
            }
        }

        void Attach(eMQ::Service *service, const void *owner)
        {
            for (const std::string &role : m_Options.m_Roles)
            {
                if (RoleIndex(role) == ROLE_COUNT)
                {
                    throw std::invalid_argument("AgentHost: unknown agent role '" + role + "'");
                }
            }
            m_Service = service;
            m_Bus = &Fabric::BusOf(*service);
            m_Owner = owner;
            m_Alive = std::make_shared<bool>(true); // completions of an earlier attachment stay cancelled
            const Fabric::Topics &topics = m_Options.m_Topics;
            Impl *self = this;
            for (const std::string &role : m_Options.m_Roles)
            {
                Fail("serving " + topics.AgentFilter(role),
                     m_Bus->Serve(
                         topics.AgentFilter(role),
                         [self, role](const Fabric::Inbound &request, Fabric::Bus::Responder respond)
                         { self->OnRequest(role, request, std::move(respond)); }, owner));
            }
            bool lifecycle = HasRole("lifecycle"), placement = HasRole("placement"), compliance = HasRole("compliance");
            if (lifecycle || placement || compliance)
            {
                Fail("listening to the harness state",
                     m_Bus->Listen(
                         topics.State("services"), [self](const std::string &, const Fabric::Envelope &envelope)
                         { self->OnState(envelope); }, 1, owner));
                Fail("exposing " + topics.Workflow(), m_Bus->Expose(topics.Workflow(), true));
            }
            if (lifecycle)
            {
                auto handler = [self](const std::string &topic, const Fabric::Envelope &envelope)
                { self->OnLifecycleMessage(topic, envelope); };
                for (const std::string &filter : {topics.TriggerFilter("training"), topics.EventFilter("drift"),
                                                  topics.EventFilter("metrics"), topics.HeartbeatFilter()})
                {
                    Fail("listening to " + filter, m_Bus->Listen(filter, handler, 1, owner));
                }
            }
            if (placement)
            {
                Fail("listening to " + topics.LocalityFilter(),
                     m_Bus->Listen(
                         topics.LocalityFilter(), [self](const std::string &topic, const Fabric::Envelope &envelope)
                         { self->OnLocality(topic, envelope); }, 1, owner));
                Fail("listening to " + topics.TriggerFilter("placement"),
                     m_Bus->Listen(
                         topics.TriggerFilter("placement"),
                         [self](const std::string &topic, const Fabric::Envelope &envelope)
                         { self->OnAudit(topic, envelope); }, 1, owner));
                // event-driven placement triggers: lost instances, latency drift, exhaustion, accelerator loss
                for (const char *kind : {"failure", "drift", "exhaustion", "accelerator"})
                {
                    Fail("listening to " + topics.EventFilter(kind),
                         m_Bus->Listen(
                             topics.EventFilter(kind),
                             [self](const std::string &topic, const Fabric::Envelope &envelope)
                             { self->OnPlacementEvent(topic, envelope); }, 1, owner));
                }
            }
            if (compliance)
            {
                Fail("listening to " + topics.EventFilter("violation"),
                     m_Bus->Listen(
                         topics.EventFilter("violation"),
                         [self](const std::string &topic, const Fabric::Envelope &envelope)
                         { self->OnViolation(topic, envelope); }, 1, owner));
            }
        }

        void Detach()
        {
            *m_Alive = false;
            if (m_Bus)
            {
                m_Bus->Remove(m_Owner);
            }
            m_Responders.clear(); // unanswered responders answer with an error reason
            m_Services.clear();
            m_Proposing.clear();
            m_Violations.clear();
            m_Bus = nullptr;
            // running cores use the environment, which only has to outlive the module
            std::unique_lock lock(m_FlightMutex);
            m_FlightDone.wait(lock, [this] { return m_InFlight == 0; });
        }
    };

    AgentHost::AgentHost(AgentHostOptions options, Agents::AgentEnvironment environment)
        : m_Impl(std::make_shared<Impl>(std::move(options), environment))
    {
    }

    AgentHost::~AgentHost()
    {
        if (m_Impl && *m_Impl->m_Alive)
        {
            m_Impl->Detach();
        }
    }

    Json AgentHost::Status() const
    {
        Json busy = Json::MakeObject();
        for (const std::string &role : m_Impl->m_Options.m_Roles)
        {
            size_t index = RoleIndex(role);
            busy.Set(role, index < ROLE_COUNT && m_Impl->m_Busy[index].load() > 0);
        }
        return Json::MakeObject({{"identity", m_Impl->m_Options.m_Identity},
                                 {"roles", Detail::Strings(m_Impl->m_Options.m_Roles)},
                                 {"requests", m_Impl->m_Requests.load()},
                                 {"failures", m_Impl->m_Failures.load()},
                                 {"proposals", m_Impl->m_Proposals.load()},
                                 {"busy", std::move(busy)}});
    }

    void AgentHost::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        try
        {
            m_Impl->Attach(service, this);
        }
        catch (...)
        {
            m_Impl->Detach();
            throw;
        }
    }

    void AgentHost::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
    }

    void AgentHost::OnDetach(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->Detach();
    }
} // namespace eOps::Harness
