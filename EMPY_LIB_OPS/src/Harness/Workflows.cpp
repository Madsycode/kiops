#include "Engine.h"
#include "Detail.h"
#include <eOps/Common/Text.h>
#include <eOps/Descriptor/Parser.h>
#include <eCore/Common/Stdout.h>
#include <charconv>
#include <algorithm>

namespace eOps::Harness
{
    namespace
    {
        constexpr size_t MAX_HISTORY = 64;  // lifecycle records kept per service
        constexpr size_t MAX_ARTIFACTS = 8; // newest artefacts kept besides the active and rollback ones
        constexpr uint32_t MAX_REPLICA = 1024;

        const Descriptor::Resources &RoleResources(const Descriptor::RichMLService &service, Agents::Role role)
        {
            if (role == Agents::Role::Inference && service.m_Inference)
            {
                return service.m_Inference->m_Resources;
            }
            if (role == Agents::Role::Training && service.m_Training)
            {
                return service.m_Training->m_Resources;
            }
            throw std::logic_error(std::string("service declares no ") + Agents::RoleName(role) + " role");
        }

        // instances of `role` in `deployment` (nullptr → none)
        std::vector<const Instance *> RoleInstances(const Deployment *deployment, std::string_view role)
        {
            std::vector<const Instance *> instances;
            if (deployment)
            {
                for (const Instance &instance : deployment->m_Instances)
                {
                    if (instance.m_Role == role)
                    {
                        instances.push_back(&instance);
                    }
                }
            }
            return instances;
        }

        Json PlanSummary(const Agents::PlacementPlan &plan)
        {
            Json candidates = Json::MakeArray();
            for (const Agents::Candidate &candidate : plan.m_Candidates)
            {
                candidates.Push(Json::MakeObject({{"node", candidate.m_Node},
                                                  {"score", candidate.m_Score},
                                                  {"feasible", candidate.m_Feasible},
                                                  {"reasons", Detail::Strings(candidate.m_Reasons)}}));
            }
            return Json::MakeObject({{"nodes", Detail::Strings(plan.m_Nodes)},
                                     {"rationale", plan.m_Rationale},
                                     {"relocate", plan.m_Relocate},
                                     {"currentScore", plan.m_CurrentScore},
                                     {"advisor", plan.m_Advisor},
                                     {"candidates", std::move(candidates)}});
        }

        Json SpecsJson(const std::vector<Substrate::DeploymentSpec> &specs)
        {
            Json items = Json::MakeArray();
            for (const Substrate::DeploymentSpec &spec : specs)
            {
                items.Push(Json::MakeObject({{"instance", spec.m_Instance},
                                             {"role", spec.m_Role},
                                             {"node", spec.m_Node},
                                             {"region", spec.m_Region},
                                             {"broker", spec.m_BrokerHost + ":" + std::to_string(spec.m_BrokerPort)}}));
            }
            return items;
        }

        std::vector<std::string> SpecInstances(const std::vector<Substrate::DeploymentSpec> &specs)
        {
            std::vector<std::string> instances;
            for (const Substrate::DeploymentSpec &spec : specs)
            {
                instances.push_back(spec.m_Instance);
            }
            return instances;
        }

        // "host:port" or "[v6]:port" of a node-local broker; throws std::invalid_argument
        std::pair<std::string, uint16_t> BrokerAddress(const std::string &node, std::string_view text)
        {
            size_t colon = text.rfind(':');
            std::string_view host = colon == std::string_view::npos ? std::string_view{} : text.substr(0, colon);
            std::string_view port = colon == std::string_view::npos ? std::string_view{} : text.substr(colon + 1);
            if (host.size() >= 2 && host.front() == '[' && host.back() == ']')
            {
                host = host.substr(1, host.size() - 2);
            }
            uint32_t number = 0;
            auto parsed = std::from_chars(port.data(), port.data() + port.size(), number);
            if (host.empty() || port.empty() || parsed.ec != std::errc{} || parsed.ptr != port.data() + port.size() ||
                number == 0 || number > 65535)
            {
                throw std::invalid_argument("node '" + node + "' names the broker '" + std::string(text) +
                                            "'; expected host:port");
            }
            return {std::string(host), static_cast<uint16_t>(number)};
        }

        // deploy every spec; on a failure the ones already deployed are retired again
        Json DeployAll(Substrate::ExecutionSubstrate &substrate, const std::vector<Substrate::DeploymentSpec> &specs)
        {
            Json results = Json::MakeArray();
            std::vector<std::string> deployed;
            for (const Substrate::DeploymentSpec &spec : specs)
            {
                Substrate::EnactResult result;
                try
                {
                    result = substrate.Deploy(spec);
                }
                catch (const std::exception &error)
                {
                    result.m_Ok = false;
                    result.m_Instance = spec.m_Instance;
                    result.m_Detail = error.what();
                }
                results.Push(result.ToJson());
                if (!result.m_Ok)
                {
                    for (const std::string &instance : deployed)
                    {
                        try
                        {
                            substrate.Retire(instance);
                        }
                        catch (const std::exception &error)
                        {
                            EMPY_WARN("ExecutionHarness: rollback of {} failed: {}", instance, error.what());
                        }
                    }
                    return Json::MakeObject({{"ok", false},
                                             {"results", std::move(results)},
                                             {"error", spec.m_Instance + ": " + result.m_Detail}});
                }
                deployed.push_back(result.m_Instance);
            }
            return Json::MakeObject({{"ok", true}, {"results", std::move(results)}});
        }

        Json RetireAll(Substrate::ExecutionSubstrate &substrate, const std::vector<std::string> &instances)
        {
            Json results = Json::MakeArray();
            bool ok = true;
            for (const std::string &instance : instances)
            {
                Substrate::EnactResult result;
                try
                {
                    result = substrate.Retire(instance);
                }
                catch (const std::exception &error)
                {
                    result.m_Ok = false;
                    result.m_Instance = instance;
                    result.m_Detail = error.what();
                }
                ok = ok && result.m_Ok;
                results.Push(result.ToJson());
            }
            return Json::MakeObject({{"ok", ok}, {"results", std::move(results)}});
        }

        Agents::ModelRecord *History(Deployment &deployment, uint32_t version)
        {
            for (Agents::ModelRecord &record : deployment.m_Lifecycle.m_History)
            {
                if (record.m_Version == version)
                {
                    return &record;
                }
            }
            return nullptr;
        }

        // the replaced active model keeps its last known accuracy: when the harness saw it degrade, the observed
        // value replaces the training-time one (a later rollback must not return to a model known to fail)
        void RecordObserved(Deployment &deployment, uint32_t version, std::optional<double> observed)
        {
            Agents::ModelRecord *record = History(deployment, version);
            if (record && observed && *observed < record->m_Accuracy)
            {
                record->m_Accuracy = *observed;
            }
        }
    } // namespace

    // -------------------------
    // shared helpers
    // -------------------------

    std::set<std::string> ExecutionHarness::Impl::TakenInstances() const
    {
        std::set<std::string> taken;
        const Json reservations = m_State.Reservations(); // named: a range over a temporary's array would dangle
        for (const Json &reservation : reservations.AsArray())
        {
            taken.insert(reservation.GetString("instance"));
        }
        for (const Deployment *deployment : m_State.Deployments())
        {
            for (const Instance &instance : deployment->m_Instances)
            {
                taken.insert(instance.m_Id);
            }
        }
        return taken;
    }

    std::vector<Substrate::DeploymentSpec> ExecutionHarness::Impl::Specs(const Descriptor::RichMLService &service,
                                                                         Agents::Role role,
                                                                         const Agents::PlacementPlan &plan,
                                                                         std::set<std::string> &taken) const
    {
        Topology::Inventory inventory = BaseInventory();
        std::vector<Substrate::DeploymentSpec> specs;
        for (size_t i = 0; i < plan.m_Nodes.size(); ++i)
        {
            Substrate::DeploymentSpec spec;
            spec.m_Service = service.m_Metadata.m_Name;
            spec.m_Version = service.m_Metadata.m_Version;
            spec.m_Role = Agents::RoleName(role);
            spec.m_Descriptor = service;
            spec.m_Node = plan.m_Nodes[i];
            const Topology::Node *node = inventory.Find(spec.m_Node);
            spec.m_Region = node ? node->m_Region : std::string();
            // session locality: an instance joins the broker of its node, else the harness broker
            spec.m_BrokerHost = m_Options.m_BrokerHost;
            spec.m_BrokerPort = m_Options.m_BrokerPort;
            if (node && !node->m_Broker.empty())
            {
                std::tie(spec.m_BrokerHost, spec.m_BrokerPort) = BrokerAddress(node->m_Id, node->m_Broker);
            }
            spec.m_TopicRoot = m_Options.m_Topics.m_Root;
            // the declared identity authenticates the workload session (with the auth key of its client config)
            if (const std::string &identity = service.m_Constraints.m_Security.m_Identity; !identity.empty())
            {
                spec.m_ExtraArgs = {"--identity", identity};
            }
            uint32_t replica = static_cast<uint32_t>(i);
            std::string name = Substrate::InstanceName(spec.m_Service, spec.m_Role, spec.m_Node, replica);
            // a redeploy or relocation starts new instances next to the ones it replaces
            while (taken.count(name))
            {
                if (++replica > MAX_REPLICA)
                {
                    throw std::runtime_error("no free instance name for " + spec.m_Service + " on " + spec.m_Node);
                }
                name = Substrate::InstanceName(spec.m_Service, spec.m_Role, spec.m_Node, replica);
            }
            taken.insert(name);
            spec.m_Replica = replica;
            spec.m_Instance = name;
            specs.push_back(std::move(spec));
        }
        return specs;
    }

    void ExecutionHarness::Impl::ClearTraining(const std::string &service)
    {
        if (Deployment *deployment = m_State.FindDeployment(service); deployment && deployment->m_Lifecycle.m_Training)
        {
            deployment->m_Lifecycle.m_Training = false;
            m_State.PutDeployment(*deployment);
        }
    }

    void ExecutionHarness::Impl::Prune(Deployment &deployment)
    {
        Agents::LifecycleState &lifecycle = deployment.m_Lifecycle;
        std::set<uint32_t> keep;
        if (lifecycle.m_Active)
        {
            keep.insert(*lifecycle.m_Active);
        }
        if (const Agents::ModelRecord *previous = lifecycle.PreviousRecord())
        {
            keep.insert(previous->m_Version);
        }
        size_t newest = 0;
        for (auto it = deployment.m_Artifacts.rbegin(); it != deployment.m_Artifacts.rend(); ++it)
        {
            if (newest++ < MAX_ARTIFACTS)
            {
                keep.insert(it->first);
            }
        }
        for (auto it = deployment.m_Artifacts.begin(); it != deployment.m_Artifacts.end();)
        {
            it = keep.count(it->first) ? std::next(it) : deployment.m_Artifacts.erase(it);
        }
        while (lifecycle.m_History.size() > MAX_HISTORY)
        {
            auto victim =
                std::find_if(lifecycle.m_History.begin(), lifecycle.m_History.end(),
                             [&](const Agents::ModelRecord &record) { return !keep.count(record.m_Version); });
            if (victim == lifecycle.m_History.end())
            {
                break;
            }
            lifecycle.m_History.erase(victim);
        }
    }

    void ExecutionHarness::Impl::Start(const std::string &id)
    {
        const WorkflowRecord *record = Record(id);
        if (!record)
        {
            return;
        }
        const std::string &kind = record->m_Kind;
        if (kind == "deploy")
        {
            StartDeploy(id);
        }
        else if (kind == "retrain")
        {
            StartRetrain(id);
        }
        else if (kind == "rollback")
        {
            StartRollback(id);
        }
        else if (kind == "relocate")
        {
            StartRelocate(id);
        }
        else if (kind == "retire")
        {
            StartRetire(id);
        }
        else
        {
            Finish(id, "rejected", Json(), "unknown workflow kind '" + kind + "'");
        }
    }

    // -------------------------
    // readiness gate
    // -------------------------

    void ExecutionHarness::Impl::WaitFor(const std::string &id, std::vector<std::string> instances,
                                         std::function<void()> ready, std::function<void(const Json &missing)> late)
    {
        if (!m_Options.m_WaitForReady)
        {
            ready();
            return;
        }
        std::shared_ptr<Run> run = Find(id);
        Begin(id, "ready");
        run->m_ReadyInstances = std::move(instances);
        run->m_OnReady = std::move(ready);
        run->m_OnLate = std::move(late);
        run->m_WaitingReady = true;
        run->m_ReadyDeadline = Clock::now() + std::chrono::milliseconds(m_Options.m_ReadyTimeoutMs);
        CheckReady();
    }

    // -------------------------
    // deploy: [interpret] → validate → review [→ HITL] → place → enact → record → schedule [→ ready]
    // -------------------------

    void ExecutionHarness::Impl::StartDeploy(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        const Json input = record->m_Input;
        run->m_DryRun = input.GetBool("dryRun");
        if (const Json *intent = input.Find("intent"); intent && intent->IsString())
        {
            Begin(id, "interpret");
            Json body = Json::MakeObject({{"intent", *intent}, {"operator", input.GetString("operator")}});
            Ask(id, "executive", "interpret", std::move(body),
                [this, id](const Json &result)
                {
                    Descriptor::RichMLService service = Descriptor::FromJson(result["descriptor"]);
                    End(id, "succeeded",
                        Json::MakeObject({{"frame", result["frame"]},
                                          {"issues", result["issues"]},
                                          {"evolved", result["evolved"]},
                                          {"plan", result["plan"]},
                                          {"valid", result["valid"]},
                                          {"yaml", result["yaml"]}}));
                    Find(id)->m_Candidate = std::move(service);
                    Validate(id);
                });
            return;
        }
        Begin(id, "parse");
        try
        {
            const Json &descriptor = input["descriptor"];
            run->m_Candidate = descriptor.IsString() ? Descriptor::ParseDescriptor(descriptor.AsString(), "<deploy>")
                                                     : Descriptor::FromJson(descriptor);
        }
        catch (const std::exception &error)
        {
            End(id, "failed", Json::MakeObject({{"error", error.what()}}));
            Finish(id, "rejected", Json(), std::string("descriptor: ") + error.what());
            return;
        }
        End(id, "succeeded",
            Json::MakeObject({{"service", run->m_Candidate->m_Metadata.m_Name},
                              {"version", run->m_Candidate->m_Metadata.m_Version}}));
        Validate(id);
    }

    void ExecutionHarness::Impl::Validate(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        const Descriptor::RichMLService &service = *run->m_Candidate;
        const std::string name = service.m_Metadata.m_Name;
        Begin(id, "validate");
        GuardResult check = m_Guard.CheckDescriptor(service, m_Options.m_Validation);
        // the requester's service name (if any) is not this workflow's claim on the service yet
        std::string requested = record->m_Service;
        record->m_Service.clear();
        if (!requested.empty() && requested != name)
        {
            check.Fail("the request names service '" + requested + "' but the descriptor declares '" + name + "'");
        }
        // declared security requirements must be enforceable by the workload sessions the substrate starts
        if (service.m_Constraints.m_Security.m_Encryption && !m_Options.m_SecureWorkloads)
        {
            check.Fail("'" + name +
                       "' declares security.encryption, but the workload MMQP sessions of this harness are not "
                       "TLS/QUIC (configure the substrate client_config with tls or quic)");
        }
        for (const char *kind : {"deploy", "relocate", "retire"})
        {
            if (m_State.HasRunning(kind, name))
            {
                check.Fail(std::string("a ") + kind + " workflow is already running for '" + name + "'");
            }
        }
        if (const Deployment *existing = m_State.FindDeployment(name);
            existing && Detail::CompareVersions(service.m_Metadata.m_Version, existing->m_Version) <= 0)
        {
            check.Fail("version " + service.m_Metadata.m_Version + " of '" + name +
                       "' is not newer than the deployed " + existing->m_Version);
        }
        record->m_Service = name;
        m_State.SaveWorkflow(*record);
        Json detail =
            Json::MakeObject({{"guard", check.ToJson()},
                              {"issues", Descriptor::ToJson(Descriptor::Validate(service, m_Options.m_Validation))}});
        if (!check.m_Ok)
        {
            End(id, "failed", std::move(detail));
            Finish(id, "rejected", Json::MakeObject({{"guard", check.ToJson()}}), Join(check.m_Violations, "; "));
            return;
        }
        End(id, "succeeded", std::move(detail));
        Review(id);
    }

    void ExecutionHarness::Impl::Review(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        const Descriptor::RichMLService &service = *run->m_Candidate;
        run->m_Reviewed = service; // an amendment may only restrict what was reviewed
        Json deployed = Json::MakeArray();
        for (const Deployment *deployment : m_State.Deployments())
        {
            if (deployment->m_Service != service.m_Metadata.m_Name)
            {
                deployed.Push(Descriptor::ToJson(deployment->m_Descriptor));
            }
        }
        Begin(id, "review");
        Json body = Json::MakeObject({{"descriptor", Descriptor::ToJson(service)}, {"deployed", std::move(deployed)}});
        Ask(id, "compliance", "review", std::move(body),
            [this, id](const Json &result)
            {
                std::shared_ptr<Run> run = Find(id);
                run->m_Report = Agents::ComplianceReport::FromJson(result);
                const Agents::ComplianceReport &report = run->m_Report;
                Json detail = report.ToJson();
                detail.Erase("amended");
                switch (report.m_Verdict)
                {
                case Agents::Verdict::Approved:
                case Agents::Verdict::Amended:
                {
                    GuardResult check = m_Guard.CheckCompliance(*run->m_Reviewed, report, m_Options.m_Validation);
                    detail.Set("guard", check.ToJson());
                    if (!check.m_Ok)
                    {
                        End(id, "failed", std::move(detail));
                        Finish(id, "rejected", Json::MakeObject({{"guard", check.ToJson()}}),
                               "compliance guard: " + Join(check.m_Violations, "; "));
                        return;
                    }
                    if (report.m_Verdict == Agents::Verdict::Amended)
                    {
                        run->m_Candidate = *report.m_Amended;
                        detail.Set("amendedDescriptor", Descriptor::EmitDescriptor(*report.m_Amended));
                    }
                    End(id, "succeeded", std::move(detail));
                    PlaceAll(id);
                    return;
                }
                case Agents::Verdict::Escalate:
                    End(id, "succeeded", std::move(detail));
                    Escalate(id);
                    return;
                case Agents::Verdict::Rejected:
                default:
                {
                    std::vector<std::string> errors;
                    for (const Agents::Finding &finding : report.m_Findings)
                    {
                        if (finding.m_Severity == "error")
                        {
                            errors.push_back(finding.m_Criterion + ": " + finding.m_Message);
                        }
                    }
                    End(id, "failed", std::move(detail));
                    Finish(id, "rejected", Json::MakeObject({{"compliance", report.ToJson()}}),
                           "compliance rejected: " +
                               (errors.empty() ? std::string("no reason given") : Join(errors, "; ")));
                    return;
                }
                }
            });
    }

    void ExecutionHarness::Impl::Escalate(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        const Agents::ComplianceReport &report = run->m_Report;
        Json findings = Json::MakeArray();
        for (const Agents::Finding &finding : report.m_Findings)
        {
            findings.Push(finding.ToJson());
        }
        Json resolutions = Json::MakeArray();
        for (const Json &resolution : report.m_Resolutions)
        {
            resolutions.Push(resolution);
        }
        // the operator approves exactly what would be enacted: the amended descriptor when there is one
        const Descriptor::RichMLService &enacted = report.m_Amended ? *report.m_Amended : *run->m_Candidate;
        Json body = Json::MakeObject({{"service", run->m_Candidate->m_Metadata.m_Name},
                                      {"question", report.m_Question},
                                      {"findings", std::move(findings)},
                                      {"resolutions", std::move(resolutions)},
                                      {"descriptor", Descriptor::EmitDescriptor(enacted)},
                                      {"amended", report.m_Amended.has_value()}});
        if (report.m_Amended)
        {
            body.Set("original", Descriptor::EmitDescriptor(*run->m_Candidate));
        }
        AskHuman(id, std::move(body),
                 [this, id](const Detail::HitlAnswer &answer)
                 {
                     std::shared_ptr<Run> run = Find(id);
                     Json detail = answer.ToJson();
                     if (!answer.m_Approve)
                     {
                         End(id, "failed", detail);
                         Finish(id, "rejected",
                                Json::MakeObject({{"hitl", detail}, {"compliance", run->m_Report.ToJson()}}),
                                "escalation rejected " + answer.Who());
                         return;
                     }
                     GuardResult check;
                     if (run->m_Report.m_Amended)
                     {
                         check = m_Guard.CheckAmendment(*run->m_Reviewed, *run->m_Report.m_Amended);
                         if (check.m_Ok)
                         {
                             run->m_Candidate = *run->m_Report.m_Amended;
                         }
                     }
                     if (check.m_Ok)
                     {
                         check = m_Guard.CheckDescriptor(*run->m_Candidate, m_Options.m_Validation);
                     }
                     detail.Set("guard", check.ToJson());
                     if (!check.m_Ok)
                     {
                         End(id, "failed", std::move(detail));
                         Finish(id, "rejected", Json(), Join(check.m_Violations, "; "));
                         return;
                     }
                     End(id, "succeeded", std::move(detail));
                     PlaceAll(id);
                 });
    }

    void ExecutionHarness::Impl::PlaceAll(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        bool inference = run->m_Candidate->m_Inference.has_value();
        bool training = run->m_Candidate->m_Training.has_value();
        auto enact = [this, id] { EnactDeploy(id); };
        auto trainingStep = [this, id, training, enact]
        {
            if (training)
            {
                PlaceRole(id, Agents::Role::Training, enact);
            }
            else
            {
                enact();
            }
        };
        if (inference)
        {
            PlaceRole(id, Agents::Role::Inference, trainingStep);
        }
        else
        {
            trainingStep();
        }
    }

    void ExecutionHarness::Impl::PlaceRole(const std::string &id, Agents::Role role, std::function<void()> next)
    {
        std::shared_ptr<Run> run = Find(id);
        const Descriptor::RichMLService &service = *run->m_Candidate;
        const std::string roleName = Agents::RoleName(role);
        std::vector<std::string> current, previous;
        for (const Instance *instance : RoleInstances(m_State.FindDeployment(service.m_Metadata.m_Name), roleName))
        {
            current.push_back(instance->m_Node);
            previous.push_back(instance->m_Id);
        }
        Topology::Inventory base = BaseInventory();
        Topology::Inventory offered = m_State.ApplyReservations(base);
        Topology::Inventory guarded = m_State.ApplyReservationsExcept(base, previous);
        if (role == Agents::Role::Training && run->m_InferencePlan)
        {
            // the inference instances of this deployment are not reserved yet
            const Descriptor::Resources &resources = RoleResources(service, Agents::Role::Inference);
            for (const std::string &node : run->m_InferencePlan->m_Nodes)
            {
                for (Topology::Inventory *inventory : {&offered, &guarded})
                {
                    if (Topology::Node *found = inventory->Find(node))
                    {
                        found->Reserve(resources);
                    }
                }
            }
        }
        Begin(id, "place-" + roleName);
        Topology::NetworkModel network = Network();
        Json body = Json::MakeObject({{"descriptor", Descriptor::ToJson(service)},
                                      {"role", roleName},
                                      {"inventory", offered.ToJson()},
                                      {"network", network.ToJson()},
                                      {"current", Detail::Strings(current)}});
        Ask(id, "placement", "place", std::move(body),
            [this, id, role, roleName, guarded, network, next = std::move(next)](const Json &result)
            {
                std::shared_ptr<Run> run = Find(id);
                Agents::PlacementPlan plan = Agents::PlacementPlan::FromJson(result);
                GuardResult check = m_Guard.CheckPlacement(*run->m_Candidate, role, plan, guarded, network);
                Json detail = PlanSummary(plan);
                detail.Set("guard", check.ToJson());
                if (!check.m_Ok)
                {
                    End(id, "failed", std::move(detail));
                    Finish(id, "rejected", Json::MakeObject({{"placement", PlanSummary(plan)}}),
                           roleName + " placement: " + Join(check.m_Violations, "; "));
                    return;
                }
                End(id, "succeeded", std::move(detail));
                (role == Agents::Role::Inference ? run->m_InferencePlan : run->m_TrainingPlan) = std::move(plan);
                next();
            });
    }

    void ExecutionHarness::Impl::EnactDeploy(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        const Descriptor::RichMLService &service = *run->m_Candidate;
        Json placement = Json::MakeObject();
        if (run->m_InferencePlan)
        {
            placement.Set("inference", Detail::Strings(run->m_InferencePlan->m_Nodes));
        }
        if (run->m_TrainingPlan)
        {
            placement.Set("training", Detail::Strings(run->m_TrainingPlan->m_Nodes));
        }
        if (run->m_DryRun)
        {
            Finish(id, "succeeded",
                   Json::MakeObject({{"dryRun", true},
                                     {"service", service.m_Metadata.m_Name},
                                     {"version", service.m_Metadata.m_Version},
                                     {"compliance", Agents::VerdictName(run->m_Report.m_Verdict)},
                                     {"placement", std::move(placement)},
                                     {"descriptor", Descriptor::EmitDescriptor(service)}}));
            return;
        }
        std::set<std::string> taken = TakenInstances();
        std::vector<Substrate::DeploymentSpec> specs;
        if (run->m_InferencePlan)
        {
            specs = Specs(service, Agents::Role::Inference, *run->m_InferencePlan, taken);
        }
        if (run->m_TrainingPlan)
        {
            for (Substrate::DeploymentSpec &spec : Specs(service, Agents::Role::Training, *run->m_TrainingPlan, taken))
            {
                specs.push_back(std::move(spec));
            }
        }
        for (const Substrate::DeploymentSpec &spec : specs)
        {
            m_State.Reserve(spec.m_Instance, spec.m_Node, RoleResources(service, Agents::ParseRole(spec.m_Role)));
        }
        Begin(id, "enact");
        auto release = [this, specs]
        {
            for (const Substrate::DeploymentSpec &spec : specs)
            {
                m_State.Release(spec.m_Instance);
            }
        };
        try
        {
            std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
            Enact(
                id, [substrate, specs] { return DeployAll(*substrate, specs); },
                [this, id, specs, release](const Json &outcome)
                {
                    if (!outcome.GetBool("ok"))
                    {
                        release();
                        End(id, "failed", outcome);
                        Finish(id, "failed", Json::MakeObject({{"enact", outcome}}),
                               "substrate deploy failed: " + outcome.GetString("error"));
                        return;
                    }
                    Json detail = outcome;
                    detail.Set("substrate", m_Substrate->Name());
                    detail.Set("instances", SpecsJson(specs));
                    End(id, "succeeded", std::move(detail));
                    RecordDeploy(id, specs);
                });
        }
        catch (...)
        {
            release();
            throw;
        }
    }

    void ExecutionHarness::Impl::RecordDeploy(const std::string &id,
                                              const std::vector<Substrate::DeploymentSpec> &specs)
    {
        std::shared_ptr<Run> run = Find(id);
        const Descriptor::RichMLService &service = *run->m_Candidate;
        const std::string name = service.m_Metadata.m_Name;
        Deployment deployment;
        deployment.m_Service = name;
        deployment.m_Version = service.m_Metadata.m_Version;
        deployment.m_Descriptor = service;
        deployment.m_Status = "deployed";
        for (const Substrate::DeploymentSpec &spec : specs)
        {
            deployment.m_Instances.push_back(Instance{spec.m_Instance, spec.m_Role, spec.m_Node, "deploying", 0});
        }
        if (const Deployment *existing = m_State.FindDeployment(name))
        {
            run->m_Previous = existing->m_Instances;
            run->m_Replaced = *existing;
            // models trained on the same features stay valid across a redeploy
            if (Workload::FeatureNames(existing->m_Descriptor) == Workload::FeatureNames(service))
            {
                deployment.m_Lifecycle = existing->m_Lifecycle;
                deployment.m_Artifacts = existing->m_Artifacts;
            }
        }
        deployment.m_Lifecycle.m_Service = name;
        deployment.m_Lifecycle.m_HasTraining = service.m_Training.has_value();
        deployment.m_Lifecycle.m_Training = false;
        m_State.PutDeployment(std::move(deployment));
        if (eMQ::Reason reason = m_Bus->Expose(m_Options.m_Topics.WorkloadFilter(name), true))
        {
            EMPY_WARN("ExecutionHarness: workload request profile of {} not registered: {}", name, reason.m_Text);
        }
        // temporal triggers: training schedule and placement audit
        m_Scheduler.Remove(name, "training");
        m_Scheduler.Remove(name, "placement");
        Begin(id, "schedule");
        if (service.m_Training && !service.m_Training->m_Schedule.empty())
        {
            m_Scheduler.AddCron(name, "training", service.m_Training->m_Schedule);
        }
        if (m_Options.m_PlacementAuditSeconds > 0)
        {
            m_Scheduler.AddInterval(name, "placement", std::chrono::seconds(m_Options.m_PlacementAuditSeconds));
        }
        Json schedules = Json::MakeArray();
        for (const ScheduleEntry &entry : m_Scheduler.Entries())
        {
            if (entry.m_Service == name)
            {
                schedules.Push(entry.ToJson());
            }
        }
        End(id, "succeeded", Json::MakeObject({{"schedules", std::move(schedules)}}));
        Publish();
        if (run->m_Previous.empty())
        {
            AwaitReady(id);
            return;
        }
        // a redeploy keeps the replaced instances until the new ones are ready (inference serving the active model,
        // training holding what the replaced trainer had collected, up to training.minSamples)
        int64_t collected = 0;
        for (const Instance &instance : run->m_Previous)
        {
            if (auto beat = m_Beats.find(instance.m_Id); instance.m_Role == "training" && beat != m_Beats.end())
            {
                collected = std::max(collected, beat->second.m_Samples);
            }
        }
        for (const Substrate::DeploymentSpec &spec : specs)
        {
            if (spec.m_Role == "training" && service.m_Training)
            {
                run->m_ReadySamples[spec.m_Instance] = std::min<int64_t>(collected, service.m_Training->m_MinSamples);
            }
        }
        std::vector<std::string> fresh = SpecInstances(specs);
        WaitFor(
            id, fresh, [this, id] { RetirePrevious(id); },
            [this, id, name, fresh](const Json &missing)
            {
                // the new instances never became ready: they are retired and the replaced deployment stays
                std::shared_ptr<Run> run = Find(id);
                Deployment restored = *run->m_Replaced;
                if (const Deployment *current = m_State.FindDeployment(name);
                    current &&
                    Workload::FeatureNames(current->m_Descriptor) == Workload::FeatureNames(restored.m_Descriptor))
                {
                    restored.m_Lifecycle = current->m_Lifecycle; // a promotion meanwhile reached every instance
                    restored.m_Artifacts = current->m_Artifacts;
                    restored.m_Lifecycle.m_HasTraining = restored.m_Descriptor.m_Training.has_value();
                }
                m_State.PutDeployment(restored);
                m_Scheduler.Remove(name, "training");
                if (restored.m_Descriptor.m_Training && !restored.m_Descriptor.m_Training->m_Schedule.empty())
                {
                    m_Scheduler.AddCron(name, "training", restored.m_Descriptor.m_Training->m_Schedule);
                }
                Publish();
                Begin(id, "retire-new");
                std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
                Enact(
                    id, [substrate, fresh] { return RetireAll(*substrate, fresh); },
                    [this, id, name, fresh, missing, version = restored.m_Version](const Json &outcome)
                    {
                        for (const std::string &instance : fresh)
                        {
                            m_State.Release(instance);
                        }
                        End(id, outcome.GetBool("ok") ? "succeeded" : "failed", outcome);
                        Finish(id, "failed", Json::MakeObject({{"service", name}, {"missing", missing}}),
                               "new instances did not report ready within " +
                                   std::to_string(m_Options.m_ReadyTimeoutMs) + " ms; version " + version +
                                   " keeps running");
                    });
            });
    }

    void ExecutionHarness::Impl::RetirePrevious(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        std::vector<std::string> previous;
        for (const Instance &instance : run->m_Previous)
        {
            previous.push_back(instance.m_Id);
        }
        Begin(id, "retire-previous");
        std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
        Enact(
            id, [substrate, previous] { return RetireAll(*substrate, previous); },
            [this, id, previous](const Json &outcome)
            {
                for (const std::string &instance : previous)
                {
                    m_State.Release(instance);
                    m_Beats.erase(instance);
                }
                End(id, outcome.GetBool("ok") ? "succeeded" : "failed", outcome);
                Publish();
                CompleteDeploy(id);
            });
    }

    void ExecutionHarness::Impl::AwaitReady(const std::string &id)
    {
        const WorkflowRecord *record = Record(id);
        std::vector<std::string> instances;
        if (const Deployment *deployment = record ? m_State.FindDeployment(record->m_Service) : nullptr)
        {
            for (const Instance &instance : deployment->m_Instances)
            {
                instances.push_back(instance.m_Id);
            }
        }
        WaitFor(
            id, std::move(instances), [this, id] { CompleteDeploy(id); },
            [this, id](const Json &missing)
            {
                WorkflowRecord *record = Record(id);
                if (Deployment *deployment = m_State.FindDeployment(record->m_Service))
                {
                    deployment->m_Status = "degraded";
                    m_State.PutDeployment(*deployment);
                }
                Finish(id, "failed", Json::MakeObject({{"service", record->m_Service}, {"missing", missing}}),
                       "instances did not report ready within " + std::to_string(m_Options.m_ReadyTimeoutMs) + " ms");
            });
    }

    void ExecutionHarness::Impl::CompleteDeploy(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        if (!run || !record)
        {
            return;
        }
        const Deployment *deployment = m_State.FindDeployment(record->m_Service);
        Json instances = Json::MakeArray();
        Json schedules = Json::MakeArray();
        if (deployment)
        {
            for (const Instance &instance : deployment->m_Instances)
            {
                instances.Push(instance.ToJson());
            }
        }
        for (const ScheduleEntry &entry : m_Scheduler.Entries())
        {
            if (entry.m_Service == record->m_Service)
            {
                schedules.Push(entry.ToJson());
            }
        }
        Json placement = Json::MakeObject();
        if (run->m_InferencePlan)
        {
            placement.Set("inference", Detail::Strings(run->m_InferencePlan->m_Nodes));
        }
        if (run->m_TrainingPlan)
        {
            placement.Set("training", Detail::Strings(run->m_TrainingPlan->m_Nodes));
        }
        Json replaced = Json::MakeArray();
        for (const Instance &instance : run->m_Previous)
        {
            replaced.Push(instance.m_Id);
        }
        Finish(id, "succeeded",
               Json::MakeObject({{"service", record->m_Service},
                                 {"version", deployment ? deployment->m_Version : std::string()},
                                 {"compliance", Agents::VerdictName(run->m_Report.m_Verdict)},
                                 {"amended", run->m_Report.m_Verdict == Agents::Verdict::Amended ||
                                                 (run->m_Report.m_Verdict == Agents::Verdict::Escalate &&
                                                  run->m_Report.m_Amended.has_value())},
                                 {"placement", std::move(placement)},
                                 {"instances", std::move(instances)},
                                 {"schedules", std::move(schedules)},
                                 {"replaced", std::move(replaced)}}));
    }

    // -------------------------
    // retrain: train → evaluate (lifecycle) → guard → load on every inference instance → record
    // -------------------------

    void ExecutionHarness::Impl::StartRetrain(const std::string &id)
    {
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        Deployment *deployment = m_State.FindDeployment(service);
        if (RoleInstances(deployment, "training").empty())
        {
            Finish(id, "rejected", Json(), "no training instance of '" + service + "' is deployed");
            return;
        }
        uint32_t version = deployment->m_Lifecycle.NextVersion();
        if (!deployment->m_Artifacts.empty())
        {
            version = std::max(version, deployment->m_Artifacts.rbegin()->first + 1);
        }
        deployment->m_Lifecycle.m_Training = true;
        deployment->m_Lifecycle.m_LastRetrain = NowMillis();
        deployment->m_Lifecycle.m_HasTraining = true;
        m_State.PutDeployment(*deployment);
        Publish();
        Begin(id, "train");
        Json body = Json::MakeObject({{"workflowId", id},
                                      {"version", version},
                                      {"reason", record->m_Input.GetString("reason")},
                                      {"trigger", record->m_Trigger}});
        Impl *self = this;
        m_Bus->Call(
            m_Options.m_Topics.Workload(service, "train"),
            Fabric::Envelope::Make("train", std::move(body), m_Options.m_Identity),
            [self, id, service, version](eMQ::Reason reason, std::optional<Fabric::Envelope> reply)
            {
                if (!self->Find(id))
                {
                    return;
                }
                self->Guarded(
                    id,
                    [&]
                    {
                        if (reason || !reply)
                        {
                            std::string text = reason ? Detail::ReasonText(reason) : "empty training response";
                            self->End(id, "failed", Json::MakeObject({{"reason", reason.m_Code}, {"error", text}}));
                            self->Finish(id, "failed", Json(), "training workload: " + text);
                            return;
                        }
                        const Json &body = reply->m_Body;
                        Workload::ModelArtifact artifact = Workload::ModelArtifact::FromJson(
                            body.IsObject() && body.Has("artifact") ? body["artifact"] : body);
                        if (artifact.m_Service.empty())
                        {
                            artifact.m_Service = service;
                        }
                        if (artifact.m_Version == 0)
                        {
                            artifact.m_Version = version;
                        }
                        if (artifact.m_Id.empty())
                        {
                            artifact.m_Id = service + "-v" + std::to_string(artifact.m_Version);
                        }
                        Deployment *deployment = self->m_State.FindDeployment(service);
                        if (!deployment)
                        {
                            throw std::runtime_error("service '" + service + "' was retired during training");
                        }
                        if (deployment->m_Artifacts.count(artifact.m_Version))
                        {
                            throw std::runtime_error("candidate v" + std::to_string(artifact.m_Version) +
                                                     " collides with a recorded artefact");
                        }
                        uint64_t now = NowMillis();
                        deployment->m_Lifecycle.m_History.push_back(Agents::ModelRecord{
                            artifact.m_Version, artifact.m_Id, artifact.m_Accuracy, now, "candidate"});
                        std::sort(deployment->m_Lifecycle.m_History.begin(), deployment->m_Lifecycle.m_History.end(),
                                  [](const Agents::ModelRecord &a, const Agents::ModelRecord &b)
                                  { return a.m_Version < b.m_Version; });
                        deployment->m_Artifacts[artifact.m_Version] = artifact;
                        self->m_State.PutDeployment(*deployment);
                        self->End(id, "succeeded",
                                  Json::MakeObject({{"version", artifact.m_Version},
                                                    {"accuracy", artifact.m_Accuracy},
                                                    {"samples", artifact.m_Samples},
                                                    {"dataset", artifact.m_Dataset},
                                                    {"source", reply->m_Source}}));
                        self->Find(id)->m_Artifact = std::move(artifact);
                        self->EvaluateCandidate(id);
                    });
            },
            m_Options.m_AgentTimeoutMs, m_Owner);
    }

    void ExecutionHarness::Impl::EvaluateCandidate(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        const Deployment *deployment = m_State.FindDeployment(service);
        Agents::LifecycleState state = deployment->m_Lifecycle;
        // the candidate's own record is not part of the state it is judged against
        std::erase_if(state.m_History,
                      [&](const Agents::ModelRecord &item) { return item.m_Version == run->m_Artifact->m_Version; });
        Json evaluation = Json::MakeObject();
        if (const Json *claimed = record->m_Input.Find("observed"); claimed && claimed->IsNumber())
        {
            evaluation.Set("claimed", *claimed); // provenance only: a requester's claim never sets the baseline
        }
        // judge against the accuracy the active model shows now (seen by the harness itself on the fabric), not
        // its training-time accuracy
        run->m_Observed = ObservedAccuracy(service);
        if (run->m_Observed)
        {
            for (Agents::ModelRecord &item : state.m_History)
            {
                if (state.m_Active && item.m_Version == *state.m_Active && *run->m_Observed < item.m_Accuracy)
                {
                    evaluation.Set("observed", *run->m_Observed);
                    evaluation.Set("recorded", item.m_Accuracy);
                    item.m_Accuracy = *run->m_Observed;
                }
            }
        }
        Begin(id, "evaluate");
        Json body = Json::MakeObject({{"state", state.ToJson()},
                                      {"policy", Detail::PolicyJson(deployment->m_Descriptor.m_Lifecycle)},
                                      {"candidate", run->m_Artifact->ToJson()},
                                      {"now", NowMillis()}});
        Ask(id, "lifecycle", "evaluate", std::move(body),
            [this, id, service, evaluation](const Json &result)
            {
                std::shared_ptr<Run> run = Find(id);
                run->m_Decision = Agents::LifecycleDecision::FromJson(result);
                const Agents::LifecycleDecision &decision = run->m_Decision;
                Deployment *deployment = m_State.FindDeployment(service);
                if (!deployment)
                {
                    throw std::runtime_error("service '" + service + "' was retired during evaluation");
                }
                const Workload::ModelArtifact artifact = *run->m_Artifact;
                Json detail = evaluation;
                detail.Set("decision", decision.ToJson());
                auto reject = [this, id, service, &artifact](const std::string &status, std::string error, Json result)
                {
                    if (Deployment *current = m_State.FindDeployment(service))
                    {
                        if (Agents::ModelRecord *item = History(*current, artifact.m_Version))
                        {
                            item->m_Status = "rejected";
                            item->m_At = NowMillis();
                        }
                        Prune(*current);
                        m_State.PutDeployment(*current);
                    }
                    Finish(id, status, std::move(result), std::move(error));
                };
                if (decision.m_Action != Agents::LifecycleAction::Promote)
                {
                    End(id, "succeeded", std::move(detail));
                    reject("rejected",
                           "candidate v" + std::to_string(artifact.m_Version) + " not promoted: " + decision.m_Reason,
                           Json::MakeObject({{"decision", decision.ToJson()},
                                             {"candidate", artifact.m_Version},
                                             {"accuracy", artifact.m_Accuracy}}));
                    return;
                }
                GuardResult check = m_Guard.CheckPromotion(*deployment, artifact, decision, run->m_Observed);
                detail.Set("guard", check.ToJson());
                if (!check.m_Ok)
                {
                    End(id, "failed", std::move(detail));
                    reject("rejected", "promotion guard: " + Join(check.m_Violations, "; "),
                           Json::MakeObject({{"decision", decision.ToJson()}, {"guard", check.ToJson()}}));
                    return;
                }
                End(id, "succeeded", std::move(detail));
                LoadModel(id, artifact,
                          [this, id, service, artifact](size_t loaded, size_t expected)
                          {
                              Deployment *deployment = m_State.FindDeployment(service);
                              if (!deployment)
                              {
                                  throw std::runtime_error("service '" + service + "' was retired during the load");
                              }
                              if (expected > 0 && loaded == 0)
                              {
                                  if (Agents::ModelRecord *item = History(*deployment, artifact.m_Version))
                                  {
                                      item->m_Status = "rejected";
                                      item->m_At = NowMillis();
                                  }
                                  m_State.PutDeployment(*deployment);
                                  Finish(id, "failed", Json(),
                                         "no inference instance loaded candidate v" +
                                             std::to_string(artifact.m_Version));
                                  return;
                              }
                              uint64_t now = NowMillis();
                              std::optional<uint32_t> previous = deployment->m_Lifecycle.m_Active;
                              if (previous)
                              {
                                  RecordObserved(*deployment, *previous, Find(id)->m_Observed);
                              }
                              for (Agents::ModelRecord &item : deployment->m_Lifecycle.m_History)
                              {
                                  if (item.m_Status == "active")
                                  {
                                      item.m_Status = "retired";
                                      item.m_At = now;
                                  }
                              }
                              if (Agents::ModelRecord *item = History(*deployment, artifact.m_Version))
                              {
                                  item->m_Status = "active";
                                  item->m_At = now;
                              }
                              deployment->m_Lifecycle.m_Active = artifact.m_Version;
                              deployment->m_Lifecycle.m_LastPromotion = now;
                              deployment->m_Lifecycle.m_Training = false;
                              Prune(*deployment);
                              m_State.PutDeployment(*deployment);
                              m_Observed.erase(service);
                              Finish(id, "succeeded",
                                     Json::MakeObject({{"decision", Find(id)->m_Decision.ToJson()},
                                                       {"version", artifact.m_Version},
                                                       {"accuracy", artifact.m_Accuracy},
                                                       {"previous", previous ? Json(*previous) : Json()},
                                                       {"loaded", loaded},
                                                       {"expected", expected}}));
                          });
            });
    }

    void ExecutionHarness::Impl::LoadModel(const std::string &id, const Workload::ModelArtifact &artifact,
                                           std::function<void(size_t loaded, size_t expected)> next)
    {
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        size_t expected = RoleInstances(m_State.FindDeployment(service), "inference").size();
        if (expected == 0)
        {
            next(0, 0);
            return;
        }
        Begin(id, "load");
        uint32_t version = artifact.m_Version;
        Json body = Json::MakeObject({{"action", "load"}, {"artifact", artifact.ToJson()}, {"workflowId", id}});
        Impl *self = this;
        m_Bus->Gather(
            m_Options.m_Topics.Workload(service, "model"),
            Fabric::Envelope::Make("model", std::move(body), m_Options.m_Identity),
            static_cast<uint8_t>(std::min<size_t>(expected, 255)),
            [self, id, expected, version, next = std::move(next)](eMQ::Reason reason,
                                                                  std::vector<Fabric::Envelope> replies)
            {
                if (!self->Find(id))
                {
                    return;
                }
                self->Guarded(id,
                              [&]
                              {
                                  size_t loaded = 0;
                                  Json sources = Json::MakeArray();
                                  Json refused = Json::MakeArray();
                                  for (const Fabric::Envelope &reply : replies)
                                  {
                                      // a reply naming another version or carrying an error did not load it
                                      const Json &body = reply.m_Body;
                                      bool error = reply.m_Kind == "error" || (body.IsObject() && body.Has("error"));
                                      bool other = body.IsObject() && body.Has("loaded") &&
                                                   body.GetInt("loaded", -1) != static_cast<int64_t>(version);
                                      if (!error && !other)
                                      {
                                          ++loaded;
                                          sources.Push(reply.m_Source);
                                      }
                                      else
                                      {
                                          refused.Push(Json::MakeObject(
                                              {{"source", reply.m_Source}, {"text", body.GetString("text")}}));
                                      }
                                  }
                                  Json detail = Json::MakeObject({{"version", version},
                                                                  {"loaded", loaded},
                                                                  {"expected", expected},
                                                                  {"sources", std::move(sources)},
                                                                  {"refused", std::move(refused)},
                                                                  {"reason", reason.m_Code},
                                                                  {"text", reason.m_Text}});
                                  self->End(id, loaded > 0 ? "succeeded" : "failed", std::move(detail));
                                  next(loaded, expected);
                              });
            },
            m_Options.m_AgentTimeoutMs, m_Owner);
    }

    // -------------------------
    // rollback: guard → load the target artefact → record
    // -------------------------

    void ExecutionHarness::Impl::StartRollback(const std::string &id)
    {
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        const Deployment *deployment = m_State.FindDeployment(service);
        Agents::LifecycleDecision decision;
        decision.m_Action = Agents::LifecycleAction::Rollback;
        decision.m_Reason = record->m_Input.GetString("reason", "rollback requested by " + record->m_Trigger);
        if (const Json *target = record->m_Input.Find("target"); target && target->IsInteger())
        {
            decision.m_Target = static_cast<uint32_t>(target->AsInt());
        }
        else if (const Agents::ModelRecord *previous = deployment->m_Lifecycle.PreviousRecord())
        {
            decision.m_Target = previous->m_Version;
        }
        Begin(id, "guard");
        GuardResult check = m_Guard.CheckRollback(*deployment, decision);
        End(id, check.m_Ok ? "succeeded" : "failed",
            Json::MakeObject({{"decision", decision.ToJson()}, {"guard", check.ToJson()}}));
        if (!check.m_Ok)
        {
            Finish(id, "rejected", Json::MakeObject({{"guard", check.ToJson()}}), Join(check.m_Violations, "; "));
            return;
        }
        uint32_t target = *decision.m_Target;
        Find(id)->m_Decision = decision;
        Find(id)->m_Observed = ObservedAccuracy(service);
        const Workload::ModelArtifact artifact = deployment->m_Artifacts.at(target);
        LoadModel(id, artifact,
                  [this, id, service, target](size_t loaded, size_t expected)
                  {
                      Deployment *deployment = m_State.FindDeployment(service);
                      if (!deployment)
                      {
                          throw std::runtime_error("service '" + service + "' was retired during the rollback");
                      }
                      if (expected > 0 && loaded == 0)
                      {
                          Finish(id, "failed", Json(),
                                 "no inference instance loaded v" + std::to_string(target) + " for the rollback");
                          return;
                      }
                      uint64_t now = NowMillis();
                      std::optional<uint32_t> previous = deployment->m_Lifecycle.m_Active;
                      if (previous)
                      {
                          RecordObserved(*deployment, *previous, Find(id)->m_Observed);
                      }
                      for (Agents::ModelRecord &item : deployment->m_Lifecycle.m_History)
                      {
                          if (item.m_Status == "active")
                          {
                              item.m_Status = "rolledback";
                              item.m_At = now;
                          }
                      }
                      if (Agents::ModelRecord *item = History(*deployment, target))
                      {
                          item->m_Status = "active";
                          item->m_At = now;
                      }
                      deployment->m_Lifecycle.m_Active = target;
                      Prune(*deployment);
                      m_State.PutDeployment(*deployment);
                      m_Observed.erase(service);
                      Finish(id, "succeeded",
                             Json::MakeObject({{"decision", Find(id)->m_Decision.ToJson()},
                                               {"version", target},
                                               {"previous", previous ? Json(*previous) : Json()},
                                               {"loaded", loaded},
                                               {"expected", expected}}));
                  });
    }

    // -------------------------
    // relocate: per role re-place with the current nodes → deploy new → [ready] → retire old (profile
    // re-registration and session migration happen as the new workload connects, receives the active model and
    // registers the same profiles; the old one is retired only then and disconnects)
    // -------------------------

    void ExecutionHarness::Impl::StartRelocate(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        const Deployment *deployment = m_State.FindDeployment(record->m_Service);
        run->m_Candidate = deployment->m_Descriptor;
        run->m_Replace = Detail::ReadStrings(record->m_Input, "replace");
        std::string role = record->m_Input.GetString("role");
        for (const char *name : {"inference", "training"})
        {
            bool replaced = false;
            for (const Instance *instance : RoleInstances(deployment, name))
            {
                replaced = replaced || Detail::Contains(run->m_Replace, instance->m_Id);
            }
            // replacements name their roles; otherwise every role (or the requested one) is examined
            bool wanted = run->m_Replace.empty() ? (role.empty() || role == name) : replaced;
            if (wanted && !RoleInstances(deployment, name).empty())
            {
                run->m_Roles.push_back(name);
            }
        }
        if (run->m_Roles.empty())
        {
            Finish(id, "rejected", Json(),
                   "'" + record->m_Service + "' has no " + (run->m_Replace.empty() ? "instance" : "named instance") +
                       " to relocate");
            return;
        }
        RelocateNext(id);
    }

    void ExecutionHarness::Impl::RelocateNext(const std::string &id)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        if (run->m_Roles.empty())
        {
            Finish(id, "succeeded",
                   Json::MakeObject(
                       {{"service", service}, {"relocated", run->m_Moves.Size() > 0}, {"moves", run->m_Moves}}));
            return;
        }
        const std::string roleName = run->m_Roles.front();
        run->m_Roles.erase(run->m_Roles.begin());
        Agents::Role role = Agents::ParseRole(roleName);
        const Deployment *deployment = m_State.FindDeployment(service);
        std::vector<std::string> current, previous;
        bool replace = false;
        for (const Instance *instance : RoleInstances(deployment, roleName))
        {
            current.push_back(instance->m_Node);
            previous.push_back(instance->m_Id);
            replace = replace || Detail::Contains(run->m_Replace, instance->m_Id);
        }
        Topology::Inventory base = BaseInventory();
        Topology::Inventory guarded = m_State.ApplyReservationsExcept(base, previous);
        Topology::NetworkModel network = Network();
        Begin(id, "place-" + roleName);
        Json body = Json::MakeObject({{"descriptor", Descriptor::ToJson(deployment->m_Descriptor)},
                                      {"role", roleName},
                                      {"inventory", m_State.ApplyReservations(base).ToJson()},
                                      {"network", network.ToJson()},
                                      {"current", Detail::Strings(current)}});
        Ask(id, "placement", "place", std::move(body),
            [this, id, service, role, roleName, current, previous, guarded, network, replace](const Json &result)
            {
                std::shared_ptr<Run> run = Find(id);
                Agents::PlacementPlan plan = Agents::PlacementPlan::FromJson(result);
                GuardResult check = m_Guard.CheckPlacement(*run->m_Candidate, role, plan, guarded, network);
                Json detail = PlanSummary(plan);
                detail.Set("guard", check.ToJson());
                detail.Set("current", Detail::Strings(current));
                detail.Set("replace", replace);
                if (!check.m_Ok)
                {
                    End(id, "failed", std::move(detail));
                    Finish(id, "rejected", Json::MakeObject({{"moves", run->m_Moves}}),
                           roleName + " placement: " + Join(check.m_Violations, "; "));
                    return;
                }
                std::vector<std::string> sortedPlan = plan.m_Nodes, sortedCurrent = current;
                std::sort(sortedPlan.begin(), sortedPlan.end());
                std::sort(sortedCurrent.begin(), sortedCurrent.end());
                // lost instances are redeployed even where the placement keeps their nodes
                if (!replace && (!plan.m_Relocate || sortedPlan == sortedCurrent))
                {
                    End(id, "succeeded", std::move(detail));
                    RelocateNext(id);
                    return;
                }
                End(id, "succeeded", std::move(detail));
                std::set<std::string> taken = TakenInstances();
                std::vector<Substrate::DeploymentSpec> specs = Specs(*run->m_Candidate, role, plan, taken);
                const Descriptor::Resources &resources = RoleResources(*run->m_Candidate, role);
                for (const Substrate::DeploymentSpec &spec : specs)
                {
                    m_State.Reserve(spec.m_Instance, spec.m_Node, resources);
                }
                auto release = [this, specs]
                {
                    for (const Substrate::DeploymentSpec &spec : specs)
                    {
                        m_State.Release(spec.m_Instance);
                    }
                };
                Begin(id, "enact-" + roleName);
                std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
                try
                {
                    Enact(
                        id, [substrate, specs] { return DeployAll(*substrate, specs); },
                        [this, id, service, roleName, current, previous, specs, release, substrate](const Json &outcome)
                        {
                            if (!outcome.GetBool("ok"))
                            {
                                release();
                                End(id, "failed", outcome);
                                Finish(id, "failed", Json::MakeObject({{"moves", Find(id)->m_Moves}}),
                                       "substrate deploy failed: " + outcome.GetString("error"));
                                return;
                            }
                            End(id, "succeeded", outcome);
                            Deployment *deployment = m_State.FindDeployment(service);
                            if (!deployment)
                            {
                                throw std::runtime_error("service '" + service + "' was retired during relocation");
                            }
                            std::vector<Instance> replaced;
                            int64_t collected = 0;
                            for (const Instance &instance : deployment->m_Instances)
                            {
                                if (instance.m_Role == roleName)
                                {
                                    replaced.push_back(instance);
                                    if (auto beat = m_Beats.find(instance.m_Id); beat != m_Beats.end())
                                    {
                                        collected = std::max(collected, beat->second.m_Samples);
                                    }
                                }
                            }
                            std::erase_if(deployment->m_Instances,
                                          [&](const Instance &instance) { return instance.m_Role == roleName; });
                            Json to = Json::MakeArray();
                            std::shared_ptr<Run> run = Find(id);
                            for (const Substrate::DeploymentSpec &spec : specs)
                            {
                                deployment->m_Instances.push_back(
                                    Instance{spec.m_Instance, spec.m_Role, spec.m_Node, "deploying", 0});
                                to.Push(spec.m_Node);
                                if (roleName == "training" && run->m_Candidate->m_Training)
                                {
                                    // the new trainer must hold what the old one collected (up to minSamples)
                                    run->m_ReadySamples[spec.m_Instance] =
                                        std::min<int64_t>(collected, run->m_Candidate->m_Training->m_MinSamples);
                                }
                            }
                            m_State.PutDeployment(*deployment);
                            Publish();
                            std::vector<std::string> fresh = SpecInstances(specs);
                            Json move = Json::MakeObject({{"role", roleName},
                                                          {"from", Detail::Strings(current)},
                                                          {"to", std::move(to)},
                                                          {"instances", Detail::Strings(fresh)},
                                                          {"retired", Detail::Strings(previous)}});
                            WaitFor(
                                id, fresh,
                                [this, id, previous, move, substrate, roleName]
                                {
                                    Find(id)->m_Moves.Push(move);
                                    Begin(id, "retire-" + roleName);
                                    Enact(
                                        id, [substrate, previous] { return RetireAll(*substrate, previous); },
                                        [this, id, previous](const Json &retired)
                                        {
                                            for (const std::string &instance : previous)
                                            {
                                                m_State.Release(instance);
                                                m_Beats.erase(instance);
                                            }
                                            End(id, retired.GetBool("ok") ? "succeeded" : "failed", retired);
                                            Publish();
                                            RelocateNext(id);
                                        });
                                },
                                [this, id, service, roleName, replaced, fresh, substrate](const Json &missing)
                                {
                                    // the relocated instances never became ready: the previous ones stay
                                    if (Deployment *deployment = m_State.FindDeployment(service))
                                    {
                                        std::erase_if(deployment->m_Instances, [&](const Instance &instance)
                                                      { return Detail::Contains(fresh, instance.m_Id); });
                                        for (const Instance &instance : replaced)
                                        {
                                            deployment->m_Instances.push_back(instance);
                                        }
                                        m_State.PutDeployment(*deployment);
                                    }
                                    Publish();
                                    Begin(id, "retire-new");
                                    Enact(
                                        id, [substrate, fresh] { return RetireAll(*substrate, fresh); },
                                        [this, id, fresh, missing, roleName](const Json &outcome)
                                        {
                                            for (const std::string &instance : fresh)
                                            {
                                                m_State.Release(instance);
                                            }
                                            End(id, outcome.GetBool("ok") ? "succeeded" : "failed", outcome);
                                            Finish(
                                                id, "failed",
                                                Json::MakeObject({{"moves", Find(id)->m_Moves}, {"missing", missing}}),
                                                "relocated " + roleName + " instances did not report ready within " +
                                                    std::to_string(m_Options.m_ReadyTimeoutMs) +
                                                    " ms; the previous instances keep running");
                                        });
                                });
                        });
                }
                catch (...)
                {
                    release();
                    throw;
                }
            });
    }

    // -------------------------
    // retire: [HITL when escalated by an agent] → substrate Retire all → release reservations → remove schedules
    // -------------------------

    void ExecutionHarness::Impl::StartRetire(const std::string &id)
    {
        WorkflowRecord *record = Record(id);
        if (!record->m_Input.GetBool("escalate"))
        {
            RetireService(id);
            return;
        }
        // an agent's proposal to take a service down (e.g. repeated policy violations) needs a human decision
        const std::string service = record->m_Service;
        std::string reason = record->m_Input.GetString("reason", "retirement proposed by " + record->m_Trigger);
        Json findings = Json::MakeArray();
        findings.Push(Json::MakeObject(
            {{"criterion", record->m_Trigger}, {"severity", "warning"}, {"message", reason}, {"service", service}}));
        Json body = Json::MakeObject({{"service", service},
                                      {"question", "Retire '" + service + "' (" + reason +
                                                       ")? Approve to stop all of its "
                                                       "instances."},
                                      {"findings", std::move(findings)},
                                      {"resolutions", Json::MakeArray()},
                                      {"proposal", record->m_Input}});
        AskHuman(id, std::move(body),
                 [this, id](const Detail::HitlAnswer &answer)
                 {
                     if (!answer.m_Approve)
                     {
                         End(id, "failed", answer.ToJson());
                         Finish(id, "rejected", Json::MakeObject({{"hitl", answer.ToJson()}}),
                                "retirement rejected " + answer.Who());
                         return;
                     }
                     End(id, "succeeded", answer.ToJson());
                     RetireService(id);
                 });
    }

    void ExecutionHarness::Impl::RetireService(const std::string &id)
    {
        WorkflowRecord *record = Record(id);
        const std::string service = record->m_Service;
        Deployment *deployment = m_State.FindDeployment(service);
        if (!deployment)
        {
            Finish(id, "rejected", Json(), "service '" + service + "' is no longer deployed");
            return;
        }
        std::vector<std::string> instances;
        for (const Instance &instance : deployment->m_Instances)
        {
            instances.push_back(instance.m_Id);
        }
        deployment->m_Status = "retiring";
        m_State.PutDeployment(*deployment);
        Publish();
        Begin(id, "retire");
        std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
        Enact(
            id, [substrate, instances] { return RetireAll(*substrate, instances); },
            [this, id, service, instances](const Json &outcome)
            {
                for (const std::string &instance : instances)
                {
                    m_State.Release(instance);
                    m_Beats.erase(instance);
                }
                size_t schedules = m_Scheduler.RemoveService(service);
                m_State.EraseDeployment(service);
                m_LastSync.erase(service);
                m_Observed.erase(service);
                Json detail = outcome;
                detail.Set("schedulesRemoved", schedules);
                End(id, "succeeded", std::move(detail));
                Finish(id, "succeeded",
                       Json::MakeObject(
                           {{"service", service}, {"retired", Detail::Strings(instances)}, {"substrate", outcome}}));
            });
    }
} // namespace eOps::Harness
