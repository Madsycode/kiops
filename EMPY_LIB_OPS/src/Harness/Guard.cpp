#include <eOps/Harness/Guard.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/TopicAlgebra.h>
#include "Detail.h"
#include <set>
#include <cmath>

namespace eOps::Harness
{
    namespace
    {
        const char *const WORKFLOW_KINDS[] = {"deploy", "retrain", "rollback", "relocate", "retire"};

        std::string Number(double value)
        {
            std::string text = Json(std::isfinite(value) ? value : 0.0).Dump();
            return text;
        }

        // workflow kinds that must not run concurrently with `kind` for the same service
        std::vector<std::string> Conflicting(std::string_view kind)
        {
            if (kind == "retire")
            {
                return {""}; // any running workflow
            }
            if (kind == "retrain" || kind == "rollback")
            {
                return {"retrain", "rollback", "retire"};
            }
            if (kind == "relocate")
            {
                return {"relocate", "deploy", "retire"};
            }
            return {"deploy", "relocate", "retire"}; // deploy
        }

        // the data region of a service, re-derived the way the Placement agent defines it: the region with the
        // densest observable telemetry, else with the densest actionable consumers ("" when nothing is known)
        std::string DataRegion(const Descriptor::RichMLService &service, const Topology::Inventory &inventory,
                               const Topology::NetworkModel &network)
        {
            std::vector<std::string> regions = network.m_Regions;
            for (const Topology::Node &node : inventory.m_Nodes)
            {
                if (!node.m_Region.empty() && !Detail::Contains(regions, node.m_Region))
                {
                    regions.push_back(node.m_Region);
                }
            }
            auto densest = [&](const auto &density)
            {
                std::string best;
                double highest = 0;
                for (const std::string &region : regions)
                {
                    double value = density(region);
                    if (value > highest)
                    {
                        highest = value;
                        best = region;
                    }
                }
                return best;
            };
            std::string region = densest(
                [&](const std::string &candidate)
                {
                    double total = 0;
                    for (const Descriptor::Observable &observable : service.m_Observables)
                    {
                        total += network.Density(observable.m_Name, candidate);
                    }
                    return total;
                });
            if (region.empty())
            {
                region = densest(
                    [&](const std::string &candidate)
                    {
                        double total = 0;
                        for (const Descriptor::Actionable &actionable : service.m_Actionables)
                        {
                            total += network.Consumers(actionable.m_Name, candidate);
                        }
                        return total;
                    });
            }
            return region;
        }

        const Agents::ModelRecord *HistoryRecord(const Deployment &deployment, uint32_t version)
        {
            for (const Agents::ModelRecord &record : deployment.m_Lifecycle.m_History)
            {
                if (record.m_Version == version)
                {
                    return &record;
                }
            }
            return nullptr;
        }

        GuardResult Placement(const Descriptor::RichMLService &service, Agents::Role role,
                              const Agents::PlacementPlan &plan, const Topology::Inventory &inventory,
                              const Topology::NetworkModel *network)
        {
            GuardResult result;
            const std::string name = Agents::RoleName(role);
            if (!plan.m_Service.empty() && plan.m_Service != service.m_Metadata.m_Name)
            {
                result.Fail("placement plan is for service '" + plan.m_Service + "', not '" +
                            service.m_Metadata.m_Name + "'");
            }
            if (plan.m_Role != role)
            {
                result.Fail(std::string("placement plan is for role '") + Agents::RoleName(plan.m_Role) + "', not '" +
                            name + "'");
            }
            const Descriptor::Resources *resources = nullptr;
            const std::vector<std::string> *regions = nullptr;
            size_t replicas = 1;
            if (role == Agents::Role::Inference && service.m_Inference)
            {
                resources = &service.m_Inference->m_Resources;
                regions = &service.m_Inference->m_Regions;
                replicas = service.m_Inference->m_Replicas;
            }
            else if (role == Agents::Role::Training && service.m_Training)
            {
                resources = &service.m_Training->m_Resources;
                regions = &service.m_Training->m_Regions;
            }
            if (!resources)
            {
                result.Fail("service '" + service.m_Metadata.m_Name + "' declares no " + name + " role");
                return result;
            }
            if (plan.m_Nodes.size() != replicas)
            {
                result.Fail(name + " placement names " + std::to_string(plan.m_Nodes.size()) + " node(s), " +
                            std::to_string(replicas) + " required");
            }
            // latency objective of the inference role: within maxMs/2 of the data region (never the agent's claim)
            std::string dataRegion;
            double maxMs = 0;
            if (network && role == Agents::Role::Inference && service.m_Inference->m_Latency &&
                service.m_Inference->m_Latency->m_MaxMs > 0)
            {
                maxMs = service.m_Inference->m_Latency->m_MaxMs;
                dataRegion = DataRegion(service, inventory, *network);
            }
            const Descriptor::PlacementConstraints &constraints = service.m_Constraints.m_Placement;
            Topology::Inventory working = inventory;
            std::set<std::string> seen;
            for (const std::string &id : plan.m_Nodes)
            {
                if (!seen.insert(id).second)
                {
                    result.Fail(name + " placement names node '" + id + "' twice");
                    continue;
                }
                Topology::Node *node = working.Find(id);
                if (!node)
                {
                    result.Fail(name + " placement names unknown node '" + id + "'");
                    continue;
                }
                if (!regions->empty() && !Detail::Contains(*regions, node->m_Region))
                {
                    result.Fail("node '" + id + "' is in region '" + node->m_Region + "', outside the declared " +
                                name + " regions");
                }
                if (Detail::Contains(constraints.m_Avoid, node->m_Region))
                {
                    result.Fail("node '" + id + "' is in the avoided region '" + node->m_Region + "'");
                }
                if (!constraints.m_Nodes.empty() && !Detail::Contains(constraints.m_Nodes, id))
                {
                    result.Fail("node '" + id + "' is not among the pinned nodes");
                }
                if (!dataRegion.empty())
                {
                    double latency = network->Latency(node->m_Region, dataRegion);
                    if (latency > maxMs / 2)
                    {
                        result.Fail("node '" + id + "' is " + Number(latency) + " ms from the data region '" +
                                    dataRegion + "', more than half the " + Number(maxMs) + " ms latency objective");
                    }
                }
                if (!node->Fits(*resources))
                {
                    std::string detail = "cpu " + Number(node->m_Cpu - node->m_UsedCpu) + " free of " +
                                         Number(resources->m_Cpu) + " required, memory " +
                                         std::to_string(node->m_Memory - std::min(node->m_Memory, node->m_UsedMemory)) +
                                         " free of " + std::to_string(resources->m_Memory);
                    if (!resources->m_Accelerator.empty())
                    {
                        detail += ", " + resources->m_Accelerator + " " +
                                  std::to_string(node->FreeAccelerators(resources->m_Accelerator)) + " free of " +
                                  std::to_string(std::max<uint32_t>(1, resources->m_AcceleratorCount));
                    }
                    result.Fail("node '" + id + "' cannot host the " + name + " role (" + detail + ")");
                    continue;
                }
                node->Reserve(*resources);
            }
            return result;
        }
    } // namespace

    void GuardResult::Fail(std::string violation)
    {
        m_Ok = false;
        m_Violations.push_back(std::move(violation));
    }

    Json GuardResult::ToJson() const
    {
        return Json::MakeObject({{"ok", m_Ok}, {"violations", Detail::Strings(m_Violations)}});
    }

    GuardResult Guard::CheckDescriptor(const Descriptor::RichMLService &service,
                                       const Descriptor::ValidationOptions &options) const
    {
        GuardResult result;
        for (const Descriptor::Issue &issue : Descriptor::Validate(service, options))
        {
            if (issue.m_Severity == Descriptor::Severity::Error)
            {
                result.Fail("descriptor " + issue.m_Path + ": " + issue.m_Message);
            }
        }
        return result;
    }

    GuardResult Guard::CheckCompliance(const Agents::ComplianceReport &report,
                                       const Descriptor::ValidationOptions &options) const
    {
        GuardResult result;
        if (report.m_Verdict != Agents::Verdict::Approved && report.m_Verdict != Agents::Verdict::Amended)
        {
            result.Fail(std::string("compliance verdict '") + Agents::VerdictName(report.m_Verdict) +
                        "' does not permit enactment");
        }
        // the verdict must follow from the findings themselves
        for (const Agents::Finding &finding : report.m_Findings)
        {
            if (finding.m_Severity == "error")
            {
                result.Fail("compliance error finding (" + finding.m_Criterion +
                            ") contradicts the verdict: " + finding.m_Message);
            }
        }
        if (report.m_Verdict == Agents::Verdict::Amended)
        {
            if (!report.m_Amended)
            {
                result.Fail("amended verdict without an amended descriptor");
            }
            else
            {
                for (const Descriptor::Issue &issue : Descriptor::Validate(*report.m_Amended, options))
                {
                    if (issue.m_Severity == Descriptor::Severity::Error)
                    {
                        result.Fail("amended descriptor " + issue.m_Path + ": " + issue.m_Message);
                    }
                }
            }
        }
        return result;
    }

    GuardResult Guard::CheckCompliance(const Descriptor::RichMLService &candidate,
                                       const Agents::ComplianceReport &report,
                                       const Descriptor::ValidationOptions &options) const
    {
        GuardResult result = CheckCompliance(report, options);
        if (report.m_Verdict == Agents::Verdict::Amended && report.m_Amended)
        {
            for (std::string &violation : CheckAmendment(candidate, *report.m_Amended).m_Violations)
            {
                result.Fail(std::move(violation));
            }
        }
        return result;
    }

    GuardResult Guard::CheckAmendment(const Descriptor::RichMLService &candidate,
                                      const Descriptor::RichMLService &amended) const
    {
        GuardResult result;
        const std::string &name = candidate.m_Metadata.m_Name;
        if (amended.m_Metadata.m_Name != name)
        {
            result.Fail("the amendment renames service '" + name + "' to '" + amended.m_Metadata.m_Name + "'");
        }
        if (amended.m_Metadata.m_Version != candidate.m_Metadata.m_Version)
        {
            result.Fail("the amendment changes the version of '" + name + "' from " + candidate.m_Metadata.m_Version +
                        " to " + amended.m_Metadata.m_Version);
        }
        // everything but the actionables and the actuation topic derived from them stays as declared
        Descriptor::RichMLService comparable = amended;
        comparable.m_Actionables = candidate.m_Actionables;
        if (comparable.m_Inference && candidate.m_Inference)
        {
            comparable.m_Inference->m_Actuation = candidate.m_Inference->m_Actuation;
        }
        if (result.m_Ok && !(Descriptor::ToJson(comparable) == Descriptor::ToJson(candidate)))
        {
            result.Fail("the amendment of '" + name +
                        "' changes more than its actionables (metadata, observables, inference, training, constraints "
                        "and lifecycle must stay as declared)");
        }
        for (const Descriptor::Actionable &actionable : amended.m_Actionables)
        {
            const Descriptor::Actionable *declared = nullptr;
            for (const Descriptor::Actionable &original : candidate.m_Actionables)
            {
                if (original.m_Name == actionable.m_Name ||
                    (IsFilter(original.m_Name) && FilterCovers(original.m_Name, actionable.m_Name)))
                {
                    declared = &original;
                    break;
                }
            }
            const std::string topic = "amended actionable '" + actionable.m_Name + "'";
            if (!declared)
            {
                result.Fail(topic + " is not covered by an actionable '" + name + "' declared");
                continue;
            }
            if (actionable.m_Unit != declared->m_Unit)
            {
                result.Fail(topic + " changes the unit '" + declared->m_Unit + "'");
            }
            if (declared->m_Min && (!actionable.m_Min || *actionable.m_Min < *declared->m_Min))
            {
                result.Fail(topic + " widens the declared minimum " + Number(*declared->m_Min));
            }
            if (declared->m_Max && (!actionable.m_Max || *actionable.m_Max > *declared->m_Max))
            {
                result.Fail(topic + " widens the declared maximum " + Number(*declared->m_Max));
            }
            if (declared->m_MaxRatePerMinute > 0 &&
                (actionable.m_MaxRatePerMinute == 0 || actionable.m_MaxRatePerMinute > declared->m_MaxRatePerMinute))
            {
                result.Fail(topic + " raises the declared rate limit of " +
                            std::to_string(declared->m_MaxRatePerMinute) + " actions per minute");
            }
            if (actionable.m_Qos > declared->m_Qos)
            {
                result.Fail(topic + " raises the declared QoS");
            }
        }
        return result;
    }

    GuardResult Guard::CheckPlacement(const Descriptor::RichMLService &service, Agents::Role role,
                                      const Agents::PlacementPlan &plan, const Topology::Inventory &inventory) const
    {
        return Placement(service, role, plan, inventory, nullptr);
    }

    GuardResult Guard::CheckPlacement(const Descriptor::RichMLService &service, Agents::Role role,
                                      const Agents::PlacementPlan &plan, const Topology::Inventory &inventory,
                                      const Topology::NetworkModel &network) const
    {
        return Placement(service, role, plan, inventory, &network);
    }

    GuardResult Guard::CheckWorkflowRequest(const Json &request, const StateStore &state) const
    {
        GuardResult result;
        if (!request.IsObject())
        {
            result.Fail("workflow request must be an object");
            return result;
        }
        const Json *kindValue = request.Find("kind");
        if (!kindValue || !kindValue->IsString())
        {
            result.Fail("workflow request needs a string 'kind'");
            return result;
        }
        const std::string &kind = kindValue->AsString();
        if (std::find(std::begin(WORKFLOW_KINDS), std::end(WORKFLOW_KINDS), kind) == std::end(WORKFLOW_KINDS))
        {
            result.Fail("unknown workflow kind '" + kind + "'");
            return result;
        }
        const Json *serviceValue = request.Find("service");
        if (serviceValue && !serviceValue->IsNull() && !serviceValue->IsString())
        {
            result.Fail("workflow 'service' must be a string");
            return result;
        }
        std::string service = serviceValue && serviceValue->IsString() ? serviceValue->AsString() : std::string();
        if (kind == "deploy")
        {
            const Json *intent = request.Find("intent");
            const Json *descriptor = request.Find("descriptor");
            bool hasIntent = intent && intent->IsString() && !Trim(intent->AsString()).empty();
            bool hasDescriptor = descriptor && ((descriptor->IsString() && !Trim(descriptor->AsString()).empty()) ||
                                                descriptor->IsObject());
            if (hasIntent == hasDescriptor)
            {
                result.Fail("a deploy workflow needs exactly one of a non-empty 'intent' or 'descriptor'");
            }
        }
        else
        {
            if (service.empty())
            {
                result.Fail("a " + kind + " workflow needs a 'service'");
                return result;
            }
            const Deployment *deployment = state.FindDeployment(service);
            if (!deployment)
            {
                result.Fail("service '" + service + "' is not deployed");
                return result;
            }
            if (kind == "retrain" && !deployment->m_Descriptor.m_Training)
            {
                result.Fail("service '" + service + "' declares no training");
            }
            if (const Json *observed = request.Find("observed"); observed && !observed->IsNull())
            {
                if (!observed->IsNumber() || !std::isfinite(observed->AsNumber()) || observed->AsNumber() < 0 ||
                    observed->AsNumber() > 1)
                {
                    result.Fail("workflow 'observed' must be an accuracy in [0, 1]");
                }
            }
            if (kind == "rollback")
            {
                const Json *target = request.Find("target");
                if (target && !target->IsNull() && (!target->IsInteger() || target->AsInt() <= 0))
                {
                    result.Fail("rollback 'target' must be a positive model version");
                }
            }
            if (kind == "relocate")
            {
                if (const Json *role = request.Find("role"); role && !role->IsNull())
                {
                    if (!role->IsString() || (role->AsString() != "inference" && role->AsString() != "training"))
                    {
                        result.Fail("relocate 'role' must be inference or training");
                    }
                }
                if (const Json *replace = request.Find("replace"); replace && !replace->IsNull())
                {
                    bool strings = replace->IsArray();
                    for (const Json &item : strings ? replace->AsArray() : Json::Array{})
                    {
                        strings = strings && item.IsString();
                    }
                    if (!strings)
                    {
                        result.Fail("relocate 'replace' must be an array of instance ids");
                    }
                }
            }
        }
        if (!service.empty())
        {
            for (const std::string &other : Conflicting(kind))
            {
                if (state.HasRunning(other, service))
                {
                    result.Fail("a " + (other.empty() ? std::string("workflow") : other + " workflow") +
                                " is already running for '" + service + "'");
                    break;
                }
            }
        }
        return result;
    }

    GuardResult Guard::CheckPromotion(const Deployment &deployment, const Workload::ModelArtifact &candidate,
                                      const Agents::LifecycleDecision &decision, std::optional<double> observed) const
    {
        GuardResult result;
        if (decision.m_Action != Agents::LifecycleAction::Promote)
        {
            result.Fail(std::string("lifecycle decision is '") + Agents::ActionName(decision.m_Action) +
                        "', not promote");
        }
        if (decision.m_Target && *decision.m_Target != candidate.m_Version)
        {
            result.Fail("decision targets v" + std::to_string(*decision.m_Target) + " but the candidate is v" +
                        std::to_string(candidate.m_Version));
        }
        // the artefact contract shared with the inference runtimes (they load what the Guard approves)
        if (std::string problem = Workload::CheckArtifact(candidate, deployment.m_Descriptor); !problem.empty())
        {
            result.Fail("candidate: " + problem);
        }
        if (deployment.m_Lifecycle.m_Active && candidate.m_Version <= *deployment.m_Lifecycle.m_Active)
        {
            result.Fail("candidate v" + std::to_string(candidate.m_Version) + " is not newer than active v" +
                        std::to_string(*deployment.m_Lifecycle.m_Active));
        }
        const Descriptor::LifecyclePolicy &policy = deployment.m_Descriptor.m_Lifecycle;
        if (!std::isfinite(candidate.m_Accuracy) || candidate.m_Accuracy < policy.m_AccuracyThreshold)
        {
            result.Fail("candidate accuracy " + Number(candidate.m_Accuracy) + " is below the threshold " +
                        Number(policy.m_AccuracyThreshold));
        }
        // improvement over the active model: its recorded accuracy, or the lower accuracy the harness observed
        const Agents::ModelRecord *active = deployment.m_Lifecycle.ActiveRecord();
        if (active && std::isfinite(candidate.m_Accuracy))
        {
            double baseline = active->m_Accuracy;
            if (observed && std::isfinite(*observed))
            {
                baseline = std::min(baseline, *observed);
            }
            double improvement = candidate.m_Accuracy - baseline;
            if (improvement < policy.m_MinImprovement)
            {
                result.Fail("candidate improves accuracy by " + Number(improvement) + " over active v" +
                            std::to_string(active->m_Version) + " (" + Number(baseline) + "), minimum " +
                            Number(policy.m_MinImprovement));
            }
        }
        return result;
    }

    GuardResult Guard::CheckRollback(const Deployment &deployment, const Agents::LifecycleDecision &decision) const
    {
        GuardResult result;
        if (decision.m_Action != Agents::LifecycleAction::Rollback)
        {
            result.Fail(std::string("lifecycle decision is '") + Agents::ActionName(decision.m_Action) +
                        "', not rollback");
        }
        if (!decision.m_Target)
        {
            result.Fail("rollback without a target version");
            return result;
        }
        const uint32_t target = *decision.m_Target;
        const std::string name = "v" + std::to_string(target);
        auto artifact = deployment.m_Artifacts.find(target);
        if (artifact == deployment.m_Artifacts.end())
        {
            result.Fail("no artefact recorded for " + name);
            return result;
        }
        if (deployment.m_Lifecycle.m_Active && *deployment.m_Lifecycle.m_Active == target)
        {
            result.Fail(name + " is already active");
        }
        if (std::string problem = Workload::CheckArtifact(artifact->second, deployment.m_Descriptor); !problem.empty())
        {
            result.Fail(name + ": " + problem);
        }
        // only a model that was promoted before, and whose last known accuracy still meets the threshold
        const Agents::ModelRecord *record = HistoryRecord(deployment, target);
        if (!record)
        {
            result.Fail(name + " has no lifecycle record: it was never promoted");
            return result;
        }
        if (record->m_Status != "retired" && record->m_Status != "rolledback")
        {
            result.Fail(name + " is '" + record->m_Status +
                        "': only a model that was promoted before (retired or rolled back) is a rollback target");
        }
        double threshold = deployment.m_Descriptor.m_Lifecycle.m_AccuracyThreshold;
        if (!std::isfinite(record->m_Accuracy) || record->m_Accuracy < threshold)
        {
            result.Fail(name + " accuracy " + Number(record->m_Accuracy) + " (last known) is below the threshold " +
                        Number(threshold));
        }
        return result;
    }
} // namespace eOps::Harness
