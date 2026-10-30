#include <eOps/Agents/Placement.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/YAML.h>
#include "Detail.h"
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>

namespace eOps::Agents
{
    namespace
    {
        // near-best window inside which the Planning advisor may reorder candidates
        constexpr double ADVISOR_WINDOW = 0.05;
        // latency normaliser when no latency objective is declared (ms)
        constexpr double DEFAULT_LATENCY_SCALE = 50;

        struct Component
        {
            const char *m_Name;
            double m_Weight;
            double m_Value;
        };

        double Clamp01(double value)
        {
            return std::isfinite(value) ? std::clamp(value, 0.0, 1.0) : 0.0;
        }

        // every region known to the inventory or the network model, in a stable order
        std::vector<std::string> KnownRegions(const Topology::Inventory &inventory,
                                              const Topology::NetworkModel &network)
        {
            std::vector<std::string> regions = network.m_Regions;
            for (const Topology::Node &node : inventory.m_Nodes)
            {
                Detail::AddUnique(regions, node.m_Region);
            }
            return regions;
        }

        double ObservableDensity(const Descriptor::RichMLService &service, const Topology::NetworkModel &network,
                                 const std::string &region)
        {
            double density = 0;
            for (const Descriptor::Observable &observable : service.m_Observables)
            {
                density += network.Density(observable.m_Name, region);
            }
            return density;
        }

        double ActionableConsumers(const Descriptor::RichMLService &service, const Topology::NetworkModel &network,
                                   const std::string &region)
        {
            double consumers = 0;
            for (const Descriptor::Actionable &actionable : service.m_Actionables)
            {
                consumers += network.Consumers(actionable.m_Name, region);
            }
            return consumers;
        }

        std::string Percent(double value)
        {
            return Detail::Text(value);
        }
    } // namespace

    // -------------------------
    // names and JSON forms
    // -------------------------

    const char *RoleName(Role role)
    {
        return role == Role::Training ? "training" : "inference";
    }

    Role ParseRole(std::string_view name)
    {
        if (name == "inference")
        {
            return Role::Inference;
        }
        if (name == "training")
        {
            return Role::Training;
        }
        throw std::invalid_argument("unknown role '" + std::string(name) + "' (expected inference or training)");
    }

    Json Candidate::ToJson() const
    {
        return Json::MakeObject({{"node", m_Node},
                                 {"region", m_Region},
                                 {"score", m_Score},
                                 {"feasible", m_Feasible},
                                 {"reasons", Detail::Strings(m_Reasons)},
                                 {"breakdown", m_Breakdown.IsNull() ? Json::MakeObject() : m_Breakdown}});
    }

    Candidate Candidate::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "placement candidate");
        Candidate candidate;
        candidate.m_Node = Detail::ReadString(json, "node");
        candidate.m_Region = Detail::ReadString(json, "region");
        candidate.m_Score = Detail::ReadNumber(json, "score");
        candidate.m_Feasible = Detail::ReadBool(json, "feasible");
        candidate.m_Reasons = Detail::ReadStrings(json, "reasons");
        if (const Json *breakdown = json.Find("breakdown"))
        {
            candidate.m_Breakdown = *breakdown;
        }
        if (candidate.m_Node.empty())
        {
            throw JsonError("placement candidate without a node");
        }
        return candidate;
    }

    Json PlacementPlan::ToJson() const
    {
        Json candidates = Json::MakeArray();
        for (const Candidate &candidate : m_Candidates)
        {
            candidates.Push(candidate.ToJson());
        }
        return Json::MakeObject({{"service", m_Service},
                                 {"role", RoleName(m_Role)},
                                 {"nodes", Detail::Strings(m_Nodes)},
                                 {"candidates", std::move(candidates)},
                                 {"rationale", m_Rationale},
                                 {"relocate", m_Relocate},
                                 {"currentScore", m_CurrentScore},
                                 {"advisor", m_Advisor}});
    }

    PlacementPlan PlacementPlan::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "placement plan");
        PlacementPlan plan;
        plan.m_Service = Detail::ReadString(json, "service");
        try
        {
            plan.m_Role = ParseRole(Detail::ReadString(json, "role", "inference"));
        }
        catch (const std::invalid_argument &error)
        {
            throw JsonError(error.what());
        }
        plan.m_Nodes = Detail::ReadStrings(json, "nodes");
        for (const Json &item : Detail::ReadArray(json, "candidates"))
        {
            plan.m_Candidates.push_back(Candidate::FromJson(item));
        }
        plan.m_Rationale = Detail::ReadString(json, "rationale");
        plan.m_Relocate = Detail::ReadBool(json, "relocate");
        plan.m_CurrentScore = Detail::ReadNumber(json, "currentScore");
        plan.m_Advisor = Detail::ReadString(json, "advisor");
        return plan;
    }

    PlacementWeights PlacementWeights::FromYaml(const eCore::YamlNode &node)
    {
        PlacementWeights weights;
        if (node.IsNull())
        {
            return weights;
        }
        if (!node.IsMap())
        {
            throw eCore::ConfigError(node.Path() + ": expected a mapping of placement weights");
        }
        std::pair<const char *, double *> fields[] = {
            {"locality", &weights.m_Locality}, {"subscribers", &weights.m_Subscribers}, {"shard", &weights.m_Shard},
            {"latency", &weights.m_Latency},   {"accelerator", &weights.m_Accelerator}, {"session", &weights.m_Session},
            {"capacity", &weights.m_Capacity}, {"preference", &weights.m_Preference}};
        for (const std::string &key : node.Keys())
        {
            auto field = std::find_if(std::begin(fields), std::end(fields),
                                      [&](const auto &entry) { return key == entry.first; });
            if (field == std::end(fields))
            {
                throw eCore::ConfigError(node.Path() + "." + key + ": unknown placement weight");
            }
            *field->second = node.Get<double>(key, *field->second, 0.0, 1000.0);
        }
        double total = weights.m_Locality + weights.m_Subscribers + weights.m_Shard + weights.m_Latency +
                       weights.m_Accelerator + weights.m_Session + weights.m_Capacity + weights.m_Preference;
        if (!(total > 0))
        {
            throw eCore::ConfigError(node.Path() + ": placement weights must not all be zero");
        }
        return weights;
    }

    Json PlacementWeights::ToJson() const
    {
        return Json::MakeObject({{"locality", m_Locality},
                                 {"subscribers", m_Subscribers},
                                 {"shard", m_Shard},
                                 {"latency", m_Latency},
                                 {"accelerator", m_Accelerator},
                                 {"session", m_Session},
                                 {"capacity", m_Capacity},
                                 {"preference", m_Preference}});
    }

    // -------------------------
    // Placement
    // -------------------------

    Placement::Placement(AgentEnvironment environment, PlacementWeights weights)
        : m_Environment(environment), m_Weights(weights)
    {
    }

    PlacementPlan Placement::Place(const Descriptor::RichMLService &service, Role role,
                                   const Topology::Inventory &inventory, const Topology::NetworkModel &network,
                                   const std::vector<std::string> &current) const
    {
        PlacementPlan plan;
        plan.m_Service = service.m_Metadata.m_Name;
        plan.m_Role = role;
        const char *roleName = RoleName(role);
        bool inference = role == Role::Inference;
        if ((inference && !service.m_Inference) || (!inference && !service.m_Training))
        {
            plan.m_Rationale = "service '" + plan.m_Service + "' declares no " + roleName;
            return plan;
        }
        const Descriptor::Resources &resources =
            inference ? service.m_Inference->m_Resources : service.m_Training->m_Resources;
        const std::vector<std::string> &regions =
            inference ? service.m_Inference->m_Regions : service.m_Training->m_Regions;
        const Descriptor::PlacementConstraints &constraints = service.m_Constraints.m_Placement;
        const std::optional<Descriptor::LatencyObjective> &objective =
            inference ? service.m_Inference->m_Latency : std::optional<Descriptor::LatencyObjective>();
        uint32_t replicas = inference ? std::max<uint32_t>(1, service.m_Inference->m_Replicas) : 1;
        uint32_t acceleratorCount =
            resources.m_Accelerator.empty() ? 0 : std::max<uint32_t>(1, resources.m_AcceleratorCount);

        // data region: densest observable telemetry, else densest actionable consumers
        std::vector<std::string> known = KnownRegions(inventory, network);
        std::string dataRegion;
        double densest = 0;
        double maxDensity = 0;
        double maxConsumers = 0;
        for (const std::string &region : known)
        {
            double density = ObservableDensity(service, network, region);
            maxDensity = std::max(maxDensity, density);
            maxConsumers = std::max(maxConsumers, ActionableConsumers(service, network, region));
            if (density > densest)
            {
                densest = density;
                dataRegion = region;
            }
        }
        if (dataRegion.empty())
        {
            for (const std::string &region : known)
            {
                double consumers = ActionableConsumers(service, network, region);
                if (consumers > densest)
                {
                    densest = consumers;
                    dataRegion = region;
                }
            }
        }
        std::vector<std::string> topics;
        for (const Descriptor::Observable &observable : service.m_Observables)
        {
            topics.push_back(observable.m_Name);
        }
        for (const Descriptor::Actionable &actionable : service.m_Actionables)
        {
            topics.push_back(actionable.m_Name);
        }

        // session locality: the node-local brokers the current instances' MMQP sessions live on
        std::vector<std::string> currentBrokers;
        for (const std::string &id : current)
        {
            if (const Topology::Node *node = inventory.Find(id); node && !node->m_Broker.empty())
            {
                Detail::AddUnique(currentBrokers, node->m_Broker);
            }
        }

        // role weighting: training discounts communication locality and emphasises accelerators
        double scale = inference ? 1.0 : 0.25;
        double acceleratorScale = inference ? 1.0 : 3.0;
        double weightLocality = m_Weights.m_Locality * scale;
        double weightSubscribers = m_Weights.m_Subscribers * scale;
        double weightShard = m_Weights.m_Shard * scale;
        double weightAccelerator = m_Weights.m_Accelerator * acceleratorScale;
        double weightTotal = weightLocality + weightSubscribers + weightShard + m_Weights.m_Latency +
                             weightAccelerator + m_Weights.m_Session + m_Weights.m_Capacity + m_Weights.m_Preference;
        if (!(weightTotal > 0))
        {
            weightTotal = 1;
        }

        for (const Topology::Node &original : inventory.m_Nodes)
        {
            Candidate candidate;
            candidate.m_Node = original.m_Id;
            candidate.m_Region = original.m_Region;
            Topology::Node node = original;
            bool hosting = Detail::Contains(current, node.m_Id);
            if (hosting)
            {
                node.Release(resources); // the service's own reservation
            }

            // hard filters
            if (!regions.empty() && !Detail::Contains(regions, node.m_Region))
            {
                candidate.m_Reasons.push_back("region '" + node.m_Region + "' is not a declared " + roleName +
                                              " region");
            }
            if (Detail::Contains(constraints.m_Avoid, node.m_Region))
            {
                candidate.m_Reasons.push_back("region '" + node.m_Region + "' is avoided");
            }
            if (!constraints.m_Nodes.empty() && !Detail::Contains(constraints.m_Nodes, node.m_Id))
            {
                candidate.m_Reasons.push_back("node is not among the pinned nodes");
            }
            double freeCpu = std::max(0.0, node.m_Cpu - node.m_UsedCpu);
            uint64_t freeMemory = node.m_Memory > node.m_UsedMemory ? node.m_Memory - node.m_UsedMemory : 0;
            if (freeCpu + 1e-9 < resources.m_Cpu)
            {
                candidate.m_Reasons.push_back("insufficient cpu (free " + Detail::Text(freeCpu) + " < required " +
                                              Detail::Text(resources.m_Cpu) + ")");
            }
            if (freeMemory < resources.m_Memory)
            {
                candidate.m_Reasons.push_back("insufficient memory (free " + Descriptor::FormatMemory(freeMemory) +
                                              " < required " + Descriptor::FormatMemory(resources.m_Memory) + ")");
            }
            uint32_t freeAccelerators = acceleratorCount ? node.FreeAccelerators(resources.m_Accelerator) : 0;
            if (acceleratorCount && freeAccelerators < acceleratorCount)
            {
                candidate.m_Reasons.push_back("accelerator " + resources.m_Accelerator + " unavailable (free " +
                                              std::to_string(freeAccelerators) + " < required " +
                                              std::to_string(acceleratorCount) + ")");
            }
            if (candidate.m_Reasons.empty() && !node.Fits(resources))
            {
                candidate.m_Reasons.push_back("resources do not fit");
            }
            double latency = dataRegion.empty() ? 0 : network.Latency(node.m_Region, dataRegion);
            if (objective && objective->m_MaxMs > 0 && !dataRegion.empty() && latency > objective->m_MaxMs / 2)
            {
                candidate.m_Reasons.push_back("latency to data region '" + dataRegion + "' " + Detail::Text(latency) +
                                              " ms exceeds half the " + Detail::Text(objective->m_MaxMs) +
                                              " ms objective");
            }
            candidate.m_Feasible = candidate.m_Reasons.empty();

            // normalised components
            double locality = maxDensity > 0 ? ObservableDensity(service, network, node.m_Region) / maxDensity : 0;
            double subscribers =
                maxConsumers > 0 ? ActionableConsumers(service, network, node.m_Region) / maxConsumers : 0;
            double shard = 0;
            if (!topics.empty())
            {
                size_t owned =
                    static_cast<size_t>(std::count_if(topics.begin(), topics.end(), [&](const std::string &topic)
                                                      { return network.OwnsShard(node.m_Region, topic); }));
                shard = static_cast<double>(owned) / static_cast<double>(topics.size());
            }
            double latencyScore = 1;
            if (!dataRegion.empty())
            {
                double limit = objective && objective->m_MaxMs > 0 ? objective->m_MaxMs : DEFAULT_LATENCY_SCALE;
                latencyScore = 1 - latency / limit;
            }
            double accelerator = 1;
            if (acceleratorCount)
            {
                uint32_t total = 0;
                if (auto found = node.m_Accelerators.find(resources.m_Accelerator); found != node.m_Accelerators.end())
                {
                    total = found->second;
                }
                // free share left after this reservation
                accelerator = total > 0 && freeAccelerators >= acceleratorCount
                                  ? static_cast<double>(freeAccelerators - acceleratorCount) / total
                                  : 0;
            }
            // 1 on a current node; 0.5 on a node behind the same node-local broker (the session reconnects there
            // without migrating to another broker); 0 elsewhere
            double session =
                hosting ? 1 : (!node.m_Broker.empty() && Detail::Contains(currentBrokers, node.m_Broker) ? 0.5 : 0);
            double capacity = node.m_Cpu > 0 ? (node.m_Cpu - node.m_UsedCpu - resources.m_Cpu) / node.m_Cpu : 0;
            double preference = Detail::Contains(constraints.m_Prefer, node.m_Region) ? 1 : 0;

            Component components[] = {{"locality", weightLocality, Clamp01(locality)},
                                      {"subscribers", weightSubscribers, Clamp01(subscribers)},
                                      {"shard", weightShard, Clamp01(shard)},
                                      {"latency", m_Weights.m_Latency, Clamp01(latencyScore)},
                                      {"accelerator", weightAccelerator, Clamp01(accelerator)},
                                      {"session", m_Weights.m_Session, session},
                                      {"capacity", m_Weights.m_Capacity, Clamp01(capacity)},
                                      {"preference", m_Weights.m_Preference, preference}};
            Json breakdown = Json::MakeObject();
            double score = 0;
            for (const Component &component : components)
            {
                breakdown.Set(component.m_Name, component.m_Value);
                score += component.m_Weight * component.m_Value;
            }
            candidate.m_Score = Clamp01(score / weightTotal);
            candidate.m_Breakdown = std::move(breakdown);
            plan.m_Candidates.push_back(std::move(candidate));
        }

        // feasible first, best score first, ties by node id
        std::stable_sort(plan.m_Candidates.begin(), plan.m_Candidates.end(),
                         [](const Candidate &a, const Candidate &b)
                         {
                             if (a.m_Feasible != b.m_Feasible)
                             {
                                 return a.m_Feasible;
                             }
                             if (a.m_Feasible && a.m_Score != b.m_Score)
                             {
                                 return a.m_Score > b.m_Score;
                             }
                             return a.m_Node < b.m_Node;
                         });
        size_t feasible = static_cast<size_t>(std::count_if(plan.m_Candidates.begin(), plan.m_Candidates.end(),
                                                            [](const Candidate &c) { return c.m_Feasible; }));

        // optional Planning advisor: may only reorder feasible candidates within the near-best window
        Intelligence::Provider *advisor =
            m_Environment.m_Providers ? m_Environment.m_Providers->Get(Intelligence::Capability::Planning) : nullptr;
        size_t nearBest = 0;
        while (nearBest < feasible &&
               plan.m_Candidates[nearBest].m_Score >= plan.m_Candidates[0].m_Score - ADVISOR_WINDOW)
        {
            ++nearBest;
        }
        if (advisor && nearBest >= 2)
        {
            Json options = Json::MakeArray();
            for (size_t i = 0; i < nearBest; ++i)
            {
                options.Push(plan.m_Candidates[i].ToJson());
            }
            using namespace Intelligence;
            Json schema = SchemaObject(
                {{"order", SchemaArray(SchemaScalar("string", "node id"))}, {"rationale", SchemaScalar("string")}});
            std::string user = "Service: " + plan.m_Service + " (" + roleName +
                               ")\nNear-equivalent feasible candidates (deterministic order): " + options.Dump() +
                               "\nNetwork knowledge: " + network.ToJson().Dump();
            std::string error;
            std::optional<Json> advice =
                AskJson(advisor, Capability::Planning,
                        "You advise the Placement agent of an MLOps framework. Order the given candidate nodes from "
                        "best to worst for communication-aware placement. Use only the given node ids.",
                        std::move(user), schema, &error);
            if (advice)
            {
                std::vector<Candidate> head(plan.m_Candidates.begin(),
                                            plan.m_Candidates.begin() + static_cast<std::ptrdiff_t>(nearBest));
                std::vector<Candidate> ordered;
                for (const std::string &id : Detail::ReadStrings(*advice, "order"))
                {
                    auto found =
                        std::find_if(head.begin(), head.end(), [&](const Candidate &c) { return c.m_Node == id; });
                    if (found != head.end())
                    {
                        ordered.push_back(std::move(*found));
                        head.erase(found);
                    }
                }
                for (Candidate &rest : head)
                {
                    ordered.push_back(std::move(rest));
                }
                bool changed = false;
                for (size_t i = 0; i < nearBest; ++i)
                {
                    changed = changed || ordered[i].m_Node != plan.m_Candidates[i].m_Node;
                }
                if (changed)
                {
                    std::move(ordered.begin(), ordered.end(), plan.m_Candidates.begin());
                    plan.m_Advisor = "llm:" + advisor->Name();
                }
            }
        }

        // replica selection
        std::vector<std::string> chosen;
        double chosenScore = 0;
        for (size_t i = 0; i < feasible && chosen.size() < replicas; ++i)
        {
            chosen.push_back(plan.m_Candidates[i].m_Node);
            chosenScore += plan.m_Candidates[i].m_Score;
        }
        if (!chosen.empty())
        {
            chosenScore /= static_cast<double>(chosen.size());
        }

        std::string rationale = std::string(roleName) + " of '" + plan.m_Service + "': " + std::to_string(feasible) +
                                " of " + std::to_string(plan.m_Candidates.size()) + " nodes feasible";
        if (!dataRegion.empty())
        {
            rationale += "; data region " + dataRegion;
        }
        if (chosen.size() < replicas)
        {
            plan.m_Rationale = rationale + "; infeasible: " + std::to_string(replicas) + " replica(s) required";
            for (const Candidate &candidate : plan.m_Candidates)
            {
                if (!candidate.m_Feasible && !candidate.m_Reasons.empty())
                {
                    plan.m_Rationale += "; " + candidate.m_Node + ": " + candidate.m_Reasons.front();
                }
            }
            // a current placement that is no longer feasible must move, but nowhere fits
            if (!current.empty())
            {
                plan.m_Nodes = current;
            }
            return plan;
        }

        // the best candidate's strongest weighted components
        const Candidate &best = plan.m_Candidates.front();
        std::vector<std::pair<double, std::string>> strongest;
        for (const auto &[name, value] : best.m_Breakdown.AsObject())
        {
            double weight = name == "locality"      ? weightLocality
                            : name == "subscribers" ? weightSubscribers
                            : name == "shard"       ? weightShard
                            : name == "accelerator" ? weightAccelerator
                            : name == "latency"     ? m_Weights.m_Latency
                            : name == "session"     ? m_Weights.m_Session
                            : name == "capacity"    ? m_Weights.m_Capacity
                                                    : m_Weights.m_Preference;
            if (value.AsNumber() > 0 && weight > 0)
            {
                strongest.emplace_back(weight * value.AsNumber(), name + " " + Percent(value.AsNumber()));
            }
        }
        std::stable_sort(strongest.begin(), strongest.end(),
                         [](const auto &a, const auto &b) { return a.first > b.first; });
        std::vector<std::string> top;
        for (size_t i = 0; i < strongest.size() && i < 3; ++i)
        {
            top.push_back(strongest[i].second);
        }
        rationale += "; best " + best.m_Node + " (" + best.m_Region + ") score " + Detail::Text(best.m_Score) +
                     (top.empty() ? std::string() : " [" + Join(top, ", ") + "]");
        if (!plan.m_Advisor.empty())
        {
            rationale += "; near-best order from " + plan.m_Advisor;
        }

        if (current.empty())
        {
            plan.m_Nodes = std::move(chosen);
            plan.m_Rationale = std::move(rationale);
            return plan;
        }

        // relocation: only when allowed, different, and better by at least the hysteresis
        double currentScore = 0;
        for (const std::string &id : current)
        {
            auto found = std::find_if(plan.m_Candidates.begin(), plan.m_Candidates.end(),
                                      [&](const Candidate &c) { return c.m_Node == id; });
            currentScore += found != plan.m_Candidates.end() && found->m_Feasible ? found->m_Score : 0;
        }
        currentScore /= static_cast<double>(current.size());
        plan.m_CurrentScore = currentScore;
        bool different = std::any_of(chosen.begin(), chosen.end(),
                                     [&](const std::string &id) { return !Detail::Contains(current, id); });
        double hysteresis = std::max(0.0, constraints.m_Hysteresis);
        plan.m_Relocate = constraints.m_AllowRelocation && different && chosenScore >= currentScore + hysteresis;
        if (plan.m_Relocate)
        {
            plan.m_Nodes = std::move(chosen);
            rationale += "; relocate from [" + Join(current, ", ") + "] (score " + Detail::Text(currentScore) + " → " +
                         Detail::Text(chosenScore) + ", hysteresis " + Detail::Text(hysteresis) + ")";
        }
        else
        {
            plan.m_Nodes = current;
            rationale +=
                "; keep [" + Join(current, ", ") + "] (score " + Detail::Text(currentScore) +
                (different ? (constraints.m_AllowRelocation ? ", gain below hysteresis " + Detail::Text(hysteresis)
                                                            : ", relocation disabled")
                           : ", already best") +
                ")";
        }
        plan.m_Rationale = std::move(rationale);
        return plan;
    }
} // namespace eOps::Agents
