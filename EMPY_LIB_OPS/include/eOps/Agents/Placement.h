#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <string_view>
#include "Agent.h"
#include "../Common/Json.h"
#include "../Descriptor/Model.h"
#include "../Topology/Inventory.h"
#include "../Topology/Network.h"

// Placement agent: communication-aware placement by topic locality, subscriber distribution,
// profile-shard ownership, communication latency, accelerator availability and session
// locality. Inference migrates toward dense telemetry/action regions, training toward
// accelerator capacity; relocation happens through profile re-registration and session
// migration of the redeployed workload.

namespace eOps::Agents
{
    enum class Role : uint8_t
    {
        Inference,
        Training
    };

    EMPY_OPS_API const char *RoleName(Role role);       // "inference" | "training"
    EMPY_OPS_API Role ParseRole(std::string_view name); // throws std::invalid_argument

    struct EMPY_OPS_API Candidate
    {
        std::string m_Node;
        std::string m_Region;
        double m_Score = 0; // [0, 1]
        bool m_Feasible = false;
        std::vector<std::string> m_Reasons; // hard-filter rejections
        Json m_Breakdown;                   // component → normalised value
        Json ToJson() const;                // {node, region, score, feasible, reasons[], breakdown}
        static Candidate FromJson(const Json &json);
    };

    struct EMPY_OPS_API PlacementPlan
    {
        std::string m_Service;
        Role m_Role = Role::Inference;
        std::vector<std::string> m_Nodes;    // chosen nodes (replica count, distinct); empty when infeasible
        std::vector<Candidate> m_Candidates; // all candidates, best first
        std::string m_Rationale;
        bool m_Relocate = false; // a better placement than `current` beats the hysteresis
        double m_CurrentScore = 0;
        std::string m_Advisor; // "" | "llm:<provider>" when the Planning advisor reordered
        // {service, role, nodes[], candidates[], rationale, relocate, currentScore, advisor}
        Json ToJson() const;
        static PlacementPlan FromJson(const Json &json);
    };

    struct EMPY_OPS_API PlacementWeights
    {
        double m_Locality = 0.25;
        double m_Subscribers = 0.10;
        double m_Shard = 0.15;
        double m_Latency = 0.20;
        double m_Accelerator = 0.10;
        double m_Session = 0.10;
        double m_Capacity = 0.05;
        double m_Preference = 0.05;
        // {locality, subscribers, shard, latency, accelerator, session, capacity, preference}; missing → default
        static PlacementWeights FromYaml(const eCore::YamlNode &node);
        Json ToJson() const;
    };

    class EMPY_OPS_API Placement
    {
    public:
        Placement(AgentEnvironment environment, PlacementWeights weights = {});

        // `current` = nodes currently hosting this service role (their reservation is released before fitting)
        PlacementPlan Place(const Descriptor::RichMLService &service, Role role, const Topology::Inventory &inventory,
                            const Topology::NetworkModel &network, const std::vector<std::string> &current = {}) const;

    private:
        AgentEnvironment m_Environment;
        PlacementWeights m_Weights;
    };
} // namespace eOps::Agents
