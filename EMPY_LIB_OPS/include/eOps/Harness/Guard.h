#pragma once
#include <string>
#include <vector>
#include <optional>
#include "State.h"
#include "../Common/Json.h"
#include "../Descriptor/Model.h"
#include "../Descriptor/Validator.h"
#include "../Topology/Inventory.h"
#include "../Topology/Network.h"
#include "../Agents/Compliance.h"
#include "../Agents/Placement.h"
#include "../Agents/Lifecycle.h"
#include "../Workload/Model.h"

// harness Guard: validates every agent output against the formally declared execution
// boundaries before any operational action. It re-derives each constraint itself and never
// trusts agent claims (scores, feasibility flags, verdict texts).

namespace eOps::Harness
{
    struct EMPY_OPS_API GuardResult
    {
        bool m_Ok = true;
        std::vector<std::string> m_Violations;
        void Fail(std::string violation); // m_Ok = false + record
        Json ToJson() const;              // {ok, violations[]}
    };

    class EMPY_OPS_API Guard
    {
    public:
        // Descriptor::Validate errors are violations
        GuardResult CheckDescriptor(const Descriptor::RichMLService &service,
                                    const Descriptor::ValidationOptions &options) const;
        // Approved/Amended only; an amended descriptor is re-validated (with `options`)
        GuardResult CheckCompliance(const Agents::ComplianceReport &report,
                                    const Descriptor::ValidationOptions &options = {}) const;
        // as above, and an amended descriptor may only restrict `candidate` (CheckAmendment)
        GuardResult CheckCompliance(const Descriptor::RichMLService &candidate, const Agents::ComplianceReport &report,
                                    const Descriptor::ValidationOptions &options = {}) const;
        // an amendment (automatic or approved by a human) restricts the reviewed candidate and nothing else: same
        // name and version, every section but the actionables (and the actuation topic derived from them) as
        // declared, each actionable covered by a declared one with bounds and rate inside the declared ones
        GuardResult CheckAmendment(const Descriptor::RichMLService &candidate,
                                   const Descriptor::RichMLService &amended) const;
        // nodes exist, count = replicas (training 1), distinct, region allowed and not avoided, pin respected,
        // resources/accelerators fit `inventory` (which already carries the current reservations)
        GuardResult CheckPlacement(const Descriptor::RichMLService &service, Agents::Role role,
                                   const Agents::PlacementPlan &plan, const Topology::Inventory &inventory) const;
        // as above, and an inference role with a latency objective keeps every node within maxMs/2 of the data
        // region (densest observable telemetry, else densest actionable consumers) re-derived from `network`
        GuardResult CheckPlacement(const Descriptor::RichMLService &service, Agents::Role role,
                                   const Agents::PlacementPlan &plan, const Topology::Inventory &inventory,
                                   const Topology::NetworkModel &network) const;
        // kind ∈ {deploy, retrain, rollback, relocate, retire}; service known (except deploy); no duplicate running;
        // retrain `observed` (when given) is an accuracy in [0, 1]
        GuardResult CheckWorkflowRequest(const Json &request, const StateStore &state) const;
        // decision is Promote; Workload::CheckArtifact; version > active; accuracy ≥ threshold; improvement over the
        // active model (its recorded accuracy, or the lower harness-observed accuracy) ≥ minImprovement
        GuardResult CheckPromotion(const Deployment &deployment, const Workload::ModelArtifact &candidate,
                                   const Agents::LifecycleDecision &decision,
                                   std::optional<double> observed = std::nullopt) const;
        // decision is Rollback; the target artefact exists, is not active, satisfies Workload::CheckArtifact, was
        // promoted before (history status retired or rolledback) and its last known accuracy meets the threshold
        GuardResult CheckRollback(const Deployment &deployment, const Agents::LifecycleDecision &decision) const;
    };
} // namespace eOps::Harness
