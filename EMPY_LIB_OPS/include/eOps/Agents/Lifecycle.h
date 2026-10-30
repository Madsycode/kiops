#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include "Agent.h"
#include "../Common/Json.h"
#include "../Descriptor/Model.h"
#include "../Workload/Model.h"

// Lifecycle agent: drift detection, retraining management, model validation, rollback and
// version promotion, triggered by MMQP events (drift topic → retrain → validate → deployment).
// Decisions are proposals; the harness enacts them after Guard checks.

namespace eOps::Agents
{
    struct EMPY_OPS_API ModelRecord
    {
        uint32_t m_Version = 0;
        std::string m_ArtifactId;
        double m_Accuracy = 0;
        uint64_t m_At = 0;    // Unix ms of the last status change
        std::string m_Status; // candidate | active | retired | rejected | rolledback
        Json ToJson() const;  // {version, artifactId, accuracy, at, status}
        static ModelRecord FromJson(const Json &json);
    };

    struct EMPY_OPS_API LifecycleState
    {
        std::string m_Service;
        std::vector<ModelRecord> m_History; // ascending version
        std::optional<uint32_t> m_Active;
        uint64_t m_LastRetrain = 0;   // Unix ms a retrain was started
        uint64_t m_LastPromotion = 0; // Unix ms of the last promotion
        bool m_Training = false;      // a retrain workflow is running
        bool m_HasTraining = false;   // the descriptor declares training
        // {service, history[], active (null when none), lastRetrain, lastPromotion, training, hasTraining}
        Json ToJson() const;
        static LifecycleState FromJson(const Json &json);
        const ModelRecord *ActiveRecord() const;
        const ModelRecord *PreviousRecord() const; // last retired before active
        uint32_t NextVersion() const;              // highest recorded version + 1
    };

    enum class LifecycleAction : uint8_t
    {
        None,
        Retrain,
        Promote,
        Reject,
        Rollback
    };

    EMPY_OPS_API const char *ActionName(LifecycleAction action);     // none | retrain | promote | reject | rollback
    EMPY_OPS_API LifecycleAction ParseAction(std::string_view name); // throws std::invalid_argument

    struct EMPY_OPS_API LifecycleDecision
    {
        LifecycleAction m_Action = LifecycleAction::None;
        std::string m_Reason;
        std::optional<uint32_t> m_Target; // promote/rollback version
        std::optional<double> m_Observed; // the observed accuracy a drift/metrics decision was based on
        Json ToJson() const;              // {action, reason, target, observed} (null when absent)
        static LifecycleDecision FromJson(const Json &json);
    };

    class EMPY_OPS_API Lifecycle
    {
    public:
        explicit Lifecycle(AgentEnvironment environment);

        // `now` is Unix ms everywhere; cooldown = policy.cooldownSeconds. Only accuracy drift (kind accuracy, or
        // no kind) is a model concern; latency drift is left to the Placement agent (None)
        LifecycleDecision OnDrift(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                  const Json &drift, uint64_t now) const;
        LifecycleDecision OnSchedule(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                     uint64_t now) const;
        // bootstrap: no active model and training declared and idle → Retrain
        LifecycleDecision OnStatus(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                   const Json &status, uint64_t now) const;
        LifecycleDecision Evaluate(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                   const Workload::ModelArtifact &candidate, uint64_t now) const;
        LifecycleDecision OnMetrics(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                    const Json &metrics, uint64_t now) const;

    private:
        AgentEnvironment m_Environment;
    };
} // namespace eOps::Agents
