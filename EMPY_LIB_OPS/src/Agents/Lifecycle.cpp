#include <eOps/Agents/Lifecycle.h>
#include "Detail.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace eOps::Agents
{
    namespace
    {
        // bootstrap retries are spaced by min(cooldown, this) so a failed first training (too few samples yet)
        // is retried soon without a retrain per heartbeat
        constexpr uint64_t BOOTSTRAP_RETRY_MS = 5000;

        LifecycleDecision Decide(LifecycleAction action, std::string reason,
                                 std::optional<uint32_t> target = std::nullopt)
        {
            LifecycleDecision decision;
            decision.m_Action = action;
            decision.m_Reason = std::move(reason);
            decision.m_Target = target;
            return decision;
        }

        uint64_t CooldownMs(const Descriptor::LifecyclePolicy &policy)
        {
            return static_cast<uint64_t>(policy.m_CooldownSeconds) * 1000;
        }

        // remaining wait of a guard window that started at `since` (0 = never started)
        uint64_t Remaining(uint64_t since, uint64_t window, uint64_t now)
        {
            if (since == 0 || window == 0 || now >= since + window)
            {
                return 0;
            }
            return since + window - std::max(now, since);
        }

        std::string Seconds(uint64_t millis)
        {
            return Detail::Text(static_cast<double>(millis) / 1000) + " s";
        }

        // shared Retrain guards; nullopt = retraining may start
        std::optional<LifecycleDecision> RetrainBlocked(const LifecycleState &state,
                                                        const Descriptor::LifecyclePolicy &policy, uint64_t now,
                                                        uint64_t cooldownMs)
        {
            if (!state.m_HasTraining)
            {
                return Decide(LifecycleAction::None, "service declares no training");
            }
            if (state.m_Training)
            {
                return Decide(LifecycleAction::None, "a retrain is already running");
            }
            if (uint64_t wait = Remaining(state.m_LastRetrain, cooldownMs, now))
            {
                return Decide(LifecycleAction::None, "retrain cooldown (" + Seconds(wait) + " remaining of " +
                                                         std::to_string(policy.m_CooldownSeconds) + " s)");
            }
            return std::nullopt;
        }

        // observed accuracy of a metrics/status body: {accuracy: x} or {accuracy: {value: x}}
        std::optional<double> Observed(const Json &json, std::string_view key)
        {
            const Json *value = json.IsObject() ? json.Find(key) : nullptr;
            if (value && value->IsObject())
            {
                value = value->Find("value");
            }
            if (!value || !value->IsNumber() || !std::isfinite(value->AsNumber()))
            {
                return std::nullopt;
            }
            return value->AsNumber();
        }
    } // namespace

    // -------------------------
    // names and JSON forms
    // -------------------------

    const char *ActionName(LifecycleAction action)
    {
        switch (action)
        {
        case LifecycleAction::Retrain:
            return "retrain";
        case LifecycleAction::Promote:
            return "promote";
        case LifecycleAction::Reject:
            return "reject";
        case LifecycleAction::Rollback:
            return "rollback";
        case LifecycleAction::None:
        default:
            return "none";
        }
    }

    LifecycleAction ParseAction(std::string_view name)
    {
        for (LifecycleAction action : {LifecycleAction::None, LifecycleAction::Retrain, LifecycleAction::Promote,
                                       LifecycleAction::Reject, LifecycleAction::Rollback})
        {
            if (name == ActionName(action))
            {
                return action;
            }
        }
        throw std::invalid_argument("unknown lifecycle action '" + std::string(name) + "'");
    }

    Json ModelRecord::ToJson() const
    {
        return Json::MakeObject({{"version", m_Version},
                                 {"artifactId", m_ArtifactId},
                                 {"accuracy", m_Accuracy},
                                 {"at", m_At},
                                 {"status", m_Status}});
    }

    ModelRecord ModelRecord::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "model record");
        ModelRecord record;
        uint64_t version = Detail::ReadUnsigned(json, "version");
        if (version == 0 || version > std::numeric_limits<uint32_t>::max())
        {
            throw JsonError("model record version must be in [1, 2^32)");
        }
        record.m_Version = static_cast<uint32_t>(version);
        record.m_ArtifactId = Detail::ReadString(json, "artifactId");
        record.m_Accuracy = Detail::ReadNumber(json, "accuracy");
        record.m_At = Detail::ReadUnsigned(json, "at");
        record.m_Status = Detail::ReadString(json, "status");
        return record;
    }

    Json LifecycleState::ToJson() const
    {
        Json history = Json::MakeArray();
        for (const ModelRecord &record : m_History)
        {
            history.Push(record.ToJson());
        }
        return Json::MakeObject({{"service", m_Service},
                                 {"history", std::move(history)},
                                 {"active", m_Active ? Json(*m_Active) : Json()},
                                 {"lastRetrain", m_LastRetrain},
                                 {"lastPromotion", m_LastPromotion},
                                 {"training", m_Training},
                                 {"hasTraining", m_HasTraining}});
    }

    LifecycleState LifecycleState::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "lifecycle state");
        LifecycleState state;
        state.m_Service = Detail::ReadString(json, "service");
        for (const Json &item : Detail::ReadArray(json, "history"))
        {
            state.m_History.push_back(ModelRecord::FromJson(item));
        }
        std::stable_sort(state.m_History.begin(), state.m_History.end(),
                         [](const ModelRecord &a, const ModelRecord &b) { return a.m_Version < b.m_Version; });
        if (const Json *active = json.Find("active"); active && !active->IsNull())
        {
            uint64_t version = active->AsUint();
            if (version == 0 || version > std::numeric_limits<uint32_t>::max())
            {
                throw JsonError("lifecycle state active version must be in [1, 2^32)");
            }
            state.m_Active = static_cast<uint32_t>(version);
        }
        state.m_LastRetrain = Detail::ReadUnsigned(json, "lastRetrain");
        state.m_LastPromotion = Detail::ReadUnsigned(json, "lastPromotion");
        state.m_Training = Detail::ReadBool(json, "training");
        state.m_HasTraining = Detail::ReadBool(json, "hasTraining");
        return state;
    }

    const ModelRecord *LifecycleState::ActiveRecord() const
    {
        if (!m_Active)
        {
            return nullptr;
        }
        for (const ModelRecord &record : m_History)
        {
            if (record.m_Version == *m_Active)
            {
                return &record;
            }
        }
        return nullptr;
    }

    const ModelRecord *LifecycleState::PreviousRecord() const
    {
        const ModelRecord *previous = nullptr;
        for (const ModelRecord &record : m_History)
        {
            if (record.m_Status != "retired" || (m_Active && record.m_Version >= *m_Active))
            {
                continue;
            }
            if (!previous || record.m_Version > previous->m_Version)
            {
                previous = &record;
            }
        }
        return previous;
    }

    uint32_t LifecycleState::NextVersion() const
    {
        uint32_t highest = m_Active.value_or(0);
        for (const ModelRecord &record : m_History)
        {
            highest = std::max(highest, record.m_Version);
        }
        return highest + 1;
    }

    Json LifecycleDecision::ToJson() const
    {
        return Json::MakeObject({{"action", ActionName(m_Action)},
                                 {"reason", m_Reason},
                                 {"target", m_Target ? Json(*m_Target) : Json()},
                                 {"observed", m_Observed ? Json(*m_Observed) : Json()}});
    }

    LifecycleDecision LifecycleDecision::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "lifecycle decision");
        LifecycleDecision decision;
        try
        {
            decision.m_Action = ParseAction(Detail::ReadString(json, "action", "none"));
        }
        catch (const std::invalid_argument &error)
        {
            throw JsonError(error.what());
        }
        decision.m_Reason = Detail::ReadString(json, "reason");
        if (const Json *target = json.Find("target"); target && !target->IsNull())
        {
            uint64_t version = target->AsUint();
            if (version > std::numeric_limits<uint32_t>::max())
            {
                throw JsonError("lifecycle decision target is out of range");
            }
            decision.m_Target = static_cast<uint32_t>(version);
        }
        decision.m_Observed = Detail::ReadOptionalNumber(json, "observed");
        return decision;
    }

    // -------------------------
    // Lifecycle
    // -------------------------

    Lifecycle::Lifecycle(AgentEnvironment environment) : m_Environment(environment)
    {
    }

    LifecycleDecision Lifecycle::OnDrift(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                         const Json &drift, uint64_t now) const
    {
        std::string kind = drift.IsObject() ? drift.GetString("kind", "drift") : std::string("drift");
        if (kind != "accuracy" && kind != "drift")
        {
            // a retrained model cannot shorten the observation-to-action path: placement re-evaluates instead
            return Decide(LifecycleAction::None, kind + " drift is not a model concern (placement re-evaluates)");
        }
        if (std::optional<LifecycleDecision> blocked = RetrainBlocked(state, policy, now, CooldownMs(policy)))
        {
            return *blocked;
        }
        std::string reason = "drift (" + kind + ")";
        std::optional<double> value = Observed(drift, "value");
        if (value)
        {
            reason += " " + Detail::Text(*value);
            if (std::optional<double> threshold = Observed(drift, "threshold"))
            {
                reason += " vs threshold " + Detail::Text(*threshold);
            }
        }
        LifecycleDecision decision = Decide(LifecycleAction::Retrain, reason);
        decision.m_Observed = value;
        return decision;
    }

    LifecycleDecision Lifecycle::OnSchedule(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                            uint64_t now) const
    {
        if (std::optional<LifecycleDecision> blocked = RetrainBlocked(state, policy, now, CooldownMs(policy)))
        {
            return *blocked;
        }
        return Decide(LifecycleAction::Retrain, "scheduled retraining");
    }

    LifecycleDecision Lifecycle::OnStatus(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                          const Json &status, uint64_t now) const
    {
        if (state.m_Active)
        {
            return Decide(LifecycleAction::None, "model v" + std::to_string(*state.m_Active) + " is active");
        }
        if (status.IsObject() && status.GetInt("modelVersion", 0) > 0)
        {
            return Decide(LifecycleAction::None,
                          "instance already serves model v" + std::to_string(status.GetInt("modelVersion", 0)));
        }
        uint64_t retry = std::min<uint64_t>(CooldownMs(policy), BOOTSTRAP_RETRY_MS);
        if (std::optional<LifecycleDecision> blocked = RetrainBlocked(state, policy, now, retry))
        {
            return *blocked;
        }
        return Decide(LifecycleAction::Retrain, "bootstrap: no active model");
    }

    LifecycleDecision Lifecycle::Evaluate(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                          const Workload::ModelArtifact &candidate, uint64_t now) const
    {
        std::string name = "candidate v" + std::to_string(candidate.m_Version);
        if (candidate.m_Version == 0)
        {
            return Decide(LifecycleAction::Reject, name + " has no version");
        }
        if (!std::isfinite(candidate.m_Accuracy) || candidate.m_Accuracy < policy.m_AccuracyThreshold)
        {
            return Decide(LifecycleAction::Reject,
                          name + " accuracy " + Detail::Text(candidate.m_Accuracy) + " below threshold " +
                              Detail::Text(policy.m_AccuracyThreshold),
                          candidate.m_Version);
        }
        if (const ModelRecord *active = state.ActiveRecord())
        {
            if (candidate.m_Version <= active->m_Version)
            {
                return Decide(LifecycleAction::Reject,
                              name + " is not newer than active v" + std::to_string(active->m_Version),
                              candidate.m_Version);
            }
            double improvement = candidate.m_Accuracy - active->m_Accuracy;
            if (improvement < policy.m_MinImprovement)
            {
                return Decide(LifecycleAction::Reject,
                              name + " improves accuracy by " + Detail::Text(improvement) + " over v" +
                                  std::to_string(active->m_Version) + " (minimum " +
                                  Detail::Text(policy.m_MinImprovement) + ")",
                              candidate.m_Version);
            }
            if (uint64_t wait = Remaining(state.m_LastPromotion, CooldownMs(policy), now))
            {
                return Decide(LifecycleAction::Reject,
                              name + " rejected by promotion cooldown (" + Seconds(wait) + " remaining)",
                              candidate.m_Version);
            }
            return Decide(LifecycleAction::Promote,
                          name + " accuracy " + Detail::Text(candidate.m_Accuracy) + " improves on v" +
                              std::to_string(active->m_Version) + " (" + Detail::Text(active->m_Accuracy) + ")",
                          candidate.m_Version);
        }
        if (state.m_Active)
        {
            // an active version without a record: promote only a newer one
            if (candidate.m_Version <= *state.m_Active)
            {
                return Decide(LifecycleAction::Reject,
                              name + " is not newer than active v" + std::to_string(*state.m_Active),
                              candidate.m_Version);
            }
        }
        return Decide(LifecycleAction::Promote,
                      name + " accuracy " + Detail::Text(candidate.m_Accuracy) + " meets threshold " +
                          Detail::Text(policy.m_AccuracyThreshold) + (state.m_Active ? "" : " (first model)"),
                      candidate.m_Version);
    }

    LifecycleDecision Lifecycle::OnMetrics(const LifecycleState &state, const Descriptor::LifecyclePolicy &policy,
                                           const Json &metrics, uint64_t now) const
    {
        const ModelRecord *active = state.ActiveRecord();
        if (!state.m_Active)
        {
            return Decide(LifecycleAction::None, "no active model");
        }
        std::optional<double> accuracy = Observed(metrics, "accuracy");
        if (!accuracy)
        {
            return Decide(LifecycleAction::None, "metrics carry no accuracy");
        }
        int64_t samples = metrics.IsObject() ? metrics.GetInt("samples", 0) : 0;
        if (samples < static_cast<int64_t>(policy.m_MinSamples))
        {
            return Decide(LifecycleAction::None, "insufficient samples (" + std::to_string(samples) + " < " +
                                                     std::to_string(policy.m_MinSamples) + ")");
        }
        if (int64_t version = metrics.GetInt("modelVersion", 0); version > 0 && version != *state.m_Active)
        {
            return Decide(LifecycleAction::None, "metrics describe model v" + std::to_string(version) +
                                                     ", not the active v" + std::to_string(*state.m_Active));
        }
        if (*accuracy >= policy.m_AccuracyThreshold)
        {
            return Decide(LifecycleAction::None, "accuracy " + Detail::Text(*accuracy) + " meets threshold " +
                                                     Detail::Text(policy.m_AccuracyThreshold));
        }
        std::string degraded = "observed accuracy " + Detail::Text(*accuracy) + " below threshold " +
                               Detail::Text(policy.m_AccuracyThreshold);
        if (const ModelRecord *previous = state.PreviousRecord(); policy.m_AutoRollback && previous &&
                                                                  previous->m_Accuracy >= policy.m_AccuracyThreshold &&
                                                                  previous->m_Accuracy > *accuracy)
        {
            // a previous record carries its last known accuracy: the harness records the degraded accuracy on a
            // model it replaced because of drift, so such a model is never chosen here
            LifecycleDecision rollback =
                Decide(LifecycleAction::Rollback,
                       degraded + " for v" + std::to_string(active ? active->m_Version : *state.m_Active) +
                           "; roll back to v" + std::to_string(previous->m_Version) + " (" +
                           Detail::Text(previous->m_Accuracy) + ")",
                       previous->m_Version);
            rollback.m_Observed = accuracy;
            return rollback;
        }
        if (std::optional<LifecycleDecision> blocked = RetrainBlocked(state, policy, now, CooldownMs(policy)))
        {
            blocked->m_Reason = degraded + "; " + blocked->m_Reason;
            return *blocked;
        }
        LifecycleDecision retrain = Decide(LifecycleAction::Retrain, degraded);
        retrain.m_Observed = accuracy;
        return retrain;
    }
} // namespace eOps::Agents
