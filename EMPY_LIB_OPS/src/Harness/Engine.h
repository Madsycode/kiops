#pragma once
#include <map>
#include <set>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <eOps/Harness/Harness.h>
#include <eOps/Agents/Compliance.h>
#include <eOps/Agents/Placement.h>
#include <eOps/Agents/Lifecycle.h>
#include <eOps/Workload/Model.h>
#include <eOps/Substrate/Substrate.h>

// ExecutionHarness internals shared by Harness.cpp (module, serving, state publication, health) and
// Workflows.cpp (the deploy/retrain/rollback/relocate/retire state machines). Owner thread only,
// except the substrate work handed to workers by Enact (and the substrate listing of the health check).

namespace eOps::Harness
{
    namespace Detail
    {
        inline std::string ReasonText(const eMQ::Reason &reason)
        {
            return reason.m_Text.empty() ? "reason " + std::to_string(reason.m_Code) : reason.m_Text;
        }

        // an operator's HITL answer (or the configured default) in normalised form
        struct HitlAnswer
        {
            bool m_Approve = false;
            std::string m_Operator;
            std::string m_Note;
            std::string m_Source; // operator | default
            Json ToJson() const
            {
                return Json::MakeObject({{"decision", m_Approve ? "approve" : "reject"},
                                         {"operator", m_Operator},
                                         {"note", m_Note},
                                         {"source", m_Source}});
            }
            std::string Who() const
            {
                return m_Source == "default" ? "by default (" + m_Note + ")" : "by operator '" + m_Operator + "'";
            }
        };
    } // namespace Detail

    struct ExecutionHarness::Impl
    {
        using Clock = std::chrono::steady_clock;
        using Continue = std::function<void(const Json &result)>;

        // transient data of one workflow between its asynchronous steps
        struct Run
        {
            std::string m_Id;
            Completion m_Done;
            Fabric::Bus::Responder m_Respond; // requester waiting for the finished record (may be empty)
            bool m_DryRun = false;
            std::optional<Descriptor::RichMLService> m_Candidate;
            std::optional<Descriptor::RichMLService> m_Reviewed; // the candidate compliance reviewed (amendments)
            Agents::ComplianceReport m_Report;
            std::optional<Agents::PlacementPlan> m_InferencePlan;
            std::optional<Agents::PlacementPlan> m_TrainingPlan;
            std::vector<Instance> m_Previous;     // instances replaced by a redeploy
            std::optional<Deployment> m_Replaced; // redeploy: the deployment record the new instances replace
            std::optional<Workload::ModelArtifact> m_Artifact;
            std::optional<double> m_Observed; // retrain: harness-observed accuracy of the active model
            Agents::LifecycleDecision m_Decision;
            std::vector<std::string> m_Roles;   // relocate: roles still to examine
            std::vector<std::string> m_Replace; // relocate: lost instances to redeploy in any case
            Json m_Moves = Json::MakeArray();   // relocate: performed moves
            // readiness gate: instances that must report before m_OnReady runs (m_OnLate at the deadline)
            bool m_WaitingReady = false;
            Clock::time_point m_ReadyDeadline;
            std::vector<std::string> m_ReadyInstances;
            std::map<std::string, int64_t> m_ReadySamples; // training instance → samples it must hold
            std::function<void()> m_OnReady;
            std::function<void(const Json &missing)> m_OnLate;
        };

        // facts of the last heartbeat of an instance
        struct Beat
        {
            std::string m_Role;
            int64_t m_Model = 0;
            int64_t m_Samples = 0;
        };

        // accuracy of a service's active model as the harness saw it on the fabric (drift/metrics events)
        struct Observation
        {
            uint32_t m_Version = 0;
            double m_Accuracy = 0;
            uint64_t m_At = 0;
        };

        HarnessOptions m_Options;
        std::shared_ptr<Substrate::ExecutionSubstrate> m_Substrate;
        std::shared_ptr<Context::Comas> m_Comas;
        StateStore m_State;
        Scheduler m_Scheduler;
        Guard m_Guard;
        std::function<void(const Json &)> m_Sink;

        eMQ::Service *m_Service = nullptr;
        Fabric::Bus *m_Bus = nullptr;
        const void *m_Owner = nullptr;
        std::shared_ptr<bool> m_Alive = std::make_shared<bool>(false); // owner thread only
        std::map<std::string, std::shared_ptr<Run>> m_Runs;
        bool m_Dirty = false;
        Clock::time_point m_LastPublish{};
        Clock::time_point m_LastConsolidation{};
        Clock::time_point m_LastHealth{};
        std::map<std::string, Clock::time_point> m_LastSync; // per-service model synchronisation
        std::shared_ptr<std::function<void()>> m_Tick;       // self-rearming scheduler timer
        std::map<std::string, Beat> m_Beats;                 // instance → last heartbeat
        std::map<std::string, Observation> m_Observed;       // service → observed accuracy of its active model
        std::map<std::string, Clock::time_point> m_Raised;   // "<kind>/<instance>" → when that event was raised
        bool m_Listing = false;                              // a substrate listing runs on a worker
        std::string m_Inventory; // the reserved inventory last checked (a registry change republishes the state)

        Impl(HarnessOptions options, std::shared_ptr<Substrate::ExecutionSubstrate> substrate,
             std::shared_ptr<Context::Comas> comas);

        // ---- module glue (Harness.cpp)
        void Attach(eMQ::Service *service, const void *owner);
        void Update();
        void Detach();
        std::string Submit(Json request, std::string trigger, Completion done, Fabric::Bus::Responder respond);
        Json Snapshot(size_t workflows = 32) const;
        void MarkDirty();
        void Publish();
        void OnHeartbeat(const std::string &topic, const Fabric::Envelope &envelope);
        void OnEvent(const std::string &topic, const Fabric::Envelope &envelope);
        void SyncModel(const std::string &service, bool force = false);
        void CheckReady();
        void FireSchedule(const ScheduleEntry &entry, std::chrono::system_clock::time_point firedAt);
        // reservations of instances no deployment holds (a workflow cut short by a stop) are released
        void ReconcileReservations();

        // ---- health: liveness, substrate states, node capacity → failure/exhaustion/accelerator events
        void CheckHealth();
        void OnSubstrateList(const std::vector<Substrate::InstanceStatus> &statuses);
        void Raise(const std::string &service, const Instance &instance, const std::string &kind, Json detail);
        std::optional<double> ObservedAccuracy(const std::string &service) const;

        // ---- step helpers (Harness.cpp)
        std::shared_ptr<Run> Find(const std::string &id) const;
        WorkflowRecord *Record(const std::string &id);
        void Begin(const std::string &id, std::string step);
        void End(const std::string &id, std::string status, Json detail = Json());
        void Finish(const std::string &id, std::string status, Json result, std::string error = {});
        // runs `fn`; an exception fails the workflow
        void Guarded(const std::string &id, const std::function<void()> &fn);
        // Call Agent(role, task) with `body`; failures finish the workflow, success continues with "result"
        void Ask(const std::string &id, const std::string &role, const std::string &task, Json body, Continue next);
        // HITL request on Hitl(); `next` gets the operator's answer or, on timeout/no responder, the default
        void AskHuman(const std::string &id, Json body, std::function<void(const Detail::HitlAnswer &)> next);
        // run `work` on a worker (substrate operations); `next` gets its JSON result on the owner thread
        void Enact(const std::string &id, std::function<Json()> work, Continue next);
        Topology::Inventory BaseInventory() const;
        Topology::NetworkModel Network() const;
        void Remember(const WorkflowRecord &record);

        // ---- workflows (Workflows.cpp)
        void Start(const std::string &id);
        void StartDeploy(const std::string &id);
        void Validate(const std::string &id);
        void Review(const std::string &id);
        void Escalate(const std::string &id);
        void PlaceAll(const std::string &id);
        void PlaceRole(const std::string &id, Agents::Role role, std::function<void()> next);
        void EnactDeploy(const std::string &id);
        void RecordDeploy(const std::string &id, const std::vector<Substrate::DeploymentSpec> &specs);
        void RetirePrevious(const std::string &id);
        void AwaitReady(const std::string &id);
        // readiness gate over `instances` of the run's service (see Run); without waitForReady `ready` runs at once
        void WaitFor(const std::string &id, std::vector<std::string> instances, std::function<void()> ready,
                     std::function<void(const Json &missing)> late);
        bool InstanceReady(const Deployment &deployment, const Instance &instance, const Run &run) const;
        void CompleteDeploy(const std::string &id);
        void StartRetrain(const std::string &id);
        void EvaluateCandidate(const std::string &id);
        void LoadModel(const std::string &id, const Workload::ModelArtifact &artifact,
                       std::function<void(size_t loaded, size_t expected)> next);
        void StartRollback(const std::string &id);
        void StartRelocate(const std::string &id);
        void RelocateNext(const std::string &id);
        void StartRetire(const std::string &id);
        void RetireService(const std::string &id);

        // deployment specs for `plan` with instance names unused by any current reservation/instance
        std::vector<Substrate::DeploymentSpec> Specs(const Descriptor::RichMLService &service, Agents::Role role,
                                                     const Agents::PlacementPlan &plan,
                                                     std::set<std::string> &taken) const;
        std::set<std::string> TakenInstances() const;
        void ClearTraining(const std::string &service);
        static void Prune(Deployment &deployment);
    };
} // namespace eOps::Harness
