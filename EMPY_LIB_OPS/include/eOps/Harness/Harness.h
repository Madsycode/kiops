#pragma once
#include <memory>
#include <string>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <eMQ/Client.h>
#include "State.h"
#include "Guard.h"
#include "Scheduler.h"
#include "../Common/Json.h"
#include "../Fabric/Bus.h"
#include "../Fabric/Topics.h"
#include "../Context/Comas.h"
#include "../Descriptor/Validator.h"
#include "../Substrate/Substrate.h"

// Harness Runtime (thesis §2.1.2): deterministic execution environment that maintains execution
// state, coordinates agents and tools and interacts with MMQP. Activation by temporal triggers,
// event-driven triggers and request/response workflows. Every workflow is an explicit state
// machine of asynchronous owner-thread steps; each agent interaction is a Bus Call to
// Agent(role, task) carrying the harness-provided context (agents stay stateless); replies are
// validated by the Guard before any enactment on the execution substrate.
//
// Serves Intent() {intent, dryRun?, operator?}, Deploy() {descriptor: YAML, dryRun?},
// Workflow() {kind: deploy|retrain|rollback|relocate|retire, service, ...} and Status() and
// responds with the workflow record (kind "workflow") when it finishes, or the snapshot.
//
// Event-driven triggers it raises itself (Event(service, kind), re-derived from its own state):
// "failure" (an instance silent for livenessMs after a heartbeat, or one the substrate reports
// exited), "exhaustion" (a node's reservations exceed its registered cpu/memory) and "accelerator"
// (a node no longer has the accelerators reserved on it). The workloads raise "drift", "metrics" and
// "violation"; the agents turn all of them into Workflow() proposals. Relocate requests may name
// `replace: [instance]` (lost instances to redeploy even when the placement keeps their node); a
// retire request with `escalate: true` asks the operators (HITL) first.

namespace eOps::Harness
{
    struct HarnessOptions
    {
        Fabric::Topics m_Topics;
        std::string m_Identity = "harness";
        uint32_t m_AgentTimeoutMs = 30000;
        uint32_t m_HitlTimeoutMs = 60000;
        bool m_HitlDefaultApprove = false;      // decision on HITL timeout / no responder
        uint32_t m_PlacementAuditSeconds = 0;   // 0 = no periodic placement trigger
        uint32_t m_ConsolidationSeconds = 60;   // 0 = no CoMaS consolidation
        std::filesystem::path m_StateDirectory; // StateStore persistence ("" = memory)
        // deploy waits for ready heartbeats; a redeploy or relocation also waits for its new instances (serving the
        // active model, or holding the samples of the training instance they replace) before retiring the old ones
        bool m_WaitForReady = false;
        uint32_t m_ReadyTimeoutMs = 30000;
        std::string m_BrokerHost = "127.0.0.1"; // handed to substrate deployments (a node's own broker wins)
        uint16_t m_BrokerPort = 5600;
        size_t m_MaxWorkflows = 256;
        Descriptor::ValidationOptions m_Validation;
        // an instance silent this long after a heartbeat is lost: Event(service, "failure"); 0 = no liveness check
        uint32_t m_LivenessMs = 15000;
        // the workload MMQP sessions use TLS or QUIC; a descriptor declaring security.encryption is refused
        // otherwise (identity is handed to the workloads as --identity)
        bool m_SecureWorkloads = false;
    };

    // requires an attached Fabric::BusModule
    class EMPY_OPS_API ExecutionHarness : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        using Completion = std::function<void(const WorkflowRecord &record)>;

        ExecutionHarness(HarnessOptions options, std::shared_ptr<Substrate::ExecutionSubstrate> substrate,
                         std::shared_ptr<Context::Comas> comas);
        ~ExecutionHarness() override;

        // owner thread: start a workflow exactly as a Workflow()/Intent()/Deploy() request would
        // (`request` = {kind, ...}; deploy accepts {intent} or {descriptor}); `done` runs once when it finishes.
        // Returns the workflow id, or "" when the Guard rejects the request (then `done` gets the rejected record).
        std::string Submit(Json request, std::string trigger, Completion done = {});

        // {identity, deployments[], workflows[] (latest first, bounded), schedules[], inventory, substrate, time}
        Json Snapshot() const;
        const StateStore &State() const;
        const Scheduler &Schedules() const;
        // the last snapshot is also pushed here after each change (HTTP endpoint)
        void SetSnapshotSink(std::function<void(const Json &)> sink);

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Harness
