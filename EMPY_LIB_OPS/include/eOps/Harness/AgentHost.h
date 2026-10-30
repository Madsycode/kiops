#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <eMQ/Client.h>
#include "../Common/Json.h"
#include "../Fabric/Bus.h"
#include "../Fabric/Topics.h"
#include "../Agents/Agent.h"
#include "../Agents/Executive.h"
#include "../Agents/Placement.h"
#include "../Agents/Lifecycle.h"
#include "../Agents/Compliance.h"

// hosts the agent swarm on an MMQP client. Serves AgentFilter(role) for each role; a request runs
// its agent core on a worker (service->Async) under that agent's mutex (cores that read or extend
// the shared Ontology — executive, compliance — also hold one ontology mutex) and returns to the
// owner thread through service->Dispatch guarded by a shared alive flag, answering
// "agent-response" {result} or an error reason. Tasks: executive/interpret {intent, operator},
// compliance/review {descriptor, deployed[]}, placement/place {descriptor, role, inventory,
// network, current[]}, lifecycle/evaluate {state, policy, candidate}.
// Event-driven roles: lifecycle listens to TriggerFilter("training"), EventFilter("drift"),
// EventFilter("metrics"), HeartbeatFilter() and the retained State("services") and proposes
// Workflow() requests {kind: retrain|rollback, service, reason, trigger, target?, observed?};
// placement listens to LocalityFilter() (CoMaS network knowledge + Observe),
// TriggerFilter("placement") (audit), EventFilter("failure") (relocate with replace: the lost
// instance is redeployed), EventFilter("drift") of kind latency, EventFilter("exhaustion") and
// EventFilter("accelerator") (re-placement of the affected role) and proposes relocate workflows;
// compliance listens to EventFilter("violation") and proposes an escalated retirement (HITL) after
// m_ViolationLimit violations within the window (rule "rate" is recorded, never escalated). Every
// decision is Remembered in CoMaS.

namespace eOps::Harness
{
    struct AgentHostOptions
    {
        std::vector<std::string> m_Roles = {"executive", "placement", "lifecycle", "compliance"};
        Fabric::Topics m_Topics;
        Agents::PolicySet m_Policies;
        Agents::PlacementWeights m_Weights;
        std::string m_Identity = "agents";
        uint32_t m_HarnessTimeoutMs = 30000;        // Workflow() proposals
        Descriptor::ValidationOptions m_Validation; // Executive validation
        std::filesystem::path m_OntologyState;      // evolved BoM ontology saved here ("" = not persisted)
        uint32_t m_ViolationLimit = 3; // violations within the window that propose a retirement (0 = never)
        uint32_t m_ViolationWindowSeconds = 300;
    };

    // requires an attached Fabric::BusModule; the environment members must outlive the module
    class EMPY_OPS_API AgentHost : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        AgentHost(AgentHostOptions options, Agents::AgentEnvironment environment);
        ~AgentHost() override;

        // {identity, roles[], requests, failures, proposals, busy{role: bool}}
        Json Status() const;

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        struct Impl;
        std::shared_ptr<Impl> m_Impl; // shared with in-flight worker tasks
    };
} // namespace eOps::Harness
