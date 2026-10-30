#pragma once
#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include "../API.h"

// MLOps control-plane topic map (all under one configurable root, default "mlops"):
//   Intent()                <r>/intent                        REQUEST → harness        intent / workflow
//   Deploy()                <r>/deploy                        REQUEST → harness        deploy / workflow
//   Workflow()              <r>/harness/workflow              REQUEST agents → harness workflow-request / workflow
//   Status()                <r>/harness/status                REQUEST → harness        status-request / snapshot
//   Agent(role, task)       <r>/agent/<role>/<task>           REQUEST harness → agent  agent-request / agent-response
//   Trigger(service, job)   <r>/trigger/time/<service>/<job>  temporal trigger         trigger
//   Event(service, kind)    <r>/event/<service>/<kind>        drift|metrics|failure|accelerator|exhaustion|violation
//   Heartbeat(service, id)  <r>/status/<service>/<instance>   retained                 status
//   Workload(service, cmd)  <r>/workload/<service>/<command>  REQUEST harness → workload train / model
//   Candidate(service)      <r>/model/<service>/candidate     the "train/model" observable: artifact
//   Hitl()                  <r>/hitl/approval                 REQUEST harness → operators hitl / hitl-decision
//   State(name)             <r>/state/<name>                  retained (services, workflows) state
//   Locality(region)        <r>/telemetry/locality/<region>   locality

namespace eOps::Fabric
{
    struct EMPY_OPS_API Topics
    {
        std::string m_Root = "mlops";

        Topics() = default;
        explicit Topics(std::string root); // canonical root without trailing '/'; throws std::invalid_argument

        std::string Intent() const;
        std::string Deploy() const;
        std::string Workflow() const;
        std::string Status() const;
        std::string Agent(std::string_view role, std::string_view task) const;
        std::string AgentFilter(std::string_view role) const; // <r>/agent/<role>/+
        std::string Trigger(std::string_view service, std::string_view job) const;
        std::string TriggerFilter(std::string_view job) const; // <r>/trigger/time/+/<job>
        std::string Event(std::string_view service, std::string_view kind) const;
        std::string EventFilter(std::string_view kind) const; // <r>/event/+/<kind>
        std::string Heartbeat(std::string_view service, std::string_view instance) const;
        std::string HeartbeatFilter() const; // <r>/status/+/+
        std::string Workload(std::string_view service, std::string_view command) const;
        std::string WorkloadFilter(std::string_view service) const; // <r>/workload/<service>/+
        std::string Candidate(std::string_view service) const;
        std::string CandidateFilter() const; // <r>/model/+/candidate
        std::string Hitl() const;
        std::string State(std::string_view name) const;
        std::string Locality(std::string_view region) const;
        std::string LocalityFilter() const; // <r>/telemetry/locality/+
        std::string All() const;            // <r>/#

        // levels of `topic` below the root ("mlops/event/home-control/drift" → event, home-control, drift);
        // nullopt when the topic is not under the root
        std::optional<std::vector<std::string>> Relative(std::string_view topic) const;
    };
} // namespace eOps::Fabric
