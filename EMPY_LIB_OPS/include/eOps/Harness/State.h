#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include "../Common/Json.h"
#include "../Descriptor/Model.h"
#include "../Topology/Inventory.h"
#include "../Workload/Model.h"
#include "../Agents/Lifecycle.h"

// harness execution state: deployments with their instances, lifecycle state and artefacts,
// bounded workflow records and inventory reservations; optionally persisted through
// eCore::MMQP::FileStore (journaled) so a restarted harness resumes its view

namespace eOps::Harness
{
    struct EMPY_OPS_API Instance
    {
        std::string m_Id;
        std::string m_Role; // inference | training
        std::string m_Node;
        std::string m_State;     // deploying | running | ready | waiting-model | lost | retired
        uint64_t m_LastSeen = 0; // Unix ms of the last heartbeat
        Json ToJson() const;     // {id, role, node, state, lastSeen}
        static Instance FromJson(const Json &json);
    };

    struct EMPY_OPS_API Deployment
    {
        std::string m_Service;
        std::string m_Version;
        Descriptor::RichMLService m_Descriptor;
        std::vector<Instance> m_Instances;
        Agents::LifecycleState m_Lifecycle;
        std::map<uint32_t, Workload::ModelArtifact> m_Artifacts; // version → artefact
        std::string m_Status;                                    // deploying | deployed | degraded | retiring
        uint64_t m_Updated = 0;
        // {service, version, descriptor (JSON), instances[], lifecycle, artifacts[], status, updated}
        Json ToJson() const;
        static Deployment FromJson(const Json &json);
    };

    struct EMPY_OPS_API StepRecord
    {
        std::string m_Name;
        std::string m_Status; // running | succeeded | failed | skipped
        uint64_t m_Started = 0;
        uint64_t m_Finished = 0;
        Json m_Detail;
        Json ToJson() const; // {name, status, started, finished, detail}
        static StepRecord FromJson(const Json &json);
    };

    struct EMPY_OPS_API WorkflowRecord
    {
        std::string m_Id;
        std::string m_Kind; // deploy | retrain | rollback | relocate | retire
        std::string m_Service;
        std::string m_Status;  // running | waiting-hitl | succeeded | failed | rejected
        std::string m_Trigger; // intent | descriptor | schedule | drift | metrics | bootstrap | operator | audit ...
        Json m_Input;
        Json m_Result;
        std::string m_Error;
        std::vector<StepRecord> m_Steps;
        uint64_t m_Created = 0;
        uint64_t m_Updated = 0;
        bool Finished() const noexcept; // succeeded | failed | rejected
        // {id, kind, service, status, trigger, input, result, error, steps[], created, updated}
        Json ToJson() const;
        static WorkflowRecord FromJson(const Json &json);
    };

    struct StateOptions
    {
        std::filesystem::path m_Directory; // "" = in memory only
        size_t m_MaxWorkflows = 256;       // finished records beyond this are evicted oldest first
    };

    // owner thread only. Pointers returned by Find* stay valid until that record is erased/evicted;
    // running workflows are never evicted.
    class EMPY_OPS_API StateStore
    {
    public:
        explicit StateStore(StateOptions options = {}); // reloads persisted state; throws on a corrupt store
        ~StateStore();
        StateStore(const StateStore &) = delete;
        StateStore &operator=(const StateStore &) = delete;

        // deployments (persisted on every Put/Erase)
        Deployment *FindDeployment(std::string_view service);
        const Deployment *FindDeployment(std::string_view service) const;
        void PutDeployment(Deployment deployment); // insert or replace by service; sets m_Updated
        bool EraseDeployment(std::string_view service);
        std::vector<const Deployment *> Deployments() const; // ordered by service name

        // workflows
        WorkflowRecord &CreateWorkflow(std::string kind, std::string service, std::string trigger, Json input);
        WorkflowRecord *FindWorkflow(std::string_view id);
        const WorkflowRecord *FindWorkflow(std::string_view id) const;
        void SaveWorkflow(const WorkflowRecord &record); // stamps m_Updated, persists, applies the bound
        std::vector<const WorkflowRecord *> Workflows(size_t limit = SIZE_MAX) const; // newest first
        // a running (or waiting-hitl) workflow of `kind` for `service` exists ("" kind = any)
        bool HasRunning(std::string_view kind, std::string_view service) const;

        // inventory reservations keyed by instance id
        void Reserve(const std::string &instance, const std::string &node, const Descriptor::Resources &resources);
        bool Release(const std::string &instance);
        // `base` with every reservation added to its nodes' used amounts (unknown nodes ignored)
        Topology::Inventory ApplyReservations(Topology::Inventory base) const;
        // same, but without the reservations of the listed instances (re-placement of a service)
        Topology::Inventory ApplyReservationsExcept(Topology::Inventory base,
                                                    const std::vector<std::string> &instances) const;
        Json Reservations() const; // [{instance, node, resources}]

        // {deployments[], workflows[] (newest first, ≤ limit), reservations[]}
        Json ToJson(size_t workflowLimit = 32) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Harness
