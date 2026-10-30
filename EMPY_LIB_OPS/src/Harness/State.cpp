#include <eOps/Harness/State.h>
#include <eOps/Descriptor/Parser.h>
#include <eCore/Common/Storage.h>
#include <eCore/Common/Stdout.h>
#include "Detail.h"
#include <stdexcept>
#include <algorithm>

namespace eOps::Harness
{
    namespace
    {
        constexpr std::string_view DEPLOYMENT_KEY = "deployment/";
        constexpr std::string_view WORKFLOW_KEY = "workflow/";
        constexpr std::string_view RESERVATION_KEY = "reservation/";
        constexpr size_t STORE_BYTES = 64u << 20;
        constexpr size_t STORE_RECORDS = 65536;

        bool Active(std::string_view status)
        {
            return status == "running" || status == "waiting-hitl";
        }

        eCore::MMQP::Buffer Encode(const Json &json)
        {
            std::string text = json.Dump();
            return eCore::MMQP::Buffer(text.begin(), text.end());
        }
    } // namespace

    // -------------------------
    // records
    // -------------------------

    Json Instance::ToJson() const
    {
        return Json::MakeObject(
            {{"id", m_Id}, {"role", m_Role}, {"node", m_Node}, {"state", m_State}, {"lastSeen", m_LastSeen}});
    }

    Instance Instance::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("instance must be an object");
        }
        Instance instance;
        instance.m_Id = Detail::ReadString(json, "id");
        instance.m_Role = Detail::ReadString(json, "role");
        instance.m_Node = Detail::ReadString(json, "node");
        instance.m_State = Detail::ReadString(json, "state");
        instance.m_LastSeen = Detail::ReadUnsigned(json, "lastSeen");
        if (instance.m_Id.empty())
        {
            throw JsonError("instance.id must not be empty");
        }
        return instance;
    }

    Json Deployment::ToJson() const
    {
        Json instances = Json::MakeArray();
        for (const Instance &instance : m_Instances)
        {
            instances.Push(instance.ToJson());
        }
        Json artifacts = Json::MakeArray();
        for (const auto &[version, artifact] : m_Artifacts)
        {
            artifacts.Push(artifact.ToJson());
        }
        return Json::MakeObject({{"service", m_Service},
                                 {"version", m_Version},
                                 {"descriptor", Descriptor::ToJson(m_Descriptor)},
                                 {"instances", std::move(instances)},
                                 {"lifecycle", m_Lifecycle.ToJson()},
                                 {"artifacts", std::move(artifacts)},
                                 {"status", m_Status},
                                 {"updated", m_Updated}});
    }

    Deployment Deployment::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("deployment must be an object");
        }
        Deployment deployment;
        deployment.m_Service = Detail::ReadString(json, "service");
        deployment.m_Version = Detail::ReadString(json, "version");
        if (deployment.m_Service.empty())
        {
            throw JsonError("deployment.service must not be empty");
        }
        try
        {
            deployment.m_Descriptor = Descriptor::FromJson(json["descriptor"]);
        }
        catch (const std::exception &error)
        {
            throw JsonError(std::string("deployment.descriptor: ") + error.what());
        }
        if (const Json *instances = json.Find("instances"); instances && !instances->IsNull())
        {
            if (!instances->IsArray())
            {
                throw JsonError("deployment.instances must be an array");
            }
            for (const Json &item : instances->AsArray())
            {
                deployment.m_Instances.push_back(Instance::FromJson(item));
            }
        }
        if (const Json *lifecycle = json.Find("lifecycle"); lifecycle && !lifecycle->IsNull())
        {
            deployment.m_Lifecycle = Agents::LifecycleState::FromJson(*lifecycle);
        }
        if (const Json *artifacts = json.Find("artifacts"); artifacts && !artifacts->IsNull())
        {
            if (!artifacts->IsArray())
            {
                throw JsonError("deployment.artifacts must be an array");
            }
            for (const Json &item : artifacts->AsArray())
            {
                Workload::ModelArtifact artifact = Workload::ModelArtifact::FromJson(item);
                uint32_t version = artifact.m_Version;
                deployment.m_Artifacts[version] = std::move(artifact);
            }
        }
        deployment.m_Status = Detail::ReadString(json, "status");
        deployment.m_Updated = Detail::ReadUnsigned(json, "updated");
        return deployment;
    }

    Json StepRecord::ToJson() const
    {
        return Json::MakeObject({{"name", m_Name},
                                 {"status", m_Status},
                                 {"started", m_Started},
                                 {"finished", m_Finished},
                                 {"detail", m_Detail}});
    }

    StepRecord StepRecord::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("step must be an object");
        }
        StepRecord step;
        step.m_Name = Detail::ReadString(json, "name");
        step.m_Status = Detail::ReadString(json, "status");
        step.m_Started = Detail::ReadUnsigned(json, "started");
        step.m_Finished = Detail::ReadUnsigned(json, "finished");
        step.m_Detail = json["detail"];
        return step;
    }

    bool WorkflowRecord::Finished() const noexcept
    {
        return m_Status == "succeeded" || m_Status == "failed" || m_Status == "rejected";
    }

    Json WorkflowRecord::ToJson() const
    {
        Json steps = Json::MakeArray();
        for (const StepRecord &step : m_Steps)
        {
            steps.Push(step.ToJson());
        }
        return Json::MakeObject({{"id", m_Id},
                                 {"kind", m_Kind},
                                 {"service", m_Service},
                                 {"status", m_Status},
                                 {"trigger", m_Trigger},
                                 {"input", m_Input},
                                 {"result", m_Result},
                                 {"error", m_Error},
                                 {"steps", std::move(steps)},
                                 {"created", m_Created},
                                 {"updated", m_Updated}});
    }

    WorkflowRecord WorkflowRecord::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("workflow must be an object");
        }
        WorkflowRecord record;
        record.m_Id = Detail::ReadString(json, "id");
        record.m_Kind = Detail::ReadString(json, "kind");
        record.m_Service = Detail::ReadString(json, "service");
        record.m_Status = Detail::ReadString(json, "status");
        record.m_Trigger = Detail::ReadString(json, "trigger");
        record.m_Input = json["input"];
        record.m_Result = json["result"];
        record.m_Error = Detail::ReadString(json, "error");
        if (const Json *steps = json.Find("steps"); steps && !steps->IsNull())
        {
            if (!steps->IsArray())
            {
                throw JsonError("workflow.steps must be an array");
            }
            for (const Json &item : steps->AsArray())
            {
                record.m_Steps.push_back(StepRecord::FromJson(item));
            }
        }
        record.m_Created = Detail::ReadUnsigned(json, "created");
        record.m_Updated = Detail::ReadUnsigned(json, "updated");
        if (record.m_Id.empty())
        {
            throw JsonError("workflow.id must not be empty");
        }
        return record;
    }

    // -------------------------
    // StateStore
    // -------------------------

    struct StateStore::Impl
    {
        struct Reservation
        {
            std::string m_Node;
            Descriptor::Resources m_Resources;
        };

        StateOptions m_Options;
        std::unique_ptr<eCore::MMQP::FileStore> m_Store;
        std::map<std::string, Deployment, std::less<>> m_Deployments;
        std::map<std::string, WorkflowRecord, std::less<>> m_Workflows; // node-based: stable pointers
        std::map<std::string, uint64_t, std::less<>> m_Sequence;        // id → creation order
        std::map<uint64_t, std::string> m_Order;                        // creation order → id
        std::map<std::string, Reservation, std::less<>> m_Reservations;
        uint64_t m_NextSequence = 1;

        void Persist(std::string key, const Json &json)
        {
            if (m_Store)
            {
                m_Store->Put(std::move(key), Encode(json));
            }
        }

        void Forget(std::string_view key)
        {
            if (m_Store)
            {
                m_Store->Erase(key);
            }
        }

        static Json ReservationJson(const std::string &instance, const Reservation &reservation)
        {
            return Json::MakeObject({{"instance", instance},
                                     {"node", reservation.m_Node},
                                     {"resources", Detail::ResourcesJson(reservation.m_Resources)}});
        }

        void Track(const std::string &id)
        {
            uint64_t sequence = m_NextSequence++;
            m_Sequence[id] = sequence;
            m_Order[sequence] = id;
        }

        void EraseWorkflow(const std::string &id)
        {
            if (auto found = m_Sequence.find(id); found != m_Sequence.end())
            {
                m_Order.erase(found->second);
                m_Sequence.erase(found);
            }
            m_Workflows.erase(id);
            Forget(std::string(WORKFLOW_KEY) + id);
        }

        // evict the oldest finished records beyond the bound (running ones are kept)
        void Bound()
        {
            size_t finished = 0;
            for (const auto &[id, record] : m_Workflows)
            {
                finished += record.Finished() ? 1 : 0;
            }
            for (auto it = m_Order.begin(); finished > m_Options.m_MaxWorkflows && it != m_Order.end();)
            {
                std::string id = it->second;
                ++it;
                auto record = m_Workflows.find(id);
                if (record != m_Workflows.end() && record->second.Finished())
                {
                    EraseWorkflow(id);
                    --finished;
                }
            }
        }

        void Load()
        {
            std::vector<WorkflowRecord> workflows;
            for (const std::string &key : m_Store->Keys())
            {
                const eCore::MMQP::Buffer *value = m_Store->Find(key);
                if (!value)
                {
                    continue;
                }
                try
                {
                    Json json =
                        Json::Parse(std::string_view(reinterpret_cast<const char *>(value->data()), value->size()));
                    if (key.starts_with(DEPLOYMENT_KEY))
                    {
                        Deployment deployment = Deployment::FromJson(json);
                        if (deployment.m_Service != key.substr(DEPLOYMENT_KEY.size()))
                        {
                            throw JsonError("key does not match the service");
                        }
                        // a retrain that was running when the harness stopped cannot resume
                        deployment.m_Lifecycle.m_Training = false;
                        m_Deployments[deployment.m_Service] = std::move(deployment);
                    }
                    else if (key.starts_with(WORKFLOW_KEY))
                    {
                        WorkflowRecord record = WorkflowRecord::FromJson(json);
                        if (record.m_Id != key.substr(WORKFLOW_KEY.size()))
                        {
                            throw JsonError("key does not match the workflow id");
                        }
                        workflows.push_back(std::move(record));
                    }
                    else if (key.starts_with(RESERVATION_KEY))
                    {
                        std::string instance = Detail::ReadString(json, "instance");
                        if (instance != key.substr(RESERVATION_KEY.size()))
                        {
                            throw JsonError("key does not match the instance");
                        }
                        Reservation reservation{Detail::ReadString(json, "node"),
                                                Detail::ResourcesFrom(json["resources"])};
                        m_Reservations[instance] = std::move(reservation);
                    }
                    else
                    {
                        throw JsonError("unknown record kind");
                    }
                }
                catch (const std::exception &error)
                {
                    throw std::runtime_error("corrupt harness state record '" + key + "': " + error.what());
                }
            }
            std::sort(workflows.begin(), workflows.end(), [](const WorkflowRecord &a, const WorkflowRecord &b)
                      { return a.m_Created != b.m_Created ? a.m_Created < b.m_Created : a.m_Id < b.m_Id; });
            for (WorkflowRecord &record : workflows)
            {
                if (!record.Finished())
                {
                    // its asynchronous steps were lost with the previous harness process
                    std::string previous = record.m_Status.empty() ? std::string("running") : record.m_Status;
                    record.m_Status = "failed";
                    record.m_Error = "harness restarted while the workflow was " + previous;
                    record.m_Updated = NowMillis();
                    Persist(std::string(WORKFLOW_KEY) + record.m_Id, record.ToJson());
                }
                std::string id = record.m_Id;
                m_Workflows[id] = std::move(record);
                Track(id);
            }
            Bound();
        }
    };

    StateStore::StateStore(StateOptions options) : m_Impl(std::make_unique<Impl>())
    {
        m_Impl->m_Options = std::move(options);
        if (!m_Impl->m_Options.m_Directory.empty())
        {
            std::filesystem::create_directories(m_Impl->m_Options.m_Directory);
            m_Impl->m_Store = std::make_unique<eCore::MMQP::FileStore>(m_Impl->m_Options.m_Directory, STORE_BYTES,
                                                                       STORE_RECORDS, true);
            m_Impl->Load();
        }
    }

    StateStore::~StateStore() = default;

    Deployment *StateStore::FindDeployment(std::string_view service)
    {
        auto found = m_Impl->m_Deployments.find(service);
        return found == m_Impl->m_Deployments.end() ? nullptr : &found->second;
    }

    const Deployment *StateStore::FindDeployment(std::string_view service) const
    {
        auto found = m_Impl->m_Deployments.find(service);
        return found == m_Impl->m_Deployments.end() ? nullptr : &found->second;
    }

    void StateStore::PutDeployment(Deployment deployment)
    {
        if (deployment.m_Service.empty())
        {
            throw std::invalid_argument("deployment without a service name");
        }
        deployment.m_Updated = NowMillis();
        std::string key = std::string(DEPLOYMENT_KEY) + deployment.m_Service;
        m_Impl->Persist(key, deployment.ToJson());
        std::string service = deployment.m_Service;
        m_Impl->m_Deployments[service] = std::move(deployment);
    }

    bool StateStore::EraseDeployment(std::string_view service)
    {
        auto found = m_Impl->m_Deployments.find(service);
        if (found == m_Impl->m_Deployments.end())
        {
            return false;
        }
        m_Impl->Forget(std::string(DEPLOYMENT_KEY) + std::string(service));
        m_Impl->m_Deployments.erase(found);
        return true;
    }

    std::vector<const Deployment *> StateStore::Deployments() const
    {
        std::vector<const Deployment *> deployments;
        for (const auto &[service, deployment] : m_Impl->m_Deployments)
        {
            deployments.push_back(&deployment);
        }
        return deployments;
    }

    WorkflowRecord &StateStore::CreateWorkflow(std::string kind, std::string service, std::string trigger, Json input)
    {
        WorkflowRecord record;
        do
        {
            record.m_Id = NewId("wf");
        } while (m_Impl->m_Workflows.count(record.m_Id));
        record.m_Kind = std::move(kind);
        record.m_Service = std::move(service);
        record.m_Status = "running";
        record.m_Trigger = std::move(trigger);
        record.m_Input = std::move(input);
        record.m_Created = NowMillis();
        record.m_Updated = record.m_Created;
        std::string id = record.m_Id;
        m_Impl->Persist(std::string(WORKFLOW_KEY) + id, record.ToJson());
        WorkflowRecord &stored = m_Impl->m_Workflows[id] = std::move(record);
        m_Impl->Track(id);
        return stored;
    }

    WorkflowRecord *StateStore::FindWorkflow(std::string_view id)
    {
        auto found = m_Impl->m_Workflows.find(id);
        return found == m_Impl->m_Workflows.end() ? nullptr : &found->second;
    }

    const WorkflowRecord *StateStore::FindWorkflow(std::string_view id) const
    {
        auto found = m_Impl->m_Workflows.find(id);
        return found == m_Impl->m_Workflows.end() ? nullptr : &found->second;
    }

    void StateStore::SaveWorkflow(const WorkflowRecord &record)
    {
        if (record.m_Id.empty())
        {
            throw std::invalid_argument("workflow record without an id");
        }
        auto found = m_Impl->m_Workflows.find(record.m_Id);
        if (found == m_Impl->m_Workflows.end())
        {
            found = m_Impl->m_Workflows.emplace(record.m_Id, record).first;
            m_Impl->Track(record.m_Id);
        }
        else if (&found->second != &record)
        {
            found->second = record;
        }
        found->second.m_Updated = NowMillis();
        m_Impl->Persist(std::string(WORKFLOW_KEY) + record.m_Id, found->second.ToJson());
        if (found->second.Finished())
        {
            m_Impl->Bound();
        }
    }

    std::vector<const WorkflowRecord *> StateStore::Workflows(size_t limit) const
    {
        std::vector<const WorkflowRecord *> records;
        for (auto it = m_Impl->m_Order.rbegin(); it != m_Impl->m_Order.rend() && records.size() < limit; ++it)
        {
            if (auto found = m_Impl->m_Workflows.find(it->second); found != m_Impl->m_Workflows.end())
            {
                records.push_back(&found->second);
            }
        }
        return records;
    }

    bool StateStore::HasRunning(std::string_view kind, std::string_view service) const
    {
        for (const auto &[id, record] : m_Impl->m_Workflows)
        {
            if (Active(record.m_Status) && record.m_Service == service && (kind.empty() || record.m_Kind == kind))
            {
                return true;
            }
        }
        return false;
    }

    void StateStore::Reserve(const std::string &instance, const std::string &node,
                             const Descriptor::Resources &resources)
    {
        if (instance.empty() || node.empty())
        {
            throw std::invalid_argument("a reservation needs an instance and a node");
        }
        Impl::Reservation reservation{node, resources};
        m_Impl->Persist(std::string(RESERVATION_KEY) + instance, Impl::ReservationJson(instance, reservation));
        m_Impl->m_Reservations[instance] = std::move(reservation);
    }

    bool StateStore::Release(const std::string &instance)
    {
        auto found = m_Impl->m_Reservations.find(instance);
        if (found == m_Impl->m_Reservations.end())
        {
            return false;
        }
        m_Impl->Forget(std::string(RESERVATION_KEY) + instance);
        m_Impl->m_Reservations.erase(found);
        return true;
    }

    Topology::Inventory StateStore::ApplyReservations(Topology::Inventory base) const
    {
        return ApplyReservationsExcept(std::move(base), {});
    }

    Topology::Inventory StateStore::ApplyReservationsExcept(Topology::Inventory base,
                                                            const std::vector<std::string> &instances) const
    {
        for (const auto &[instance, reservation] : m_Impl->m_Reservations)
        {
            if (Detail::Contains(instances, instance))
            {
                continue;
            }
            if (Topology::Node *node = base.Find(reservation.m_Node))
            {
                node->Reserve(reservation.m_Resources);
            }
        }
        return base;
    }

    Json StateStore::Reservations() const
    {
        Json reservations = Json::MakeArray();
        for (const auto &[instance, reservation] : m_Impl->m_Reservations)
        {
            reservations.Push(Impl::ReservationJson(instance, reservation));
        }
        return reservations;
    }

    Json StateStore::ToJson(size_t workflowLimit) const
    {
        Json deployments = Json::MakeArray();
        for (const auto &[service, deployment] : m_Impl->m_Deployments)
        {
            deployments.Push(deployment.ToJson());
        }
        Json workflows = Json::MakeArray();
        for (const WorkflowRecord *record : Workflows(workflowLimit))
        {
            workflows.Push(record->ToJson());
        }
        return Json::MakeObject({{"deployments", std::move(deployments)},
                                 {"workflows", std::move(workflows)},
                                 {"reservations", Reservations()}});
    }
} // namespace eOps::Harness
