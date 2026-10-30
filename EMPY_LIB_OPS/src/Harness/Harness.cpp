#include "Engine.h"
#include "Detail.h"
#include <eOps/Common/Text.h>
#include <eOps/Descriptor/Parser.h>
#include <eCore/Common/Stdout.h>

namespace eOps::Harness
{
    namespace
    {
        constexpr auto TICK = std::chrono::milliseconds(200);
        constexpr auto PUBLISH_INTERVAL = std::chrono::milliseconds(250);
        constexpr auto SYNC_INTERVAL = std::chrono::seconds(5);
        constexpr auto FORCED_SYNC_INTERVAL = std::chrono::seconds(1); // instances a readiness gate waits for
        constexpr auto HEALTH_INTERVAL = std::chrono::seconds(1);
        constexpr auto RERAISE_INTERVAL = std::chrono::seconds(30); // a standing condition is raised again
        constexpr size_t PUBLISHED_WORKFLOWS = 16;
        constexpr size_t MAX_STATE_BYTES = 512 * 1024; // the published snapshot stays well below 1 MiB packets

        const Descriptor::Resources *RoleResources(const Descriptor::RichMLService &service, std::string_view role)
        {
            if (role == "inference" && service.m_Inference)
            {
                return &service.m_Inference->m_Resources;
            }
            if (role == "training" && service.m_Training)
            {
                return &service.m_Training->m_Resources;
            }
            return nullptr;
        }

        // observed accuracy in a drift/metrics body: {accuracy: x} | {accuracy: {value: x}} | drift {kind, value}
        std::optional<double> EventAccuracy(std::string_view kind, const Json &body)
        {
            const Json *value = nullptr;
            if (kind == "drift")
            {
                value = body.GetString("kind") == "accuracy" ? body.Find("value") : nullptr;
            }
            else
            {
                value = body.Find("accuracy");
                if (value && value->IsObject())
                {
                    value = value->Find("value");
                }
            }
            if (!value || !value->IsNumber() || !std::isfinite(value->AsNumber()) || value->AsNumber() < 0 ||
                value->AsNumber() > 1)
            {
                return std::nullopt;
            }
            return value->AsNumber();
        }
    } // namespace

    ExecutionHarness::Impl::Impl(HarnessOptions options, std::shared_ptr<Substrate::ExecutionSubstrate> substrate,
                                 std::shared_ptr<Context::Comas> comas)
        : m_Options(std::move(options)), m_Substrate(std::move(substrate)), m_Comas(std::move(comas)),
          m_State(StateOptions{m_Options.m_StateDirectory, m_Options.m_MaxWorkflows}),
          m_Scheduler([this](const ScheduleEntry &entry, std::chrono::system_clock::time_point firedAt)
                      { FireSchedule(entry, firedAt); })
    {
        if (!m_Substrate)
        {
            throw std::invalid_argument("ExecutionHarness needs an execution substrate");
        }
        if (m_Options.m_Validation.m_TopicRoot.empty())
        {
            m_Options.m_Validation.m_TopicRoot = m_Options.m_Topics.m_Root;
        }
        // persisted deployments resume their temporal triggers
        for (const Deployment *deployment : m_State.Deployments())
        {
            const Descriptor::RichMLService &service = deployment->m_Descriptor;
            try
            {
                if (service.m_Training && !service.m_Training->m_Schedule.empty())
                {
                    m_Scheduler.AddCron(deployment->m_Service, "training", service.m_Training->m_Schedule);
                }
                if (m_Options.m_PlacementAuditSeconds > 0)
                {
                    m_Scheduler.AddInterval(deployment->m_Service, "placement",
                                            std::chrono::seconds(m_Options.m_PlacementAuditSeconds));
                }
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("ExecutionHarness: schedule of {} not restored: {}", deployment->m_Service, error.what());
            }
        }
        // no workflow survives a restart: reservations of instances no deployment holds are leaks
        ReconcileReservations();
    }

    void ExecutionHarness::Impl::ReconcileReservations()
    {
        std::set<std::string> held;
        for (const Deployment *deployment : m_State.Deployments())
        {
            for (const Instance &instance : deployment->m_Instances)
            {
                held.insert(instance.m_Id);
            }
        }
        std::vector<std::string> orphans;
        const Json reservations = m_State.Reservations(); // named: a range over a temporary's array would dangle
        for (const Json &reservation : reservations.AsArray())
        {
            std::string instance = reservation.GetString("instance");
            if (!held.count(instance))
            {
                orphans.push_back(std::move(instance));
            }
        }
        for (const std::string &instance : orphans)
        {
            EMPY_WARN("ExecutionHarness: releasing the reservation of {}, which no deployment holds", instance);
            m_State.Release(instance);
        }
    }

    // -------------------------
    // module glue
    // -------------------------

    void ExecutionHarness::Impl::Attach(eMQ::Service *service, const void *owner)
    {
        m_Service = service;
        m_Bus = &Fabric::BusOf(*service);
        m_Owner = owner;
        // a fresh flag per attachment: work handed out during an earlier attachment stays cancelled
        m_Alive = std::make_shared<bool>(true);
        m_Listing = false;
        ReconcileReservations();
        const Fabric::Topics &topics = m_Options.m_Topics;
        auto require = [](const std::string &what, const eMQ::Reason &reason)
        {
            if (reason)
            {
                throw std::runtime_error("ExecutionHarness: " + what + " failed: " + Detail::ReasonText(reason));
            }
        };
        // one wildcard publication profile per request family keeps the session profile budget small
        require("exposing agent requests", m_Bus->Expose(topics.m_Root + "/agent/+/+", true));
        require("exposing temporal triggers", m_Bus->Expose(topics.m_Root + "/trigger/time/+/+", false));
        require("exposing event triggers", m_Bus->Expose(topics.m_Root + "/event/+/+", false));
        require("exposing HITL requests", m_Bus->Expose(topics.Hitl(), true));
        require("exposing the state topic", m_Bus->Expose(topics.State("services"), false, 1));
        for (const Deployment *deployment : m_State.Deployments())
        {
            require("exposing workload requests", m_Bus->Expose(topics.WorkloadFilter(deployment->m_Service), true));
        }

        Impl *self = this;
        auto serve =
            [&](const std::string &topic, std::function<void(const Fabric::Inbound &, Fabric::Bus::Responder)> fn)
        { require("serving " + topic, m_Bus->Serve(topic, std::move(fn), owner)); };
        auto submit = [self](Json request, std::string trigger, const Json &body, Fabric::Bus::Responder respond)
        {
            if (body.IsObject() && !body.GetBool("wait", true))
            {
                // answer with the running record at once
                std::string id = self->Submit(std::move(request), std::move(trigger), {}, {});
                const WorkflowRecord *record = id.empty() ? nullptr : self->m_State.FindWorkflow(id);
                Json reply = record ? record->ToJson() : self->m_State.Workflows(1).front()->ToJson();
                respond(Fabric::Envelope::Make("workflow", std::move(reply), self->m_Options.m_Identity), {});
                return;
            }
            self->Submit(std::move(request), std::move(trigger), {}, std::move(respond));
        };
        serve(topics.Intent(),
              [submit](const Fabric::Inbound &request, Fabric::Bus::Responder respond)
              {
                  const Json &body = request.m_Envelope.m_Body;
                  Json workflow = Json::MakeObject({{"kind", "deploy"}});
                  if (body.IsString())
                  {
                      workflow.Set("intent", body);
                  }
                  else if (body.IsObject())
                  {
                      for (const char *key : {"intent", "operator", "dryRun"})
                      {
                          if (const Json *value = body.Find(key))
                          {
                              workflow.Set(key, *value);
                          }
                      }
                  }
                  submit(std::move(workflow), "intent", body, std::move(respond));
              });
        serve(topics.Deploy(),
              [submit](const Fabric::Inbound &request, Fabric::Bus::Responder respond)
              {
                  const Json &body = request.m_Envelope.m_Body;
                  Json workflow = Json::MakeObject({{"kind", "deploy"}});
                  if (body.IsString())
                  {
                      workflow.Set("descriptor", body);
                  }
                  else if (body.IsObject())
                  {
                      for (const char *key : {"descriptor", "dryRun", "operator"})
                      {
                          if (const Json *value = body.Find(key))
                          {
                              workflow.Set(key, *value);
                          }
                      }
                  }
                  submit(std::move(workflow), "descriptor", body, std::move(respond));
              });
        serve(topics.Workflow(),
              [submit](const Fabric::Inbound &request, Fabric::Bus::Responder respond)
              {
                  const Json &body = request.m_Envelope.m_Body;
                  std::string trigger = body.IsObject() ? body.GetString("trigger", "operator") : "operator";
                  submit(body, std::move(trigger), body, std::move(respond));
              });
        serve(topics.Status(),
              [self](const Fabric::Inbound &, Fabric::Bus::Responder respond)
              {
                  Json snapshot = self->Snapshot();
                  if (snapshot.Dump().size() > MAX_STATE_BYTES)
                  {
                      snapshot = self->Snapshot(PUBLISHED_WORKFLOWS / 4);
                  }
                  respond(Fabric::Envelope::Make("snapshot", std::move(snapshot), self->m_Options.m_Identity), {});
              });
        require("listening to heartbeats",
                m_Bus->Listen(
                    topics.HeartbeatFilter(), [self](const std::string &topic, const Fabric::Envelope &envelope)
                    { self->OnHeartbeat(topic, envelope); }, 1, owner));
        // the accuracy the active models show on the fabric (judging candidates never trusts a requester's claim)
        for (const char *kind : {"drift", "metrics"})
        {
            require(std::string("listening to ") + kind + " events",
                    m_Bus->Listen(
                        topics.EventFilter(kind), [self](const std::string &topic, const Fabric::Envelope &envelope)
                        { self->OnEvent(topic, envelope); }, 1, owner));
        }

        // timer backup for the scheduler when the runtime's update interval is long; the alive flag keeps a
        // re-armed timer from touching a detached harness
        std::shared_ptr<bool> alive = m_Alive;
        auto tick = std::make_shared<std::function<void()>>();
        *tick = [self, alive, service, weak = std::weak_ptr<std::function<void()>>(tick)]()
        {
            if (!*alive)
            {
                return;
            }
            self->Update();
            if (auto next = weak.lock())
            {
                service->Schedule(TICK, eCore::Task([next] { (*next)(); }));
            }
        };
        m_Tick = tick;
        service->Schedule(TICK, eCore::Task([tick] { (*tick)(); }));
        m_LastConsolidation = Clock::now();
        m_LastHealth = Clock::now();
        Publish();
    }

    void ExecutionHarness::Impl::Update()
    {
        m_Scheduler.Tick();
        CheckReady();
        auto now = Clock::now();
        if (now - m_LastHealth >= HEALTH_INTERVAL)
        {
            m_LastHealth = now;
            try
            {
                CheckHealth();
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("ExecutionHarness: health check failed: {}", error.what());
            }
        }
        if (m_Dirty && now - m_LastPublish >= PUBLISH_INTERVAL)
        {
            Publish();
        }
        if (m_Comas && m_Options.m_ConsolidationSeconds > 0 &&
            now - m_LastConsolidation >= std::chrono::seconds(m_Options.m_ConsolidationSeconds))
        {
            m_LastConsolidation = now;
            try
            {
                size_t groups = m_Comas->Consolidate(NowMillis());
                if (groups > 0)
                {
                    EMPY_DEBUG("ExecutionHarness: consolidated {} memory group(s)", groups);
                }
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("ExecutionHarness: memory consolidation failed: {}", error.what());
            }
        }
    }

    void ExecutionHarness::Impl::Detach()
    {
        *m_Alive = false;
        m_Tick.reset();
        if (m_Bus)
        {
            m_Bus->Remove(m_Owner);
        }
        // workflows cannot continue without the runtime: record them as failed
        for (auto &[id, run] : m_Runs)
        {
            if (WorkflowRecord *record = m_State.FindWorkflow(id); record && !record->Finished())
            {
                record->m_Status = "failed";
                record->m_Error = "harness stopped before the workflow finished";
                try
                {
                    m_State.SaveWorkflow(*record);
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("ExecutionHarness: workflow {} not saved: {}", id, error.what());
                }
            }
        }
        m_Runs.clear();
        m_Bus = nullptr;
    }

    std::string ExecutionHarness::Impl::Submit(Json request, std::string trigger, Completion done,
                                               Fabric::Bus::Responder respond)
    {
        GuardResult check = m_Guard.CheckWorkflowRequest(request, m_State);
        std::string kind = request.IsObject() ? request.GetString("kind") : std::string();
        std::string service = request.IsObject() ? request.GetString("service") : std::string();
        if (!check.m_Ok)
        {
            WorkflowRecord &created =
                m_State.CreateWorkflow(kind.empty() ? "unknown" : kind, service, std::move(trigger), request);
            created.m_Status = "rejected";
            created.m_Error = Join(check.m_Violations, "; ");
            created.m_Result = Json::MakeObject({{"guard", check.ToJson()}});
            m_State.SaveWorkflow(created);
            WorkflowRecord record = created;
            EMPY_INFO("ExecutionHarness: {} request rejected: {}", record.m_Kind, record.m_Error);
            MarkDirty();
            if (respond)
            {
                respond(Fabric::Envelope::Make("workflow", record.ToJson(), m_Options.m_Identity), {});
            }
            if (done)
            {
                done(record);
            }
            return "";
        }
        WorkflowRecord &record = m_State.CreateWorkflow(kind, service, std::move(trigger), std::move(request));
        auto run = std::make_shared<Run>();
        run->m_Id = record.m_Id;
        run->m_Done = std::move(done);
        run->m_Respond = std::move(respond);
        std::string id = record.m_Id;
        m_Runs[id] = run;
        EMPY_INFO("ExecutionHarness: workflow {} ({} {}) started by {}", id, record.m_Kind, record.m_Service,
                  record.m_Trigger);
        MarkDirty();
        Guarded(id, [&] { Start(id); });
        return id;
    }

    Json ExecutionHarness::Impl::Snapshot(size_t workflows) const
    {
        Json deployments = Json::MakeArray();
        for (const Deployment *deployment : m_State.Deployments())
        {
            deployments.Push(deployment->ToJson());
        }
        Json records = Json::MakeArray();
        for (const WorkflowRecord *record : m_State.Workflows(workflows))
        {
            records.Push(record->ToJson());
        }
        Json substrate;
        try
        {
            substrate = m_Substrate->Describe();
        }
        catch (const std::exception &error)
        {
            substrate = Json::MakeObject({{"type", m_Substrate->Name()}, {"error", error.what()}});
        }
        return Json::MakeObject({{"identity", m_Options.m_Identity},
                                 {"deployments", std::move(deployments)},
                                 {"workflows", std::move(records)},
                                 {"schedules", m_Scheduler.ToJson()},
                                 {"inventory", m_State.ApplyReservations(BaseInventory()).ToJson()},
                                 {"reservations", m_State.Reservations()},
                                 {"substrate", std::move(substrate)},
                                 {"time", NowMillis()}});
    }

    void ExecutionHarness::Impl::MarkDirty()
    {
        m_Dirty = true;
    }

    void ExecutionHarness::Impl::Publish()
    {
        m_Dirty = false;
        m_LastPublish = Clock::now();
        Json snapshot = Snapshot(PUBLISHED_WORKFLOWS);
        if (m_Sink)
        {
            try
            {
                m_Sink(snapshot);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("ExecutionHarness: snapshot sink failed: {}", error.what());
            }
        }
        if (!m_Bus)
        {
            return;
        }
        Fabric::Envelope envelope = Fabric::Envelope::Make("state", std::move(snapshot), m_Options.m_Identity);
        if (envelope.m_Body.Dump().size() > MAX_STATE_BYTES)
        {
            envelope.m_Body.Set("workflows", Json::MakeArray());
        }
        // the retained publication is acknowledged on the network strand; nobody waits for it here
        try
        {
            auto published = m_Bus->Publish(m_Options.m_Topics.State("services"), envelope, 1, 1);
            if (published.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            {
                if (eMQ::Reason reason = published.get())
                {
                    EMPY_WARN("ExecutionHarness: state publication rejected: {}", Detail::ReasonText(reason));
                }
            }
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("ExecutionHarness: state publication failed: {}", error.what());
        }
    }

    void ExecutionHarness::Impl::OnHeartbeat(const std::string &topic, const Fabric::Envelope &envelope)
    {
        std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
        if (!levels || levels->size() != 3)
        {
            return;
        }
        Deployment *deployment = m_State.FindDeployment((*levels)[1]);
        if (!deployment)
        {
            return;
        }
        const std::string &instanceId = (*levels)[2];
        const Json &body = envelope.m_Body;
        std::string state = body.IsObject() ? body.GetString("state", "running") : std::string("running");
        bool known = false;
        for (Instance &instance : deployment->m_Instances)
        {
            if (instance.m_Id != instanceId)
            {
                continue;
            }
            known = true;
            if (instance.m_State != state)
            {
                instance.m_State = state;
                MarkDirty();
            }
            instance.m_LastSeen = NowMillis();
            break;
        }
        if (!known)
        {
            return; // a replaced or foreign instance
        }
        Beat &beat = m_Beats[instanceId];
        if (body.IsObject())
        {
            beat.m_Role = body.GetString("role");
            beat.m_Model = body.GetInt("modelVersion", 0);
            beat.m_Samples = body.GetInt("samples", 0);
        }
        // a lost instance that reports again is healthy: a later loss is raised again
        if (state != "stopped")
        {
            m_Raised.erase("failure/" + instanceId);
        }
        // an inference instance that does not serve the active model (restart, relocation) gets it again; one a
        // readiness gate waits for is synchronised at once
        if (body.IsObject() && body.GetString("role") == "inference" && deployment->m_Lifecycle.m_Active &&
            body.GetInt("modelVersion", 0) != static_cast<int64_t>(*deployment->m_Lifecycle.m_Active))
        {
            bool awaited = false;
            for (const auto &[id, run] : m_Runs)
            {
                awaited = awaited || (run->m_WaitingReady && Detail::Contains(run->m_ReadyInstances, instanceId));
            }
            SyncModel(deployment->m_Service, awaited);
        }
        CheckReady();
    }

    void ExecutionHarness::Impl::OnEvent(const std::string &topic, const Fabric::Envelope &envelope)
    {
        std::optional<std::vector<std::string>> levels = m_Options.m_Topics.Relative(topic);
        if (!levels || levels->size() != 3 || !envelope.m_Body.IsObject())
        {
            return;
        }
        const std::string &service = (*levels)[1];
        const Deployment *deployment = m_State.FindDeployment(service);
        if (!deployment || !deployment->m_Lifecycle.m_Active)
        {
            return;
        }
        const Json &body = envelope.m_Body;
        uint32_t active = *deployment->m_Lifecycle.m_Active;
        std::optional<double> accuracy = EventAccuracy((*levels)[2], body);
        if (!accuracy || body.GetInt("modelVersion", 0) != static_cast<int64_t>(active) ||
            body.GetInt("samples", 0) < static_cast<int64_t>(deployment->m_Descriptor.m_Lifecycle.m_MinSamples))
        {
            return;
        }
        m_Observed[service] = Observation{active, *accuracy, NowMillis()};
    }

    std::optional<double> ExecutionHarness::Impl::ObservedAccuracy(const std::string &service) const
    {
        const Deployment *deployment = m_State.FindDeployment(service);
        auto found = m_Observed.find(service);
        if (!deployment || !deployment->m_Lifecycle.m_Active || found == m_Observed.end() ||
            found->second.m_Version != *deployment->m_Lifecycle.m_Active)
        {
            return std::nullopt;
        }
        return found->second.m_Accuracy;
    }

    void ExecutionHarness::Impl::SyncModel(const std::string &service, bool force)
    {
        Deployment *deployment = m_State.FindDeployment(service);
        if (!deployment || !deployment->m_Lifecycle.m_Active || m_State.HasRunning("retrain", service) ||
            m_State.HasRunning("rollback", service) || m_State.HasRunning("retire", service))
        {
            return;
        }
        auto artifact = deployment->m_Artifacts.find(*deployment->m_Lifecycle.m_Active);
        if (artifact == deployment->m_Artifacts.end())
        {
            return;
        }
        auto now = Clock::now();
        if (auto last = m_LastSync.find(service);
            last != m_LastSync.end() && now - last->second < (force ? FORCED_SYNC_INTERVAL : SYNC_INTERVAL))
        {
            return;
        }
        m_LastSync[service] = now;
        size_t expected = 0;
        for (const Instance &instance : deployment->m_Instances)
        {
            expected += instance.m_Role == "inference" ? 1 : 0;
        }
        if (expected == 0)
        {
            return;
        }
        uint32_t version = artifact->first;
        EMPY_INFO("ExecutionHarness: synchronising {} instances with model v{}", service, version);
        m_Bus->Gather(
            m_Options.m_Topics.Workload(service, "model"),
            Fabric::Envelope::Make("model",
                                   Json::MakeObject({{"action", "load"}, {"artifact", artifact->second.ToJson()}}),
                                   m_Options.m_Identity),
            static_cast<uint8_t>(std::min<size_t>(expected, 255)),
            [service, version](eMQ::Reason reason, std::vector<Fabric::Envelope> replies)
            {
                if (reason && replies.empty())
                {
                    EMPY_WARN("ExecutionHarness: model v{} synchronisation of {} failed: {}", version, service,
                              Detail::ReasonText(reason));
                }
            },
            m_Options.m_AgentTimeoutMs, m_Owner);
    }

    bool ExecutionHarness::Impl::InstanceReady(const Deployment &deployment, const Instance &instance,
                                               const Run &run) const
    {
        if (instance.m_LastSeen == 0 || instance.m_State == "lost")
        {
            return false;
        }
        auto beat = m_Beats.find(instance.m_Id);
        if (instance.m_Role == "inference" && deployment.m_Lifecycle.m_Active)
        {
            // serves the active model (the harness loads it on the first heartbeat)
            return beat != m_Beats.end() &&
                   beat->second.m_Model == static_cast<int64_t>(*deployment.m_Lifecycle.m_Active);
        }
        if (instance.m_Role == "training")
        {
            auto required = run.m_ReadySamples.find(instance.m_Id);
            return required == run.m_ReadySamples.end() ||
                   (beat != m_Beats.end() && beat->second.m_Samples >= required->second);
        }
        return true;
    }

    void ExecutionHarness::Impl::CheckReady()
    {
        std::vector<std::string> ready, late;
        auto now = Clock::now();
        for (const auto &[id, run] : m_Runs)
        {
            if (!run->m_WaitingReady)
            {
                continue;
            }
            const WorkflowRecord *record = m_State.FindWorkflow(id);
            const Deployment *deployment = record ? m_State.FindDeployment(record->m_Service) : nullptr;
            bool all = deployment != nullptr;
            for (const std::string &instanceId : run->m_ReadyInstances)
            {
                const Instance *instance = nullptr;
                for (const Instance &item : deployment ? deployment->m_Instances : std::vector<Instance>{})
                {
                    instance = item.m_Id == instanceId ? &item : instance;
                }
                all = all && instance && InstanceReady(*deployment, *instance, *run);
                if (!all)
                {
                    break;
                }
            }
            if (all)
            {
                ready.push_back(id);
            }
            else if (now >= run->m_ReadyDeadline)
            {
                late.push_back(id);
            }
        }
        for (const std::string &id : ready)
        {
            std::shared_ptr<Run> run = Find(id);
            if (!run || !run->m_WaitingReady)
            {
                continue;
            }
            run->m_WaitingReady = false;
            std::function<void()> next = std::move(run->m_OnReady);
            End(id, "succeeded", Json::MakeObject({{"instances", Detail::Strings(run->m_ReadyInstances)}}));
            Guarded(id, [&] { next(); });
        }
        for (const std::string &id : late)
        {
            std::shared_ptr<Run> run = Find(id);
            WorkflowRecord *record = Record(id);
            if (!run || !record || !run->m_WaitingReady)
            {
                continue;
            }
            run->m_WaitingReady = false;
            Json missing = Json::MakeArray();
            if (const Deployment *deployment = m_State.FindDeployment(record->m_Service))
            {
                for (const std::string &instanceId : run->m_ReadyInstances)
                {
                    bool ready = false;
                    for (const Instance &instance : deployment->m_Instances)
                    {
                        ready = ready || (instance.m_Id == instanceId && InstanceReady(*deployment, instance, *run));
                    }
                    if (!ready)
                    {
                        missing.Push(instanceId);
                    }
                }
            }
            std::function<void(const Json &)> next = std::move(run->m_OnLate);
            End(id, "failed", Json::MakeObject({{"missing", missing}}));
            Guarded(id, [&] { next(missing); });
        }
    }

    void ExecutionHarness::Impl::FireSchedule(const ScheduleEntry &entry, std::chrono::system_clock::time_point firedAt)
    {
        if (!m_Bus)
        {
            return;
        }
        uint64_t at = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(firedAt.time_since_epoch()).count());
        Json body = Json::MakeObject({{"service", entry.m_Service},
                                      {"job", entry.m_Job},
                                      {"schedule", entry.m_Expression},
                                      {"firedAt", at},
                                      {"fired", entry.m_Fired}});
        EMPY_DEBUG("ExecutionHarness: temporal trigger {}/{}", entry.m_Service, entry.m_Job);
        (void)m_Bus->Publish(m_Options.m_Topics.Trigger(entry.m_Service, entry.m_Job),
                             Fabric::Envelope::Make("trigger", std::move(body), m_Options.m_Identity), 1);
        MarkDirty();
    }

    // -------------------------
    // health: event-driven triggers the harness derives from its own state
    // -------------------------

    void ExecutionHarness::Impl::Raise(const std::string &service, const Instance &instance, const std::string &kind,
                                       Json detail)
    {
        std::string key = kind + "/" + instance.m_Id;
        auto now = Clock::now();
        if (auto raised = m_Raised.find(key); raised != m_Raised.end() && now - raised->second < RERAISE_INTERVAL)
        {
            return;
        }
        m_Raised[key] = now;
        Json body = Json::MakeObject({{"service", service},
                                      {"instance", instance.m_Id},
                                      {"role", instance.m_Role},
                                      {"node", instance.m_Node},
                                      {"kind", kind}});
        if (detail.IsObject())
        {
            for (const auto &[name, value] : detail.AsObject())
            {
                body.Set(name, value);
            }
        }
        EMPY_WARN("ExecutionHarness: {} of {} ({}): {}", kind, instance.m_Id, service, body.GetString("reason"));
        // the agents re-plan from the retained state: it goes out before the event that activates them
        if (m_Dirty)
        {
            Publish();
        }
        if (m_Comas)
        {
            try
            {
                m_Comas->Observe("harness", kind, body);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("ExecutionHarness: CoMaS observation failed: {}", error.what());
            }
        }
        if (m_Bus)
        {
            (void)m_Bus->Publish(m_Options.m_Topics.Event(service, kind),
                                 Fabric::Envelope::Make("event", std::move(body), m_Options.m_Identity), 1);
        }
    }

    void ExecutionHarness::Impl::CheckHealth()
    {
        uint64_t now = NowMillis();
        std::vector<std::string> services;
        for (const Deployment *deployment : m_State.Deployments())
        {
            services.push_back(deployment->m_Service);
        }
        // liveness: an instance silent for livenessMs after a heartbeat is lost (accelerator or node failure,
        // crashed workload, partitioned session)
        if (m_Options.m_LivenessMs > 0)
        {
            for (const std::string &service : services)
            {
                Deployment *deployment = m_State.FindDeployment(service);
                if (!deployment || deployment->m_Status == "retiring")
                {
                    continue;
                }
                bool changed = false;
                for (Instance &instance : deployment->m_Instances)
                {
                    if (instance.m_State == "lost")
                    {
                        // still lost (a remedy was refused or failed): raised again after RERAISE_INTERVAL
                        Raise(service, instance, "failure",
                              Json::MakeObject({{"reason", "the instance is still lost"}}));
                        continue;
                    }
                    if (instance.m_LastSeen == 0 || now < instance.m_LastSeen + m_Options.m_LivenessMs)
                    {
                        continue;
                    }
                    instance.m_State = "lost";
                    changed = true;
                    Raise(service, instance, "failure",
                          Json::MakeObject(
                              {{"reason", "no heartbeat for " + std::to_string(now - instance.m_LastSeen) +
                                              " ms (liveness " + std::to_string(m_Options.m_LivenessMs) + " ms)"},
                               {"lastSeen", instance.m_LastSeen}}));
                }
                if (changed)
                {
                    m_State.PutDeployment(*deployment);
                    MarkDirty();
                }
            }
        }
        // capacity: reservations beyond a node's registered resources (the resource registry shrank, a node lost
        // its accelerators)
        Topology::Inventory inventory = m_State.ApplyReservations(BaseInventory());
        if (std::string current = inventory.ToJson().Dump(); current != m_Inventory)
        {
            m_Inventory = std::move(current); // the resource registry changed: the agents see it in the state
            MarkDirty();
        }
        std::set<std::string> standing;
        for (const std::string &service : services)
        {
            const Deployment *deployment = m_State.FindDeployment(service);
            for (const Instance &instance : deployment ? deployment->m_Instances : std::vector<Instance>{})
            {
                const Topology::Node *node = inventory.Find(instance.m_Node);
                const Descriptor::Resources *resources = RoleResources(deployment->m_Descriptor, instance.m_Role);
                if (!node || !resources)
                {
                    continue;
                }
                if (node->m_UsedCpu > node->m_Cpu + 1e-9 || node->m_UsedMemory > node->m_Memory)
                {
                    standing.insert("exhaustion/" + instance.m_Id);
                    Raise(service, instance, "exhaustion",
                          Json::MakeObject({{"reason", "node '" + node->m_Id + "' is over-committed"},
                                            {"cpu", node->m_Cpu},
                                            {"usedCpu", node->m_UsedCpu},
                                            {"memory", node->m_Memory},
                                            {"usedMemory", node->m_UsedMemory}}));
                }
                if (!resources->m_Accelerator.empty())
                {
                    auto total = node->m_Accelerators.find(resources->m_Accelerator);
                    auto used = node->m_UsedAccelerators.find(resources->m_Accelerator);
                    uint32_t available = total == node->m_Accelerators.end() ? 0 : total->second;
                    uint32_t reserved = used == node->m_UsedAccelerators.end() ? 0 : used->second;
                    if (reserved > available)
                    {
                        standing.insert("accelerator/" + instance.m_Id);
                        Raise(
                            service, instance, "accelerator",
                            Json::MakeObject({{"reason", "node '" + node->m_Id + "' has " + std::to_string(available) +
                                                             " " + resources->m_Accelerator + " for " +
                                                             std::to_string(reserved) + " reserved"},
                                              {"accelerator", resources->m_Accelerator},
                                              {"available", available},
                                              {"reserved", reserved}}));
                    }
                }
            }
        }
        // a condition that cleared is forgotten, so a recurrence is raised at once
        std::erase_if(m_Raised,
                      [&](const auto &entry)
                      {
                          return (entry.first.starts_with("exhaustion/") || entry.first.starts_with("accelerator/")) &&
                                 !standing.count(entry.first);
                      });
        // substrate states (a crashed EmpyWorkload, a deleted pod) are read on a worker
        if (!m_Listing && m_Service)
        {
            m_Listing = true;
            Impl *self = this;
            std::shared_ptr<bool> alive = m_Alive;
            std::shared_ptr<Substrate::ExecutionSubstrate> substrate = m_Substrate;
            eMQ::Service *service = m_Service;
            try
            {
                (void)m_Service->Async(
                    [self, alive, substrate, service]
                    {
                        auto statuses = std::make_shared<std::vector<Substrate::InstanceStatus>>();
                        try
                        {
                            *statuses = substrate->List();
                        }
                        catch (const std::exception &error)
                        {
                            EMPY_WARN("ExecutionHarness: substrate listing failed: {}", error.what());
                        }
                        service->Dispatch(eCore::Task(
                            [self, alive, statuses]
                            {
                                if (*alive)
                                {
                                    self->m_Listing = false;
                                    self->OnSubstrateList(*statuses);
                                }
                            }));
                    });
            }
            catch (const std::exception &error)
            {
                m_Listing = false;
                EMPY_WARN("ExecutionHarness: substrate listing not scheduled: {}", error.what());
            }
        }
    }

    void ExecutionHarness::Impl::OnSubstrateList(const std::vector<Substrate::InstanceStatus> &statuses)
    {
        for (const Substrate::InstanceStatus &status : statuses)
        {
            if (status.m_State != "exited" && status.m_State != "failed")
            {
                continue;
            }
            Deployment *deployment = m_State.FindDeployment(status.m_Service);
            if (!deployment || deployment->m_Status == "retiring")
            {
                continue;
            }
            for (Instance &instance : deployment->m_Instances)
            {
                if (instance.m_Id != status.m_Instance || instance.m_State == "lost")
                {
                    continue;
                }
                instance.m_State = "lost";
                Instance lost = instance;
                m_State.PutDeployment(*deployment);
                MarkDirty();
                Raise(status.m_Service, lost, "failure",
                      Json::MakeObject({{"reason", "the " + m_Substrate->Name() + " substrate reports the instance " +
                                                       status.m_State}}));
                break;
            }
        }
    }

    // -------------------------
    // step helpers
    // -------------------------

    std::shared_ptr<ExecutionHarness::Impl::Run> ExecutionHarness::Impl::Find(const std::string &id) const
    {
        auto found = m_Runs.find(id);
        return found == m_Runs.end() ? nullptr : found->second;
    }

    WorkflowRecord *ExecutionHarness::Impl::Record(const std::string &id)
    {
        return m_Runs.count(id) ? m_State.FindWorkflow(id) : nullptr;
    }

    void ExecutionHarness::Impl::Begin(const std::string &id, std::string step)
    {
        WorkflowRecord *record = Record(id);
        if (!record)
        {
            return;
        }
        StepRecord entry;
        entry.m_Name = std::move(step);
        entry.m_Status = "running";
        entry.m_Started = NowMillis();
        record->m_Steps.push_back(std::move(entry));
        m_State.SaveWorkflow(*record);
        MarkDirty();
    }

    void ExecutionHarness::Impl::End(const std::string &id, std::string status, Json detail)
    {
        WorkflowRecord *record = Record(id);
        if (!record)
        {
            return;
        }
        for (auto step = record->m_Steps.rbegin(); step != record->m_Steps.rend(); ++step)
        {
            if (step->m_Status == "running")
            {
                step->m_Status = std::move(status);
                step->m_Finished = NowMillis();
                step->m_Detail = std::move(detail);
                break;
            }
        }
        m_State.SaveWorkflow(*record);
        MarkDirty();
    }

    void ExecutionHarness::Impl::Finish(const std::string &id, std::string status, Json result, std::string error)
    {
        std::shared_ptr<Run> run = Find(id);
        WorkflowRecord *stored = m_State.FindWorkflow(id);
        if (!run || !stored || stored->Finished())
        {
            return;
        }
        m_Runs.erase(id);
        for (StepRecord &step : stored->m_Steps)
        {
            if (step.m_Status == "running")
            {
                step.m_Status = status == "succeeded" ? "succeeded" : "failed";
                step.m_Finished = NowMillis();
            }
        }
        stored->m_Status = std::move(status);
        stored->m_Result = std::move(result);
        stored->m_Error = std::move(error);
        if (stored->m_Kind == "retrain")
        {
            try
            {
                ClearTraining(stored->m_Service);
            }
            catch (const std::exception &failure)
            {
                EMPY_ERROR("ExecutionHarness: training flag of {} not cleared: {}", stored->m_Service, failure.what());
            }
        }
        try
        {
            m_State.SaveWorkflow(*stored);
        }
        catch (const std::exception &failure)
        {
            EMPY_ERROR("ExecutionHarness: workflow {} not persisted: {}", id, failure.what());
        }
        WorkflowRecord record = *stored;
        if (record.m_Status == "succeeded")
        {
            EMPY_INFO("ExecutionHarness: workflow {} ({} {}) succeeded", id, record.m_Kind, record.m_Service);
        }
        else
        {
            EMPY_INFO("ExecutionHarness: workflow {} ({} {}) {}: {}", id, record.m_Kind, record.m_Service,
                      record.m_Status, record.m_Error);
        }
        Remember(record);
        Publish();
        if (run->m_Respond)
        {
            run->m_Respond(Fabric::Envelope::Make("workflow", record.ToJson(), m_Options.m_Identity), {});
        }
        if (run->m_Done)
        {
            try
            {
                run->m_Done(record);
            }
            catch (const std::exception &failure)
            {
                EMPY_WARN("ExecutionHarness: completion of workflow {} failed: {}", id, failure.what());
            }
        }
    }

    void ExecutionHarness::Impl::Guarded(const std::string &id, const std::function<void()> &fn)
    {
        try
        {
            fn();
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("ExecutionHarness: workflow {} step failed: {}", id, error.what());
            try
            {
                Finish(id, "failed", Json(), error.what());
            }
            catch (const std::exception &again)
            {
                EMPY_ERROR("ExecutionHarness: workflow {} could not be finished: {}", id, again.what());
            }
        }
    }

    void ExecutionHarness::Impl::Ask(const std::string &id, const std::string &role, const std::string &task, Json body,
                                     Continue next)
    {
        Impl *self = this;
        std::string label = role + "/" + task;
        m_Bus->Call(
            m_Options.m_Topics.Agent(role, task),
            Fabric::Envelope::Make("agent-request", std::move(body), m_Options.m_Identity),
            [self, id, label, next = std::move(next)](eMQ::Reason reason, std::optional<Fabric::Envelope> reply)
            {
                if (!self->Find(id))
                {
                    return;
                }
                self->Guarded(
                    id,
                    [&]
                    {
                        if (reason || !reply)
                        {
                            std::string text = reason ? Detail::ReasonText(reason) : "empty agent response";
                            self->End(id, "failed",
                                      Json::MakeObject({{"agent", label}, {"reason", reason.m_Code}, {"error", text}}));
                            self->Finish(id, "failed", Json(), "agent " + label + ": " + text);
                            return;
                        }
                        next(reply->m_Body["result"]);
                    });
            },
            m_Options.m_AgentTimeoutMs, m_Owner);
    }

    void ExecutionHarness::Impl::AskHuman(const std::string &id, Json body,
                                          std::function<void(const Detail::HitlAnswer &)> next)
    {
        WorkflowRecord *record = Record(id);
        record->m_Status = "waiting-hitl";
        m_State.SaveWorkflow(*record);
        body.Set("workflowId", id);
        body.Set("default", m_Options.m_HitlDefaultApprove ? "approve" : "reject");
        body.Set("timeoutMs", m_Options.m_HitlTimeoutMs);
        Begin(id, "hitl");
        Publish();
        Impl *self = this;
        m_Bus->Call(
            m_Options.m_Topics.Hitl(), Fabric::Envelope::Make("hitl", std::move(body), m_Options.m_Identity),
            [self, id, next = std::move(next)](eMQ::Reason reason, std::optional<Fabric::Envelope> reply)
            {
                if (!self->Find(id))
                {
                    return;
                }
                self->Guarded(id,
                              [&]
                              {
                                  Detail::HitlAnswer answer;
                                  std::string decision;
                                  answer.m_Source = "operator";
                                  if (!reason && reply && reply->m_Body.IsObject())
                                  {
                                      decision = Lower(reply->m_Body.GetString("decision"));
                                      answer.m_Operator = reply->m_Body.GetString("operator", reply->m_Source);
                                      answer.m_Note = reply->m_Body.GetString("note");
                                  }
                                  if (decision == "approved" || decision == "approve" || decision == "yes")
                                  {
                                      answer.m_Approve = true;
                                  }
                                  else if (decision == "rejected" || decision == "reject" || decision == "deny" ||
                                           decision == "no")
                                  {
                                      answer.m_Approve = false;
                                  }
                                  else
                                  {
                                      // timeout, no responder or an unusable answer → configured default
                                      answer.m_Note = reason
                                                          ? reason.m_Text
                                                          : (reply ? "unrecognised HITL decision" : "no HITL answer");
                                      answer.m_Approve = self->m_Options.m_HitlDefaultApprove;
                                      answer.m_Source = "default";
                                      answer.m_Operator.clear();
                                  }
                                  WorkflowRecord *record = self->Record(id);
                                  record->m_Status = "running";
                                  self->m_State.SaveWorkflow(*record);
                                  next(answer);
                              });
            },
            m_Options.m_HitlTimeoutMs, m_Owner);
    }

    void ExecutionHarness::Impl::Enact(const std::string &id, std::function<Json()> work, Continue next)
    {
        Impl *self = this;
        std::shared_ptr<bool> alive = m_Alive;
        eMQ::Service *service = m_Service;
        auto task = [self, alive, service, id, work = std::move(work), next = std::move(next)]() mutable
        {
            auto outcome = std::make_shared<Json>();
            auto error = std::make_shared<std::string>();
            try
            {
                *outcome = work();
            }
            catch (const std::exception &failure)
            {
                *error = failure.what();
            }
            service->Dispatch(eCore::Task(
                [self, alive, id, outcome, error, next = std::move(next)]
                {
                    if (!*alive || !self->Find(id))
                    {
                        return;
                    }
                    self->Guarded(id,
                                  [&]
                                  {
                                      if (!error->empty())
                                      {
                                          throw std::runtime_error("substrate: " + *error);
                                      }
                                      next(*outcome);
                                  });
                }));
        };
        (void)m_Service->Async(std::move(task));
    }

    Topology::Inventory ExecutionHarness::Impl::BaseInventory() const
    {
        return m_Comas ? m_Comas->Plane().Inventory() : Topology::Inventory{};
    }

    Topology::NetworkModel ExecutionHarness::Impl::Network() const
    {
        return m_Comas ? m_Comas->Plane().Network() : Topology::NetworkModel{};
    }

    void ExecutionHarness::Impl::Remember(const WorkflowRecord &record)
    {
        if (!m_Comas)
        {
            return;
        }
        try
        {
            m_Comas->Remember("harness", record.m_Kind, record.m_Service,
                              Json::MakeObject({{"workflowId", record.m_Id},
                                                {"trigger", record.m_Trigger},
                                                {"status", record.m_Status},
                                                {"error", record.m_Error},
                                                {"steps", record.m_Steps.size()},
                                                {"outcome", record.m_Status}}));
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("ExecutionHarness: CoMaS memory write failed: {}", error.what());
        }
    }

    // -------------------------
    // ExecutionHarness
    // -------------------------

    ExecutionHarness::ExecutionHarness(HarnessOptions options, std::shared_ptr<Substrate::ExecutionSubstrate> substrate,
                                       std::shared_ptr<Context::Comas> comas)
        : m_Impl(std::make_unique<Impl>(std::move(options), std::move(substrate), std::move(comas)))
    {
    }

    ExecutionHarness::~ExecutionHarness() = default;

    std::string ExecutionHarness::Submit(Json request, std::string trigger, Completion done)
    {
        if (!m_Impl->m_Bus)
        {
            throw std::logic_error("ExecutionHarness is not attached");
        }
        return m_Impl->Submit(std::move(request), std::move(trigger), std::move(done), {});
    }

    Json ExecutionHarness::Snapshot() const
    {
        return m_Impl->Snapshot();
    }

    const StateStore &ExecutionHarness::State() const
    {
        return m_Impl->m_State;
    }

    const Scheduler &ExecutionHarness::Schedules() const
    {
        return m_Impl->m_Scheduler;
    }

    void ExecutionHarness::SetSnapshotSink(std::function<void(const Json &)> sink)
    {
        m_Impl->m_Sink = std::move(sink);
        if (m_Impl->m_Sink)
        {
            m_Impl->m_Sink(m_Impl->Snapshot(PUBLISHED_WORKFLOWS));
        }
    }

    void ExecutionHarness::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        try
        {
            m_Impl->Attach(service, this);
        }
        catch (...)
        {
            m_Impl->Detach();
            throw;
        }
    }

    void ExecutionHarness::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->Update();
    }

    void ExecutionHarness::OnDetach(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->Detach();
    }
} // namespace eOps::Harness
