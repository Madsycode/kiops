#include "RuntimeDetail.h"
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/Stdout.h>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <map>

namespace eOps::Workload
{
    using eCore::MMQP::MMQP_REASON_MALFORMED_PACKET;
    using eCore::MMQP::MMQP_REASON_POLICY_VIOLATION;
    using eCore::MMQP::MMQP_REASON_UNSPECIFIED;

    namespace
    {
        using Clock = std::chrono::steady_clock;

        // latency samples kept per window (bounded memory at high action rates)
        constexpr size_t MAX_LATENCY_SAMPLES = 4096;
        // the latency statistic sorts its window, so the objective is evaluated at most this often
        constexpr auto LATENCY_CHECK_PERIOD = std::chrono::milliseconds(100);
        constexpr auto STOP_ACK_TIMEOUT = std::chrono::milliseconds(1000); // final heartbeat acknowledgement
    } // namespace

    struct InferenceRuntime::Impl
    {
        WorkloadOptions m_Options;
        std::vector<std::string> m_Features;
        std::vector<std::optional<double>> m_Values; // latest value per observable, declaration order
        std::string m_Label;
        std::string m_Actuation;
        std::string m_Unit;
        uint8_t m_ActionQos = 1;
        bool m_Declared = false; // the actuation topic is a concrete declared actionable
        ActionBoundary m_Boundary;
        std::string m_Statistic = "p95";
        uint32_t m_WindowSeconds = 60;
        double m_MaxMs = 0; // 0 = no latency objective
        mutable WindowStats m_Latency;
        std::deque<bool> m_Hits; // accuracy window: prediction within tolerance of the next label
        size_t m_Window = 64;
        std::optional<ModelArtifact> m_Active;
        std::optional<ModelArtifact> m_Previous;
        std::optional<double> m_Prediction; // latest prediction not yet paired with a label
        uint64_t m_Predictions = 0;
        uint64_t m_Actions = 0;
        uint64_t m_Rejected = 0;
        uint64_t m_Clamped = 0;
        uint64_t m_Ignored = 0; // unparsable data-plane payloads
        uint64_t m_Drifts = 0;
        std::map<std::string, Clock::time_point> m_LastDrift;     // per drift kind (debounce)
        std::map<std::string, Clock::time_point> m_LastViolation; // per boundary rule (debounce)
        std::string m_LastRejection;
        Clock::time_point m_NextLatencyCheck{};
        eMQ::Service *m_Service = nullptr;
        const void *m_Owner = nullptr;
        Detail::Ticker m_Heartbeat;
        Detail::Ticker m_Metrics;
        Detail::Outbox m_Outbox;
        bool m_Stopped = false;

        static WindowStats MakeLatency(const WorkloadOptions &options)
        {
            const auto &inference = options.m_Service.m_Inference;
            uint32_t window = inference && inference->m_Latency ? inference->m_Latency->m_WindowSeconds : 60;
            if (!window)
            {
                throw std::invalid_argument("Inference latency window must be at least 1 s");
            }
            return WindowStats(window, MAX_LATENCY_SAMPLES);
        }

        explicit Impl(WorkloadOptions options)
            : m_Options(std::move(options)), m_Boundary(m_Options.m_Service), m_Latency(MakeLatency(m_Options))
        {
            const auto &service = m_Options.m_Service;
            if (!service.m_Inference)
            {
                throw std::invalid_argument("Inference runtime needs a descriptor with an inference section");
            }
            Detail::Resolve(m_Options, "inference");
            m_Features = FeatureNames(service);
            m_Values.resize(m_Features.size());
            if (service.m_Training && !service.m_Training->m_Label.empty())
            {
                m_Label = service.m_Training->m_Label;
                if (!ValidTopic(m_Label, false) || IsFilter(m_Label))
                {
                    throw std::invalid_argument("training.label must be a concrete topic: " + m_Label);
                }
            }
            m_Actuation = service.ActuationTopic();
            if (const auto *declared = m_Actuation.empty() ? nullptr : service.FindActionable(m_Actuation))
            {
                m_Unit = declared->m_Unit;
                m_ActionQos = declared->m_Qos > 1 ? 1 : declared->m_Qos;
                m_Declared = ValidTopic(m_Actuation, false) && !IsFilter(m_Actuation);
            }
            if (const auto &latency = service.m_Inference->m_Latency)
            {
                if (!WindowStats::KnownStatistic(latency->m_Statistic))
                {
                    throw std::invalid_argument("Unknown latency statistic: " + latency->m_Statistic);
                }
                m_Statistic = latency->m_Statistic;
                m_WindowSeconds = latency->m_WindowSeconds;
                m_MaxMs = latency->m_MaxMs > 0 ? latency->m_MaxMs : 0;
            }
            m_Window = std::max<size_t>(1, service.m_Lifecycle.m_EvaluationWindow);
            if (m_Options.m_InitialModel)
            {
                auto problem = CheckArtifact(*m_Options.m_InitialModel, service);
                if (!problem.empty())
                {
                    throw std::invalid_argument("Initial model rejected: " + problem);
                }
                m_Active = m_Options.m_InitialModel;
            }
        }

        const std::string &Name() const
        {
            return m_Options.m_Service.m_Metadata.m_Name;
        }

        std::optional<double> Accuracy() const
        {
            if (m_Hits.empty())
            {
                return std::nullopt;
            }
            size_t hits = static_cast<size_t>(std::count(m_Hits.begin(), m_Hits.end(), true));
            return static_cast<double>(hits) / static_cast<double>(m_Hits.size());
        }

        uint32_t Version() const
        {
            return m_Active ? m_Active->m_Version : 0;
        }

        Json Latency() const
        {
            Json latency = Json::Object{{"statistic", m_Statistic},
                                        {"value", Detail::Number(m_Latency.Statistic(m_Statistic))},
                                        {"windowSeconds", m_WindowSeconds},
                                        {"samples", m_Latency.Count()}};
            if (m_MaxMs > 0)
            {
                latency.Set("maxMs", m_MaxMs);
            }
            return latency;
        }

        Json Status() const
        {
            const char *state = m_Stopped ? "stopped" : m_Active ? "ready" : "waiting-model";
            return Json::Object{{"service", Name()},
                                {"instance", m_Options.m_Instance},
                                {"role", "inference"},
                                {"node", m_Options.m_Node},
                                {"region", m_Options.m_Region},
                                {"state", state},
                                {"modelVersion", this->Version()},
                                {"artifact", m_Active ? Json(m_Active->m_Id) : Json()},
                                {"previousVersion", m_Previous ? Json(m_Previous->m_Version) : Json()},
                                {"accuracy", Detail::Number(this->Accuracy())},
                                {"samples", m_Hits.size()},
                                {"latency", this->Latency()},
                                {"actuation", m_Actuation},
                                {"predictions", m_Predictions},
                                {"actions", m_Actions},
                                {"rejected", m_Rejected},
                                {"clamped", m_Clamped},
                                {"drifts", m_Drifts},
                                {"publishFailures", m_Outbox.Failed()},
                                {"time", NowMillis()}};
        }

        // -- control plane

        void PublishHeartbeat()
        {
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            auto topic = m_Options.m_Topics.Heartbeat(this->Name(), m_Options.m_Instance);
            auto message = Fabric::Envelope::Make("status", this->Status());
            m_Outbox.Send(topic, [&] { return bus->Publish(topic, message, 1, 1); });
        }

        void PublishEvent(std::string_view kind, Json body)
        {
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            auto topic = m_Options.m_Topics.Event(this->Name(), kind);
            auto message = Fabric::Envelope::Make("event", std::move(body));
            m_Outbox.Send(topic, [&] { return bus->Publish(topic, message, 1); });
        }

        void PublishMetrics()
        {
            m_Outbox.Harvest();
            this->CheckLatency();
            this->PublishEvent("metrics", Json::Object{{"service", this->Name()},
                                                       {"instance", m_Options.m_Instance},
                                                       {"node", m_Options.m_Node},
                                                       {"region", m_Options.m_Region},
                                                       {"accuracy", Detail::Number(this->Accuracy())},
                                                       {"samples", m_Hits.size()},
                                                       {"latency", this->Latency()},
                                                       {"actions", m_Actions},
                                                       {"rejected", m_Rejected},
                                                       {"clamped", m_Clamped},
                                                       {"predictions", m_Predictions},
                                                       {"modelVersion", this->Version()}});
        }

        // debounced drift event of `kind` (accuracy | latency)
        void Drift(const std::string &kind, double value, double threshold, size_t samples)
        {
            const auto now = Clock::now();
            auto last = m_LastDrift.find(kind);
            if (last != m_LastDrift.end() &&
                now - last->second < std::chrono::milliseconds(m_Options.m_DriftDebounceMs))
            {
                return;
            }
            m_LastDrift[kind] = now;
            ++m_Drifts;
            EMPY_INFO("Workload {}: {} drift {} vs {} over {} samples (model v{})", m_Options.m_Instance, kind, value,
                      threshold, samples, this->Version());
            this->PublishEvent("drift", Json::Object{{"service", this->Name()},
                                                     {"instance", m_Options.m_Instance},
                                                     {"node", m_Options.m_Node},
                                                     {"region", m_Options.m_Region},
                                                     {"kind", kind},
                                                     {"value", value},
                                                     {"threshold", threshold},
                                                     {"modelVersion", this->Version()},
                                                     {"samples", samples}});
        }

        // policy violation event, debounced per rule: an action the execution boundary rejected (undeclared topic,
        // non-finite value, rate limit)
        void Violation(const ActionDecision &decision, double value)
        {
            const auto now = Clock::now();
            auto last = m_LastViolation.find(decision.m_Rule);
            if (last != m_LastViolation.end() &&
                now - last->second < std::chrono::milliseconds(m_Options.m_DriftDebounceMs))
            {
                return;
            }
            m_LastViolation[decision.m_Rule] = now;
            this->PublishEvent("violation", Json::Object{{"service", this->Name()},
                                                         {"instance", m_Options.m_Instance},
                                                         {"node", m_Options.m_Node},
                                                         {"region", m_Options.m_Region},
                                                         {"topic", m_Actuation},
                                                         {"value", std::isfinite(value) ? Json(value) : Json()},
                                                         {"reason", decision.m_Reason},
                                                         {"rule", decision.m_Rule},
                                                         {"rejected", m_Rejected},
                                                         {"modelVersion", this->Version()}});
        }

        void CheckAccuracy()
        {
            const auto &policy = m_Options.m_Service.m_Lifecycle;
            auto accuracy = this->Accuracy();
            if (m_Active && accuracy && m_Hits.size() >= policy.m_MinSamples && *accuracy < policy.m_AccuracyThreshold)
            {
                this->Drift("accuracy", *accuracy, policy.m_AccuracyThreshold, m_Hits.size());
            }
        }

        void CheckLatency()
        {
            const auto now = Clock::now();
            if (!m_Active || m_MaxMs <= 0 || now < m_NextLatencyCheck)
            {
                return;
            }
            m_NextLatencyCheck = now + LATENCY_CHECK_PERIOD;
            size_t count = m_Latency.Count();
            auto value = m_Latency.Statistic(m_Statistic);
            if (value && count >= m_Options.m_Service.m_Lifecycle.m_MinSamples && *value > m_MaxMs)
            {
                this->Drift("latency", *value, m_MaxMs, count);
            }
        }

        // -- data plane

        void OnObservation(size_t index, const eMQ::PublishRequest &message)
        {
            const auto received = Clock::now();
            if (message.m_Topic == m_Label)
            {
                return; // ground truth is never a feature
            }
            auto value = Fabric::ParseValue(message.m_Payload);
            if (!value)
            {
                ++m_Ignored;
                return;
            }
            m_Values[index] = *value;
            this->Infer(received);
        }

        void Infer(Clock::time_point received)
        {
            if (!m_Active)
            {
                return;
            }
            std::vector<double> features;
            features.reserve(m_Values.size());
            for (const auto &value : m_Values)
            {
                if (!value)
                {
                    return; // not every feature observed yet
                }
                features.push_back(*value);
            }
            double prediction = LinearModel::Predict(*m_Active, features);
            ++m_Predictions;
            m_Prediction = prediction;
            if (m_Actuation.empty())
            {
                return; // prediction-only service
            }
            // formal execution boundary: declared topic, bounds and rate (never trust the model)
            auto decision = m_Boundary.Check(m_Actuation, prediction);
            if (!decision.m_Allowed)
            {
                if (m_Rejected++ % 1000 == 0 || decision.m_Reason != m_LastRejection)
                {
                    EMPY_WARN("Workload {}: action rejected: {}", m_Options.m_Instance, decision.m_Reason);
                }
                m_LastRejection = decision.m_Reason;
                this->Violation(decision, prediction);
                return;
            }
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            if (decision.m_Clamped)
            {
                ++m_Clamped;
            }
            auto payload =
                Fabric::EncodeAction(decision.m_Value, m_Unit, this->Name(), m_Options.m_Instance, this->Version());
            m_Outbox.Send(m_Actuation, [&] { return bus->PublishRaw(m_Actuation, std::move(payload), m_ActionQos); });
            ++m_Actions;
            // observation-to-action latency: receive → publish admission
            m_Latency.Add(std::chrono::duration<double, std::milli>(Clock::now() - received).count());
            this->CheckLatency();
        }

        void OnLabel(const eMQ::PublishRequest &message)
        {
            auto label = Fabric::ParseValue(message.m_Payload);
            if (!label)
            {
                ++m_Ignored;
                return;
            }
            if (!m_Active || !m_Prediction)
            {
                return;
            }
            const auto &policy = m_Options.m_Service.m_Lifecycle;
            m_Hits.push_back(std::fabs(*m_Prediction - *label) <= policy.m_Tolerance);
            m_Prediction.reset();
            while (m_Hits.size() > m_Window)
            {
                m_Hits.pop_front();
            }
            this->CheckAccuracy();
        }

        // -- model management

        void Reset()
        {
            m_Hits.clear();
            m_Prediction.reset();
            m_LastDrift.clear();
        }

        void Activate(ModelArtifact artifact)
        {
            if (m_Active && m_Active->m_Version != artifact.m_Version)
            {
                m_Previous = std::move(m_Active);
            }
            m_Active = std::move(artifact);
            this->Reset();
            EMPY_INFO("Workload {}: model v{} active", m_Options.m_Instance, m_Active->m_Version);
            this->PublishHeartbeat();
        }

        Json Loaded(std::string_view action) const
        {
            return Json::Object{{"service", this->Name()},
                                {"instance", m_Options.m_Instance},
                                {"action", action},
                                {"loaded", this->Version()},
                                {"artifact", m_Active ? Json(m_Active->m_Id) : Json()},
                                {"previous", m_Previous ? Json(m_Previous->m_Version) : Json()}};
        }

        void OnModel(const Fabric::Inbound &inbound, Fabric::Bus::Responder respond)
        {
            const Json &body = inbound.m_Envelope.m_Body;
            const Json *action = body.IsObject() ? body.Find("action") : nullptr;
            if (!action || !action->IsString())
            {
                return respond({}, {MMQP_REASON_MALFORMED_PACKET, "Model request needs a string 'action'"});
            }
            const std::string &verb = action->AsString();
            if (verb == "load")
            {
                const Json *json = body.Find("artifact");
                if (!json || !json->IsObject())
                {
                    return respond({}, {MMQP_REASON_MALFORMED_PACKET, "Model load needs an 'artifact' object"});
                }
                ModelArtifact artifact;
                try
                {
                    artifact = ModelArtifact::FromJson(*json);
                }
                catch (const std::exception &error)
                {
                    return respond({},
                                   {MMQP_REASON_MALFORMED_PACKET, std::string("Invalid artifact: ") + error.what()});
                }
                auto problem = CheckArtifact(artifact, m_Options.m_Service);
                if (!problem.empty())
                {
                    EMPY_WARN("Workload {}: artifact {} rejected: {}", m_Options.m_Instance, artifact.m_Id, problem);
                    return respond({}, {MMQP_REASON_POLICY_VIOLATION, "Artifact rejected: " + problem});
                }
                this->Activate(std::move(artifact));
                return respond(Fabric::Envelope::Make("model", this->Loaded("load")), {});
            }
            if (verb == "rollback")
            {
                if (!m_Previous)
                {
                    return respond({}, {MMQP_REASON_UNSPECIFIED, "No previous model to roll back to"});
                }
                const Json *target = body.Find("target");
                if (target && !target->IsNull() &&
                    (!target->IsInteger() || target->AsInt() != static_cast<int64_t>(m_Previous->m_Version)))
                {
                    return respond({}, {MMQP_REASON_UNSPECIFIED, "Rollback target is not the previous model (v" +
                                                                     std::to_string(m_Previous->m_Version) + ")"});
                }
                std::swap(m_Active, m_Previous);
                this->Reset();
                EMPY_INFO("Workload {}: rolled back to model v{}", m_Options.m_Instance, m_Active->m_Version);
                this->PublishHeartbeat();
                Json result = this->Loaded("rollback");
                result.Set("rolledBack", true);
                return respond(Fabric::Envelope::Make("model", std::move(result)), {});
            }
            if (verb == "status")
            {
                return respond(Fabric::Envelope::Make("status", this->Status()), {});
            }
            respond({}, {MMQP_REASON_MALFORMED_PACKET, "Unknown model action '" + verb + "'"});
        }

        // -- lifecycle

        void Attach(eMQ::Service *service, const void *owner)
        {
            m_Service = service;
            m_Owner = owner;
            m_Stopped = false;
            auto &bus = Fabric::BusOf(*service);
            const auto &topics = m_Options.m_Topics;
            const auto &name = this->Name();
            if (m_Declared)
            {
                Detail::Require(bus.Expose(m_Actuation), "Publication profile " + m_Actuation);
            }
            else if (!m_Actuation.empty())
            {
                EMPY_WARN("Workload {}: actuation topic '{}' is not a concrete declared actionable; every action "
                          "will be rejected",
                          m_Options.m_Instance, m_Actuation);
            }
            Detail::Require(bus.Expose(topics.Heartbeat(name, m_Options.m_Instance), false, 1), "Heartbeat profile");
            Detail::Require(bus.Expose(topics.Event(name, "drift")), "Drift event profile");
            Detail::Require(bus.Expose(topics.Event(name, "metrics")), "Metrics event profile");
            Detail::Require(bus.Expose(topics.Event(name, "violation")), "Violation event profile");
            Detail::Require(bus.Serve(
                                topics.Workload(name, "model"),
                                [this](const Fabric::Inbound &inbound, Fabric::Bus::Responder respond)
                                { this->OnModel(inbound, std::move(respond)); }, owner),
                            "Model responder");
            const auto &observables = m_Options.m_Service.m_Observables;
            for (size_t i = 0; i < observables.size(); ++i)
            {
                Detail::Require(bus.ListenRaw(
                                    observables[i].m_Name, [this, i](const eMQ::PublishRequest &message)
                                    { this->OnObservation(i, message); }, observables[i].m_Qos, owner),
                                "Observable subscription " + observables[i].m_Name);
            }
            if (!m_Label.empty())
            {
                Detail::Require(
                    bus.ListenRaw(
                        m_Label, [this](const eMQ::PublishRequest &message) { this->OnLabel(message); }, 1, owner),
                    "Label subscription " + m_Label);
            }
            m_Heartbeat.Start(*service, m_Options.m_HeartbeatMs, [this] { this->PublishHeartbeat(); });
            m_Metrics.Start(*service, m_Options.m_MetricsMs, [this] { this->PublishMetrics(); });
            EMPY_INFO("Workload {}: inference ready for {} (model v{})", m_Options.m_Instance, name, this->Version());
            this->PublishHeartbeat();
        }

        void Detach(eMQ::Service *service)
        {
            m_Heartbeat.Stop();
            m_Metrics.Stop();
            m_Stopped = true;
            if (auto *bus = Detail::FindBus(service))
            {
                this->PublishHeartbeat(); // retained "stopped" state, acknowledged before the module goes
                m_Outbox.Drain(STOP_ACK_TIMEOUT);
                bus->Remove(m_Owner);
            }
            m_Service = nullptr;
        }
    };

    InferenceRuntime::InferenceRuntime(WorkloadOptions options) : m_Impl(std::make_unique<Impl>(std::move(options)))
    {
    }

    InferenceRuntime::~InferenceRuntime() = default;

    Json InferenceRuntime::Status() const
    {
        return m_Impl->Status();
    }

    std::optional<ModelArtifact> InferenceRuntime::Active() const
    {
        return m_Impl->m_Active;
    }

    void InferenceRuntime::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        m_Impl->Attach(service, this);
    }

    void InferenceRuntime::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->m_Outbox.Harvest();
    }

    void InferenceRuntime::OnDetach(eMQ::Service *service, eMQ::Context *)
    {
        m_Impl->Detach(service);
    }
} // namespace eOps::Workload
