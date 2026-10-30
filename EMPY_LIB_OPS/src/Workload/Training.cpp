#include "RuntimeDetail.h"
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/Stdout.h>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace eOps::Workload
{
    using eCore::MMQP::MMQP_REASON_BUSY;
    using eCore::MMQP::MMQP_REASON_MALFORMED_PACKET;
    using eCore::MMQP::MMQP_REASON_UNSPECIFIED;

    namespace
    {
        // result of one fit computed on a worker thread
        struct Outcome
        {
            std::optional<ModelArtifact> m_Artifact;
            std::string m_Error;
            size_t m_Train = 0;
            size_t m_Test = 0;
        };

        void Columns(const std::vector<Sample> &samples, std::vector<std::vector<double>> &x, std::vector<double> &y)
        {
            x.clear();
            y.clear();
            x.reserve(samples.size());
            y.reserve(samples.size());
            for (const auto &sample : samples)
            {
                x.push_back(sample.m_Features);
                y.push_back(sample.m_Label);
            }
        }
    } // namespace

    struct TrainingRuntime::Impl
    {
        // one training request waiting for its worker result
        struct Job
        {
            Fabric::Bus::Responder m_Respond;
            uint32_t m_Version = 0;
            Json m_Workflow;
        };

        WorkloadOptions m_Options;
        std::vector<std::string> m_Features;
        std::vector<std::optional<double>> m_Values; // latest value per observable, declaration order
        std::string m_Label;
        Dataset m_Dataset;
        uint64_t m_Labels = 0;   // labels received
        uint64_t m_Unpaired = 0; // labels that arrived before every feature was observed
        uint64_t m_Ignored = 0;  // unparsable data-plane payloads
        uint64_t m_Trained = 0;
        uint32_t m_Counter = 0; // highest version produced (local versioning)
        std::optional<uint32_t> m_LastVersion;
        std::optional<double> m_LastAccuracy;
        std::optional<Job> m_Job;
        std::shared_ptr<bool> m_Alive = std::make_shared<bool>(true); // guards worker hand-backs
        eMQ::Service *m_Service = nullptr;
        const void *m_Owner = nullptr;
        Detail::Ticker m_Heartbeat;
        Detail::Outbox m_Outbox;
        bool m_Stopped = false;

        static size_t Window(const WorkloadOptions &options)
        {
            const auto &training = options.m_Service.m_Training;
            if (!training)
            {
                throw std::invalid_argument("Training runtime needs a descriptor with a training section");
            }
            if (!training->m_Window)
            {
                throw std::invalid_argument("training.window must be at least 1");
            }
            return training->m_Window;
        }

        explicit Impl(WorkloadOptions options) : m_Options(std::move(options)), m_Dataset(Window(m_Options))
        {
            Detail::Resolve(m_Options, "training");
            const auto &training = *m_Options.m_Service.m_Training;
            m_Label = training.m_Label;
            if (m_Label.empty() || !ValidTopic(m_Label, false) || IsFilter(m_Label))
            {
                throw std::invalid_argument("Training runtime needs a concrete training.label topic");
            }
            if (!(training.m_Holdout > 0 && training.m_Holdout < 1))
            {
                throw std::invalid_argument("training.holdout must lie strictly between 0 and 1");
            }
            m_Features = FeatureNames(m_Options.m_Service);
            m_Values.resize(m_Features.size());
            (void)m_Options.m_Topics.Candidate(this->Name());
        }

        ~Impl()
        {
            *m_Alive = false;
        }

        const std::string &Name() const
        {
            return m_Options.m_Service.m_Metadata.m_Name;
        }

        const Descriptor::TrainingSpec &Spec() const
        {
            return *m_Options.m_Service.m_Training;
        }

        Json Status() const
        {
            const char *state = m_Stopped                                       ? "stopped"
                                : m_Job                                         ? "training"
                                : m_Dataset.Size() >= this->Spec().m_MinSamples ? "ready"
                                                                                : "collecting";
            return Json::Object{{"service", this->Name()},
                                {"instance", m_Options.m_Instance},
                                {"role", "training"},
                                {"node", m_Options.m_Node},
                                {"region", m_Options.m_Region},
                                {"state", state},
                                {"samples", m_Dataset.Size()},
                                {"minSamples", this->Spec().m_MinSamples},
                                {"window", m_Dataset.Capacity()},
                                {"dataset", this->Spec().m_Dataset},
                                {"labels", m_Labels},
                                {"unpaired", m_Unpaired},
                                {"trained", m_Trained},
                                {"lastVersion", m_LastVersion ? Json(*m_LastVersion) : Json()},
                                {"lastAccuracy", Detail::Number(m_LastAccuracy)},
                                {"publishFailures", m_Outbox.Failed()},
                                {"time", NowMillis()}};
        }

        void PublishHeartbeat()
        {
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            m_Outbox.Harvest();
            auto topic = m_Options.m_Topics.Heartbeat(this->Name(), m_Options.m_Instance);
            auto message = Fabric::Envelope::Make("status", this->Status());
            m_Outbox.Send(topic, [&] { return bus->Publish(topic, message, 1, 1); });
        }

        // -- data plane

        void OnObservation(size_t index, const eMQ::PublishRequest &message)
        {
            if (message.m_Topic == m_Label)
            {
                return;
            }
            auto value = Fabric::ParseValue(message.m_Payload);
            if (!value)
            {
                ++m_Ignored;
                return;
            }
            m_Values[index] = *value;
        }

        // each label is paired with the latest value of every feature
        void OnLabel(const eMQ::PublishRequest &message)
        {
            auto label = Fabric::ParseValue(message.m_Payload);
            if (!label)
            {
                ++m_Ignored;
                return;
            }
            ++m_Labels;
            Sample sample;
            sample.m_Label = *label;
            sample.m_Time = NowMillis();
            sample.m_Features.reserve(m_Values.size());
            for (const auto &value : m_Values)
            {
                if (!value)
                {
                    ++m_Unpaired;
                    return;
                }
                sample.m_Features.push_back(*value);
            }
            m_Dataset.Add(std::move(sample));
        }

        // -- train requests

        void OnTrain(const Fabric::Inbound &inbound, Fabric::Bus::Responder respond)
        {
            const Json &body = inbound.m_Envelope.m_Body;
            if (!body.IsNull() && !body.IsObject())
            {
                return respond({}, {MMQP_REASON_MALFORMED_PACKET, "Train request body must be an object"});
            }
            std::optional<uint32_t> requested;
            if (const Json *version = body.IsObject() ? body.Find("version") : nullptr; version && !version->IsNull())
            {
                if (!version->IsInteger() || version->AsInt() < 1 ||
                    version->AsInt() > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
                {
                    return respond({}, {MMQP_REASON_MALFORMED_PACKET, "Train 'version' must be a positive integer"});
                }
                requested = static_cast<uint32_t>(version->AsInt());
            }
            if (m_Job)
            {
                return respond({}, {MMQP_REASON_BUSY, "Training already in progress"});
            }
            const auto &spec = this->Spec();
            if (m_Dataset.Size() < spec.m_MinSamples)
            {
                return respond({},
                               {MMQP_REASON_UNSPECIFIED, "Insufficient samples: " + std::to_string(m_Dataset.Size()) +
                                                             " of " + std::to_string(spec.m_MinSamples) + " required"});
            }
            std::vector<Sample> train;
            std::vector<Sample> test;
            m_Dataset.Split(spec.m_Holdout, train, test);
            if (test.empty() || train.size() < m_Features.size() + 1)
            {
                return respond({}, {MMQP_REASON_UNSPECIFIED,
                                    "Insufficient samples for a holdout split: " + std::to_string(train.size()) +
                                        " training and " + std::to_string(test.size()) + " test samples"});
            }
            uint32_t version = requested ? *requested : m_Counter + 1;
            Json workflow = body.IsObject() && body.Has("workflowId") ? body["workflowId"] : Json();
            m_Job = Job{std::move(respond), version, std::move(workflow)};

            // the fit runs on a worker; its result is handed back to the owner thread
            ModelArtifact shape;
            shape.m_Service = this->Name();
            shape.m_Version = version;
            shape.m_Id = this->Name() + "-v" + std::to_string(version);
            shape.m_Features = m_Features;
            shape.m_Dataset = spec.m_Dataset;
            double tolerance = m_Options.m_Service.m_Lifecycle.m_Tolerance;
            std::weak_ptr<bool> alive = m_Alive;
            eMQ::Service *service = m_Service;
            auto work = [service, alive, self = this, shape = std::move(shape), train = std::move(train),
                         test = std::move(test), tolerance]() mutable
            {
                Outcome outcome;
                outcome.m_Train = train.size();
                outcome.m_Test = test.size();
                try
                {
                    std::vector<std::vector<double>> x;
                    std::vector<double> y;
                    Columns(train, x, y);
                    ModelArtifact artifact = LinearModel::Fit(x, y);
                    Columns(test, x, y);
                    artifact.m_Accuracy = LinearModel::Accuracy(artifact, x, y, tolerance);
                    artifact.m_Id = std::move(shape.m_Id);
                    artifact.m_Service = std::move(shape.m_Service);
                    artifact.m_Version = shape.m_Version;
                    artifact.m_Features = std::move(shape.m_Features);
                    artifact.m_Dataset = std::move(shape.m_Dataset);
                    outcome.m_Artifact = std::move(artifact);
                }
                catch (const std::exception &error)
                {
                    outcome.m_Error = error.what();
                }
                // dropped when the runtime is shutting down (dispatch queue closed)
                service->Dispatch(eCore::Task(
                    [alive, self, outcome = std::move(outcome)]() mutable
                    {
                        if (auto living = alive.lock(); living && *living)
                        {
                            self->Finish(std::move(outcome));
                        }
                    }));
            };
            try
            {
                (void)m_Service->Async(std::move(work));
            }
            catch (const std::exception &error)
            {
                auto job = std::move(*m_Job);
                m_Job.reset();
                job.m_Respond({}, {MMQP_REASON_BUSY, std::string("Training worker unavailable: ") + error.what()});
                return;
            }
            EMPY_INFO("Workload {}: training v{} on {} samples", m_Options.m_Instance, version, m_Dataset.Size());
            this->PublishHeartbeat();
        }

        void Finish(Outcome outcome)
        {
            if (!m_Job)
            {
                return;
            }
            Job job = std::move(*m_Job);
            m_Job.reset();
            if (!outcome.m_Artifact)
            {
                EMPY_WARN("Workload {}: training v{} failed: {}", m_Options.m_Instance, job.m_Version, outcome.m_Error);
                job.m_Respond({}, {MMQP_REASON_UNSPECIFIED, "Training failed: " + outcome.m_Error});
                this->PublishHeartbeat();
                return;
            }
            const ModelArtifact &artifact = *outcome.m_Artifact;
            m_Counter = std::max(m_Counter, artifact.m_Version);
            m_LastVersion = artifact.m_Version;
            m_LastAccuracy = artifact.m_Accuracy;
            ++m_Trained;
            Json body = Json::Object{{"service", this->Name()},      {"instance", m_Options.m_Instance},
                                     {"workflowId", job.m_Workflow}, {"artifact", artifact.ToJson()},
                                     {"samples", outcome.m_Train},   {"testSamples", outcome.m_Test}};
            EMPY_INFO("Workload {}: trained v{} (holdout accuracy {})", m_Options.m_Instance, artifact.m_Version,
                      artifact.m_Accuracy);
            job.m_Respond(Fabric::Envelope::Make("artifact", body), {});
            // the "train/model" observable: candidate artefacts for the lifecycle
            if (auto *bus = Detail::FindBus(m_Service))
            {
                auto topic = m_Options.m_Topics.Candidate(this->Name());
                auto message = Fabric::Envelope::Make("artifact", std::move(body));
                m_Outbox.Send(topic, [&] { return bus->Publish(topic, message, 1); });
            }
            this->PublishHeartbeat();
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
            Detail::Require(bus.Expose(topics.Heartbeat(name, m_Options.m_Instance), false, 1), "Heartbeat profile");
            Detail::Require(bus.Expose(topics.Candidate(name)), "Candidate profile");
            Detail::Require(bus.Serve(
                                topics.Workload(name, "train"),
                                [this](const Fabric::Inbound &inbound, Fabric::Bus::Responder respond)
                                { this->OnTrain(inbound, std::move(respond)); }, owner),
                            "Train responder");
            const auto &observables = m_Options.m_Service.m_Observables;
            for (size_t i = 0; i < observables.size(); ++i)
            {
                Detail::Require(bus.ListenRaw(
                                    observables[i].m_Name, [this, i](const eMQ::PublishRequest &message)
                                    { this->OnObservation(i, message); }, observables[i].m_Qos, owner),
                                "Observable subscription " + observables[i].m_Name);
            }
            Detail::Require(
                bus.ListenRaw(
                    m_Label, [this](const eMQ::PublishRequest &message) { this->OnLabel(message); }, 1, owner),
                "Label subscription " + m_Label);
            m_Heartbeat.Start(*service, m_Options.m_HeartbeatMs, [this] { this->PublishHeartbeat(); });
            EMPY_INFO("Workload {}: training ready for {} (window {})", m_Options.m_Instance, name,
                      m_Dataset.Capacity());
            this->PublishHeartbeat();
        }

        void Detach(eMQ::Service *service)
        {
            m_Heartbeat.Stop();
            m_Stopped = true;
            *m_Alive = false; // a fit still running on a worker is discarded
            if (m_Job)
            {
                Job job = std::move(*m_Job);
                m_Job.reset();
                job.m_Respond({}, {MMQP_REASON_UNSPECIFIED, "Training runtime stopped"});
            }
            if (auto *bus = Detail::FindBus(service))
            {
                this->PublishHeartbeat(); // retained "stopped" state, acknowledged before the module goes
                m_Outbox.Drain(std::chrono::milliseconds(1000));
                bus->Remove(m_Owner);
            }
            m_Service = nullptr;
        }
    };

    TrainingRuntime::TrainingRuntime(WorkloadOptions options) : m_Impl(std::make_unique<Impl>(std::move(options)))
    {
    }

    TrainingRuntime::~TrainingRuntime() = default;

    Json TrainingRuntime::Status() const
    {
        return m_Impl->Status();
    }

    size_t TrainingRuntime::Samples() const
    {
        return m_Impl->m_Dataset.Size();
    }

    void TrainingRuntime::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        m_Impl->Attach(service, this);
    }

    void TrainingRuntime::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->m_Outbox.Harvest();
    }

    void TrainingRuntime::OnDetach(eMQ::Service *service, eMQ::Context *)
    {
        m_Impl->Detach(service);
    }
} // namespace eOps::Workload
