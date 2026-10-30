#pragma once
#include <memory>
#include <string>
#include <cstdint>
#include <optional>
#include <eMQ/Client.h>
#include "Model.h"
#include "Stats.h"
#include "Dataset.h"
#include "Boundary.h"
#include "../Fabric/Bus.h"
#include "../Fabric/Topics.h"
#include "../Descriptor/Model.h"

// reference ML workloads as eMQ::Client modules (each requires an attached Fabric::BusModule):
//  - InferenceRuntime: subscribes the observables, predicts with the active model, enforces the
//    ActionBoundary, publishes actions, tracks accuracy/latency windows, raises debounced drift
//    events, publishes metrics and a retained heartbeat, and serves Workload(service, "model")
//    ({action: load|rollback|status}).
//  - TrainingRuntime: pairs the latest features with each label into a bounded Dataset, serves
//    Workload(service, "train") (split, fit, evaluate, respond {artifact}, publish Candidate).
//  - Simulator: synthetic IoT gateway publishing deterministic observable/label signals (with
//    optional concept drift) and locality reports.

namespace eOps::Workload
{
    struct WorkloadOptions
    {
        Descriptor::RichMLService m_Service;
        std::string m_Instance; // "<service>-<role>-<node>-<replica>" (Substrate) or any unique id
        std::string m_Node;
        std::string m_Region;
        Fabric::Topics m_Topics;
        uint32_t m_HeartbeatMs = 2000;
        uint32_t m_MetricsMs = 2000;
        uint32_t m_DriftDebounceMs = 5000;
        std::optional<ModelArtifact> m_InitialModel;
    };

    class EMPY_OPS_API InferenceRuntime : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        explicit InferenceRuntime(WorkloadOptions options);
        ~InferenceRuntime() override;

        // {service, instance, role: "inference", node, region, state: ready|waiting-model, modelVersion, accuracy,
        //  samples, latency: {statistic, value, windowSeconds}, actions, rejected}
        Json Status() const;
        std::optional<ModelArtifact> Active() const;

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    class EMPY_OPS_API TrainingRuntime : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        explicit TrainingRuntime(WorkloadOptions options);
        ~TrainingRuntime() override;

        // {service, instance, role: "training", node, region, samples, trained, lastVersion, lastAccuracy}
        Json Status() const;
        size_t Samples() const;

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    struct SimulatorOptions
    {
        Descriptor::RichMLService m_Service;
        Fabric::Topics m_Topics;
        uint32_t m_PeriodMs = 100;
        uint32_t m_DriftAfterMs = 0; // 0 = never drift
        std::string m_Region = "ran";
        double m_LocalityRate = 10; // reported msgs/s per observable
        uint64_t m_Seed = 7;
        uint32_t m_LocalityMs = 2000;
    };

    class EMPY_OPS_API Simulator : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        explicit Simulator(SimulatorOptions options);
        ~Simulator() override;

        size_t Published() const; // observable + label publications admitted so far
        bool Drifted() const;     // the drifted coefficients are in effect

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Workload
