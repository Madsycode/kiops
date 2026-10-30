#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <optional>
#include <string_view>
#include "../API.h"

// RichML-Service descriptor (thesis §2.1.1, Listing 2.1): the self-describing ML Bill of
// Materials declaring observables (subscription profiles), bounded actionables (publication
// profiles / permitted action domains), inference and training requirements, constraints and
// lifecycle policy. YAML/JSON field names are lowerCamelCase (see Parser.h).

namespace eOps::Descriptor
{
    // subscription profile: an MMQP topic filter (wildcards allowed)
    struct Observable
    {
        std::string m_Name;
        std::string m_Unit;
        std::string m_Description;
        uint8_t m_Qos = 1;
    };

    // publication profile: a concrete topic or a wildcard permission domain with value bounds
    struct Actionable
    {
        std::string m_Name;
        std::string m_Unit;
        std::optional<double> m_Min;
        std::optional<double> m_Max;
        uint32_t m_MaxRatePerMinute = 0; // 0 = unlimited
        std::string m_Description;
        uint8_t m_Qos = 1;
    };

    struct Resources
    {
        double m_Cpu = 0;                // cores ("500m" = 0.5)
        uint64_t m_Memory = 0;           // bytes
        std::string m_Accelerator;       // "" | gpu | npu | tpu
        uint32_t m_AcceleratorCount = 0; // parser defaults to 1 when an accelerator is named
    };

    // "<statistic> of <observable> over <path> within <windowSeconds> ≤ maxMs"
    struct LatencyObjective
    {
        std::string m_Observable = "control-latency";
        std::string m_Path = "observation-to-action";
        std::string m_Statistic = "p95";
        uint32_t m_WindowSeconds = 60;
        double m_MaxMs = 0;
    };

    struct InferenceSpec
    {
        Resources m_Resources;
        std::string m_Endpoint = "/v1/predict";
        std::vector<std::string> m_Regions;
        std::optional<LatencyObjective> m_Latency;
        uint32_t m_Replicas = 1;
        std::string m_Image;
        std::string m_Actuation; // concrete actionable topic driven by predictions; default first concrete actionable
    };

    struct TrainingSpec
    {
        Resources m_Resources;
        std::vector<std::string> m_Regions;
        std::string m_Schedule; // cron expression (Cron.h), "" = event-driven only
        std::string m_Dataset;
        std::string m_Label; // ground-truth topic (extension)
        uint32_t m_Window = 512;
        uint32_t m_MinSamples = 32;
        double m_Holdout = 0.25;
        std::string m_Image;
    };

    struct SecurityConstraints
    {
        bool m_Encryption = false;
        std::string m_Tenant;
        std::string m_Identity;
    };

    struct PlacementConstraints
    {
        std::vector<std::string> m_Prefer;
        std::vector<std::string> m_Avoid;
        std::vector<std::string> m_Nodes; // pin to these node ids when non-empty
        bool m_AllowRelocation = true;
        double m_Hysteresis = 0.1;
    };

    struct Constraints
    {
        SecurityConstraints m_Security;
        PlacementConstraints m_Placement;
        int32_t m_Priority = 0;
        std::string m_Owner;
        std::vector<std::string> m_Objectives;
    };

    struct LifecyclePolicy
    {
        double m_AccuracyThreshold = 0.9;
        double m_MinImprovement = 0.02;
        uint32_t m_CooldownSeconds = 60;
        uint32_t m_EvaluationWindow = 64;
        uint32_t m_MinSamples = 20;
        double m_Tolerance = 0.5;
        bool m_AutoRollback = true;
    };

    struct Metadata
    {
        std::string m_Name;
        std::string m_Version = "1.0.0";
        std::string m_Description;
        std::string m_Intent;
        std::vector<std::pair<std::string, std::string>> m_Labels;
    };

    struct EMPY_OPS_API RichMLService
    {
        std::string m_ApiVersion = "mlops/v1";
        std::string m_Kind = "RichMLService";
        Metadata m_Metadata;
        std::vector<Observable> m_Observables;
        std::vector<Actionable> m_Actionables;
        std::optional<InferenceSpec> m_Inference;
        std::optional<TrainingSpec> m_Training;
        Constraints m_Constraints;
        LifecyclePolicy m_Lifecycle;

        // the declared actionable whose filter covers concrete `topic` (nullptr when undeclared)
        const Actionable *FindActionable(std::string_view topic) const;
        // inference.actuation, else the first concrete actionable, else ""
        std::string ActuationTopic() const;
    };

    // quantities: cpu "1", "500m", 2, 0.5 (throws std::invalid_argument when malformed or negative)
    EMPY_OPS_API double ParseCpu(std::string_view text);
    // memory "1Gi", "512Mi", "64Ki", "1G", "500M", "1k", "1024" bytes (throws std::invalid_argument)
    EMPY_OPS_API uint64_t ParseMemory(std::string_view text);
    // whole cores as "2", fractions as millicores "500m"
    EMPY_OPS_API std::string FormatCpu(double cores);
    // "4Gi" / "512Mi" / "64Ki" when exact, else plain bytes
    EMPY_OPS_API std::string FormatMemory(uint64_t bytes);
} // namespace eOps::Descriptor
