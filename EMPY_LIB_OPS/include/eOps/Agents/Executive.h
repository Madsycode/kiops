#pragma once
#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include "Agent.h"
#include "../Common/Json.h"
#include "../Descriptor/Model.h"
#include "../Descriptor/Validator.h"
#include "../Semantics/Ontology.h"

// Executive agent: intent interpretation, BoM generation, profile synthesis, workflow
// decomposition and task delegation. Semantic parsing is grounded in the BoM-Ontology; LLM
// advice only fills gaps and may evolve the ontology through validated Extend. Never touches
// infrastructure.

namespace eOps::Agents
{
    struct EMPY_OPS_API PlanStep
    {
        std::string m_Name;  // validate | review | place-inference | place-training | enact | schedule
        std::string m_Agent; // harness | compliance | placement | ...
        std::string m_Task;  // agent task (topic level), "" for harness steps
        Json m_Input;
        Json ToJson() const; // {name, agent, task, input}
    };

    struct EMPY_OPS_API WorkflowPlan
    {
        std::string m_Kind; // deploy | retrain | ...
        std::vector<PlanStep> m_Steps;
        Json ToJson() const; // {kind, steps[]}
        static WorkflowPlan FromJson(const Json &json);
    };

    // ontology-grounded reading of an intent
    struct EMPY_OPS_API SemanticFrame
    {
        std::string m_Name;    // DNS label "<domain>-<pattern suffix>"
        std::string m_Domain;  // ontology domain id ("app" when none)
        std::string m_Pattern; // controller | predictor | detector
        std::vector<Descriptor::Observable> m_Observables;
        std::vector<Descriptor::Actionable> m_Actionables;
        std::vector<std::string> m_Objectives;
        std::vector<std::string> m_InferenceRegions;
        std::vector<std::string> m_TrainingRegions;
        std::optional<double> m_LatencyMs;
        std::string m_Statistic;            // p50 | p90 | p95 | p99 ("" = default)
        std::string m_Schedule;             // cron
        std::string m_Accelerator;          // training accelerator kind
        std::optional<double> m_Accuracy;   // lifecycle accuracy threshold in [0, 1]
        std::vector<std::string> m_Unknown; // unmatched content words
        std::vector<std::string> m_Notes;   // explanations (defaults applied, LLM advice accepted/rejected)
        std::string m_Source;               // "deterministic" | "deterministic+llm:<provider>"
        Json ToJson() const;
    };

    struct EMPY_OPS_API Interpretation
    {
        SemanticFrame m_Frame;
        Descriptor::RichMLService m_Descriptor;
        std::string m_Yaml; // EmitDescriptor(m_Descriptor)
        std::vector<Descriptor::Issue> m_Issues;
        std::vector<Semantics::Concept> m_Evolved; // concepts accepted into the ontology
        WorkflowPlan m_Plan;
        // {frame, descriptor (JSON), yaml, issues[], evolved[], plan, valid}
        Json ToJson() const;
    };

    class EMPY_OPS_API Executive
    {
    public:
        explicit Executive(AgentEnvironment environment);

        // Parse → optional Reasoning advice (AskJson; concepts via Ontology::Extend; numbers only fill empty
        // fields) → CoMaS retrieval for operator context → Generate → Validate → Decompose → Remember
        Interpretation Interpret(std::string_view intent, std::string_view operatorId = {});
        // deterministic ontology-grounded semantic parsing
        SemanticFrame Parse(std::string_view intent) const;
        // BoM synthesis with defaults (Listing 2.1 for the reference intent)
        Descriptor::RichMLService Generate(const SemanticFrame &frame, std::string_view intent) const;
        // deploy plan: validate, review, place-inference?, place-training?, enact, schedule?
        WorkflowPlan Decompose(const Descriptor::RichMLService &service) const;

        // validation options used by Interpret (regions from the ontology/network when known)
        void SetValidation(Descriptor::ValidationOptions options);

    private:
        AgentEnvironment m_Environment;
        Descriptor::ValidationOptions m_Validation;
    };
} // namespace eOps::Agents
