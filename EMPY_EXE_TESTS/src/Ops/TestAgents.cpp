#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Agents/Executive.h>
#include <eOps/Agents/Placement.h>
#include <eOps/Agents/Lifecycle.h>
#include <eOps/Agents/Compliance.h>
#include <eOps/Context/Comas.h>
#include <eOps/Context/Knowledge.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Descriptor/Validator.h>
#include <eOps/Intelligence/Provider.h>
#include <eOps/Semantics/Ontology.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <filesystem>
#include <functional>
#include <memory>

using namespace eOps;
using namespace eOps::Agents;

namespace
{
    using eTest::Ops::ConfigFile;

    // the intent of thesis Listing 2.1 (spec §10)
    const char *LISTING_INTENT =
        "Deploy a home temperature controller that keeps the temperature between 16 and 24 degrees using temperature "
        "and humidity sensors, with a p95 latency under 5 ms, retrained daily on a GPU";

    // a fresh copy per test: the Executive may extend it
    Semantics::Ontology LoadOntology()
    {
        return Semantics::Ontology::Load(ConfigFile("Ontology.yaml"));
    }

    Descriptor::RichMLService HomeControl()
    {
        return Descriptor::LoadDescriptor(ConfigFile("HomeControl.yaml"));
    }

    Descriptor::RichMLService EnergySaver()
    {
        return Descriptor::LoadDescriptor(ConfigFile("EnergySaver.yaml"));
    }

    PolicySet SeedPolicies()
    {
        std::string path = ConfigFile("Policies.yaml").string();
        return PolicySet::FromYaml(eCore::YamlNode::Load(path));
    }

    Descriptor::ValidationOptions Regions()
    {
        Descriptor::ValidationOptions options;
        options.m_Regions = {"ran", "edge", "cloud"};
        return options;
    }

    double Or(const std::optional<double> &value, double fallback = -1e300)
    {
        return value.value_or(fallback);
    }

    bool Has(const std::vector<std::string> &items, const std::string &item)
    {
        return std::find(items.begin(), items.end(), item) != items.end();
    }

    std::string Errors(const std::vector<Descriptor::Issue> &issues)
    {
        std::string text;
        for (const Descriptor::Issue &issue : issues)
        {
            if (issue.m_Severity == Descriptor::Severity::Error)
            {
                text += issue.m_Path + ": " + issue.m_Message + "; ";
            }
        }
        return text;
    }

    void CheckResources(const Descriptor::Resources &actual, const Descriptor::Resources &expected)
    {
        EMPY_CHECK_EQ(actual.m_Cpu, expected.m_Cpu);
        EMPY_CHECK_EQ(actual.m_Memory, expected.m_Memory);
        EMPY_CHECK_EQ(actual.m_Accelerator, expected.m_Accelerator);
        EMPY_CHECK_EQ(actual.m_AcceleratorCount, expected.m_AcceleratorCount);
    }

    // field-by-field descriptor equality with readable failures
    void CheckSameDescriptor(const Descriptor::RichMLService &actual, const Descriptor::RichMLService &expected)
    {
        EMPY_CHECK_EQ(actual.m_ApiVersion, expected.m_ApiVersion);
        EMPY_CHECK_EQ(actual.m_Kind, expected.m_Kind);
        EMPY_CHECK_EQ(actual.m_Metadata.m_Name, expected.m_Metadata.m_Name);
        EMPY_CHECK_EQ(actual.m_Metadata.m_Version, expected.m_Metadata.m_Version);
        EMPY_CHECK_EQ(actual.m_Metadata.m_Description, expected.m_Metadata.m_Description);
        EMPY_CHECK_EQ(actual.m_Metadata.m_Intent, expected.m_Metadata.m_Intent);
        EMPY_CHECK(actual.m_Metadata.m_Labels == expected.m_Metadata.m_Labels);
        EMPY_REQUIRE_EQ(actual.m_Observables.size(), expected.m_Observables.size());
        for (size_t i = 0; i < expected.m_Observables.size(); ++i)
        {
            EMPY_CHECK_EQ(actual.m_Observables[i].m_Name, expected.m_Observables[i].m_Name);
            EMPY_CHECK_EQ(actual.m_Observables[i].m_Unit, expected.m_Observables[i].m_Unit);
            EMPY_CHECK_EQ(actual.m_Observables[i].m_Description, expected.m_Observables[i].m_Description);
            EMPY_CHECK_EQ(actual.m_Observables[i].m_Qos, expected.m_Observables[i].m_Qos);
        }
        EMPY_REQUIRE_EQ(actual.m_Actionables.size(), expected.m_Actionables.size());
        for (size_t i = 0; i < expected.m_Actionables.size(); ++i)
        {
            const Descriptor::Actionable &a = actual.m_Actionables[i];
            const Descriptor::Actionable &e = expected.m_Actionables[i];
            EMPY_CHECK_EQ(a.m_Name, e.m_Name);
            EMPY_CHECK_EQ(a.m_Unit, e.m_Unit);
            EMPY_CHECK_EQ(Or(a.m_Min), Or(e.m_Min));
            EMPY_CHECK_EQ(Or(a.m_Max), Or(e.m_Max));
            EMPY_CHECK_EQ(a.m_MaxRatePerMinute, e.m_MaxRatePerMinute);
            EMPY_CHECK_EQ(a.m_Description, e.m_Description);
            EMPY_CHECK_EQ(a.m_Qos, e.m_Qos);
        }
        EMPY_REQUIRE_EQ(actual.m_Inference.has_value(), expected.m_Inference.has_value());
        if (expected.m_Inference)
        {
            const Descriptor::InferenceSpec &a = *actual.m_Inference;
            const Descriptor::InferenceSpec &e = *expected.m_Inference;
            CheckResources(a.m_Resources, e.m_Resources);
            EMPY_CHECK_EQ(a.m_Endpoint, e.m_Endpoint);
            EMPY_CHECK(a.m_Regions == e.m_Regions);
            EMPY_CHECK_EQ(a.m_Replicas, e.m_Replicas);
            EMPY_CHECK_EQ(a.m_Image, e.m_Image);
            EMPY_CHECK_EQ(a.m_Actuation, e.m_Actuation);
            EMPY_REQUIRE_EQ(a.m_Latency.has_value(), e.m_Latency.has_value());
            if (e.m_Latency)
            {
                EMPY_CHECK_EQ(a.m_Latency->m_Observable, e.m_Latency->m_Observable);
                EMPY_CHECK_EQ(a.m_Latency->m_Path, e.m_Latency->m_Path);
                EMPY_CHECK_EQ(a.m_Latency->m_Statistic, e.m_Latency->m_Statistic);
                EMPY_CHECK_EQ(a.m_Latency->m_WindowSeconds, e.m_Latency->m_WindowSeconds);
                EMPY_CHECK_EQ(a.m_Latency->m_MaxMs, e.m_Latency->m_MaxMs);
            }
        }
        EMPY_REQUIRE_EQ(actual.m_Training.has_value(), expected.m_Training.has_value());
        if (expected.m_Training)
        {
            const Descriptor::TrainingSpec &a = *actual.m_Training;
            const Descriptor::TrainingSpec &e = *expected.m_Training;
            CheckResources(a.m_Resources, e.m_Resources);
            EMPY_CHECK(a.m_Regions == e.m_Regions);
            EMPY_CHECK_EQ(a.m_Schedule, e.m_Schedule);
            EMPY_CHECK_EQ(a.m_Dataset, e.m_Dataset);
            EMPY_CHECK_EQ(a.m_Label, e.m_Label);
            EMPY_CHECK_EQ(a.m_Window, e.m_Window);
            EMPY_CHECK_EQ(a.m_MinSamples, e.m_MinSamples);
            EMPY_CHECK_EQ(a.m_Holdout, e.m_Holdout);
            EMPY_CHECK_EQ(a.m_Image, e.m_Image);
        }
        const Descriptor::Constraints &ac = actual.m_Constraints;
        const Descriptor::Constraints &ec = expected.m_Constraints;
        EMPY_CHECK_EQ(ac.m_Priority, ec.m_Priority);
        EMPY_CHECK_EQ(ac.m_Owner, ec.m_Owner);
        EMPY_CHECK(ac.m_Objectives == ec.m_Objectives);
        EMPY_CHECK_EQ(ac.m_Security.m_Encryption, ec.m_Security.m_Encryption);
        EMPY_CHECK_EQ(ac.m_Security.m_Tenant, ec.m_Security.m_Tenant);
        EMPY_CHECK_EQ(ac.m_Security.m_Identity, ec.m_Security.m_Identity);
        EMPY_CHECK(ac.m_Placement.m_Prefer == ec.m_Placement.m_Prefer);
        EMPY_CHECK(ac.m_Placement.m_Avoid == ec.m_Placement.m_Avoid);
        EMPY_CHECK(ac.m_Placement.m_Nodes == ec.m_Placement.m_Nodes);
        EMPY_CHECK_EQ(ac.m_Placement.m_AllowRelocation, ec.m_Placement.m_AllowRelocation);
        EMPY_CHECK_EQ(ac.m_Placement.m_Hysteresis, ec.m_Placement.m_Hysteresis);
        const Descriptor::LifecyclePolicy &al = actual.m_Lifecycle;
        const Descriptor::LifecyclePolicy &el = expected.m_Lifecycle;
        EMPY_CHECK_EQ(al.m_AccuracyThreshold, el.m_AccuracyThreshold);
        EMPY_CHECK_EQ(al.m_MinImprovement, el.m_MinImprovement);
        EMPY_CHECK_EQ(al.m_CooldownSeconds, el.m_CooldownSeconds);
        EMPY_CHECK_EQ(al.m_EvaluationWindow, el.m_EvaluationWindow);
        EMPY_CHECK_EQ(al.m_MinSamples, el.m_MinSamples);
        EMPY_CHECK_EQ(al.m_Tolerance, el.m_Tolerance);
        EMPY_CHECK_EQ(al.m_AutoRollback, el.m_AutoRollback);
        // and the whole document
        EMPY_CHECK(Descriptor::ToJson(actual) == Descriptor::ToJson(expected));
    }

    // scripted structured-output provider standing in for an LLM
    class FakeProvider : public Intelligence::Provider
    {
    public:
        explicit FakeProvider(std::function<Json(const Intelligence::Prompt &)> reply) : m_Reply(std::move(reply))
        {
        }
        std::string Name() const override
        {
            return "fake";
        }
        bool Available() const override
        {
            return true;
        }
        Intelligence::Completion Complete(const Intelligence::Prompt &prompt) override
        {
            ++m_Calls;
            Intelligence::Completion completion;
            completion.m_Ok = true;
            completion.m_Provider = "fake";
            completion.m_Json = m_Reply(prompt);
            completion.m_Text = completion.m_Json->Dump();
            return completion;
        }
        size_t m_Calls = 0;

    private:
        std::function<Json(const Intelligence::Prompt &)> m_Reply;
    };

    // knowledge source returning fixed operator-context entries (CoMaS R′), recording the queries
    class ScriptedSource : public Context::KnowledgeSource
    {
    public:
        std::vector<Context::Entry> Execute(const Context::Query &query) override
        {
            ++m_Queries;
            (void)query;
            return m_Entries;
        }
        std::vector<Context::Entry> m_Entries;
        size_t m_Queries = 0;
    };

    Context::Entry OperatorEntry(std::string id, std::string text, std::vector<std::string> tags)
    {
        Context::Entry entry;
        entry.m_Id = std::move(id);
        entry.m_Collection = Context::OPERATOR;
        entry.m_Data = Json::MakeObject({{"text", std::move(text)}});
        entry.m_Tags = std::move(tags);
        entry.m_Score = 1;
        return entry;
    }

    // ---- Knowledge.yaml-like inventory and network (spec §5)

    Topology::Node MakeNode(std::string id, std::string region, double cpu, uint64_t memoryGi,
                            std::map<std::string, uint32_t> accelerators = {})
    {
        Topology::Node node;
        node.m_Id = std::move(id);
        node.m_Region = std::move(region);
        node.m_Cpu = cpu;
        node.m_Memory = memoryGi << 30;
        node.m_Accelerators = std::move(accelerators);
        return node;
    }

    Topology::Inventory KnowledgeInventory()
    {
        Topology::Inventory inventory;
        inventory.m_Nodes.push_back(MakeNode("ran-1", "ran", 4, 8, {{"npu", 1}}));
        inventory.m_Nodes.push_back(MakeNode("edge-1", "edge", 16, 32, {{"gpu", 1}}));
        inventory.m_Nodes.push_back(MakeNode("cloud-1", "cloud", 64, 256, {{"gpu", 8}}));
        return inventory;
    }

    Topology::NetworkModel KnowledgeNetwork()
    {
        Json network = Json::Parse(R"({"regions": ["ran", "edge", "cloud"],
            "latency": {"ran": {"edge": 2, "cloud": 25}, "edge": {"cloud": 15}},
            "topics": {"home/#": {"ran": 40, "edge": 5}},
            "consumers": {"home/set/#": {"ran": 10}},
            "shards": {"ran": ["home/#"], "cloud": ["factory/#"]}})");
        return Topology::NetworkModel::FromJson(network);
    }

    const Candidate *FindCandidate(const PlacementPlan &plan, const std::string &node)
    {
        for (const Candidate &candidate : plan.m_Candidates)
        {
            if (candidate.m_Node == node)
            {
                return &candidate;
            }
        }
        return nullptr;
    }

    bool ReasonContains(const Candidate *candidate, const std::string &text)
    {
        if (!candidate)
        {
            return false;
        }
        for (const std::string &reason : candidate->m_Reasons)
        {
            if (reason.find(text) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    // ---- lifecycle fixtures

    ModelRecord Record(uint32_t version, double accuracy, std::string status)
    {
        ModelRecord record;
        record.m_Version = version;
        record.m_ArtifactId = "svc-v" + std::to_string(version);
        record.m_Accuracy = accuracy;
        record.m_At = 1000 * version;
        record.m_Status = std::move(status);
        return record;
    }

    Workload::ModelArtifact Artifact(uint32_t version, double accuracy)
    {
        Workload::ModelArtifact artifact;
        artifact.m_Service = "svc";
        artifact.m_Version = version;
        artifact.m_Accuracy = accuracy;
        artifact.m_Features = {"home/temp", "home/hum"};
        artifact.m_Weights = {1, 2, 3};
        return artifact;
    }

    // ---- compliance fixtures

    bool HasFinding(const ComplianceReport &report, const std::string &criterion, const std::string &severity = {})
    {
        for (const Finding &finding : report.m_Findings)
        {
            if (finding.m_Criterion == criterion && (severity.empty() || finding.m_Severity == severity))
            {
                return true;
            }
        }
        return false;
    }

    std::string Findings(const ComplianceReport &report)
    {
        std::string text;
        for (const Finding &finding : report.m_Findings)
        {
            text += finding.m_Criterion + "/" + finding.m_Severity + ": " + finding.m_Message + "\n";
        }
        return text;
    }

    Descriptor::Actionable MakeActionable(std::string name, std::string unit, std::optional<double> min,
                                          std::optional<double> max)
    {
        Descriptor::Actionable actionable;
        actionable.m_Name = std::move(name);
        actionable.m_Unit = std::move(unit);
        actionable.m_Min = min;
        actionable.m_Max = max;
        return actionable;
    }
} // namespace

// ============================================================================ Executive

EMPY_TEST(OpsAgents, ExecutiveListing21)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});
    executive.SetValidation(Regions());

    SemanticFrame frame = executive.Parse(LISTING_INTENT);
    EMPY_CHECK_EQ(frame.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(frame.m_Domain, std::string("home"));
    EMPY_CHECK_EQ(frame.m_Pattern, std::string("controller"));
    EMPY_CHECK_EQ(Or(frame.m_LatencyMs), 5.0);
    EMPY_CHECK_EQ(frame.m_Statistic, std::string("p95"));
    EMPY_CHECK_EQ(frame.m_Schedule, std::string("0 0 * * *"));
    EMPY_CHECK_EQ(frame.m_Accelerator, std::string("gpu"));
    EMPY_CHECK(frame.m_InferenceRegions.empty());
    EMPY_CHECK(!frame.m_Accuracy.has_value());
    EMPY_CHECK_EQ(Join(frame.m_Unknown, ","), std::string());
    EMPY_CHECK_EQ(frame.m_Source, std::string("deterministic"));

    // Parse → Generate yields exactly Configs/MLOps/HomeControl.yaml (Listing 2.1 + label + intent)
    Descriptor::RichMLService expected = HomeControl();
    CheckSameDescriptor(executive.Generate(frame, LISTING_INTENT), expected);

    Interpretation interpretation = executive.Interpret(LISTING_INTENT, "operator-1");
    CheckSameDescriptor(interpretation.m_Descriptor, expected);
    EMPY_CHECK_EQ(Errors(interpretation.m_Issues), std::string());
    EMPY_CHECK(Descriptor::Validate(interpretation.m_Descriptor, Regions()).empty());
    EMPY_CHECK(interpretation.m_Evolved.empty());
    // the emitted YAML round-trips to the same service
    CheckSameDescriptor(Descriptor::ParseDescriptor(interpretation.m_Yaml), expected);
    Json json = interpretation.ToJson();
    EMPY_CHECK(json["valid"].AsBool());
    EMPY_CHECK_EQ(json["frame"]["name"].AsString(), std::string("home-control"));
    EMPY_CHECK_EQ(json["plan"]["kind"].AsString(), std::string("deploy"));
    EMPY_CHECK(Descriptor::FromJson(json["descriptor"]).m_Metadata.m_Name == "home-control");

    // whitespace inside a multi-line intent is folded like the YAML descriptor's folded scalar
    std::string folded = LISTING_INTENT;
    folded.replace(folded.find(" using"), 1, "\n    ");
    EMPY_CHECK_EQ(executive.Generate(executive.Parse(folded), folded).m_Metadata.m_Intent,
                  expected.m_Metadata.m_Intent);
}

EMPY_TEST(OpsAgents, ExecutiveFactoryDetector)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});
    executive.SetValidation(Regions());
    Interpretation result =
        executive.Interpret("Deploy a factory vibration anomaly detector at the edge with hourly retraining");
    const SemanticFrame &frame = result.m_Frame;
    EMPY_CHECK_EQ(frame.m_Domain, std::string("factory"));
    EMPY_CHECK_EQ(frame.m_Pattern, std::string("detector"));
    EMPY_CHECK_EQ(frame.m_Name, std::string("factory-detector"));
    EMPY_CHECK_EQ(Join(frame.m_Unknown, ","), std::string());
    const Descriptor::RichMLService &service = result.m_Descriptor;
    EMPY_REQUIRE_EQ(service.m_Observables.size(), size_t(1));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Name, std::string("factory/vibration"));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Unit, std::string("mm-per-second"));
    EMPY_REQUIRE_EQ(service.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(service.m_Actionables[0].m_Name, std::string("factory/detector/vibration"));
    EMPY_REQUIRE(service.m_Inference.has_value());
    EMPY_CHECK(service.m_Inference->m_Regions == std::vector<std::string>{"edge"});
    EMPY_CHECK_EQ(service.m_Inference->m_Latency->m_MaxMs, 50.0);
    EMPY_REQUIRE(service.m_Training.has_value());
    EMPY_CHECK_EQ(service.m_Training->m_Schedule, std::string("0 * * * *"));
    EMPY_CHECK_EQ(service.m_Training->m_Label, std::string("factory/label/vibration"));
    EMPY_CHECK_EQ(service.m_Training->m_Dataset, std::string("factory-telemetry"));
    EMPY_CHECK(service.m_Training->m_Regions == (std::vector<std::string>{"edge", "cloud"}));
    EMPY_CHECK_EQ(service.ActuationTopic(), std::string("factory/detector/vibration"));
    EMPY_CHECK_EQ(Errors(result.m_Issues), std::string());
}

EMPY_TEST(OpsAgents, ExecutiveVehiclePredictor)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});
    executive.SetValidation(Regions());
    Interpretation result = executive.Interpret("Deploy a vehicle speed predictor with 90% accuracy");
    const SemanticFrame &frame = result.m_Frame;
    EMPY_CHECK_EQ(frame.m_Domain, std::string("vehicle")); // "vehicle speed" is one quantity phrase
    EMPY_CHECK_EQ(frame.m_Pattern, std::string("predictor"));
    EMPY_CHECK_EQ(frame.m_Name, std::string("vehicle-forecast"));
    EMPY_CHECK_EQ(Or(frame.m_Accuracy), 0.9);
    EMPY_CHECK_EQ(Join(frame.m_Unknown, ","), std::string());
    const Descriptor::RichMLService &service = result.m_Descriptor;
    EMPY_REQUIRE_EQ(service.m_Observables.size(), size_t(1));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Name, std::string("vehicle/speed"));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Unit, std::string("km-per-hour"));
    EMPY_REQUIRE_EQ(service.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(service.m_Actionables[0].m_Name, std::string("vehicle/forecast/speed"));
    EMPY_CHECK_EQ(Or(service.m_Actionables[0].m_Min), 0.0);
    EMPY_CHECK(!service.m_Actionables[0].m_Max.has_value());
    EMPY_CHECK_EQ(service.m_Lifecycle.m_AccuracyThreshold, 0.9);
    EMPY_CHECK(service.m_Inference->m_Regions == std::vector<std::string>{"edge"});
    EMPY_CHECK_EQ(service.m_Inference->m_Latency->m_MaxMs, 100.0);
    EMPY_CHECK_EQ(service.m_Training->m_Label, std::string("vehicle/label/speed"));
    EMPY_CHECK(service.m_Training->m_Schedule.empty());
    EMPY_CHECK(service.m_Training->m_Resources.m_Accelerator.empty());
    EMPY_CHECK_EQ(Errors(result.m_Issues), std::string());

    // the other accuracy phrasings
    EMPY_CHECK_EQ(Or(executive.Parse("a vehicle speed predictor with an accuracy of at least 95%").m_Accuracy), 0.95);
    EMPY_CHECK_EQ(Or(executive.Parse("a vehicle speed predictor with accuracy above 0.8").m_Accuracy), 0.8);
    EMPY_CHECK(!executive.Parse("a vehicle speed predictor with 150% accuracy").m_Accuracy.has_value());
}

EMPY_TEST(OpsAgents, ExecutiveUnknownWords)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});

    // nothing grounded: unknown words reported, defaults applied, the descriptor fails validation honestly
    Interpretation empty = executive.Interpret("Deploy a zorblax quantum flux predictor");
    EMPY_CHECK(Has(empty.m_Frame.m_Unknown, "zorblax"));
    EMPY_CHECK(Has(empty.m_Frame.m_Unknown, "quantum"));
    EMPY_CHECK(Has(empty.m_Frame.m_Unknown, "flux"));
    EMPY_CHECK(!Has(empty.m_Frame.m_Unknown, "deploy"));
    EMPY_CHECK(!Has(empty.m_Frame.m_Unknown, "predictor"));
    EMPY_CHECK_EQ(empty.m_Frame.m_Domain, std::string("app"));
    EMPY_CHECK_EQ(empty.m_Frame.m_Name, std::string("app-forecast"));
    EMPY_CHECK(empty.m_Descriptor.m_Observables.empty());
    EMPY_CHECK(Descriptor::HasErrors(empty.m_Issues));
    EMPY_CHECK(!empty.ToJson()["valid"].AsBool());

    // a grounded intent with one stray word stays valid and reports only that word
    Interpretation partial = executive.Interpret("Deploy a home humidity predictor with zorblax smoothing");
    EMPY_CHECK_EQ(Join(partial.m_Frame.m_Unknown, ","), std::string("zorblax,smoothing"));
    EMPY_CHECK_EQ(Errors(partial.m_Issues), std::string());
    EMPY_CHECK_EQ(partial.m_Descriptor.m_Observables.at(0).m_Name, std::string("home/hum"));

    // without an ontology nothing can be grounded, but parsing never fails
    Executive bare(AgentEnvironment{});
    SemanticFrame frame = bare.Parse(LISTING_INTENT);
    EMPY_CHECK_EQ(frame.m_Domain, std::string("app"));
    EMPY_CHECK(frame.m_Observables.empty());
    EMPY_CHECK(Has(frame.m_Unknown, "temperature"));
}

EMPY_TEST(OpsAgents, ExecutivePhrases)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});

    EMPY_CHECK_EQ(executive.Parse("home temperature predictor retrained every 15 minutes").m_Schedule,
                  std::string("*/15 * * * *"));
    EMPY_CHECK_EQ(executive.Parse("home temperature predictor retrained every 6 hours").m_Schedule,
                  std::string("0 */6 * * *"));
    EMPY_CHECK_EQ(executive.Parse("home temperature predictor retrained every 7 hours").m_Schedule,
                  std::string("@every 7h"));
    EMPY_CHECK_EQ(executive.Parse("home temperature predictor retrained every 24 hours").m_Schedule,
                  std::string("0 0 * * *"));
    EMPY_CHECK_EQ(executive.Parse("home temperature predictor trained weekly").m_Schedule, std::string("0 0 * * 0"));

    SemanticFrame latency = executive.Parse("home humidity predictor with p99 latency below 20 ms");
    EMPY_CHECK_EQ(Or(latency.m_LatencyMs), 20.0);
    EMPY_CHECK_EQ(latency.m_Statistic, std::string("p99"));
    EMPY_CHECK_EQ(Or(executive.Parse("home humidity predictor responding within 0.5 seconds").m_LatencyMs), 500.0);
    Descriptor::RichMLService slow =
        executive.Generate(latency, "home humidity predictor with p99 latency below 20 ms");
    EMPY_CHECK(slow.m_Inference->m_Regions == std::vector<std::string>{"edge"});
    EMPY_CHECK_EQ(slow.m_Inference->m_Latency->m_Statistic, std::string("p99"));

    // explicit regions: inference by default, training when the phrase follows a training word
    SemanticFrame regions =
        executive.Parse("home temperature controller at the far edge, retrained nightly in the data center");
    EMPY_CHECK(regions.m_InferenceRegions == std::vector<std::string>{"ran"});
    EMPY_CHECK(regions.m_TrainingRegions == std::vector<std::string>{"cloud"});
    EMPY_CHECK_EQ(regions.m_Schedule, std::string("0 0 * * *"));
    SemanticFrame negated = executive.Parse("home temperature controller, not in the cloud");
    EMPY_CHECK(negated.m_InferenceRegions.empty());

    // ranges in another unit are converted into the actionable's unit; explicit actuators are used
    SemanticFrame fahrenheit = executive.Parse("keep the home temperature between 60 and 75 °F");
    EMPY_REQUIRE_EQ(fahrenheit.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(fahrenheit.m_Actionables[0].m_Name, std::string("home/set/temp"));
    EMPY_CHECK(std::abs(Or(fahrenheit.m_Actionables[0].m_Min) - 15.5556) < 1e-3);
    EMPY_CHECK(std::abs(Or(fahrenheit.m_Actionables[0].m_Max) - 23.8889) < 1e-3);
    SemanticFrame fan = executive.Parse("control the home humidity with the fan below 80 percent");
    EMPY_REQUIRE_EQ(fan.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(fan.m_Actionables[0].m_Name, std::string("home/set/fan"));
    EMPY_CHECK_EQ(Or(fan.m_Actionables[0].m_Max), 80.0);
    EMPY_CHECK_EQ(Or(fan.m_Actionables[0].m_Min), 0.0);

    // objectives and a building domain; an accelerator for training
    SemanticFrame energy = executive.Parse("an energy efficient building lighting controller trained on a tpu");
    EMPY_CHECK_EQ(energy.m_Domain, std::string("building"));
    EMPY_CHECK(energy.m_Objectives == std::vector<std::string>{"energy"});
    EMPY_CHECK_EQ(energy.m_Accelerator, std::string("tpu"));
    EMPY_REQUIRE_EQ(energy.m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(energy.m_Actionables[0].m_Name, std::string("building/set/light"));
    EMPY_REQUIRE_EQ(energy.m_Observables.size(), size_t(1));
    EMPY_CHECK_EQ(energy.m_Observables[0].m_Name, std::string("building/lux"));

    // the domain is inferred when only one domain contains the named quantity
    EMPY_CHECK_EQ(executive.Parse("a vibration anomaly detector").m_Domain, std::string("factory"));
    Json reread = Json::Parse(executive.Parse("a vibration anomaly detector").ToJson().Dump());
    EMPY_CHECK_EQ(std::string(reread["domain"].AsString()), std::string("factory"));
}

EMPY_TEST(OpsAgents, ExecutiveDecompose)
{
    Semantics::Ontology ontology = LoadOntology();
    Executive executive(AgentEnvironment{nullptr, nullptr, &ontology});
    WorkflowPlan plan = executive.Decompose(HomeControl());
    EMPY_CHECK_EQ(plan.m_Kind, std::string("deploy"));
    std::vector<std::string> names;
    for (const PlanStep &step : plan.m_Steps)
    {
        names.push_back(step.m_Name + "@" + step.m_Agent + "/" + step.m_Task);
    }
    EMPY_CHECK_EQ(Join(names, " "), std::string("validate@harness/ review@compliance/review "
                                                "place-inference@placement/place place-training@placement/place "
                                                "enact@harness/ schedule@harness/"));
    EMPY_CHECK_EQ(plan.m_Steps[2].m_Input["role"].AsString(), std::string("inference"));
    EMPY_CHECK_EQ(plan.m_Steps[3].m_Input["accelerator"].AsString(), std::string("gpu"));
    EMPY_CHECK_EQ(plan.m_Steps[4].m_Input["instances"].AsInt(), int64_t(2));
    EMPY_CHECK_EQ(plan.m_Steps[5].m_Input["schedule"].AsString(), std::string("0 0 * * *"));

    // inference only, no schedule
    Descriptor::RichMLService inferenceOnly = HomeControl();
    inferenceOnly.m_Training.reset();
    inferenceOnly.m_Inference->m_Replicas = 3;
    WorkflowPlan short_ = executive.Decompose(inferenceOnly);
    names.clear();
    for (const PlanStep &step : short_.m_Steps)
    {
        names.push_back(step.m_Name);
    }
    EMPY_CHECK_EQ(Join(names, ","), std::string("validate,review,place-inference,enact"));
    EMPY_CHECK_EQ(short_.m_Steps.back().m_Input["instances"].AsInt(), int64_t(3));

    // JSON round trip
    WorkflowPlan copy = WorkflowPlan::FromJson(Json::Parse(plan.ToJson().Dump()));
    EMPY_CHECK(copy.ToJson() == plan.ToJson());
    EMPY_CHECK_THROWS(WorkflowPlan::FromJson(Json::Parse(R"({"kind": "deploy", "steps": [{"agent": "x"}]})")));
    EMPY_CHECK_THROWS(WorkflowPlan::FromJson(Json("plan")));
}

EMPY_TEST(OpsAgents, ExecutiveAdvisor)
{
    Semantics::Ontology ontology = LoadOntology();
    uint64_t revision = ontology.Revision();
    const char *intent = "Deploy a greenhouse temperature predictor with latency under 20 ms";
    auto provider = std::make_shared<FakeProvider>(
        [](const Intelligence::Prompt &prompt)
        {
            EMPY_CHECK(prompt.m_Schema.has_value());
            EMPY_CHECK(prompt.m_User.find("greenhouse") != std::string::npos);
            return Json::Parse(R"({
                "concepts": [
                    {"id": "greenhouse", "kind": "domain", "term": "greenhouse", "unit": null, "topic": null,
                     "quantity": null, "min": null, "max": null, "cron": null},
                    {"id": "soil-moisture", "kind": "quantity", "term": "soil moisture", "unit": "percent",
                     "topic": "soil", "quantity": null, "min": null, "max": null, "cron": null},
                    {"id": "temperature", "kind": "region", "term": "temperature", "unit": null, "topic": null,
                     "quantity": null, "min": null, "max": null, "cron": null}
                ],
                "latencyMs": 999, "statistic": "p90", "accuracy": 0.8, "schedule": "0 6 * * *",
                "accelerator": "quantum", "inferenceRegions": ["cloud", "mars"], "trainingRegions": [],
                "objectives": ["comfort"]})");
        });
    Intelligence::ProviderSet providers;
    providers.Set(Intelligence::Capability::Reasoning, provider);
    Executive executive(AgentEnvironment{nullptr, &providers, &ontology});

    Interpretation result = executive.Interpret(intent);
    EMPY_CHECK_EQ(provider->m_Calls, size_t(1));
    // only the grounded, consistent concept evolves the ontology
    EMPY_REQUIRE_EQ(result.m_Evolved.size(), size_t(1));
    EMPY_CHECK_EQ(result.m_Evolved[0].m_Id, std::string("greenhouse"));
    EMPY_CHECK(ontology.Find("greenhouse") != nullptr);
    EMPY_CHECK(ontology.Find("soil-moisture") == nullptr); // its term is not in the intent
    EMPY_CHECK_EQ(ontology.Revision(), revision + 1);
    const SemanticFrame &frame = result.m_Frame;
    EMPY_CHECK_EQ(frame.m_Source, std::string("deterministic+llm:fake"));
    EMPY_CHECK_EQ(frame.m_Domain, std::string("greenhouse"));
    EMPY_CHECK_EQ(Join(frame.m_Unknown, ","), std::string());
    // numbers only fill what the deterministic reading left empty
    EMPY_CHECK_EQ(Or(frame.m_LatencyMs), 20.0);
    EMPY_CHECK_EQ(frame.m_Statistic, std::string("p90"));
    EMPY_CHECK_EQ(Or(frame.m_Accuracy), 0.8);
    EMPY_CHECK_EQ(frame.m_Schedule, std::string("0 6 * * *"));
    EMPY_CHECK(frame.m_Accelerator.empty());
    EMPY_CHECK(frame.m_InferenceRegions == std::vector<std::string>{"cloud"});
    EMPY_CHECK(frame.m_Objectives == std::vector<std::string>{"comfort"});
    const Descriptor::RichMLService &service = result.m_Descriptor;
    EMPY_CHECK_EQ(service.m_Metadata.m_Name, std::string("greenhouse-forecast"));
    EMPY_CHECK_EQ(service.m_Observables.at(0).m_Name, std::string("greenhouse/temp"));
    EMPY_CHECK_EQ(service.m_Lifecycle.m_AccuracyThreshold, 0.8);
    EMPY_CHECK_EQ(Errors(result.m_Issues), std::string());

    // a provider answering garbage is ignored: the deterministic reading stands
    auto broken = std::make_shared<FakeProvider>([](const Intelligence::Prompt &) { return Json("nonsense"); });
    Intelligence::ProviderSet brokenSet;
    brokenSet.Set(Intelligence::Capability::Reasoning, broken);
    Semantics::Ontology fresh = LoadOntology();
    Executive fallback(AgentEnvironment{nullptr, &brokenSet, &fresh});
    Interpretation plain = fallback.Interpret(LISTING_INTENT);
    EMPY_CHECK_EQ(broken->m_Calls, size_t(1));
    EMPY_CHECK_EQ(plain.m_Frame.m_Source, std::string("deterministic"));
    CheckSameDescriptor(plain.m_Descriptor, HomeControl());
}

EMPY_TEST(OpsAgents, ExecutiveOperatorContext)
{
    Semantics::Ontology ontology = LoadOntology();
    Context::KnowledgePlane plane;
    auto source = std::make_shared<ScriptedSource>();
    source->m_Entries = {
        OperatorEntry("site-policy", "Prefer RAN placement for closed-loop home control.", {"home", "placement"}),
        OperatorEntry("factory-policy", "Prefer the cloud for factory anomaly detectors.", {"factory"}),
        OperatorEntry("home-forecasts", "Avoid the cloud for home forecasts.", {"home"}),
        OperatorEntry("home-forecasts-edge", "Avoid the edge for home forecasts.", {"home"})};
    Context::Comas comas(plane, nullptr, source);
    Executive executive(AgentEnvironment{&comas, nullptr, &ontology});
    executive.SetValidation(Regions());

    Interpretation result = executive.Interpret(LISTING_INTENT, "operator-1");
    EMPY_CHECK(source->m_Queries >= 1);
    const Descriptor::PlacementConstraints &placement = result.m_Descriptor.m_Constraints.m_Placement;
    // only the policy about home controllers applies (domain and pattern must match)
    EMPY_CHECK(placement.m_Prefer == std::vector<std::string>{"ran"});
    EMPY_CHECK(placement.m_Avoid.empty());
    EMPY_CHECK_EQ(Errors(result.m_Issues), std::string());
    bool noted = false;
    for (const std::string &note : result.m_Frame.m_Notes)
    {
        noted = noted || note.find("site-policy") != std::string::npos;
    }
    EMPY_CHECK(noted);
    // everything else still equals Listing 2.1
    Descriptor::RichMLService expected = HomeControl();
    expected.m_Constraints.m_Placement.m_Prefer = {"ran"};
    CheckSameDescriptor(result.m_Descriptor, expected);

    // the interpretation is remembered in the agents memory
    std::vector<Context::Entry> memory = plane.Scan(Context::MEMORY);
    EMPY_REQUIRE_EQ(memory.size(), size_t(1));
    EMPY_CHECK_EQ(memory[0].m_Data.GetString("agent"), std::string("executive"));
    EMPY_CHECK_EQ(memory[0].m_Data.GetString("service"), std::string("home-control"));
    EMPY_CHECK_EQ(memory[0].m_Data["data"].GetString("operator"), std::string("operator-1"));
    EMPY_CHECK(memory[0].m_Data["data"].GetBool("valid"));

    // a home predictor is steered by the forecast policy instead
    Interpretation forecast = executive.Interpret("Deploy a home humidity predictor");
    EMPY_CHECK(forecast.m_Descriptor.m_Constraints.m_Placement.m_Prefer.empty());
    // avoiding the cloud keeps the edge for training; avoiding the only inference region (edge) is ignored
    EMPY_CHECK(forecast.m_Descriptor.m_Constraints.m_Placement.m_Avoid == std::vector<std::string>{"cloud"});
    EMPY_CHECK_EQ(Errors(forecast.m_Issues), std::string());
    EMPY_CHECK_EQ(plane.Size(Context::MEMORY), size_t(2));

    // the same flow over the real local retrieval engine and a seeded plane never fails the interpretation
    Context::KnowledgePlane seeded;
    seeded.Seed(eCore::YamlNode::Parse(R"(operator:
  - {id: site-policy, text: "Prefer RAN placement for closed-loop home control.", tags: [home, placement]}
)",
                                       "knowledge.yaml"));
    Context::Comas local(seeded);
    Executive grounded(AgentEnvironment{&local, nullptr, &ontology});
    Interpretation remembered = grounded.Interpret(LISTING_INTENT);
    EMPY_CHECK_EQ(remembered.m_Descriptor.m_Metadata.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(seeded.Size(Context::MEMORY), size_t(1));
}

// ============================================================================ Placement

EMPY_TEST(OpsAgents, PlacementFilters)
{
    Placement placement(AgentEnvironment{});
    Topology::Inventory inventory = KnowledgeInventory();
    Topology::NetworkModel network = KnowledgeNetwork();
    Descriptor::RichMLService service = HomeControl();

    // region filter + locality preference toward the dense ran region
    PlacementPlan plan = placement.Place(service, Role::Inference, inventory, network);
    EMPY_CHECK(plan.m_Nodes == std::vector<std::string>{"ran-1"});
    EMPY_REQUIRE_EQ(plan.m_Candidates.size(), size_t(3));
    EMPY_CHECK_EQ(plan.m_Candidates[0].m_Node, std::string("ran-1"));
    EMPY_CHECK_EQ(plan.m_Candidates[1].m_Node, std::string("edge-1"));
    EMPY_CHECK(plan.m_Candidates[0].m_Score > plan.m_Candidates[1].m_Score);
    EMPY_CHECK_EQ(plan.m_Candidates[0].m_Breakdown["locality"].AsNumber(), 1.0);
    EMPY_CHECK_EQ(plan.m_Candidates[0].m_Breakdown["shard"].AsNumber(), 1.0);
    EMPY_CHECK_EQ(plan.m_Candidates[1].m_Breakdown["locality"].AsNumber(), 0.125);
    EMPY_CHECK(ReasonContains(FindCandidate(plan, "cloud-1"), "region"));
    EMPY_CHECK(!FindCandidate(plan, "cloud-1")->m_Feasible);
    EMPY_CHECK(!plan.m_Relocate);
    EMPY_CHECK(plan.m_Rationale.find("ran-1") != std::string::npos);
    for (const Candidate &candidate : plan.m_Candidates)
    {
        EMPY_CHECK(candidate.m_Score >= 0 && candidate.m_Score <= 1);
    }

    // avoid
    Descriptor::RichMLService avoid = service;
    avoid.m_Constraints.m_Placement.m_Avoid = {"ran"};
    PlacementPlan avoided = placement.Place(avoid, Role::Inference, inventory, network);
    EMPY_CHECK(avoided.m_Nodes == std::vector<std::string>{"edge-1"});
    EMPY_CHECK(ReasonContains(FindCandidate(avoided, "ran-1"), "avoided"));

    // pin
    Descriptor::RichMLService pinned = service;
    pinned.m_Constraints.m_Placement.m_Nodes = {"edge-1"};
    EMPY_CHECK(placement.Place(pinned, Role::Inference, inventory, network).m_Nodes ==
               std::vector<std::string>{"edge-1"});

    // capacity: ran-1 almost full
    Topology::Inventory busy = inventory;
    busy.Find("ran-1")->m_UsedCpu = 3.5;
    PlacementPlan full = placement.Place(service, Role::Inference, busy, network);
    EMPY_CHECK(full.m_Nodes == std::vector<std::string>{"edge-1"});
    EMPY_CHECK(ReasonContains(FindCandidate(full, "ran-1"), "insufficient cpu"));
    // ... unless the service itself holds that reservation (released before fitting)
    busy.Find("ran-1")->m_UsedCpu = 3.5;
    PlacementPlan own = placement.Place(service, Role::Inference, busy, network, {"ran-1"});
    EMPY_CHECK(FindCandidate(own, "ran-1")->m_Feasible);
    EMPY_CHECK(own.m_Nodes == std::vector<std::string>{"ran-1"});
    Topology::Inventory memory = inventory;
    memory.Find("ran-1")->m_UsedMemory = uint64_t(7800) << 20;
    EMPY_CHECK(ReasonContains(FindCandidate(placement.Place(service, Role::Inference, memory, network), "ran-1"),
                              "insufficient memory"));

    // accelerator kind and count
    Descriptor::RichMLService gpu = service;
    gpu.m_Inference->m_Resources.m_Accelerator = "gpu";
    gpu.m_Inference->m_Resources.m_AcceleratorCount = 1;
    PlacementPlan accelerated = placement.Place(gpu, Role::Inference, inventory, network);
    EMPY_CHECK(accelerated.m_Nodes == std::vector<std::string>{"edge-1"});
    EMPY_CHECK(ReasonContains(FindCandidate(accelerated, "ran-1"), "accelerator gpu"));
    gpu.m_Inference->m_Resources.m_AcceleratorCount = 2;
    EMPY_CHECK(placement.Place(gpu, Role::Inference, inventory, network).m_Nodes.empty());

    // latency feasibility: Latency(node region, densest data region ran) ≤ maxMs / 2
    Descriptor::RichMLService tight = service;
    tight.m_Inference->m_Latency->m_MaxMs = 3; // edge (2 ms) > 1.5 ms
    PlacementPlan latency = placement.Place(tight, Role::Inference, inventory, network);
    EMPY_CHECK(latency.m_Nodes == std::vector<std::string>{"ran-1"});
    EMPY_CHECK(ReasonContains(FindCandidate(latency, "edge-1"), "latency"));
    tight.m_Inference->m_Latency->m_MaxMs = 0.8; // even same-region 0.5 ms > 0.4 ms
    PlacementPlan infeasible = placement.Place(tight, Role::Inference, inventory, network);
    EMPY_CHECK(infeasible.m_Nodes.empty());
    EMPY_CHECK(infeasible.m_Rationale.find("infeasible") != std::string::npos);

    // a role the service does not declare
    Descriptor::RichMLService noTraining = service;
    noTraining.m_Training.reset();
    EMPY_CHECK(placement.Place(noTraining, Role::Training, inventory, network).m_Nodes.empty());
}

EMPY_TEST(OpsAgents, PlacementTrainingAndReplicas)
{
    Placement placement(AgentEnvironment{});
    Topology::Inventory inventory = KnowledgeInventory();
    Topology::NetworkModel network = KnowledgeNetwork();
    Descriptor::RichMLService service = HomeControl();

    // training goes to gpu capacity in edge/cloud; ran (npu only, not a training region) is filtered
    PlacementPlan training = placement.Place(service, Role::Training, inventory, network);
    EMPY_REQUIRE_EQ(training.m_Nodes.size(), size_t(1));
    EMPY_CHECK_EQ(training.m_Nodes[0], std::string("cloud-1")); // 7 of 8 GPUs stay free there, none on edge-1
    EMPY_CHECK(FindCandidate(training, "edge-1")->m_Feasible);
    EMPY_CHECK(!FindCandidate(training, "ran-1")->m_Feasible);
    EMPY_CHECK(FindCandidate(training, "cloud-1")->m_Breakdown["accelerator"].AsNumber() > 0.8);
    EMPY_CHECK_EQ(training.m_Role, Role::Training);
    // edge-1's single GPU taken: only cloud-1 remains
    Topology::Inventory used = inventory;
    used.Find("cloud-1")->m_UsedAccelerators["gpu"] = 8;
    PlacementPlan edgeOnly = placement.Place(service, Role::Training, used, network);
    EMPY_CHECK(edgeOnly.m_Nodes == std::vector<std::string>{"edge-1"});

    // replicas land on distinct nodes; too few feasible nodes → no placement
    Topology::Inventory wide = inventory;
    wide.m_Nodes.push_back(MakeNode("ran-2", "ran", 4, 8));
    Descriptor::RichMLService replicated = service;
    replicated.m_Inference->m_Replicas = 3;
    PlacementPlan three = placement.Place(replicated, Role::Inference, wide, network);
    EMPY_REQUIRE_EQ(three.m_Nodes.size(), size_t(3));
    EMPY_CHECK_EQ(three.m_Nodes[0], std::string("ran-1"));
    EMPY_CHECK_EQ(three.m_Nodes[1], std::string("ran-2"));
    EMPY_CHECK_EQ(three.m_Nodes[2], std::string("edge-1"));
    replicated.m_Inference->m_Replicas = 4;
    EMPY_CHECK(placement.Place(replicated, Role::Inference, wide, network).m_Nodes.empty());

    // JSON round trip
    PlacementPlan copy = PlacementPlan::FromJson(Json::Parse(three.ToJson().Dump()));
    EMPY_CHECK(copy.ToJson() == three.ToJson());
    EMPY_CHECK_EQ(ParseRole("training"), Role::Training);
    EMPY_CHECK_EQ(std::string(RoleName(Role::Inference)), std::string("inference"));
    EMPY_CHECK_THROWS(ParseRole("gpu"));
}

EMPY_TEST(OpsAgents, PlacementRelocationAndSession)
{
    Placement placement(AgentEnvironment{});
    Topology::Inventory inventory = KnowledgeInventory();
    Topology::NetworkModel network = KnowledgeNetwork();
    Descriptor::RichMLService service = HomeControl();

    PlacementPlan fresh = placement.Place(service, Role::Inference, inventory, network);
    double ran = FindCandidate(fresh, "ran-1")->m_Score;
    double edge = FindCandidate(fresh, "edge-1")->m_Score;
    EMPY_CHECK(ran - edge > 0.2);

    // running on edge-1: ran-1 is better by more than the default 0.1 hysteresis → relocate
    PlacementPlan move = placement.Place(service, Role::Inference, inventory, network, {"edge-1"});
    EMPY_CHECK(move.m_Relocate);
    EMPY_CHECK(move.m_Nodes == std::vector<std::string>{"ran-1"});
    EMPY_CHECK(move.m_CurrentScore > edge); // session bonus for the hosting node
    EMPY_CHECK(move.m_Rationale.find("relocate") != std::string::npos);

    // a hysteresis above the gain keeps the current placement
    Descriptor::RichMLService sticky = service;
    sticky.m_Constraints.m_Placement.m_Hysteresis = 0.9;
    PlacementPlan keep = placement.Place(sticky, Role::Inference, inventory, network, {"edge-1"});
    EMPY_CHECK(!keep.m_Relocate);
    EMPY_CHECK(keep.m_Nodes == std::vector<std::string>{"edge-1"});
    Descriptor::RichMLService pinnedSession = service;
    pinnedSession.m_Constraints.m_Placement.m_AllowRelocation = false;
    PlacementPlan disabled = placement.Place(pinnedSession, Role::Inference, inventory, network, {"edge-1"});
    EMPY_CHECK(!disabled.m_Relocate);
    EMPY_CHECK(disabled.m_Nodes == std::vector<std::string>{"edge-1"});
    EMPY_CHECK(disabled.m_Rationale.find("relocation disabled") != std::string::npos);
    // already on the best node
    PlacementPlan stay = placement.Place(service, Role::Inference, inventory, network, {"ran-1"});
    EMPY_CHECK(!stay.m_Relocate);
    EMPY_CHECK(stay.m_Nodes == std::vector<std::string>{"ran-1"});

    // session stickiness between two equivalent nodes: the hosting node wins the tie-break
    Topology::Inventory twins = inventory;
    twins.m_Nodes.push_back(MakeNode("ran-2", "ran", 4, 8, {{"npu", 1}}));
    PlacementPlan tie = placement.Place(service, Role::Inference, twins, network);
    EMPY_CHECK(tie.m_Nodes == std::vector<std::string>{"ran-1"}); // equal scores → node id
    EMPY_CHECK_EQ(FindCandidate(tie, "ran-1")->m_Score, FindCandidate(tie, "ran-2")->m_Score);
    PlacementPlan session = placement.Place(service, Role::Inference, twins, network, {"ran-2"});
    EMPY_CHECK_EQ(session.m_Candidates[0].m_Node, std::string("ran-2"));
    EMPY_CHECK_EQ(session.m_Candidates[0].m_Breakdown["session"].AsNumber(), 1.0);
    EMPY_CHECK(!session.m_Relocate);
    EMPY_CHECK(session.m_Nodes == std::vector<std::string>{"ran-2"});

    // session locality by node-local broker: a node behind the broker the current session lives on keeps the
    // session (0.5), a node behind another broker migrates it (0)
    Topology::Inventory brokered = twins;
    brokered.m_Nodes.push_back(MakeNode("ran-3", "ran", 4, 8, {{"npu", 1}}));
    brokered.Find("ran-1")->m_Broker = "10.0.0.1:5600";
    brokered.Find("ran-2")->m_Broker = "10.0.0.1:5600";
    brokered.Find("ran-3")->m_Broker = "10.0.0.2:5600";
    PlacementPlan local = placement.Place(service, Role::Inference, brokered, network, {"ran-1"});
    EMPY_CHECK_EQ(FindCandidate(local, "ran-1")->m_Breakdown["session"].AsNumber(), 1.0);
    EMPY_CHECK_EQ(FindCandidate(local, "ran-2")->m_Breakdown["session"].AsNumber(), 0.5);
    EMPY_CHECK_EQ(FindCandidate(local, "ran-3")->m_Breakdown["session"].AsNumber(), 0.0);
    EMPY_CHECK(FindCandidate(local, "ran-2")->m_Score > FindCandidate(local, "ran-3")->m_Score);

    // a current node that vanished from the inventory scores 0, so relocation happens
    PlacementPlan gone = placement.Place(service, Role::Inference, inventory, network, {"ran-9"});
    EMPY_CHECK(gone.m_Relocate);
    EMPY_CHECK_EQ(gone.m_CurrentScore, 0.0);
}

EMPY_TEST(OpsAgents, PlacementAdvisorAndWeights)
{
    Topology::Inventory twins = KnowledgeInventory();
    twins.m_Nodes.push_back(MakeNode("ran-2", "ran", 4, 8, {{"npu", 1}}));
    Topology::NetworkModel network = KnowledgeNetwork();
    Descriptor::RichMLService service = HomeControl();

    // the Planning advisor may reorder near-best candidates only
    auto advisor = std::make_shared<FakeProvider>(
        [](const Intelligence::Prompt &)
        { return Json::Parse(R"({"order": ["edge-1", "ran-2", "ran-1"], "rationale": "spread load"})"); });
    Intelligence::ProviderSet providers;
    providers.Set(Intelligence::Capability::Planning, advisor);
    Placement advised(AgentEnvironment{nullptr, &providers, nullptr});
    PlacementPlan plan = advised.Place(service, Role::Inference, twins, network);
    EMPY_CHECK_EQ(advisor->m_Calls, size_t(1));
    EMPY_CHECK(plan.m_Nodes == std::vector<std::string>{"ran-2"}); // edge-1 is outside the 0.05 window
    EMPY_CHECK_EQ(plan.m_Advisor, std::string("llm:fake"));
    EMPY_CHECK_EQ(plan.m_Candidates[2].m_Node, std::string("edge-1"));

    // a single near-best candidate: the advisor is not consulted
    Topology::Inventory single = KnowledgeInventory();
    PlacementPlan alone = advised.Place(service, Role::Inference, single, network);
    EMPY_CHECK_EQ(advisor->m_Calls, size_t(1));
    EMPY_CHECK(alone.m_Advisor.empty());

    // weights from YAML
    PlacementWeights defaults = PlacementWeights::FromYaml(eCore::YamlNode());
    EMPY_CHECK_EQ(defaults.m_Locality, 0.25);
    PlacementWeights custom =
        PlacementWeights::FromYaml(eCore::YamlNode::Parse("weights: {locality: 0, latency: 1}")["weights"]);
    EMPY_CHECK_EQ(custom.m_Locality, 0.0);
    EMPY_CHECK_EQ(custom.m_Latency, 1.0);
    EMPY_CHECK_EQ(custom.m_Session, 0.10);
    EMPY_CHECK_EQ(custom.ToJson()["latency"].AsNumber(), 1.0);
    EMPY_CHECK_THROWS(PlacementWeights::FromYaml(eCore::YamlNode::Parse("w: {localty: 1}")["w"]));
    EMPY_CHECK_THROWS(PlacementWeights::FromYaml(eCore::YamlNode::Parse("w: {latency: -1}")["w"]));
    // only latency counts: the same-region ran nodes are best, without locality
    Placement latencyOnly(AgentEnvironment{}, custom);
    PlacementPlan byLatency = latencyOnly.Place(service, Role::Inference, twins, network);
    EMPY_CHECK_EQ(byLatency.m_Candidates[0].m_Region, std::string("ran"));
}

// ============================================================================ Lifecycle

EMPY_TEST(OpsAgents, LifecycleDecisionTable)
{
    Lifecycle lifecycle(AgentEnvironment{});
    Descriptor::LifecyclePolicy policy; // 0.9 / 0.02 / 60 s, minSamples 20, autoRollback
    const uint64_t now = 10'000'000;
    Json drift = Json::Parse(R"({"kind": "accuracy", "value": 0.7, "threshold": 0.9})");

    LifecycleState state;
    state.m_Service = "svc";
    state.m_HasTraining = true;

    // drift / schedule guards
    EMPY_CHECK_EQ(lifecycle.OnDrift(state, policy, drift, now).m_Action, LifecycleAction::Retrain);
    EMPY_CHECK(lifecycle.OnDrift(state, policy, drift, now).m_Reason.find("accuracy") != std::string::npos);
    EMPY_CHECK_EQ(lifecycle.OnSchedule(state, policy, now).m_Action, LifecycleAction::Retrain);
    LifecycleState noTraining = state;
    noTraining.m_HasTraining = false;
    EMPY_CHECK_EQ(lifecycle.OnDrift(noTraining, policy, drift, now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(lifecycle.OnSchedule(noTraining, policy, now).m_Action, LifecycleAction::None);
    LifecycleState running = state;
    running.m_Training = true;
    EMPY_CHECK_EQ(lifecycle.OnDrift(running, policy, drift, now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(lifecycle.OnSchedule(running, policy, now).m_Action, LifecycleAction::None);
    LifecycleState cooling = state;
    cooling.m_LastRetrain = now - 30'000;
    LifecycleDecision cooled = lifecycle.OnDrift(cooling, policy, drift, now);
    EMPY_CHECK_EQ(cooled.m_Action, LifecycleAction::None);
    EMPY_CHECK(cooled.m_Reason.find("cooldown") != std::string::npos);
    EMPY_CHECK_EQ(lifecycle.OnSchedule(cooling, policy, now).m_Action, LifecycleAction::None);
    cooling.m_LastRetrain = now - 60'000;
    EMPY_CHECK_EQ(lifecycle.OnDrift(cooling, policy, drift, now).m_Action, LifecycleAction::Retrain);
    // the decision carries the accuracy it was based on (the harness forwards it as provenance)
    EMPY_CHECK_EQ(lifecycle.OnDrift(state, policy, drift, now).m_Observed.value_or(-1), 0.7);
    // latency drift is a placement concern: retraining cannot shorten the observation-to-action path
    LifecycleDecision latency =
        lifecycle.OnDrift(state, policy, Json::Parse(R"({"kind": "latency", "value": 12, "threshold": 5})"), now);
    EMPY_CHECK_EQ(latency.m_Action, LifecycleAction::None);
    EMPY_CHECK(latency.m_Reason.find("placement") != std::string::npos);
    EMPY_CHECK(!latency.m_Observed.has_value());
    EMPY_CHECK_EQ(lifecycle.OnDrift(state, policy, Json::Parse(R"({"value": 0.5})"), now).m_Action,
                  LifecycleAction::Retrain); // no kind: accuracy drift

    // bootstrap
    LifecycleDecision bootstrap = lifecycle.OnStatus(state, policy, Json::Parse(R"({"state": "waiting-model"})"), now);
    EMPY_CHECK_EQ(bootstrap.m_Action, LifecycleAction::Retrain);
    EMPY_CHECK(bootstrap.m_Reason.find("bootstrap") != std::string::npos);
    EMPY_CHECK_EQ(lifecycle.OnStatus(running, policy, Json::MakeObject(), now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(lifecycle.OnStatus(noTraining, policy, Json::MakeObject(), now).m_Action, LifecycleAction::None);
    LifecycleState retried = state;
    retried.m_LastRetrain = now - 2'000; // a failed bootstrap is retried after min(cooldown, 5 s)
    EMPY_CHECK_EQ(lifecycle.OnStatus(retried, policy, Json::MakeObject(), now).m_Action, LifecycleAction::None);
    retried.m_LastRetrain = now - 5'000;
    EMPY_CHECK_EQ(lifecycle.OnStatus(retried, policy, Json::MakeObject(), now).m_Action, LifecycleAction::Retrain);
    EMPY_CHECK_EQ(lifecycle.OnStatus(state, policy, Json::Parse(R"({"modelVersion": 2})"), now).m_Action,
                  LifecycleAction::None);

    // evaluate: first model
    LifecycleDecision first = lifecycle.Evaluate(state, policy, Artifact(1, 0.95), now);
    EMPY_CHECK_EQ(first.m_Action, LifecycleAction::Promote);
    EMPY_CHECK_EQ(first.m_Target.value_or(0), uint32_t(1));
    EMPY_CHECK_EQ(lifecycle.Evaluate(state, policy, Artifact(1, 0.85), now).m_Action, LifecycleAction::Reject);
    EMPY_CHECK_EQ(lifecycle.Evaluate(state, policy, Artifact(0, 0.99), now).m_Action, LifecycleAction::Reject);

    // evaluate against an active model
    LifecycleState active = state;
    active.m_History = {Record(1, 0.92, "retired"), Record(2, 0.93, "active")};
    active.m_Active = 2;
    active.m_LastPromotion = now - 120'000;
    EMPY_CHECK_EQ(active.NextVersion(), uint32_t(3));
    EMPY_CHECK_EQ(lifecycle.Evaluate(active, policy, Artifact(3, 0.96), now).m_Action, LifecycleAction::Promote);
    LifecycleDecision small = lifecycle.Evaluate(active, policy, Artifact(3, 0.94), now);
    EMPY_CHECK_EQ(small.m_Action, LifecycleAction::Reject);
    EMPY_CHECK(small.m_Reason.find("improves") != std::string::npos);
    EMPY_CHECK_EQ(lifecycle.Evaluate(active, policy, Artifact(3, 0.80), now).m_Action, LifecycleAction::Reject);
    EMPY_CHECK_EQ(lifecycle.Evaluate(active, policy, Artifact(2, 0.99), now).m_Action, LifecycleAction::Reject);
    LifecycleState recent = active;
    recent.m_LastPromotion = now - 10'000;
    LifecycleDecision cooldown = lifecycle.Evaluate(recent, policy, Artifact(3, 0.99), now);
    EMPY_CHECK_EQ(cooldown.m_Action, LifecycleAction::Reject);
    EMPY_CHECK(cooldown.m_Reason.find("cooldown") != std::string::npos);

    // metrics: rollback to a good previous model, otherwise retrain
    auto metrics = [](double accuracy, int64_t samples)
    { return Json::MakeObject({{"accuracy", accuracy}, {"samples", samples}, {"modelVersion", 2}}); };
    EMPY_CHECK_EQ(lifecycle.OnMetrics(active, policy, metrics(0.95, 50), now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(lifecycle.OnMetrics(active, policy, metrics(0.50, 5), now).m_Action, LifecycleAction::None);
    LifecycleDecision rollback = lifecycle.OnMetrics(active, policy, metrics(0.60, 50), now);
    EMPY_CHECK_EQ(rollback.m_Action, LifecycleAction::Rollback);
    EMPY_CHECK_EQ(rollback.m_Target.value_or(0), uint32_t(1));
    EMPY_CHECK_EQ(rollback.m_Observed.value_or(-1), 0.60);
    EMPY_CHECK_EQ(lifecycle
                      .OnMetrics(active, policy,
                                 Json::Parse(R"({"accuracy": {"value": 0.6}, "samples": 50, "modelVersion": 2})"), now)
                      .m_Action,
                  LifecycleAction::Rollback);
    Descriptor::LifecyclePolicy manual = policy;
    manual.m_AutoRollback = false;
    LifecycleDecision retrain = lifecycle.OnMetrics(active, manual, metrics(0.60, 50), now);
    EMPY_CHECK_EQ(retrain.m_Action, LifecycleAction::Retrain);
    EMPY_CHECK_EQ(retrain.m_Observed.value_or(-1), 0.60);
    LifecycleState weakPrevious = active;
    weakPrevious.m_History[0].m_Accuracy = 0.85; // below threshold: no rollback target
    EMPY_CHECK_EQ(lifecycle.OnMetrics(weakPrevious, policy, metrics(0.60, 50), now).m_Action, LifecycleAction::Retrain);
    weakPrevious.m_LastRetrain = now - 1'000;
    EMPY_CHECK_EQ(lifecycle.OnMetrics(weakPrevious, policy, metrics(0.60, 50), now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(lifecycle.OnMetrics(state, policy, metrics(0.60, 50), now).m_Action, LifecycleAction::None);
    EMPY_CHECK_EQ(
        lifecycle.OnMetrics(active, policy, Json::Parse(R"({"accuracy": null, "samples": 50})"), now).m_Action,
        LifecycleAction::None);
    // metrics of an older model version are ignored
    EMPY_CHECK_EQ(
        lifecycle
            .OnMetrics(active, policy, Json::MakeObject({{"accuracy", 0.1}, {"samples", 50}, {"modelVersion", 1}}), now)
            .m_Action,
        LifecycleAction::None);
}

EMPY_TEST(OpsAgents, LifecycleStateJson)
{
    LifecycleState state;
    state.m_Service = "home-control";
    state.m_History = {Record(1, 0.91, "retired"), Record(2, 0.5, "rejected"), Record(3, 0.95, "active")};
    state.m_Active = 3;
    state.m_LastRetrain = 123;
    state.m_LastPromotion = 456;
    state.m_HasTraining = true;
    EMPY_REQUIRE(state.ActiveRecord() != nullptr);
    EMPY_CHECK_EQ(state.ActiveRecord()->m_Version, uint32_t(3));
    EMPY_REQUIRE(state.PreviousRecord() != nullptr);
    EMPY_CHECK_EQ(state.PreviousRecord()->m_Version, uint32_t(1));
    EMPY_CHECK_EQ(state.NextVersion(), uint32_t(4));
    LifecycleState copy = LifecycleState::FromJson(Json::Parse(state.ToJson().Dump()));
    EMPY_CHECK(copy.ToJson() == state.ToJson());
    EMPY_CHECK(LifecycleState().ToJson()["active"].IsNull());
    EMPY_CHECK_EQ(LifecycleState().NextVersion(), uint32_t(1));
    EMPY_CHECK_THROWS(LifecycleState::FromJson(Json::Parse(R"({"history": [{"version": 0}]})")));
    EMPY_CHECK_THROWS(LifecycleState::FromJson(Json::Parse(R"({"training": "yes"})")));

    LifecycleDecision decision;
    decision.m_Action = LifecycleAction::Rollback;
    decision.m_Reason = "drift";
    decision.m_Target = 2;
    decision.m_Observed = 0.42;
    LifecycleDecision decoded = LifecycleDecision::FromJson(Json::Parse(decision.ToJson().Dump()));
    EMPY_CHECK_EQ(decoded.m_Action, LifecycleAction::Rollback);
    EMPY_CHECK_EQ(decoded.m_Target.value_or(0), uint32_t(2));
    EMPY_CHECK_EQ(decoded.m_Observed.value_or(-1), 0.42);
    EMPY_CHECK(LifecycleDecision().ToJson()["observed"].IsNull());
    EMPY_CHECK(!LifecycleDecision::FromJson(Json::Parse(R"({"action": "none"})")).m_Observed.has_value());
    EMPY_CHECK_THROWS(LifecycleDecision::FromJson(Json::Parse(R"({"action": "none", "observed": "high"})")));
    EMPY_CHECK_EQ(std::string(ActionName(LifecycleAction::Promote)), std::string("promote"));
    EMPY_CHECK_EQ(ParseAction("retrain"), LifecycleAction::Retrain);
    EMPY_CHECK_THROWS(ParseAction("explode"));
    EMPY_CHECK_THROWS(LifecycleDecision::FromJson(Json::Parse(R"({"action": "explode"})")));
}

// ============================================================================ Compliance

EMPY_TEST(OpsAgents, PolicySetYaml)
{
    PolicySet policies = SeedPolicies();
    EMPY_REQUIRE_EQ(policies.m_Ownership.size(), size_t(1));
    EMPY_CHECK_EQ(policies.m_Ownership[0].m_Filter, std::string("home/set/#"));
    EMPY_CHECK_EQ(policies.m_Ownership[0].m_Owner, std::string("home-ops"));
    EMPY_REQUIRE_EQ(policies.m_Safety.size(), size_t(1));
    EMPY_CHECK_EQ(policies.m_Safety[0].m_Filter, std::string("home/set/temp"));
    EMPY_CHECK_EQ(policies.m_Safety[0].m_Unit, std::string("celsius"));
    EMPY_CHECK_EQ(Or(policies.m_Safety[0].m_Min), 10.0);
    EMPY_CHECK_EQ(Or(policies.m_Safety[0].m_Max), 30.0);
    EMPY_CHECK(!policies.m_Permissions.empty());
    EMPY_CHECK(!policies.m_Tenants.empty());
    EMPY_CHECK(Has(policies.m_DeniedTopics, "home/set/door/#"));
    EMPY_CHECK(PolicySet::FromJson(Json::Parse(policies.ToJson().Dump())).ToJson() == policies.ToJson());
    EMPY_CHECK(PolicySet::FromYaml(eCore::YamlNode()).ToJson() == PolicySet().ToJson());

    auto load = [](const char *text) { return PolicySet::FromYaml(eCore::YamlNode::Parse(text, "policies.yaml")); };
    EMPY_CHECK_THROWS(load("deny_topics: [a/b]"));
    EMPY_CHECK_THROWS(load("denied_topics: ['a/#/b']"));
    EMPY_CHECK_THROWS(load("ownership: [{filter: a/#}]"));
    EMPY_CHECK_THROWS(load("safety: [{filter: a/b, min: 5, max: 1}]"));
    EMPY_CHECK_THROWS(load("safety: [{filter: a/b}]"));
    EMPY_CHECK_THROWS(load("tenants: [{name: x, root: 'x/+'}]"));
    EMPY_CHECK_THROWS(load("permissions: [{allow: [a/b]}]"));
    EMPY_CHECK_THROWS(load("max_action_rate_per_minute: -1"));
    try
    {
        load("safety: [{filter: a/b, min: 5, max: 1}]");
    }
    catch (const eCore::ConfigError &error)
    {
        EMPY_CHECK(std::string(error.what()).find("safety[0]") != std::string::npos);
    }
    PolicySet tenant = load("tenants: [{name: acme}]");
    EMPY_CHECK_EQ(tenant.m_Tenants.at(0).m_Root, std::string("acme"));
    EMPY_CHECK_EQ(ParseVerdict("escalate"), Verdict::Escalate);
    EMPY_CHECK_EQ(std::string(VerdictName(Verdict::Amended)), std::string("amended"));
    EMPY_CHECK_THROWS(ParseVerdict("maybe"));
}

EMPY_TEST(OpsAgents, CompliancePolicyCriteria)
{
    Semantics::Ontology ontology = LoadOntology();
    Compliance compliance(AgentEnvironment{nullptr, nullptr, &ontology}, SeedPolicies());
    Descriptor::RichMLService home = HomeControl();

    // HomeControl alone: approved (an unowned service on an owned domain is only a warning)
    ComplianceReport approved = compliance.Review(home, {});
    EMPY_CHECK_EQ(approved.m_Verdict, Verdict::Approved);
    EMPY_CHECK(HasFinding(approved, "ownership", "warning"));
    EMPY_CHECK(!approved.m_Amended.has_value());
    EMPY_CHECK(approved.m_Question.empty());
    Descriptor::RichMLService owned = home;
    owned.m_Constraints.m_Owner = "home-ops";
    ComplianceReport clean = compliance.Review(owned, {});
    EMPY_CHECK_EQ(clean.m_Verdict, Verdict::Approved);
    EMPY_CHECK(clean.m_Findings.empty());

    // denied topic
    Descriptor::RichMLService door = owned;
    door.m_Actionables.push_back(MakeActionable("home/set/door", "", 0, 1));
    ComplianceReport denied = compliance.Review(door, {});
    EMPY_CHECK_EQ(denied.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(denied, "policy", "error"));

    // ownership by another team
    Descriptor::RichMLService foreign = home;
    foreign.m_Constraints.m_Owner = "facility-team";
    ComplianceReport stolen = compliance.Review(foreign, {});
    EMPY_CHECK_EQ(stolen.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(stolen, "ownership", "error"));

    // permissions: "*" deny, and the energy-saver allow list
    Descriptor::RichMLService alarm = owned;
    alarm.m_Actionables.push_back(MakeActionable("home/set/alarm/siren", "", 0, 1));
    alarm.m_Metadata.m_Intent += " and the alarm siren";
    ComplianceReport deny = compliance.Review(alarm, {});
    EMPY_CHECK_EQ(deny.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(deny, "permission", "error"));
    Descriptor::RichMLService saver = EnergySaver();
    ComplianceReport allowed = compliance.Review(saver, {});
    EMPY_CHECK_EQ(allowed.m_Verdict, Verdict::Approved);
    EMPY_CHECK(!HasFinding(allowed, "permission"));
    saver.m_Actionables.push_back(MakeActionable("home/set/fan", "percent", 0, 100));
    saver.m_Metadata.m_Intent += " and the fan";
    ComplianceReport outside = compliance.Review(saver, {});
    EMPY_CHECK_EQ(outside.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(outside, "permission", "error"));

    // safety: out-of-envelope bounds are clamped (amended), missing bounds filled
    Descriptor::RichMLService wide = owned;
    wide.m_Actionables[0].m_Min = 5;
    wide.m_Actionables[0].m_Max = 40;
    ComplianceReport clamped = compliance.Review(wide, {});
    EMPY_CHECK_EQ(clamped.m_Verdict, Verdict::Amended);
    EMPY_REQUIRE(clamped.m_Amended.has_value());
    EMPY_CHECK_EQ(Or(clamped.m_Amended->m_Actionables[0].m_Min), 10.0);
    EMPY_CHECK_EQ(Or(clamped.m_Amended->m_Actionables[0].m_Max), 30.0);
    EMPY_REQUIRE_EQ(clamped.m_Resolutions.size(), size_t(1));
    EMPY_CHECK_EQ(clamped.m_Resolutions[0]["type"].AsString(), std::string("clamp"));
    EMPY_CHECK_EQ(clamped.m_Resolutions[0]["topic"].AsString(), std::string("home/set/temp"));
    Descriptor::RichMLService unbounded = owned;
    unbounded.m_Actionables[0].m_Min.reset();
    unbounded.m_Actionables[0].m_Max = 20;
    ComplianceReport filled = compliance.Review(unbounded, {});
    EMPY_CHECK_EQ(filled.m_Verdict, Verdict::Amended);
    EMPY_CHECK_EQ(Or(filled.m_Amended->m_Actionables[0].m_Min), 10.0);
    EMPY_CHECK_EQ(Or(filled.m_Amended->m_Actionables[0].m_Max), 20.0);
    // "celcius" is the same unit for the ontology
    Descriptor::RichMLService spelled = wide;
    spelled.m_Actionables[0].m_Unit = "celcius";
    EMPY_CHECK_EQ(compliance.Review(spelled, {}).m_Verdict, Verdict::Amended);
    Descriptor::RichMLService kelvin = owned;
    kelvin.m_Actionables[0].m_Unit = "fahrenheit";
    EMPY_CHECK_EQ(compliance.Review(kelvin, {}).m_Verdict, Verdict::Rejected);

    // safety: an empty intersection rejects
    Descriptor::RichMLService hot = owned;
    hot.m_Actionables[0].m_Min = 35;
    hot.m_Actionables[0].m_Max = 40;
    ComplianceReport empty = compliance.Review(hot, {});
    EMPY_CHECK_EQ(empty.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(empty, "safety", "error"));
    EMPY_CHECK(!empty.m_Amended.has_value());
}

EMPY_TEST(OpsAgents, ComplianceTenantAndIntent)
{
    Semantics::Ontology ontology = LoadOntology();
    PolicySet policies = SeedPolicies();
    Compliance compliance(AgentEnvironment{nullptr, nullptr, &ontology}, policies);

    // a tenant service must stay under its root
    Descriptor::RichMLService tenant;
    tenant.m_Metadata.m_Name = "acme-forecast";
    tenant.m_Metadata.m_Intent = "forecast the acme temperature";
    Descriptor::Observable temp;
    temp.m_Name = "tenants/acme/temp";
    tenant.m_Observables = {temp};
    tenant.m_Actionables = {MakeActionable("tenants/acme/forecast/temp", "celsius", {}, {})};
    tenant.m_Inference = Descriptor::InferenceSpec{};
    tenant.m_Inference->m_Resources = {1, uint64_t(1) << 30, "", 0};
    tenant.m_Inference->m_Regions = {"edge"};
    tenant.m_Constraints.m_Security.m_Tenant = "acme";
    ComplianceReport inside = compliance.Review(tenant, {});
    EMPY_CHECK_EQ(inside.m_Verdict, Verdict::Approved);
    EMPY_CHECK(!HasFinding(inside, "tenant"));
    Descriptor::RichMLService leaking = tenant;
    leaking.m_Observables.push_back(Descriptor::Observable{"home/temp", "celsius", "", 1});
    ComplianceReport leak = compliance.Review(leaking, {});
    EMPY_CHECK_EQ(leak.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(leak, "tenant", "error"));
    Descriptor::RichMLService unknownTenant = tenant;
    unknownTenant.m_Constraints.m_Security.m_Tenant = "globex";
    EMPY_CHECK_EQ(compliance.Review(unknownTenant, {}).m_Verdict, Verdict::Rejected);
    // another service may not reach into a tenant root
    Descriptor::RichMLService intruder = HomeControl();
    intruder.m_Observables.push_back(Descriptor::Observable{"tenants/+/temp", "celsius", "", 1});
    ComplianceReport intrusion = compliance.Review(intruder, {});
    EMPY_CHECK_EQ(intrusion.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(intrusion, "tenant", "error"));
    // encryption required for tenants
    PolicySet strict = policies;
    strict.m_RequireEncryptionForTenants = true;
    Compliance encrypted(AgentEnvironment{nullptr, nullptr, &ontology}, strict);
    EMPY_CHECK_EQ(encrypted.Review(tenant, {}).m_Verdict, Verdict::Rejected);
    tenant.m_Constraints.m_Security.m_Encryption = true;
    EMPY_CHECK_EQ(encrypted.Review(tenant, {}).m_Verdict, Verdict::Approved);

    // declared intent mismatch → escalate with a question
    Descriptor::RichMLService mismatch = HomeControl();
    mismatch.m_Constraints.m_Owner = "home-ops";
    mismatch.m_Metadata.m_Intent = "Forecast the home humidity for the dashboard";
    ComplianceReport escalated = compliance.Review(mismatch, {});
    EMPY_CHECK_EQ(escalated.m_Verdict, Verdict::Escalate);
    EMPY_CHECK(HasFinding(escalated, "intent", "warning"));
    EMPY_CHECK(escalated.m_Question.find("home/set/temp") != std::string::npos);
    // the actuator's ontology quantity justifies it ("thermostat" → cooling → temperature)
    mismatch.m_Metadata.m_Intent = "Drive the thermostat from humidity readings";
    EMPY_CHECK_EQ(compliance.Review(mismatch, {}).m_Verdict, Verdict::Approved);
    // a leaf level named verbatim also justifies it
    mismatch.m_Metadata.m_Intent = "Set temp from humidity";
    EMPY_CHECK_EQ(compliance.Review(mismatch, {}).m_Verdict, Verdict::Approved);
    // no intent declared: nothing to compare
    mismatch.m_Metadata.m_Intent.clear();
    EMPY_CHECK_EQ(compliance.Review(mismatch, {}).m_Verdict, Verdict::Approved);

    // action-rate ceiling amends unlimited declarations
    PolicySet limited = policies;
    limited.m_MaxActionRatePerMinute = 12;
    Compliance capped(AgentEnvironment{nullptr, nullptr, &ontology}, limited);
    ComplianceReport rate = capped.Review(HomeControl(), {});
    EMPY_CHECK_EQ(rate.m_Verdict, Verdict::Amended);
    EMPY_CHECK_EQ(rate.m_Amended->m_Actionables[0].m_MaxRatePerMinute, uint32_t(12));
    EMPY_CHECK_EQ(rate.m_Resolutions.at(0)["type"].AsString(), std::string("rate"));
}

EMPY_TEST(OpsAgents, ComplianceConflicts)
{
    Semantics::Ontology ontology = LoadOntology();
    Compliance compliance(AgentEnvironment{nullptr, nullptr, &ontology}, SeedPolicies());
    Descriptor::RichMLService home = HomeControl();
    Descriptor::RichMLService saver = EnergySaver();

    // lower priority with another actionable left → amended: the overlapping actionable is dropped
    Descriptor::RichMLService heating = saver;
    heating.m_Metadata.m_Name = "heat-saver";
    heating.m_Metadata.m_Intent = "Save energy on home heating and the temperature setpoint";
    heating.m_Actionables.push_back(MakeActionable("home/set/heating", "percent", 0, 100));
    heating.m_Inference->m_Actuation = "home/set/temp";
    ComplianceReport dropped = compliance.Review(heating, {home});
    EMPY_CHECK_EQ(dropped.m_Verdict, Verdict::Amended);
    EMPY_REQUIRE(dropped.m_Amended.has_value());
    EMPY_REQUIRE_EQ(dropped.m_Amended->m_Actionables.size(), size_t(1));
    EMPY_CHECK_EQ(dropped.m_Amended->m_Actionables[0].m_Name, std::string("home/set/heating"));
    EMPY_CHECK_EQ(dropped.m_Amended->ActuationTopic(), std::string("home/set/heating"));
    EMPY_CHECK(HasFinding(dropped, "conflict"));
    bool drop = false;
    for (const Json &resolution : dropped.m_Resolutions)
    {
        drop = drop || (resolution["type"].AsString() == "drop" && resolution["topic"].AsString() == "home/set/temp");
    }
    EMPY_CHECK(drop);
    EMPY_CHECK(Descriptor::Validate(*dropped.m_Amended).empty() ||
               !Descriptor::HasErrors(Descriptor::Validate(*dropped.m_Amended)));

    // HomeControl vs EnergySaver: the lower-priority saver has nothing left → rejected
    ComplianceReport saverReport = compliance.Review(saver, {home});
    EMPY_CHECK_EQ(saverReport.m_Verdict, Verdict::Rejected);
    EMPY_CHECK(HasFinding(saverReport, "conflict", "error"));
    // HomeControl after EnergySaver: higher priority → escalate, naming services, topic and objectives
    ComplianceReport homeReport = compliance.Review(home, {saver});
    EMPY_CHECK_EQ(homeReport.m_Verdict, Verdict::Escalate);
    EMPY_CHECK(homeReport.m_Question.find("home-control") != std::string::npos);
    EMPY_CHECK(homeReport.m_Question.find("energy-saver") != std::string::npos);
    EMPY_CHECK(homeReport.m_Question.find("home/set/temp") != std::string::npos);
    EMPY_CHECK(homeReport.m_Question.find("energy") != std::string::npos);
    EMPY_CHECK_EQ(homeReport.m_Findings.back().m_Service, std::string("energy-saver"));

    // equal priority → escalate; conflicting ontology objectives are named
    Descriptor::RichMLService comfort = home;
    comfort.m_Metadata.m_Name = "comfort-control";
    comfort.m_Constraints.m_Objectives = {"comfort"};
    Descriptor::RichMLService peer = saver;
    peer.m_Constraints.m_Priority = 0;
    ComplianceReport equal = compliance.Review(comfort, {peer});
    EMPY_CHECK_EQ(equal.m_Verdict, Verdict::Escalate);
    EMPY_CHECK(equal.m_Question.find("neither has priority") != std::string::npos);
    EMPY_CHECK(Findings(equal).find("conflicting objectives") != std::string::npos);
    EMPY_CHECK(Findings(equal).find("energy vs comfort") != std::string::npos ||
               Findings(equal).find("comfort vs energy") != std::string::npos);

    // the same service name (a redeploy) and non-overlapping services do not conflict
    EMPY_CHECK_EQ(compliance.Review(home, {home}).m_Verdict, Verdict::Approved);
    Descriptor::RichMLService humidity = home;
    humidity.m_Metadata.m_Name = "humidity-control";
    humidity.m_Actionables = {MakeActionable("home/set/fan", "percent", 0, 100)};
    humidity.m_Metadata.m_Intent = "control the home humidity with the fan";
    EMPY_CHECK_EQ(compliance.Review(humidity, {home, saver}).m_Verdict, Verdict::Approved);

    // precedence: an error beats an escalation; an escalation keeps its amendments
    Descriptor::RichMLService both = home;
    both.m_Actionables[0].m_Min = 0; // clamped
    ComplianceReport amendedEscalation = compliance.Review(both, {saver});
    EMPY_CHECK_EQ(amendedEscalation.m_Verdict, Verdict::Escalate);
    EMPY_CHECK(amendedEscalation.m_Amended.has_value());
    both.m_Constraints.m_Owner = "facility-team";
    EMPY_CHECK_EQ(compliance.Review(both, {saver}).m_Verdict, Verdict::Rejected);

    // report JSON round trip
    ComplianceReport copy = ComplianceReport::FromJson(Json::Parse(amendedEscalation.ToJson().Dump()));
    EMPY_CHECK(copy.ToJson() == amendedEscalation.ToJson());
    EMPY_CHECK(ComplianceReport().ToJson()["amended"].IsNull());
    EMPY_CHECK_THROWS(ComplianceReport::FromJson(Json::Parse(R"({"verdict": "perhaps"})")));
    EMPY_CHECK_THROWS(Finding::FromJson(Json::Parse(R"({"severity": "fatal"})")));
}

EMPY_TEST(OpsAgents, ComplianceQuestionWriter)
{
    Semantics::Ontology ontology = LoadOntology();
    auto writer = std::make_shared<FakeProvider>(
        [](const Intelligence::Prompt &prompt)
        {
            EMPY_CHECK(prompt.m_User.find("Draft question") != std::string::npos);
            return Json::Parse(
                R"({"question": "Allow home-control to share home/set/temp with energy-saver despite the risk?"})");
        });
    Intelligence::ProviderSet providers;
    providers.Set(Intelligence::Capability::Generation, writer);
    Compliance compliance(AgentEnvironment{nullptr, &providers, &ontology}, SeedPolicies());
    ComplianceReport report = compliance.Review(HomeControl(), {EnergySaver()});
    EMPY_CHECK_EQ(report.m_Verdict, Verdict::Escalate);
    EMPY_CHECK_EQ(report.m_Question,
                  std::string("Allow home-control to share home/set/temp with energy-saver despite the risk?"));
    EMPY_CHECK_EQ(writer->m_Calls, size_t(1));
    // not consulted when nothing escalates
    EMPY_CHECK_EQ(compliance.Review(HomeControl(), {}).m_Verdict, Verdict::Approved);
    EMPY_CHECK_EQ(writer->m_Calls, size_t(1));

    // a question that does not name the candidate falls back to the deterministic text
    auto vague = std::make_shared<FakeProvider>([](const Intelligence::Prompt &)
                                                { return Json::Parse(R"({"question": "Approve?"})"); });
    Intelligence::ProviderSet vagueSet;
    vagueSet.Set(Intelligence::Capability::Generation, vague);
    Compliance fallback(AgentEnvironment{nullptr, &vagueSet, &ontology}, SeedPolicies());
    ComplianceReport plain = fallback.Review(HomeControl(), {EnergySaver()});
    EMPY_CHECK(plain.m_Question.find("energy-saver") != std::string::npos);
    EMPY_CHECK(plain.m_Question != "Approve?");
}
