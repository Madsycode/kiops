#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Descriptor/Model.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Descriptor/Validator.h>
#include <eOps/Descriptor/Profiles.h>
#include <eOps/Descriptor/Cron.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <eCore/Network/MMQP.h>
#include <filesystem>
#include <functional>

using namespace eOps;
using namespace eOps::Descriptor;

namespace
{
    using eTest::Ops::RepositoryRoot;

    // Listing 2.1 as printed (typographic hyphen gaps removed; "celcius" and the elided constraints kept)
    const char *LISTING_21 = R"(apiVersion: mlops/v1
kind: RichMLService

metadata:
    name: home-control
    version: "1.0.0"

specs:
    observables:
        - name: home/temp
          unit: celcius
        - name: home/hum
          unit: percent

    actionables:
        - name: home/set/temp
          unit: celcius
          min: 16
          max: 24

    inference:
        resources:
            cpu: "1"
            memory: "1Gi"
        endpoint: "/v1/predict"
        regions: ["ran", "edge"]
        latency:
            observable: control-latency
            path: observation-to-action
            statistic: p95
            windowSeconds: 60
            maxMs: 5

    training:
        resources:
            cpu: "2"
            memory: "4Gi"
            accelerator: gpu
        regions: ["edge", "cloud"]
        schedule: "0 0 * * *"
        dataset: traffic-telemetry

    constraints: {}

    lifecycle:
        accuracyThreshold: 0.90
        minImprovement: 0.02
        cooldownSeconds: 60
)";

    bool HasIssue(const std::vector<Issue> &issues, std::string_view path, Severity severity = Severity::Error)
    {
        for (const Issue &issue : issues)
        {
            if (issue.m_Path == path && issue.m_Severity == severity)
            {
                return true;
            }
        }
        return false;
    }

    size_t ErrorCount(const std::vector<Issue> &issues)
    {
        size_t count = 0;
        for (const Issue &issue : issues)
        {
            count += issue.m_Severity == Severity::Error ? 1 : 0;
        }
        return count;
    }

    std::string Describe(const std::vector<Issue> &issues)
    {
        std::string text;
        for (const Issue &issue : issues)
        {
            text += std::string(SeverityName(issue.m_Severity)) + " " + issue.m_Path + ": " + issue.m_Message + "; ";
        }
        return text;
    }

    std::string ErrorText(const std::function<void()> &body)
    {
        try
        {
            body();
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        return {};
    }

    std::string Iso(std::chrono::system_clock::time_point time)
    {
        return IsoTime(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count()));
    }

    std::chrono::system_clock::time_point At(int32_t y, uint32_t mo, uint32_t d, int32_t h, int32_t mi, int32_t s = 0)
    {
        using namespace std::chrono;
        return sys_days(year(y) / month(mo) / day(d)) + hours(h) + minutes(mi) + seconds(s);
    }

    // a fully populated descriptor exercising every extension field
    RichMLService Populated()
    {
        RichMLService service = ParseDescriptor(LISTING_21);
        service.m_Metadata.m_Description = "Multi-line \"quoted\" text:\n  indented: yes\n# not a comment";
        service.m_Metadata.m_Intent = "keep 16-24 °C: comfort first";
        service.m_Metadata.m_Labels = {{"flag", "true"}, {"mlops.empy.io/tier", "gold"}, {"team", "home"}};
        service.m_Observables.push_back(Observable{"home/+/presence", "", "occupancy sensors", 0});
        service.m_Actionables.push_back(Actionable{"home/cooling/#", "percent", 0.0, 100.5, 12, "fan domain", 0});
        service.m_Inference->m_Replicas = 2;
        service.m_Inference->m_Image = "registry.local/home:1";
        service.m_Inference->m_Actuation = "home/set/temp";
        service.m_Inference->m_Resources.m_Cpu = 0.25;
        service.m_Inference->m_Resources.m_Memory = 1536;
        service.m_Training->m_Resources.m_AcceleratorCount = 2;
        service.m_Training->m_Label = "home/label/temp";
        service.m_Training->m_Window = 256;
        service.m_Training->m_MinSamples = 16;
        service.m_Training->m_Holdout = 0.2;
        service.m_Constraints.m_Priority = -3;
        service.m_Constraints.m_Owner = "home-operator";
        service.m_Constraints.m_Objectives = {"comfort", "energy"};
        service.m_Constraints.m_Security = {true, "", "svc-home"};
        service.m_Constraints.m_Placement = {{"ran"}, {"cloud"}, {"ran-1"}, false, 0.25};
        service.m_Lifecycle.m_EvaluationWindow = 32;
        service.m_Lifecycle.m_MinSamples = 8;
        service.m_Lifecycle.m_Tolerance = 0.75;
        service.m_Lifecycle.m_AutoRollback = false;
        return service;
    }
} // namespace

EMPY_TEST(OpsDescriptor, Listing21)
{
    RichMLService service = ParseDescriptor(LISTING_21, "listing-2.1");
    EMPY_CHECK_EQ(service.m_ApiVersion, std::string("mlops/v1"));
    EMPY_CHECK_EQ(service.m_Kind, std::string("RichMLService"));
    EMPY_CHECK_EQ(service.m_Metadata.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(service.m_Metadata.m_Version, std::string("1.0.0"));
    EMPY_REQUIRE_EQ(service.m_Observables.size(), size_t{2});
    EMPY_CHECK_EQ(service.m_Observables[0].m_Name, std::string("home/temp"));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Unit, std::string("celcius"));
    EMPY_CHECK_EQ(service.m_Observables[1].m_Name, std::string("home/hum"));
    EMPY_CHECK_EQ(service.m_Observables[1].m_Unit, std::string("percent"));
    EMPY_CHECK_EQ(service.m_Observables[0].m_Qos, uint8_t{1});
    EMPY_REQUIRE_EQ(service.m_Actionables.size(), size_t{1});
    const Actionable &setpoint = service.m_Actionables[0];
    EMPY_CHECK_EQ(setpoint.m_Name, std::string("home/set/temp"));
    EMPY_REQUIRE(setpoint.m_Min.has_value() && setpoint.m_Max.has_value());
    EMPY_CHECK_EQ(*setpoint.m_Min, 16.0);
    EMPY_CHECK_EQ(*setpoint.m_Max, 24.0);

    EMPY_REQUIRE(service.m_Inference.has_value());
    const InferenceSpec &inference = *service.m_Inference;
    EMPY_CHECK_EQ(inference.m_Resources.m_Cpu, 1.0);
    EMPY_CHECK_EQ(inference.m_Resources.m_Memory, uint64_t{1} << 30);
    EMPY_CHECK(inference.m_Resources.m_Accelerator.empty());
    EMPY_CHECK_EQ(inference.m_Resources.m_AcceleratorCount, 0u);
    EMPY_CHECK_EQ(inference.m_Endpoint, std::string("/v1/predict"));
    EMPY_CHECK(inference.m_Regions == std::vector<std::string>({"ran", "edge"}));
    EMPY_CHECK_EQ(inference.m_Replicas, 1u);
    EMPY_REQUIRE(inference.m_Latency.has_value());
    EMPY_CHECK_EQ(inference.m_Latency->m_Observable, std::string("control-latency"));
    EMPY_CHECK_EQ(inference.m_Latency->m_Path, std::string("observation-to-action"));
    EMPY_CHECK_EQ(inference.m_Latency->m_Statistic, std::string("p95"));
    EMPY_CHECK_EQ(inference.m_Latency->m_WindowSeconds, 60u);
    EMPY_CHECK_EQ(inference.m_Latency->m_MaxMs, 5.0);

    EMPY_REQUIRE(service.m_Training.has_value());
    const TrainingSpec &training = *service.m_Training;
    EMPY_CHECK_EQ(training.m_Resources.m_Cpu, 2.0);
    EMPY_CHECK_EQ(training.m_Resources.m_Memory, uint64_t{4} << 30);
    EMPY_CHECK_EQ(training.m_Resources.m_Accelerator, std::string("gpu"));
    EMPY_CHECK_EQ(training.m_Resources.m_AcceleratorCount, 1u); // implied by a named accelerator
    EMPY_CHECK(training.m_Regions == std::vector<std::string>({"edge", "cloud"}));
    EMPY_CHECK_EQ(training.m_Schedule, std::string("0 0 * * *"));
    EMPY_CHECK_EQ(training.m_Dataset, std::string("traffic-telemetry"));
    EMPY_CHECK(training.m_Label.empty());
    EMPY_CHECK_EQ(training.m_Window, 512u);
    EMPY_CHECK_EQ(training.m_Holdout, 0.25);

    EMPY_CHECK_EQ(service.m_Lifecycle.m_AccuracyThreshold, 0.9);
    EMPY_CHECK_EQ(service.m_Lifecycle.m_MinImprovement, 0.02);
    EMPY_CHECK_EQ(service.m_Lifecycle.m_CooldownSeconds, 60u);
    EMPY_CHECK_EQ(service.m_Constraints.m_Priority, 0);

    EMPY_CHECK_EQ(service.ActuationTopic(), std::string("home/set/temp"));
    std::vector<Issue> issues = Validate(service);
    EMPY_CHECK_EQ(ErrorCount(issues), size_t{0});
    EMPY_CHECK_EQ(Describe(issues), std::string());
    EMPY_CHECK(!HasErrors(Validate(service, ValidationOptions{{"ran", "edge", "cloud"}})));
}

EMPY_TEST(OpsDescriptor, ShippedDescriptors)
{
    std::filesystem::path directory = RepositoryRoot() / "Configs" / "MLOps";
    RichMLService home = LoadDescriptor(directory / "HomeControl.yaml");
    EMPY_CHECK_EQ(home.m_Metadata.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(home.m_Metadata.m_Version, std::string("1.0.0"));
    EMPY_CHECK(home.m_Metadata.m_Intent.starts_with("Deploy a home temperature controller"));
    EMPY_CHECK(home.m_Metadata.m_Intent.find("retrained daily on a GPU") != std::string::npos);
    EMPY_CHECK(home.m_Metadata.m_Intent.find('\n') == std::string::npos);
    EMPY_REQUIRE_EQ(home.m_Observables.size(), size_t{2});
    EMPY_CHECK_EQ(home.m_Observables[0].m_Unit, std::string("celsius"));
    EMPY_REQUIRE(home.m_Training.has_value() && home.m_Inference.has_value());
    EMPY_CHECK_EQ(home.m_Training->m_Label, std::string("home/label/temp"));
    EMPY_CHECK_EQ(home.m_Training->m_Dataset, std::string("home-telemetry"));
    EMPY_CHECK_EQ(home.m_Training->m_Resources.m_Accelerator, std::string("gpu"));
    EMPY_CHECK_EQ(home.m_Inference->m_Latency->m_MaxMs, 5.0);
    EMPY_CHECK(home.m_Inference->m_Regions == std::vector<std::string>({"ran", "edge"}));
    EMPY_CHECK_EQ(Describe(Validate(home, ValidationOptions{{"ran", "edge", "cloud"}})), std::string());

    std::string text = ReadFile(directory / "HomeControl.yaml");
    Json document = FromYaml(eCore::YamlNode::Parse(text));
    EMPY_CHECK(UnknownKeys(document).empty());

    RichMLService saver = LoadDescriptor(directory / "EnergySaver.yaml");
    EMPY_CHECK_EQ(saver.m_Metadata.m_Name, std::string("energy-saver"));
    EMPY_CHECK(saver.m_Constraints.m_Priority < home.m_Constraints.m_Priority);
    EMPY_CHECK(saver.m_Constraints.m_Objectives == std::vector<std::string>({"energy"}));
    EMPY_REQUIRE(saver.FindActionable("home/set/temp") != nullptr);
    EMPY_CHECK_EQ(Describe(Validate(saver, ValidationOptions{{"ran", "edge", "cloud"}})), std::string());
    EMPY_CHECK(UnknownKeys(FromYaml(eCore::YamlNode::Load((directory / "EnergySaver.yaml").string()))).empty());

    // the CRD is valid YAML for the mlops.empy.io group
    eCore::YamlNode crd = eCore::YamlNode::Load((directory / "Kubernetes" / "RichMLService.crd.yaml").string());
    EMPY_CHECK_EQ(crd["kind"].As<std::string>(), std::string("CustomResourceDefinition"));
    EMPY_CHECK_EQ(crd["spec"]["group"].As<std::string>(), std::string("mlops.empy.io"));
    EMPY_CHECK_EQ(crd["spec"]["names"]["kind"].As<std::string>(), std::string("RichMLService"));
    EMPY_CHECK(
        crd["spec"]["versions"]
            .Items()[0]["schema"]["openAPIV3Schema"]["properties"]["spec"]["x-kubernetes-preserve-unknown-fields"]
            .As<bool>());
}

EMPY_TEST(OpsDescriptor, SpecAliasAndErrors)
{
    std::string alias = LISTING_21;
    alias.replace(alias.find("specs:"), 6, "spec:");
    RichMLService service = ParseDescriptor(alias);
    EMPY_CHECK_EQ(service.m_Observables.size(), size_t{2});
    EMPY_CHECK(service.m_Training.has_value());

    std::string both = std::string(LISTING_21) + "spec: {}\n";
    EMPY_CHECK(ErrorText([&] { ParseDescriptor(both); }).find("both 'specs' and 'spec'") != std::string::npos);

    // errors name the origin and the dotted path
    std::string cpu = LISTING_21;
    cpu.replace(cpu.find("cpu: \"1\""), 8, "cpu: \"lots\"");
    std::string message = ErrorText([&] { ParseDescriptor(cpu, "bad.yaml"); });
    EMPY_CHECK(message.starts_with("bad.yaml: specs.inference.resources.cpu:"));

    std::string qos = "apiVersion: mlops/v1\nspecs:\n  observables:\n    - {name: a/b, qos: x}\n";
    EMPY_CHECK(ErrorText([&] { ParseDescriptor(qos); }).find("specs.observables[0].qos") != std::string::npos);
    std::string list = "specs:\n  observables: {name: a}\n";
    EMPY_CHECK(ErrorText([&] { ParseDescriptor(list); }).find("specs.observables: expected a list") !=
               std::string::npos);
    std::string count = "specs:\n  training:\n    resources: {acceleratorCount: 1.5}\n";
    EMPY_CHECK(ErrorText([&] { ParseDescriptor(count); }).find("specs.training.resources.acceleratorCount") !=
               std::string::npos);
    EMPY_CHECK_THROWS(ParseDescriptor("- not\n- a mapping\n"));
    EMPY_CHECK_THROWS(ParseDescriptor("metadata: [unclosed\n"));
    EMPY_CHECK_THROWS(ParseDescriptor("specs:\n  lifecycle:\n    autoRollback: maybe\n"));
    EMPY_CHECK_THROWS(LoadDescriptor(std::filesystem::temp_directory_path() / ("missing-" + NewId({}) + ".yaml")));

    // plain-string list items name observables and actionables; identity fields have no parse defaults
    RichMLService minimal = ParseDescriptor("specs:\n  observables: [home/temp]\n  actionables: [home/set/#]\n");
    EMPY_CHECK(minimal.m_ApiVersion.empty() && minimal.m_Kind.empty());
    EMPY_CHECK_EQ(minimal.m_Observables[0].m_Name, std::string("home/temp"));
    EMPY_CHECK_EQ(minimal.m_Actionables[0].m_Name, std::string("home/set/#"));
    EMPY_CHECK(HasIssue(Validate(minimal), "apiVersion"));
    EMPY_CHECK(HasIssue(Validate(minimal), "kind"));
    EMPY_CHECK(HasIssue(Validate(minimal), "metadata.name"));
    EMPY_CHECK(HasIssue(Validate(minimal), "specs"));
}

EMPY_TEST(OpsDescriptor, Quantities)
{
    EMPY_CHECK_EQ(ParseCpu("1"), 1.0);
    EMPY_CHECK_EQ(ParseCpu("500m"), 0.5);
    EMPY_CHECK_EQ(ParseCpu("2"), 2.0);
    EMPY_CHECK_EQ(ParseCpu("0.5"), 0.5);
    EMPY_CHECK_EQ(ParseCpu(" 250m "), 0.25);
    EMPY_CHECK_THROWS(ParseCpu(""));
    EMPY_CHECK_THROWS(ParseCpu("m"));
    EMPY_CHECK_THROWS(ParseCpu("-1"));
    EMPY_CHECK_THROWS(ParseCpu("1 core"));
    EMPY_CHECK_THROWS(ParseCpu("1e400"));

    EMPY_CHECK_EQ(ParseMemory("1Gi"), uint64_t{1} << 30);
    EMPY_CHECK_EQ(ParseMemory("512Mi"), uint64_t{512} << 20);
    EMPY_CHECK_EQ(ParseMemory("64Ki"), uint64_t{64} << 10);
    EMPY_CHECK_EQ(ParseMemory("1G"), uint64_t{1000000000});
    EMPY_CHECK_EQ(ParseMemory("500M"), uint64_t{500000000});
    EMPY_CHECK_EQ(ParseMemory("1k"), uint64_t{1000});
    EMPY_CHECK_EQ(ParseMemory("1024"), uint64_t{1024});
    EMPY_CHECK_EQ(ParseMemory("1.5Gi"), uint64_t{3} << 29);
    EMPY_CHECK_EQ(ParseMemory("15Ei"), uint64_t{15} << 60);
    EMPY_CHECK_THROWS(ParseMemory("16Ei"));
    EMPY_CHECK_THROWS(ParseMemory("Gi"));
    EMPY_CHECK_THROWS(ParseMemory("1Gb"));
    EMPY_CHECK_THROWS(ParseMemory("-1Gi"));
    EMPY_CHECK_THROWS(ParseMemory("99999999999999999999"));

    EMPY_CHECK_EQ(FormatCpu(1), std::string("1"));
    EMPY_CHECK_EQ(FormatCpu(2), std::string("2"));
    EMPY_CHECK_EQ(FormatCpu(0.5), std::string("500m"));
    EMPY_CHECK_EQ(FormatCpu(1.25), std::string("1250m"));
    EMPY_CHECK_EQ(FormatCpu(0), std::string("0"));
    EMPY_CHECK_THROWS(FormatCpu(-1));
    EMPY_CHECK_EQ(FormatMemory(uint64_t{4} << 30), std::string("4Gi"));
    EMPY_CHECK_EQ(FormatMemory(uint64_t{512} << 20), std::string("512Mi"));
    EMPY_CHECK_EQ(FormatMemory(uint64_t{64} << 10), std::string("64Ki"));
    EMPY_CHECK_EQ(FormatMemory(1536), std::string("1536"));
    EMPY_CHECK_EQ(FormatMemory(1000), std::string("1000"));
    EMPY_CHECK_EQ(FormatMemory(0), std::string("0"));
    for (uint64_t bytes : {uint64_t{1}, uint64_t{1} << 30, uint64_t{3} << 29, uint64_t{7} << 50})
    {
        EMPY_CHECK_EQ(ParseMemory(FormatMemory(bytes)), bytes);
    }

    // numeric YAML quantities are accepted too
    RichMLService numeric =
        ParseDescriptor("specs:\n  inference:\n    resources: {cpu: 0.5, memory: 2048, accelerator: NPU}\n");
    EMPY_CHECK_EQ(numeric.m_Inference->m_Resources.m_Cpu, 0.5);
    EMPY_CHECK_EQ(numeric.m_Inference->m_Resources.m_Memory, uint64_t{2048});
    EMPY_CHECK_EQ(numeric.m_Inference->m_Resources.m_Accelerator, std::string("npu"));
    EMPY_CHECK_THROWS(ParseDescriptor("specs:\n  inference:\n    resources: {memory: -5}\n"));
}

EMPY_TEST(OpsDescriptor, ValidatorMatrix)
{
    const RichMLService base = ParseDescriptor(LISTING_21);
    const ValidationOptions regions{{"ran", "edge", "cloud"}};
    auto check = [&](const std::function<void(RichMLService &)> &mutate, std::string_view path,
                     Severity severity = Severity::Error, const ValidationOptions &options = {})
    {
        RichMLService service = base;
        mutate(service);
        std::vector<Issue> issues = Validate(service, options);
        bool found = HasIssue(issues, path, severity);
        if (!found)
        {
            ::eTest::Fail(__FILE__, __LINE__, "missing issue at '" + std::string(path) + "': " + Describe(issues));
        }
        return issues;
    };

    check([](RichMLService &s) { s.m_ApiVersion = "mlops/v2"; }, "apiVersion");
    check([](RichMLService &s) { s.m_Kind = "Deployment"; }, "kind");
    check([](RichMLService &s) { s.m_Metadata.m_Name = "Home_Control"; }, "metadata.name");
    check([](RichMLService &s) { s.m_Metadata.m_Name = std::string(64, 'a'); }, "metadata.name");
    check([](RichMLService &s) { s.m_Metadata.m_Version = "1.0"; }, "metadata.version");
    check([](RichMLService &s) { s.m_Metadata.m_Version = "1.02.0"; }, "metadata.version");
    check([](RichMLService &s) { s.m_Observables.clear(); }, "specs.observables");
    check([](RichMLService &s) { s.m_Observables[0].m_Name = "home/#/temp"; }, "specs.observables[0].name");
    check([](RichMLService &s) { s.m_Observables[1].m_Name = "home/temp"; }, "specs.observables[1].name");
    check([](RichMLService &s) { s.m_Observables[0].m_Name = "$SYS/broker"; }, "specs.observables[0].name");
    check([](RichMLService &s) { s.m_Observables[0].m_Name = "mlops/intent"; }, "specs.observables[0].name");
    check([](RichMLService &s) { s.m_Observables[0].m_Name = "+/temp"; }, "specs.observables[0].name",
          Severity::Warning);
    check([](RichMLService &s) { s.m_Observables[0].m_Qos = 2; }, "specs.observables[0].qos");
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "home/set+"; }, "specs.actionables[0].name");
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "$share/g/home/set"; }, "specs.actionables[0].name");
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "#"; }, "specs.actionables[0].name");
    check([](RichMLService &s) { s.m_Actionables.push_back(s.m_Actionables[0]); }, "specs.actionables[1].name");
    check(
        [](RichMLService &s)
        {
            s.m_Actionables[0].m_Min = 30;
            s.m_Actionables[0].m_Max = 10;
        },
        "specs.actionables[0]");
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "home/#"; }, "specs.actionables[0].name",
          Severity::Warning);
    check([](RichMLService &s) { s.m_Inference->m_Resources.m_Cpu = 0; }, "specs.inference.resources.cpu");
    check([](RichMLService &s) { s.m_Inference->m_Resources.m_Memory = 0; }, "specs.inference.resources.memory");
    check([](RichMLService &s) { s.m_Training->m_Resources.m_Cpu = 0; }, "specs.training.resources.cpu");
    check([](RichMLService &s) { s.m_Training->m_Resources.m_Accelerator = "fpga"; },
          "specs.training.resources.accelerator");
    check([](RichMLService &s) { s.m_Training->m_Resources.m_AcceleratorCount = 0; },
          "specs.training.resources.acceleratorCount");
    check([](RichMLService &s) { s.m_Inference->m_Regions.clear(); }, "specs.inference.regions");
    check([](RichMLService &s) { s.m_Training->m_Regions.push_back("mars"); }, "specs.training.regions[2]",
          Severity::Error, regions);
    check([](RichMLService &s) { s.m_Inference->m_Latency->m_Statistic = "p42"; }, "specs.inference.latency.statistic");
    check([](RichMLService &s) { s.m_Inference->m_Latency->m_WindowSeconds = 0; },
          "specs.inference.latency.windowSeconds");
    check([](RichMLService &s) { s.m_Inference->m_Latency->m_MaxMs = 0; }, "specs.inference.latency.maxMs");
    check([](RichMLService &s) { s.m_Inference->m_Replicas = 0; }, "specs.inference.replicas");
    check([](RichMLService &s) { s.m_Training->m_Schedule = "61 * * * *"; }, "specs.training.schedule");
    check([](RichMLService &s) { s.m_Training->m_Dataset.clear(); }, "specs.training.dataset");
    check([](RichMLService &s) { s.m_Training->m_Label = "home/label/+"; }, "specs.training.label");
    check([](RichMLService &s) { s.m_Training->m_Holdout = 1; }, "specs.training.holdout");
    check([](RichMLService &s) { s.m_Training->m_Holdout = 0; }, "specs.training.holdout");
    check([](RichMLService &s) { s.m_Training->m_MinSamples = 1000; }, "specs.training.minSamples");
    check([](RichMLService &s) { s.m_Lifecycle.m_AccuracyThreshold = 1.5; }, "specs.lifecycle.accuracyThreshold");
    check([](RichMLService &s) { s.m_Lifecycle.m_MinImprovement = -0.1; }, "specs.lifecycle.minImprovement");
    check([](RichMLService &s) { s.m_Inference->m_Actuation = "home/set/humidity"; }, "specs.inference.actuation");
    check([](RichMLService &s) { s.m_Inference->m_Actuation = "home/set/+"; }, "specs.inference.actuation");
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "home/set/#"; }, "specs.inference.actuation");
    check([](RichMLService &s) { s.m_Constraints.m_Security.m_Tenant = "acme"; }, "specs.observables[0].name");
    check([](RichMLService &s) { s.m_Constraints.m_Security.m_Tenant = "acme"; }, "specs.actionables[0].name");
    check([](RichMLService &s) { s.m_Constraints.m_Security.m_Tenant = "a/b"; }, "specs.constraints.security.tenant");
    check(
        [](RichMLService &s)
        {
            s.m_Constraints.m_Placement.m_Prefer = {"ran"};
            s.m_Constraints.m_Placement.m_Avoid = {"ran"};
        },
        "specs.constraints.placement", Severity::Warning);
    check([](RichMLService &s) { s.m_Actionables[0].m_Name = "home/temp"; }, "specs.actionables[0].name",
          Severity::Warning);
    check(
        [](RichMLService &s)
        {
            s.m_Inference.reset();
            s.m_Training.reset();
        },
        "specs");

    // valid variants: a wildcard permission domain with an explicit actuation, tenant-prefixed topics,
    // training-only services and predictors without actionables
    RichMLService domain = base;
    domain.m_Actionables[0].m_Name = "home/set/#";
    domain.m_Inference->m_Actuation = "home/set/temp";
    EMPY_CHECK_EQ(Describe(Validate(domain)), std::string());
    EMPY_CHECK_EQ(domain.FindActionable("home/set/temp"), &domain.m_Actionables[0]);
    EMPY_CHECK(domain.FindActionable("home/set") != nullptr); // '#' also matches the parent level
    EMPY_CHECK(domain.FindActionable("home/get/temp") == nullptr);
    EMPY_CHECK(domain.FindActionable("home/set/+") == nullptr);

    RichMLService tenant = base;
    tenant.m_Constraints.m_Security.m_Tenant = "home";
    EMPY_CHECK(!HasErrors(Validate(tenant)));
    // shared subscriptions are judged by the filter behind "$share/<group>/"
    tenant.m_Observables[1].m_Name = "$share/workers/home/hum";
    EMPY_CHECK_EQ(Describe(Validate(tenant)), std::string());
    tenant.m_Observables[1].m_Name = "$share/workers/mlops/intent";
    EMPY_CHECK(HasIssue(Validate(tenant), "specs.observables[1].name"));

    RichMLService trainingOnly = base;
    trainingOnly.m_Inference.reset();
    trainingOnly.m_Actionables.clear();
    EMPY_CHECK_EQ(Describe(Validate(trainingOnly)), std::string());

    RichMLService predictor = base;
    predictor.m_Actionables.clear();
    EMPY_CHECK_EQ(Describe(Validate(predictor)), std::string());
    EMPY_CHECK(predictor.ActuationTopic().empty());

    // a different control root moves the reserved space
    RichMLService rooted = base;
    rooted.m_Observables[0].m_Name = "mlops/home/temp";
    EMPY_CHECK(!HasErrors(Validate(rooted, ValidationOptions{{}, {"gpu"}, "ops"})));

    const Json json = ToJson(Validate(rooted));
    EMPY_REQUIRE(json.IsArray() && json.Size() >= 1);
    EMPY_CHECK_EQ(json[0].GetString("path"), std::string("specs.observables[0].name"));
    EMPY_CHECK_EQ(json[0].GetString("severity"), std::string("error"));
    EMPY_CHECK(!json[0].GetString("message").empty());
}

EMPY_TEST(OpsDescriptor, UnknownKeysWarn)
{
    std::string text = std::string(LISTING_21) + "extra: 1\n";
    text.replace(text.find("maxMs: 5"), 8, "maxMs: 5\n            percentile: 95");
    text.replace(text.find("version: \"1.0.0\""), 16, "version: \"1.0.0\"\n    owner: someone");
    Json document = FromYaml(eCore::YamlNode::Parse(text));
    std::vector<std::string> unknown = UnknownKeys(document);
    EMPY_CHECK(unknown == std::vector<std::string>({"extra", "metadata.owner", "specs.inference.latency.percentile"}));
    RichMLService service = FromJson(document);
    std::vector<Issue> issues = ValidateDocument(document, service);
    EMPY_CHECK(!HasErrors(issues));
    EMPY_CHECK(HasIssue(issues, "extra", Severity::Warning));
    EMPY_CHECK(HasIssue(issues, "specs.inference.latency.percentile", Severity::Warning));
    EMPY_CHECK(UnknownKeys(ToJson(Populated())).empty());
}

EMPY_TEST(OpsDescriptor, EmitRoundTrip)
{
    RichMLService listing = ParseDescriptor(LISTING_21);
    std::string yaml = EmitDescriptor(listing);
    EMPY_CHECK(yaml.starts_with("apiVersion: mlops/v1\nkind: RichMLService\nmetadata:\n  name: home-control\n"
                                "  version: \"1.0.0\"\nspecs:\n  observables:\n    - name: home/temp\n"
                                "      unit: celcius\n"));
    EMPY_CHECK(yaml.find("regions: [ran, edge]") != std::string::npos);
    EMPY_CHECK(yaml.find("cpu: \"1\"") != std::string::npos);
    EMPY_CHECK(yaml.find("memory: \"4Gi\"") != std::string::npos);
    EMPY_CHECK(yaml.find("schedule: \"0 0 * * *\"") != std::string::npos);
    EMPY_CHECK(yaml.find("constraints") == std::string::npos); // defaults are not emitted
    // Listing 2.1 order: metadata, observables, actionables, inference, training, lifecycle
    EMPY_CHECK(yaml.find("observables:") < yaml.find("actionables:"));
    EMPY_CHECK(yaml.find("actionables:") < yaml.find("inference:"));
    EMPY_CHECK(yaml.find("inference:") < yaml.find("training:"));
    EMPY_CHECK(yaml.find("training:") < yaml.find("lifecycle:"));
    EMPY_CHECK(ToJson(ParseDescriptor(yaml)) == ToJson(listing));
    EMPY_CHECK_EQ(EmitDescriptor(ParseDescriptor(yaml)), yaml);

    RichMLService full = Populated();
    std::string text = EmitDescriptor(full);
    EMPY_CHECK_EQ(text, EmitDescriptor(full)); // deterministic
    RichMLService back = ParseDescriptor(text, "emitted");
    EMPY_CHECK(ToJson(back) == ToJson(full));
    EMPY_CHECK_EQ(back.m_Metadata.m_Description, full.m_Metadata.m_Description);
    EMPY_CHECK_EQ(back.m_Metadata.m_Intent, full.m_Metadata.m_Intent);
    EMPY_CHECK(back.m_Metadata.m_Labels == full.m_Metadata.m_Labels);
    EMPY_CHECK_EQ(back.m_Actionables[1].m_Name, std::string("home/cooling/#"));
    EMPY_CHECK_EQ(*back.m_Actionables[1].m_Max, 100.5);
    EMPY_CHECK_EQ(back.m_Constraints.m_Priority, -3);
    EMPY_CHECK_EQ(back.m_Inference->m_Resources.m_Cpu, 0.25);
    EMPY_CHECK_EQ(back.m_Inference->m_Resources.m_Memory, uint64_t{1536});
    EMPY_CHECK_EQ(back.m_Training->m_Resources.m_AcceleratorCount, 2u);
    EMPY_CHECK(!back.m_Constraints.m_Placement.m_AllowRelocation);

    // multi-line text uses a literal block that survives the round trip
    RichMLService block = listing;
    block.m_Metadata.m_Description = "line one\nline two: with colon\n\n# hash line\n";
    std::string blockYaml = EmitDescriptor(block);
    EMPY_CHECK(blockYaml.find("description: |\n    line one\n") != std::string::npos);
    EMPY_CHECK_EQ(ParseDescriptor(blockYaml).m_Metadata.m_Description, block.m_Metadata.m_Description);

    // ambiguous strings are quoted
    RichMLService tricky = listing;
    tricky.m_Metadata.m_Description = "true";
    tricky.m_Observables[0].m_Unit = "123";
    tricky.m_Training->m_Dataset = "null";
    RichMLService parsed = ParseDescriptor(EmitDescriptor(tricky));
    EMPY_CHECK_EQ(parsed.m_Metadata.m_Description, std::string("true"));
    EMPY_CHECK_EQ(parsed.m_Observables[0].m_Unit, std::string("123"));
    EMPY_CHECK_EQ(parsed.m_Training->m_Dataset, std::string("null"));
}

EMPY_TEST(OpsDescriptor, JsonCodec)
{
    RichMLService full = Populated();
    Json json = ToJson(full);
    EMPY_CHECK_EQ(json.GetString("apiVersion"), std::string("mlops/v1"));
    EMPY_REQUIRE(json.At("specs.inference.resources.cpu") != nullptr);
    EMPY_CHECK_EQ(json.At("specs.inference.resources.cpu")->AsString(), std::string("250m"));
    EMPY_CHECK_EQ(json.At("specs.training.resources.memory")->AsString(), std::string("4Gi"));
    EMPY_CHECK_EQ(json.At("specs.actionables.0.min")->AsNumber(), 16.0);
    EMPY_CHECK(json.At("specs.constraints.placement.allowRelocation")->IsBool());
    RichMLService back = FromJson(json);
    EMPY_CHECK(ToJson(back) == json);
    EMPY_CHECK(ToJson(FromJson(Json::Parse(json.Dump()))) == json);

    Json bad = json;
    bad.Find("specs")->Find("training")->Set("holdout", Json("most"));
    EMPY_CHECK(ErrorText([&] { FromJson(bad); }).starts_with("<json>: specs.training.holdout:"));
    EMPY_CHECK_THROWS(FromJson(Json(42)));
}

EMPY_TEST(OpsDescriptor, Profiles)
{
    RichMLService service = Populated();
    std::vector<eCore::MMQP::ProfileEntry> subscriptions = SubscriptionProfiles(service);
    EMPY_REQUIRE_EQ(subscriptions.size(), size_t{4}); // three observables + training label
    EMPY_CHECK_EQ(subscriptions[0].m_Topic, std::string("home/temp"));
    EMPY_CHECK_EQ(subscriptions[2].m_Topic, std::string("home/+/presence"));
    EMPY_CHECK_EQ(subscriptions[2].m_Qos, uint8_t{0});
    EMPY_CHECK_EQ(subscriptions[3].m_Topic, std::string("home/label/temp"));
    for (const auto &entry : subscriptions)
    {
        EMPY_CHECK_EQ(entry.m_Type, eCore::MMQP::MMQP_PROREQ_TYPE_SUB);
        EMPY_CHECK(!entry.m_Request);
    }
    EMPY_CHECK_EQ(subscriptions[0].m_Qos, uint8_t{1});

    std::vector<eCore::MMQP::ProfileEntry> publications = PublicationProfiles(service);
    EMPY_REQUIRE_EQ(publications.size(), size_t{1}); // the wildcard domain registers lazily
    EMPY_CHECK_EQ(publications[0].m_Type, eCore::MMQP::MMQP_PROREQ_TYPE_REG);
    EMPY_CHECK_EQ(publications[0].m_Topic, std::string("home/set/temp"));
    EMPY_CHECK_EQ(publications[0].m_Qos, uint8_t{0});

    // the label is not repeated when it is also an observable
    service.m_Training->m_Label = "home/temp";
    EMPY_CHECK_EQ(SubscriptionProfiles(service).size(), size_t{3});
    service.m_Training.reset();
    EMPY_CHECK_EQ(SubscriptionProfiles(service).size(), size_t{3});
}

EMPY_TEST(OpsDescriptor, CronFields)
{
    auto next = [](std::string_view expression, std::chrono::system_clock::time_point after)
    { return Iso(CronSchedule::Parse(expression).Next(after)); };
    auto t = At(2026, 10, 20, 13, 45, 30); // a Tuesday

    EMPY_CHECK_EQ(next("0 0 * * *", t), std::string("2026-10-21T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("*/15 * * * *", t), std::string("2026-10-20T14:00:00.000Z"));
    EMPY_CHECK_EQ(next("* * * * *", t), std::string("2026-10-20T13:46:00.000Z"));
    EMPY_CHECK_EQ(next("5,50 13 * * *", t), std::string("2026-10-20T13:50:00.000Z"));
    EMPY_CHECK_EQ(next("10-20/5 * * * *", t), std::string("2026-10-20T14:10:00.000Z"));
    EMPY_CHECK_EQ(next("30/10 * * * *", t), std::string("2026-10-20T13:50:00.000Z"));
    EMPY_CHECK_EQ(next("0 9 * * mon-fri", At(2026, 10, 24, 12, 0)), std::string("2026-10-26T09:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 0 * * 7", t), std::string("2026-10-25T00:00:00.000Z")); // 7 = Sunday
    EMPY_CHECK_EQ(next("0 0 * * SUN", t), std::string("2026-10-25T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 0 1 jan *", t), std::string("2027-01-01T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 0 29 2 *", t), std::string("2028-02-29T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 12 31 * *", t), std::string("2026-10-31T12:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 12 31 * *", At(2026, 11, 1, 0, 0)), std::string("2026-12-31T12:00:00.000Z"));
    // Vixie OR semantics: the 13th or any Friday
    EMPY_CHECK_EQ(next("0 0 13 * fri", t), std::string("2026-10-23T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("0 0 13 * fri", At(2026, 11, 7, 0, 0)), std::string("2026-11-13T00:00:00.000Z"));
    // a star-prefixed day field switches to AND semantics: odd days that are Mondays
    EMPY_CHECK_EQ(next("0 0 */2 * mon", t), std::string("2026-11-09T00:00:00.000Z"));
    // strictly after: a firing instant yields the next one
    EMPY_CHECK_EQ(next("0 0 * * *", At(2026, 10, 21, 0, 0)), std::string("2026-10-22T00:00:00.000Z"));

    EMPY_CHECK_EQ(next("@daily", t), std::string("2026-10-21T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("@midnight", t), std::string("2026-10-21T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("@hourly", t), std::string("2026-10-20T14:00:00.000Z"));
    EMPY_CHECK_EQ(next("@weekly", t), std::string("2026-10-25T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("@monthly", t), std::string("2026-11-01T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("@yearly", t), std::string("2027-01-01T00:00:00.000Z"));
    EMPY_CHECK_EQ(next("@annually", t), std::string("2027-01-01T00:00:00.000Z"));
    EMPY_CHECK_EQ(std::string(CronSchedule::Parse(" @Daily ").Expression()), std::string("@Daily"));

    for (const char *invalid :
         {"",          "* * * *",     "* * * * * *", "60 * * * *", "* 24 * * *",   "* * 0 * *",  "* * * 13 *",
          "* * * * 8", "*/0 * * * *", "5-1 * * * *", "a * * * *",  "1,,2 * * * *", "0 0 30 2 *", "0 0 31 4,6 *",
          "@often",    "@every",      "@every 0s",   "@every 5d",  "@every -5m",   "@everyday"})
    {
        EMPY_CHECK_THROWS(CronSchedule::Parse(invalid));
    }
}

EMPY_TEST(OpsDescriptor, CronEvery)
{
    CronSchedule every = CronSchedule::Parse("@every 30s");
    // periods are anchored at the Unix epoch, independent of when the schedule was parsed
    EMPY_CHECK_EQ(Iso(every.Next(At(2026, 10, 20, 13, 45, 10))), std::string("2026-10-20T13:45:30.000Z"));
    EMPY_CHECK_EQ(Iso(every.Next(At(2026, 10, 20, 13, 45, 30))), std::string("2026-10-20T13:46:00.000Z"));
    auto inside = At(2026, 10, 20, 13, 45, 29) + std::chrono::milliseconds(999);
    EMPY_CHECK_EQ(Iso(every.Next(inside)), std::string("2026-10-20T13:45:30.000Z"));
    EMPY_CHECK_EQ(Iso(CronSchedule::Parse("@every 5m").Next(At(2026, 10, 20, 13, 47))),
                  std::string("2026-10-20T13:50:00.000Z"));
    EMPY_CHECK_EQ(Iso(CronSchedule::Parse("@every 2h").Next(At(2026, 10, 20, 13, 0))),
                  std::string("2026-10-20T14:00:00.000Z"));
    EMPY_CHECK_EQ(std::string(CronSchedule::Parse("@every 30s").Expression()), std::string("@every 30s"));
    auto now = std::chrono::system_clock::now();
    auto due = every.Next(now);
    EMPY_CHECK(due > now && due - now <= std::chrono::seconds(30));
}
