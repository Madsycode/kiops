#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Substrate/Substrate.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <atomic>
#include <filesystem>
#include <thread>

using namespace eOps;
using namespace eOps::Substrate;

namespace
{
    using eTest::Ops::ConfigFile;

    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsSubstrate")
        {
        }
    };

    using Clock = std::chrono::steady_clock;

    Descriptor::RichMLService HomeControl()
    {
        return Descriptor::LoadDescriptor(ConfigFile("HomeControl.yaml"));
    }

    DeploymentSpec Spec(std::string role, std::string node, uint32_t replica = 0)
    {
        DeploymentSpec spec;
        spec.m_Descriptor = HomeControl();
        spec.m_Service = spec.m_Descriptor.m_Metadata.m_Name;
        spec.m_Role = std::move(role);
        spec.m_Node = std::move(node);
        spec.m_Region = "ran";
        spec.m_Replica = replica;
        spec.m_BrokerPort = 5611;
        return spec;
    }

    // split a multi-document YAML stream
    std::vector<eCore::YamlNode> Documents(const std::string &text)
    {
        std::vector<eCore::YamlNode> documents;
        size_t start = 0;
        while (true)
        {
            size_t end = text.find("\n---\n", start);
            documents.push_back(eCore::YamlNode::Parse(text.substr(start, end - start)));
            if (end == std::string::npos)
            {
                return documents;
            }
            start = end + 5;
        }
    }

    std::vector<std::string> Strings(const eCore::YamlNode &list)
    {
        std::vector<std::string> items;
        for (const eCore::YamlNode &item : list.Items())
        {
            items.push_back(item.As<std::string>());
        }
        return items;
    }

    // a harmless child command line: `script` for Windows batch, `shell` for /bin/sh
    std::vector<std::string> Command(std::vector<std::string> windows, std::vector<std::string> posix)
    {
#ifdef _WIN32
        (void)posix;
        return windows;
#else
        (void)windows;
        return posix;
#endif
    }

    // writes a stand-in workload: echoes its arguments (and EMPY_TEST_FLAG), then sleeps or exits
    std::filesystem::path Script(const std::filesystem::path &directory, const std::string &name, bool linger)
    {
#ifdef _WIN32
        std::filesystem::path path = directory / (name + ".cmd");
        std::string body = "@echo off\r\necho workload %* flag=%EMPY_TEST_FLAG%\r\n";
        body += linger ? "ping -n 60 127.0.0.1 >nul\r\n" : "exit /b 0\r\n";
#else
        std::filesystem::path path = directory / (name + ".sh");
        std::string body = "#!/bin/sh\necho workload \"$@\" flag=$EMPY_TEST_FLAG\n";
        body += linger ? "sleep 60\n" : "exit 0\n";
#endif
        WriteFileAtomic(path, body);
#ifndef _WIN32
        std::filesystem::permissions(path, std::filesystem::perms::owner_all, std::filesystem::perm_options::add);
#endif
        return path;
    }

    std::string StateOf(ExecutionSubstrate &substrate, const std::string &instance)
    {
        for (const InstanceStatus &status : substrate.List())
        {
            if (status.m_Instance == instance)
            {
                return status.m_State;
            }
        }
        return {};
    }

    std::string ReadText(const std::filesystem::path &path)
    {
        try
        {
            return ReadFile(path);
        }
        catch (const std::exception &)
        {
            return {};
        }
    }
} // namespace

EMPY_TEST(OpsSubstrate, InstanceNamesAndArgs)
{
    EMPY_CHECK_EQ(InstanceName("home-control", "inference", "ran-1", 0), std::string("home-control-inference-ran-1-0"));
    EMPY_CHECK_EQ(InstanceName("Home", "training", "Edge_1.local", 2), std::string("home-training-edge-1-local-2"));
    std::string longName = InstanceName(std::string(40, 'a'), "inference", std::string(40, 'b'), 1);
    std::string otherName = InstanceName(std::string(40, 'a'), "inference", std::string(40, 'b'), 2);
    EMPY_CHECK(IsDnsLabel(longName));
    EMPY_CHECK_EQ(longName.size(), size_t{63});
    EMPY_CHECK(longName != otherName); // the hash suffix keeps truncated names unique
    EMPY_CHECK(IsDnsLabel(InstanceName("--", "", "", 0)));

    DeploymentSpec spec = Spec("inference", "ran-1");
    spec.m_Instance = "home-control-inference-ran-1-0";
    spec.m_ExtraArgs = {"--verbose"};
    std::vector<std::string> args = WorkloadArgs(spec, "/etc/mlops/descriptor.yaml");
    EMPY_CHECK(args ==
               std::vector<std::string>({"inference", "--descriptor", "/etc/mlops/descriptor.yaml", "--instance",
                                         "home-control-inference-ran-1-0", "--node", "ran-1", "--region", "ran",
                                         "--host", "127.0.0.1", "--port", "5611", "--root", "mlops", "--verbose"}));
    std::filesystem::path workload = DefaultWorkloadExecutable();
    EMPY_CHECK(workload.is_absolute());
    EMPY_CHECK(workload.stem() == "EmpyWorkload");
}

EMPY_TEST(OpsSubstrate, Simulated)
{
    SimulatedSubstrate substrate;
    EMPY_CHECK_EQ(substrate.Name(), std::string("simulated"));
    EnactResult first = substrate.Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(first.m_Ok);
    EMPY_CHECK_EQ(first.m_Instance, std::string("home-control-inference-ran-1-0"));
    EMPY_CHECK(substrate.Deploy(Spec("training", "edge-1")).m_Ok);
    EnactResult duplicate = substrate.Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(!duplicate.m_Ok);
    EMPY_CHECK(duplicate.m_Detail.find("already deployed") != std::string::npos);

    std::vector<InstanceStatus> listed = substrate.List();
    EMPY_REQUIRE_EQ(listed.size(), size_t{2});
    EMPY_CHECK_EQ(listed[0].m_Instance, std::string("home-control-inference-ran-1-0"));
    EMPY_CHECK_EQ(listed[0].m_State, std::string("running"));
    EMPY_CHECK_EQ(listed[1].m_Role, std::string("training"));
    std::vector<DeploymentSpec> deployed = substrate.Deployed();
    EMPY_REQUIRE_EQ(deployed.size(), size_t{2});
    EMPY_CHECK_EQ(deployed[0].m_Version, std::string("1.0.0")); // filled from the descriptor
    EMPY_CHECK_EQ(deployed[1].m_Node, std::string("edge-1"));

    Json described = substrate.Describe();
    EMPY_CHECK_EQ(described.GetString("type"), std::string("simulated"));
    EMPY_CHECK_EQ(described["instances"].Size(), size_t{2});
    EMPY_CHECK_EQ(std::string(described.At("instances.1.node")->AsString()), std::string("edge-1"));

    // invalid specs fail without side effects
    DeploymentSpec bad = Spec("inference", "ran-1");
    bad.m_Service = "Not A Label";
    EMPY_CHECK(!substrate.Deploy(bad).m_Ok);
    EMPY_CHECK(!substrate.Deploy(Spec("inference", "")).m_Ok);
    DeploymentSpec named = Spec("inference", "ran-2");
    named.m_Instance = "custom_name";
    EMPY_CHECK(!substrate.Deploy(named).m_Ok);
    named.m_Instance = "custom-name";
    EMPY_CHECK_EQ(substrate.Deploy(named).m_Instance, std::string("custom-name"));

    // the failure hook runs unlocked and may inspect the substrate
    std::atomic<int32_t> calls{0};
    substrate.SetFailure(
        [&](const DeploymentSpec &spec) -> std::optional<std::string>
        {
            ++calls;
            if (spec.m_Node == "ran-9")
            {
                return "accelerator failure on " + spec.m_Node + " (" + std::to_string(substrate.List().size()) +
                       " running)";
            }
            return std::nullopt;
        });
    EnactResult failed = substrate.Deploy(Spec("inference", "ran-9"));
    EMPY_CHECK(!failed.m_Ok);
    EMPY_CHECK_EQ(failed.m_Detail, std::string("accelerator failure on ran-9 (3 running)"));
    EMPY_CHECK(substrate.Deploy(Spec("inference", "ran-3")).m_Ok);
    EMPY_CHECK_EQ(calls.load(), 2);
    substrate.SetFailure(nullptr);

    EMPY_CHECK(substrate.Retire("home-control-inference-ran-1-0").m_Ok);
    EnactResult unknown = substrate.Retire("home-control-inference-ran-1-0");
    EMPY_CHECK(!unknown.m_Ok);
    EMPY_CHECK(unknown.m_Detail.find("unknown instance") != std::string::npos);
    EMPY_CHECK_EQ(substrate.List().size(), size_t{3});
    EMPY_CHECK(substrate.Deploy(Spec("inference", "ran-1")).m_Ok); // retired names can be reused
}

EMPY_TEST(OpsSubstrate, ManifestContent)
{
    DeploymentSpec inference = Spec("inference", "ran-1");
    inference.m_BrokerHost = "broker.empy-mlops.svc";
    std::string text = RenderManifests(inference, "empy-mlops", "empy/workload:latest");
    EMPY_CHECK_EQ(text, RenderManifests(inference, "empy-mlops", "empy/workload:latest")); // deterministic
    EMPY_CHECK(text.starts_with("# empy-mlops: {\"instance\":\"home-control-inference-ran-1-0\""));
    std::vector<eCore::YamlNode> documents = Documents(text);
    EMPY_REQUIRE_EQ(documents.size(), size_t{3});

    const eCore::YamlNode &config = documents[0];
    EMPY_CHECK_EQ(config["kind"].As<std::string>(), std::string("ConfigMap"));
    EMPY_CHECK_EQ(config["metadata"]["name"].As<std::string>(),
                  std::string("home-control-inference-ran-1-0-descriptor"));
    EMPY_CHECK_EQ(config["metadata"]["namespace"].As<std::string>(), std::string("empy-mlops"));
    // the embedded descriptor is the deployed descriptor
    Descriptor::RichMLService embedded =
        Descriptor::ParseDescriptor(config["data"]["descriptor.yaml"].As<std::string>(), "configmap");
    EMPY_CHECK(Descriptor::ToJson(embedded) == Descriptor::ToJson(inference.m_Descriptor));

    const eCore::YamlNode &deployment = documents[1];
    EMPY_CHECK_EQ(deployment["apiVersion"].As<std::string>(), std::string("apps/v1"));
    EMPY_CHECK_EQ(deployment["kind"].As<std::string>(), std::string("Deployment"));
    eCore::YamlNode labels = deployment["metadata"]["labels"];
    EMPY_CHECK_EQ(labels["app.kubernetes.io/name"].As<std::string>(), std::string("home-control"));
    EMPY_CHECK_EQ(labels["app.kubernetes.io/component"].As<std::string>(), std::string("inference"));
    EMPY_CHECK_EQ(labels["mlops.empy.io/service"].As<std::string>(), std::string("home-control"));
    EMPY_CHECK_EQ(labels["mlops.empy.io/version"].As<std::string>(), std::string("1.0.0"));
    eCore::YamlNode pod = deployment["spec"]["template"]["spec"];
    EMPY_CHECK_EQ(pod["nodeSelector"]["kubernetes.io/hostname"].As<std::string>(), std::string("ran-1"));
    EMPY_CHECK_EQ(deployment["spec"]["selector"]["matchLabels"]["app.kubernetes.io/instance"].As<std::string>(),
                  std::string("home-control-inference-ran-1-0"));
    std::vector<eCore::YamlNode> containers = pod["containers"].Items();
    EMPY_REQUIRE_EQ(containers.size(), size_t{1});
    eCore::YamlNode container = containers[0];
    EMPY_CHECK_EQ(container["image"].As<std::string>(), std::string("empy/workload:latest"));
    EMPY_CHECK(Strings(container["args"]) ==
               std::vector<std::string>({"inference", "--descriptor", "/etc/mlops/descriptor.yaml", "--instance",
                                         "home-control-inference-ran-1-0", "--node", "ran-1", "--region", "ran",
                                         "--host", "broker.empy-mlops.svc", "--port", "5611", "--root", "mlops"}));
    EMPY_CHECK_EQ(container["resources"]["requests"]["cpu"].As<std::string>(), std::string("1"));
    EMPY_CHECK_EQ(container["resources"]["requests"]["memory"].As<std::string>(), std::string("1Gi"));
    EMPY_CHECK_EQ(container["resources"]["limits"]["cpu"].As<std::string>(), std::string("1"));
    EMPY_CHECK_EQ(container["resources"]["limits"]["memory"].As<std::string>(), std::string("1Gi"));
    EMPY_CHECK(!container["resources"]["limits"].Has("nvidia.com/gpu"));
    EMPY_CHECK_EQ(container["volumeMounts"].Items()[0]["mountPath"].As<std::string>(), std::string("/etc/mlops"));
    EMPY_CHECK_EQ(pod["volumes"].Items()[0]["configMap"]["name"].As<std::string>(),
                  std::string("home-control-inference-ran-1-0-descriptor"));

    const eCore::YamlNode &resource = documents[2];
    EMPY_CHECK_EQ(resource["apiVersion"].As<std::string>(), std::string("mlops.empy.io/v1"));
    EMPY_CHECK_EQ(resource["kind"].As<std::string>(), std::string("RichMLService"));
    EMPY_CHECK_EQ(resource["metadata"]["name"].As<std::string>(), std::string("home-control-inference-ran-1-0"));
    EMPY_CHECK_EQ(resource["metadata"]["namespace"].As<std::string>(), std::string("empy-mlops"));
    EMPY_CHECK_EQ(resource["spec"]["observables"].Items()[0]["name"].As<std::string>(), std::string("home/temp"));
    EMPY_CHECK_EQ(resource["spec"]["inference"]["latency"]["statistic"].As<std::string>(), std::string("p95"));
    EMPY_CHECK_EQ(resource["spec"]["training"]["schedule"].As<std::string>(), std::string("0 0 * * *"));

    // training: accelerator limits and the descriptor's own image
    DeploymentSpec training = Spec("training", "edge-1");
    training.m_Descriptor.m_Training->m_Image = "registry.local/trainer:2";
    eCore::YamlNode trainer = Documents(RenderManifests(training, "lab", "unused"))[1];
    eCore::YamlNode trainContainer = trainer["spec"]["template"]["spec"]["containers"].Items()[0];
    EMPY_CHECK_EQ(trainContainer["name"].As<std::string>(), std::string("training"));
    EMPY_CHECK_EQ(trainContainer["image"].As<std::string>(), std::string("registry.local/trainer:2"));
    EMPY_CHECK_EQ(trainContainer["resources"]["limits"]["nvidia.com/gpu"].As<int32_t>(), 1);
    EMPY_CHECK_EQ(trainContainer["resources"]["requests"]["cpu"].As<std::string>(), std::string("2"));
    EMPY_CHECK_EQ(trainContainer["resources"]["limits"]["memory"].As<std::string>(), std::string("4Gi"));
    EMPY_CHECK(!trainContainer["resources"]["requests"].Has("nvidia.com/gpu"));
    EMPY_CHECK_EQ(trainer["metadata"]["namespace"].As<std::string>(), std::string("lab"));

    training.m_Descriptor.m_Training->m_Resources.m_Accelerator = "npu";
    training.m_Descriptor.m_Training->m_Resources.m_AcceleratorCount = 2;
    training.m_Descriptor.m_Training->m_Resources.m_Cpu = 0.5;
    eCore::YamlNode npu = Documents(RenderManifests(training, "lab", "unused"))[1];
    eCore::YamlNode limits = npu["spec"]["template"]["spec"]["containers"].Items()[0]["resources"]["limits"];
    EMPY_CHECK_EQ(limits["empy.io/npu"].As<int32_t>(), 2);
    EMPY_CHECK_EQ(limits["cpu"].As<std::string>(), std::string("500m"));
    training.m_Descriptor.m_Training->m_Resources.m_Accelerator = "tpu";
    EMPY_CHECK(RenderManifests(training, "lab", "unused").find("google.com/tpu: 2") != std::string::npos);

    // specs that cannot be rendered
    EMPY_CHECK_THROWS(RenderManifests(Spec("simulate", "ran-1"), "empy-mlops", "image"));
    EMPY_CHECK_THROWS(RenderManifests(inference, "Bad_Namespace", "image"));
    EMPY_CHECK_THROWS(RenderManifests(Spec("inference", "ran-1"), "empy-mlops", ""));
    DeploymentSpec noTraining = Spec("training", "edge-1");
    noTraining.m_Descriptor.m_Training.reset();
    EMPY_CHECK_THROWS(RenderManifests(noTraining, "empy-mlops", "image"));
    DeploymentSpec badNode = Spec("inference", "node with spaces");
    EMPY_CHECK_THROWS(RenderManifests(badNode, "empy-mlops", "image"));
}

EMPY_TEST(OpsSubstrate, KubernetesCommands)
{
    TempDir temp;
    EMPY_CHECK_THROWS(KubernetesSubstrate(KubernetesOptions{}));
    KubernetesOptions badNamespace;
    badNamespace.m_OutputDirectory = temp.m_Path;
    badNamespace.m_Namespace = "Not Valid";
    EMPY_CHECK_THROWS(KubernetesSubstrate(badNamespace));

    KubernetesOptions options;
    options.m_OutputDirectory = temp.m_Path / "manifests";
    options.m_ApplyCommand = Command({"cmd", "/c", "echo", "applied"}, {"/bin/sh", "-c", "echo applied \"$0\""});
    options.m_DeleteCommand = Command({"cmd", "/c", "exit", "0"}, {"/bin/sh", "-c", "exit 0"});
    KubernetesSubstrate substrate(options);
    EMPY_CHECK_EQ(substrate.Name(), std::string("kubernetes"));
    EMPY_CHECK(substrate.List().empty());

    EnactResult deployed = substrate.Deploy(Spec("inference", "ran-1"));
    EMPY_REQUIRE(deployed.m_Ok);
    std::filesystem::path file = options.m_OutputDirectory / "home-control-inference-ran-1-0.yaml";
    EMPY_CHECK(std::filesystem::exists(file));
    EMPY_REQUIRE_EQ(deployed.m_Artifacts.size(), size_t{1});
    EMPY_CHECK(std::filesystem::equivalent(deployed.m_Artifacts[0], file));
    // the command saw the manifest path appended and its output is reported
    EMPY_CHECK(deployed.m_Detail.find("exited with code 0") != std::string::npos);
    EMPY_CHECK(deployed.m_Detail.find("applied") != std::string::npos);
    EMPY_CHECK(deployed.m_Detail.find("home-control-inference-ran-1-0.yaml") != std::string::npos);
    EMPY_CHECK_EQ(ReadText(file), RenderManifests(Spec("inference", "ran-1"), "empy-mlops", "empy/workload:latest"));

    EnactResult duplicate = substrate.Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(!duplicate.m_Ok);
    EMPY_CHECK(substrate.Deploy(Spec("training", "edge-1")).m_Ok);
    std::vector<InstanceStatus> listed = substrate.List();
    EMPY_REQUIRE_EQ(listed.size(), size_t{2});
    EMPY_CHECK_EQ(listed[0].m_Instance, std::string("home-control-inference-ran-1-0"));
    EMPY_CHECK_EQ(listed[0].m_Service, std::string("home-control"));
    EMPY_CHECK_EQ(listed[0].m_Node, std::string("ran-1"));
    EMPY_CHECK_EQ(listed[0].m_State, std::string("applied"));
    EMPY_CHECK_EQ(listed[1].m_Role, std::string("training"));

    // a fresh substrate over the same directory recovers the rendered instances
    KubernetesSubstrate recovered(options);
    EMPY_CHECK_EQ(recovered.List().size(), size_t{2});

    EnactResult retired = substrate.Retire("home-control-inference-ran-1-0");
    EMPY_CHECK(retired.m_Ok);
    EMPY_CHECK(!std::filesystem::exists(file));
    EMPY_CHECK_EQ(substrate.List().size(), size_t{1});
    EMPY_CHECK(!substrate.Retire("home-control-inference-ran-1-0").m_Ok);
    EMPY_CHECK(!substrate.Retire("../escape").m_Ok);

    // failing commands: apply leaves nothing behind, delete keeps the manifest for a retry
    KubernetesOptions failing = options;
    failing.m_ApplyCommand =
        Command({"cmd", "/c", "echo", "denied", "&", "exit", "3"}, {"/bin/sh", "-c", "echo denied >&2; exit 3"});
    failing.m_DeleteCommand = Command({"cmd", "/c", "exit", "4"}, {"/bin/sh", "-c", "exit 4"});
    KubernetesSubstrate broken(failing);
    EnactResult rejected = broken.Deploy(Spec("inference", "ran-2"));
    EMPY_CHECK(!rejected.m_Ok);
    EMPY_CHECK(rejected.m_Detail.find("exited with code 3") != std::string::npos);
    EMPY_CHECK(rejected.m_Detail.find("denied") != std::string::npos);
    EMPY_CHECK(!std::filesystem::exists(options.m_OutputDirectory / "home-control-inference-ran-2-0.yaml"));
    EnactResult stuck = broken.Retire("home-control-training-edge-1-0");
    EMPY_CHECK(!stuck.m_Ok);
    EMPY_CHECK(stuck.m_Detail.find("exited with code 4") != std::string::npos);
    EMPY_CHECK(std::filesystem::exists(options.m_OutputDirectory / "home-control-training-edge-1-0.yaml"));
    EMPY_CHECK(substrate.Retire("home-control-training-edge-1-0").m_Ok);

    KubernetesOptions missing = options;
    missing.m_ApplyCommand = {(temp.m_Path / "no-such-kubectl.exe").string()};
    EnactResult unavailable = KubernetesSubstrate(missing).Deploy(Spec("inference", "ran-4"));
    EMPY_CHECK(!unavailable.m_Ok);
    EMPY_CHECK(unavailable.m_Detail.find("cannot run") != std::string::npos);

    KubernetesOptions slow = options;
    slow.m_ApplyCommand =
        Command({"cmd", "/c", "ping", "-n", "30", "127.0.0.1", ">nul", "&", "rem"}, {"/bin/sh", "-c", "sleep 30"});
    slow.m_CommandTimeoutMs = 300;
    Clock::time_point start = Clock::now();
    EnactResult timedOut = KubernetesSubstrate(slow).Deploy(Spec("inference", "ran-5"));
    EMPY_CHECK(!timedOut.m_Ok);
    EMPY_CHECK(timedOut.m_Detail.find("timed out") != std::string::npos);
    EMPY_CHECK(Clock::now() - start < std::chrono::seconds(10));

    // the apply command runs outside the substrate lock: List (the harness status on its owner thread) answers at
    // once and reports the instance as applying while a worker waits for the slow command
    slow.m_CommandTimeoutMs = 1500;
    KubernetesSubstrate busy(slow);
    std::atomic<bool> finished{false};
    std::thread worker(
        [&]
        {
            busy.Deploy(Spec("inference", "ran-6"));
            finished = true;
        });
    bool applying = false;
    Clock::time_point wait = Clock::now();
    while (!applying && !finished && Clock::now() - wait < std::chrono::seconds(5))
    {
        Clock::time_point listed = Clock::now();
        std::vector<InstanceStatus> statuses = busy.List();
        EMPY_CHECK(Clock::now() - listed < std::chrono::milliseconds(500));
        applying = statuses.size() == 1 && statuses[0].m_State == "applying";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EMPY_CHECK(applying);
    EMPY_CHECK(!busy.Retire("home-control-inference-ran-6-0").m_Ok); // a command is in progress
    worker.join();
    EMPY_CHECK(busy.List().empty()); // the timed-out apply left nothing behind

    // without commands the manifests are only rendered
    KubernetesOptions renderOnly;
    renderOnly.m_OutputDirectory = temp.m_Path / "render";
    KubernetesSubstrate renderer(renderOnly);
    EMPY_CHECK(renderer.Deploy(Spec("inference", "ran-1")).m_Ok);
    EMPY_CHECK_EQ(renderer.List()[0].m_State, std::string("rendered"));
    EMPY_CHECK_EQ(renderer.Describe()["instances"].Size(), size_t{1});
    EMPY_CHECK(renderer.Retire("home-control-inference-ran-1-0").m_Ok);
    EMPY_CHECK(renderer.List().empty());
    // only marker-carrying manifests are listed
    WriteFileAtomic(renderOnly.m_OutputDirectory / "foreign.yaml", "kind: ConfigMap\n");
    EMPY_CHECK(renderer.List().empty());
}

EMPY_TEST(OpsSubstrate, ProcessLaunchAndKill)
{
    TempDir temp;
    ProcessSubstrateOptions options;
    options.m_Executable = Script(temp.m_Path, "workload", true);
    options.m_StateDirectory = temp.m_Path / "state";
    options.m_LogDirectory = temp.m_Path / "logs";
    options.m_ClientConfig = temp.m_Path / "client.yaml";
    options.m_Environment = {{"EMPY_TEST_FLAG", "on"}};
    auto substrate = std::make_unique<ProcessSubstrate>(options);
    EMPY_CHECK_EQ(substrate->Name(), std::string("process"));

    EnactResult started = substrate->Deploy(Spec("inference", "ran-1"));
    EMPY_REQUIRE(started.m_Ok);
    const std::string instance = "home-control-inference-ran-1-0";
    EMPY_CHECK_EQ(started.m_Instance, instance);
    EMPY_CHECK(started.m_Detail.starts_with("pid "));
    std::filesystem::path descriptor = options.m_StateDirectory / (instance + ".descriptor.yaml");
    std::filesystem::path log = options.m_LogDirectory / (instance + ".log");
    EMPY_REQUIRE_EQ(started.m_Artifacts.size(), size_t{2});
    EMPY_CHECK(std::filesystem::exists(descriptor));
    EMPY_CHECK_EQ(Descriptor::LoadDescriptor(descriptor).m_Metadata.m_Name, std::string("home-control"));
    EMPY_CHECK_EQ(StateOf(*substrate, instance), std::string("running"));

    // the child received the workload arguments, the client config and the extra environment
    EMPY_REQUIRE(eTest::WaitUntil([&] { return ReadText(log).find("flag=on") != std::string::npos; },
                                  std::chrono::milliseconds(10000)));
    std::string output = ReadText(log);
    EMPY_CHECK(output.find("workload inference --descriptor") != std::string::npos);
    EMPY_CHECK(output.find("--instance " + instance) != std::string::npos);
    EMPY_CHECK(output.find("--port 5611") != std::string::npos);
    EMPY_CHECK(output.find("--config") != std::string::npos);
    EMPY_CHECK(output.find("client.yaml") != std::string::npos);

    EnactResult again = substrate->Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(!again.m_Ok);
    EMPY_CHECK(again.m_Detail.find("already running") != std::string::npos);
    EMPY_CHECK(substrate->Deploy(Spec("training", "edge-1")).m_Ok);
    EMPY_CHECK_EQ(substrate->List().size(), size_t{2});

    Clock::time_point start = Clock::now();
    EnactResult killed = substrate->Retire(instance);
    EMPY_CHECK(killed.m_Ok);
    EMPY_CHECK(killed.m_Detail.starts_with("killed"));
    EMPY_CHECK(Clock::now() - start < std::chrono::seconds(10));
    EMPY_CHECK(!std::filesystem::exists(descriptor));
    EMPY_CHECK(std::filesystem::exists(log)); // logs are kept for diagnosis
    EMPY_CHECK(!substrate->Retire(instance).m_Ok);
    EMPY_CHECK_EQ(substrate->List().size(), size_t{1});

    // destruction takes down remaining children promptly
    start = Clock::now();
    substrate.reset();
    EMPY_CHECK(Clock::now() - start < std::chrono::seconds(10));
}

EMPY_TEST(OpsSubstrate, ProcessShortRuns)
{
    TempDir temp;
    ProcessSubstrateOptions options;
    options.m_Executable = Script(temp.m_Path, "quick", false);
    options.m_StateDirectory = temp.m_Path / "state";
    ProcessSubstrate substrate(options);

    const std::string instance = "home-control-training-edge-1-0";
    EMPY_REQUIRE(substrate.Deploy(Spec("training", "edge-1")).m_Ok);
    EMPY_CHECK(
        eTest::WaitUntil([&] { return StateOf(substrate, instance) == "exited"; }, std::chrono::milliseconds(15000)));
    // the log defaults to the state directory; no --config without a client config
    std::string output = ReadText(options.m_StateDirectory / (instance + ".log"));
    EMPY_CHECK(output.find("workload training") != std::string::npos);
    EMPY_CHECK(output.find("--config") == std::string::npos);

    // an exited instance may be deployed again, and retiring it reports the exit
    EMPY_REQUIRE(substrate.Deploy(Spec("training", "edge-1")).m_Ok);
    EMPY_CHECK(
        eTest::WaitUntil([&] { return StateOf(substrate, instance) == "exited"; }, std::chrono::milliseconds(15000)));
    EnactResult retired = substrate.Retire(instance);
    EMPY_CHECK(retired.m_Ok);
    EMPY_CHECK_EQ(retired.m_Detail, std::string("already exited (exit code 0)"));

    // a missing executable fails the deployment and leaves no descriptor behind
    ProcessSubstrateOptions missing = options;
    missing.m_Executable = temp.m_Path / "NoSuchWorkload.exe";
    ProcessSubstrate unavailable(missing);
    EnactResult failed = unavailable.Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(!failed.m_Ok);
    EMPY_CHECK(failed.m_Detail.find("cannot launch") != std::string::npos);
    EMPY_CHECK(!std::filesystem::exists(options.m_StateDirectory / "home-control-inference-ran-1-0.descriptor.yaml"));
    EMPY_CHECK(unavailable.List().empty());
    EMPY_CHECK(!unavailable.Deploy(Spec("Inference!", "ran-1")).m_Ok);
}

EMPY_TEST(OpsSubstrate, MakeFromYaml)
{
    TempDir temp;
    std::string config = (temp.m_Path / "Harness.yaml").string();
    EMPY_CHECK_EQ(MakeSubstrate(eCore::YamlNode(), config)->Name(), std::string("simulated"));
    auto parse = [&](const char *text)
    { return MakeSubstrate(eCore::YamlNode::Parse(text, "substrate")["substrate"], config); };
    EMPY_CHECK_EQ(parse("substrate: {type: simulated}")->Name(), std::string("simulated"));
    EMPY_CHECK_EQ(parse("substrate: {}")->Name(), std::string("simulated"));

    std::shared_ptr<ExecutionSubstrate> kubernetes =
        parse("substrate:\n  type: kubernetes\n  output_directory: out/manifests\n  namespace: lab\n"
              "  image: empy/workload:1\n  apply_command: [cmd, /c, rem]\n  command_timeout_ms: 5000\n");
    EMPY_REQUIRE_EQ(kubernetes->Name(), std::string("kubernetes"));
    EnactResult rendered = kubernetes->Deploy(Spec("inference", "ran-1"));
#ifdef _WIN32
    EMPY_CHECK(rendered.m_Ok);
#endif
    // relative paths resolve against the configuration file
    std::filesystem::path manifest = temp.m_Path / "out" / "manifests" / "home-control-inference-ran-1-0.yaml";
#ifdef _WIN32
    EMPY_CHECK(std::filesystem::exists(manifest));
    EMPY_CHECK(ReadText(manifest).find("namespace: lab") != std::string::npos);
    EMPY_CHECK(ReadText(manifest).find("image: \"empy/workload:1\"") != std::string::npos); // ':' is quoted
#endif

    std::shared_ptr<ExecutionSubstrate> process =
        parse("substrate:\n  type: process\n  executable: bin/EmpyWorkload\n  state_directory: state\n"
              "  environment: {EMPY_LOG: debug}\n");
    EMPY_CHECK_EQ(process->Name(), std::string("process"));
    EnactResult launch = process->Deploy(Spec("inference", "ran-1"));
    EMPY_CHECK(!launch.m_Ok);
    EMPY_CHECK(launch.m_Detail.find((std::filesystem::path("bin") / "EmpyWorkload").string()) != std::string::npos);

    EMPY_CHECK_THROWS(parse("substrate: {type: docker}"));
    EMPY_CHECK_THROWS(parse("substrate: {type: kubernetes}"));
    EMPY_CHECK_THROWS(parse("substrate: {type: kubernetes, output_directory: out, namespace: Bad_NS}"));
    EMPY_CHECK_THROWS(parse("substrate: {type: kubernetes, output_directory: out, apply_command: kubectl}"));
    EMPY_CHECK_THROWS(parse("substrate: {type: kubernetes, output_directory: out, command_timeout_ms: 0}"));
    EMPY_CHECK_THROWS(parse("substrate: {type: simulated, outputdirectory: typo}"));
    EMPY_CHECK_THROWS(parse("substrate: [simulated]"));
    try
    {
        parse("substrate: {type: docker}");
    }
    catch (const eCore::ConfigError &error)
    {
        EMPY_CHECK(std::string(error.what()).find("substrate.type") != std::string::npos);
    }
}
