#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <optional>
#include <functional>
#include <filesystem>
#include "../Common/Json.h"
#include "../Descriptor/Model.h"

// execution substrate (thesis Fig. 23 bottom): enacts validated deployments. Simulated
// (in-memory, deterministic), Kubernetes (rendered manifests: ConfigMap + Deployment +
// RichMLService custom resource, optionally applied with a command) and Process (local
// EmpyWorkload children). Substrates are internally synchronised.

namespace eOps::Substrate
{
    struct DeploymentSpec
    {
        std::string m_Service;
        std::string m_Version;
        std::string m_Role; // inference | training
        Descriptor::RichMLService m_Descriptor;
        std::string m_Node;
        std::string m_Region;
        uint32_t m_Replica = 0;
        std::string m_Instance; // <service>-<role>-<node>-<replica>
        std::string m_BrokerHost = "127.0.0.1";
        uint16_t m_BrokerPort = 5600;
        std::string m_TopicRoot = "mlops";
        std::vector<std::string> m_ExtraArgs; // appended to the workload arguments
    };

    // "<service>-<role>-<node>-<replica>" sanitised to a DNS label (≤ 63, lowercase, '-' separators)
    EMPY_OPS_API std::string InstanceName(const std::string &service, const std::string &role, const std::string &node,
                                          uint32_t replica);

    struct EMPY_OPS_API EnactResult
    {
        bool m_Ok = false;
        std::string m_Instance;
        std::string m_Detail;
        std::vector<std::string> m_Artifacts; // files written / resources applied
        Json ToJson() const;                  // {ok, instance, detail, artifacts[]}
    };

    struct EMPY_OPS_API InstanceStatus
    {
        std::string m_Instance;
        std::string m_Service;
        std::string m_Role;
        std::string m_Node;
        std::string m_State; // running | exited | applied | ...
        Json ToJson() const; // {instance, service, role, node, state}
    };

    struct EMPY_OPS_API ExecutionSubstrate
    {
        virtual ~ExecutionSubstrate();
        virtual std::string Name() const = 0;
        virtual EnactResult Deploy(const DeploymentSpec &spec) = 0;
        virtual EnactResult Retire(const std::string &instance) = 0;
        virtual std::vector<InstanceStatus> List() = 0;
        virtual Json Describe(); // {type: Name(), instances: [...]}
    };

    // workload argv for a spec: [<role>, --descriptor, <descriptorPath>, --instance, <instance>, --node, <node>,
    // --region, <region>, --host, <broker host>, --port, <port>, --root, <root>] + m_ExtraArgs
    EMPY_OPS_API std::vector<std::string> WorkloadArgs(const DeploymentSpec &spec, const std::string &descriptorPath);

    class EMPY_OPS_API SimulatedSubstrate : public ExecutionSubstrate
    {
    public:
        using Failure = std::function<std::optional<std::string>(const DeploymentSpec &)>;

        SimulatedSubstrate();
        ~SimulatedSubstrate() override;

        std::string Name() const override;                       // "simulated"
        EnactResult Deploy(const DeploymentSpec &spec) override; // records the instance as "running"
        EnactResult Retire(const std::string &instance) override;
        std::vector<InstanceStatus> List() override;

        // test hook: a non-empty result fails that Deploy with the returned detail
        void SetFailure(Failure failure);
        std::vector<DeploymentSpec> Deployed() const; // specs of current instances

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    struct KubernetesOptions
    {
        std::filesystem::path m_OutputDirectory; // <out>/<instance>.yaml
        std::string m_Namespace = "empy-mlops";
        std::string m_Image = "empy/workload:latest";
        std::vector<std::string> m_ApplyCommand;  // e.g. [kubectl, apply, -f]; file path appended
        std::vector<std::string> m_DeleteCommand; // e.g. [kubectl, delete, -f]; file path appended
        uint32_t m_CommandTimeoutMs = 60000;
    };

    class EMPY_OPS_API KubernetesSubstrate : public ExecutionSubstrate
    {
    public:
        explicit KubernetesSubstrate(KubernetesOptions options);
        ~KubernetesSubstrate() override;

        std::string Name() const override; // "kubernetes"
        // writes the manifests atomically; runs the apply command (non-zero exit → failure with its output)
        EnactResult Deploy(const DeploymentSpec &spec) override;
        // runs the delete command then removes the file
        EnactResult Retire(const std::string &instance) override;
        std::vector<InstanceStatus> List() override; // instances with a manifest file ("applied")

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    struct ProcessSubstrateOptions
    {
        std::filesystem::path m_Executable;     // EmpyWorkload; "" → EmpyWorkload(.exe) beside the running executable
        std::filesystem::path m_StateDirectory; // <state>/<instance>.descriptor.yaml
        std::filesystem::path m_ClientConfig;   // "--config <client yaml>" when set
        std::filesystem::path m_LogDirectory;   // <log>/<instance>.log (stdout + stderr); "" → state directory
        std::vector<std::pair<std::string, std::string>> m_Environment; // extra child environment
    };

    class EMPY_OPS_API ProcessSubstrate : public ExecutionSubstrate
    {
    public:
        explicit ProcessSubstrate(ProcessSubstrateOptions options);
        ~ProcessSubstrate() override; // kills every child

        std::string Name() const override; // "process"
        EnactResult Deploy(const DeploymentSpec &spec) override;
        EnactResult Retire(const std::string &instance) override; // kills the child
        std::vector<InstanceStatus> List() override;              // running | exited

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    // multi-document YAML (ConfigMap with descriptor.yaml, Deployment, RichMLService custom resource), deterministic
    EMPY_OPS_API std::string RenderManifests(const DeploymentSpec &spec, const std::string &nameSpace,
                                             const std::string &image);

    // `substrate: {type: simulated | kubernetes | process, output_directory, namespace, image,
    // apply_command: [...], delete_command: [...], executable, state_directory, log_directory, client_config}`;
    // paths relative to `configPath` (eCore::ConfigPath); a null node → simulated; throws eCore::ConfigError
    EMPY_OPS_API std::shared_ptr<ExecutionSubstrate> MakeSubstrate(const eCore::YamlNode &substrate,
                                                                   const std::string &configPath);

    // EmpyWorkload(.exe) beside the running executable
    EMPY_OPS_API std::filesystem::path DefaultWorkloadExecutable();
} // namespace eOps::Substrate
