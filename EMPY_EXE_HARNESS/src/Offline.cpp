#include "Cli.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// EmpyHarness offline commands: they load the configured ontology, knowledge and policies but never connect to a
// broker or touch an execution substrate (validate, interpret, render, query, mcp-server)

namespace eHarness
{
    namespace
    {
        eOps::Descriptor::ValidationOptions Validation(const eOps::Harness::HarnessConfig &config,
                                                       const eOps::Harness::HarnessComponents &components)
        {
            eOps::Descriptor::ValidationOptions options;
            options.m_Regions = components.m_Regions;
            options.m_TopicRoot = config.m_TopicRoot;
            return options;
        }

        // nodes for `role`: --node, else the Placement agent over the configured knowledge, else "local"
        std::vector<std::string> Nodes(const Options &options, const eOps::Harness::HarnessComponents &components,
                                       const eOps::Descriptor::RichMLService &service, eOps::Agents::Role role)
        {
            size_t replicas = role == eOps::Agents::Role::Inference ? service.m_Inference->m_Replicas : 1;
            if (!options.m_Node.empty())
            {
                return std::vector<std::string>(1, options.m_Node);
            }
            eOps::Agents::Placement placement(components.Environment());
            eOps::Agents::PlacementPlan plan =
                placement.Place(service, role, components.m_Knowledge->Inventory(), components.m_Knowledge->Network());
            if (plan.m_Nodes.size() == replicas)
            {
                return plan.m_Nodes;
            }
            std::cerr << "EmpyHarness render: no feasible " << eOps::Agents::RoleName(role)
                      << " placement in the configured knowledge; using node 'local'\n";
            return std::vector<std::string>(1, "local");
        }
    } // namespace

    int32_t Validate(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        eOps::Harness::HarnessConfig config = Offline(Effective(options));
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        const std::string &file = options.m_Positional.front();
        std::string text = eOps::ReadFile(file, 1 << 20);
        eOps::Json report = eOps::Json::MakeObject({{"file", file}});
        try
        {
            eOps::Descriptor::RichMLService service = eOps::Descriptor::ParseDescriptor(text, file);
            eOps::Json document = eOps::FromYaml(eCore::YamlNode::Parse(text, file));
            std::vector<eOps::Descriptor::Issue> issues =
                eOps::Descriptor::ValidateDocument(document, service, Validation(config, components));
            bool valid = !eOps::Descriptor::HasErrors(issues);
            report.Set("service", service.m_Metadata.m_Name);
            report.Set("version", service.m_Metadata.m_Version);
            report.Set("valid", valid);
            report.Set("issues", eOps::Descriptor::ToJson(issues));
            Print(report);
            return valid ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        catch (const std::exception &error)
        {
            report.Set("valid", false);
            report.Set("error", error.what());
            Print(report);
            return EXIT_FAILURE;
        }
    }

    int32_t Interpret(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        eOps::Harness::HarnessConfig config = Offline(Effective(options));
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        eOps::Agents::Executive executive(components.Environment());
        executive.SetValidation(Validation(config, components));
        eOps::Agents::Interpretation interpretation =
            executive.Interpret(options.m_Positional.front(), options.m_Operator);
        // the BoM on stdout, the reading of the intent on stderr
        Out() << interpretation.m_Yaml;
        if (!interpretation.m_Yaml.empty() && interpretation.m_Yaml.back() != '\n')
        {
            Out() << '\n';
        }
        Out().flush();
        eOps::Json notes = eOps::Json::MakeObject({{"frame", interpretation.m_Frame.ToJson()},
                                                   {"issues", eOps::Descriptor::ToJson(interpretation.m_Issues)},
                                                   {"plan", interpretation.m_Plan.ToJson()}});
        std::cerr << notes.Dump(2) << std::endl;
        return eOps::Descriptor::HasErrors(interpretation.m_Issues) ? EXIT_FAILURE : EXIT_SUCCESS;
    }

    int32_t Render(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        eOps::Harness::HarnessConfig raw = Effective(options);
        eOps::Harness::HarnessConfig config = Offline(raw);
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        eOps::Descriptor::RichMLService service = eOps::Descriptor::LoadDescriptor(options.m_Positional.front());
        eOps::Substrate::KubernetesOptions kubernetes;
        if (raw.m_Substrate.IsMap() && raw.m_Substrate.Get<std::string>("type", "") == "kubernetes")
        {
            kubernetes.m_Namespace = raw.m_Substrate.Get<std::string>("namespace", kubernetes.m_Namespace);
            kubernetes.m_Image = raw.m_Substrate.Get<std::string>("image", kubernetes.m_Image);
        }
        if (!options.m_Namespace.empty())
        {
            kubernetes.m_Namespace = options.m_Namespace;
        }
        if (!options.m_Image.empty())
        {
            kubernetes.m_Image = options.m_Image;
        }
        std::filesystem::path out(options.m_Out);
        std::filesystem::create_directories(out);
        eOps::Topology::Inventory inventory = components.m_Knowledge->Inventory();
        eOps::Json files = eOps::Json::MakeArray();
        for (eOps::Agents::Role role : {eOps::Agents::Role::Inference, eOps::Agents::Role::Training})
        {
            if ((role == eOps::Agents::Role::Inference && !service.m_Inference) ||
                (role == eOps::Agents::Role::Training && !service.m_Training))
            {
                continue;
            }
            std::vector<std::string> nodes = Nodes(options, components, service, role);
            for (size_t replica = 0; replica < nodes.size(); ++replica)
            {
                eOps::Substrate::DeploymentSpec spec;
                spec.m_Service = service.m_Metadata.m_Name;
                spec.m_Version = service.m_Metadata.m_Version;
                spec.m_Role = eOps::Agents::RoleName(role);
                spec.m_Descriptor = service;
                spec.m_Node = nodes[replica];
                const eOps::Topology::Node *node = inventory.Find(spec.m_Node);
                spec.m_Region = node ? node->m_Region : std::string();
                spec.m_Replica = static_cast<uint32_t>(replica);
                spec.m_Instance =
                    eOps::Substrate::InstanceName(spec.m_Service, spec.m_Role, spec.m_Node, spec.m_Replica);
                spec.m_BrokerHost = config.m_BrokerHost;
                spec.m_BrokerPort = config.m_BrokerPort;
                spec.m_TopicRoot = config.m_TopicRoot;
                std::filesystem::path file = out / (spec.m_Instance + ".yaml");
                eOps::WriteFileAtomic(
                    file, eOps::Substrate::RenderManifests(spec, kubernetes.m_Namespace, kubernetes.m_Image));
                files.Push(eOps::Json::MakeObject({{"instance", spec.m_Instance},
                                                   {"role", spec.m_Role},
                                                   {"node", spec.m_Node},
                                                   {"file", file.string()}}));
            }
        }
        Print(eOps::Json::MakeObject({{"service", service.m_Metadata.m_Name},
                                      {"namespace", kubernetes.m_Namespace},
                                      {"image", kubernetes.m_Image},
                                      {"manifests", std::move(files)}}));
        return EXIT_SUCCESS;
    }

    int32_t Query(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        eOps::Harness::HarnessConfig config = Offline(Effective(options), true);
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        eOps::Context::RetrievalRequest request;
        request.m_Agent = options.m_Operator.empty() ? "operator" : options.m_Operator;
        request.m_Intent = options.m_Positional.front();
        request.m_Limit = options.m_Limit.value_or(20);
        Print(components.m_Comas->Retrieve(request).ToJson());
        return EXIT_SUCCESS;
    }

    int32_t McpServe(const Options &options)
    {
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        // stdout carries the protocol only (logging is routed to stderr by the StdoutRoute of main)
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        eOps::Harness::HarnessConfig config = Offline(Effective(options), true);
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        eOps::Context::McpServer server(*components.m_Comas);
        return server.Run(std::cin, Out());
    }
} // namespace eHarness
