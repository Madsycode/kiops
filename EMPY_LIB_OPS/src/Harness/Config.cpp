#include <eOps/Harness/Config.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/Process.h>
#include <eOps/Fabric/Topics.h>
#include <eCore/Common/Stdout.h>
#include <algorithm>

namespace eOps::Harness
{
    namespace
    {
        const char *const TOP_LEVEL[] = {
            "broker",         "topic_root", "identity",     "roles",           "ontology",
            "ontology_state", "knowledge",  "policies",     "state_directory", "knowledge_directory",
            "status",         "timeouts",   "hitl",         "wait_for_ready",  "placement",
            "comas",          "compliance", "intelligence", "substrate"};
        const char *const ROLES[] = {"harness", "executive", "placement", "lifecycle", "compliance"};

        void RequireKeys(const eCore::YamlNode &node, std::initializer_list<const char *> allowed)
        {
            if (node.IsNull())
            {
                return;
            }
            if (!node.IsMap())
            {
                throw eCore::ConfigError(node.Path() + ": expected a mapping");
            }
            for (const std::string &key : node.Keys())
            {
                if (std::none_of(allowed.begin(), allowed.end(), [&](const char *name) { return key == name; }))
                {
                    throw eCore::ConfigError(node.Path() + "." + key + ": unknown key");
                }
            }
        }

        std::vector<std::string> StringList(const eCore::YamlNode &node)
        {
            std::vector<std::string> items;
            for (const eCore::YamlNode &item : node.Items())
            {
                items.push_back(item.As<std::string>());
            }
            return items;
        }

        // a relative path is taken relative to the configuration file
        std::string Resolve(const std::string &config, const std::string &value)
        {
            return eCore::ConfigPath(config.c_str(), value);
        }

        bool AgentRole(std::string_view role)
        {
            return role == "executive" || role == "placement" || role == "lifecycle" || role == "compliance";
        }

        // the workload sessions are secure when the process substrate hands them a TLS or QUIC client config
        bool SecureWorkloads(const eCore::YamlNode &substrate, const std::string &base)
        {
            if (!substrate.IsMap() || substrate.Get<std::string>("type", "") != "process")
            {
                return false;
            }
            std::string client = substrate.Get<std::string>("client_config", "");
            if (client.empty())
            {
                return false;
            }
            try
            {
                eMQ::Configs configs = eMQ::LoadConfigs(Resolve(base, client).c_str());
                return configs.m_Transport.m_Tls || configs.m_Transport.m_Quic;
            }
            catch (const std::exception &error)
            {
                throw eCore::ConfigError(substrate.Path() + ".client_config: " + error.what());
            }
        }
    } // namespace

    HarnessConfig ParseHarnessConfig(std::string_view yaml, const std::string &path)
    {
        eCore::YamlNode root = eCore::YamlNode::Parse(yaml, path.empty() ? "<harness>" : path);
        HarnessConfig config;
        config.m_Path = path.empty() ? std::string() : std::filesystem::absolute(path).lexically_normal().string();
        if (root.IsNull())
        {
            return config;
        }
        if (!root.IsMap())
        {
            throw eCore::ConfigError(root.Path() + ": expected a mapping");
        }
        for (const std::string &key : root.Keys())
        {
            if (std::none_of(std::begin(TOP_LEVEL), std::end(TOP_LEVEL), [&](const char *name) { return key == name; }))
            {
                throw eCore::ConfigError(root.Path() + "." + key + ": unknown key");
            }
        }
        const std::string &base = config.m_Path;

        eCore::YamlNode broker = root["broker"];
        RequireKeys(broker, {"host", "port", "client_config", "secure_workloads"});
        config.m_BrokerHost = broker.Get<std::string>("host", config.m_BrokerHost);
        config.m_BrokerPort = broker.Get<uint16_t>("port", config.m_BrokerPort, 1, UINT16_MAX);
        config.m_ClientConfig = Resolve(base, broker.Get<std::string>("client_config", ""));
        if (config.m_BrokerHost.empty())
        {
            throw eCore::ConfigError(broker.Path() + ".host: must not be empty");
        }

        config.m_TopicRoot = root.Get<std::string>("topic_root", config.m_TopicRoot);
        try
        {
            config.m_TopicRoot = Fabric::Topics(config.m_TopicRoot).m_Root;
        }
        catch (const std::exception &error)
        {
            throw eCore::ConfigError(root.Path() + ".topic_root: " + error.what());
        }
        config.m_Identity = root.Get<std::string>("identity", config.m_Identity);
        if (config.m_Identity.empty())
        {
            throw eCore::ConfigError(root.Path() + ".identity: must not be empty");
        }
        if (root.Has("roles"))
        {
            config.m_Roles = StringList(root["roles"]);
            for (const std::string &role : config.m_Roles)
            {
                if (std::none_of(std::begin(ROLES), std::end(ROLES), [&](const char *name) { return role == name; }))
                {
                    throw eCore::ConfigError(root.Path() + ".roles: unknown role '" + role + "'");
                }
            }
        }

        config.m_Ontology = Resolve(base, root.Get<std::string>("ontology", ""));
        config.m_OntologyState = Resolve(base, root.Get<std::string>("ontology_state", ""));
        config.m_Knowledge = Resolve(base, root.Get<std::string>("knowledge", ""));
        config.m_Policies = Resolve(base, root.Get<std::string>("policies", ""));
        config.m_StateDirectory = Resolve(base, root.Get<std::string>("state_directory", ""));
        config.m_KnowledgeDirectory = Resolve(base, root.Get<std::string>("knowledge_directory", ""));

        eCore::YamlNode status = root["status"];
        RequireKeys(status, {"address", "port", "enabled"});
        config.m_StatusAddress = status.Get<std::string>("address", config.m_StatusAddress);
        config.m_StatusPort = status.Get<uint16_t>("port", config.m_StatusPort, 0, UINT16_MAX);
        config.m_StatusEnabled = status.Get<bool>("enabled", config.m_StatusEnabled);

        eCore::YamlNode timeouts = root["timeouts"];
        RequireKeys(timeouts, {"agent_ms", "hitl_ms", "ready_ms", "liveness_ms"});
        config.m_AgentTimeoutMs = timeouts.Get<uint32_t>("agent_ms", config.m_AgentTimeoutMs, 1, 3600000);
        config.m_HitlTimeoutMs = timeouts.Get<uint32_t>("hitl_ms", config.m_HitlTimeoutMs, 1, 86400000);
        config.m_ReadyTimeoutMs = timeouts.Get<uint32_t>("ready_ms", config.m_ReadyTimeoutMs, 1, 3600000);
        config.m_LivenessMs = timeouts.Get<uint32_t>("liveness_ms", config.m_LivenessMs, 0, 3600000);

        eCore::YamlNode hitl = root["hitl"];
        RequireKeys(hitl, {"default"});
        std::string decision = Lower(hitl.Get<std::string>("default", "reject"));
        if (decision != "approve" && decision != "reject")
        {
            throw eCore::ConfigError(hitl.Path() + ".default: expected approve or reject");
        }
        config.m_HitlDefaultApprove = decision == "approve";
        config.m_WaitForReady = root.Get<bool>("wait_for_ready", config.m_WaitForReady);

        eCore::YamlNode placement = root["placement"];
        RequireKeys(placement, {"audit_seconds", "weights"});
        config.m_PlacementAuditSeconds = placement.Get<uint32_t>("audit_seconds", 0, 0, 86400);
        config.m_Weights = Agents::PlacementWeights::FromYaml(placement["weights"]);

        eCore::YamlNode comas = root["comas"];
        RequireKeys(comas, {"consolidation_seconds", "mcp_command"});
        config.m_ConsolidationSeconds = comas.Get<uint32_t>("consolidation_seconds", 60, 0, 86400);
        config.m_McpCommand = StringList(comas["mcp_command"]);
        if (!config.m_McpCommand.empty() && config.m_McpCommand.front().find_first_of("/\\") != std::string::npos)
        {
            config.m_McpCommand.front() = Resolve(base, config.m_McpCommand.front());
        }

        eCore::YamlNode compliance = root["compliance"];
        RequireKeys(compliance, {"violation_limit", "violation_window_seconds"});
        config.m_ViolationLimit = compliance.Get<uint32_t>("violation_limit", config.m_ViolationLimit, 0, 1000000);
        config.m_ViolationWindowSeconds =
            compliance.Get<uint32_t>("violation_window_seconds", config.m_ViolationWindowSeconds, 1, 86400);

        config.m_Intelligence = root["intelligence"];
        config.m_Substrate = root["substrate"];
        config.m_SecureWorkloads = broker.Has("secure_workloads") ? broker.Get<bool>("secure_workloads", false)
                                                                  : SecureWorkloads(config.m_Substrate, base);
        return config;
    }

    HarnessConfig LoadHarnessConfig(const std::filesystem::path &path)
    {
        std::string text;
        try
        {
            text = ReadFile(path, 1 << 20);
        }
        catch (const std::exception &error)
        {
            throw eCore::ConfigError("cannot read harness configuration '" + path.string() + "': " + error.what());
        }
        return ParseHarnessConfig(text, path.string());
    }

    HarnessComponents BuildComponents(const HarnessConfig &config)
    {
        HarnessComponents components;
        // the evolved ontology of an earlier run wins over the seed (the map/evolve loop accumulates)
        std::error_code missing;
        std::string ontology =
            !config.m_OntologyState.empty() && std::filesystem::exists(config.m_OntologyState, missing)
                ? config.m_OntologyState
                : config.m_Ontology;
        components.m_Ontology = std::make_shared<Semantics::Ontology>(
            ontology.empty() ? Semantics::Ontology() : Semantics::Ontology::Load(ontology));

        Context::KnowledgeOptions knowledge;
        knowledge.m_Directory = config.m_KnowledgeDirectory;
        components.m_Knowledge = std::make_shared<Context::KnowledgePlane>(knowledge);
        if (!config.m_Knowledge.empty())
        {
            components.m_Knowledge->Seed(eCore::YamlNode::Load(config.m_Knowledge));
        }

        components.m_Providers = std::make_shared<Intelligence::ProviderSet>(
            Intelligence::ProviderSet::FromYaml(config.m_Intelligence, config.m_Path));

        std::shared_ptr<Context::KnowledgeSource> source;
        if (!config.m_McpCommand.empty())
        {
            ProcessOptions server;
            server.m_Program = config.m_McpCommand.front();
            server.m_Args.assign(config.m_McpCommand.begin() + 1, config.m_McpCommand.end());
            auto client = std::make_shared<Context::McpClient>();
            std::string error;
            if (!client->Start(server, &error))
            {
                throw std::runtime_error("comas.mcp_command: MCP knowledge server failed to start: " + error);
            }
            components.m_Mcp = client;
            source = std::make_shared<Context::McpSource>(client);
        }
        // the deleter keeps the plane, providers and MCP client alive as long as CoMaS
        std::shared_ptr<Context::KnowledgePlane> plane = components.m_Knowledge;
        std::shared_ptr<Intelligence::ProviderSet> providers = components.m_Providers;
        std::shared_ptr<Context::McpClient> mcp = components.m_Mcp;
        components.m_Comas =
            std::shared_ptr<Context::Comas>(new Context::Comas(*plane, providers.get(), source),
                                            [plane, providers, mcp](Context::Comas *comas) { delete comas; });

        components.m_Substrate = Substrate::MakeSubstrate(config.m_Substrate, config.m_Path);
        if (!config.m_Policies.empty())
        {
            components.m_Policies = Agents::PolicySet::FromYaml(eCore::YamlNode::Load(config.m_Policies));
        }
        components.m_Regions = components.m_Knowledge->Network().m_Regions;
        return components;
    }

    HarnessOptions MakeHarnessOptions(const HarnessConfig &config, const HarnessComponents &components)
    {
        HarnessOptions options;
        options.m_Topics = Fabric::Topics(config.m_TopicRoot);
        options.m_Identity = config.m_Identity;
        options.m_AgentTimeoutMs = config.m_AgentTimeoutMs;
        options.m_HitlTimeoutMs = config.m_HitlTimeoutMs;
        options.m_HitlDefaultApprove = config.m_HitlDefaultApprove;
        options.m_PlacementAuditSeconds = config.m_PlacementAuditSeconds;
        options.m_ConsolidationSeconds = config.m_ConsolidationSeconds;
        options.m_StateDirectory = config.m_StateDirectory;
        options.m_WaitForReady = config.m_WaitForReady;
        options.m_ReadyTimeoutMs = config.m_ReadyTimeoutMs;
        options.m_BrokerHost = config.m_BrokerHost;
        options.m_BrokerPort = config.m_BrokerPort;
        options.m_LivenessMs = config.m_LivenessMs;
        options.m_SecureWorkloads = config.m_SecureWorkloads;
        options.m_Validation.m_Regions = components.m_Regions;
        options.m_Validation.m_TopicRoot = config.m_TopicRoot;
        return options;
    }

    AgentHostOptions MakeAgentHostOptions(const HarnessConfig &config, const HarnessComponents &components)
    {
        AgentHostOptions options;
        options.m_Roles.clear();
        for (const std::string &role : config.m_Roles)
        {
            if (AgentRole(role))
            {
                options.m_Roles.push_back(role);
            }
        }
        options.m_Topics = Fabric::Topics(config.m_TopicRoot);
        options.m_Policies = components.m_Policies;
        options.m_Weights = config.m_Weights;
        options.m_Identity = config.HasRole("harness") ? config.m_Identity + "-agents" : config.m_Identity;
        // a proposal is answered when its workflow finishes: training, evaluation and the model load
        uint64_t timeout = static_cast<uint64_t>(config.m_AgentTimeoutMs) * 3 + 5000;
        options.m_HarnessTimeoutMs = static_cast<uint32_t>(std::min<uint64_t>(timeout, UINT32_MAX));
        options.m_OntologyState = config.m_OntologyState;
        options.m_ViolationLimit = config.m_ViolationLimit;
        options.m_ViolationWindowSeconds = config.m_ViolationWindowSeconds;
        options.m_Validation.m_Regions = components.m_Regions;
        options.m_Validation.m_TopicRoot = config.m_TopicRoot;
        return options;
    }
} // namespace eOps::Harness
