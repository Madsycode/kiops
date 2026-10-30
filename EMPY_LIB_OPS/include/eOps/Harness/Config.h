#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <eCore/Common/YAML.h>
#include "Harness.h"
#include "AgentHost.h"
#include "../Agents/Agent.h"
#include "../Agents/Compliance.h"
#include "../Agents/Placement.h"
#include "../Context/Comas.h"
#include "../Context/Knowledge.h"
#include "../Context/Mcp.h"
#include "../Intelligence/Provider.h"
#include "../Semantics/Ontology.h"
#include "../Substrate/Substrate.h"

// Configs/MLOps/Harness.yaml loader (paths relative to the file) and the assembly of the
// harness components (ontology, knowledge plane, CoMaS, providers, substrate, policies)
//
//   broker: {host, port, client_config, secure_workloads}   topic_root     identity     roles: [...]
//   ontology, ontology_state, knowledge, policies           state_directory              knowledge_directory
//   status: {address, port}                  timeouts: {agent_ms, hitl_ms, ready_ms, liveness_ms}
//   hitl: {default: approve|reject}          wait_for_ready
//   placement: {audit_seconds, weights: {...}}   comas: {consolidation_seconds, mcp_command: [...]}
//   compliance: {violation_limit, violation_window_seconds}
//   intelligence: {...} (Intelligence/Provider.h)   substrate: {...} (Substrate/Substrate.h)
//
// ontology_state: the evolved BoM ontology is saved there and loaded in preference to `ontology`
// when it exists. broker.secure_workloads: the workload MMQP sessions use TLS or QUIC (default:
// derived from the process substrate's client_config); descriptors declaring security.encryption
// are refused otherwise.

namespace eOps::Harness
{
    struct HarnessConfig
    {
        std::string m_Path; // absolute configuration file path ("" when parsed from text without one)
        std::string m_BrokerHost = "127.0.0.1";
        uint16_t m_BrokerPort = 5600;
        std::string m_ClientConfig; // eMQ client YAML (absolute), "" = defaults
        std::string m_TopicRoot = "mlops";
        std::string m_Identity = "harness";
        std::vector<std::string> m_Roles = {"harness", "executive", "placement", "lifecycle", "compliance"};
        std::string m_Ontology;      // absolute paths, "" = none
        std::string m_OntologyState; // evolved ontology (saved by the executive, loaded in preference to m_Ontology)
        std::string m_Knowledge;
        std::string m_Policies;
        std::string m_StateDirectory;
        std::string m_KnowledgeDirectory;
        std::string m_StatusAddress = "127.0.0.1";
        uint16_t m_StatusPort = 0;
        bool m_StatusEnabled = true; // false when the status section sets enabled: false
        uint32_t m_AgentTimeoutMs = 30000;
        uint32_t m_HitlTimeoutMs = 60000;
        uint32_t m_ReadyTimeoutMs = 30000;
        uint32_t m_LivenessMs = 15000; // 0 = no liveness check
        bool m_SecureWorkloads = false;
        uint32_t m_ViolationLimit = 3; // 0 = violations never propose a retirement
        uint32_t m_ViolationWindowSeconds = 300;
        bool m_HitlDefaultApprove = false;
        bool m_WaitForReady = false;
        uint32_t m_PlacementAuditSeconds = 0;
        Agents::PlacementWeights m_Weights;
        uint32_t m_ConsolidationSeconds = 60;
        std::vector<std::string> m_McpCommand;
        eCore::YamlNode m_Intelligence; // raw sections, resolved against m_Path by BuildComponents
        eCore::YamlNode m_Substrate;

        bool HasRole(std::string_view role) const
        {
            for (const std::string &item : m_Roles)
            {
                if (item == role)
                {
                    return true;
                }
            }
            return false;
        }
    };

    // throws eCore::ConfigError naming the offending key
    EMPY_OPS_API HarnessConfig LoadHarnessConfig(const std::filesystem::path &path);
    EMPY_OPS_API HarnessConfig ParseHarnessConfig(std::string_view yaml, const std::string &path);

    struct HarnessComponents
    {
        std::shared_ptr<Semantics::Ontology> m_Ontology;
        std::shared_ptr<Context::KnowledgePlane> m_Knowledge;
        std::shared_ptr<Intelligence::ProviderSet> m_Providers;
        std::shared_ptr<Context::McpClient> m_Mcp; // null without comas.mcp_command
        std::shared_ptr<Context::Comas> m_Comas;   // keeps m_Knowledge/m_Providers/m_Mcp alive
        std::shared_ptr<Substrate::ExecutionSubstrate> m_Substrate;
        Agents::PolicySet m_Policies;
        std::vector<std::string> m_Regions; // network regions (validation)

        Agents::AgentEnvironment Environment() const
        {
            return Agents::AgentEnvironment{m_Comas.get(), m_Providers.get(), m_Ontology.get()};
        }
    };

    // loads ontology/knowledge/policies, seeds the plane, builds providers, MCP source and substrate;
    // throws eCore::ConfigError / std::runtime_error
    EMPY_OPS_API HarnessComponents BuildComponents(const HarnessConfig &config);
    EMPY_OPS_API HarnessOptions MakeHarnessOptions(const HarnessConfig &config, const HarnessComponents &components);
    EMPY_OPS_API AgentHostOptions MakeAgentHostOptions(const HarnessConfig &config,
                                                       const HarnessComponents &components);
} // namespace eOps::Harness
