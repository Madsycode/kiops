#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <string_view>
#include "Agent.h"
#include "../Common/Json.h"
#include "../Descriptor/Model.h"

// Compliance agent: governance and conflict resolution across policy constraints, topic
// ownership, action permissions, safety requirements, tenant isolation and declared deployment
// intent; actionable-topic conflicts between services (oscillation risk) are resolved by
// priority or escalated to a human (HITL via MMQP request/response).

namespace eOps::Agents
{
    struct EMPY_OPS_API PolicySet
    {
        struct Ownership
        {
            std::string m_Filter;
            std::string m_Owner;
        };
        struct Permission
        {
            std::string m_Subject; // service name, owner, or "*"
            std::vector<std::string> m_Allow;
            std::vector<std::string> m_Deny;
        };
        struct Safety
        {
            std::string m_Filter;
            std::string m_Unit;
            std::optional<double> m_Min;
            std::optional<double> m_Max;
        };
        struct Tenant
        {
            std::string m_Name;
            std::string m_Root;
        };

        std::vector<Ownership> m_Ownership;
        std::vector<Permission> m_Permissions;
        std::vector<Safety> m_Safety;
        std::vector<Tenant> m_Tenants;
        std::vector<std::string> m_DeniedTopics;
        bool m_RequireEncryptionForTenants = false;
        uint32_t m_MaxActionRatePerMinute = 0; // 0 = unlimited

        // Policies.yaml: denied_topics, ownership[{filter, owner}], permissions[{subject, allow, deny}],
        // safety[{filter, unit, min, max}], tenants[{name, root}], require_encryption_for_tenants,
        // max_action_rate_per_minute; throws eCore::ConfigError
        static PolicySet FromYaml(const eCore::YamlNode &node);
        Json ToJson() const; // same keys as the YAML form
        static PolicySet FromJson(const Json &json);
    };

    enum class Verdict : uint8_t
    {
        Approved,
        Amended,
        Rejected,
        Escalate
    };

    EMPY_OPS_API const char *VerdictName(Verdict verdict);    // approved | amended | rejected | escalate
    EMPY_OPS_API Verdict ParseVerdict(std::string_view name); // throws std::invalid_argument

    struct EMPY_OPS_API Finding
    {
        std::string m_Criterion; // policy | ownership | permission | safety | tenant | intent | conflict
        std::string m_Severity;  // info | warning | error
        std::string m_Message;
        std::string m_Topic;
        std::string m_Service; // the other service for conflicts
        Json ToJson() const;   // {criterion, severity, message, topic, service}
        static Finding FromJson(const Json &json);
    };

    struct EMPY_OPS_API ComplianceReport
    {
        Verdict m_Verdict = Verdict::Approved;
        std::vector<Finding> m_Findings;
        std::vector<Json> m_Resolutions; // {type: clamp|drop|..., topic, detail}
        std::optional<Descriptor::RichMLService> m_Amended;
        std::string m_Question; // HITL question when escalating
        // {verdict, findings[], resolutions[], amended (descriptor JSON or null), question}
        Json ToJson() const;
        static ComplianceReport FromJson(const Json &json);
    };

    class EMPY_OPS_API Compliance
    {
    public:
        Compliance(AgentEnvironment environment, PolicySet policies);

        // verdict precedence: any error → Rejected; else escalation → Escalate; else amendments → Amended;
        // else Approved. `deployed` services with the candidate's own name are ignored (redeploys).
        ComplianceReport Review(const Descriptor::RichMLService &candidate,
                                const std::vector<Descriptor::RichMLService> &deployed) const;

        const PolicySet &Policies() const noexcept;

    private:
        AgentEnvironment m_Environment;
        PolicySet m_Policies;
    };
} // namespace eOps::Agents
