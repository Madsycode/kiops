#include <eOps/Agents/Compliance.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Descriptor/Validator.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/YAML.h>
#include "Detail.h"
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>

namespace eOps::Agents
{
    namespace
    {
        constexpr size_t MAX_QUESTION_BYTES = 4096;

        void RequireFilter(const std::string &filter, const std::string &path)
        {
            if (!ValidTopic(filter, true))
            {
                throw JsonError(path + ": '" + filter + "' is not a valid MMQP topic filter");
            }
        }

        void RejectUnknown(const Json &json, std::initializer_list<std::string_view> keys, const std::string &path)
        {
            for (const auto &[key, value] : json.AsObject())
            {
                if (std::find(keys.begin(), keys.end(), key) == keys.end())
                {
                    throw JsonError(path + "." + key + ": unknown policy key");
                }
            }
        }

        std::string At(const std::string &path, size_t index)
        {
            return path + "[" + std::to_string(index) + "]";
        }

        // every topic a service touches: observables, actionables and the training label
        std::vector<std::string> ServiceTopics(const Descriptor::RichMLService &service)
        {
            std::vector<std::string> topics;
            for (const Descriptor::Observable &observable : service.m_Observables)
            {
                topics.push_back(observable.m_Name);
            }
            for (const Descriptor::Actionable &actionable : service.m_Actionables)
            {
                topics.push_back(actionable.m_Name);
            }
            if (service.m_Training && !service.m_Training->m_Label.empty())
            {
                topics.push_back(service.m_Training->m_Label);
            }
            return topics;
        }

        std::string Bound(const std::optional<double> &value)
        {
            return value ? Detail::Text(*value) : std::string("unbounded");
        }

        std::string Range(const std::optional<double> &min, const std::optional<double> &max)
        {
            return "[" + Bound(min) + ", " + Bound(max) + "]";
        }

        std::string ObjectiveText(const std::vector<std::string> &objectives)
        {
            return objectives.empty() ? std::string("none") : Join(objectives, ", ");
        }

        // working state of one review
        struct ReviewState
        {
            ComplianceReport m_Report;
            Descriptor::RichMLService m_Amended;
            bool m_Error = false;
            bool m_Escalate = false;
            bool m_Changed = false;
            std::vector<std::string> m_Questions;

            void Add(std::string criterion, std::string severity, std::string message, std::string topic,
                     std::string service)
            {
                m_Error = m_Error || severity == "error";
                m_Report.m_Findings.push_back(Finding{std::move(criterion), std::move(severity), std::move(message),
                                                      std::move(topic), std::move(service)});
            }

            void Resolve(std::string type, std::string topic, std::string detail)
            {
                m_Changed = true;
                m_Report.m_Resolutions.push_back(
                    Json::MakeObject({{"type", std::move(type)}, {"topic", std::move(topic)}, {"detail", detail}}));
            }
        };
    } // namespace

    // -------------------------
    // PolicySet
    // -------------------------

    PolicySet PolicySet::FromJson(const Json &json)
    {
        PolicySet policies;
        if (json.IsNull())
        {
            return policies;
        }
        Detail::RequireObject(json, "policies");
        RejectUnknown(json,
                      {"denied_topics", "ownership", "permissions", "safety", "tenants",
                       "require_encryption_for_tenants", "max_action_rate_per_minute"},
                      "policies");
        policies.m_DeniedTopics = Detail::ReadStrings(json, "denied_topics");
        for (size_t i = 0; i < policies.m_DeniedTopics.size(); ++i)
        {
            RequireFilter(policies.m_DeniedTopics[i], At("denied_topics", i));
        }
        const Json::Array &ownership = Detail::ReadArray(json, "ownership");
        for (size_t i = 0; i < ownership.size(); ++i)
        {
            std::string path = At("ownership", i);
            Detail::RequireObject(ownership[i], path);
            RejectUnknown(ownership[i], {"filter", "owner"}, path);
            Ownership entry{Detail::ReadString(ownership[i], "filter"), Detail::ReadString(ownership[i], "owner")};
            RequireFilter(entry.m_Filter, path + ".filter");
            if (entry.m_Owner.empty())
            {
                throw JsonError(path + ".owner: required");
            }
            policies.m_Ownership.push_back(std::move(entry));
        }
        const Json::Array &permissions = Detail::ReadArray(json, "permissions");
        for (size_t i = 0; i < permissions.size(); ++i)
        {
            std::string path = At("permissions", i);
            Detail::RequireObject(permissions[i], path);
            RejectUnknown(permissions[i], {"subject", "allow", "deny"}, path);
            Permission entry;
            entry.m_Subject = Detail::ReadString(permissions[i], "subject");
            entry.m_Allow = Detail::ReadStrings(permissions[i], "allow");
            entry.m_Deny = Detail::ReadStrings(permissions[i], "deny");
            if (entry.m_Subject.empty())
            {
                throw JsonError(path + ".subject: required (service name, owner or '*')");
            }
            for (size_t j = 0; j < entry.m_Allow.size(); ++j)
            {
                RequireFilter(entry.m_Allow[j], At(path + ".allow", j));
            }
            for (size_t j = 0; j < entry.m_Deny.size(); ++j)
            {
                RequireFilter(entry.m_Deny[j], At(path + ".deny", j));
            }
            policies.m_Permissions.push_back(std::move(entry));
        }
        const Json::Array &safety = Detail::ReadArray(json, "safety");
        for (size_t i = 0; i < safety.size(); ++i)
        {
            std::string path = At("safety", i);
            Detail::RequireObject(safety[i], path);
            RejectUnknown(safety[i], {"filter", "unit", "min", "max"}, path);
            Safety entry;
            entry.m_Filter = Detail::ReadString(safety[i], "filter");
            entry.m_Unit = Detail::ReadString(safety[i], "unit");
            entry.m_Min = Detail::ReadOptionalNumber(safety[i], "min");
            entry.m_Max = Detail::ReadOptionalNumber(safety[i], "max");
            RequireFilter(entry.m_Filter, path + ".filter");
            if (!entry.m_Min && !entry.m_Max)
            {
                throw JsonError(path + ": a safety envelope needs min and/or max");
            }
            if (entry.m_Min && entry.m_Max && *entry.m_Min > *entry.m_Max)
            {
                throw JsonError(path + ": min must not exceed max");
            }
            policies.m_Safety.push_back(std::move(entry));
        }
        const Json::Array &tenants = Detail::ReadArray(json, "tenants");
        for (size_t i = 0; i < tenants.size(); ++i)
        {
            std::string path = At("tenants", i);
            Detail::RequireObject(tenants[i], path);
            RejectUnknown(tenants[i], {"name", "root"}, path);
            Tenant entry{Detail::ReadString(tenants[i], "name"), Detail::ReadString(tenants[i], "root")};
            if (entry.m_Name.empty())
            {
                throw JsonError(path + ".name: required");
            }
            if (entry.m_Root.empty())
            {
                entry.m_Root = entry.m_Name;
            }
            while (entry.m_Root.size() > 1 && entry.m_Root.back() == '/')
            {
                entry.m_Root.pop_back();
            }
            if (!ValidTopic(entry.m_Root, false) || IsFilter(entry.m_Root))
            {
                throw JsonError(path + ".root: '" + entry.m_Root + "' is not a concrete topic prefix");
            }
            policies.m_Tenants.push_back(std::move(entry));
        }
        policies.m_RequireEncryptionForTenants = Detail::ReadBool(json, "require_encryption_for_tenants");
        uint64_t rate = Detail::ReadUnsigned(json, "max_action_rate_per_minute");
        if (rate > std::numeric_limits<uint32_t>::max())
        {
            throw JsonError("max_action_rate_per_minute: out of range");
        }
        policies.m_MaxActionRatePerMinute = static_cast<uint32_t>(rate);
        return policies;
    }

    PolicySet PolicySet::FromYaml(const eCore::YamlNode &node)
    {
        try
        {
            return FromJson(eOps::FromYaml(node));
        }
        catch (const eCore::ConfigError &)
        {
            throw;
        }
        catch (const std::exception &error)
        {
            throw eCore::ConfigError(node.Path() + ": " + error.what());
        }
    }

    Json PolicySet::ToJson() const
    {
        Json ownership = Json::MakeArray();
        for (const Ownership &entry : m_Ownership)
        {
            ownership.Push(Json::MakeObject({{"filter", entry.m_Filter}, {"owner", entry.m_Owner}}));
        }
        Json permissions = Json::MakeArray();
        for (const Permission &entry : m_Permissions)
        {
            permissions.Push(Json::MakeObject({{"subject", entry.m_Subject},
                                               {"allow", Detail::Strings(entry.m_Allow)},
                                               {"deny", Detail::Strings(entry.m_Deny)}}));
        }
        Json safety = Json::MakeArray();
        for (const Safety &entry : m_Safety)
        {
            safety.Push(Json::MakeObject({{"filter", entry.m_Filter},
                                          {"unit", entry.m_Unit},
                                          {"min", Detail::Number(entry.m_Min)},
                                          {"max", Detail::Number(entry.m_Max)}}));
        }
        Json tenants = Json::MakeArray();
        for (const Tenant &entry : m_Tenants)
        {
            tenants.Push(Json::MakeObject({{"name", entry.m_Name}, {"root", entry.m_Root}}));
        }
        return Json::MakeObject({{"denied_topics", Detail::Strings(m_DeniedTopics)},
                                 {"ownership", std::move(ownership)},
                                 {"permissions", std::move(permissions)},
                                 {"safety", std::move(safety)},
                                 {"tenants", std::move(tenants)},
                                 {"require_encryption_for_tenants", m_RequireEncryptionForTenants},
                                 {"max_action_rate_per_minute", m_MaxActionRatePerMinute}});
    }

    // -------------------------
    // verdicts, findings, reports
    // -------------------------

    const char *VerdictName(Verdict verdict)
    {
        switch (verdict)
        {
        case Verdict::Amended:
            return "amended";
        case Verdict::Rejected:
            return "rejected";
        case Verdict::Escalate:
            return "escalate";
        case Verdict::Approved:
        default:
            return "approved";
        }
    }

    Verdict ParseVerdict(std::string_view name)
    {
        for (Verdict verdict : {Verdict::Approved, Verdict::Amended, Verdict::Rejected, Verdict::Escalate})
        {
            if (name == VerdictName(verdict))
            {
                return verdict;
            }
        }
        throw std::invalid_argument("unknown compliance verdict '" + std::string(name) + "'");
    }

    Json Finding::ToJson() const
    {
        return Json::MakeObject({{"criterion", m_Criterion},
                                 {"severity", m_Severity},
                                 {"message", m_Message},
                                 {"topic", m_Topic},
                                 {"service", m_Service}});
    }

    Finding Finding::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "finding");
        Finding finding;
        finding.m_Criterion = Detail::ReadString(json, "criterion");
        finding.m_Severity = Detail::ReadString(json, "severity", "info");
        finding.m_Message = Detail::ReadString(json, "message");
        finding.m_Topic = Detail::ReadString(json, "topic");
        finding.m_Service = Detail::ReadString(json, "service");
        if (finding.m_Severity != "info" && finding.m_Severity != "warning" && finding.m_Severity != "error")
        {
            throw JsonError("finding severity must be info, warning or error");
        }
        return finding;
    }

    Json ComplianceReport::ToJson() const
    {
        Json findings = Json::MakeArray();
        for (const Finding &finding : m_Findings)
        {
            findings.Push(finding.ToJson());
        }
        Json resolutions = Json::MakeArray();
        for (const Json &resolution : m_Resolutions)
        {
            resolutions.Push(resolution);
        }
        return Json::MakeObject({{"verdict", VerdictName(m_Verdict)},
                                 {"findings", std::move(findings)},
                                 {"resolutions", std::move(resolutions)},
                                 {"amended", m_Amended ? Descriptor::ToJson(*m_Amended) : Json()},
                                 {"question", m_Question}});
    }

    ComplianceReport ComplianceReport::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "compliance report");
        ComplianceReport report;
        try
        {
            report.m_Verdict = ParseVerdict(Detail::ReadString(json, "verdict", "approved"));
        }
        catch (const std::invalid_argument &error)
        {
            throw JsonError(error.what());
        }
        for (const Json &item : Detail::ReadArray(json, "findings"))
        {
            report.m_Findings.push_back(Finding::FromJson(item));
        }
        for (const Json &item : Detail::ReadArray(json, "resolutions"))
        {
            report.m_Resolutions.push_back(item);
        }
        if (const Json *amended = json.Find("amended"); amended && !amended->IsNull())
        {
            try
            {
                report.m_Amended = Descriptor::FromJson(*amended);
            }
            catch (const Descriptor::DescriptorError &error)
            {
                throw JsonError(std::string("amended descriptor: ") + error.what());
            }
        }
        report.m_Question = Detail::ReadString(json, "question");
        return report;
    }

    // -------------------------
    // Compliance
    // -------------------------

    Compliance::Compliance(AgentEnvironment environment, PolicySet policies)
        : m_Environment(environment), m_Policies(std::move(policies))
    {
    }

    const PolicySet &Compliance::Policies() const noexcept
    {
        return m_Policies;
    }

    ComplianceReport Compliance::Review(const Descriptor::RichMLService &candidate,
                                        const std::vector<Descriptor::RichMLService> &deployed) const
    {
        ReviewState review;
        review.m_Amended = candidate;
        const std::string &name = candidate.m_Metadata.m_Name;
        const std::string &owner = candidate.m_Constraints.m_Owner;
        std::vector<std::string> topics = ServiceTopics(candidate);
        const Semantics::Ontology *ontology = m_Environment.m_Ontology;

        // policy constraints: denied topics, action rate ceiling
        for (const std::string &topic : topics)
        {
            for (const std::string &denied : m_Policies.m_DeniedTopics)
            {
                if (FiltersOverlap(topic, denied))
                {
                    review.Add("policy", "error", "topic '" + topic + "' is denied by policy ('" + denied + "')", topic,
                               name);
                }
            }
        }
        if (uint32_t ceiling = m_Policies.m_MaxActionRatePerMinute)
        {
            for (Descriptor::Actionable &actionable : review.m_Amended.m_Actionables)
            {
                if (actionable.m_MaxRatePerMinute == 0 || actionable.m_MaxRatePerMinute > ceiling)
                {
                    std::string before = actionable.m_MaxRatePerMinute ? std::to_string(actionable.m_MaxRatePerMinute)
                                                                       : std::string("unlimited");
                    actionable.m_MaxRatePerMinute = ceiling;
                    review.Add("policy", "warning",
                               "action rate of '" + actionable.m_Name + "' limited to the policy ceiling " +
                                   std::to_string(ceiling) + "/min (was " + before + ")",
                               actionable.m_Name, name);
                    review.Resolve("rate", actionable.m_Name, before + " → " + std::to_string(ceiling) + " per minute");
                }
            }
        }

        // topic ownership
        for (const Descriptor::Actionable &actionable : candidate.m_Actionables)
        {
            for (const PolicySet::Ownership &entry : m_Policies.m_Ownership)
            {
                if (!FiltersOverlap(actionable.m_Name, entry.m_Filter) || entry.m_Owner == owner)
                {
                    continue;
                }
                if (owner.empty())
                {
                    review.Add("ownership", "warning",
                               "'" + actionable.m_Name + "' lies in '" + entry.m_Filter + "' owned by '" +
                                   entry.m_Owner + "'; the service declares no owner",
                               actionable.m_Name, name);
                }
                else
                {
                    review.Add("ownership", "error",
                               "'" + actionable.m_Name + "' lies in '" + entry.m_Filter + "' owned by '" +
                                   entry.m_Owner + "', not by '" + owner + "'",
                               actionable.m_Name, name);
                }
            }
        }

        // action permissions
        std::vector<const PolicySet::Permission *> applicable;
        for (const PolicySet::Permission &entry : m_Policies.m_Permissions)
        {
            if (entry.m_Subject == "*" || entry.m_Subject == name || (!owner.empty() && entry.m_Subject == owner))
            {
                applicable.push_back(&entry);
            }
        }
        bool allowLists = std::any_of(applicable.begin(), applicable.end(),
                                      [](const PolicySet::Permission *entry) { return !entry->m_Allow.empty(); });
        for (const Descriptor::Actionable &actionable : candidate.m_Actionables)
        {
            bool allowed = false;
            for (const PolicySet::Permission *entry : applicable)
            {
                for (const std::string &deny : entry->m_Deny)
                {
                    if (FiltersOverlap(actionable.m_Name, deny))
                    {
                        review.Add("permission", "error",
                                   "actions on '" + actionable.m_Name + "' are denied for '" + entry->m_Subject +
                                       "' ('" + deny + "')",
                                   actionable.m_Name, name);
                    }
                }
                for (const std::string &allow : entry->m_Allow)
                {
                    allowed = allowed || FilterCovers(allow, actionable.m_Name);
                }
            }
            if (allowLists && !allowed)
            {
                review.Add("permission", "error",
                           "actions on '" + actionable.m_Name + "' are not covered by any permission granted to '" +
                               name + "'",
                           actionable.m_Name, name);
            }
        }

        // safety envelopes: clamp declared bounds to the envelope, reject an empty intersection
        for (Descriptor::Actionable &actionable : review.m_Amended.m_Actionables)
        {
            for (const PolicySet::Safety &envelope : m_Policies.m_Safety)
            {
                if (!FiltersOverlap(actionable.m_Name, envelope.m_Filter))
                {
                    continue;
                }
                auto canonical = [&](const std::string &unit) -> std::string
                {
                    const Semantics::Concept *entry = ontology && !unit.empty() ? ontology->Lookup(unit) : nullptr;
                    return entry && entry->m_Kind == "unit" ? entry->m_Id : Lower(unit);
                };
                if (!envelope.m_Unit.empty() && !actionable.m_Unit.empty() &&
                    canonical(envelope.m_Unit) != canonical(actionable.m_Unit))
                {
                    review.Add("safety", "error",
                               "'" + actionable.m_Name + "' is declared in '" + actionable.m_Unit +
                                   "' but its safety envelope is in '" + envelope.m_Unit + "'",
                               actionable.m_Name, name);
                    continue;
                }
                std::optional<double> low = actionable.m_Min;
                std::optional<double> high = actionable.m_Max;
                if (envelope.m_Min && (!low || *low < *envelope.m_Min))
                {
                    low = envelope.m_Min;
                }
                if (envelope.m_Max && (!high || *high > *envelope.m_Max))
                {
                    high = envelope.m_Max;
                }
                if (low && high && *low > *high)
                {
                    review.Add("safety", "error",
                               "bounds " + Range(actionable.m_Min, actionable.m_Max) + " of '" + actionable.m_Name +
                                   "' do not intersect the safety envelope " + Range(envelope.m_Min, envelope.m_Max) +
                                   " of '" + envelope.m_Filter + "'",
                               actionable.m_Name, name);
                    continue;
                }
                if (low != actionable.m_Min || high != actionable.m_Max)
                {
                    std::string detail = Range(actionable.m_Min, actionable.m_Max) + " → " + Range(low, high);
                    review.Add("safety", "warning",
                               "bounds of '" + actionable.m_Name + "' clamped to the safety envelope of '" +
                                   envelope.m_Filter + "': " + detail,
                               actionable.m_Name, name);
                    actionable.m_Min = low;
                    actionable.m_Max = high;
                    review.Resolve("clamp", actionable.m_Name, detail);
                }
            }
        }

        // tenant isolation
        const std::string &tenant = candidate.m_Constraints.m_Security.m_Tenant;
        const PolicySet::Tenant *own = nullptr;
        for (const PolicySet::Tenant &entry : m_Policies.m_Tenants)
        {
            if (entry.m_Name == tenant)
            {
                own = &entry;
            }
        }
        if (!tenant.empty() && !m_Policies.m_Tenants.empty() && !own)
        {
            review.Add("tenant", "error", "tenant '" + tenant + "' is not declared by policy", "", name);
        }
        for (const std::string &topic : topics)
        {
            if (own && !FilterCovers(own->m_Root + "/#", topic))
            {
                review.Add("tenant", "error",
                           "'" + topic + "' is outside the root '" + own->m_Root + "/' of tenant '" + tenant + "'",
                           topic, name);
            }
            for (const PolicySet::Tenant &entry : m_Policies.m_Tenants)
            {
                if (&entry != own && FiltersOverlap(topic, entry.m_Root + "/#"))
                {
                    review.Add("tenant", "error",
                               "'" + topic + "' reaches into the root '" + entry.m_Root + "/' of tenant '" +
                                   entry.m_Name + "'",
                               topic, name);
                }
            }
        }
        if (!tenant.empty() && m_Policies.m_RequireEncryptionForTenants &&
            !candidate.m_Constraints.m_Security.m_Encryption)
        {
            review.Add("tenant", "error", "tenant service '" + name + "' must enable security.encryption", "", name);
        }

        // declared deployment intent: every actionable must be motivated by the intent text
        if (!candidate.m_Metadata.m_Intent.empty())
        {
            std::vector<std::string> tokens = Tokenize(candidate.m_Metadata.m_Intent);
            std::set<std::string> words;
            for (const std::string &token : tokens)
            {
                words.insert(token);
                words.insert(Semantics::Normalize(token));
            }
            std::set<std::string> mentioned;
            if (ontology)
            {
                for (const Semantics::ConceptMatch &match : ontology->Match(tokens))
                {
                    mentioned.insert(match.m_Concept->m_Id);
                }
            }
            for (const Descriptor::Actionable &actionable : candidate.m_Actionables)
            {
                bool justified = false;
                std::vector<std::string> levels = Split(actionable.m_Name, '/');
                for (size_t i = 0; i < levels.size() && !justified; ++i)
                {
                    const std::string &level = levels[i];
                    if (level.empty() || level == "+" || level == "#" || (i == 0 && levels.size() > 1))
                    {
                        continue; // the domain root alone does not justify an action domain
                    }
                    justified = words.contains(Lower(level)) || words.contains(Semantics::Normalize(level));
                    if (!justified && ontology)
                    {
                        const Semantics::Concept *entry = ontology->Lookup(level);
                        justified = entry && mentioned.contains(entry->m_Id);
                    }
                }
                if (!justified && ontology)
                {
                    // the actuator whose topic suffix this is, or the quantity it drives
                    std::string suffix = levels.size() > 1 ? actionable.m_Name.substr(levels.front().size() + 1) : "";
                    for (const Semantics::Concept *actuator : ontology->OfKind("actuator"))
                    {
                        const Json *topic =
                            actuator->m_Attributes.IsObject() ? actuator->m_Attributes.Find("topic") : nullptr;
                        if (!topic || !topic->IsString() || topic->AsString() != suffix)
                        {
                            continue;
                        }
                        justified = mentioned.contains(actuator->m_Id);
                        for (const Semantics::Concept *quantity : ontology->Related(actuator->m_Id, "actuates"))
                        {
                            justified = justified || mentioned.contains(quantity->m_Id);
                        }
                        if (justified)
                        {
                            break;
                        }
                    }
                }
                if (!justified)
                {
                    review.Add("intent", "warning",
                               "actions on '" + actionable.m_Name + "' are not motivated by the declared intent",
                               actionable.m_Name, name);
                    review.m_Escalate = true;
                    review.m_Questions.push_back("Service '" + name + "' declares actions on '" + actionable.m_Name +
                                                 "', which its intent \"" + candidate.m_Metadata.m_Intent +
                                                 "\" does not mention. Approve this action domain?");
                }
            }
        }

        // conflicts on actionable topics with deployed services (oscillation risk)
        for (const Descriptor::RichMLService &other : deployed)
        {
            const std::string &otherName = other.m_Metadata.m_Name;
            if (otherName == name)
            {
                continue; // a redeploy replaces the running version
            }
            std::vector<std::string> overlapping;
            std::vector<std::string> pairs;
            for (const Descriptor::Actionable &mine : candidate.m_Actionables)
            {
                for (const Descriptor::Actionable &theirs : other.m_Actionables)
                {
                    if (FiltersOverlap(mine.m_Name, theirs.m_Name))
                    {
                        Detail::AddUnique(overlapping, mine.m_Name);
                        Detail::AddUnique(pairs, mine.m_Name == theirs.m_Name ? mine.m_Name
                                                                              : mine.m_Name + " ~ " + theirs.m_Name);
                    }
                }
            }
            if (overlapping.empty())
            {
                continue;
            }
            // objectives the ontology declares as conflicting raise the severity
            std::vector<std::string> clashes;
            if (ontology)
            {
                for (const std::string &objective : candidate.m_Constraints.m_Objectives)
                {
                    for (const std::string &theirs : other.m_Constraints.m_Objectives)
                    {
                        const Semantics::Concept *a = ontology->Lookup(objective);
                        const Semantics::Concept *b = ontology->Lookup(theirs);
                        if (!a || !b)
                        {
                            continue;
                        }
                        auto conflicts = [&](const Semantics::Concept *from, const Semantics::Concept *to)
                        {
                            std::vector<const Semantics::Concept *> related =
                                ontology->Related(from->m_Id, "conflictsWith");
                            return std::find(related.begin(), related.end(), to) != related.end();
                        };
                        if (conflicts(a, b) || conflicts(b, a))
                        {
                            Detail::AddUnique(clashes, a->m_Id + " vs " + b->m_Id);
                        }
                    }
                }
            }
            int32_t mine = candidate.m_Constraints.m_Priority;
            int32_t theirs = other.m_Constraints.m_Priority;
            std::string summary = "'" + name + "' (priority " + std::to_string(mine) + ", objectives " +
                                  ObjectiveText(candidate.m_Constraints.m_Objectives) + ") and deployed '" + otherName +
                                  "' (priority " + std::to_string(theirs) + ", objectives " +
                                  ObjectiveText(other.m_Constraints.m_Objectives) + ") both act on " +
                                  Join(pairs, ", ") +
                                  (clashes.empty() ? std::string() : "; conflicting objectives " + Join(clashes, ", "));
            if (mine < theirs)
            {
                std::vector<Descriptor::Actionable> &actionables = review.m_Amended.m_Actionables;
                size_t before = actionables.size();
                std::erase_if(actionables, [&](const Descriptor::Actionable &actionable)
                              { return Detail::Contains(overlapping, actionable.m_Name); });
                if (actionables.empty() || actionables.size() == before)
                {
                    review.Add("conflict", "error",
                               summary + "; the lower-priority candidate has no other actionable to keep",
                               overlapping.front(), otherName);
                    continue;
                }
                review.Add("conflict", clashes.empty() ? "info" : "warning",
                           summary + "; dropped from the lower-priority candidate", overlapping.front(), otherName);
                for (const std::string &topic : overlapping)
                {
                    review.Resolve("drop", topic,
                                   "yields to '" + otherName + "' (priority " + std::to_string(theirs) + ")");
                }
                if (review.m_Amended.m_Inference &&
                    Detail::Contains(overlapping, review.m_Amended.m_Inference->m_Actuation))
                {
                    review.m_Amended.m_Inference->m_Actuation.clear();
                }
                continue;
            }
            review.Add("conflict", "warning",
                       summary + (mine == theirs ? "; equal priorities" : "; the candidate has the higher priority"),
                       overlapping.front(), otherName);
            review.m_Escalate = true;
            review.m_Questions.push_back(
                "Service " + summary + ". Deploying both risks oscillating actuation" +
                (mine == theirs ? " and neither has priority" : "; '" + otherName + "' would have to yield") +
                ". Approve deploying '" + name + "' alongside '" + otherName + "'?");
        }

        // an amendment must leave a valid descriptor (the harness Guard re-validates it as well)
        if (review.m_Changed && !review.m_Error)
        {
            for (const Descriptor::Issue &issue : Descriptor::Validate(review.m_Amended))
            {
                if (issue.m_Severity == Descriptor::Severity::Error)
                {
                    review.Add("policy", "error",
                               "the amended descriptor is invalid: " + issue.m_Path + ": " + issue.m_Message, "", name);
                }
            }
        }

        ComplianceReport report = std::move(review.m_Report);
        if (review.m_Error)
        {
            report.m_Verdict = Verdict::Rejected;
            return report;
        }
        if (review.m_Changed)
        {
            report.m_Amended = std::move(review.m_Amended);
        }
        if (!review.m_Escalate)
        {
            report.m_Verdict = review.m_Changed ? Verdict::Amended : Verdict::Approved;
            return report;
        }
        report.m_Verdict = Verdict::Escalate;
        std::string question = Join(review.m_Questions, "\n");
        // optional Generation provider phrases the operator question; the deterministic text is the fallback
        Intelligence::Provider *writer =
            m_Environment.m_Providers ? m_Environment.m_Providers->Get(Intelligence::Capability::Generation) : nullptr;
        if (writer)
        {
            Json findings = Json::MakeArray();
            for (const Finding &finding : report.m_Findings)
            {
                findings.Push(finding.ToJson());
            }
            using namespace Intelligence;
            Json schema = SchemaObject({{"question", SchemaScalar("string", "one concise approval question")}});
            std::optional<Json> answer =
                AskJson(writer, Capability::Generation,
                        "You write approval questions for human operators of an MLOps platform. Name the services, "
                        "topics and objectives involved and ask for an approve/reject decision. Be concise.",
                        "Findings: " + findings.Dump() + "\nDraft question: " + question, schema);
            std::string text = answer ? Trim(answer->GetString("question")) : std::string();
            if (!text.empty() && text.size() <= MAX_QUESTION_BYTES && text.find(name) != std::string::npos)
            {
                question = text;
            }
        }
        if (question.size() > MAX_QUESTION_BYTES)
        {
            question.resize(MAX_QUESTION_BYTES);
        }
        report.m_Question = std::move(question);
        return report;
    }
} // namespace eOps::Agents
