#include <eOps/Agents/Executive.h>
#include <eOps/Descriptor/Cron.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/Stdout.h>
#include "Detail.h"
#include <set>
#include <cmath>
#include <algorithm>

namespace eOps::Agents
{
    namespace
    {
        using Semantics::Concept;
        using Semantics::ConceptMatch;
        using Semantics::Ontology;

        using WordSet = std::set<std::string, std::less<>>;

        // intents are operator sentences; longer input is parsed only up to this bound
        constexpr size_t MAX_INTENT_BYTES = 64 * 1024;

        // grammar, filler and MLOps vocabulary the parser consumes or ignores (never "unknown")
        const WordSet &FillerWords()
        {
            static const WordSet s_Words = {"a",
                                            "an",
                                            "the",
                                            "this",
                                            "that",
                                            "these",
                                            "those",
                                            "which",
                                            "who",
                                            "whom",
                                            "whose",
                                            "what",
                                            "where",
                                            "when",
                                            "while",
                                            "it",
                                            "its",
                                            "is",
                                            "are",
                                            "was",
                                            "were",
                                            "be",
                                            "been",
                                            "being",
                                            "am",
                                            "do",
                                            "does",
                                            "did",
                                            "has",
                                            "have",
                                            "had",
                                            "will",
                                            "would",
                                            "shall",
                                            "should",
                                            "can",
                                            "could",
                                            "may",
                                            "might",
                                            "must",
                                            "please",
                                            "i",
                                            "we",
                                            "you",
                                            "they",
                                            "me",
                                            "us",
                                            "our",
                                            "my",
                                            "your",
                                            "their",
                                            "them",
                                            "deploy",
                                            "deploys",
                                            "deployed",
                                            "deploying",
                                            "deployment",
                                            "create",
                                            "creates",
                                            "build",
                                            "builds",
                                            "make",
                                            "makes",
                                            "run",
                                            "runs",
                                            "running",
                                            "start",
                                            "launch",
                                            "provide",
                                            "provides",
                                            "setup",
                                            "up",
                                            "service",
                                            "services",
                                            "model",
                                            "models",
                                            "ml",
                                            "machine",
                                            "learning",
                                            "ai",
                                            "application",
                                            "app",
                                            "system",
                                            "workload",
                                            "agent",
                                            "pipeline",
                                            "sensor",
                                            "sensors",
                                            "sensing",
                                            "measurement",
                                            "measurements",
                                            "reading",
                                            "readings",
                                            "data",
                                            "telemetry",
                                            "signal",
                                            "signals",
                                            "value",
                                            "values",
                                            "using",
                                            "use",
                                            "uses",
                                            "used",
                                            "via",
                                            "by",
                                            "with",
                                            "without",
                                            "for",
                                            "from",
                                            "to",
                                            "of",
                                            "in",
                                            "on",
                                            "at",
                                            "into",
                                            "onto",
                                            "over",
                                            "under",
                                            "below",
                                            "above",
                                            "beneath",
                                            "between",
                                            "and",
                                            "or",
                                            "but",
                                            "nor",
                                            "also",
                                            "plus",
                                            "as",
                                            "than",
                                            "then",
                                            "so",
                                            "such",
                                            "per",
                                            "each",
                                            "every",
                                            "all",
                                            "any",
                                            "some",
                                            "both",
                                            "within",
                                            "across",
                                            "near",
                                            "around",
                                            "about",
                                            "after",
                                            "before",
                                            "during",
                                            "based",
                                            "through",
                                            "new",
                                            "need",
                                            "needs",
                                            "want",
                                            "wants",
                                            "accuracy",
                                            "accurate",
                                            "latency",
                                            "latencies",
                                            "delay",
                                            "response",
                                            "reaction",
                                            "deadline",
                                            "time",
                                            "percentile",
                                            "quantile",
                                            "target",
                                            "objective",
                                            "goal",
                                            "threshold",
                                            "less",
                                            "more",
                                            "most",
                                            "least",
                                            "lower",
                                            "higher",
                                            "greater",
                                            "fewer",
                                            "max",
                                            "maximum",
                                            "min",
                                            "minimum",
                                            "range",
                                            "low",
                                            "high",
                                            "real",
                                            "realtime",
                                            "inference",
                                            "schedule",
                                            "scheduled",
                                            "periodically",
                                            "regularly",
                                            "automatically",
                                            "continuously",
                                            "online",
                                            "device",
                                            "devices",
                                            "iot",
                                            "not",
                                            "never",
                                            "avoid",
                                            "avoiding",
                                            "except",
                                            "excluding",
                                            "no",
                                            "exceeding",
                                            "th",
                                            "median",
                                            "train",
                                            "trained",
                                            "training",
                                            "trains",
                                            "retrain",
                                            "retrained",
                                            "retraining",
                                            "retrains",
                                            "refit",
                                            "relearn",
                                            "level",
                                            "levels",
                                            "placement",
                                            "place",
                                            "placed",
                                            "hosted",
                                            "host",
                                            "located",
                                            "keeping",
                                            "ensure",
                                            "ensuring",
                                            "guarantee",
                                            "day",
                                            "days",
                                            "week",
                                            "weeks",
                                            "h",
                                            "s",
                                            "m"};
            return s_Words;
        }

        bool TrainingWord(std::string_view token)
        {
            static const WordSet s_Words = {"train",     "trained",    "training", "trains",  "retrain",
                                            "retrained", "retraining", "retrains", "refit",   "relearn",
                                            "learn",     "learns",     "learned",  "learning"};
            return s_Words.contains(token);
        }

        bool NegationWord(std::string_view token)
        {
            static const WordSet s_Words = {"not",    "never",     "avoid",   "avoiding",
                                            "except", "excluding", "without", "no"};
            return s_Words.contains(token);
        }

        bool LatencyWord(std::string_view token)
        {
            static const WordSet s_Words = {"latency", "latencies", "delay", "response", "reaction", "deadline"};
            return s_Words.contains(token);
        }

        std::string StringAttribute(const Concept *entry, std::string_view key)
        {
            if (!entry || !entry->m_Attributes.IsObject())
            {
                return {};
            }
            const Json *value = entry->m_Attributes.Find(key);
            return value && value->IsString() ? value->AsString() : std::string();
        }

        std::optional<double> NumberAttribute(const Concept *entry, std::string_view key)
        {
            if (!entry || !entry->m_Attributes.IsObject())
            {
                return std::nullopt;
            }
            const Json *value = entry->m_Attributes.Find(key);
            if (!value || !value->IsNumber() || !std::isfinite(value->AsNumber()))
            {
                return std::nullopt;
            }
            return value->AsNumber();
        }

        bool BoolAttribute(const Concept *entry, std::string_view key, bool fallback)
        {
            if (!entry || !entry->m_Attributes.IsObject())
            {
                return fallback;
            }
            const Json *value = entry->m_Attributes.Find(key);
            return value && value->IsBool() ? value->AsBool() : fallback;
        }

        bool IsKind(const Concept *entry, std::string_view kind)
        {
            return entry && entry->m_Kind == kind;
        }

        // seconds per unit for time units ("ms" 0.001, "minute" 60), else nullopt
        std::optional<double> TimeUnitSeconds(const Concept *unit)
        {
            if (!IsKind(unit, "unit") || StringAttribute(unit, "dimension") != "time")
            {
                return std::nullopt;
            }
            return NumberAttribute(unit, "seconds");
        }

        // DNS label: lowercase alphanumerics with single '-' separators, ≤ 63 characters
        std::string DnsName(std::string_view text)
        {
            std::string name;
            for (char c : Lower(text))
            {
                bool alnum = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
                if (alnum)
                {
                    name.push_back(c);
                }
                else if (!name.empty() && name.back() != '-')
                {
                    name.push_back('-');
                }
            }
            if (name.size() > 63)
            {
                name.resize(63);
            }
            while (!name.empty() && name.back() == '-')
            {
                name.pop_back();
            }
            return name.empty() ? std::string("service") : name;
        }

        // topic level text from a concept id ("air-quality" stays, spaces become '-')
        std::string TopicWord(std::string_view text)
        {
            std::string word = DnsName(text);
            return word == "service" && Lower(text) != "service" ? std::string() : word;
        }

        std::string CollapseSpaces(std::string_view text)
        {
            std::string result;
            bool space = false;
            for (char c : Trim(text))
            {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')
                {
                    space = true;
                    continue;
                }
                if (space && !result.empty())
                {
                    result.push_back(' ');
                }
                space = false;
                result.push_back(c);
            }
            return result;
        }

        bool ValidCron(const std::string &expression)
        {
            try
            {
                (void)Descriptor::CronSchedule::Parse(expression);
                return true;
            }
            catch (const std::exception &)
            {
                return false;
            }
        }

        // pattern archetype defaults (ontology attributes override the built-in table)
        struct PatternDefaults
        {
            double m_LatencyMs = 100;
            std::string m_Statistic = "p95";
            std::string m_Suffix = "forecast";
            bool m_Actionable = false;
        };

        PatternDefaults DefaultsOf(const Ontology *ontology, const std::string &pattern)
        {
            PatternDefaults defaults;
            if (pattern == "controller")
            {
                defaults = PatternDefaults{5, "p95", "control", true};
            }
            else if (pattern == "detector")
            {
                defaults = PatternDefaults{50, "p95", "detector", false};
            }
            else if (pattern != "predictor" && !pattern.empty())
            {
                defaults.m_Suffix = TopicWord(pattern);
            }
            const Concept *entry = ontology ? ontology->Find(pattern) : nullptr;
            if (IsKind(entry, "pattern"))
            {
                if (std::optional<double> latency = NumberAttribute(entry, "latencyMs"); latency && *latency > 0)
                {
                    defaults.m_LatencyMs = *latency;
                }
                if (std::string statistic = StringAttribute(entry, "statistic"); !statistic.empty())
                {
                    defaults.m_Statistic = statistic;
                }
                if (std::string suffix = StringAttribute(entry, "suffix"); IsDnsLabel(suffix))
                {
                    defaults.m_Suffix = suffix;
                }
                defaults.m_Actionable = BoolAttribute(entry, "actionable", defaults.m_Actionable);
            }
            if (defaults.m_Suffix.empty())
            {
                defaults.m_Suffix = "service";
            }
            return defaults;
        }

        std::optional<double> ConvertUnit(double value, const std::string &from, const std::string &to)
        {
            auto same = [](const std::string &a, const std::string &b)
            { return a == b || (a == "celcius" && b == "celsius") || (a == "celsius" && b == "celcius"); };
            if (from.empty() || to.empty() || same(from, to))
            {
                return value;
            }
            if (from == "fahrenheit" && same(to, "celsius"))
            {
                return (value - 32) * 5 / 9;
            }
            if (same(from, "celsius") && to == "fahrenheit")
            {
                return value * 9 / 5 + 32;
            }
            return std::nullopt;
        }

        // cron for "every N <unit>" (seconds per unit); nullopt when not expressible
        std::optional<std::string> EveryCron(double count, double unitSeconds)
        {
            if (count < 1 || count != std::floor(count) || count > 1e6)
            {
                return std::nullopt;
            }
            uint64_t n = static_cast<uint64_t>(count);
            uint64_t seconds = static_cast<uint64_t>(unitSeconds) * n;
            if (unitSeconds < 1 || seconds == 0)
            {
                return std::nullopt;
            }
            std::string cron;
            if (seconds % 604800 == 0 && seconds / 604800 == 1)
            {
                cron = "0 0 * * 0";
            }
            else if (seconds % 86400 == 0)
            {
                uint64_t days = seconds / 86400;
                cron = days == 1 ? "0 0 * * *" : days <= 31 ? "0 0 */" + std::to_string(days) + " * *" : "";
            }
            else if (seconds % 3600 == 0 && 24 % (seconds / 3600) == 0)
            {
                uint64_t hours = seconds / 3600;
                cron = hours == 1 ? "0 * * * *" : "0 */" + std::to_string(hours) + " * * *";
            }
            else if (seconds % 60 == 0 && 60 % (seconds / 60) == 0)
            {
                uint64_t minutes = seconds / 60;
                cron = minutes == 1 ? "* * * * *" : "*/" + std::to_string(minutes) + " * * * *";
            }
            else if (seconds % 3600 == 0)
            {
                cron = "@every " + std::to_string(seconds / 3600) + "h";
            }
            else if (seconds % 60 == 0)
            {
                cron = "@every " + std::to_string(seconds / 60) + "m";
            }
            else
            {
                cron = "@every " + std::to_string(seconds) + "s";
            }
            if (cron.empty() || !ValidCron(cron))
            {
                return std::nullopt;
            }
            return cron;
        }

        // deterministic, ontology-grounded reading of one intent
        class IntentReader
        {
        public:
            IntentReader(const Ontology *ontology, std::string_view intent)
                : m_Ontology(ontology), m_Tokens(Tokenize(intent)), m_Concepts(m_Tokens.size(), nullptr),
                  m_MatchAt(m_Tokens.size(), -1), m_Used(m_Tokens.size(), false)
            {
                if (m_Ontology)
                {
                    m_Matches = m_Ontology->Match(m_Tokens);
                }
                for (size_t i = 0; i < m_Matches.size(); ++i)
                {
                    const ConceptMatch &match = m_Matches[i];
                    m_MatchAt[match.m_Token] = static_cast<int32_t>(i);
                    for (size_t t = match.m_Token; t < match.m_Token + match.m_Length && t < m_Tokens.size(); ++t)
                    {
                        m_Concepts[t] = match.m_Concept;
                        m_Used[t] = true;
                    }
                }
            }

            SemanticFrame Read()
            {
                if (!m_Ontology)
                {
                    m_Frame.m_Notes.push_back("no ontology: intent words cannot be grounded");
                }
                Domain();
                Pattern();
                Quantities();
                Actionables();
                Accuracy();
                Bounds();
                Latency();
                Statistic();
                Schedule();
                Regions();
                Accelerator();
                Objectives();
                Unknown();
                m_Frame.m_Name = DnsName(m_Frame.m_Domain + "-" + m_Defaults.m_Suffix);
                m_Frame.m_Source = "deterministic";
                return std::move(m_Frame);
            }

        private:
            bool Word(size_t index, std::string_view word) const
            {
                return index < m_Tokens.size() && m_Tokens[index] == word;
            }

            // free number token (not part of a concept phrase)
            std::optional<double> Number(size_t index) const
            {
                if (index >= m_Tokens.size() || m_Concepts[index])
                {
                    return std::nullopt;
                }
                return ParseNumber(m_Tokens[index]);
            }

            // unit concept match starting at `index`
            const ConceptMatch *UnitAt(size_t index) const
            {
                if (index >= m_Tokens.size() || m_MatchAt[index] < 0)
                {
                    return nullptr;
                }
                const ConceptMatch &match = m_Matches[static_cast<size_t>(m_MatchAt[index])];
                return IsKind(match.m_Concept, "unit") ? &match : nullptr;
            }

            void Use(size_t from, size_t count)
            {
                for (size_t i = from; i < from + count && i < m_Used.size(); ++i)
                {
                    m_Used[i] = true;
                }
            }

            std::vector<const ConceptMatch *> OfKind(std::string_view kind) const
            {
                std::vector<const ConceptMatch *> found;
                for (const ConceptMatch &match : m_Matches)
                {
                    if (IsKind(match.m_Concept, kind))
                    {
                        found.push_back(&match);
                    }
                }
                return found;
            }

            void Domain()
            {
                const Concept *domain = nullptr;
                if (std::vector<const ConceptMatch *> domains = OfKind("domain"); !domains.empty())
                {
                    domain = domains.front()->m_Concept;
                }
                // a domain word swallowed by a longer phrase ("vehicle speed")
                for (size_t i = 0; !domain && m_Ontology && i < m_Tokens.size(); ++i)
                {
                    const Concept *entry = m_Ontology->Lookup(m_Tokens[i]);
                    if (IsKind(entry, "domain"))
                    {
                        domain = entry;
                        m_Frame.m_Notes.push_back("domain '" + entry->m_Id + "' read from the word '" + m_Tokens[i] +
                                                  "' inside a longer phrase");
                    }
                }
                if (!domain && m_Ontology)
                {
                    // the single domain containing every grounded quantity
                    std::vector<const Concept *> owners;
                    for (const ConceptMatch *quantity : OfKind("quantity"))
                    {
                        std::vector<const Concept *> containing =
                            m_Ontology->RelatedFrom(quantity->m_Concept->m_Id, "contains");
                        if (owners.empty())
                        {
                            owners = containing;
                            continue;
                        }
                        std::erase_if(
                            owners, [&](const Concept *owner)
                            { return std::find(containing.begin(), containing.end(), owner) == containing.end(); });
                    }
                    if (owners.size() == 1)
                    {
                        domain = owners.front();
                        m_Frame.m_Notes.push_back("domain '" + domain->m_Id +
                                                  "' inferred: the only domain containing the named quantities");
                    }
                }
                if (domain)
                {
                    m_Frame.m_Domain = domain->m_Id;
                    m_Root = StringAttribute(domain, "root");
                    if (m_Root.empty() || !ValidTopic(m_Root, false) || IsFilter(m_Root))
                    {
                        m_Root = TopicWord(domain->m_Id);
                    }
                }
                else
                {
                    m_Frame.m_Domain = "app";
                    m_Root = "app";
                    m_Frame.m_Notes.push_back("no domain grounded: topics use the root 'app'");
                }
            }

            void Pattern()
            {
                std::vector<const ConceptMatch *> patterns = OfKind("pattern");
                const Concept *chosen = nullptr;
                for (const ConceptMatch *match : patterns)
                {
                    if (DefaultsOf(m_Ontology, match->m_Concept->m_Id).m_Actionable)
                    {
                        chosen = match->m_Concept;
                        break;
                    }
                }
                if (!chosen && !patterns.empty())
                {
                    chosen = patterns.front()->m_Concept;
                }
                if (chosen)
                {
                    m_Frame.m_Pattern = chosen->m_Id;
                }
                else
                {
                    m_Frame.m_Pattern = OfKind("actuator").empty() ? "predictor" : "controller";
                    m_Frame.m_Notes.push_back("no service pattern named: defaulting to " + m_Frame.m_Pattern);
                }
                m_Defaults = DefaultsOf(m_Ontology, m_Frame.m_Pattern);
            }

            void AddObservable(const Concept *quantity)
            {
                if (std::find(m_Quantities.begin(), m_Quantities.end(), quantity) != m_Quantities.end())
                {
                    return;
                }
                std::string topic = StringAttribute(quantity, "topic");
                if (topic.empty())
                {
                    topic = TopicWord(quantity->m_Id);
                }
                std::string name = m_Root + "/" + topic;
                if (!ValidTopic(name, false) || IsFilter(name))
                {
                    m_Frame.m_Notes.push_back("quantity '" + quantity->m_Id + "' has no usable topic");
                    return;
                }
                m_Quantities.push_back(quantity);
                Descriptor::Observable observable;
                observable.m_Name = name;
                observable.m_Unit = StringAttribute(quantity, "unit");
                m_Frame.m_Observables.push_back(std::move(observable));
            }

            void Quantities()
            {
                for (const ConceptMatch *match : OfKind("quantity"))
                {
                    AddObservable(match->m_Concept);
                }
            }

            // the quantity an actuator drives (attribute, else the actuates relation)
            const Concept *ActuatedQuantity(const Concept *actuator) const
            {
                if (!m_Ontology)
                {
                    return nullptr;
                }
                const Concept *quantity = m_Ontology->Find(StringAttribute(actuator, "quantity"));
                if (!IsKind(quantity, "quantity"))
                {
                    std::vector<const Concept *> related = m_Ontology->Related(actuator->m_Id, "actuates");
                    quantity = related.empty() ? nullptr : related.front();
                }
                return IsKind(quantity, "quantity") ? quantity : nullptr;
            }

            void AddActionable(const Concept *source, const Concept *quantity, std::string topic, std::string unit,
                               std::optional<double> min, std::optional<double> max, std::string description)
            {
                std::string name = m_Root + "/" + topic;
                if (!ValidTopic(name, false) || IsFilter(name))
                {
                    m_Frame.m_Notes.push_back("'" + source->m_Id + "' has no usable actionable topic");
                    return;
                }
                for (const Descriptor::Actionable &existing : m_Frame.m_Actionables)
                {
                    if (existing.m_Name == name)
                    {
                        return;
                    }
                }
                Descriptor::Actionable actionable;
                actionable.m_Name = std::move(name);
                actionable.m_Unit = std::move(unit);
                actionable.m_Min = min;
                actionable.m_Max = max;
                actionable.m_Description = std::move(description);
                m_Frame.m_Actionables.push_back(std::move(actionable));
                m_ActionSources.emplace_back(source, quantity);
            }

            void AddActuator(const Concept *actuator)
            {
                std::string topic = StringAttribute(actuator, "topic");
                if (topic.empty())
                {
                    topic = "set/" + TopicWord(actuator->m_Id);
                }
                const Concept *quantity = ActuatedQuantity(actuator);
                AddActionable(actuator, quantity, topic, StringAttribute(actuator, "unit"),
                              NumberAttribute(actuator, "min"), NumberAttribute(actuator, "max"), {});
                if (quantity && std::find(m_Quantities.begin(), m_Quantities.end(), quantity) == m_Quantities.end())
                {
                    AddObservable(quantity);
                    m_Frame.m_Notes.push_back("observing '" + quantity->m_Id + "' as feedback for actuator '" +
                                              actuator->m_Id + "'");
                }
            }

            void Actionables()
            {
                for (const ConceptMatch *match : OfKind("actuator"))
                {
                    AddActuator(match->m_Concept);
                }
                if (!m_Frame.m_Actionables.empty())
                {
                    return;
                }
                if (m_Quantities.empty())
                {
                    m_Frame.m_Notes.push_back("no quantity grounded: no observable or actionable can be derived");
                    return;
                }
                const Concept *primary = m_Quantities.front();
                if (m_Defaults.m_Actionable)
                {
                    // a controller drives the setpoint actuator of its primary quantity
                    std::vector<const Concept *> actuators = m_Ontology
                                                                 ? m_Ontology->RelatedFrom(primary->m_Id, "actuates")
                                                                 : std::vector<const Concept *>{};
                    if (actuators.empty())
                    {
                        m_Frame.m_Notes.push_back("no actuator known for '" + primary->m_Id + "'");
                        return;
                    }
                    AddActuator(actuators.front());
                    m_Frame.m_Notes.push_back("setpoint actuator '" + actuators.front()->m_Id + "' derived for '" +
                                              primary->m_Id + "'");
                    return;
                }
                // predictors and detectors publish their output on <root>/<suffix>/<quantity topic>
                std::string topic = StringAttribute(primary, "topic");
                if (topic.empty())
                {
                    topic = TopicWord(primary->m_Id);
                }
                std::string description = m_Frame.m_Pattern == "detector"
                                              ? "expected " + primary->m_Id + " (deviations indicate anomalies)"
                                              : m_Frame.m_Pattern + " output for " + primary->m_Id;
                AddActionable(primary, primary, m_Defaults.m_Suffix + "/" + topic, StringAttribute(primary, "unit"),
                              NumberAttribute(primary, "min"), NumberAttribute(primary, "max"), description);
                m_Frame.m_Notes.push_back("output topic " + m_Root + "/" + m_Defaults.m_Suffix + "/" + topic +
                                          " published by the " + m_Frame.m_Pattern);
            }

            // actionable addressed by a phrase at `index`: nearest preceding quantity/actuator mention
            Descriptor::Actionable *TargetOf(size_t index)
            {
                if (m_Frame.m_Actionables.empty())
                {
                    return nullptr;
                }
                for (size_t i = index; i-- > 0;)
                {
                    const Concept *entry = m_Concepts[i];
                    if (!IsKind(entry, "quantity") && !IsKind(entry, "actuator"))
                    {
                        continue;
                    }
                    for (size_t a = 0; a < m_ActionSources.size(); ++a)
                    {
                        if (m_ActionSources[a].first == entry || m_ActionSources[a].second == entry)
                        {
                            return &m_Frame.m_Actionables[a];
                        }
                    }
                }
                return &m_Frame.m_Actionables.front();
            }

            // value in the actionable's unit, or nullopt when the stated unit is incompatible
            std::optional<double> InUnit(double value, const ConceptMatch *unit, const Descriptor::Actionable &target)
            {
                if (!unit)
                {
                    return value;
                }
                std::optional<double> converted = ConvertUnit(value, unit->m_Concept->m_Id, target.m_Unit);
                if (!converted)
                {
                    m_Frame.m_Notes.push_back("bound in '" + unit->m_Concept->m_Id + "' ignored: " + target.m_Name +
                                              " is measured in '" + target.m_Unit + "'");
                }
                else if (*converted != value)
                {
                    m_Frame.m_Notes.push_back("bound " + Detail::Text(value) + " " + unit->m_Concept->m_Id +
                                              " converted to " + Detail::Text(*converted) + " " + target.m_Unit);
                }
                return converted;
            }

            // comparator phrase starting at `index`: {upper bound?, token count}
            std::optional<std::pair<bool, size_t>> Comparator(size_t index) const
            {
                if (Word(index, "under") || Word(index, "below") || Word(index, "beneath") || Word(index, "within") ||
                    Word(index, "max") || Word(index, "maximum"))
                {
                    return std::pair<bool, size_t>{true, 1};
                }
                if ((Word(index, "less") || Word(index, "lower") || Word(index, "fewer")) && Word(index + 1, "than"))
                {
                    return std::pair<bool, size_t>{true, 2};
                }
                if ((Word(index, "at") && Word(index + 1, "most")) || (Word(index, "up") && Word(index + 1, "to")))
                {
                    return std::pair<bool, size_t>{true, 2};
                }
                if (Word(index, "no") && Word(index + 1, "more") && Word(index + 2, "than"))
                {
                    return std::pair<bool, size_t>{true, 3};
                }
                if (Word(index, "above") || Word(index, "over") || Word(index, "exceeding") || Word(index, "min") ||
                    Word(index, "minimum"))
                {
                    return std::pair<bool, size_t>{false, 1};
                }
                if ((Word(index, "more") || Word(index, "greater") || Word(index, "higher")) && Word(index + 1, "than"))
                {
                    return std::pair<bool, size_t>{false, 2};
                }
                if (Word(index, "at") && Word(index + 1, "least"))
                {
                    return std::pair<bool, size_t>{false, 2};
                }
                return std::nullopt;
            }

            void Bounds()
            {
                for (size_t i = 0; i < m_Tokens.size(); ++i)
                {
                    bool between = Word(i, "between");
                    bool from = Word(i, "from");
                    if (!between && !from)
                    {
                        continue;
                    }
                    std::optional<double> low = Number(i + 1);
                    if (!low)
                    {
                        continue;
                    }
                    size_t at = i + 2;
                    const ConceptMatch *lowUnit = UnitAt(at);
                    if (lowUnit)
                    {
                        at += lowUnit->m_Length;
                    }
                    if (!Word(at, between ? "and" : "to"))
                    {
                        continue;
                    }
                    std::optional<double> high = Number(at + 1);
                    if (!high)
                    {
                        continue;
                    }
                    const ConceptMatch *highUnit = UnitAt(at + 2);
                    const ConceptMatch *unit = highUnit ? highUnit : lowUnit;
                    if (TimeUnitSeconds(unit ? unit->m_Concept : nullptr))
                    {
                        continue; // a time span, not an actionable range
                    }
                    Use(i, at + 2 - i + (highUnit ? highUnit->m_Length : 0));
                    Descriptor::Actionable *target = TargetOf(i);
                    if (!target)
                    {
                        m_Frame.m_Notes.push_back("range " + Detail::Text(*low) + ".." + Detail::Text(*high) +
                                                  " ignored: no actionable to bound");
                        continue;
                    }
                    std::optional<double> lowValue = InUnit(*low, lowUnit ? lowUnit : highUnit, *target);
                    std::optional<double> highValue = InUnit(*high, unit, *target);
                    if (!lowValue || !highValue)
                    {
                        continue;
                    }
                    if (*lowValue > *highValue)
                    {
                        std::swap(*lowValue, *highValue);
                        m_Frame.m_Notes.push_back("range bounds given in descending order were swapped");
                    }
                    target->m_Min = *lowValue;
                    target->m_Max = *highValue;
                    m_Frame.m_Notes.push_back("bounds " + Detail::Text(*lowValue) + ".." + Detail::Text(*highValue) +
                                              " for " + target->m_Name + " taken from the intent");
                }
                // single-sided bounds need an explicit, non-time unit ("below 30 degrees")
                for (size_t i = 0; i < m_Tokens.size(); ++i)
                {
                    std::optional<std::pair<bool, size_t>> comparator = Comparator(i);
                    if (!comparator || m_Used[i])
                    {
                        continue;
                    }
                    size_t at = i + comparator->second;
                    std::optional<double> value = Number(at);
                    const ConceptMatch *unit = UnitAt(at + 1);
                    if (!value || !unit || m_Used[at] || TimeUnitSeconds(unit->m_Concept))
                    {
                        continue;
                    }
                    Descriptor::Actionable *target = TargetOf(i);
                    if (!target)
                    {
                        continue;
                    }
                    if (std::optional<double> converted = InUnit(*value, unit, *target))
                    {
                        Use(i, comparator->second + 1 + unit->m_Length);
                        (comparator->first ? target->m_Max : target->m_Min) = *converted;
                        m_Frame.m_Notes.push_back(std::string(comparator->first ? "upper" : "lower") + " bound " +
                                                  Detail::Text(*converted) + " for " + target->m_Name);
                    }
                }
            }

            void Latency()
            {
                for (size_t n = 0; n < m_Tokens.size() && !m_Frame.m_LatencyMs; ++n)
                {
                    std::optional<double> value = Number(n);
                    const ConceptMatch *unit = UnitAt(n + 1);
                    std::optional<double> seconds = unit ? TimeUnitSeconds(unit->m_Concept) : std::nullopt;
                    if (!value || !seconds || m_Used[n] || (n > 0 && Word(n - 1, "every")))
                    {
                        continue;
                    }
                    bool context = false;
                    size_t comparatorStart = n;
                    for (size_t back = 1; back <= 3 && back <= n; ++back)
                    {
                        if (std::optional<std::pair<bool, size_t>> comparator = Comparator(n - back);
                            comparator && comparator->first && n - back + comparator->second == n)
                        {
                            context = true;
                            comparatorStart = n - back;
                        }
                    }
                    for (size_t i = n >= 4 ? n - 4 : 0; i < std::min(m_Tokens.size(), n + 2 + unit->m_Length + 2); ++i)
                    {
                        context = context || LatencyWord(m_Tokens[i]);
                    }
                    if (!context)
                    {
                        continue;
                    }
                    double ms = *value * *seconds * 1000;
                    if (!(ms > 0) || !std::isfinite(ms))
                    {
                        continue;
                    }
                    Use(comparatorStart, n + 1 + unit->m_Length - comparatorStart);
                    m_Frame.m_LatencyMs = ms;
                }
            }

            void Statistic()
            {
                for (size_t i = 0; i < m_Tokens.size() && m_Frame.m_Statistic.empty(); ++i)
                {
                    const std::string &token = m_Tokens[i];
                    if (token == "p50" || token == "p90" || token == "p95" || token == "p99")
                    {
                        m_Frame.m_Statistic = token;
                        Use(i, 1);
                    }
                    else if (token == "median")
                    {
                        m_Frame.m_Statistic = "p50";
                        Use(i, 1);
                    }
                    else if (std::optional<double> value = Number(i);
                             value && Word(i + 1, "th") && (Word(i + 2, "percentile") || Word(i + 2, "quantile")))
                    {
                        std::string name = "p" + Detail::Text(*value);
                        if (name == "p50" || name == "p90" || name == "p95" || name == "p99")
                        {
                            m_Frame.m_Statistic = name;
                            Use(i, 3);
                        }
                    }
                }
            }

            void SetAccuracy(double value, bool percent, size_t from, size_t count)
            {
                if (percent || value > 1)
                {
                    value /= 100;
                }
                if (!(value > 0) || value > 1)
                {
                    m_Frame.m_Notes.push_back("accuracy " + Detail::Text(value) + " ignored: outside (0, 1]");
                    return;
                }
                m_Frame.m_Accuracy = value;
                Use(from, count);
            }

            void Accuracy()
            {
                for (size_t i = 0; i < m_Tokens.size() && !m_Frame.m_Accuracy; ++i)
                {
                    // "90% accuracy", "90 percent prediction accuracy", "95% accurate"
                    if (std::optional<double> value = Number(i); value && !m_Used[i])
                    {
                        bool percent = Word(i + 1, "percent");
                        size_t after = i + (percent ? 2 : 1);
                        for (size_t j = after; j < std::min(m_Tokens.size(), after + 2); ++j)
                        {
                            if (Word(j, "accuracy") || Word(j, "accurate"))
                            {
                                SetAccuracy(*value, percent, i, j + 1 - i);
                                break;
                            }
                        }
                        if (m_Frame.m_Accuracy)
                        {
                            break;
                        }
                    }
                    // "accuracy of 90%", "accuracy above 0.9", "accuracy of at least 95 percent"
                    if (Word(i, "accuracy") || Word(i, "accurate"))
                    {
                        for (size_t j = i + 1; j < std::min(m_Tokens.size(), i + 5); ++j)
                        {
                            if (std::optional<double> value = Number(j); value && !m_Used[j])
                            {
                                SetAccuracy(*value, Word(j + 1, "percent"), i, j + 1 - i + (Word(j + 1, "percent")));
                                break;
                            }
                        }
                    }
                }
            }

            void Schedule()
            {
                if (std::vector<const ConceptMatch *> schedules = OfKind("schedule"); !schedules.empty())
                {
                    std::string cron = StringAttribute(schedules.front()->m_Concept, "cron");
                    if (ValidCron(cron))
                    {
                        m_Frame.m_Schedule = cron;
                        return;
                    }
                    m_Frame.m_Notes.push_back("schedule '" + schedules.front()->m_Concept->m_Id +
                                              "' has no valid cron expression");
                }
                for (size_t i = 0; i < m_Tokens.size(); ++i)
                {
                    if (!Word(i, "every") || m_Concepts[i])
                    {
                        continue;
                    }
                    size_t at = i + 1;
                    double count = 1;
                    if (std::optional<double> value = Number(at))
                    {
                        count = *value;
                        ++at;
                    }
                    std::optional<double> seconds;
                    size_t length = 1;
                    if (const ConceptMatch *unit = UnitAt(at))
                    {
                        seconds = TimeUnitSeconds(unit->m_Concept);
                        length = unit->m_Length;
                    }
                    else if (Word(at, "day") || Word(at, "days"))
                    {
                        seconds = 86400;
                    }
                    else if (Word(at, "week") || Word(at, "weeks"))
                    {
                        seconds = 604800;
                    }
                    if (!seconds)
                    {
                        continue;
                    }
                    if (std::optional<std::string> cron = EveryCron(count, *seconds))
                    {
                        m_Frame.m_Schedule = *cron;
                        Use(i, at + length - i);
                        return;
                    }
                    m_Frame.m_Notes.push_back("period 'every " + Detail::Text(count) + " " + m_Tokens[at] +
                                              "' cannot be expressed as a schedule");
                }
            }

            void Regions()
            {
                for (const ConceptMatch *match : OfKind("region"))
                {
                    size_t start = match->m_Token;
                    bool negated = false;
                    bool training = false;
                    for (size_t back = 1; back <= 4 && back <= start; ++back)
                    {
                        const std::string &token = m_Tokens[start - back];
                        negated = negated || (back <= 3 && NegationWord(token));
                        training = training || TrainingWord(token);
                    }
                    const std::string &region = match->m_Concept->m_Id;
                    if (negated)
                    {
                        m_Frame.m_Notes.push_back("region '" + region + "' is excluded by the intent");
                        continue;
                    }
                    Detail::AddUnique(training ? m_Frame.m_TrainingRegions : m_Frame.m_InferenceRegions, region);
                }
            }

            void Accelerator()
            {
                if (std::vector<const ConceptMatch *> accelerators = OfKind("accelerator"); !accelerators.empty())
                {
                    m_Frame.m_Accelerator = accelerators.front()->m_Concept->m_Id;
                }
            }

            void Objectives()
            {
                for (const ConceptMatch *match : OfKind("objective"))
                {
                    Detail::AddUnique(m_Frame.m_Objectives, match->m_Concept->m_Id);
                }
            }

            void Unknown()
            {
                const WordSet &filler = FillerWords();
                for (size_t i = 0; i < m_Tokens.size(); ++i)
                {
                    const std::string &token = m_Tokens[i];
                    if (m_Used[i] || token.size() < 2 || filler.contains(token) || ParseNumber(token))
                    {
                        continue;
                    }
                    if (std::none_of(token.begin(), token.end(), [](char c) { return c >= 'a' && c <= 'z'; }))
                    {
                        continue;
                    }
                    Detail::AddUnique(m_Frame.m_Unknown, token);
                }
            }

            const Ontology *m_Ontology = nullptr;
            std::vector<std::string> m_Tokens;
            std::vector<ConceptMatch> m_Matches;
            std::vector<const Concept *> m_Concepts; // concept covering each token
            std::vector<int32_t> m_MatchAt;          // match index starting at each token
            std::vector<bool> m_Used;                // tokens consumed by a concept or a phrase
            std::vector<const Concept *> m_Quantities;
            std::vector<std::pair<const Concept *, const Concept *>> m_ActionSources; // per actionable
            SemanticFrame m_Frame;
            PatternDefaults m_Defaults;
            std::string m_Root;
        };

        // ---- LLM advisory (Reasoning provider): concepts only through Ontology::Extend

        const std::vector<std::string> &AdvisoryKinds()
        {
            static const std::vector<std::string> s_Kinds = {
                "domain", "quantity", "actuator", "region", "objective", "accelerator", "unit", "schedule", "pattern"};
            return s_Kinds;
        }

        Json AdvisorySchema()
        {
            using namespace Intelligence;
            Json text = SchemaNullable(SchemaScalar("string"));
            Json number = SchemaNullable(SchemaScalar("number"));
            Json entry = SchemaObject(
                {{"id", SchemaScalar("string", "lowercase concept id")},
                 {"kind", SchemaEnum(AdvisoryKinds())},
                 {"term", SchemaScalar("string", "the exact intent words this concept grounds")},
                 {"unit", text},
                 {"topic", SchemaNullable(SchemaScalar("string", "topic suffix, e.g. 'co2' or 'set/valve'"))},
                 {"quantity", SchemaNullable(SchemaScalar("string", "quantity id an actuator drives"))},
                 {"min", number},
                 {"max", number},
                 {"cron", text}});
            return SchemaObject({{"concepts", SchemaArray(entry)},
                                 {"latencyMs", number},
                                 {"statistic", SchemaNullable(SchemaEnum({"p50", "p90", "p95", "p99"}))},
                                 {"accuracy", SchemaNullable(SchemaScalar("number", "accuracy threshold in [0, 1]"))},
                                 {"schedule", SchemaNullable(SchemaScalar("string", "cron expression"))},
                                 {"accelerator", text},
                                 {"inferenceRegions", SchemaArray(SchemaScalar("string"))},
                                 {"trainingRegions", SchemaArray(SchemaScalar("string"))},
                                 {"objectives", SchemaArray(SchemaScalar("string"))}});
        }

        // "a b" is a contiguous token run of the intent
        bool Grounded(const std::vector<std::string> &intent, const std::vector<std::string> &term)
        {
            if (term.empty() || term.size() > intent.size())
            {
                return false;
            }
            return std::search(intent.begin(), intent.end(), term.begin(), term.end()) != intent.end();
        }

        std::optional<double> AdviceNumber(const Json &json, std::string_view key)
        {
            const Json *value = json.Find(key);
            if (!value || !value->IsNumber() || !std::isfinite(value->AsNumber()))
            {
                return std::nullopt;
            }
            return value->AsNumber();
        }

        // builds the Extend proposal for one advised concept; nullopt (with reason) when ill-formed
        std::optional<Concept> Proposal(const Json &advice, const Ontology &ontology,
                                        const std::vector<std::string> &intent, std::string &reason)
        {
            std::string id = DnsName(advice.GetString("id"));
            std::string kind = advice.GetString("kind");
            std::string term = Trim(advice.GetString("term"));
            if (advice.GetString("id").empty() || !Detail::Contains(AdvisoryKinds(), kind))
            {
                reason = "missing id or unsupported kind";
                return std::nullopt;
            }
            if (!Grounded(intent, Tokenize(term)))
            {
                reason = "term '" + term + "' does not occur in the intent";
                return std::nullopt;
            }
            Concept entry;
            entry.m_Id = id;
            entry.m_Kind = kind;
            if (Lower(term) != id)
            {
                entry.m_Synonyms.push_back(Lower(term));
            }
            Json attributes = Json::MakeObject();
            std::string unit = advice.GetString("unit");
            const Concept *unitConcept = unit.empty() ? nullptr : ontology.Lookup(unit);
            std::string topic = advice.GetString("topic");
            std::optional<double> min = AdviceNumber(advice, "min");
            std::optional<double> max = AdviceNumber(advice, "max");
            if (kind == "quantity" || kind == "actuator")
            {
                if (topic.empty())
                {
                    topic = kind == "actuator" ? "set/" + TopicWord(id) : TopicWord(id);
                }
                if (IsKind(unitConcept, "unit"))
                {
                    attributes.Set("unit", unitConcept->m_Id);
                    entry.m_Relations.push_back(Semantics::Relation{"measuredIn", unitConcept->m_Id});
                }
                attributes.Set("topic", topic);
                if (min)
                {
                    attributes.Set("min", *min);
                }
                if (max)
                {
                    attributes.Set("max", *max);
                }
                if (kind == "actuator")
                {
                    const Concept *quantity = ontology.Lookup(advice.GetString("quantity"));
                    if (!IsKind(quantity, "quantity"))
                    {
                        reason = "actuator without a known quantity";
                        return std::nullopt;
                    }
                    attributes.Set("quantity", quantity->m_Id);
                    entry.m_Relations.push_back(Semantics::Relation{"actuates", quantity->m_Id});
                }
            }
            else if (kind == "domain")
            {
                attributes.Set("root", topic.empty() ? TopicWord(id) : topic);
            }
            else if (kind == "schedule")
            {
                std::string cron = advice.GetString("cron");
                if (!ValidCron(cron))
                {
                    reason = "schedule without a valid cron expression";
                    return std::nullopt;
                }
                attributes.Set("cron", cron);
            }
            else if (kind == "pattern")
            {
                attributes.Set("actionable", false);
                attributes.Set("suffix", id);
            }
            if (attributes.Size() > 0)
            {
                entry.m_Attributes = std::move(attributes);
            }
            return entry;
        }

        // regions/objectives the ontology knows, in advice order
        std::vector<std::string> KnownIds(const Json &advice, std::string_view key, const Ontology *ontology,
                                          std::string_view kind)
        {
            std::vector<std::string> ids;
            const Json *items = advice.Find(key);
            if (!items || !items->IsArray())
            {
                return ids;
            }
            for (const Json &item : items->AsArray())
            {
                const Concept *entry = item.IsString() && ontology ? ontology->Lookup(item.AsString()) : nullptr;
                if (IsKind(entry, kind))
                {
                    Detail::AddUnique(ids, entry->m_Id);
                }
            }
            return ids;
        }
    } // namespace

    // -------------------------
    // JSON forms
    // -------------------------

    Json PlanStep::ToJson() const
    {
        return Json::MakeObject({{"name", m_Name}, {"agent", m_Agent}, {"task", m_Task}, {"input", m_Input}});
    }

    Json WorkflowPlan::ToJson() const
    {
        Json steps = Json::MakeArray();
        for (const PlanStep &step : m_Steps)
        {
            steps.Push(step.ToJson());
        }
        return Json::MakeObject({{"kind", m_Kind}, {"steps", std::move(steps)}});
    }

    WorkflowPlan WorkflowPlan::FromJson(const Json &json)
    {
        Detail::RequireObject(json, "workflow plan");
        WorkflowPlan plan;
        plan.m_Kind = Detail::ReadString(json, "kind");
        for (const Json &item : Detail::ReadArray(json, "steps"))
        {
            Detail::RequireObject(item, "plan step");
            PlanStep step;
            step.m_Name = Detail::ReadString(item, "name");
            step.m_Agent = Detail::ReadString(item, "agent");
            step.m_Task = Detail::ReadString(item, "task");
            if (const Json *input = item.Find("input"))
            {
                step.m_Input = *input;
            }
            if (step.m_Name.empty())
            {
                throw JsonError("plan step without a name");
            }
            plan.m_Steps.push_back(std::move(step));
        }
        return plan;
    }

    Json SemanticFrame::ToJson() const
    {
        Json observables = Json::MakeArray();
        for (const Descriptor::Observable &observable : m_Observables)
        {
            observables.Push(Detail::ObservableJson(observable));
        }
        Json actionables = Json::MakeArray();
        for (const Descriptor::Actionable &actionable : m_Actionables)
        {
            actionables.Push(Detail::ActionableJson(actionable));
        }
        return Json::MakeObject({{"name", m_Name},
                                 {"domain", m_Domain},
                                 {"pattern", m_Pattern},
                                 {"observables", std::move(observables)},
                                 {"actionables", std::move(actionables)},
                                 {"objectives", Detail::Strings(m_Objectives)},
                                 {"inferenceRegions", Detail::Strings(m_InferenceRegions)},
                                 {"trainingRegions", Detail::Strings(m_TrainingRegions)},
                                 {"latencyMs", Detail::Number(m_LatencyMs)},
                                 {"statistic", m_Statistic},
                                 {"schedule", m_Schedule},
                                 {"accelerator", m_Accelerator},
                                 {"accuracy", Detail::Number(m_Accuracy)},
                                 {"unknown", Detail::Strings(m_Unknown)},
                                 {"notes", Detail::Strings(m_Notes)},
                                 {"source", m_Source}});
    }

    Json Interpretation::ToJson() const
    {
        Json evolved = Json::MakeArray();
        for (const Semantics::Concept &entry : m_Evolved)
        {
            evolved.Push(entry.ToJson());
        }
        return Json::MakeObject({{"frame", m_Frame.ToJson()},
                                 {"descriptor", Descriptor::ToJson(m_Descriptor)},
                                 {"yaml", m_Yaml},
                                 {"issues", Descriptor::ToJson(m_Issues)},
                                 {"evolved", std::move(evolved)},
                                 {"plan", m_Plan.ToJson()},
                                 {"valid", !Descriptor::HasErrors(m_Issues)}});
    }

    // -------------------------
    // Executive
    // -------------------------

    Executive::Executive(AgentEnvironment environment) : m_Environment(environment)
    {
    }

    void Executive::SetValidation(Descriptor::ValidationOptions options)
    {
        m_Validation = std::move(options);
    }

    SemanticFrame Executive::Parse(std::string_view intent) const
    {
        if (intent.size() <= MAX_INTENT_BYTES)
        {
            return IntentReader(m_Environment.m_Ontology, intent).Read();
        }
        SemanticFrame frame = IntentReader(m_Environment.m_Ontology, intent.substr(0, MAX_INTENT_BYTES)).Read();
        frame.m_Notes.push_back("intent truncated to " + std::to_string(MAX_INTENT_BYTES) + " bytes for parsing");
        return frame;
    }

    Descriptor::RichMLService Executive::Generate(const SemanticFrame &frame, std::string_view intent) const
    {
        PatternDefaults defaults = DefaultsOf(m_Environment.m_Ontology, frame.m_Pattern);
        Descriptor::RichMLService service;
        service.m_Metadata.m_Name = IsDnsLabel(frame.m_Name) ? frame.m_Name : DnsName(frame.m_Name);
        service.m_Metadata.m_Intent = CollapseSpaces(intent);
        service.m_Observables = frame.m_Observables;
        service.m_Actionables = frame.m_Actionables;

        Descriptor::InferenceSpec inference;
        inference.m_Resources.m_Cpu = 1;
        inference.m_Resources.m_Memory = uint64_t(1) << 30;
        Descriptor::LatencyObjective latency;
        latency.m_MaxMs = frame.m_LatencyMs.value_or(defaults.m_LatencyMs);
        latency.m_Statistic = frame.m_Statistic.empty() ? defaults.m_Statistic : frame.m_Statistic;
        inference.m_Latency = latency;
        inference.m_Regions = frame.m_InferenceRegions;
        if (inference.m_Regions.empty())
        {
            inference.m_Regions =
                latency.m_MaxMs <= 10 ? std::vector<std::string>{"ran", "edge"} : std::vector<std::string>{"edge"};
        }
        service.m_Inference = std::move(inference);

        Descriptor::TrainingSpec training;
        training.m_Resources.m_Cpu = 2;
        training.m_Resources.m_Memory = uint64_t(4) << 30;
        if (!frame.m_Accelerator.empty())
        {
            training.m_Resources.m_Accelerator = frame.m_Accelerator;
            training.m_Resources.m_AcceleratorCount = 1;
        }
        training.m_Regions = frame.m_TrainingRegions;
        if (training.m_Regions.empty())
        {
            training.m_Regions = {"edge", "cloud"};
        }
        training.m_Schedule = frame.m_Schedule;
        training.m_Dataset = DnsName(frame.m_Domain.empty() ? std::string("app") : frame.m_Domain) + "-telemetry";
        // ground truth for the predicted quantity: <root>/label/<leaf of the output topic>
        std::string output = !service.m_Actionables.empty()   ? service.m_Actionables.front().m_Name
                             : !service.m_Observables.empty() ? service.m_Observables.front().m_Name
                                                              : std::string();
        if (!output.empty() && ValidTopic(output, false) && !IsFilter(output))
        {
            std::vector<std::string> levels = Split(output, '/');
            std::string label = levels.front() + "/label/" + levels.back();
            if (ValidTopic(label, false))
            {
                training.m_Label = label;
            }
        }
        service.m_Training = std::move(training);

        service.m_Constraints.m_Objectives = frame.m_Objectives;
        if (frame.m_Accuracy)
        {
            service.m_Lifecycle.m_AccuracyThreshold = *frame.m_Accuracy;
        }
        return service;
    }

    WorkflowPlan Executive::Decompose(const Descriptor::RichMLService &service) const
    {
        const std::string &name = service.m_Metadata.m_Name;
        WorkflowPlan plan;
        plan.m_Kind = "deploy";
        plan.m_Steps.push_back(PlanStep{"validate", "harness", "", Json::MakeObject({{"service", name}})});
        plan.m_Steps.push_back(PlanStep{"review", "compliance", "review", Json::MakeObject({{"service", name}})});
        uint32_t instances = 0;
        if (service.m_Inference)
        {
            plan.m_Steps.push_back(PlanStep{
                "place-inference", "placement", "place",
                Json::MakeObject(
                    {{"service", name}, {"role", "inference"}, {"replicas", service.m_Inference->m_Replicas}})});
            instances += service.m_Inference->m_Replicas;
        }
        if (service.m_Training)
        {
            plan.m_Steps.push_back(
                PlanStep{"place-training", "placement", "place",
                         Json::MakeObject({{"service", name},
                                           {"role", "training"},
                                           {"replicas", 1},
                                           {"accelerator", service.m_Training->m_Resources.m_Accelerator}})});
            instances += 1;
        }
        plan.m_Steps.push_back(
            PlanStep{"enact", "harness", "",
                     Json::MakeObject(
                         {{"service", name}, {"version", service.m_Metadata.m_Version}, {"instances", instances}})});
        if (service.m_Training && !service.m_Training->m_Schedule.empty())
        {
            plan.m_Steps.push_back(
                PlanStep{"schedule", "harness", "",
                         Json::MakeObject(
                             {{"service", name}, {"job", "training"}, {"schedule", service.m_Training->m_Schedule}})});
        }
        return plan;
    }

    Interpretation Executive::Interpret(std::string_view intent, std::string_view operatorId)
    {
        Interpretation result;
        SemanticFrame frame = Parse(intent);
        std::vector<std::string> notes;

        // optional Reasoning advice: concepts only through Ontology::Extend, numbers only fill gaps
        Intelligence::Provider *provider =
            m_Environment.m_Providers ? m_Environment.m_Providers->Get(Intelligence::Capability::Reasoning) : nullptr;
        if (provider)
        {
            Json context = Json::MakeObject({{"frame", frame.ToJson()}, {"kinds", Detail::Strings(AdvisoryKinds())}});
            std::string system =
                "You advise the Executive agent of an MLOps framework that turns operator intents into ML "
                "service descriptors. A deterministic ontology parser already read the intent. Propose ontology "
                "concepts only for intent words it did not ground ('unknown'), quoting each word exactly as 'term'. "
                "Report numbers only when the intent states them. Use null when unsure.";
            std::string user = "Intent: " + std::string(intent) + "\nDeterministic reading: " + context.Dump();
            std::string error;
            std::optional<Json> advice =
                Intelligence::AskJson(provider, Intelligence::Capability::Reasoning, std::move(system), std::move(user),
                                      AdvisorySchema(), &error);
            if (!advice)
            {
                notes.push_back("reasoning advice unavailable: " + error);
            }
            else
            {
                std::vector<std::string> tokens = Tokenize(intent);
                bool evolved = false;
                for (const Json &item : Detail::ReadArray(*advice, "concepts"))
                {
                    std::string reason;
                    std::optional<Semantics::Concept> proposal =
                        m_Environment.m_Ontology ? Proposal(item, *m_Environment.m_Ontology, tokens, reason)
                                                 : std::nullopt;
                    if (!m_Environment.m_Ontology)
                    {
                        reason = "no ontology to extend";
                    }
                    if (proposal && m_Environment.m_Ontology->Extend(*proposal, &reason))
                    {
                        const Semantics::Concept *stored = m_Environment.m_Ontology->Find(proposal->m_Id);
                        result.m_Evolved.push_back(stored ? *stored : *proposal);
                        notes.push_back("ontology evolved with " + proposal->m_Kind + " '" + proposal->m_Id + "'");
                        evolved = true;
                        continue;
                    }
                    notes.push_back("advised concept '" + item.GetString("id") + "' rejected: " + reason);
                }
                if (evolved)
                {
                    frame = Parse(intent); // ground the intent again with the evolved ontology
                }
                const Semantics::Ontology *ontology = m_Environment.m_Ontology;
                if (std::optional<double> latency = AdviceNumber(*advice, "latencyMs");
                    !frame.m_LatencyMs && latency && *latency > 0)
                {
                    frame.m_LatencyMs = *latency;
                    notes.push_back("latency objective " + Detail::Text(*latency) + " ms from reasoning advice");
                }
                if (std::string statistic = advice->GetString("statistic");
                    frame.m_Statistic.empty() && !statistic.empty())
                {
                    frame.m_Statistic = statistic;
                }
                if (std::optional<double> accuracy = AdviceNumber(*advice, "accuracy");
                    !frame.m_Accuracy && accuracy && *accuracy > 0 && *accuracy <= 1)
                {
                    frame.m_Accuracy = *accuracy;
                    notes.push_back("accuracy threshold " + Detail::Text(*accuracy) + " from reasoning advice");
                }
                if (std::string schedule = advice->GetString("schedule"); frame.m_Schedule.empty() && !schedule.empty())
                {
                    if (ValidCron(schedule))
                    {
                        frame.m_Schedule = schedule;
                        notes.push_back("training schedule '" + schedule + "' from reasoning advice");
                    }
                    else
                    {
                        notes.push_back("advised schedule '" + schedule + "' rejected: not a cron expression");
                    }
                }
                if (std::string accelerator = advice->GetString("accelerator");
                    frame.m_Accelerator.empty() && !accelerator.empty())
                {
                    const Semantics::Concept *entry = ontology ? ontology->Lookup(accelerator) : nullptr;
                    if (IsKind(entry, "accelerator"))
                    {
                        frame.m_Accelerator = entry->m_Id;
                    }
                    else
                    {
                        notes.push_back("advised accelerator '" + accelerator + "' rejected: unknown");
                    }
                }
                if (frame.m_InferenceRegions.empty())
                {
                    frame.m_InferenceRegions = KnownIds(*advice, "inferenceRegions", ontology, "region");
                }
                if (frame.m_TrainingRegions.empty())
                {
                    frame.m_TrainingRegions = KnownIds(*advice, "trainingRegions", ontology, "region");
                }
                if (frame.m_Objectives.empty())
                {
                    frame.m_Objectives = KnownIds(*advice, "objectives", ontology, "objective");
                }
                frame.m_Source = "deterministic+llm:" + provider->Name();
            }
        }

        Descriptor::RichMLService service = Generate(frame, intent);

        // CoMaS retrieval: operator-provided context (e.g. "Prefer RAN placement for closed-loop home control.")
        if (m_Environment.m_Comas)
        {
            try
            {
                Context::RetrievalRequest request;
                request.m_Agent = "executive";
                request.m_Intent = std::string(intent);
                request.m_Limit = 8;
                Context::RetrievalResult context = m_Environment.m_Comas->Retrieve(request);
                notes.push_back("context: " + std::to_string(context.m_Entries.size()) + " entries via " +
                                context.m_Translator);
                for (const Context::Entry &entry : context.m_Entries)
                {
                    std::string text = entry.m_Data.IsObject() ? entry.m_Data.GetString("text") : std::string();
                    if (entry.m_Collection != Context::OPERATOR || text.empty() || !m_Environment.m_Ontology)
                    {
                        continue;
                    }
                    SemanticFrame hint = IntentReader(m_Environment.m_Ontology, text).Read();
                    std::vector<std::string> tokens = Tokenize(text);
                    std::vector<Semantics::ConceptMatch> matches = m_Environment.m_Ontology->Match(tokens);
                    bool domainNamed = false;
                    bool patternNamed = false;
                    for (const Semantics::ConceptMatch &match : matches)
                    {
                        domainNamed = domainNamed || IsKind(match.m_Concept, "domain");
                        patternNamed = patternNamed || IsKind(match.m_Concept, "pattern");
                    }
                    bool tagged = Detail::Contains(entry.m_Tags, frame.m_Domain);
                    bool relevant = (domainNamed ? hint.m_Domain == frame.m_Domain : tagged) &&
                                    (!patternNamed || hint.m_Pattern == frame.m_Pattern);
                    if (!relevant)
                    {
                        continue;
                    }
                    Descriptor::PlacementConstraints &placement = service.m_Constraints.m_Placement;
                    for (const Semantics::ConceptMatch &match : matches)
                    {
                        if (!IsKind(match.m_Concept, "region"))
                        {
                            continue;
                        }
                        bool prefer = false;
                        bool avoid = false;
                        for (size_t back = 1; back <= 3 && back <= match.m_Token; ++back)
                        {
                            const std::string &word = tokens[match.m_Token - back];
                            prefer = prefer || word == "prefer" || word == "preferred" || word == "prefers" ||
                                     word == "favour" || word == "favor";
                            avoid = avoid || NegationWord(word);
                        }
                        const std::string &region = match.m_Concept->m_Id;
                        bool declared = Detail::Contains(service.m_Inference->m_Regions, region) ||
                                        Detail::Contains(service.m_Training->m_Regions, region);
                        if (prefer && !avoid && declared && !Detail::Contains(placement.m_Prefer, region))
                        {
                            placement.m_Prefer.push_back(region);
                            notes.push_back("operator context '" + entry.m_Id + "': prefer region " + region);
                        }
                        else if (avoid && !Detail::Contains(placement.m_Avoid, region))
                        {
                            // never avoid the last region of a role: the service would become unplaceable
                            auto remains = [&](const std::vector<std::string> &regions)
                            {
                                return std::any_of(regions.begin(), regions.end(), [&](const std::string &r)
                                                   { return r != region && !Detail::Contains(placement.m_Avoid, r); });
                            };
                            if (remains(service.m_Inference->m_Regions) && remains(service.m_Training->m_Regions))
                            {
                                placement.m_Avoid.push_back(region);
                                notes.push_back("operator context '" + entry.m_Id + "': avoid region " + region);
                            }
                            else
                            {
                                notes.push_back("operator context '" + entry.m_Id + "': avoiding region " + region +
                                                " ignored (it would leave a role without a region)");
                            }
                        }
                    }
                }
            }
            catch (const std::exception &error)
            {
                notes.push_back(std::string("context retrieval failed: ") + error.what());
            }
        }

        frame.m_Notes.insert(frame.m_Notes.end(), notes.begin(), notes.end());
        result.m_Frame = std::move(frame);
        result.m_Descriptor = std::move(service);
        result.m_Issues = Descriptor::Validate(result.m_Descriptor, m_Validation);
        result.m_Yaml = Descriptor::EmitDescriptor(result.m_Descriptor);
        result.m_Plan = Decompose(result.m_Descriptor);

        if (m_Environment.m_Comas)
        {
            try
            {
                Json evolved = Json::MakeArray();
                for (const Semantics::Concept &entry : result.m_Evolved)
                {
                    evolved.Push(entry.m_Id);
                }
                size_t errors = static_cast<size_t>(
                    std::count_if(result.m_Issues.begin(), result.m_Issues.end(), [](const Descriptor::Issue &issue)
                                  { return issue.m_Severity == Descriptor::Severity::Error; }));
                m_Environment.m_Comas->Remember(
                    "executive", "interpretation", result.m_Descriptor.m_Metadata.m_Name,
                    Json::MakeObject({{"intent", result.m_Descriptor.m_Metadata.m_Intent},
                                      {"operator", std::string(operatorId)},
                                      {"valid", errors == 0},
                                      {"errors", errors},
                                      {"source", result.m_Frame.m_Source},
                                      {"unknown", Detail::Strings(result.m_Frame.m_Unknown)},
                                      {"evolved", std::move(evolved)},
                                      {"outcome", errors == 0 ? "success" : "failure"}}));
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("executive: remembering the interpretation failed: {}", error.what());
            }
        }
        return result;
    }
} // namespace eOps::Agents
