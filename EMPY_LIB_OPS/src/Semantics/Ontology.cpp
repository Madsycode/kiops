#include <eOps/Semantics/Ontology.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/YAML.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace eOps::Semantics
{
    namespace
    {
        constexpr size_t MAX_TERM_BYTES = 128;
        constexpr size_t MAX_TERM_WORDS = 8;
        constexpr double NORMALISED_SCORE = 0.9;

        // plural folding of one lowercase word (Common/Text: shared with the CoMaS retrieval engine)
        std::string NormalizeWord(const std::string &word)
        {
            return Singular(word);
        }

        // index keys of a term: tokens joined by ' ' as written and plural-folded
        struct TermKeys
        {
            std::string m_Exact;
            std::string m_Normal;
            size_t m_Words = 0;
        };

        TermKeys KeysOf(std::string_view term)
        {
            TermKeys keys;
            std::vector<std::string> tokens = Tokenize(term);
            std::vector<std::string> normal;
            normal.reserve(tokens.size());
            for (const std::string &token : tokens)
            {
                normal.push_back(NormalizeWord(token));
            }
            keys.m_Exact = Join(tokens, " ");
            keys.m_Normal = Join(normal, " ");
            keys.m_Words = tokens.size();
            return keys;
        }

        bool Reject(std::string *reason, std::string message)
        {
            if (reason)
            {
                *reason = std::move(message);
            }
            return false;
        }

        bool ValidSuffixTopic(const Json &value)
        {
            return value.IsString() && !value.AsString().empty() && value.AsString().front() != '$' &&
                   value.AsString().front() != '/' && ValidTopic(value.AsString(), false);
        }

        // allowed source and target kinds per relation type
        struct RelationRule
        {
            const char *m_Type;
            std::vector<std::string> m_Sources;
            std::vector<std::string> m_Targets;
        };

        const std::vector<RelationRule> &RelationRules()
        {
            static const std::vector<RelationRule> s_Rules = {
                {"actuates", {"actuator"}, {"quantity"}},
                {"measuredIn", {"quantity", "actuator"}, {"unit"}},
                {"contains", {"domain"}, {"quantity", "actuator"}},
                {"conflictsWith", {"objective"}, {"objective"}},
                {"prefers", {"pattern"}, {"region"}},
            };
            return s_Rules;
        }

        const RelationRule *FindRule(std::string_view type)
        {
            for (const RelationRule &rule : RelationRules())
            {
                if (type == rule.m_Type)
                {
                    return &rule;
                }
            }
            return nullptr;
        }

        bool Contains(const std::vector<std::string> &values, std::string_view value)
        {
            return std::find(values.begin(), values.end(), value) != values.end();
        }

        // YAML flow text for a JSON value (JSON scalars are valid YAML double-quoted scalars)
        std::string Flow(const Json &value)
        {
            if (value.IsObject())
            {
                std::string out = "{";
                bool first = true;
                for (const auto &[key, member] : value.AsObject())
                {
                    out += first ? "" : ", ";
                    out += Json(key).Dump() + ": " + Flow(member);
                    first = false;
                }
                return out + "}";
            }
            if (value.IsArray())
            {
                std::string out = "[";
                bool first = true;
                for (const Json &item : value.AsArray())
                {
                    out += first ? "" : ", ";
                    out += Flow(item);
                    first = false;
                }
                return out + "]";
            }
            return value.Dump();
        }
    } // namespace

    std::string Normalize(std::string_view term)
    {
        std::vector<std::string> words;
        std::string current;
        for (char c : Lower(Trim(term)))
        {
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')
            {
                if (!current.empty())
                {
                    words.push_back(NormalizeWord(std::move(current)));
                    current.clear();
                }
                continue;
            }
            current.push_back(c);
        }
        if (!current.empty())
        {
            words.push_back(NormalizeWord(std::move(current)));
        }
        return Join(words, " ");
    }

    // -------------------------
    // Concept
    // -------------------------

    Json Concept::ToJson() const
    {
        Json synonyms = Json::MakeArray();
        for (const std::string &synonym : m_Synonyms)
        {
            synonyms.Push(synonym);
        }
        Json relations = Json::MakeArray();
        for (const Relation &relation : m_Relations)
        {
            relations.Push(Json::MakeObject({{"type", relation.m_Type}, {"target", relation.m_Target}}));
        }
        return Json::MakeObject({{"id", m_Id},
                                 {"kind", m_Kind},
                                 {"synonyms", std::move(synonyms)},
                                 {"attributes", m_Attributes.IsObject() ? m_Attributes : Json::MakeObject()},
                                 {"relations", std::move(relations)}});
    }

    Concept Concept::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("concept: expected an object");
        }
        const Json *id = json.Find("id");
        const Json *kind = json.Find("kind");
        if (!id || !id->IsString() || !kind || !kind->IsString())
        {
            throw JsonError("concept: 'id' and 'kind' must be strings");
        }
        Concept entry;
        entry.m_Id = id->AsString();
        entry.m_Kind = kind->AsString();
        if (const Json *synonyms = json.Find("synonyms"); synonyms && !synonyms->IsNull())
        {
            if (!synonyms->IsArray())
            {
                throw JsonError("concept '" + entry.m_Id + "': 'synonyms' must be a list");
            }
            for (const Json &synonym : synonyms->AsArray())
            {
                if (!synonym.IsString())
                {
                    throw JsonError("concept '" + entry.m_Id + "': every synonym must be a string");
                }
                entry.m_Synonyms.push_back(synonym.AsString());
            }
        }
        if (const Json *attributes = json.Find("attributes"); attributes && !attributes->IsNull())
        {
            if (!attributes->IsObject())
            {
                throw JsonError("concept '" + entry.m_Id + "': 'attributes' must be a mapping");
            }
            entry.m_Attributes = *attributes;
        }
        if (const Json *relations = json.Find("relations"); relations && !relations->IsNull())
        {
            if (!relations->IsArray())
            {
                throw JsonError("concept '" + entry.m_Id + "': 'relations' must be a list");
            }
            for (const Json &relation : relations->AsArray())
            {
                const Json *type = relation.Find("type");
                const Json *target = relation.Find("target");
                if (!relation.IsObject() || !type || !type->IsString() || !target || !target->IsString())
                {
                    throw JsonError("concept '" + entry.m_Id + "': a relation needs string 'type' and 'target'");
                }
                entry.m_Relations.push_back(Relation{type->AsString(), target->AsString()});
            }
        }
        return entry;
    }

    // -------------------------
    // Ontology
    // -------------------------

    struct Ontology::Impl
    {
        std::vector<std::unique_ptr<Concept>> m_Concepts; // stable addresses across Extend
        std::unordered_map<std::string, size_t> m_Ids;
        std::unordered_map<std::string, size_t> m_Exact;  // term key as written → concept
        std::unordered_map<std::string, size_t> m_Normal; // plural-folded term key → concept
        size_t m_MaxWords = 1;
        uint64_t m_Revision = 0;

        Impl() = default;
        Impl(const Impl &other)
            : m_Ids(other.m_Ids), m_Exact(other.m_Exact), m_Normal(other.m_Normal), m_MaxWords(other.m_MaxWords),
              m_Revision(other.m_Revision)
        {
            m_Concepts.reserve(other.m_Concepts.size());
            for (const auto &entry : other.m_Concepts)
            {
                m_Concepts.push_back(std::make_unique<Concept>(*entry));
            }
        }

        const Concept *ById(std::string_view id) const
        {
            auto found = m_Ids.find(std::string(id));
            return found == m_Ids.end() ? nullptr : m_Concepts[found->second].get();
        }

        // a concept of another kind already owns `key`
        bool Conflicts(const std::unordered_map<std::string, size_t> &index, const std::string &key,
                       const std::string &kind, std::string &owner) const
        {
            auto found = index.find(key);
            if (found == index.end() || m_Concepts[found->second]->m_Kind == kind)
            {
                return false;
            }
            owner = m_Concepts[found->second]->m_Id + " (" + m_Concepts[found->second]->m_Kind + ")";
            return true;
        }

        bool CheckAttributes(const Concept &entry, std::string *reason) const
        {
            const Json &attributes = entry.m_Attributes;
            auto number = [&](std::string_view key) { return attributes.Find(key); };
            auto bad = [&](std::string_view key, std::string_view expected)
            {
                return Reject(reason, "concept '" + entry.m_Id + "': attribute '" + std::string(key) + "' must be " +
                                          std::string(expected));
            };
            for (const char *key : {"min", "max", "latencyMs"})
            {
                if (const Json *value = number(key); value && !value->IsNumber())
                {
                    return bad(key, "a number");
                }
            }
            if (const Json *min = attributes.Find("min"), *max = attributes.Find("max");
                min && max && min->AsNumber() > max->AsNumber())
            {
                return Reject(reason, "concept '" + entry.m_Id + "': attribute min exceeds max");
            }
            for (const char *key : {"unit", "quantity", "statistic", "suffix", "cron"})
            {
                if (const Json *value = attributes.Find(key); value && !value->IsString())
                {
                    return bad(key, "a string");
                }
            }
            const std::string &kind = entry.m_Kind;
            if (kind == "domain")
            {
                if (const Json *root = attributes.Find("root"); root && !ValidSuffixTopic(*root))
                {
                    return bad("root", "a concrete topic");
                }
            }
            else if (kind == "quantity" || kind == "actuator")
            {
                const Json *topic = attributes.Find("topic");
                if (!topic || !ValidSuffixTopic(*topic))
                {
                    return bad("topic", "a concrete topic suffix");
                }
                if (const Json *quantity = attributes.Find("quantity"); kind == "actuator" && quantity)
                {
                    const Concept *target = ById(quantity->AsString());
                    if (!target || target->m_Kind != "quantity")
                    {
                        return bad("quantity", "the id of a known quantity");
                    }
                }
            }
            else if (kind == "pattern")
            {
                if (const Json *actionable = attributes.Find("actionable"); actionable && !actionable->IsBool())
                {
                    return bad("actionable", "a boolean");
                }
                if (const Json *latency = attributes.Find("latencyMs"); latency && latency->AsNumber() <= 0)
                {
                    return bad("latencyMs", "positive");
                }
                static const std::vector<std::string> s_Statistics = {"p50", "p90", "p95", "p99", "mean", "max"};
                if (const Json *statistic = attributes.Find("statistic");
                    statistic && !Contains(s_Statistics, statistic->AsString()))
                {
                    return bad("statistic", "one of p50, p90, p95, p99, mean, max");
                }
                if (const Json *suffix = attributes.Find("suffix"); suffix && !IsDnsLabel(suffix->AsString()))
                {
                    return bad("suffix", "a DNS label");
                }
            }
            else if (kind == "schedule")
            {
                const Json *cron = attributes.Find("cron");
                if (!cron)
                {
                    return bad("cron", "present");
                }
                std::string expression = Trim(cron->AsString());
                size_t fields = 0;
                for (const std::string &field : Split(expression, ' '))
                {
                    fields += field.empty() ? 0 : 1;
                }
                if (expression.empty() || (expression.front() != '@' && fields != 5))
                {
                    return bad("cron", "a 5-field cron expression or an @macro");
                }
            }
            return true;
        }
    };

    Ontology::Ontology() : m_Impl(std::make_unique<Impl>())
    {
    }

    Ontology::~Ontology() = default;

    Ontology::Ontology(const Ontology &other) : m_Impl(std::make_unique<Impl>(*other.m_Impl))
    {
    }

    Ontology &Ontology::operator=(const Ontology &other)
    {
        if (this != &other)
        {
            m_Impl = std::make_unique<Impl>(*other.m_Impl);
        }
        return *this;
    }

    Ontology::Ontology(Ontology &&other) noexcept : m_Impl(std::move(other.m_Impl))
    {
    }

    Ontology &Ontology::operator=(Ontology &&other) noexcept
    {
        std::swap(m_Impl, other.m_Impl);
        return *this;
    }

    Ontology Ontology::FromYaml(const eCore::YamlNode &root)
    {
        if (!root.IsMap() || !root["concepts"].IsList())
        {
            throw std::invalid_argument(root.Path() + ": an ontology needs a 'concepts' list");
        }
        // concepts first, references (relations, actuator quantity) second, so a document may reference
        // concepts declared later
        std::vector<Concept> entries;
        for (const eCore::YamlNode &item : root["concepts"].Items())
        {
            try
            {
                entries.push_back(Concept::FromJson(eOps::FromYaml(item)));
            }
            catch (const std::exception &error)
            {
                throw std::invalid_argument(item.Path() + ": " + error.what());
            }
        }
        Ontology ontology;
        for (const Concept &entry : entries)
        {
            Concept bare = entry;
            bare.m_Relations.clear();
            if (bare.m_Attributes.IsObject())
            {
                bare.m_Attributes.Erase("quantity");
            }
            std::string reason;
            if (!ontology.Extend(std::move(bare), &reason))
            {
                throw std::invalid_argument(root.Path() + ": concept '" + entry.m_Id + "' rejected: " + reason);
            }
        }
        for (const Concept &entry : entries)
        {
            const Json *quantity = entry.m_Attributes.Find("quantity");
            if (entry.m_Relations.empty() && !quantity)
            {
                continue;
            }
            Concept links;
            links.m_Id = entry.m_Id;
            links.m_Kind = entry.m_Kind;
            links.m_Relations = entry.m_Relations;
            if (quantity)
            {
                links.m_Attributes = Json::MakeObject({{"quantity", *quantity}});
            }
            std::string reason;
            if (!ontology.Extend(std::move(links), &reason))
            {
                throw std::invalid_argument(root.Path() + ": concept '" + entry.m_Id + "' rejected: " + reason);
            }
        }
        ontology.m_Impl->m_Revision = root.Get<uint64_t>("revision", 0);
        return ontology;
    }

    Ontology Ontology::Load(const std::filesystem::path &path)
    {
        return FromYaml(eCore::YamlNode::Load(path.string()));
    }

    std::string Ontology::ToYaml() const
    {
        std::string out = "# BoM-Ontology: concepts [{id, kind, synonyms, attributes, relations}]\n";
        out += "revision: " + std::to_string(m_Impl->m_Revision) + "\n";
        out += "concepts:\n";
        for (const auto &entry : m_Impl->m_Concepts)
        {
            Json synonyms = Json::MakeArray();
            for (const std::string &synonym : entry->m_Synonyms)
            {
                synonyms.Push(synonym);
            }
            out += "  - id: " + Json(entry->m_Id).Dump() + "\n";
            out += "    kind: " + Json(entry->m_Kind).Dump() + "\n";
            out += "    synonyms: " + Flow(synonyms) + "\n";
            out +=
                "    attributes: " + Flow(entry->m_Attributes.IsObject() ? entry->m_Attributes : Json::MakeObject()) +
                "\n";
            if (entry->m_Relations.empty())
            {
                out += "    relations: []\n";
                continue;
            }
            out += "    relations:\n";
            for (const Relation &relation : entry->m_Relations)
            {
                out += "      - {type: " + Json(relation.m_Type).Dump() +
                       ", target: " + Json(relation.m_Target).Dump() + "}\n";
            }
        }
        return out;
    }

    void Ontology::Save(const std::filesystem::path &path) const
    {
        WriteFileAtomic(path, ToYaml());
    }

    const Concept *Ontology::Find(std::string_view id) const
    {
        return m_Impl->ById(id);
    }

    const Concept *Ontology::Lookup(std::string_view term) const
    {
        if (const Concept *byId = m_Impl->ById(Trim(term)))
        {
            return byId;
        }
        TermKeys keys = KeysOf(term);
        if (keys.m_Words == 0)
        {
            return nullptr;
        }
        if (auto found = m_Impl->m_Exact.find(keys.m_Exact); found != m_Impl->m_Exact.end())
        {
            return m_Impl->m_Concepts[found->second].get();
        }
        if (auto found = m_Impl->m_Normal.find(keys.m_Normal); found != m_Impl->m_Normal.end())
        {
            return m_Impl->m_Concepts[found->second].get();
        }
        return nullptr;
    }

    std::vector<ConceptMatch> Ontology::Match(const std::vector<std::string> &tokens) const
    {
        std::vector<std::string> exact;
        std::vector<std::string> normal;
        exact.reserve(tokens.size());
        normal.reserve(tokens.size());
        for (const std::string &token : tokens)
        {
            exact.push_back(Lower(Trim(token)));
            normal.push_back(NormalizeWord(exact.back()));
        }
        // every span that names a concept, then longest first without overlap
        std::vector<ConceptMatch> candidates;
        for (size_t start = 0; start < exact.size(); ++start)
        {
            std::string exactKey;
            std::string normalKey;
            size_t limit = std::min(m_Impl->m_MaxWords, exact.size() - start);
            for (size_t length = 1; length <= limit; ++length)
            {
                const std::string &word = exact[start + length - 1];
                if (word.empty() || word.find(' ') != std::string::npos)
                {
                    break;
                }
                exactKey += (length > 1 ? " " : "") + word;
                normalKey += (length > 1 ? " " : "") + normal[start + length - 1];
                const Concept *hit = nullptr;
                double score = 1;
                if (auto found = m_Impl->m_Exact.find(exactKey); found != m_Impl->m_Exact.end())
                {
                    hit = m_Impl->m_Concepts[found->second].get();
                }
                else if (auto folded = m_Impl->m_Normal.find(normalKey); folded != m_Impl->m_Normal.end())
                {
                    hit = m_Impl->m_Concepts[folded->second].get();
                    score = NORMALISED_SCORE;
                }
                if (hit)
                {
                    candidates.push_back(ConceptMatch{hit, start, length, exactKey, score});
                }
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const ConceptMatch &a, const ConceptMatch &b)
                         {
                             if (a.m_Length != b.m_Length)
                             {
                                 return a.m_Length > b.m_Length;
                             }
                             if (a.m_Score != b.m_Score)
                             {
                                 return a.m_Score > b.m_Score;
                             }
                             return a.m_Token < b.m_Token;
                         });
        std::vector<bool> taken(tokens.size(), false);
        std::vector<ConceptMatch> matches;
        for (const ConceptMatch &candidate : candidates)
        {
            bool free = true;
            for (size_t i = candidate.m_Token; i < candidate.m_Token + candidate.m_Length; ++i)
            {
                free = free && !taken[i];
            }
            if (!free)
            {
                continue;
            }
            for (size_t i = candidate.m_Token; i < candidate.m_Token + candidate.m_Length; ++i)
            {
                taken[i] = true;
            }
            matches.push_back(candidate);
        }
        std::sort(matches.begin(), matches.end(),
                  [](const ConceptMatch &a, const ConceptMatch &b) { return a.m_Token < b.m_Token; });
        return matches;
    }

    std::vector<const Concept *> Ontology::Related(std::string_view id, std::string_view relation) const
    {
        std::vector<const Concept *> related;
        const Concept *source = Find(id);
        if (!source)
        {
            return related;
        }
        for (const Relation &link : source->m_Relations)
        {
            if (link.m_Type == relation)
            {
                if (const Concept *target = Find(link.m_Target))
                {
                    related.push_back(target);
                }
            }
        }
        return related;
    }

    std::vector<const Concept *> Ontology::RelatedFrom(std::string_view id, std::string_view relation) const
    {
        std::vector<const Concept *> related;
        for (const auto &entry : m_Impl->m_Concepts)
        {
            for (const Relation &link : entry->m_Relations)
            {
                if (link.m_Type == relation && link.m_Target == id)
                {
                    related.push_back(entry.get());
                    break;
                }
            }
        }
        return related;
    }

    std::vector<const Concept *> Ontology::OfKind(std::string_view kind) const
    {
        std::vector<const Concept *> concepts;
        for (const auto &entry : m_Impl->m_Concepts)
        {
            if (entry->m_Kind == kind)
            {
                concepts.push_back(entry.get());
            }
        }
        return concepts;
    }

    bool Ontology::Extend(Concept entry, std::string *reason)
    {
        Impl &impl = *m_Impl;
        entry.m_Id = Trim(entry.m_Id);
        entry.m_Kind = Trim(entry.m_Kind);
        if (entry.m_Id.empty() || entry.m_Id.size() > MAX_TERM_BYTES)
        {
            return Reject(reason, "concept id must be 1.." + std::to_string(MAX_TERM_BYTES) + " characters");
        }
        for (char c : entry.m_Id)
        {
            if (static_cast<uint8_t>(c) <= 0x20 || c == 0x7F)
            {
                return Reject(reason, "concept id '" + entry.m_Id + "' contains whitespace or control characters");
            }
        }
        if (!Contains(Kinds(), entry.m_Kind))
        {
            return Reject(reason, "concept '" + entry.m_Id + "': unknown kind '" + entry.m_Kind + "'");
        }
        if (!entry.m_Attributes.IsNull() && !entry.m_Attributes.IsObject())
        {
            return Reject(reason, "concept '" + entry.m_Id + "': attributes must be an object");
        }

        // merge into the existing same-kind concept (attributes are add-only: a differing value is a conflict)
        auto existing = impl.m_Ids.find(entry.m_Id);
        Concept merged;
        if (existing != impl.m_Ids.end())
        {
            merged = *impl.m_Concepts[existing->second];
            if (merged.m_Kind != entry.m_Kind)
            {
                return Reject(reason, "concept '" + entry.m_Id + "' already exists with kind '" + merged.m_Kind + "'");
            }
            if (!merged.m_Attributes.IsObject())
            {
                merged.m_Attributes = Json::MakeObject();
            }
            if (entry.m_Attributes.IsObject())
            {
                for (const auto &[key, value] : entry.m_Attributes.AsObject())
                {
                    const Json *current = merged.m_Attributes.Find(key);
                    if (current && !(*current == value))
                    {
                        return Reject(reason, "concept '" + entry.m_Id + "': attribute '" + key + "' conflicts with " +
                                                  current->Dump());
                    }
                    merged.m_Attributes.Set(key, value);
                }
            }
        }
        else
        {
            merged.m_Id = entry.m_Id;
            merged.m_Kind = entry.m_Kind;
            merged.m_Attributes = entry.m_Attributes.IsObject() ? entry.m_Attributes : Json::MakeObject();
        }
        for (const std::string &raw : entry.m_Synonyms)
        {
            std::string synonym = Trim(raw);
            TermKeys keys = KeysOf(synonym);
            if (synonym.empty() || synonym.size() > MAX_TERM_BYTES || keys.m_Words == 0 ||
                keys.m_Words > MAX_TERM_WORDS)
            {
                return Reject(reason, "concept '" + entry.m_Id + "': synonym '" + synonym + "' is not a usable term");
            }
            if (!Contains(merged.m_Synonyms, synonym))
            {
                merged.m_Synonyms.push_back(std::move(synonym));
            }
        }
        for (const Relation &relation : entry.m_Relations)
        {
            const RelationRule *rule = FindRule(relation.m_Type);
            if (!rule)
            {
                return Reject(reason, "concept '" + entry.m_Id + "': unknown relation '" + relation.m_Type + "'");
            }
            const Concept *target = impl.ById(relation.m_Target);
            if (!target)
            {
                return Reject(reason, "concept '" + entry.m_Id + "': relation target '" + relation.m_Target +
                                          "' does not exist");
            }
            if (!Contains(rule->m_Sources, merged.m_Kind) || !Contains(rule->m_Targets, target->m_Kind))
            {
                return Reject(reason, "concept '" + entry.m_Id + "': relation '" + relation.m_Type + "' cannot link " +
                                          merged.m_Kind + " to " + target->m_Kind);
            }
            bool known = std::any_of(merged.m_Relations.begin(), merged.m_Relations.end(), [&](const Relation &link)
                                     { return link.m_Type == relation.m_Type && link.m_Target == relation.m_Target; });
            if (!known)
            {
                merged.m_Relations.push_back(relation);
            }
        }
        if (!impl.CheckAttributes(merged, reason))
        {
            return false;
        }

        // every term of the concept must be free of other-kind bindings
        std::vector<TermKeys> terms;
        terms.push_back(KeysOf(merged.m_Id));
        for (const std::string &synonym : merged.m_Synonyms)
        {
            terms.push_back(KeysOf(synonym));
        }
        for (const TermKeys &keys : terms)
        {
            std::string owner;
            if (keys.m_Words > 0 && (impl.Conflicts(impl.m_Exact, keys.m_Exact, merged.m_Kind, owner) ||
                                     impl.Conflicts(impl.m_Normal, keys.m_Normal, merged.m_Kind, owner)))
            {
                return Reject(reason, "concept '" + entry.m_Id + "': term '" + keys.m_Exact + "' is bound to " + owner);
            }
        }

        size_t index = 0;
        if (existing != impl.m_Ids.end())
        {
            index = existing->second;
            *impl.m_Concepts[index] = std::move(merged);
        }
        else
        {
            index = impl.m_Concepts.size();
            impl.m_Concepts.push_back(std::make_unique<Concept>(std::move(merged)));
            impl.m_Ids.emplace(impl.m_Concepts.back()->m_Id, index);
        }
        // first binding wins among same-kind concepts
        for (const TermKeys &keys : terms)
        {
            if (keys.m_Words == 0)
            {
                continue;
            }
            impl.m_Exact.emplace(keys.m_Exact, index);
            impl.m_Normal.emplace(keys.m_Normal, index);
            impl.m_MaxWords = std::max(impl.m_MaxWords, keys.m_Words);
        }
        ++impl.m_Revision;
        return true;
    }

    size_t Ontology::Size() const noexcept
    {
        return m_Impl ? m_Impl->m_Concepts.size() : 0;
    }

    uint64_t Ontology::Revision() const noexcept
    {
        return m_Impl ? m_Impl->m_Revision : 0;
    }

    const std::vector<std::string> &Ontology::Kinds()
    {
        static const std::vector<std::string> s_Kinds = {"domain", "quantity",    "actuator", "pattern", "objective",
                                                         "region", "accelerator", "unit",     "schedule"};
        return s_Kinds;
    }

    const std::vector<std::string> &Ontology::RelationTypes()
    {
        static const std::vector<std::string> s_Types = {"actuates", "measuredIn", "contains", "conflictsWith",
                                                         "prefers"};
        return s_Types;
    }
} // namespace eOps::Semantics
