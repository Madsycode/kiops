#include <eOps/Context/Comas.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include "Lexicon.h"
#include <array>
#include <algorithm>

// Translate(x1, c) → x2: the Reasoning provider proposes a structured query (validated before use)
// or the deterministic vocabulary translator maps intent words to collections, conditions and
// relevance keywords

namespace eOps::Context
{
    namespace
    {
        // collection bits in the fixed emission order
        enum : uint32_t
        {
            RES = 1,
            NET = 2,
            TOOL = 4,
            MEM = 8, // memory + consolidated
            OPR = 16,
            OBS = 32
        };

        struct Word
        {
            std::string_view m_Stem;
            uint32_t m_Collections;
            bool m_Keyword; // also kept as a relevance keyword (content-bearing)
        };

        constexpr std::array<Word, 57> VOCABULARY = {{
            {"node", RES, false},     {"cpu", RES, false},         {"core", RES, false},
            {"memory", RES, false},   {"ram", RES, false},         {"accelerator", RES, false},
            {"capacity", RES, false}, {"resource", RES, false},    {"machine", RES, false},
            {"host", RES, false},     {"server", RES, false},      {"hardware", RES, false},
            {"compute", RES, false},  {"latency", NET, false},     {"delay", NET, false},
            {"locality", NET, false}, {"region", NET, false},      {"topic", NET, false},
            {"shard", NET, false},    {"traffic", NET, false},     {"subscriber", NET, false},
            {"consumer", NET, false}, {"density", NET, false},     {"link", NET, false},
            {"network", NET, false},  {"bandwidth", NET, false},   {"tool", TOOL, false},
            {"skill", TOOL, false},   {"capability", TOOL, false}, {"history", MEM, false},
            {"previous", MEM, false}, {"past", MEM, false},        {"experience", MEM, false},
            {"episode", MEM, false},  {"decision", MEM, true},     {"outcome", MEM, true},
            {"failure", MEM, true},   {"fail", MEM, true},         {"drift", MEM, true},
            {"retrain", MEM, true},   {"lesson", MEM, true},       {"incident", MEM, true},
            {"rollback", MEM, true},  {"promotion", MEM, true},    {"policy", OPR, true},
            {"operator", OPR, false}, {"preference", OPR, true},   {"prefer", OPR, true},
            {"site", OPR, true},      {"rule", OPR, true},         {"guideline", OPR, true},
            {"metric", OBS, true},    {"observation", OBS, false}, {"status", OBS, true},
            {"telemetry", OBS, true}, {"measurement", OBS, true},  {"heartbeat", OBS, true},
        }};

        constexpr std::array<std::string_view, 3> ACCELERATORS = {"gpu", "npu", "tpu"};
        constexpr std::array<std::string_view, 3> DEFAULT_REGIONS = {"ran", "edge", "cloud"};
        constexpr std::array<std::pair<std::string_view, std::string_view>, 2> REGION_SYNONYMS = {
            {{"datacenter", "cloud"}, {"datacentre", "cloud"}}};
        // two-word region synonyms (Tokenize splits "far-edge" into "far edge")
        constexpr std::array<std::array<std::string_view, 3>, 5> REGION_PHRASES = {{{"data", "center", "cloud"},
                                                                                    {"data", "centre", "cloud"},
                                                                                    {"far", "edge", "ran"},
                                                                                    {"base", "station", "ran"},
                                                                                    {"radio", "access", "ran"}}};
        constexpr std::array<std::string_view, 8> MOST = {"most",    "largest", "highest",  "biggest",
                                                          "maximum", "max",     "greatest", "top"};
        constexpr std::array<std::string_view, 7> LEAST = {"least", "lowest", "smallest", "minimum",
                                                           "min",   "fewest", "fastest"};
        constexpr std::array<std::pair<std::string_view, std::string_view>, 6> ORDER_FIELDS = {
            {{"cpu", "cpu"},
             {"core", "cpu"},
             {"memory", "memoryBytes"},
             {"ram", "memoryBytes"},
             {"latency", "ms"},
             {"delay", "ms"}}};
        constexpr std::array<std::string_view, 5> SERVICE_CUES = {"for", "of", "service", "about", "by"};
        constexpr std::string_view PUNCTUATION = ",.;:!?()[]{}<>\"'`";

        const char *SYSTEM_PROMPT =
            "You translate an MLOps agent's knowledge request into one structured query for the CoMaS knowledge "
            "plane. Collections: resources (compute nodes: id, region, cpu, memory, memoryBytes, accelerators{kind: "
            "count}, labels, broker), network (kind latency {from, to, region[], ms}; kind density {topic, rates, "
            "region[]}; kind consumers {topic, counts, region[]}; kind shard {region, topics[]}; id model = the whole "
            "network model), tools (tool/skill registry: kind, description, capabilities[]), memory (agents memory: "
            "agent, kind, service, outcome, data), consolidated (service, kind, count, successes, failures, "
            "lastOutcome, lastAt), operator (operator context: text), observations (source, kind, service, region, "
            "topic, data). Conditions all must hold; field is a dotted path into the entry data or id, tags, "
            "collection, updated; an array field matches when any element matches; op matches = MMQP topic filter "
            "overlap. Use text for relevance keywords. Prefer few, precise conditions. Use orderBy \"\" for "
            "relevance order, limit 20 unless the request implies another number, select [] for whole entries. "
            "Only use values grounded in the request or the provided context.";

        bool Contains(auto &&values, std::string_view value)
        {
            return std::find(values.begin(), values.end(), value) != values.end();
        }

        void AddUnique(std::vector<std::string> &values, std::string value)
        {
            if (!value.empty() && !Contains(values, value))
            {
                values.push_back(std::move(value));
            }
        }

        std::vector<std::string> ContextStrings(const Json &context, std::string_view member)
        {
            std::vector<std::string> values;
            if (const Json *list = context.Find(member); list && list->IsArray())
            {
                for (const Json &item : list->AsArray())
                {
                    if (item.IsString() && !item.AsString().empty())
                    {
                        values.push_back(Lower(item.AsString()));
                    }
                }
            }
            return values;
        }

        const Word *Lookup(std::string_view stem)
        {
            for (const Word &word : VOCABULARY)
            {
                if (word.m_Stem == stem)
                {
                    return &word;
                }
            }
            return nullptr;
        }

        std::string StripPunctuation(std::string_view word)
        {
            while (!word.empty() && PUNCTUATION.find(word.front()) != std::string_view::npos)
            {
                word.remove_prefix(1);
            }
            while (!word.empty() && PUNCTUATION.find(word.back()) != std::string_view::npos)
            {
                word.remove_suffix(1);
            }
            return std::string(word);
        }

        bool IsTopicWord(std::string_view word)
        {
            return word.find('/') != std::string_view::npos && ValidTopic(word, true);
        }

        // every target collection carries the field a condition would test
        bool Within(uint32_t targets, uint32_t allowed)
        {
            return targets != 0 && (targets & ~allowed) == 0;
        }

        void AddText(Query &query, std::string_view text)
        {
            for (std::string &keyword : Detail::Keywords(text))
            {
                AddUnique(query.m_Text, std::move(keyword));
            }
        }

        bool Acceptable(const Query &query, const std::vector<std::string> &known, std::string *reason)
        {
            if (query.m_Collections.empty() && query.m_Where.empty() && query.m_Text.empty())
            {
                *reason = "the query is empty";
                return false;
            }
            for (const std::string &collection : query.m_Collections)
            {
                if (!Contains(known, collection))
                {
                    *reason = "unknown collection '" + collection + "'";
                    return false;
                }
            }
            return true;
        }
    } // namespace

    Query Comas::Translate(std::string_view intent, const Json &context, std::string *translator) const
    {
        auto record = [translator](std::string name)
        {
            if (translator)
            {
                *translator = std::move(name);
            }
        };
        Intelligence::Provider *provider =
            m_Providers ? m_Providers->Get(Intelligence::Capability::Reasoning) : nullptr;
        if (!provider)
        {
            record("deterministic");
            return TranslateDeterministic(intent, context);
        }
        std::string user = Json(Json::Object{{"intent", std::string(intent)}, {"context", context}}).Dump();
        std::string reason;
        if (std::optional<Json> proposal =
                Intelligence::AskJson(provider, Intelligence::Capability::Reasoning, SYSTEM_PROMPT, std::move(user),
                                      Query::Schema(), &reason))
        {
            try
            {
                Query query = Query::FromJson(*proposal);
                std::vector<std::string> known = {MEMORY,   TOOLS,        RESOURCES,   NETWORK,
                                                  OPERATOR, OBSERVATIONS, CONSOLIDATED};
                for (const std::string &collection : m_Plane.Collections())
                {
                    AddUnique(known, collection);
                }
                if (Acceptable(query, known, &reason))
                {
                    record("llm:" + provider->Name());
                    return query;
                }
            }
            catch (const std::exception &error)
            {
                reason = error.what();
            }
        }
        EMPY_DEBUG("comas: LLM translation rejected, using the deterministic translator: {}", reason);
        record("deterministic(fallback)");
        return TranslateDeterministic(intent, context);
    }

    Query Comas::TranslateDeterministic(std::string_view intent, const Json &context) const
    {
        Query query;
        std::vector<std::string> regions = ContextStrings(context, "regions");
        for (std::string_view region : DEFAULT_REGIONS)
        {
            AddUnique(regions, std::string(region));
        }
        std::vector<std::string> knownServices = ContextStrings(context, "services");

        // 1. quoted segments and topic/service-shaped words, taken from the raw text
        std::vector<std::string> topics;
        std::vector<std::string> services;
        std::string residual;
        std::string text(intent);
        for (size_t i = 0; i < text.size(); ++i)
        {
            char quote = text[i];
            if (quote != '"' && quote != '`' && !(quote == '\'' && (i == 0 || text[i - 1] == ' ')))
            {
                continue;
            }
            size_t end = text.find(quote, i + 1);
            if (end == std::string::npos)
            {
                break;
            }
            std::string inner = Trim(text.substr(i + 1, end - i - 1));
            if (IsTopicWord(inner))
            {
                AddUnique(topics, inner);
                text.replace(i, end - i + 1, end - i + 1, ' ');
            }
            else if (IsDnsLabel(Lower(inner)))
            {
                AddUnique(services, Lower(inner));
                text.replace(i, end - i + 1, end - i + 1, ' ');
            }
            i = end;
        }
        std::vector<std::string> words;
        for (const std::string &part : Split(text, ' '))
        {
            if (!Trim(part).empty())
            {
                words.push_back(Trim(part));
            }
        }
        for (size_t i = 0; i < words.size(); ++i)
        {
            std::string word = StripPunctuation(words[i]);
            std::string lower = Lower(word);
            std::string previous = i > 0 ? Lower(StripPunctuation(words[i - 1])) : std::string();
            std::string next = i + 1 < words.size() ? Lower(StripPunctuation(words[i + 1])) : std::string();
            if (IsTopicWord(word))
            {
                AddUnique(topics, word);
                continue;
            }
            bool known = std::any_of(knownServices.begin(), knownServices.end(),
                                     [&lower](const std::string &service) { return service == lower; });
            bool shaped = IsDnsLabel(lower) && lower.find('-') != std::string::npos &&
                          (Contains(SERVICE_CUES, previous) || next == "service");
            if (known || shaped)
            {
                AddUnique(services, lower);
                continue;
            }
            residual += words[i] + " ";
        }

        // 2. vocabulary, regions, accelerators, ordering and residual keywords
        uint32_t targets = 0;
        std::vector<std::string> matchedRegions;
        std::vector<std::string> accelerators;
        int32_t direction = 0; // +1 most, -1 least
        std::string orderField;
        std::vector<std::string> tokens = Tokenize(residual);
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            const std::string &token = tokens[i];
            if (std::all_of(token.begin(), token.end(),
                            [](char c) { return (c >= '0' && c <= '9') || c == '.' || c == '-'; }))
            {
                continue;
            }
            bool phrase = false;
            for (const auto &[first, second, region] : REGION_PHRASES)
            {
                if (token == first && i + 1 < tokens.size() && tokens[i + 1] == second)
                {
                    AddUnique(matchedRegions, std::string(region));
                    ++i;
                    phrase = true;
                    break;
                }
            }
            if (phrase)
            {
                continue;
            }
            std::string stem = Detail::Stem(token);
            if (Contains(regions, token) || Contains(regions, stem))
            {
                AddUnique(matchedRegions, Contains(regions, token) ? token : stem);
                continue;
            }
            if (auto synonym = std::find_if(REGION_SYNONYMS.begin(), REGION_SYNONYMS.end(),
                                            [&stem](const auto &pair) { return pair.first == stem; });
                synonym != REGION_SYNONYMS.end())
            {
                AddUnique(matchedRegions, std::string(synonym->second));
                continue;
            }
            if (Contains(ACCELERATORS, stem))
            {
                AddUnique(accelerators, stem);
                targets |= RES;
                continue;
            }
            if (Contains(MOST, stem))
            {
                direction = 1;
                continue;
            }
            if (Contains(LEAST, stem))
            {
                direction = -1;
                continue;
            }
            if (const Word *word = Lookup(stem))
            {
                targets |= word->m_Collections;
                if (word->m_Keyword)
                {
                    AddUnique(query.m_Text, stem);
                }
                if (orderField.empty())
                {
                    for (const auto &[name, field] : ORDER_FIELDS)
                    {
                        if (name == stem)
                        {
                            orderField = field;
                        }
                    }
                }
                continue;
            }
            if (Detail::IsStopword(stem))
            {
                continue;
            }
            AddUnique(query.m_Text, stem);
        }
        if (targets == 0)
        {
            if (!matchedRegions.empty())
            {
                targets = RES | NET;
            }
            else if (!services.empty())
            {
                targets = MEM | OBS;
            }
            else if (!topics.empty())
            {
                targets = NET | OBS;
            }
        }
        if (targets & RES)
        {
            query.m_Collections.emplace_back(RESOURCES);
        }
        if (targets & NET)
        {
            query.m_Collections.emplace_back(NETWORK);
        }
        if (targets & TOOL)
        {
            query.m_Collections.emplace_back(TOOLS);
        }
        if (targets & MEM)
        {
            query.m_Collections.emplace_back(MEMORY);
            query.m_Collections.emplace_back(CONSOLIDATED);
        }
        if (targets & OPR)
        {
            query.m_Collections.emplace_back(OPERATOR);
        }
        if (targets & OBS)
        {
            query.m_Collections.emplace_back(OBSERVATIONS);
        }

        // 3. conditions only where every targeted collection carries the field; else keywords
        for (const std::string &kind : accelerators)
        {
            if (targets == RES)
            {
                query.m_Where.push_back(Condition{"accelerators." + kind, Op::Exists, Json(true)});
            }
            else
            {
                AddUnique(query.m_Text, kind);
            }
        }
        if (!matchedRegions.empty())
        {
            if (!Within(targets, RES | NET))
            {
                for (const std::string &region : matchedRegions)
                {
                    AddUnique(query.m_Text, region);
                }
            }
            else if (matchedRegions.size() == 1)
            {
                query.m_Where.push_back(Condition{"region", Op::Eq, Json(matchedRegions.front())});
            }
            else if (targets == NET)
            {
                // links between all named regions
                for (const std::string &region : matchedRegions)
                {
                    query.m_Where.push_back(Condition{"region", Op::Eq, Json(region)});
                }
            }
            else
            {
                Json options = Json::MakeArray();
                for (const std::string &region : matchedRegions)
                {
                    options.Push(Json(region));
                }
                query.m_Where.push_back(Condition{"region", Op::In, std::move(options)});
            }
        }
        if (!services.empty())
        {
            if (!Within(targets, MEM | OBS))
            {
                for (const std::string &service : services)
                {
                    AddText(query, service);
                }
            }
            else if (services.size() == 1)
            {
                query.m_Where.push_back(Condition{"service", Op::Eq, Json(services.front())});
            }
            else
            {
                Json options = Json::MakeArray();
                for (const std::string &service : services)
                {
                    options.Push(Json(service));
                }
                query.m_Where.push_back(Condition{"service", Op::In, std::move(options)});
            }
        }
        for (size_t i = 0; i < topics.size(); ++i)
        {
            if (i == 0 && Within(targets, NET | OBS))
            {
                query.m_Where.push_back(Condition{"topic", Op::Matches, Json(topics[i])});
            }
            else
            {
                AddText(query, topics[i]);
            }
        }
        if (direction != 0 && !orderField.empty())
        {
            query.m_OrderBy = orderField;
            query.m_Descending = direction > 0;
        }
        return query;
    }
} // namespace eOps::Context
