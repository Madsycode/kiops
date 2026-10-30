#include <eOps/Context/Comas.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include "Lexicon.h"
#include <set>
#include <map>
#include <array>
#include <atomic>
#include <format>
#include <algorithm>

namespace eOps::Context
{
    namespace
    {
        constexpr size_t SUMMARY_BYTES = 4096;
        constexpr size_t LINE_BYTES = 240;
        constexpr size_t CONTEXT_TEXT_BYTES = 512;
        constexpr size_t MAX_SERVICES = 64;
        constexpr size_t MAX_LIMIT = 1000;
        constexpr std::array<std::string_view, 12> SUCCESSES = {"success",  "succeeded", "ok",       "approved",
                                                                "amended",  "promoted",  "promote",  "loaded",
                                                                "deployed", "completed", "accepted", "done"};
        constexpr std::array<std::string_view, 10> FAILURES = {"failure",  "failed",  "error",  "rejected",
                                                               "reject",   "timeout", "denied", "rolledback",
                                                               "rollback", "aborted"};

        // chronologically ordered unique id: zero-padded Unix ms, a process-wide sequence (orders entries
        // created within one millisecond) and a random suffix (unique across processes)
        std::string SequenceId(uint64_t now)
        {
            static std::atomic<uint64_t> s_Sequence{0};
            return std::format("{:013}-{:06}-{}", now, s_Sequence.fetch_add(1) % 1000000, NewId({}));
        }

        std::string Scalar(const Json &value)
        {
            if (value.IsString())
            {
                return value.AsString();
            }
            if (value.IsNull())
            {
                return {};
            }
            return value.Dump();
        }

        void AddTag(std::vector<std::string> &tags, std::string_view tag)
        {
            if (!tag.empty() && std::find(tags.begin(), tags.end(), tag) == tags.end())
            {
                tags.emplace_back(tag);
            }
        }

        // outcome of an agents-memory entry: an explicit "outcome", else the first verdict-like member
        std::string Outcome(const Json &memory)
        {
            if (const Json *outcome = memory.Find("outcome"); outcome && outcome->IsString())
            {
                return Lower(outcome->AsString());
            }
            const Json &inner = memory["data"];
            for (const char *member : {"outcome", "status", "result", "verdict", "action", "decision"})
            {
                if (const Json *value = inner.Find(member); value && value->IsString())
                {
                    return Lower(value->AsString());
                }
            }
            for (const char *member : {"ok", "success"})
            {
                if (const Json *value = inner.Find(member); value && value->IsBool())
                {
                    return value->AsBool() ? "success" : "failure";
                }
            }
            return {};
        }

        bool IsIn(std::string_view value, auto &&set)
        {
            return std::find(set.begin(), set.end(), value) != set.end();
        }

        std::string Rates(const Json &rates)
        {
            std::string text;
            if (rates.IsObject())
            {
                for (const auto &[region, rate] : rates.AsObject())
                {
                    text += " " + region + "=" + Scalar(rate);
                }
            }
            return text;
        }

        std::string Describe(const Entry &entry)
        {
            const Json &data = entry.m_Data;
            if (entry.m_Collection == RESOURCES)
            {
                std::string accelerators;
                if (const Json *kinds = data.Find("accelerators"); kinds && kinds->IsObject())
                {
                    for (const auto &[kind, count] : kinds->AsObject())
                    {
                        accelerators += (accelerators.empty() ? "" : ",") + kind + ":" + Scalar(count);
                    }
                }
                return std::format("region={} cpu={} memory={}{}", Scalar(data["region"]), Scalar(data["cpu"]),
                                   Scalar(data["memory"]),
                                   accelerators.empty() ? std::string() : " accelerators=" + accelerators);
            }
            if (entry.m_Collection == NETWORK)
            {
                std::string kind = data.GetString("kind");
                if (kind == "latency")
                {
                    return std::format("latency {}-{} {} ms", data.GetString("from"), data.GetString("to"),
                                       Scalar(data["ms"]));
                }
                if (kind == "density")
                {
                    return "density " + data.GetString("topic") + Rates(data["rates"]) + " msg/s";
                }
                if (kind == "consumers")
                {
                    return "consumers " + data.GetString("topic") + Rates(data["counts"]);
                }
                if (kind == "shard")
                {
                    std::vector<std::string> topics;
                    for (const Json &topic : data["topics"].IsArray() ? data["topics"].AsArray() : Json::Array{})
                    {
                        topics.push_back(Scalar(topic));
                    }
                    return "shard owner " + data.GetString("region") + " of " + Join(topics, ", ");
                }
                if (entry.m_Id == "model")
                {
                    std::vector<std::string> regions;
                    for (const Json &region : data["regions"].IsArray() ? data["regions"].AsArray() : Json::Array{})
                    {
                        regions.push_back(Scalar(region));
                    }
                    return "network model, regions " + Join(regions, ", ");
                }
            }
            if (entry.m_Collection == MEMORY)
            {
                return Trim(data.GetString("agent") + " " + data.GetString("kind") + " " + data.GetString("service")) +
                       ": " + Scalar(data["data"]);
            }
            if (entry.m_Collection == CONSOLIDATED)
            {
                return std::format("{} {} count={} successes={} failures={} last={}", data.GetString("service"),
                                   data.GetString("kind"), Scalar(data["count"]), Scalar(data["successes"]),
                                   Scalar(data["failures"]), data.GetString("lastOutcome", "-"));
            }
            if (entry.m_Collection == OPERATOR)
            {
                return data.GetString("text");
            }
            if (entry.m_Collection == TOOLS)
            {
                std::vector<std::string> capabilities;
                if (const Json *list = data.Find("capabilities"); list && list->IsArray())
                {
                    for (const Json &capability : list->AsArray())
                    {
                        capabilities.push_back(Scalar(capability));
                    }
                }
                return data.GetString("description") +
                       (capabilities.empty() ? std::string() : " [" + Join(capabilities, ", ") + "]");
            }
            if (entry.m_Collection == OBSERVATIONS)
            {
                return Trim(data.GetString("source") + " " + data.GetString("kind")) + ": " + Scalar(data["data"]);
            }
            return Scalar(data);
        }

        std::string Summary(const std::vector<Entry> &entries)
        {
            std::string summary;
            for (size_t i = 0; i < entries.size(); ++i)
            {
                std::string line =
                    Detail::Truncate(entries[i].m_Collection + "/" + entries[i].m_Id + ": " + Describe(entries[i]),
                                     LINE_BYTES) +
                    "\n";
                std::string more = std::format("... ({} more)\n", entries.size() - i);
                // keep room for the "more" marker unless this is the last line
                size_t reserve =
                    i + 1 < entries.size() ? std::format("... ({} more)\n", entries.size() - i - 1).size() : 0;
                if (summary.size() + line.size() + reserve > SUMMARY_BYTES)
                {
                    if (summary.size() + more.size() <= SUMMARY_BYTES)
                    {
                        summary += more;
                    }
                    break;
                }
                summary += line;
            }
            return summary;
        }
    } // namespace

    Json RetrievalResult::ToJson() const
    {
        Json entries = Json::MakeArray();
        for (const Entry &entry : m_Entries)
        {
            entries.Push(entry.ToJson());
        }
        return Json(Json::Object{{"context", m_Context},
                                 {"query", m_Query.ToJson()},
                                 {"entries", std::move(entries)},
                                 {"aggregates", m_Aggregates},
                                 {"summary", m_Summary},
                                 {"translator", m_Translator}});
    }

    Comas::Comas(KnowledgePlane &plane, const Intelligence::ProviderSet *providers,
                 std::shared_ptr<KnowledgeSource> source)
        : m_Plane(plane), m_Providers(providers),
          m_Source(source ? std::move(source) : std::make_shared<LocalSource>(plane))
    {
    }

    RetrievalResult Comas::Retrieve(const RetrievalRequest &request)
    {
        Json context = Augment(request.m_Intent);
        if (!request.m_Agent.empty())
        {
            context.Set("agent", request.m_Agent);
        }
        std::string translator;
        Query query = Translate(request.m_Intent, context, &translator);
        if (request.m_Limit > 0 && (query.m_Limit == 0 || query.m_Limit > request.m_Limit))
        {
            query.m_Limit = request.m_Limit;
        }
        std::vector<Entry> entries = Execute(query);
        return Contextualise(request, std::move(context), std::move(query), std::move(entries), std::move(translator));
    }

    Json Comas::Augment(std::string_view intent, size_t limit) const
    {
        limit = std::clamp<size_t>(limit, 1, MAX_LIMIT);
        Json collections = Json::MakeObject();
        for (const std::string &name : m_Plane.Collections())
        {
            collections.Set(name, static_cast<uint64_t>(m_Plane.Size(name)));
        }
        std::set<std::string> regions;
        std::set<std::string> accelerators;
        for (const Entry &entry : m_Plane.Scan(RESOURCES))
        {
            if (std::string region = entry.m_Data.GetString("region"); !region.empty())
            {
                regions.insert(region);
            }
            if (const Json *kinds = entry.m_Data.Find("accelerators"); kinds && kinds->IsObject())
            {
                for (const auto &[kind, count] : kinds->AsObject())
                {
                    if (count.IsNumber() && count.AsNumber() > 0)
                    {
                        accelerators.insert(kind);
                    }
                }
            }
        }
        for (const std::string &region : m_Plane.Network().m_Regions)
        {
            regions.insert(region);
        }
        std::set<std::string> services;
        for (const char *collection : {MEMORY, CONSOLIDATED, OBSERVATIONS})
        {
            for (const Entry &entry : m_Plane.Scan(collection))
            {
                if (std::string service = entry.m_Data.GetString("service");
                    !service.empty() && services.size() < MAX_SERVICES)
                {
                    services.insert(service);
                }
            }
        }
        Query relevant;
        relevant.m_Collections = {OPERATOR};
        relevant.m_Text = {std::string(intent)};
        relevant.m_Limit = limit;
        Json operatorContext = Json::MakeArray();
        for (const Entry &entry : Evaluate(relevant, m_Plane.Scan(OPERATOR)))
        {
            Json tags = Json::MakeArray();
            for (const std::string &tag : entry.m_Tags)
            {
                tags.Push(Json(tag));
            }
            operatorContext.Push(Json(Json::Object{{"id", entry.m_Id},
                                                   {"text", Detail::Truncate(Describe(entry), CONTEXT_TEXT_BYTES)},
                                                   {"tags", std::move(tags)},
                                                   {"score", entry.m_Score}}));
        }
        std::vector<Entry> memory = m_Plane.Scan(MEMORY);
        std::sort(memory.begin(), memory.end(), [](const Entry &a, const Entry &b)
                  { return a.m_Updated != b.m_Updated ? a.m_Updated > b.m_Updated : a.m_Id > b.m_Id; });
        Json recent = Json::MakeArray();
        for (size_t i = 0; i < memory.size() && i < limit; ++i)
        {
            const Entry &entry = memory[i];
            recent.Push(Json(Json::Object{{"id", entry.m_Id},
                                          {"agent", entry.m_Data.GetString("agent")},
                                          {"kind", entry.m_Data.GetString("kind")},
                                          {"service", entry.m_Data.GetString("service")},
                                          {"outcome", Outcome(entry.m_Data)},
                                          {"updated", entry.m_Updated}}));
        }
        auto list = [](const std::set<std::string> &values)
        {
            Json array = Json::MakeArray();
            for (const std::string &value : values)
            {
                array.Push(Json(value));
            }
            return array;
        };
        return Json(Json::Object{{"intent", std::string(intent)},
                                 {"collections", std::move(collections)},
                                 {"regions", list(regions)},
                                 {"accelerators", list(accelerators)},
                                 {"services", list(services)},
                                 {"operator", std::move(operatorContext)},
                                 {"recent", std::move(recent)}});
    }

    std::vector<Entry> Comas::Execute(const Query &query) const
    {
        auto fetch = [](KnowledgeSource &source, const Query &request, const char *what)
        {
            try
            {
                return source.Execute(request);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("comas: {} knowledge source failed: {}", what, error.what());
                return std::vector<Entry>();
            }
        };
        if (dynamic_cast<const LocalSource *>(m_Source.get()))
        {
            return fetch(*m_Source, query, "local");
        }
        // an external (MCP) source holds the operator-provided knowledge; what CoMaS accumulates at run time (agents
        // memory, observations, consolidated experience, updated network knowledge) lives in this plane and is
        // merged in, so the retrieval R' sees the same experience as the augmentation c
        Query full = query;
        full.m_Select.clear(); // projected once, after the merge
        Query local = full;
        local.m_Collections.clear();
        for (const char *collection : {MEMORY, OBSERVATIONS, CONSOLIDATED, NETWORK})
        {
            if (query.m_Collections.empty() || std::find(query.m_Collections.begin(), query.m_Collections.end(),
                                                         collection) != query.m_Collections.end())
            {
                local.m_Collections.push_back(collection);
            }
        }
        LocalSource plane(m_Plane);
        std::vector<Entry> live = local.m_Collections.empty() ? std::vector<Entry>() : fetch(plane, local, "local");
        if (live.empty())
        {
            return fetch(*m_Source, query, "external");
        }
        std::vector<Entry> entries = fetch(*m_Source, full, "external");
        // the local entry wins over an external copy of the same (collection, id)
        std::set<std::pair<std::string, std::string>> ids;
        for (const Entry &entry : live)
        {
            ids.insert({entry.m_Collection, entry.m_Id});
        }
        std::erase_if(entries, [&](const Entry &entry) { return ids.count({entry.m_Collection, entry.m_Id}) > 0; });
        entries.insert(entries.end(), std::make_move_iterator(live.begin()), std::make_move_iterator(live.end()));
        // both sides are already filtered: one shared ordering, limit and projection (scores recomputed alike)
        Query merge = query;
        merge.m_Collections.clear();
        merge.m_Where.clear();
        return Evaluate(merge, std::move(entries));
    }

    RetrievalResult Comas::Contextualise(const RetrievalRequest &request, Json context, Query query,
                                         std::vector<Entry> entries, std::string translator) const
    {
        RetrievalResult result;
        size_t limit = request.m_Limit ? request.m_Limit : (query.m_Limit ? query.m_Limit : 20);
        limit = std::min(limit, MAX_LIMIT);
        // relevance filter only when nothing but keywords selected the entries
        bool dropIrrelevant = !Detail::Keywords(Join(query.m_Text, " ")).empty() && query.m_Where.empty();
        std::set<std::pair<std::string, std::string>> seen;
        for (Entry &entry : entries)
        {
            if (result.m_Entries.size() >= limit)
            {
                break;
            }
            if (dropIrrelevant && entry.m_Score <= 0)
            {
                continue;
            }
            if (!seen.insert({entry.m_Collection, entry.m_Id}).second)
            {
                continue;
            }
            result.m_Entries.push_back(std::move(entry));
        }
        std::map<std::string, uint64_t> counts;
        std::map<std::string, uint64_t> regions;
        std::map<std::string, uint64_t> accelerators;
        for (const Entry &entry : result.m_Entries)
        {
            ++counts[entry.m_Collection];
            if (entry.m_Collection != RESOURCES)
            {
                continue;
            }
            if (std::string region = entry.m_Data.GetString("region"); !region.empty())
            {
                ++regions[region];
            }
            if (const Json *kinds = entry.m_Data.Find("accelerators"); kinds && kinds->IsObject())
            {
                for (const auto &[kind, count] : kinds->AsObject())
                {
                    if (count.IsNumber() && count.AsNumber() > 0)
                    {
                        accelerators[kind] += static_cast<uint64_t>(count.AsNumber());
                    }
                }
            }
        }
        auto object = [](const std::map<std::string, uint64_t> &values)
        {
            Json json = Json::MakeObject();
            for (const auto &[key, value] : values)
            {
                json.Set(key, value);
            }
            return json;
        };
        result.m_Aggregates = Json(Json::Object{{"total", static_cast<uint64_t>(result.m_Entries.size())},
                                                {"count", object(counts)},
                                                {"regions", object(regions)},
                                                {"accelerators", object(accelerators)}});
        result.m_Summary = Summary(result.m_Entries);
        result.m_Context = std::move(context);
        result.m_Query = std::move(query);
        result.m_Translator = std::move(translator);
        return result;
    }

    void Comas::Remember(std::string_view agent, std::string_view kind, std::string_view service, Json data)
    {
        std::scoped_lock lock(m_Mutex);
        uint64_t now = NowMillis();
        Entry entry;
        entry.m_Collection = MEMORY;
        entry.m_Id = SequenceId(now);
        entry.m_Data = Json(Json::Object{{"agent", std::string(agent)},
                                         {"kind", std::string(kind)},
                                         {"service", std::string(service)},
                                         {"data", std::move(data)}});
        if (std::string outcome = Outcome(entry.m_Data); !outcome.empty())
        {
            entry.m_Data.Set("outcome", outcome);
        }
        AddTag(entry.m_Tags, agent);
        AddTag(entry.m_Tags, kind);
        AddTag(entry.m_Tags, service);
        entry.m_Created = entry.m_Updated = now;
        m_Plane.Put(std::move(entry));
    }

    void Comas::Observe(std::string_view source, std::string_view kind, Json data)
    {
        uint64_t now = NowMillis();
        Entry entry;
        entry.m_Collection = OBSERVATIONS;
        entry.m_Id = SequenceId(now);
        entry.m_Data = Json(Json::Object{{"source", std::string(source)}, {"kind", std::string(kind)}});
        // identifying members are hoisted so conditions (service, region, topic) can select them
        for (const char *member : {"service", "instance", "node", "region", "topic"})
        {
            if (const Json *value = data.Find(member); value && value->IsString())
            {
                entry.m_Data.Set(member, *value);
            }
        }
        entry.m_Data.Set("data", std::move(data));
        AddTag(entry.m_Tags, source);
        AddTag(entry.m_Tags, kind);
        entry.m_Created = entry.m_Updated = now;
        m_Plane.Put(std::move(entry));
    }

    size_t Comas::Consolidate(uint64_t nowMillis, uint64_t maxAgeMillis)
    {
        std::scoped_lock lock(m_Mutex);
        std::map<std::pair<std::string, std::string>, std::vector<Entry>> groups;
        for (Entry &entry : m_Plane.Scan(MEMORY))
        {
            if (entry.m_Updated > nowMillis || nowMillis - entry.m_Updated < maxAgeMillis)
            {
                continue;
            }
            auto key = std::make_pair(entry.m_Data.GetString("service"), entry.m_Data.GetString("kind"));
            groups[key].push_back(std::move(entry));
        }
        for (auto &[key, group] : groups)
        {
            const auto &[service, kind] = key;
            std::sort(group.begin(), group.end(), [](const Entry &a, const Entry &b)
                      { return a.m_Updated != b.m_Updated ? a.m_Updated < b.m_Updated : a.m_Id < b.m_Id; });
            std::string id = (service.empty() ? "-" : service) + "/" + (kind.empty() ? "-" : kind);
            std::optional<Entry> existing = m_Plane.Get(CONSOLIDATED, id);
            Json previous = existing ? existing->m_Data : Json();
            int64_t count = previous.GetInt("count");
            int64_t successes = previous.GetInt("successes");
            int64_t failures = previous.GetInt("failures");
            uint64_t firstAt =
                existing ? static_cast<uint64_t>(previous.GetInt("firstAt", INT64_MAX)) : group.front().m_Updated;
            uint64_t lastAt = static_cast<uint64_t>(previous.GetInt("lastAt"));
            std::string lastOutcome = previous.GetString("lastOutcome");
            Json last = previous["last"];
            std::vector<std::string> agents;
            if (const Json *known = previous.Find("agents"); known && known->IsArray())
            {
                for (const Json &agent : known->AsArray())
                {
                    AddTag(agents, Scalar(agent));
                }
            }
            for (const Entry &entry : group)
            {
                std::string outcome = Outcome(entry.m_Data);
                ++count;
                successes += IsIn(outcome, SUCCESSES) ? 1 : 0;
                failures += IsIn(outcome, FAILURES) ? 1 : 0;
                firstAt = std::min(firstAt, entry.m_Updated);
                AddTag(agents, entry.m_Data.GetString("agent"));
                if (entry.m_Updated >= lastAt)
                {
                    lastAt = entry.m_Updated;
                    lastOutcome = outcome;
                    last = entry.m_Data["data"];
                }
            }
            Json agentList = Json::MakeArray();
            for (const std::string &agent : agents)
            {
                agentList.Push(Json(agent));
            }
            Entry consolidated;
            consolidated.m_Collection = CONSOLIDATED;
            consolidated.m_Id = id;
            consolidated.m_Data = Json(Json::Object{{"service", service},
                                                    {"kind", kind},
                                                    {"count", count},
                                                    {"successes", successes},
                                                    {"failures", failures},
                                                    {"lastOutcome", lastOutcome},
                                                    {"lastAt", lastAt},
                                                    {"firstAt", firstAt},
                                                    {"agents", std::move(agentList)},
                                                    {"last", std::move(last)}});
            AddTag(consolidated.m_Tags, service);
            AddTag(consolidated.m_Tags, kind);
            AddTag(consolidated.m_Tags, "consolidated");
            consolidated.m_Created = existing ? existing->m_Created : nowMillis;
            consolidated.m_Updated = std::max<uint64_t>(nowMillis, 1);
            m_Plane.Put(std::move(consolidated));
            for (const Entry &entry : group)
            {
                m_Plane.Erase(MEMORY, entry.m_Id);
            }
        }
        return groups.size();
    }

    KnowledgePlane &Comas::Plane() noexcept
    {
        return m_Plane;
    }
} // namespace eOps::Context
