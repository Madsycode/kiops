#include <eOps/Topology/Network.h>
#include <eOps/Common/TopicAlgebra.h>
#include <cmath>
#include <algorithm>

namespace eOps::Topology
{
    namespace
    {
        constexpr double SAME_REGION_MS = 0.5;
        constexpr double UNKNOWN_MS = 50.0;

        using Table = std::map<std::string, std::map<std::string, double>>;

        double NonNegative(const Json &value, const std::string &field)
        {
            if (!value.IsNumber())
            {
                throw JsonError("network " + field + " must be a number");
            }
            double number = value.AsNumber();
            if (!std::isfinite(number) || number < 0)
            {
                throw JsonError("network " + field + " must be a non-negative number");
            }
            return number;
        }

        const Json *Section(const Json &json, std::string_view key)
        {
            const Json *section = json.Find(key);
            if (!section || section->IsNull())
            {
                return nullptr;
            }
            if (!section->IsObject())
            {
                throw JsonError("network " + std::string(key) + " must be a mapping");
            }
            return section;
        }

        // {outer: {inner: number}}
        Table ReadTable(const Json &json, std::string_view key)
        {
            Table table;
            const Json *section = Section(json, key);
            if (!section)
            {
                return table;
            }
            for (const auto &[outer, row] : section->AsObject())
            {
                if (!row.IsObject())
                {
                    throw JsonError("network " + std::string(key) + "." + outer + " must be a mapping");
                }
                auto &entries = table[outer];
                for (const auto &[inner, value] : row.AsObject())
                {
                    entries[inner] = NonNegative(value, std::string(key) + "." + outer + "." + inner);
                }
            }
            return table;
        }

        Json WriteTable(const Table &table)
        {
            Json object = Json::MakeObject();
            for (const auto &[outer, row] : table)
            {
                Json entries = Json::MakeObject();
                for (const auto &[inner, value] : row)
                {
                    entries.Set(inner, value);
                }
                object.Set(outer, std::move(entries));
            }
            return object;
        }

        double Sum(const Table &table, std::string_view topic, std::string_view region)
        {
            double total = 0;
            for (const auto &[filter, regions] : table)
            {
                if (!FiltersOverlap(filter, topic))
                {
                    continue;
                }
                auto entry = regions.find(std::string(region));
                if (entry != regions.end())
                {
                    total += entry->second;
                }
            }
            return total;
        }

        const double *Lookup(const Table &table, std::string_view from, std::string_view to)
        {
            auto row = table.find(std::string(from));
            if (row == table.end())
            {
                return nullptr;
            }
            auto entry = row->second.find(std::string(to));
            return entry == row->second.end() ? nullptr : &entry->second;
        }
    } // namespace

    double NetworkModel::Latency(std::string_view a, std::string_view b) const
    {
        if (const double *forward = Lookup(m_LatencyMs, a, b))
        {
            return *forward;
        }
        if (const double *backward = Lookup(m_LatencyMs, b, a))
        {
            return *backward;
        }
        return a == b ? SAME_REGION_MS : UNKNOWN_MS;
    }

    double NetworkModel::Density(std::string_view topic, std::string_view region) const
    {
        return Sum(m_TopicDensity, topic, region);
    }

    double NetworkModel::Consumers(std::string_view topic, std::string_view region) const
    {
        return Sum(m_ConsumerDensity, topic, region);
    }

    bool NetworkModel::OwnsShard(std::string_view region, std::string_view topic) const
    {
        auto owned = m_ShardOwners.find(std::string(region));
        if (owned == m_ShardOwners.end())
        {
            return false;
        }
        return std::any_of(owned->second.begin(), owned->second.end(),
                           [&](const std::string &filter) { return FiltersOverlap(filter, topic); });
    }

    void NetworkModel::ApplyLocality(std::string_view region, const Json &topics)
    {
        if (region.empty())
        {
            throw JsonError("locality report without a region");
        }
        if (!topics.IsObject())
        {
            throw JsonError("locality topics must be a mapping of topic to rate");
        }
        // validate the whole report first so a malformed one changes nothing
        std::vector<std::pair<std::string, double>> rates;
        for (const auto &[topic, rate] : topics.AsObject())
        {
            if (!ValidTopic(topic, true))
            {
                throw JsonError("locality topic '" + topic + "' is not a valid MMQP topic or filter");
            }
            rates.emplace_back(topic, NonNegative(rate, "locality rate of '" + topic + "'"));
        }
        std::string name(region);
        for (auto &[topic, rate] : rates)
        {
            m_TopicDensity[topic][name] = rate;
        }
        if (std::find(m_Regions.begin(), m_Regions.end(), name) == m_Regions.end())
        {
            m_Regions.push_back(name);
        }
    }

    Json NetworkModel::ToJson() const
    {
        Json regions = Json::MakeArray();
        for (const std::string &region : m_Regions)
        {
            regions.Push(Json(region));
        }
        Json shards = Json::MakeObject();
        for (const auto &[region, filters] : m_ShardOwners)
        {
            Json list = Json::MakeArray();
            for (const std::string &filter : filters)
            {
                list.Push(Json(filter));
            }
            shards.Set(region, std::move(list));
        }
        return Json(Json::Object{{"regions", std::move(regions)},
                                 {"latency", WriteTable(m_LatencyMs)},
                                 {"topics", WriteTable(m_TopicDensity)},
                                 {"consumers", WriteTable(m_ConsumerDensity)},
                                 {"shards", std::move(shards)}});
    }

    NetworkModel NetworkModel::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("network model must be a mapping");
        }
        NetworkModel model;
        if (const Json *regions = json.Find("regions"); regions && !regions->IsNull())
        {
            if (!regions->IsArray())
            {
                throw JsonError("network regions must be a list");
            }
            for (const Json &region : regions->AsArray())
            {
                const std::string &name = region.AsString();
                if (std::find(model.m_Regions.begin(), model.m_Regions.end(), name) == model.m_Regions.end())
                {
                    model.m_Regions.push_back(name);
                }
            }
        }
        model.m_LatencyMs = ReadTable(json, "latency");
        model.m_TopicDensity = ReadTable(json, "topics");
        model.m_ConsumerDensity = ReadTable(json, "consumers");
        for (const auto *table : {&model.m_TopicDensity, &model.m_ConsumerDensity})
        {
            for (const auto &[filter, regions] : *table)
            {
                if (!ValidTopic(filter, true))
                {
                    throw JsonError("network filter '" + filter + "' is not a valid MMQP topic filter");
                }
            }
        }
        if (const Json *shards = Section(json, "shards"))
        {
            for (const auto &[region, filters] : shards->AsObject())
            {
                if (!filters.IsArray())
                {
                    throw JsonError("network shards." + region + " must be a list");
                }
                auto &owned = model.m_ShardOwners[region];
                for (const Json &filter : filters.AsArray())
                {
                    if (!ValidTopic(filter.AsString(), true))
                    {
                        throw JsonError("network shard filter '" + filter.AsString() + "' is not valid");
                    }
                    owned.push_back(filter.AsString());
                }
            }
        }
        return model;
    }
} // namespace eOps::Topology
