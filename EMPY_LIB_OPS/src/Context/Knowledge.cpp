#include <eOps/Context/Knowledge.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Storage.h>
#include <eCore/Common/Stdout.h>
#include <eCore/Common/YAML.h>
#include <set>
#include <array>
#include <map>
#include <mutex>
#include <algorithm>
#include <stdexcept>
#include "../Common/JsonRead.h"

namespace eOps::Context
{
    namespace
    {
        constexpr size_t MAX_COLLECTIONS = 64;
        constexpr size_t MAX_COLLECTION_NAME = 64;
        constexpr size_t MAX_KEY_BYTES = 255; // eCore::MMQP::FileStore key limit
        constexpr std::string_view NETWORK_MODEL = "model";
        constexpr std::array<std::string_view, 4> NETWORK_DERIVED = {"latency/", "topics/", "consumers/", "shard/"};

        bool ValidCollection(std::string_view name)
        {
            return !name.empty() && name.size() <= MAX_COLLECTION_NAME &&
                   std::all_of(name.begin(), name.end(), [](char c)
                               { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_'; });
        }

        bool ValidId(std::string_view id)
        {
            return !id.empty() &&
                   std::none_of(id.begin(), id.end(), [](char c) { return static_cast<uint8_t>(c) < 0x20; });
        }

        std::string Key(std::string_view collection, std::string_view id)
        {
            return std::string(collection) + "/" + std::string(id);
        }

        bool Pinned(std::string_view collection, std::string_view id)
        {
            return collection == NETWORK && id == NETWORK_MODEL;
        }

        bool Derived(std::string_view id)
        {
            return std::any_of(NETWORK_DERIVED.begin(), NETWORK_DERIVED.end(),
                               [id](std::string_view prefix) { return id.starts_with(prefix); });
        }

        // persisted form: the entry without its transient score
        std::string Serialise(const Entry &entry)
        {
            Json json = entry.ToJson();
            json.Erase("score");
            return json.Dump();
        }

        using eOps::Detail::Strings;

        std::vector<std::string> StringList(const Json &value, const std::string &path)
        {
            std::vector<std::string> items;
            if (value.IsNull())
            {
                return items;
            }
            if (!value.IsArray())
            {
                throw eCore::ConfigError(path + ": expected a list of strings");
            }
            for (const Json &item : value.AsArray())
            {
                if (!item.IsString())
                {
                    throw eCore::ConfigError(path + ": expected a list of strings");
                }
                items.push_back(item.AsString());
            }
            return items;
        }

        void AddTag(std::vector<std::string> &tags, const std::string &tag)
        {
            if (!tag.empty() && std::find(tags.begin(), tags.end(), tag) == tags.end())
            {
                tags.push_back(tag);
            }
        }

        // NETWORK entries derived from the model so links, densities and shards are retrievable
        std::map<std::string, std::pair<Json, std::vector<std::string>>> DeriveNetwork(
            const Topology::NetworkModel &model)
        {
            std::map<std::string, std::pair<Json, std::vector<std::string>>> derived;
            auto add = [&derived](std::string id, Json data, std::vector<std::string> tags)
            {
                if (Key(NETWORK, id).size() <= MAX_KEY_BYTES)
                {
                    derived[std::move(id)] = {std::move(data), std::move(tags)};
                }
            };
            for (const auto &[from, row] : model.m_LatencyMs)
            {
                for (const auto &[to, ms] : row)
                {
                    add("latency/" + from + "/" + to,
                        Json(Json::Object{{"kind", "latency"},
                                          {"from", from},
                                          {"to", to},
                                          {"region", Strings({from, to})},
                                          {"ms", ms}}),
                        {"latency", from, to});
                }
            }
            auto rates = [&add](const std::map<std::string, std::map<std::string, double>> &table, const char *kind,
                                const char *prefix, const char *member)
            {
                for (const auto &[filter, regions] : table)
                {
                    Json values = Json::MakeObject();
                    std::vector<std::string> active;
                    double total = 0;
                    for (const auto &[region, rate] : regions)
                    {
                        values.Set(region, rate);
                        total += rate;
                        if (rate > 0)
                        {
                            active.push_back(region);
                        }
                    }
                    std::vector<std::string> tags = {kind};
                    tags.insert(tags.end(), active.begin(), active.end());
                    add(prefix + filter,
                        Json(Json::Object{{"kind", kind},
                                          {"topic", filter},
                                          {member, std::move(values)},
                                          {"region", Strings(active)},
                                          {"total", total}}),
                        std::move(tags));
                }
            };
            rates(model.m_TopicDensity, "density", "topics/", "rates");
            rates(model.m_ConsumerDensity, "consumers", "consumers/", "counts");
            for (const auto &[region, filters] : model.m_ShardOwners)
            {
                add("shard/" + region,
                    Json(Json::Object{{"kind", "shard"}, {"region", region}, {"topics", Strings(filters)}}),
                    {"shard", region});
            }
            return derived;
        }
    } // namespace

    Json Entry::ToJson() const
    {
        return Json(Json::Object{{"id", m_Id},
                                 {"collection", m_Collection},
                                 {"data", m_Data},
                                 {"tags", Strings(m_Tags)},
                                 {"created", m_Created},
                                 {"updated", m_Updated},
                                 {"score", m_Score}});
    }

    Entry Entry::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("knowledge entry must be an object");
        }
        Entry entry;
        const Json *id = json.Find("id");
        const Json *collection = json.Find("collection");
        if (!id || !id->IsString() || !collection || !collection->IsString())
        {
            throw JsonError("knowledge entry needs string 'id' and 'collection' members");
        }
        entry.m_Id = id->AsString();
        entry.m_Collection = collection->AsString();
        if (const Json *data = json.Find("data"))
        {
            entry.m_Data = *data;
        }
        if (const Json *tags = json.Find("tags"); tags && !tags->IsNull())
        {
            if (!tags->IsArray())
            {
                throw JsonError("knowledge entry 'tags' must be a list of strings");
            }
            for (const Json &tag : tags->AsArray())
            {
                if (!tag.IsString())
                {
                    throw JsonError("knowledge entry 'tags' must be a list of strings");
                }
                entry.m_Tags.push_back(tag.AsString());
            }
        }
        if (const Json *created = json.Find("created"); created && !created->IsNull())
        {
            entry.m_Created = created->AsUint();
        }
        if (const Json *updated = json.Find("updated"); updated && !updated->IsNull())
        {
            entry.m_Updated = updated->AsUint();
        }
        if (const Json *score = json.Find("score"); score && !score->IsNull())
        {
            entry.m_Score = score->AsNumber();
        }
        return entry;
    }

    struct KnowledgePlane::Impl
    {
        struct Slot
        {
            Entry m_Entry;
            size_t m_Bytes = 0;
        };
        struct Collection
        {
            std::map<std::string, Slot, std::less<>> m_Entries;
            std::set<std::pair<uint64_t, std::string>> m_ByAge; // (updated, id), oldest first
        };

        KnowledgeOptions m_Options;
        mutable std::mutex m_Mutex;
        std::map<std::string, Collection, std::less<>> m_Collections;
        size_t m_Bytes = 0;
        std::unique_ptr<eCore::MMQP::FileStore> m_Store;

        explicit Impl(KnowledgeOptions options) : m_Options(std::move(options))
        {
            if (m_Options.m_MaxEntries == 0 || m_Options.m_MaxBytes == 0)
            {
                throw std::invalid_argument("knowledge plane bounds must be positive");
            }
        }

        // ---- unlocked helpers (callers hold m_Mutex)

        const Slot *FindSlot(std::string_view collection, std::string_view id) const
        {
            auto group = m_Collections.find(collection);
            if (group == m_Collections.end())
            {
                return nullptr;
            }
            auto slot = group->second.m_Entries.find(id);
            return slot == group->second.m_Entries.end() ? nullptr : &slot->second;
        }

        void Remove(std::string_view collection, std::string_view id, bool persist)
        {
            auto group = m_Collections.find(collection);
            if (group == m_Collections.end())
            {
                return;
            }
            auto slot = group->second.m_Entries.find(id);
            if (slot == group->second.m_Entries.end())
            {
                return;
            }
            if (persist && m_Store)
            {
                m_Store->Erase(Key(collection, id));
            }
            m_Bytes -= slot->second.m_Bytes;
            group->second.m_ByAge.erase({slot->second.m_Entry.m_Updated, slot->first});
            group->second.m_Entries.erase(slot);
            if (group->second.m_Entries.empty())
            {
                m_Collections.erase(group);
            }
        }

        // oldest evictable entry of one collection other than `keep`
        const std::pair<uint64_t, std::string> *Oldest(const std::string &collection, const Collection &group,
                                                       std::string_view keep) const
        {
            for (const auto &aged : group.m_ByAge)
            {
                if (aged.second != keep && !Pinned(collection, aged.second))
                {
                    return &aged;
                }
            }
            return nullptr;
        }

        // drop the globally oldest evictable entry; false when none is left
        bool EvictOldest(std::string_view keepCollection, std::string_view keepId)
        {
            std::string collection;
            const std::pair<uint64_t, std::string> *victim = nullptr;
            for (const auto &[name, group] : m_Collections)
            {
                const auto *candidate = Oldest(name, group, name == keepCollection ? keepId : std::string_view{});
                if (candidate && (!victim || *candidate < *victim))
                {
                    victim = candidate;
                    collection = name;
                }
            }
            if (!victim)
            {
                return false;
            }
            std::string id = victim->second;
            Remove(collection, id, true);
            return true;
        }

        void Insert(Entry entry, bool persist)
        {
            if (!ValidCollection(entry.m_Collection))
            {
                throw std::invalid_argument("knowledge collection '" + entry.m_Collection +
                                            "' must be 1-64 characters of [a-z0-9_-]");
            }
            if (!ValidId(entry.m_Id) || Key(entry.m_Collection, entry.m_Id).size() > MAX_KEY_BYTES)
            {
                throw std::invalid_argument("knowledge id '" + entry.m_Id + "' is empty, too long or has controls");
            }
            const Slot *previous = FindSlot(entry.m_Collection, entry.m_Id);
            if (!previous && !m_Collections.contains(entry.m_Collection) && m_Collections.size() >= MAX_COLLECTIONS)
            {
                throw std::length_error("knowledge plane holds the maximum number of collections");
            }
            uint64_t now = NowMillis();
            if (entry.m_Updated == 0)
            {
                entry.m_Updated = now;
            }
            if (entry.m_Created == 0)
            {
                entry.m_Created = previous ? previous->m_Entry.m_Created : entry.m_Updated;
            }
            entry.m_Score = 0;
            std::string text = Serialise(entry); // throws JsonError for unrepresentable data
            if (text.size() > m_Options.m_MaxBytes)
            {
                throw std::length_error("knowledge entry exceeds the knowledge plane byte budget");
            }
            // byte budget: evict the globally oldest entries until the new value fits
            while (m_Bytes - (previous ? previous->m_Bytes : 0) + text.size() > m_Options.m_MaxBytes)
            {
                if (!EvictOldest(entry.m_Collection, entry.m_Id))
                {
                    throw std::length_error("knowledge entry does not fit the knowledge plane byte budget");
                }
                previous = FindSlot(entry.m_Collection, entry.m_Id);
            }
            if (persist && m_Store)
            {
                std::string key = Key(entry.m_Collection, entry.m_Id);
                m_Store->Put(std::move(key), eCore::MMQP::Buffer(text.begin(), text.end()));
            }
            auto &group = m_Collections[entry.m_Collection];
            if (auto slot = group.m_Entries.find(entry.m_Id); slot != group.m_Entries.end())
            {
                m_Bytes -= slot->second.m_Bytes;
                group.m_ByAge.erase({slot->second.m_Entry.m_Updated, slot->first});
                group.m_Entries.erase(slot);
            }
            std::string id = entry.m_Id;
            std::string collection = entry.m_Collection;
            group.m_ByAge.insert({entry.m_Updated, id});
            group.m_Entries[id] = Slot{std::move(entry), text.size()};
            m_Bytes += text.size();
            // entry budget per collection: evict the oldest other entries
            while (group.m_Entries.size() > m_Options.m_MaxEntries)
            {
                const auto *victim = Oldest(collection, group, id);
                if (!victim)
                {
                    break;
                }
                std::string evicted = victim->second;
                Remove(collection, evicted, true);
            }
        }

        std::optional<Topology::NetworkModel> StoredNetwork() const
        {
            const Slot *slot = FindSlot(NETWORK, NETWORK_MODEL);
            if (!slot)
            {
                return std::nullopt;
            }
            return Topology::NetworkModel::FromJson(slot->m_Entry.m_Data);
        }

        // model entry plus derived link/density/shard entries; unchanged entries keep their timestamps
        void WriteNetwork(const Topology::NetworkModel &model)
        {
            auto derived = DeriveNetwork(model);
            std::vector<std::string> stale;
            if (auto group = m_Collections.find(NETWORK); group != m_Collections.end())
            {
                for (const auto &[id, slot] : group->second.m_Entries)
                {
                    if (Derived(id) && !derived.contains(id))
                    {
                        stale.push_back(id);
                    }
                }
            }
            for (const std::string &id : stale)
            {
                Remove(NETWORK, id, true);
            }
            for (auto &[id, value] : derived)
            {
                const Slot *slot = FindSlot(NETWORK, id);
                if (slot && slot->m_Entry.m_Data == value.first && slot->m_Entry.m_Tags == value.second)
                {
                    continue;
                }
                Insert(Entry{id, NETWORK, std::move(value.first), std::move(value.second), 0, 0, 0}, true);
            }
            Json data = model.ToJson();
            const Slot *slot = FindSlot(NETWORK, NETWORK_MODEL);
            if (!slot || !(slot->m_Entry.m_Data == data))
            {
                Insert(Entry{std::string(NETWORK_MODEL), NETWORK, std::move(data), {"network", "model"}, 0, 0, 0},
                       true);
            }
        }

        void Load()
        {
            std::vector<Entry> entries;
            for (const std::string &key : m_Store->Keys())
            {
                const eCore::MMQP::Buffer *value = m_Store->Find(key);
                try
                {
                    Entry entry = Entry::FromJson(FromBytes(eCore::ByteSpan(value->data(), value->size())));
                    if (key != Key(entry.m_Collection, entry.m_Id) || entry.m_Updated == 0 ||
                        !ValidCollection(entry.m_Collection) || !ValidId(entry.m_Id))
                    {
                        throw std::runtime_error("the key does not match a valid entry");
                    }
                    entries.push_back(std::move(entry));
                }
                catch (const std::exception &error)
                {
                    throw std::runtime_error("corrupt knowledge record '" + key + "': " + error.what());
                }
            }
            // oldest first, so tighter bounds keep the newest entries; evictions are erased from the store
            std::stable_sort(entries.begin(), entries.end(),
                             [](const Entry &a, const Entry &b) { return a.m_Updated < b.m_Updated; });
            for (Entry &entry : entries)
            {
                Insert(std::move(entry), false);
            }
        }
    };

    KnowledgePlane::KnowledgePlane(KnowledgeOptions options) : m_Impl(std::make_unique<Impl>(std::move(options)))
    {
        if (!m_Impl->m_Options.m_Directory.empty())
        {
            size_t perCollection = std::min<size_t>(m_Impl->m_Options.m_MaxEntries, SIZE_MAX / (2 * MAX_COLLECTIONS));
            size_t records = MAX_COLLECTIONS * perCollection + 1;
            m_Impl->m_Store = std::make_unique<eCore::MMQP::FileStore>(m_Impl->m_Options.m_Directory,
                                                                       m_Impl->m_Options.m_MaxBytes, records, true);
            m_Impl->Load();
        }
    }

    KnowledgePlane::~KnowledgePlane() = default;

    void KnowledgePlane::Put(Entry entry)
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        m_Impl->Insert(std::move(entry), true);
    }

    std::optional<Entry> KnowledgePlane::Get(std::string_view collection, std::string_view id) const
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        const Impl::Slot *slot = m_Impl->FindSlot(collection, id);
        return slot ? std::optional<Entry>(slot->m_Entry) : std::nullopt;
    }

    bool KnowledgePlane::Erase(std::string_view collection, std::string_view id)
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        if (!m_Impl->FindSlot(collection, id))
        {
            return false;
        }
        m_Impl->Remove(collection, id, true);
        return true;
    }

    std::vector<Entry> KnowledgePlane::Scan(std::string_view collection) const
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        std::vector<Entry> entries;
        auto group = m_Impl->m_Collections.find(collection);
        if (group == m_Impl->m_Collections.end())
        {
            return entries;
        }
        entries.reserve(group->second.m_Entries.size());
        for (const auto &[id, slot] : group->second.m_Entries)
        {
            entries.push_back(slot.m_Entry);
        }
        return entries;
    }

    std::vector<std::string> KnowledgePlane::Collections() const
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        std::vector<std::string> names;
        for (const auto &[name, group] : m_Impl->m_Collections)
        {
            names.push_back(name);
        }
        return names;
    }

    size_t KnowledgePlane::Size(std::string_view collection) const
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        auto group = m_Impl->m_Collections.find(collection);
        return group == m_Impl->m_Collections.end() ? 0 : group->second.m_Entries.size();
    }

    void KnowledgePlane::Seed(const eCore::YamlNode &root)
    {
        if (root.IsNull())
        {
            return;
        }
        if (!root.IsMap())
        {
            throw eCore::ConfigError(root.Path() + ": knowledge seed must be a mapping");
        }
        static const std::vector<std::string> s_Sections = {"resources", "network", "tools", "operator", "memory"};
        for (const std::string &key : root.Keys())
        {
            if (std::find(s_Sections.begin(), s_Sections.end(), key) == s_Sections.end())
            {
                throw eCore::ConfigError(root.Path() + "." + key +
                                         ": unknown knowledge section (resources, network, tools, operator, memory)");
            }
        }
        // parse and validate everything first so a bad seed changes nothing
        std::vector<Entry> entries;
        std::optional<Topology::NetworkModel> network;
        auto wrap = [](const std::string &path, auto &&parse)
        {
            try
            {
                return parse();
            }
            catch (const eCore::ConfigError &)
            {
                throw;
            }
            catch (const std::exception &error)
            {
                throw eCore::ConfigError(path + ": " + error.what());
            }
        };
        std::set<std::string> nodes;
        for (const eCore::YamlNode &item : root["resources"].Items())
        {
            Topology::Node node = wrap(item.Path(), [&] { return Topology::Node::FromJson(FromYaml(item)); });
            if (!nodes.insert(node.m_Id).second)
            {
                throw eCore::ConfigError(item.Path() + ".id: duplicate node '" + node.m_Id + "'");
            }
            Json data = node.ToJson();
            data.Set("memoryBytes", node.m_Memory);
            std::vector<std::string> tags = {"node"};
            AddTag(tags, node.m_Region);
            for (const auto &[kind, count] : node.m_Accelerators)
            {
                if (count > 0)
                {
                    AddTag(tags, kind);
                }
            }
            entries.push_back(Entry{node.m_Id, RESOURCES, std::move(data), std::move(tags), 0, 0, 0});
        }
        if (root.Has("network"))
        {
            network = wrap(root["network"].Path(),
                           [&] { return Topology::NetworkModel::FromJson(FromYaml(root["network"])); });
        }
        auto records = [&](const char *section, const char *collection, bool requireId)
        {
            std::set<std::string> seen;
            for (const eCore::YamlNode &item : root[section].Items())
            {
                Json data = wrap(item.Path(), [&] { return FromYaml(item); });
                if (!data.IsObject())
                {
                    throw eCore::ConfigError(item.Path() + ": expected a mapping");
                }
                const Json *id = data.Find("id");
                if (id && !id->IsString())
                {
                    throw eCore::ConfigError(item.Path() + ".id: expected a string");
                }
                std::string name = id ? id->AsString() : std::string();
                if (name.empty())
                {
                    if (requireId)
                    {
                        throw eCore::ConfigError(item.Path() + ".id: required");
                    }
                    name = NewId(collection);
                }
                if (!ValidId(name) || Key(collection, name).size() > MAX_KEY_BYTES)
                {
                    throw eCore::ConfigError(item.Path() + ".id: invalid identifier '" + name + "'");
                }
                if (!seen.insert(name).second)
                {
                    throw eCore::ConfigError(item.Path() + ".id: duplicate '" + name + "'");
                }
                std::vector<std::string> tags = StringList(data["tags"], item.Path() + ".tags");
                data.Erase("id");
                data.Erase("tags");
                if (std::string_view(collection) == TOOLS)
                {
                    AddTag(tags, data.GetString("kind"));
                    for (const std::string &capability :
                         StringList(data["capabilities"], item.Path() + ".capabilities"))
                    {
                        AddTag(tags, capability);
                    }
                }
                else if (std::string_view(collection) == OPERATOR)
                {
                    const Json *text = data.Find("text");
                    if (!text || !text->IsString() || Trim(text->AsString()).empty())
                    {
                        throw eCore::ConfigError(item.Path() + ".text: operator context needs a text");
                    }
                }
                else
                {
                    for (const char *member : {"agent", "kind", "service"})
                    {
                        AddTag(tags, data.GetString(member));
                    }
                }
                entries.push_back(Entry{name, collection, std::move(data), std::move(tags), 0, 0, 0});
            }
        };
        records("tools", TOOLS, true);
        records("operator", OPERATOR, true);
        records("memory", MEMORY, false);

        std::scoped_lock lock(m_Impl->m_Mutex);
        for (Entry &entry : entries)
        {
            m_Impl->Insert(std::move(entry), true);
        }
        if (network)
        {
            m_Impl->WriteNetwork(*network);
        }
    }

    Topology::Inventory KnowledgePlane::Inventory() const
    {
        Topology::Inventory inventory;
        for (const Entry &entry : Scan(RESOURCES))
        {
            try
            {
                Topology::Node node = Topology::Node::FromJson(entry.m_Data);
                if (!inventory.Find(node.m_Id))
                {
                    inventory.m_Nodes.push_back(std::move(node));
                }
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("knowledge: resource '{}' is not a valid node: {}", entry.m_Id, error.what());
            }
        }
        return inventory;
    }

    Topology::NetworkModel KnowledgePlane::Network() const
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        try
        {
            return m_Impl->StoredNetwork().value_or(Topology::NetworkModel{});
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("knowledge: stored network model is invalid: {}", error.what());
            return {};
        }
    }

    void KnowledgePlane::UpdateNetwork(const std::function<void(Topology::NetworkModel &)> &update)
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        Topology::NetworkModel model;
        try
        {
            model = m_Impl->StoredNetwork().value_or(Topology::NetworkModel{});
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("knowledge: replacing an invalid stored network model: {}", error.what());
        }
        update(model); // works on a copy: an exception leaves the plane unchanged
        m_Impl->WriteNetwork(model);
    }
} // namespace eOps::Context
