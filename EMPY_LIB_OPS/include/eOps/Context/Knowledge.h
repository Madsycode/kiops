#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <functional>
#include <filesystem>
#include <string_view>
#include "../Common/Json.h"
#include "../Topology/Inventory.h"
#include "../Topology/Network.h"

// CoMaS knowledge plane (thesis §2.1.4): Agents Memory, Tool/Skill Registry, Resource Registry,
// Network Knowledge, operator-provided context, operational observations and consolidated
// experience, held as bounded per-collection entry maps with optional journaled persistence

namespace eOps::Context
{
    struct EMPY_OPS_API Entry
    {
        std::string m_Id;
        std::string m_Collection;
        Json m_Data;
        std::vector<std::string> m_Tags;
        uint64_t m_Created = 0; // Unix ms; Put fills 0 with now (or keeps the existing entry's value)
        uint64_t m_Updated = 0; // Unix ms; Put fills 0 with now
        double m_Score = 0;     // retrieval relevance (not persisted)

        Json ToJson() const; // {id, collection, data, tags[], created, updated, score}
        static Entry FromJson(const Json &json);
    };

    inline constexpr const char *MEMORY = "memory", *TOOLS = "tools", *RESOURCES = "resources", *NETWORK = "network",
                                *OPERATOR = "operator", *OBSERVATIONS = "observations", *CONSOLIDATED = "consolidated";

    struct KnowledgeOptions
    {
        std::filesystem::path m_Directory; // durable store (eCore::MMQP::FileStore, journaled) when set
        size_t m_MaxEntries = 4096;        // per collection; the oldest m_Updated is evicted first
        size_t m_MaxBytes = 64 << 20;      // persisted bytes budget
    };

    // thread-safe (internal mutex)
    class EMPY_OPS_API KnowledgePlane
    {
    public:
        explicit KnowledgePlane(KnowledgeOptions options = {}); // reloads persisted entries; throws on a corrupt store
        ~KnowledgePlane();
        KnowledgePlane(const KnowledgePlane &) = delete;
        KnowledgePlane &operator=(const KnowledgePlane &) = delete;

        void Put(Entry entry); // insert or replace by (collection, id); both must be non-empty
        std::optional<Entry> Get(std::string_view collection, std::string_view id) const;
        bool Erase(std::string_view collection, std::string_view id);
        std::vector<Entry> Scan(std::string_view collection) const; // ordered by id
        std::vector<std::string> Collections() const;               // non-empty collections, sorted
        size_t Size(std::string_view collection) const;

        // Knowledge.yaml: resources[] → RESOURCES (id = node id, data = Node JSON), network → NETWORK id
        // "model" (NetworkModel JSON), tools[] → TOOLS, operator[] → OPERATOR (data {text, ...}, tags),
        // memory[] → MEMORY; throws eCore::ConfigError / JsonError
        void Seed(const eCore::YamlNode &root);

        Topology::Inventory Inventory() const;  // from RESOURCES entries
        Topology::NetworkModel Network() const; // from NETWORK entry id "model" (empty model when absent)
        // atomically read-modify-write the network model entry
        void UpdateNetwork(const std::function<void(Topology::NetworkModel &)> &update);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Context
