#pragma once
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <string_view>
#include "Knowledge.h"
#include "Query.h"
#include "../Intelligence/Provider.h"

// CoMaS context retrieval (thesis Fig. 26): an agent sends intent x1 → Augment adds context c →
// Translate(x1, c) yields structured query x2 (LLM via AskJson with Query::Schema(), or the
// deterministic vocabulary translator) → Execute(x2) against the knowledge source returns R′ →
// Contextualise formats, aggregates and filters R′ into R. Also records agents memory and
// observations, and consolidates old episodic memory.

namespace eOps::Context
{
    struct RetrievalRequest
    {
        std::string m_Agent;
        std::string m_Intent;
        size_t m_Limit = 20;
    };

    struct EMPY_OPS_API RetrievalResult
    {
        Json m_Context; // c
        Query m_Query;  // x2
        std::vector<Entry> m_Entries;
        Json m_Aggregates;        // {count: {collection: n}, regions: {region: n}}
        std::string m_Summary;    // one line per entry, ≤ 4 KiB
        std::string m_Translator; // "llm:<provider>" | "deterministic" | "deterministic(fallback)"

        // {context, query, entries[], aggregates, summary, translator}
        Json ToJson() const;
    };

    // thread-safe; the plane (and providers/source) must outlive it
    class EMPY_OPS_API Comas
    {
    public:
        Comas(KnowledgePlane &plane, const Intelligence::ProviderSet *providers = nullptr,
              std::shared_ptr<KnowledgeSource> source = nullptr); // null source → LocalSource(plane)

        RetrievalResult Retrieve(const RetrievalRequest &request); // all four steps
        Json Augment(std::string_view intent,
                     size_t limit = 8) const; // c: {collections{name: size}, operator[], recent[]}
        // x2 via the Reasoning provider when available, else deterministic; `translator` records which
        Query Translate(std::string_view intent, const Json &context, std::string *translator = nullptr) const;
        Query TranslateDeterministic(std::string_view intent, const Json &context) const;
        std::vector<Entry> Execute(const Query &query) const; // R′ via the source
        RetrievalResult Contextualise(const RetrievalRequest &request, Json context, Query query,
                                      std::vector<Entry> entries, std::string translator) const; // R

        // agents memory entry {agent, kind, service, data} tagged [agent, kind, service]
        void Remember(std::string_view agent, std::string_view kind, std::string_view service, Json data);
        // observation entry {source, kind, data} tagged [source, kind]
        void Observe(std::string_view source, std::string_view kind, Json data);
        // MEMORY entries older than maxAge grouped by (service, kind) into CONSOLIDATED entries
        // {service, kind, count, successes, failures, lastOutcome, lastAt}; grouped entries are erased.
        // Returns the number of consolidated groups.
        size_t Consolidate(uint64_t nowMillis, uint64_t maxAgeMillis = 3600000);

        KnowledgePlane &Plane() noexcept;

    private:
        KnowledgePlane &m_Plane;
        const Intelligence::ProviderSet *m_Providers = nullptr;
        std::shared_ptr<KnowledgeSource> m_Source;
        mutable std::mutex m_Mutex; // serialises Consolidate against Remember
    };
} // namespace eOps::Context
