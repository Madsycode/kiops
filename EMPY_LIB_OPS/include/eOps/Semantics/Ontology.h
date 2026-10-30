#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include "../Common/Json.h"

// BoM-Ontology (thesis Fig. 25): domain vocabulary grounding the Executive's semantic parsing.
// Concepts carry a kind, synonyms (single words or multi-word phrases), attributes and typed
// relations; the ontology evolves through validated Extend calls ("map / evolve").
//
// Kinds: domain (attr root), quantity (unit, topic suffix, optional min/max), actuator (quantity,
// topic suffix, unit, min, max), pattern (actionable bool, latencyMs, statistic, suffix),
// objective, region, accelerator, unit, schedule (cron).
// Relations: actuates (actuator→quantity), measuredIn (quantity→unit), contains
// (domain→quantity), conflictsWith (objective→objective), prefers (pattern→region).
// Thread-safety: none; callers serialise access (the AgentHost holds one ontology mutex).

namespace eOps::Semantics
{
    struct Relation
    {
        std::string m_Type;
        std::string m_Target; // concept id
    };

    struct EMPY_OPS_API Concept
    {
        std::string m_Id;
        std::string m_Kind;
        std::vector<std::string> m_Synonyms;
        Json m_Attributes; // object (null when none)
        std::vector<Relation> m_Relations;

        Json ToJson() const;                       // {id, kind, synonyms[], attributes{}, relations[{type, target}]}
        static Concept FromJson(const Json &json); // throws JsonError on a malformed concept
    };

    // one ontology hit inside a token sequence
    struct ConceptMatch
    {
        const Concept *m_Concept = nullptr;
        size_t m_Token = 0;  // index of the first matched token
        size_t m_Length = 1; // matched token count (multi-word synonyms > 1)
        std::string m_Term;  // matched text, tokens joined by ' '
        double m_Score = 1;  // 1 = exact id/synonym, < 1 = normalised (plural) match
    };

    // Lower + Trim; strips a plural "es"/"s" when the word is longer than 3 characters
    EMPY_OPS_API std::string Normalize(std::string_view term);

    class EMPY_OPS_API Ontology
    {
    public:
        Ontology(); // empty
        ~Ontology();
        Ontology(const Ontology &other);
        Ontology &operator=(const Ontology &other);
        Ontology(Ontology &&other) noexcept;
        Ontology &operator=(Ontology &&other) noexcept;

        // `concepts: [{id, kind, synonyms[], attributes{}, relations[{type, target}]}]`; every concept goes
        // through Extend (throws std::invalid_argument naming the rejected concept)
        static Ontology FromYaml(const eCore::YamlNode &root);
        static Ontology Load(const std::filesystem::path &path);
        std::string ToYaml() const;                         // same format, deterministic (insertion order)
        void Save(const std::filesystem::path &path) const; // atomic (WriteFileAtomic)

        const Concept *Find(std::string_view id) const;     // by id
        const Concept *Lookup(std::string_view term) const; // id or synonym, after Tokenize/Normalize
        // longest multi-word matches first, non-overlapping, ordered by token position
        std::vector<ConceptMatch> Match(const std::vector<std::string> &tokens) const;
        // targets of `relation` from concept `id`
        std::vector<const Concept *> Related(std::string_view id, std::string_view relation) const;
        // sources whose `relation` points at `id` (e.g. actuators that actuate a quantity)
        std::vector<const Concept *> RelatedFrom(std::string_view id, std::string_view relation) const;
        std::vector<const Concept *> OfKind(std::string_view kind) const;

        // evolution with consistency checks: rejects an empty id, an unknown kind, an existing id of
        // another kind, a missing relation target, or a synonym bound to a concept of another kind;
        // an existing same-kind concept gets the new synonyms/attributes/relations merged in.
        // Concept pointers stay valid across Extend.
        bool Extend(Concept entry, std::string *reason = nullptr);

        size_t Size() const noexcept;
        uint64_t Revision() const noexcept; // increments on every successful Extend

        static const std::vector<std::string> &Kinds();
        static const std::vector<std::string> &RelationTypes();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Semantics
