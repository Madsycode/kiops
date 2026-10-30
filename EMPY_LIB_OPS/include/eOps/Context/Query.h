#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <string_view>
#include "Knowledge.h"

// structured knowledge query x2 (thesis Fig. 26) and the lightweight retrieval engine Execute(x2).
// Field = dotted path into Entry::m_Data, or "id", "tags", "collection", "updated". Text keywords
// score relevance by token overlap with the entry's JSON text and tags; empty m_Collections = all.

namespace eOps::Context
{
    enum class Op : uint8_t
    {
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        Contains, // string substring or array element
        In,       // field value ∈ m_Value array
        Exists,
        Matches // MMQP topic filter m_Value overlaps the field's topic/filter
    };

    EMPY_OPS_API const char *OpName(Op op); // "eq" "ne" "lt" "le" "gt" "ge" "contains" "in" "exists" "matches"
    EMPY_OPS_API std::optional<Op> ParseOp(std::string_view name);

    struct Condition
    {
        std::string m_Field;
        Op m_Op = Op::Eq;
        Json m_Value;
    };

    struct EMPY_OPS_API Query
    {
        std::vector<std::string> m_Collections;
        std::vector<Condition> m_Where;  // all must hold
        std::vector<std::string> m_Text; // relevance keywords
        std::string m_OrderBy;           // field; "" → score then updated
        bool m_Descending = false;
        size_t m_Limit = 20;
        std::vector<std::string> m_Select; // data fields to keep; empty = whole entry

        // {collections[], where[{field, op, value}], text[], orderBy, descending, limit, select[]}
        Json ToJson() const;
        static Query FromJson(const Json &json); // throws JsonError on unknown ops / bad shapes
        // strict structured-output schema for the LLM translator (Structured.h subset) whose documents
        // FromJson accepts directly: every field required, condition "value" is anyOf string | number |
        // boolean | string array | null, "orderBy" "" when unordered
        static Json Schema();
    };

    struct EMPY_OPS_API KnowledgeSource
    {
        virtual ~KnowledgeSource();
        virtual std::vector<Entry> Execute(const Query &query) = 0;
    };

    // in-process retrieval engine over a KnowledgePlane
    class EMPY_OPS_API LocalSource : public KnowledgeSource
    {
    public:
        explicit LocalSource(const KnowledgePlane &plane);
        std::vector<Entry> Execute(const Query &query) override;

    private:
        const KnowledgePlane &m_Plane;
    };

    // shared evaluation used by LocalSource and the MCP server: entries of `query` from `entries`,
    // filtered, scored (m_Score), ordered, limited and projected
    EMPY_OPS_API std::vector<Entry> Evaluate(const Query &query, std::vector<Entry> entries);
} // namespace eOps::Context
