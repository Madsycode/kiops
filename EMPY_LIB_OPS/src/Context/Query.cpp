#include <eOps/Context/Query.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eOps/Common/Text.h>
#include "Lexicon.h"
#include <array>
#include <algorithm>
#include "../Common/JsonRead.h"

namespace eOps::Context
{
    namespace
    {
        constexpr size_t MAX_LIMIT = 1000;
        constexpr std::array<std::string_view, 7> MEMBERS = {"collections", "where", "text",  "orderBy",
                                                             "descending",  "limit", "select"};
        constexpr std::array<std::pair<std::string_view, Op>, 17> OPS = {{{"eq", Op::Eq},
                                                                          {"ne", Op::Ne},
                                                                          {"lt", Op::Lt},
                                                                          {"le", Op::Le},
                                                                          {"gt", Op::Gt},
                                                                          {"ge", Op::Ge},
                                                                          {"contains", Op::Contains},
                                                                          {"in", Op::In},
                                                                          {"exists", Op::Exists},
                                                                          {"matches", Op::Matches},
                                                                          {"=", Op::Eq},
                                                                          {"==", Op::Eq},
                                                                          {"!=", Op::Ne},
                                                                          {"<", Op::Lt},
                                                                          {"<=", Op::Le},
                                                                          {">", Op::Gt},
                                                                          {">=", Op::Ge}}};

        std::vector<std::string> StringArray(const Json &value, std::string_view member, bool allowString = false)
        {
            std::vector<std::string> items;
            if (value.IsNull())
            {
                return items;
            }
            if (allowString && value.IsString())
            {
                items.push_back(value.AsString());
                return items;
            }
            if (!value.IsArray())
            {
                throw JsonError("query '" + std::string(member) + "' must be a list of strings");
            }
            for (const Json &item : value.AsArray())
            {
                if (!item.IsString())
                {
                    throw JsonError("query '" + std::string(member) + "' must be a list of strings");
                }
                items.push_back(item.AsString());
            }
            return items;
        }

        using eOps::Detail::Strings;

        // value of `field` for one entry; `scratch` holds synthesised values
        const Json *Resolve(const Entry &entry, std::string_view field, Json &scratch)
        {
            if (field == "id")
            {
                return &(scratch = Json(entry.m_Id));
            }
            if (field == "collection")
            {
                return &(scratch = Json(entry.m_Collection));
            }
            if (field == "tags")
            {
                return &(scratch = Strings(entry.m_Tags));
            }
            if (field == "updated")
            {
                return &(scratch = Json(entry.m_Updated));
            }
            if (field == "created")
            {
                return &(scratch = Json(entry.m_Created));
            }
            if (field.starts_with("data."))
            {
                field.remove_prefix(5);
            }
            if (field.empty())
            {
                return nullptr;
            }
            const Json *value = entry.m_Data.At(field);
            return value && !value->IsNull() ? value : nullptr;
        }

        bool Same(const Json &a, const Json &b)
        {
            if (a.IsString() && b.IsString())
            {
                return Detail::EqualsIgnoreCase(a.AsString(), b.AsString());
            }
            if (a.IsNumber() && b.IsNumber())
            {
                return a.AsNumber() == b.AsNumber();
            }
            return a == b;
        }

        // -1 / 0 / 1 for comparable scalars (numbers or strings), nullopt otherwise
        std::optional<int32_t> Order(const Json &a, const Json &b)
        {
            if (a.IsNumber() && b.IsNumber())
            {
                double x = a.AsNumber(), y = b.AsNumber();
                return x < y ? -1 : (x > y ? 1 : 0);
            }
            if (a.IsString() && b.IsString())
            {
                int32_t c = Lower(a.AsString()).compare(Lower(b.AsString()));
                return c < 0 ? -1 : (c > 0 ? 1 : 0);
            }
            return std::nullopt;
        }

        // predicate on one scalar field value (arrays are expanded by the caller)
        bool Scalar(const Json &field, const Condition &condition)
        {
            const Json &value = condition.m_Value;
            switch (condition.m_Op)
            {
            case Op::Eq:
                return Same(field, value);
            case Op::Lt:
            case Op::Le:
            case Op::Gt:
            case Op::Ge:
            {
                std::optional<int32_t> order = Order(field, value);
                if (!order)
                {
                    return false;
                }
                switch (condition.m_Op)
                {
                case Op::Lt:
                    return *order < 0;
                case Op::Le:
                    return *order <= 0;
                case Op::Gt:
                    return *order > 0;
                default:
                    return *order >= 0;
                }
            }
            case Op::Contains:
                if (field.IsString() && value.IsString())
                {
                    return Lower(field.AsString()).find(Lower(value.AsString())) != std::string::npos;
                }
                if (field.IsObject() && value.IsString())
                {
                    return field.Has(value.AsString());
                }
                return false;
            case Op::In:
                if (value.IsArray())
                {
                    return std::any_of(value.AsArray().begin(), value.AsArray().end(),
                                       [&field](const Json &option) { return Same(field, option); });
                }
                return Same(field, value);
            case Op::Matches:
                return field.IsString() && value.IsString() && FiltersOverlap(value.AsString(), field.AsString());
            default:
                return false;
            }
        }

        bool AnyElement(const Json &field, const Condition &condition)
        {
            if (field.IsArray())
            {
                // Contains also matches a whole element; arrays compare element-wise
                return std::any_of(field.AsArray().begin(), field.AsArray().end(),
                                   [&condition](const Json &item)
                                   {
                                       if (condition.m_Op == Op::Contains)
                                       {
                                           return Same(item, condition.m_Value);
                                       }
                                       return Scalar(item, condition);
                                   }) ||
                       (condition.m_Op == Op::Eq && Same(field, condition.m_Value));
            }
            return Scalar(field, condition);
        }

        bool Holds(const Entry &entry, const Condition &condition)
        {
            Json scratch;
            const Json *field = Resolve(entry, condition.m_Field, scratch);
            if (condition.m_Op == Op::Matches && !field && condition.m_Field == "topic")
            {
                // topic-bearing aliases: network densities use "topic", shards "topics", others "filter(s)"
                for (std::string_view alias : {"filter", "topics", "filters"})
                {
                    if ((field = Resolve(entry, alias, scratch)))
                    {
                        break;
                    }
                }
            }
            switch (condition.m_Op)
            {
            case Op::Exists:
            {
                bool wanted = !(condition.m_Value.IsBool() && !condition.m_Value.AsBool());
                return (field != nullptr) == wanted;
            }
            case Op::Ne:
            {
                if (!field)
                {
                    return true;
                }
                Condition equal{condition.m_Field, Op::Eq, condition.m_Value};
                return !AnyElement(*field, equal);
            }
            default:
                return field && AnyElement(*field, condition);
            }
        }

        double Relevance(const Entry &entry, const std::vector<std::string> &keywords)
        {
            if (keywords.empty())
            {
                return 1;
            }
            std::vector<std::string> tokens = Detail::EntryTokens(entry);
            double total = 0;
            for (const std::string &keyword : keywords)
            {
                if (std::binary_search(tokens.begin(), tokens.end(), keyword))
                {
                    total += 1;
                    continue;
                }
                // a longer keyword prefix ("thermo" → "thermostat") counts half
                if (keyword.size() >= 4)
                {
                    auto next = std::lower_bound(tokens.begin(), tokens.end(), keyword);
                    if (next != tokens.end() && next->starts_with(keyword))
                    {
                        total += 0.5;
                    }
                }
            }
            return total / static_cast<double>(keywords.size());
        }

        Json Project(const Entry &entry, const std::vector<std::string> &select)
        {
            Json projected = Json::MakeObject();
            for (const std::string &field : select)
            {
                Json scratch;
                if (const Json *value = Resolve(entry, field, scratch))
                {
                    projected.Set(field, *value);
                }
            }
            return projected;
        }
    } // namespace

    const char *OpName(Op op)
    {
        switch (op)
        {
        case Op::Eq:
            return "eq";
        case Op::Ne:
            return "ne";
        case Op::Lt:
            return "lt";
        case Op::Le:
            return "le";
        case Op::Gt:
            return "gt";
        case Op::Ge:
            return "ge";
        case Op::Contains:
            return "contains";
        case Op::In:
            return "in";
        case Op::Exists:
            return "exists";
        case Op::Matches:
            return "matches";
        }
        return "eq";
    }

    std::optional<Op> ParseOp(std::string_view name)
    {
        std::string lower = Lower(Trim(name));
        for (const auto &[text, op] : OPS)
        {
            if (lower == text)
            {
                return op;
            }
        }
        return std::nullopt;
    }

    Json Query::ToJson() const
    {
        Json where = Json::MakeArray();
        for (const Condition &condition : m_Where)
        {
            where.Push(Json(Json::Object{
                {"field", condition.m_Field}, {"op", OpName(condition.m_Op)}, {"value", condition.m_Value}}));
        }
        return Json(Json::Object{{"collections", Strings(m_Collections)},
                                 {"where", std::move(where)},
                                 {"text", Strings(m_Text)},
                                 {"orderBy", m_OrderBy},
                                 {"descending", m_Descending},
                                 {"limit", static_cast<uint64_t>(m_Limit)},
                                 {"select", Strings(m_Select)}});
    }

    Query Query::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("query must be an object");
        }
        for (const auto &[key, value] : json.AsObject())
        {
            if (std::find(MEMBERS.begin(), MEMBERS.end(), key) == MEMBERS.end())
            {
                throw JsonError("query has unknown member '" + key + "'");
            }
        }
        Query query;
        query.m_Collections = StringArray(json["collections"], "collections", true);
        for (const std::string &collection : query.m_Collections)
        {
            if (collection.empty())
            {
                throw JsonError("query collections must not be empty strings");
            }
        }
        if (const Json *where = json.Find("where"); where && !where->IsNull())
        {
            if (!where->IsArray())
            {
                throw JsonError("query 'where' must be a list of conditions");
            }
            for (const Json &item : where->AsArray())
            {
                if (!item.IsObject())
                {
                    throw JsonError("query condition must be an object {field, op, value}");
                }
                for (const auto &[key, value] : item.AsObject())
                {
                    if (key != "field" && key != "op" && key != "value")
                    {
                        throw JsonError("query condition has unknown member '" + key + "'");
                    }
                }
                const Json &field = item["field"];
                const Json &op = item["op"];
                if (!field.IsString() || Trim(field.AsString()).empty())
                {
                    throw JsonError("query condition needs a non-empty string 'field'");
                }
                if (!op.IsString())
                {
                    throw JsonError("query condition needs a string 'op'");
                }
                std::optional<Op> parsed = ParseOp(op.AsString());
                if (!parsed)
                {
                    throw JsonError("query condition has unknown op '" + op.AsString() + "'");
                }
                query.m_Where.push_back(Condition{Trim(field.AsString()), *parsed, item["value"]});
            }
        }
        query.m_Text = StringArray(json["text"], "text", true);
        if (const Json *order = json.Find("orderBy"); order && !order->IsNull())
        {
            if (!order->IsString())
            {
                throw JsonError("query 'orderBy' must be a string");
            }
            query.m_OrderBy = Trim(order->AsString());
        }
        if (const Json *descending = json.Find("descending"); descending && !descending->IsNull())
        {
            query.m_Descending = descending->AsBool();
        }
        if (const Json *limit = json.Find("limit"); limit && !limit->IsNull())
        {
            if (!limit->IsNumber() || limit->AsNumber() < 0)
            {
                throw JsonError("query 'limit' must be a non-negative integer");
            }
            query.m_Limit = static_cast<size_t>(std::min<uint64_t>(limit->AsUint(), MAX_LIMIT));
        }
        query.m_Select = StringArray(json["select"], "select");
        return query;
    }

    Json Query::Schema()
    {
        using namespace Intelligence;
        Json value = Json::MakeObject(
            {{"anyOf", Json::MakeArray({SchemaScalar("string"), SchemaScalar("number"), SchemaScalar("boolean"),
                                        SchemaArray(SchemaScalar("string")), Json::MakeObject({{"type", "null"}})})},
             {"description", "comparison operand; a list for 'in'; true/false or null for 'exists'"}});
        Json condition = SchemaObject(
            {{"field", SchemaScalar("string", "dotted path into the entry data, or id, tags, collection, updated")},
             {"op", SchemaEnum({"eq", "ne", "lt", "le", "gt", "ge", "contains", "in", "exists", "matches"},
                               "matches = MMQP topic filter overlap")},
             {"value", std::move(value)}});
        return SchemaObject(
            {{"collections",
              SchemaArray(SchemaEnum({MEMORY, TOOLS, RESOURCES, NETWORK, OPERATOR, OBSERVATIONS, CONSOLIDATED}),
                          "collections to search; empty = all")},
             {"where", SchemaArray(std::move(condition), "conditions that must all hold")},
             {"text", SchemaArray(SchemaScalar("string"), "relevance keywords")},
             {"orderBy", SchemaScalar("string", "field to order by; empty = relevance then recency")},
             {"descending", SchemaScalar("boolean")},
             {"limit", SchemaScalar("integer", "maximum number of entries")},
             {"select", SchemaArray(SchemaScalar("string"), "data fields to keep; empty = whole entry")}},
            "structured knowledge query");
    }

    KnowledgeSource::~KnowledgeSource() = default;

    LocalSource::LocalSource(const KnowledgePlane &plane) : m_Plane(plane)
    {
    }

    std::vector<Entry> LocalSource::Execute(const Query &query)
    {
        std::vector<std::string> collections =
            query.m_Collections.empty() ? m_Plane.Collections() : query.m_Collections;
        std::sort(collections.begin(), collections.end());
        collections.erase(std::unique(collections.begin(), collections.end()), collections.end());
        std::vector<Entry> entries;
        for (const std::string &collection : collections)
        {
            std::vector<Entry> scanned = m_Plane.Scan(collection);
            entries.insert(entries.end(), std::make_move_iterator(scanned.begin()),
                           std::make_move_iterator(scanned.end()));
        }
        return Evaluate(query, std::move(entries));
    }

    std::vector<Entry> Evaluate(const Query &query, std::vector<Entry> entries)
    {
        std::vector<std::string> keywords = Detail::Keywords(Join(query.m_Text, " "));
        std::vector<Entry> selected;
        for (Entry &entry : entries)
        {
            if (!query.m_Collections.empty() && std::find(query.m_Collections.begin(), query.m_Collections.end(),
                                                          entry.m_Collection) == query.m_Collections.end())
            {
                continue;
            }
            if (!std::all_of(query.m_Where.begin(), query.m_Where.end(),
                             [&entry](const Condition &condition) { return Holds(entry, condition); }))
            {
                continue;
            }
            entry.m_Score = Relevance(entry, keywords);
            selected.push_back(std::move(entry));
        }
        auto tail = [](const Entry &a, const Entry &b)
        {
            if (a.m_Score != b.m_Score)
            {
                return a.m_Score > b.m_Score;
            }
            if (a.m_Updated != b.m_Updated)
            {
                return a.m_Updated > b.m_Updated;
            }
            if (a.m_Collection != b.m_Collection)
            {
                return a.m_Collection < b.m_Collection;
            }
            return a.m_Id < b.m_Id;
        };
        if (query.m_OrderBy.empty())
        {
            std::sort(selected.begin(), selected.end(), tail);
        }
        else
        {
            // resolve once; entries without a comparable value go last in either direction
            std::vector<std::pair<std::optional<Json>, size_t>> keys;
            for (size_t i = 0; i < selected.size(); ++i)
            {
                Json scratch;
                const Json *value = Resolve(selected[i], query.m_OrderBy, scratch);
                keys.emplace_back(
                    value && (value->IsNumber() || value->IsString()) ? std::optional<Json>(*value) : std::nullopt, i);
            }
            std::sort(keys.begin(), keys.end(),
                      [&](const auto &a, const auto &b)
                      {
                          if (a.first.has_value() != b.first.has_value())
                          {
                              return a.first.has_value();
                          }
                          if (a.first)
                          {
                              std::optional<int32_t> order = Order(*a.first, *b.first);
                              if (!order)
                              {
                                  // numbers before strings
                                  if (a.first->IsNumber() != b.first->IsNumber())
                                  {
                                      return a.first->IsNumber();
                                  }
                              }
                              else if (*order != 0)
                              {
                                  return query.m_Descending ? *order > 0 : *order < 0;
                              }
                          }
                          return tail(selected[a.second], selected[b.second]);
                      });
            std::vector<Entry> ordered;
            ordered.reserve(selected.size());
            for (const auto &key : keys)
            {
                ordered.push_back(std::move(selected[key.second]));
            }
            selected = std::move(ordered);
        }
        size_t limit = query.m_Limit == 0 ? MAX_LIMIT : std::min(query.m_Limit, MAX_LIMIT);
        if (selected.size() > limit)
        {
            selected.resize(limit);
        }
        if (!query.m_Select.empty())
        {
            for (Entry &entry : selected)
            {
                entry.m_Data = Project(entry, query.m_Select);
            }
        }
        return selected;
    }
} // namespace eOps::Context
