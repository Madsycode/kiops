#pragma once
#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <string_view>
#include "../Common/Json.h"

// helpers for structured LLM output. Schemas sent to providers use the strict subset accepted by
// the Messages API structured outputs: every object has "additionalProperties": false and lists
// all its properties as required; no numeric or string length constraints. Conforms validates a
// returned document against that subset before any caller trusts it.

namespace eOps::Intelligence
{
    // {"type": "string"} etc.; `kind` ∈ string | number | integer | boolean
    EMPY_OPS_API Json SchemaScalar(std::string_view kind, std::string_view description = {});
    // {"type": "string", "enum": [...]}
    EMPY_OPS_API Json SchemaEnum(const std::vector<std::string> &values, std::string_view description = {});
    // {"type": "array", "items": items}
    EMPY_OPS_API Json SchemaArray(Json items, std::string_view description = {});
    // strict object: every property required, additionalProperties false
    EMPY_OPS_API Json SchemaObject(std::vector<std::pair<std::string, Json>> properties,
                                   std::string_view description = {});
    // {"anyOf": [schema, {"type": "null"}]} for optional values
    EMPY_OPS_API Json SchemaNullable(Json schema);

    // validates type, enum, properties, required, additionalProperties:false, items and anyOf;
    // false with a dotted `error` path on the first violation
    EMPY_OPS_API bool Conforms(const Json &value, const Json &schema, std::string *error = nullptr);

    // parse model text as JSON: the whole text, else a ```json fenced block, else the outermost
    // {...} object; nullopt when none parses
    EMPY_OPS_API std::optional<Json> ExtractJson(std::string_view text);
} // namespace eOps::Intelligence
