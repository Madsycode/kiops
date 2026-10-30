#pragma once
#include <string>
#include <string_view>
#include <eOps/Common/Json.h>

// deterministic block-style YAML for JSON documents (private to EmpyOps, shared on purpose by the
// Descriptor emitter and the Substrate manifest renderer): members in insertion order, two-space
// indentation, scalar lists in flow style, strings plain only when unambiguous and otherwise
// double-quoted with JSON escapes, multi-line strings as literal blocks

namespace eOps::Detail
{
    // a whole document (mapping, sequence or scalar), newline terminated
    std::string WriteYaml(const Json &document);

    // one scalar as it appears after "key: " (strings quoted when needed); containers throw JsonError
    std::string YamlScalar(const Json &value);

    // true when `text` reads back as the same string without quotes
    bool PlainScalar(std::string_view text);
} // namespace eOps::Detail
