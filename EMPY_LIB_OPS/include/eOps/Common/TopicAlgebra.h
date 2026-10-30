#pragma once
#include <string>
#include <cstddef>
#include <string_view>
#include "../API.h"

// MMQP topic-filter algebra used for permission domains, ownership and conflict detection.
// Inputs are canonicalised like the broker does (eCore::MMQP::ParseTopic): "$share/<group>/"
// prefixes are stripped, '+' matches one level, a trailing '#' matches zero or more levels
// (so "a/#" also matches "a"), and root wildcards never match topics whose first level starts
// with '$'. Invalid inputs make every predicate false; nothing here throws.

namespace eOps
{
    // true when some concrete topic matches both filters
    EMPY_OPS_API bool FiltersOverlap(std::string_view a, std::string_view b);

    // true when every topic matching `inner` also matches `outer`
    EMPY_OPS_API bool FilterCovers(std::string_view outer, std::string_view inner);

    // true when the text carries a wildcard level or a shared prefix (it cannot be published to)
    EMPY_OPS_API bool IsFilter(std::string_view topic);

    // ParseTopic without throwing: `filter` admits wildcards and shared prefixes
    EMPY_OPS_API bool ValidTopic(std::string_view topic, bool filter);

    // level `index` of the canonical topic ("" when out of range or invalid)
    EMPY_OPS_API std::string TopicLevel(std::string_view topic, size_t index);
} // namespace eOps
