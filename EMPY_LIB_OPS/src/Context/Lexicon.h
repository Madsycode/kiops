#pragma once
#include <string>
#include <vector>
#include <string_view>
#include <eOps/Context/Knowledge.h>

// private lexical helpers of the CoMaS retrieval engine: light stemming, stopwords, keyword
// extraction and the token set of a knowledge entry used for text relevance

namespace eOps::Context::Detail
{
    // lowercase word folded to a light stem ("retraining" → "retrain", "decisions" → "decision")
    std::string Stem(std::string_view word);

    // function words that carry no retrieval meaning
    bool IsStopword(std::string_view stem);

    // stemmed, deduplicated, stopword-free tokens of `text` (numbers dropped)
    std::vector<std::string> Keywords(std::string_view text);

    // sorted unique stems of an entry's id, collection, tags and JSON data (keys and values)
    std::vector<std::string> EntryTokens(const Entry &entry);

    // ASCII case-insensitive equality
    bool EqualsIgnoreCase(std::string_view a, std::string_view b);

    // `text` cut to at most `maxBytes` on a UTF-8 code point boundary
    std::string Truncate(std::string_view text, size_t maxBytes);
} // namespace eOps::Context::Detail
