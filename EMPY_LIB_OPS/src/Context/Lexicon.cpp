#include "Lexicon.h"
#include <eOps/Common/Text.h>
#include <set>
#include <array>
#include <algorithm>

namespace eOps::Context::Detail
{
    namespace
    {
        constexpr std::array<std::string_view, 72> STOPWORDS = {
            "a",      "an",    "the",  "which", "what",  "who",  "whom", "where", "when", "how",     "is",   "are",
            "was",    "were",  "be",   "been",  "being", "have", "has",  "had",   "do",   "doe",     "did",  "of",
            "for",    "to",    "in",   "on",    "at",    "by",   "with", "and",   "or",   "between", "from", "about",
            "any",    "all",   "me",   "show",  "list",  "find", "get",  "give",  "tell", "there",   "that", "this",
            "these",  "those", "i",    "we",    "you",   "it",   "its",  "our",   "my",   "please",  "can",  "could",
            "should", "would", "will", "known", "need",  "into", "than", "their", "them", "us",      "some", "each"};

        bool IsDigitWord(std::string_view word)
        {
            return !word.empty() && std::all_of(word.begin(), word.end(),
                                                [](char c) { return (c >= '0' && c <= '9') || c == '.' || c == '-'; });
        }

        void Collect(const Json &value, std::set<std::string> &tokens)
        {
            switch (value.GetType())
            {
            case Json::Type::String:
                for (const std::string &token : Tokenize(value.AsString()))
                {
                    tokens.insert(Stem(token));
                }
                break;
            case Json::Type::Array:
                for (const Json &item : value.AsArray())
                {
                    Collect(item, tokens);
                }
                break;
            case Json::Type::Object:
                for (const auto &[key, member] : value.AsObject())
                {
                    for (const std::string &token : Tokenize(key))
                    {
                        tokens.insert(Stem(token));
                    }
                    Collect(member, tokens);
                }
                break;
            default:
                break;
            }
        }
    } // namespace

    std::string Stem(std::string_view word)
    {
        std::string stem = Lower(word);
        auto ends = [&stem](std::string_view suffix) { return stem.ends_with(suffix); };
        if (stem.size() > 5 && ends("ing"))
        {
            stem.resize(stem.size() - 3);
        }
        else if (stem.size() > 4 && ends("ed") && !ends("eed"))
        {
            stem.resize(stem.size() - 2);
        }
        else
        {
            // plurals fold exactly like the ontology terms (gpus → gpu, policies → policy)
            stem = Singular(stem);
        }
        return stem;
    }

    bool IsStopword(std::string_view stem)
    {
        return std::find(STOPWORDS.begin(), STOPWORDS.end(), stem) != STOPWORDS.end();
    }

    std::vector<std::string> Keywords(std::string_view text)
    {
        std::vector<std::string> keywords;
        for (const std::string &token : Tokenize(text))
        {
            if (IsDigitWord(token))
            {
                continue;
            }
            std::string stem = Stem(token);
            if (IsStopword(stem) || std::find(keywords.begin(), keywords.end(), stem) != keywords.end())
            {
                continue;
            }
            keywords.push_back(std::move(stem));
        }
        return keywords;
    }

    std::vector<std::string> EntryTokens(const Entry &entry)
    {
        std::set<std::string> tokens;
        Collect(Json(entry.m_Id), tokens);
        Collect(Json(entry.m_Collection), tokens);
        for (const std::string &tag : entry.m_Tags)
        {
            Collect(Json(tag), tokens);
        }
        Collect(entry.m_Data, tokens);
        return {tokens.begin(), tokens.end()};
    }

    bool EqualsIgnoreCase(std::string_view a, std::string_view b)
    {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(),
                                                  [](char x, char y)
                                                  {
                                                      auto lower = [](char c)
                                                      { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + 32) : c; };
                                                      return lower(x) == lower(y);
                                                  });
    }

    std::string Truncate(std::string_view text, size_t maxBytes)
    {
        if (text.size() <= maxBytes)
        {
            return std::string(text);
        }
        size_t end = maxBytes;
        // step back over UTF-8 continuation bytes so a code point is never split
        while (end > 0 && (static_cast<uint8_t>(text[end]) & 0xC0) == 0x80)
        {
            --end;
        }
        return std::string(text.substr(0, end));
    }
} // namespace eOps::Context::Detail
