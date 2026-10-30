#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <string_view>
#include <eOps/Common/Json.h>

// typed JSON member readers and string-list helpers shared by every EmpyOps module that decodes
// agent, harness or knowledge documents (private, not installed). One error format: a member of
// the wrong type throws JsonError "'<key>' must be ..."; an absent or null member yields the fallback.

namespace eOps::Detail
{
    inline Json Strings(const std::vector<std::string> &items)
    {
        Json array = Json::MakeArray();
        for (const std::string &item : items)
        {
            array.Push(Json(item));
        }
        return array;
    }

    inline bool Contains(const std::vector<std::string> &items, std::string_view item)
    {
        return std::find(items.begin(), items.end(), item) != items.end();
    }

    inline void AddUnique(std::vector<std::string> &items, const std::string &item)
    {
        if (!item.empty() && !Contains(items, item))
        {
            items.push_back(item);
        }
    }

    inline void RequireObject(const Json &json, std::string_view what)
    {
        if (!json.IsObject())
        {
            throw JsonError(std::string(what) + " must be an object");
        }
    }

    inline std::string ReadString(const Json &json, std::string_view key, std::string fallback = {})
    {
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return fallback;
        }
        if (!value->IsString())
        {
            throw JsonError("'" + std::string(key) + "' must be a string");
        }
        return value->AsString();
    }

    inline double ReadNumber(const Json &json, std::string_view key, double fallback = 0)
    {
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return fallback;
        }
        if (!value->IsNumber() || !std::isfinite(value->AsNumber()))
        {
            throw JsonError("'" + std::string(key) + "' must be a finite number");
        }
        return value->AsNumber();
    }

    inline std::optional<double> ReadOptionalNumber(const Json &json, std::string_view key)
    {
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return std::nullopt;
        }
        return ReadNumber(json, key);
    }

    // a non-negative integer (an integral double such as 5.0 is accepted; a fraction or a negative number is not)
    inline uint64_t ReadUnsigned(const Json &json, std::string_view key, uint64_t fallback = 0)
    {
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return fallback;
        }
        try
        {
            if (value->IsNumber())
            {
                return value->AsUint();
            }
        }
        catch (const JsonError &)
        {
        }
        throw JsonError("'" + std::string(key) + "' must be a non-negative integer");
    }

    inline bool ReadBool(const Json &json, std::string_view key, bool fallback = false)
    {
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return fallback;
        }
        if (!value->IsBool())
        {
            throw JsonError("'" + std::string(key) + "' must be a boolean");
        }
        return value->AsBool();
    }

    inline std::vector<std::string> ReadStrings(const Json &json, std::string_view key)
    {
        std::vector<std::string> items;
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return items;
        }
        if (!value->IsArray())
        {
            throw JsonError("'" + std::string(key) + "' must be an array of strings");
        }
        for (const Json &item : value->AsArray())
        {
            if (!item.IsString())
            {
                throw JsonError("'" + std::string(key) + "' must be an array of strings");
            }
            items.push_back(item.AsString());
        }
        return items;
    }

    inline const Json::Array &ReadArray(const Json &json, std::string_view key)
    {
        static const Json::Array s_Empty;
        const Json *value = json.Find(key);
        if (!value || value->IsNull())
        {
            return s_Empty;
        }
        if (!value->IsArray())
        {
            throw JsonError("'" + std::string(key) + "' must be an array");
        }
        return value->AsArray();
    }
} // namespace eOps::Detail
