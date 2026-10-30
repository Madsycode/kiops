#pragma once
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <algorithm>
#include <string_view>
#include <eOps/Common/Json.h>
#include <eOps/Common/Text.h>
#include <eOps/Descriptor/Model.h>
#include "../Common/JsonRead.h"

// private helpers shared by the harness sources (not installed)

namespace eOps::Harness::Detail
{
    // the shared typed readers and string-list helpers (Common/JsonRead.h)
    using eOps::Detail::Contains;
    using eOps::Detail::ReadBool;
    using eOps::Detail::ReadNumber;
    using eOps::Detail::ReadString;
    using eOps::Detail::ReadStrings;
    using eOps::Detail::ReadUnsigned;
    using eOps::Detail::Strings;

    // {cpu, memory (bytes), accelerator, acceleratorCount}
    inline Json ResourcesJson(const Descriptor::Resources &resources)
    {
        return Json::MakeObject({{"cpu", resources.m_Cpu},
                                 {"memory", resources.m_Memory},
                                 {"accelerator", resources.m_Accelerator},
                                 {"acceleratorCount", resources.m_AcceleratorCount}});
    }

    inline Descriptor::Resources ResourcesFrom(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("resources must be an object");
        }
        Descriptor::Resources resources;
        resources.m_Cpu = ReadNumber(json, "cpu");
        resources.m_Memory = ReadUnsigned(json, "memory");
        resources.m_Accelerator = ReadString(json, "accelerator");
        resources.m_AcceleratorCount =
            static_cast<uint32_t>(std::min<uint64_t>(ReadUnsigned(json, "acceleratorCount"), UINT32_MAX));
        if (resources.m_Cpu < 0)
        {
            throw JsonError("resources.cpu must not be negative");
        }
        return resources;
    }

    // lifecycle policy with the descriptor field names (Listing 2.1 lifecycle section)
    inline Json PolicyJson(const Descriptor::LifecyclePolicy &policy)
    {
        return Json::MakeObject({{"accuracyThreshold", policy.m_AccuracyThreshold},
                                 {"minImprovement", policy.m_MinImprovement},
                                 {"cooldownSeconds", policy.m_CooldownSeconds},
                                 {"evaluationWindow", policy.m_EvaluationWindow},
                                 {"minSamples", policy.m_MinSamples},
                                 {"tolerance", policy.m_Tolerance},
                                 {"autoRollback", policy.m_AutoRollback}});
    }

    inline Descriptor::LifecyclePolicy PolicyFrom(const Json &json)
    {
        Descriptor::LifecyclePolicy policy;
        if (json.IsNull())
        {
            return policy;
        }
        if (!json.IsObject())
        {
            throw JsonError("policy must be an object");
        }
        auto narrow = [](uint64_t value) { return static_cast<uint32_t>(std::min<uint64_t>(value, UINT32_MAX)); };
        policy.m_AccuracyThreshold = ReadNumber(json, "accuracyThreshold", policy.m_AccuracyThreshold);
        policy.m_MinImprovement = ReadNumber(json, "minImprovement", policy.m_MinImprovement);
        policy.m_CooldownSeconds = narrow(ReadUnsigned(json, "cooldownSeconds", policy.m_CooldownSeconds));
        policy.m_EvaluationWindow = narrow(ReadUnsigned(json, "evaluationWindow", policy.m_EvaluationWindow));
        policy.m_MinSamples = narrow(ReadUnsigned(json, "minSamples", policy.m_MinSamples));
        policy.m_Tolerance = ReadNumber(json, "tolerance", policy.m_Tolerance);
        policy.m_AutoRollback = ReadBool(json, "autoRollback", policy.m_AutoRollback);
        return policy;
    }

    // MAJOR.MINOR.PATCH (pre-release/build suffixes ignored); nullopt when malformed
    inline std::optional<std::array<uint64_t, 3>> SemVer(std::string_view text)
    {
        size_t end = text.find_first_of("-+");
        std::vector<std::string> parts = Split(text.substr(0, end), '.');
        if (parts.size() != 3)
        {
            return std::nullopt;
        }
        std::array<uint64_t, 3> version{};
        for (size_t i = 0; i < 3; ++i)
        {
            std::optional<double> number = ParseNumber(parts[i]);
            if (parts[i].empty() || !number || *number < 0 || std::floor(*number) != *number ||
                parts[i].find_first_not_of("0123456789") != std::string::npos)
            {
                return std::nullopt;
            }
            version[i] = static_cast<uint64_t>(*number);
        }
        return version;
    }

    // <0 / 0 / >0 like strcmp; malformed versions compare as text
    inline int32_t CompareVersions(std::string_view a, std::string_view b)
    {
        std::optional<std::array<uint64_t, 3>> left = SemVer(a), right = SemVer(b);
        if (!left || !right)
        {
            return a < b ? -1 : (a == b ? 0 : 1);
        }
        return *left < *right ? -1 : (*left == *right ? 0 : 1);
    }
} // namespace eOps::Harness::Detail
