#pragma once
#include <cmath>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include <string_view>
#include <eOps/Common/Json.h>
#include <eOps/Descriptor/Model.h>
#include "../Common/JsonRead.h"

// private JSON helpers shared by the agent cores (not installed)

namespace eOps::Agents::Detail
{
    // the shared typed readers and string-list helpers (Common/JsonRead.h)
    using eOps::Detail::AddUnique;
    using eOps::Detail::Contains;
    using eOps::Detail::ReadArray;
    using eOps::Detail::ReadBool;
    using eOps::Detail::ReadNumber;
    using eOps::Detail::ReadOptionalNumber;
    using eOps::Detail::ReadString;
    using eOps::Detail::ReadStrings;
    using eOps::Detail::ReadUnsigned;
    using eOps::Detail::RequireObject;
    using eOps::Detail::Strings;

    inline Json Number(const std::optional<double> &value)
    {
        return value ? Json(*value) : Json();
    }

    // compact number text for rationales and findings ("16", "0.25")
    inline std::string Text(double value)
    {
        if (!std::isfinite(value))
        {
            return "non-finite";
        }
        double rounded = std::round(value * 1000) / 1000;
        return Json(rounded == 0 ? 0.0 : rounded).Dump();
    }

    inline Json ObservableJson(const Descriptor::Observable &observable)
    {
        return Json::MakeObject({{"name", observable.m_Name},
                                 {"unit", observable.m_Unit},
                                 {"description", observable.m_Description},
                                 {"qos", static_cast<uint32_t>(observable.m_Qos)}});
    }

    inline Json ActionableJson(const Descriptor::Actionable &actionable)
    {
        return Json::MakeObject({{"name", actionable.m_Name},
                                 {"unit", actionable.m_Unit},
                                 {"min", Number(actionable.m_Min)},
                                 {"max", Number(actionable.m_Max)},
                                 {"maxRatePerMinute", actionable.m_MaxRatePerMinute},
                                 {"description", actionable.m_Description},
                                 {"qos", static_cast<uint32_t>(actionable.m_Qos)}});
    }
} // namespace eOps::Agents::Detail
