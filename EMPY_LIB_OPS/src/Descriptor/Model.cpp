#include <eOps/Descriptor/Model.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <cmath>
#include <limits>
#include <charconv>
#include <stdexcept>

namespace eOps::Descriptor
{
    namespace
    {
        struct Suffix
        {
            std::string_view m_Text;
            uint64_t m_Factor;
        };

        // two-character binary suffixes are matched before the one-character decimal ones
        constexpr Suffix MEMORY_SUFFIXES[] = {{"Ki", 1ull << 10},
                                              {"Mi", 1ull << 20},
                                              {"Gi", 1ull << 30},
                                              {"Ti", 1ull << 40},
                                              {"Pi", 1ull << 50},
                                              {"Ei", 1ull << 60},
                                              {"k", 1000ull},
                                              {"K", 1000ull},
                                              {"M", 1000000ull},
                                              {"G", 1000000000ull},
                                              {"T", 1000000000000ull},
                                              {"P", 1000000000000000ull},
                                              {"E", 1000000000000000000ull}};

        constexpr Suffix BINARY_UNITS[] = {{"Ei", 1ull << 60}, {"Pi", 1ull << 50}, {"Ti", 1ull << 40},
                                           {"Gi", 1ull << 30}, {"Mi", 1ull << 20}, {"Ki", 1ull << 10}};

        [[noreturn]] void Malformed(std::string_view kind, std::string_view text)
        {
            throw std::invalid_argument("invalid " + std::string(kind) + " quantity '" + std::string(text) + "'");
        }

        bool AllDigits(std::string_view text) noexcept
        {
            if (text.empty())
            {
                return false;
            }
            for (char c : text)
            {
                if (c < '0' || c > '9')
                {
                    return false;
                }
            }
            return true;
        }
    } // namespace

    const Actionable *RichMLService::FindActionable(std::string_view topic) const
    {
        if (!ValidTopic(topic, false) || IsFilter(topic))
        {
            return nullptr;
        }
        // an exact declaration wins over a covering permission domain
        for (const Actionable &actionable : m_Actionables)
        {
            if (actionable.m_Name == topic)
            {
                return &actionable;
            }
        }
        for (const Actionable &actionable : m_Actionables)
        {
            if (FilterCovers(actionable.m_Name, topic))
            {
                return &actionable;
            }
        }
        return nullptr;
    }

    std::string RichMLService::ActuationTopic() const
    {
        if (m_Inference && !m_Inference->m_Actuation.empty())
        {
            return m_Inference->m_Actuation;
        }
        for (const Actionable &actionable : m_Actionables)
        {
            if (ValidTopic(actionable.m_Name, false) && !IsFilter(actionable.m_Name))
            {
                return actionable.m_Name;
            }
        }
        return {};
    }

    double ParseCpu(std::string_view text)
    {
        std::string value = Trim(text);
        bool milli = !value.empty() && value.back() == 'm';
        if (milli)
        {
            value.pop_back();
        }
        std::optional<double> number = ParseNumber(value);
        if (!number || *number < 0)
        {
            Malformed("cpu", text);
        }
        return milli ? *number / 1000.0 : *number;
    }

    uint64_t ParseMemory(std::string_view text)
    {
        std::string value = Trim(text);
        uint64_t factor = 1;
        for (const Suffix &suffix : MEMORY_SUFFIXES)
        {
            if (value.size() > suffix.m_Text.size() && value.ends_with(suffix.m_Text))
            {
                factor = suffix.m_Factor;
                value.resize(value.size() - suffix.m_Text.size());
                break;
            }
        }
        if (AllDigits(value))
        {
            uint64_t number = 0;
            auto [end, code] = std::from_chars(value.data(), value.data() + value.size(), number);
            if (code != std::errc() || end != value.data() + value.size() ||
                number > std::numeric_limits<uint64_t>::max() / factor)
            {
                Malformed("memory", text);
            }
            return number * factor;
        }
        // fractional quantities ("1.5Gi") round up to whole bytes
        std::optional<double> number = ParseNumber(value);
        if (!number || *number < 0)
        {
            Malformed("memory", text);
        }
        double bytes = std::ceil(*number * static_cast<double>(factor));
        if (!std::isfinite(bytes) || bytes >= 18446744073709551616.0)
        {
            Malformed("memory", text);
        }
        return static_cast<uint64_t>(bytes);
    }

    std::string FormatCpu(double cores)
    {
        if (!std::isfinite(cores) || cores < 0)
        {
            throw std::invalid_argument("cpu quantity must be a finite non-negative number");
        }
        long long millis = std::llround(cores * 1000.0);
        if (millis % 1000 == 0)
        {
            return std::to_string(millis / 1000);
        }
        return std::to_string(millis) + "m";
    }

    std::string FormatMemory(uint64_t bytes)
    {
        if (bytes != 0)
        {
            for (const Suffix &unit : BINARY_UNITS)
            {
                if (bytes % unit.m_Factor == 0)
                {
                    return std::to_string(bytes / unit.m_Factor) + std::string(unit.m_Text);
                }
            }
        }
        return std::to_string(bytes);
    }
} // namespace eOps::Descriptor
