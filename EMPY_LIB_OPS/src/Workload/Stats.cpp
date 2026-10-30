#include <eOps/Workload/Stats.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace eOps::Workload
{
    namespace
    {
        // nearest-rank percentile of `percent` over sorted values
        double Rank(const std::vector<double> &sorted, double percent)
        {
            size_t rank = static_cast<size_t>(std::ceil(percent / 100.0 * static_cast<double>(sorted.size())));
            rank = std::clamp<size_t>(rank, 1, sorted.size());
            return sorted[rank - 1];
        }

        std::optional<double> Percent(std::string_view name)
        {
            if (name == "p50")
            {
                return 50.0;
            }
            if (name == "p90")
            {
                return 90.0;
            }
            if (name == "p95")
            {
                return 95.0;
            }
            if (name == "p99")
            {
                return 99.0;
            }
            return std::nullopt;
        }
    } // namespace

    WindowStats::WindowStats(uint32_t windowSeconds, size_t maxSamples)
        : m_WindowSeconds(windowSeconds), m_MaxSamples(maxSamples)
    {
        if (windowSeconds == 0 || maxSamples == 0)
        {
            throw std::invalid_argument("window statistics need a window of at least 1 s and 1 sample");
        }
    }

    void WindowStats::Add(double value, std::chrono::steady_clock::time_point at)
    {
        if (!std::isfinite(value))
        {
            return; // a non-finite measurement carries no information for the objective
        }
        Expire(at);
        if (m_Samples.size() >= m_MaxSamples)
        {
            m_Samples.pop_front();
        }
        m_Samples.emplace_back(at, value);
    }

    std::optional<double> WindowStats::Statistic(std::string_view name, std::chrono::steady_clock::time_point now)
    {
        if (!KnownStatistic(name))
        {
            return std::nullopt;
        }
        Expire(now);
        if (m_Samples.empty())
        {
            return std::nullopt;
        }
        std::vector<double> values;
        values.reserve(m_Samples.size());
        for (const auto &sample : m_Samples)
        {
            values.push_back(sample.second);
        }
        if (name == "mean")
        {
            double sum = 0;
            for (double value : values)
            {
                sum += value;
            }
            return sum / static_cast<double>(values.size());
        }
        if (name == "max")
        {
            return *std::max_element(values.begin(), values.end());
        }
        std::sort(values.begin(), values.end());
        return Rank(values, *Percent(name));
    }

    size_t WindowStats::Count(std::chrono::steady_clock::time_point now)
    {
        Expire(now);
        return m_Samples.size();
    }

    uint32_t WindowStats::WindowSeconds() const noexcept
    {
        return m_WindowSeconds;
    }

    bool WindowStats::KnownStatistic(std::string_view name)
    {
        return Percent(name).has_value() || name == "mean" || name == "max";
    }

    void WindowStats::Expire(std::chrono::steady_clock::time_point now)
    {
        auto window = std::chrono::seconds(m_WindowSeconds);
        while (!m_Samples.empty() && now - m_Samples.front().first > window)
        {
            m_Samples.pop_front();
        }
    }
} // namespace eOps::Workload
