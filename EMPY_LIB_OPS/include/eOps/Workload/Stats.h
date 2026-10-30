#pragma once
#include <deque>
#include <chrono>
#include <cstdint>
#include <utility>
#include <optional>
#include <string_view>
#include "../API.h"

// time-windowed statistics for latency objectives (Listing 2.1 "p95 over 60 s"); owner-thread use

namespace eOps::Workload
{
    class EMPY_OPS_API WindowStats
    {
    public:
        explicit WindowStats(uint32_t windowSeconds = 60, size_t maxSamples = 4096);

        // samples older than the window are discarded; beyond maxSamples the oldest is dropped
        void Add(double value, std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now());
        // p50 p90 p95 p99 (nearest rank) | mean | max over samples inside the window; nullopt when empty
        // or for an unknown statistic name
        std::optional<double> Statistic(std::string_view name,
                                        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
        size_t Count(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());
        uint32_t WindowSeconds() const noexcept;

        static bool KnownStatistic(std::string_view name); // p50 p90 p95 p99 mean max

    private:
        void Expire(std::chrono::steady_clock::time_point now);

        uint32_t m_WindowSeconds = 60;
        size_t m_MaxSamples = 4096;
        std::deque<std::pair<std::chrono::steady_clock::time_point, double>> m_Samples;
    };
} // namespace eOps::Workload
