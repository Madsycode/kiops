#pragma once
#include <chrono>
#include <string>
#include <cstdint>
#include <string_view>
#include "../API.h"

// cron schedules for temporal training triggers (UTC). Standard five fields
// "minute hour day-of-month month day-of-week" with '*', lists, ranges, steps ("*/5", "1-5/2"),
// month/day names ("jan", "mon"), Vixie OR-semantics when both day fields are restricted and
// day-of-week 0 or 7 = Sunday; macros @yearly @annually @monthly @weekly @daily @midnight
// @hourly; extension "@every <N>(s|m|h)" (N ≥ 1) with periods anchored at the Unix epoch.

namespace eOps::Descriptor
{
    class EMPY_OPS_API CronSchedule
    {
    public:
        static CronSchedule Parse(std::string_view expression); // throws std::invalid_argument

        // first firing strictly after `after`
        std::chrono::system_clock::time_point Next(std::chrono::system_clock::time_point after) const;

        const std::string &Expression() const noexcept;

    private:
        CronSchedule() = default;

        std::string m_Expression;
        uint64_t m_Minutes = 0; // bit i = minute i (0..59)
        uint32_t m_Hours = 0;   // bit i = hour i (0..23)
        uint32_t m_Days = 0;    // bit i = day-of-month i (1..31)
        uint16_t m_Months = 0;  // bit i = month i (1..12)
        uint8_t m_Weekdays = 0; // bit i = weekday i (0 = Sunday .. 6)
        bool m_DayRestricted = false;
        bool m_WeekdayRestricted = false;
        uint64_t m_EveryMillis = 0; // non-zero for "@every"
    };
} // namespace eOps::Descriptor
