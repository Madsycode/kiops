#pragma once
#include <map>
#include <chrono>
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <optional>
#include <functional>
#include <string_view>
#include "../Common/Json.h"
#include "../Descriptor/Cron.h"

// temporal trigger source: cron jobs (training schedules) and fixed-interval jobs (placement
// audits), keyed by (service, job). Missed firings collapse into one. Owner thread only; the
// harness drives Tick from OnUpdate and a timer.

namespace eOps::Harness
{
    struct EMPY_OPS_API ScheduleEntry
    {
        std::string m_Service;
        std::string m_Job;        // training | placement | ...
        std::string m_Expression; // cron expression or "@every <N>ms"-style description
        std::chrono::system_clock::time_point m_Next;
        uint64_t m_Fired = 0;
        Json ToJson() const; // {service, job, schedule, next (Unix ms), fired}
    };

    class EMPY_OPS_API Scheduler
    {
    public:
        using Fire = std::function<void(const ScheduleEntry &entry, std::chrono::system_clock::time_point firedAt)>;

        explicit Scheduler(Fire fire);

        // insert or replace; throws std::invalid_argument for a malformed expression
        void AddCron(std::string service, std::string job, std::string_view expression,
                     std::chrono::system_clock::time_point now = std::chrono::system_clock::now());
        void AddInterval(std::string service, std::string job, std::chrono::milliseconds period,
                         std::chrono::system_clock::time_point now = std::chrono::system_clock::now());
        bool Remove(std::string_view service, std::string_view job);
        size_t RemoveService(std::string_view service);

        // fires every due entry once and advances it past `now`; returns the number fired
        size_t Tick(std::chrono::system_clock::time_point now = std::chrono::system_clock::now());
        std::optional<std::chrono::system_clock::time_point> NextDue() const;
        std::vector<ScheduleEntry> Entries() const; // ordered by (service, job)
        Json ToJson() const;                        // [entries]

    private:
        struct Job
        {
            ScheduleEntry m_Entry;
            std::optional<Descriptor::CronSchedule> m_Cron;
            std::chrono::milliseconds m_Period{0};
        };

        Fire m_Fire;
        std::map<std::pair<std::string, std::string>, Job, std::less<>> m_Jobs;
    };
} // namespace eOps::Harness
