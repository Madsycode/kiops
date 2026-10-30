#include <eOps/Harness/Scheduler.h>
#include <eCore/Common/Stdout.h>
#include <stdexcept>

namespace eOps::Harness
{
    namespace
    {
        uint64_t UnixMillis(std::chrono::system_clock::time_point at)
        {
            auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(at.time_since_epoch()).count();
            return millis < 0 ? 0 : static_cast<uint64_t>(millis);
        }

        std::string Describe(std::chrono::milliseconds period)
        {
            if (period.count() % 1000 == 0)
            {
                return "@every " + std::to_string(period.count() / 1000) + "s";
            }
            return "@every " + std::to_string(period.count()) + "ms";
        }
    } // namespace

    Json ScheduleEntry::ToJson() const
    {
        return Json::MakeObject({{"service", m_Service},
                                 {"job", m_Job},
                                 {"schedule", m_Expression},
                                 {"next", UnixMillis(m_Next)},
                                 {"fired", m_Fired}});
    }

    Scheduler::Scheduler(Fire fire) : m_Fire(std::move(fire))
    {
    }

    void Scheduler::AddCron(std::string service, std::string job, std::string_view expression,
                            std::chrono::system_clock::time_point now)
    {
        if (service.empty() || job.empty())
        {
            throw std::invalid_argument("a schedule needs a service and a job");
        }
        Job entry;
        entry.m_Cron = Descriptor::CronSchedule::Parse(expression);
        entry.m_Entry.m_Service = service;
        entry.m_Entry.m_Job = job;
        entry.m_Entry.m_Expression = entry.m_Cron->Expression();
        entry.m_Entry.m_Next = entry.m_Cron->Next(now);
        m_Jobs[{std::move(service), std::move(job)}] = std::move(entry);
    }

    void Scheduler::AddInterval(std::string service, std::string job, std::chrono::milliseconds period,
                                std::chrono::system_clock::time_point now)
    {
        if (service.empty() || job.empty())
        {
            throw std::invalid_argument("a schedule needs a service and a job");
        }
        if (period.count() <= 0)
        {
            throw std::invalid_argument("an interval schedule needs a positive period");
        }
        Job entry;
        entry.m_Period = period;
        entry.m_Entry.m_Service = service;
        entry.m_Entry.m_Job = job;
        entry.m_Entry.m_Expression = Describe(period);
        entry.m_Entry.m_Next = now + period;
        m_Jobs[{std::move(service), std::move(job)}] = std::move(entry);
    }

    bool Scheduler::Remove(std::string_view service, std::string_view job)
    {
        auto found = m_Jobs.find(std::make_pair(std::string(service), std::string(job)));
        if (found == m_Jobs.end())
        {
            return false;
        }
        m_Jobs.erase(found);
        return true;
    }

    size_t Scheduler::RemoveService(std::string_view service)
    {
        size_t removed = 0;
        for (auto it = m_Jobs.begin(); it != m_Jobs.end();)
        {
            if (it->first.first == service)
            {
                it = m_Jobs.erase(it);
                ++removed;
            }
            else
            {
                ++it;
            }
        }
        return removed;
    }

    size_t Scheduler::Tick(std::chrono::system_clock::time_point now)
    {
        std::vector<std::pair<std::string, std::string>> due;
        for (const auto &[key, job] : m_Jobs)
        {
            if (job.m_Entry.m_Next <= now)
            {
                due.push_back(key);
            }
        }
        size_t fired = 0;
        for (const auto &key : due)
        {
            auto found = m_Jobs.find(key); // a previous Fire may have removed it
            if (found == m_Jobs.end())
            {
                continue;
            }
            Job &job = found->second;
            // missed firings collapse into one: the next firing lies strictly after `now`
            if (job.m_Cron)
            {
                job.m_Entry.m_Next = job.m_Cron->Next(now);
            }
            else
            {
                auto behind = (now - job.m_Entry.m_Next) / job.m_Period + 1;
                job.m_Entry.m_Next += job.m_Period * behind;
            }
            ++job.m_Entry.m_Fired;
            ++fired;
            ScheduleEntry entry = job.m_Entry;
            if (!m_Fire)
            {
                continue;
            }
            try
            {
                m_Fire(entry, now);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("Scheduler: {}/{} trigger failed: {}", entry.m_Service, entry.m_Job, error.what());
            }
        }
        return fired;
    }

    std::optional<std::chrono::system_clock::time_point> Scheduler::NextDue() const
    {
        std::optional<std::chrono::system_clock::time_point> next;
        for (const auto &[key, job] : m_Jobs)
        {
            if (!next || job.m_Entry.m_Next < *next)
            {
                next = job.m_Entry.m_Next;
            }
        }
        return next;
    }

    std::vector<ScheduleEntry> Scheduler::Entries() const
    {
        std::vector<ScheduleEntry> entries;
        for (const auto &[key, job] : m_Jobs)
        {
            entries.push_back(job.m_Entry);
        }
        return entries;
    }

    Json Scheduler::ToJson() const
    {
        Json entries = Json::MakeArray();
        for (const auto &[key, job] : m_Jobs)
        {
            entries.Push(job.m_Entry.ToJson());
        }
        return entries;
    }
} // namespace eOps::Harness
