#include <eOps/Workload/Boundary.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eOps/Common/Json.h>
#include <iterator>
#include <cmath>

namespace eOps::Workload
{
    namespace
    {
        constexpr auto RATE_WINDOW = std::chrono::seconds(60);
        // concrete topics tracked for rate limits; wildcard domains could otherwise grow without bound
        constexpr size_t MAX_TRACKED_TOPICS = 4096;

        std::string Number(double value)
        {
            return Json(value).Dump();
        }
    } // namespace

    ActionBoundary::ActionBoundary(const Descriptor::RichMLService &service) : m_Actionables(service.m_Actionables)
    {
    }

    ActionDecision ActionBoundary::Check(std::string_view topic, double value,
                                         std::chrono::steady_clock::time_point now)
    {
        ActionDecision decision;
        decision.m_Value = value;
        if (!ValidTopic(topic, false) || IsFilter(topic))
        {
            decision.m_Reason = "'" + std::string(topic) + "' is not a concrete topic";
            decision.m_Rule = "domain";
            return decision;
        }
        // an exact declaration wins over a covering permission domain (as RichMLService::FindActionable)
        const Descriptor::Actionable *declared = nullptr;
        for (const Descriptor::Actionable &actionable : m_Actionables)
        {
            if (actionable.m_Name == topic)
            {
                declared = &actionable;
                break;
            }
        }
        for (size_t i = 0; !declared && i < m_Actionables.size(); ++i)
        {
            if (FilterCovers(m_Actionables[i].m_Name, topic))
            {
                declared = &m_Actionables[i];
            }
        }
        if (!declared)
        {
            decision.m_Reason = "'" + std::string(topic) + "' is not a declared actionable";
            decision.m_Rule = "domain";
            return decision;
        }
        if (!std::isfinite(value))
        {
            decision.m_Reason = "action value is not finite";
            decision.m_Rule = "value";
            return decision;
        }
        if (declared->m_Min && value < *declared->m_Min)
        {
            decision.m_Value = *declared->m_Min;
            decision.m_Clamped = true;
        }
        else if (declared->m_Max && value > *declared->m_Max)
        {
            decision.m_Value = *declared->m_Max;
            decision.m_Clamped = true;
        }
        if (decision.m_Clamped)
        {
            decision.m_Reason =
                Number(value) + " clamped to " + Number(decision.m_Value) + " (bounds of '" + declared->m_Name + "')";
        }

        if (declared->m_MaxRatePerMinute > 0)
        {
            auto recent = m_Recent.find(topic);
            if (recent == m_Recent.end())
            {
                if (m_Recent.size() >= MAX_TRACKED_TOPICS)
                {
                    // drop topics whose history has fully expired before refusing to track a new one
                    for (auto it = m_Recent.begin(); it != m_Recent.end();)
                    {
                        it = it->second.empty() || now - it->second.back() >= RATE_WINDOW ? m_Recent.erase(it)
                                                                                          : std::next(it);
                    }
                    if (m_Recent.size() >= MAX_TRACKED_TOPICS)
                    {
                        decision.m_Clamped = false;
                        decision.m_Value = value;
                        decision.m_Reason = "rate tracking capacity exhausted";
                        decision.m_Rule = "rate";
                        return decision;
                    }
                }
                recent =
                    m_Recent.emplace(std::string(topic), std::deque<std::chrono::steady_clock::time_point>()).first;
            }
            auto &times = recent->second;
            while (!times.empty() && now - times.front() >= RATE_WINDOW)
            {
                times.pop_front();
            }
            if (times.size() >= declared->m_MaxRatePerMinute)
            {
                decision.m_Clamped = false;
                decision.m_Value = value;
                decision.m_Reason = "rate limit of " + std::to_string(declared->m_MaxRatePerMinute) +
                                    " actions per minute exceeded on '" + std::string(topic) + "'";
                decision.m_Rule = "rate";
                return decision;
            }
            times.push_back(now);
        }
        decision.m_Allowed = true;
        return decision;
    }
} // namespace eOps::Workload
