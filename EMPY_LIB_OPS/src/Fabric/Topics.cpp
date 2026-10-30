#include <eOps/Fabric/Topics.h>
#include <eOps/Common/TopicAlgebra.h>
#include <stdexcept>

namespace eOps::Fabric
{
    namespace
    {
        // profile topics are length-prefixed by one byte on the wire
        constexpr size_t MAX_TOPIC = 255;
        constexpr size_t MAX_ROOT = 128;

        // one concrete level supplied by a caller (service, role, task, ...)
        std::string_view Level(std::string_view level, const char *what)
        {
            if (level.empty() || level.find_first_of("/+#") != std::string_view::npos ||
                level.find('\0') != std::string_view::npos)
            {
                throw std::invalid_argument(std::string("Invalid topic level for ") + what + ": '" +
                                            std::string(level) + "'");
            }
            return level;
        }

        std::string Bounded(std::string topic)
        {
            if (topic.size() > MAX_TOPIC)
            {
                throw std::invalid_argument("Topic exceeds 255 bytes: " + topic.substr(0, 64) + "...");
            }
            return topic;
        }

        // levels of a topic ("a/b" → a, b)
        std::vector<std::string> SplitLevels(std::string_view topic)
        {
            std::vector<std::string> levels;
            size_t start = 0;
            while (true)
            {
                size_t end = topic.find('/', start);
                levels.emplace_back(
                    topic.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
                if (end == std::string_view::npos)
                {
                    return levels;
                }
                start = end + 1;
            }
        }

        std::string Compose(const std::string &root, std::initializer_list<std::string_view> levels)
        {
            std::string topic = root;
            for (auto level : levels)
            {
                topic += '/';
                topic += level;
            }
            return Bounded(std::move(topic));
        }
    } // namespace

    Topics::Topics(std::string root)
    {
        while (!root.empty() && root.back() == '/')
        {
            root.pop_back();
        }
        if (root.empty() || root.size() > MAX_ROOT || root.front() == '$' || IsFilter(root) || !ValidTopic(root, false))
        {
            throw std::invalid_argument("Invalid MLOps topic root: '" + root + "'");
        }
        for (const auto &level : SplitLevels(root))
        {
            if (level.empty())
            {
                throw std::invalid_argument("MLOps topic root has an empty level: '" + root + "'");
            }
        }
        m_Root = std::move(root);
    }

    std::string Topics::Intent() const
    {
        return Compose(m_Root, {"intent"});
    }

    std::string Topics::Deploy() const
    {
        return Compose(m_Root, {"deploy"});
    }

    std::string Topics::Workflow() const
    {
        return Compose(m_Root, {"harness", "workflow"});
    }

    std::string Topics::Status() const
    {
        return Compose(m_Root, {"harness", "status"});
    }

    std::string Topics::Agent(std::string_view role, std::string_view task) const
    {
        return Compose(m_Root, {"agent", Level(role, "role"), Level(task, "task")});
    }

    std::string Topics::AgentFilter(std::string_view role) const
    {
        return Compose(m_Root, {"agent", Level(role, "role"), "+"});
    }

    std::string Topics::Trigger(std::string_view service, std::string_view job) const
    {
        return Compose(m_Root, {"trigger", "time", Level(service, "service"), Level(job, "job")});
    }

    std::string Topics::TriggerFilter(std::string_view job) const
    {
        return Compose(m_Root, {"trigger", "time", "+", Level(job, "job")});
    }

    std::string Topics::Event(std::string_view service, std::string_view kind) const
    {
        return Compose(m_Root, {"event", Level(service, "service"), Level(kind, "event kind")});
    }

    std::string Topics::EventFilter(std::string_view kind) const
    {
        return Compose(m_Root, {"event", "+", Level(kind, "event kind")});
    }

    std::string Topics::Heartbeat(std::string_view service, std::string_view instance) const
    {
        return Compose(m_Root, {"status", Level(service, "service"), Level(instance, "instance")});
    }

    std::string Topics::HeartbeatFilter() const
    {
        return Compose(m_Root, {"status", "+", "+"});
    }

    std::string Topics::Workload(std::string_view service, std::string_view command) const
    {
        return Compose(m_Root, {"workload", Level(service, "service"), Level(command, "command")});
    }

    std::string Topics::WorkloadFilter(std::string_view service) const
    {
        return Compose(m_Root, {"workload", Level(service, "service"), "+"});
    }

    std::string Topics::Candidate(std::string_view service) const
    {
        return Compose(m_Root, {"model", Level(service, "service"), "candidate"});
    }

    std::string Topics::CandidateFilter() const
    {
        return Compose(m_Root, {"model", "+", "candidate"});
    }

    std::string Topics::Hitl() const
    {
        return Compose(m_Root, {"hitl", "approval"});
    }

    std::string Topics::State(std::string_view name) const
    {
        return Compose(m_Root, {"state", Level(name, "state name")});
    }

    std::string Topics::Locality(std::string_view region) const
    {
        return Compose(m_Root, {"telemetry", "locality", Level(region, "region")});
    }

    std::string Topics::LocalityFilter() const
    {
        return Compose(m_Root, {"telemetry", "locality", "+"});
    }

    std::string Topics::All() const
    {
        return Compose(m_Root, {"#"});
    }

    std::optional<std::vector<std::string>> Topics::Relative(std::string_view topic) const
    {
        if (topic == m_Root)
        {
            return std::vector<std::string>{};
        }
        if (topic.size() <= m_Root.size() + 1 || topic.substr(0, m_Root.size()) != m_Root ||
            topic[m_Root.size()] != '/')
        {
            return std::nullopt;
        }
        return SplitLevels(topic.substr(m_Root.size() + 1));
    }
} // namespace eOps::Fabric
