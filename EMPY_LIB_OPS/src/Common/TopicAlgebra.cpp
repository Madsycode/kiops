#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Network/Topic.h>
#include <optional>
#include <vector>

namespace eOps
{
    namespace
    {
        using Levels = std::vector<std::string>;

        // canonical filter levels with any shared prefix removed; nullopt when invalid
        std::optional<Levels> FilterLevels(std::string_view filter)
        {
            try
            {
                std::string canonical = eCore::MMQP::ParseTopic(filter, true).m_Filter;
                Levels levels;
                size_t start = 0;
                while (true)
                {
                    size_t end = canonical.find('/', start);
                    levels.push_back(
                        canonical.substr(start, end == std::string::npos ? std::string::npos : end - start));
                    if (end == std::string::npos)
                    {
                        return levels;
                    }
                    start = end + 1;
                }
            }
            catch (const std::exception &)
            {
                return std::nullopt;
            }
        }

        bool Wildcard(const std::string &level) noexcept
        {
            return level == "+" || level == "#";
        }

        bool Reserved(const std::string &level) noexcept
        {
            return !level.empty() && level.front() == '$';
        }

        bool Overlap(const Levels &a, size_t i, const Levels &b, size_t j)
        {
            while (true)
            {
                // '#' matches the remaining levels of the other side, including none
                if ((i < a.size() && a[i] == "#") || (j < b.size() && b[j] == "#"))
                {
                    return true;
                }
                if (i == a.size() || j == b.size())
                {
                    return i == a.size() && j == b.size();
                }
                if (a[i] != "+" && b[j] != "+" && a[i] != b[j])
                {
                    return false;
                }
                ++i;
                ++j;
            }
        }

        bool Covers(const Levels &outer, const Levels &inner)
        {
            size_t i = 0;
            size_t j = 0;
            while (true)
            {
                if (i < outer.size() && outer[i] == "#")
                {
                    return true;
                }
                if (j < inner.size() && inner[j] == "#")
                {
                    return false;
                }
                if (i == outer.size() || j == inner.size())
                {
                    return i == outer.size() && j == inner.size();
                }
                if (outer[i] != "+" && (inner[j] == "+" || outer[i] != inner[j]))
                {
                    return false;
                }
                ++i;
                ++j;
            }
        }
    } // namespace

    bool FiltersOverlap(std::string_view a, std::string_view b)
    {
        auto left = FilterLevels(a);
        auto right = FilterLevels(b);
        if (!left || !right)
        {
            return false;
        }
        // root wildcards never match '$' topics, so a reserved first level needs an exact peer
        const std::string &first = left->front();
        const std::string &second = right->front();
        if ((Reserved(first) && Wildcard(second)) || (Reserved(second) && Wildcard(first)))
        {
            return false;
        }
        return Overlap(*left, 0, *right, 0);
    }

    bool FilterCovers(std::string_view outer, std::string_view inner)
    {
        auto wide = FilterLevels(outer);
        auto narrow = FilterLevels(inner);
        if (!wide || !narrow)
        {
            return false;
        }
        // topics matching inner may start with '$' only when inner names that level literally
        if (Wildcard(wide->front()) && Reserved(narrow->front()))
        {
            return false;
        }
        return Covers(*wide, *narrow);
    }

    bool IsFilter(std::string_view topic)
    {
        return topic.starts_with(eCore::MMQP::MMQP_SHARED_TOPIC_PREFIX) || eCore::MMQP::HasWildcard(topic);
    }

    bool ValidTopic(std::string_view topic, bool filter)
    {
        try
        {
            eCore::MMQP::ParseTopic(topic, filter);
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    std::string TopicLevel(std::string_view topic, size_t index)
    {
        auto levels = FilterLevels(topic);
        if (!levels || index >= levels->size())
        {
            return {};
        }
        return (*levels)[index];
    }
} // namespace eOps
