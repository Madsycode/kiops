#include <eCore/Network/State.h>

namespace eCore::MMQP
{
    void ProfileTable::Apply(const ProfileRequest &request, size_t limit)
    {
        // Encoding validates the same field rules as the wire decoder.
        Codec::Encode(request, {false, 1024 * 1024, static_cast<uint32_t>(limit)});
        auto next = *this;
        for (auto entry : request.m_Entries)
        {
            auto &aliases = entry.m_Type == MMQP_PROREQ_TYPE_REG ? next.m_Publishers : next.m_Subscribers;
            if (entry.m_Topic.empty())
            {
                entry.m_Topic = aliases.Resolve(entry.m_Alias);
            }
            entry.m_Topic = CanonicalTopic(entry.m_Topic);
            auto pattern = ParseTopic(entry.m_Topic, true);
            if (entry.m_Type != MMQP_PROREQ_TYPE_REG && entry.m_Alias > 32767 &&
                next.m_Subscribers.Find(entry.m_Topic) != entry.m_Alias)
            {
                throw ProtocolError("New client subscription aliases must be in 1..32767", MMQP_REASON_INVALID_ALIAS);
            }
            if (entry.m_Type == MMQP_PROREQ_TYPE_UNS)
            {
                auto removed = std::erase_if(
                    next.m_Entries,
                    [&](const ProfileEntry &existing)
                    {
                        if (existing.m_Type != MMQP_PROREQ_TYPE_SUB)
                        {
                            return false;
                        }
                        auto previous = ParseTopic(existing.m_Topic, true);
                        return previous.m_Group == pattern.m_Group &&
                               (previous.m_Filter == pattern.m_Filter ||
                                (!HasWildcard(previous.m_Filter) && TopicMatches(pattern.m_Filter, previous.m_Filter)));
                    });
                if (!removed)
                {
                    throw ProtocolError("No matching subscription to remove", MMQP_REASON_INVALID_TOPIC);
                }
                continue;
            }
            if (entry.m_Alias)
            {
                aliases.Map(entry.m_Alias, entry.m_Topic, true);
            }
            auto existing = std::find_if(next.m_Entries.begin(), next.m_Entries.end(), [&](const auto &value)
                                         { return value.m_Type == entry.m_Type && value.m_Topic == entry.m_Topic; });
            if (existing != next.m_Entries.end())
            {
                *existing = std::move(entry);
            }
            else
            {
                if (next.m_Entries.size() >= limit)
                {
                    throw ProtocolError("Session profile budget exhausted", MMQP_REASON_POLICY_VIOLATION);
                }
                next.m_Entries.emplace_back(std::move(entry));
            }
        }
        *this = std::move(next);
    }

    std::optional<ProfileEntry> ProfileTable::Registration(std::string_view topic, uint8_t type) const
    {
        std::optional<ProfileEntry> best;
        size_t specificity = 0;
        for (const auto &entry : m_Entries)
        {
            bool allowed = type == MMQP_PUBREQ_TYPE_RES ? entry.m_Type == MMQP_PROREQ_TYPE_SUB && entry.m_Request
                                                        : entry.m_Type == MMQP_PROREQ_TYPE_REG &&
                                                              (type != MMQP_PUBREQ_TYPE_REQ || entry.m_Request);
            if (!allowed || !MatchTopic(entry.m_Topic, topic))
            {
                continue;
            }
            auto score = entry.m_Topic.find_first_of("+#");
            if (score == std::string::npos)
            {
                score = entry.m_Topic.size() + 256;
            }
            if (!best || score > specificity || (score == specificity && entry.m_Topic < best->m_Topic))
            {
                best = entry;
                specificity = score;
            }
        }
        return best;
    }

    std::vector<ProfileEntry> ProfileTable::Subscriptions(std::string_view topic, uint8_t qos, bool responders) const
    {
        std::vector<ProfileEntry> matches;
        this->ForEachSubscription(topic, qos, responders,
                                  [&](const ProfileEntry &entry) { matches.emplace_back(entry); });
        return matches;
    }

    bool ProfileTable::Subscribed(std::string_view topic, uint8_t qos, bool responders) const
    {
        return std::any_of(m_Entries.begin(), m_Entries.end(),
                           [&](const ProfileEntry &entry)
                           {
                               return entry.m_Type == MMQP_PROREQ_TYPE_SUB && qos <= entry.m_Qos &&
                                      (!responders || entry.m_Request) && MatchTopic(entry.m_Topic, topic);
                           });
    }
} // namespace eCore::MMQP
