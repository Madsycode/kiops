#include <eOps/Descriptor/Profiles.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Network/Topic.h>
#include <algorithm>

namespace eOps::Descriptor
{
    namespace
    {
        // publication registrations mirror eMQ::Service::Register defaults (no retain, 60 s expiry, queue 10)
        constexpr uint32_t REGISTRATION_EXPIRY_SECONDS = 60;
        constexpr uint8_t REGISTRATION_QUEUE = 10;

        bool Listed(const std::vector<eCore::MMQP::ProfileEntry> &entries, const std::string &topic)
        {
            return std::any_of(entries.begin(), entries.end(),
                               [&](const eCore::MMQP::ProfileEntry &entry) { return entry.m_Topic == topic; });
        }
    } // namespace

    std::vector<eCore::MMQP::ProfileEntry> SubscriptionProfiles(const RichMLService &service)
    {
        std::vector<eCore::MMQP::ProfileEntry> entries;
        for (const Observable &observable : service.m_Observables)
        {
            if (!Listed(entries, observable.m_Name))
            {
                eCore::MMQP::ProfileEntry entry;
                entry.m_Type = eCore::MMQP::MMQP_PROREQ_TYPE_SUB;
                entry.m_Topic = observable.m_Name;
                entry.m_Qos = observable.m_Qos;
                entries.push_back(std::move(entry));
            }
        }
        if (service.m_Training && !service.m_Training->m_Label.empty() && !Listed(entries, service.m_Training->m_Label))
        {
            eCore::MMQP::ProfileEntry entry;
            entry.m_Type = eCore::MMQP::MMQP_PROREQ_TYPE_SUB;
            entry.m_Topic = service.m_Training->m_Label;
            entry.m_Qos = 1;
            entries.push_back(std::move(entry));
        }
        return entries;
    }

    std::vector<eCore::MMQP::ProfileEntry> PublicationProfiles(const RichMLService &service)
    {
        std::vector<eCore::MMQP::ProfileEntry> entries;
        for (const Actionable &actionable : service.m_Actionables)
        {
            if (!ValidTopic(actionable.m_Name, false) || IsFilter(actionable.m_Name) ||
                Listed(entries, actionable.m_Name))
            {
                continue;
            }
            eCore::MMQP::ProfileEntry entry;
            entry.m_Type = eCore::MMQP::MMQP_PROREQ_TYPE_REG;
            entry.m_Topic = actionable.m_Name;
            entry.m_Qos = 0; // REG entries carry the retain flag in this field
            entry.m_Request = false;
            entry.m_Expiry = eCore::MMQP::EncodeInterval(REGISTRATION_EXPIRY_SECONDS);
            entry.m_Queue = REGISTRATION_QUEUE;
            entries.push_back(std::move(entry));
        }
        return entries;
    }
} // namespace eOps::Descriptor
