#pragma once
#include <eCore/API.h>
#include <eCore/Network/Events.h>
#include "Configs.h"

namespace eMQ
{
    using namespace eCore::MMQP;
    struct EMPY_MQ_API Service : eCore::ImplService<eCore::CLIENT>
    {
        explicit Service(Configs options = {});
        explicit Service(const char *filename);
        ~Service() override;

        std::future<std::string> AsyncConnect(std::string host, uint16_t port, bool resume = false);
        std::string Connect(std::string host, uint16_t port, bool resume = false);
        void Disconnect(bool graceful = true);
        void Stop();
        bool Connected() const;
        std::string Identifier();

        std::future<Reason> UpdateProfile(std::vector<ProfileEntry> entries);
        std::future<Reason> Register(std::string topic, uint8_t retain = 0, uint32_t expiry = 60, uint8_t queue = 10,
                                     bool requests = false);
        std::future<Reason> Subscribe(std::string topic, uint8_t qos = 1, bool responder = false);
        std::future<Reason> Unsubscribe(std::string topic);

        std::future<Reason> Publish(std::string topic, eCore::SharedBytes payload, uint8_t qos = 0);
        // Zero selects an unused correlation ID. Supply an explicit ID to match
        // concurrent calls before their Receive events arrive. At QoS 1 the
        // future reports PUBACK; at QoS 0 it reports local send admission.
        // Application responses arrive through Receive.
        std::future<Reason> Request(std::string topic, eCore::SharedBytes payload, uint8_t responses = 1,
                                    uint8_t qos = 1, uint16_t correlationId = 0);
        std::future<Reason> Respond(std::string topic, uint16_t correlationId, eCore::SharedBytes payload,
                                    Reason reason = {}, uint8_t qos = 1);
        std::future<Reason> Respond(const PublishRequest &request, eCore::SharedBytes payload, Reason reason = {});

        std::future<Reason> Send(PublishRequest publication);

        using eCore::ImplService<eCore::CLIENT>::On;
        using eCore::ImplService<eCore::CLIENT>::Off;

        template <typename _Event, typename _Callback> eCore::ListenerId On(std::string topic, _Callback &&callback)
        {
            static_assert(std::is_same_v<_Event, eMQ::Receive> || std::is_same_v<_Event, eMQ::Publish> ||
                          std::is_same_v<_Event, eMQ::Profile>);
            auto filter = ParseTopic(topic, true).m_Filter;
            return eCore::ImplService<eCore::CLIENT>::On<_Event>(
                [filter = std::move(filter), fn = std::forward<_Callback>(callback)](const _Event &event)
                {
                    if constexpr (std::is_same_v<_Event, eMQ::Receive> || std::is_same_v<_Event, eMQ::Publish>)
                    {
                        if (TopicMatches(filter, event.m_Message.m_Topic))
                        {
                            fn(event);
                        }
                    }
                    else
                    {
                        if (filter == event.topic || (!HasWildcard(event.topic) && TopicMatches(filter, event.topic)))
                        {
                            fn(event);
                        }
                    }
                });
        }

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eMQ
