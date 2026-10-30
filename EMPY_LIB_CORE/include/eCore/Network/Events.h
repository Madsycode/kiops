#pragma once
#include "Codecs.h"

namespace eCore::MMQP
{
    struct Connect
    {
        std::string m_Identifier;
        bool m_Resumed = false;
    };

    struct Disconnect
    {
        Reason m_Reason;
        bool m_Reconnecting = false;
    };
    struct Receive
    {
        PublishRequest m_Message;
        bool m_Decrypted = false;
    };
    struct Profile
    {
        std::string m_Topic;
        uint16_t m_PacketId = 0;
        Reason m_Reason;
    };
    struct Delivery
    {
        uint16_t m_PacketId = 0;
        Reason m_Reason;
    };
    struct Publish
    {
        PublishRequest m_Message;
        Reason m_Reason;
    };
    struct Session
    {
        std::string m_Identifier;
        bool m_Connected = false;
        bool m_Resumed = false;
        Reason m_Reason;
    };
    struct SessionConnected
    {
        std::string m_Identifier;
        bool m_Resumed = false;
    };
    struct SessionDisconnected
    {
        std::string m_Identifier;
        std::string m_Reason;
    };
    struct PublicationRouted
    {
        std::string m_Topic;
        size_t m_Recipients = 0;
        size_t m_Bytes = 0;
    };
} // namespace eCore::MMQP
