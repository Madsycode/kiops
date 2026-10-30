#pragma once
#include "Topic.h"
#include <variant>

namespace eCore::MMQP
{
    struct Reason
    {
        uint8_t m_Code = MMQP_REASON_SUCCESS;
        std::string m_Text;
        explicit operator bool() const noexcept
        {
            return m_Code >= 0x80;
        }
    };

    struct ProfileEntry
    {
        uint8_t m_Type = MMQP_PROREQ_TYPE_SUB;
        uint16_t m_Alias = 0;
        std::string m_Topic;
        uint8_t m_Qos = 0;
        bool m_Request = false;
        uint8_t m_Expiry = 10;
        uint8_t m_Queue = 10;
    };

    struct ProfileRequest
    {
        uint16_t m_PacketId = 0;
        std::vector<ProfileEntry> m_Entries;
    };

    struct SessionAuth
    {
        std::string m_Identifier;
        Buffer m_Token;
    };

    struct SessionPolicy
    {
        bool m_Reasons = false;
        std::optional<uint8_t> m_Expiry;
        std::optional<uint8_t> m_Keepalive;
        std::optional<uint32_t> m_MaxPacket;
    };

    struct BulkPolicy
    {
        uint8_t m_Streams = 1;
        uint32_t m_Threshold = 65536;
        bool operator==(const BulkPolicy &) const = default;
    };

    struct QuicPolicy
    {
        std::optional<uint8_t> m_Strategy;
        std::optional<BulkPolicy> m_Bulk;
        std::optional<uint32_t> m_Datagram;
        bool operator==(const QuicPolicy &) const = default;
    };

    struct SessionWill
    {
        std::optional<Buffer> m_Birth;
        std::optional<Buffer> m_Last;
        std::string m_Topic;
        uint8_t m_Qos = 0;
    };

    struct SessionProperties
    {
        std::optional<SessionAuth> m_Auth;
        std::optional<SessionPolicy> m_Policy;
        std::optional<QuicPolicy> m_Quic;
        std::optional<SessionWill> m_Will;
        std::optional<ProfileRequest> m_Profile;
    };

    struct ConnectionRequest
    {
        bool m_Resume = false;
        uint8_t m_Version = MMQPv1;
        SessionProperties m_Properties;
    };

    struct ConnectionAck
    {
        Reason m_Reason;
        SessionProperties m_Properties;
    };

    struct ConnectionSync
    {
        std::optional<Buffer> m_Token;
    };

    struct ConnectionEnd
    {
        Reason m_Reason;
    };

    struct ProfileAck
    {
        uint16_t m_PacketId = 0;
        Reason m_Reason;
    };

    struct ProfileSync
    {
        uint16_t m_Alias = 0;
        std::string m_Topic;
        Reason m_Reason;
    };

    struct EncryptionHeader
    {
        uint32_t m_Epoch = 0;
        std::array<uint8_t, 16> m_Nonce{};
        std::array<uint8_t, 16> m_Tag{};
    };

    struct PublishRequest
    {
        uint8_t m_Type = MMQP_PUBREQ_TYPE_MSG;
        uint16_t m_Alias = 0;
        std::string m_Topic;
        uint8_t m_Qos = 0;
        uint16_t m_PacketId = 0;
        bool m_Duplicate = false;
        uint16_t m_CorrelationId = 0; // Mandatory for PRT REQ/RES, independent of the QoS packet ID.
        uint8_t m_Responses = 1;
        Reason m_Reason;
        std::optional<EncryptionHeader> m_Encryption;
        SharedBytes m_Payload;
    };

    struct PublishAck
    {
        uint16_t m_PacketId = 0;
        Reason m_Reason;
    };

    using Message = std::variant<ConnectionRequest, ConnectionAck, ConnectionSync, ConnectionEnd, ProfileRequest,
                                 ProfileAck, ProfileSync, PublishRequest, PublishAck>;

    struct CodecOptions
    {
        uint32_t m_MaxPacket = 1024 * 1024;
        uint32_t m_MaxProfiles = 256;
        bool m_Reasons = false;

        CodecOptions(bool includeReasons = false, uint32_t packetLimit = 1024 * 1024, uint32_t profileLimit = 256)
            : m_MaxPacket(packetLimit), m_MaxProfiles(profileLimit), m_Reasons(includeReasons)
        {
        }
    };

    // Routing view of an encoded frame; the topic refers into the frame bytes.
    struct PacketSummary
    {
        uint8_t m_Type = 0;
        uint8_t m_Qos = 0;
        size_t m_Payload = 0;
        std::string_view m_Topic;
        EMPY_INLINE bool Publication() const
        {
            return m_Type == MMQP_PACKET_PUBREQ;
        }
    };

    struct EMPY_CORE_API Codec
    {
        static Message Decode(std::span<const uint8_t> bytes, CodecOptions options = {});
        // Reads the fixed and publication headers only; it does not validate the body.
        static PacketSummary Peek(std::span<const uint8_t> bytes, CodecOptions options = {});
        static Buffer Encode(const Message &message, CodecOptions options = {});
        static WireFrame EncodeFrame(const Message &message, CodecOptions options = {});
        static WireFrame EncodeFrame(const PublishRequest &message, CodecOptions options = {});
        static PacketSummary PeekFrame(const WireFrame &frame, CodecOptions options = {});
        // Encodes a publication whose payload is held elsewhere; header.payload must be empty.
        static Buffer Encode(const PublishRequest &header, CodecOptions options, std::span<const uint8_t> payload);
        static uint8_t Type(const Message &message);
    };
} // namespace eCore::MMQP
