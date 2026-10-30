#pragma once
#include "../Common/Buffers.h"

// MMQP protocol

namespace eCore::MMQP
{
    // -------------------------
    // fixed header
    // -------------------------

    // control packet types (bits 7-4 of the control octet)
    enum class PacketType : uint8_t
    {
        ConReq = 0x1,
        ConAck = 0x2,
        ConSyn = 0x3,
        ConEnd = 0x4,
        ProReq = 0x5,
        ProAck = 0x6,
        ProSyn = 0x7,
        PubReq = 0x8,
        PubAck = 0x9
    };

    // complete encoded packet size limits: 1 control octet + up to 4 length octets + body
    inline constexpr uint32_t MMQP_MAX_BODY_SIZE = UINT32_MAX;
    inline constexpr uint32_t MMQP_MIN_PACKET_SIZE = 64;

    struct FrameHeader
    {
        PacketType m_Type = PacketType::ConReq;
        uint8_t m_Flags = 0;      // control packet flags (bits 1-0)
        uint8_t m_HeaderSize = 0; // 1 + w
        uint32_t m_BodySize = 0;  // remaining length
        uint64_t m_FrameSize = 0; // HeaderSize + BodySize
    };

    enum class FrameStatus : uint8_t
    {
        Complete, // header valid and the whole frame is present
        NeedMore, // header valid (or incomplete) but more bytes are required
        Invalid,  // unsupported packet type or non-minimal remaining length
        TooLarge  // frame exceeds the configured/negotiated packet size limit
    };

    // smallest remaining-length width w for a body length (RLF value is w - 1)
    EMPY_INLINE constexpr uint8_t RemainingLengthWidth(uint32_t bodySize) noexcept
    {
        return bodySize <= 0xFFu ? 1 : bodySize <= 0xFFFFu ? 2 : bodySize <= 0xFFFFFFu ? 3 : 4;
    }

    // Inspect the fixed header at the start of `data` without consuming it. Constant cost:
    // the RLF in the control octet gives the length-field width, so no continuation loop is
    // needed. The size limit is enforced from the header alone, before any body is buffered.
    EMPY_INLINE FrameStatus PeekFrame(ByteSpan data, uint32_t maxPacketSize, FrameHeader &out) noexcept
    {
        if (data.empty())
        {
            return FrameStatus::NeedMore;
        }
        uint8_t control = data[0];
        uint8_t type = control >> 4;
        if (type < 0x1 || type > 0x9)
        {
            return FrameStatus::Invalid;
        }
        uint8_t flags = control & 0x3;
        if ((type == 0x8 && flags == 0) || ((type == 0x1 || type == 0x3) && (flags & 0x1)) ||
            (type != 0x1 && type != 0x3 && type != 0x8 && flags != 0))
        {
            return FrameStatus::Invalid;
        }
        uint8_t width = static_cast<uint8_t>(((control >> 2) & 0x3) + 1);
        if (data.size() < size_t(1) + width)
        {
            return FrameStatus::NeedMore;
        }
        uint32_t body = static_cast<uint32_t>(LoadBEN(data.data() + 1, width));
        if (RemainingLengthWidth(body) != width)
        {
            return FrameStatus::Invalid;
        }
        out.m_Type = static_cast<PacketType>(type);
        out.m_Flags = control & 0x3;
        out.m_HeaderSize = static_cast<uint8_t>(1 + width);
        out.m_BodySize = body;
        out.m_FrameSize = uint64_t(out.m_HeaderSize) + body;
        if (out.m_FrameSize > maxPacketSize)
        {
            return FrameStatus::TooLarge;
        }
        return data.size() >= out.m_FrameSize ? FrameStatus::Complete : FrameStatus::NeedMore;
    }
} // namespace eCore::MMQP

namespace eCore::MMQP
{

    // topic prefixes
    inline constexpr const char *MMQP_BIRTH_STATUS_TOPIC_PREFIX = "$bwm/";
    inline constexpr const char *MMQP_LAST_STATUS_TOPIC_PREFIX = "$lwm/";
    inline constexpr const char *MMQP_SHARED_TOPIC_PREFIX = "$share/";

    // topic delimiter
    inline constexpr char MMQP_TOPIC_WILDCARD_PLUS = '+';
    inline constexpr char MMQP_TOPIC_WILDCARD_TAG = '#';
    inline constexpr char MMQP_TOPIC_DELIMETER = '/';

    // protocol configs
    inline constexpr unsigned char MMQP_STRLEN_SIZE = 1;
    inline constexpr unsigned char MMQP_FIXED_LEN = 1;
    inline constexpr unsigned char MMQP_TOKEN_SIZE = 32; // CONREQ token and CONSYN HMAC-SHA256 response

    // protocol alias
    inline constexpr const char *MMQP_ALIAS = "MMQP";
    inline constexpr unsigned char MMQPv1 = 0x01;
    using _byte_ = unsigned char;

    // control packets
    inline constexpr _byte_ MMQP_PACKET_CONREQ = 0x10;
    inline constexpr _byte_ MMQP_PACKET_CONACK = 0x20;
    inline constexpr _byte_ MMQP_PACKET_CONSYN = 0x30;
    inline constexpr _byte_ MMQP_PACKET_CONEND = 0x40;
    inline constexpr _byte_ MMQP_PACKET_PROREQ = 0x50;
    inline constexpr _byte_ MMQP_PACKET_PROACK = 0x60;
    inline constexpr _byte_ MMQP_PACKET_PROSYN = 0x70;
    inline constexpr _byte_ MMQP_PACKET_PUBREQ = 0x80;
    inline constexpr _byte_ MMQP_PACKET_PUBACK = 0x90;

    // control masks
    inline constexpr _byte_ MMQP_PACKET_TYPE_MASK = 0xF0;
    inline constexpr _byte_ MMQP_RLEN_FLAGS_MASK = 0x0C;
    inline constexpr _byte_ MMQP_CTRL_FLAGS_MASK = 0x03;
    inline constexpr _byte_ MMQP_QOS_LEVEL_MASK = 0x03;

    // conreq flags
    inline constexpr _byte_ MMQP_CONREQ_FLAGS_RES = 0x02;

    // consyn flags
    inline constexpr _byte_ MMQP_CONSYN_FLAGS_TOK = 0x02;

    // session flags
    inline constexpr _byte_ MMQP_SESSION_FLAGS_SAF = 0x80;
    inline constexpr _byte_ MMQP_SESSION_FLAGS_SPF = 0x40;
    inline constexpr _byte_ MMQP_SESSION_FLAGS_QPF = 0x20;
    inline constexpr _byte_ MMQP_SESSION_FLAGS_SWF = 0x10;
    inline constexpr _byte_ MMQP_SESSION_FLAGS_TPF = 0x08;

    // authen flags
    inline constexpr _byte_ MMQP_AUTHEN_FLAGS_SID = 0x80;
    inline constexpr _byte_ MMQP_AUTHEN_FLAGS_TOK = 0x40;

    // policy flags
    inline constexpr _byte_ MMQP_POLICY_FLAGS_IRS = 0x80;
    inline constexpr _byte_ MMQP_POLICY_FLAGS_SEI = 0x40;
    inline constexpr _byte_ MMQP_POLICY_FLAGS_KAI = 0x20;
    inline constexpr _byte_ MMQP_POLICY_FLAGS_MPS = 0x10;

    // wills flags
    inline constexpr _byte_ MMQP_STATUS_FLAGS_BSM = 0x80;
    inline constexpr _byte_ MMQP_STATUS_FLAGS_LSM = 0x40;
    inline constexpr _byte_ MMQP_STATUS_FLAGS_TOP = 0x20;
    inline constexpr _byte_ MMQP_STATUS_FLAGS_BWM = 0x80;
    inline constexpr _byte_ MMQP_STATUS_FLAGS_LWM = 0x40;

    // quic policy flags and allocation strategies (README.pdf, Section 1.3.1.1)
    inline constexpr _byte_ MMQP_QUIC_FLAGS_SAS = 0x80;
    inline constexpr _byte_ MMQP_QUIC_FLAGS_BTS = 0x40;
    inline constexpr _byte_ MMQP_QUIC_FLAGS_QDG = 0x20;
    inline constexpr _byte_ MMQP_QUIC_STRATEGY_CUSTOM = 0x00;
    inline constexpr _byte_ MMQP_QUIC_STRATEGY_SINGLE = 0x01;
    inline constexpr _byte_ MMQP_QUIC_STRATEGY_SEMANTIC = 0x02;
    inline constexpr _byte_ MMQP_QUIC_STRATEGY_BULK = 0x03;

    // profile flags
    inline constexpr _byte_ MMQP_PROREQ_FLAGS_TOP = 0x20;
    inline constexpr _byte_ MMQP_PROREQ_FLAGS_REQ = 0x10;
    inline constexpr _byte_ MMQP_PROREQ_TYPE_MASK = 0xC0;
    inline constexpr _byte_ MMQP_PROREQ_TYPE_UNS = 0xC0;
    inline constexpr _byte_ MMQP_PROREQ_TYPE_REG = 0x80;
    inline constexpr _byte_ MMQP_PROREQ_TYPE_SUB = 0x40;

    // publish flags
    inline constexpr _byte_ MMQP_PUBREQ_FLAGS_DUP = 0x80;
    inline constexpr _byte_ MMQP_PUBREQ_FLAGS_E2E = 0x40;
    inline constexpr _byte_ MMQP_PUBREQ_FLAGS_TOP = 0x20;
    inline constexpr _byte_ MMQP_PUBREQ_TYPE_MASK = 0x03;
    inline constexpr _byte_ MMQP_PUBREQ_TYPE_RES = 0x03;
    inline constexpr _byte_ MMQP_PUBREQ_TYPE_REQ = 0x02;
    inline constexpr _byte_ MMQP_PUBREQ_TYPE_MSG = 0x01;

    // success codes
    inline constexpr _byte_ MMQP_REASON_SUCCESS = 0x00;
    inline constexpr _byte_ MMQP_REASON_NOSUBSCRIBER = 0x01;
    inline constexpr _byte_ MMQP_REASON_ALIAS_MAPPED = 0x02;
    inline constexpr _byte_ MMQP_REASON_UNSPECIFIED = 0x80;
    inline constexpr _byte_ MMQP_REASON_AUTH_FAILED = 0x81;
    inline constexpr _byte_ MMQP_REASON_BANNED = 0x82;
    inline constexpr _byte_ MMQP_REASON_POLICY_VIOLATION = 0x83;
    inline constexpr _byte_ MMQP_REASON_MALFORMED_PACKET = 0x84;
    inline constexpr _byte_ MMQP_REASON_SESSION_EXPIRED = 0x85;
    inline constexpr _byte_ MMQP_REASON_UNAUTHORIZED = 0x86;
    inline constexpr _byte_ MMQP_REASON_KEEPALIVE_TIMEOUT = 0x87;
    inline constexpr _byte_ MMQP_REASON_UNSUPPORTED_VERSION = 0x88;
    inline constexpr _byte_ MMQP_REASON_INVALID_TOPIC = 0x89;
    inline constexpr _byte_ MMQP_REASON_INVALID_ALIAS = 0x8A;
    inline constexpr _byte_ MMQP_REASON_INVALID_PID = 0x8B;
    inline constexpr _byte_ MMQP_REASON_INTERNAL_ERROR = 0x8C;
    inline constexpr _byte_ MMQP_REASON_NO_RESPONDER = 0x8D;
    inline constexpr _byte_ MMQP_REASON_BUSY = 0x8E;
    inline constexpr _byte_ MMQP_REASON_TIMEOUT = 0x8F;
    inline constexpr _byte_ MMQP_REASON_INVALID_CORRELATION = 0x90;

} // namespace eCore::MMQP
