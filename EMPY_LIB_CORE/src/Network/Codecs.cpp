#include <eCore/Network/Codecs.h>

namespace eCore::MMQP
{
    namespace
    {
        void Require(bool valid, const char *text, uint8_t reason = MMQP_REASON_MALFORMED_PACKET)
        {
            if (!valid)
            {
                throw ProtocolError(text, reason);
            }
        }

        void Flags(uint8_t value, uint8_t allowed)
        {
            Require((value & ~allowed) == 0, "Reserved variable flags are set");
        }

        void Pid(uint16_t value)
        {
            Require(value != 0, "Packet identifier must be nonzero", MMQP_REASON_INVALID_PID);
        }

        void Qos(uint8_t value)
        {
            Require(value <= 1, "Only QoS 0 and 1 are supported");
        }

        void Token(const Buffer &token)
        {
            Require(token.size() == MMQP_TOKEN_SIZE, "Authentication tokens contain exactly 32 bytes",
                    MMQP_REASON_AUTH_FAILED);
        }

        std::string Text(Reader &reader)
        {
            auto text = reader.String();
            Require(ValidUtf8(text), "Malformed UTF-8 text");
            return text;
        }

        void WriteReason(Writer &writer, const Reason &reason, CodecOptions options)
        {
            writer.U8(reason.m_Code);
            if (options.m_Reasons && reason.m_Code)
            {
                Require(ValidUtf8(reason.m_Text), "Malformed reason text");
                writer.String(reason.m_Text);
            }
        }

        Reason ReadReason(Reader &reader, CodecOptions options)
        {
            Reason reason{reader.U8(), {}};
            if (options.m_Reasons && reason.m_Code)
            {
                reason.m_Text = Text(reader);
            }
            return reason;
        }

        void ValidateProfile(const ProfileEntry &entry)
        {
            Require(entry.m_Type == MMQP_PROREQ_TYPE_SUB || entry.m_Type == MMQP_PROREQ_TYPE_REG ||
                        entry.m_Type == MMQP_PROREQ_TYPE_UNS,
                    "Unknown profile operation");
            Qos(entry.m_Qos);
            if (entry.m_Type == MMQP_PROREQ_TYPE_UNS)
            {
                Require(!entry.m_Qos && !entry.m_Request, "Unsubscribe has reserved flags");
            }
            if (!entry.m_Topic.empty())
            {
                auto pattern = ParseTopic(entry.m_Topic, true);
                Require(entry.m_Type != MMQP_PROREQ_TYPE_REG || pattern.m_Group.empty(),
                        "Publish registrations cannot be shared", MMQP_REASON_INVALID_TOPIC);
                Require(HasWildcard(pattern.m_Filter) ? entry.m_Alias == 0 : entry.m_Alias != 0,
                        "Only wildcard profiles use alias zero", MMQP_REASON_INVALID_ALIAS);
            }
            else
            {
                Require(entry.m_Alias != 0, "A profile requires a topic or a mapped alias", MMQP_REASON_INVALID_ALIAS);
            }
            if (entry.m_Type == MMQP_PROREQ_TYPE_REG)
            {
                DecodeInterval(entry.m_Expiry);
                Require(!entry.m_Qos || entry.m_Queue, "Retained queue length must be positive");
            }
        }

        void WriteProfiles(Writer &writer, const ProfileRequest &request, CodecOptions options)
        {
            Pid(request.m_PacketId);
            Require(!request.m_Entries.empty() && request.m_Entries.size() <= options.m_MaxProfiles,
                    "Profile count exceeds configured bounds", MMQP_REASON_POLICY_VIOLATION);
            writer.U16(request.m_PacketId);
            for (const auto &entry : request.m_Entries)
            {
                ValidateProfile(entry);
                writer.U8(entry.m_Type | (entry.m_Topic.empty() ? 0 : MMQP_PROREQ_FLAGS_TOP) |
                          (entry.m_Request ? MMQP_PROREQ_FLAGS_REQ : 0) | entry.m_Qos);
                writer.U16(entry.m_Alias);
                if (!entry.m_Topic.empty())
                {
                    writer.String(CanonicalTopic(entry.m_Topic));
                }
                if (entry.m_Type == MMQP_PROREQ_TYPE_REG)
                {
                    writer.U8(entry.m_Expiry);
                    if (entry.m_Qos)
                    {
                        writer.U8(entry.m_Queue);
                    }
                }
            }
        }

        ProfileRequest ReadProfiles(Reader &reader, CodecOptions options)
        {
            ProfileRequest request;
            request.m_PacketId = reader.U16();
            Pid(request.m_PacketId);
            while (!reader.Empty())
            {
                Require(request.m_Entries.size() < options.m_MaxProfiles, "Too many profile entries",
                        MMQP_REASON_POLICY_VIOLATION);
                auto flags = reader.U8();
                Flags(flags, 0xF3);
                ProfileEntry entry;
                entry.m_Type = flags & MMQP_PROREQ_TYPE_MASK;
                entry.m_Qos = flags & MMQP_QOS_LEVEL_MASK;
                entry.m_Request = flags & MMQP_PROREQ_FLAGS_REQ;
                entry.m_Alias = reader.U16();
                if (flags & MMQP_PROREQ_FLAGS_TOP)
                {
                    entry.m_Topic = CanonicalTopic(Text(reader));
                }
                if (entry.m_Type == MMQP_PROREQ_TYPE_REG)
                {
                    entry.m_Expiry = reader.U8();
                    if (entry.m_Qos)
                    {
                        entry.m_Queue = reader.U8();
                    }
                }
                ValidateProfile(entry);
                request.m_Entries.emplace_back(std::move(entry));
            }
            Require(!request.m_Entries.empty(), "Empty profile request");
            return request;
        }

        void ValidateQuic(const QuicPolicy &policy)
        {
            if (policy.m_Strategy)
            {
                auto value = *policy.m_Strategy;
                Require(value == MMQP_QUIC_STRATEGY_CUSTOM || value == MMQP_QUIC_STRATEGY_SINGLE ||
                            value == MMQP_QUIC_STRATEGY_SEMANTIC || value == MMQP_QUIC_STRATEGY_BULK,
                        "Unknown QUIC allocation strategy", MMQP_REASON_POLICY_VIOLATION);
            }
            if (policy.m_Bulk)
            {
                Require(policy.m_Bulk->m_Streams && policy.m_Bulk->m_Threshold, "Invalid QUIC bulk policy");
            }
            if (policy.m_Datagram)
            {
                Require(*policy.m_Datagram != 0, "Invalid QUIC datagram threshold");
            }
        }

        void WriteProperties(Writer &writer, const SessionProperties &properties, CodecOptions options, bool ack)
        {
            writer.U8(
                (properties.m_Auth ? MMQP_SESSION_FLAGS_SAF : 0) | (properties.m_Policy ? MMQP_SESSION_FLAGS_SPF : 0) |
                (properties.m_Quic ? MMQP_SESSION_FLAGS_QPF : 0) | (properties.m_Will ? MMQP_SESSION_FLAGS_SWF : 0) |
                (properties.m_Profile ? MMQP_SESSION_FLAGS_TPF : 0));
            if (properties.m_Auth)
            {
                const auto &auth = *properties.m_Auth;
                Require(!ack || auth.m_Token.empty(), "CONACK cannot contain an authentication token");
                if (!auth.m_Token.empty())
                {
                    Token(auth.m_Token);
                }
                Require(ValidUtf8(auth.m_Identifier), "Invalid session identifier");
                writer.U8((auth.m_Identifier.empty() ? 0 : MMQP_AUTHEN_FLAGS_SID) |
                          (auth.m_Token.empty() ? 0 : MMQP_AUTHEN_FLAGS_TOK));
                if (!auth.m_Identifier.empty())
                {
                    writer.String(auth.m_Identifier);
                }
                if (!auth.m_Token.empty())
                {
                    writer.Blob(auth.m_Token);
                }
            }
            if (properties.m_Policy)
            {
                const auto &policy = *properties.m_Policy;
                writer.U8((policy.m_Reasons ? MMQP_POLICY_FLAGS_IRS : 0) |
                          (policy.m_Expiry ? MMQP_POLICY_FLAGS_SEI : 0) |
                          (policy.m_Keepalive ? MMQP_POLICY_FLAGS_KAI : 0) |
                          (policy.m_MaxPacket ? MMQP_POLICY_FLAGS_MPS : 0));
                if (policy.m_Expiry)
                {
                    DecodeInterval(*policy.m_Expiry);
                    writer.U8(*policy.m_Expiry);
                }
                if (policy.m_Keepalive)
                {
                    DecodeInterval(*policy.m_Keepalive);
                    writer.U8(*policy.m_Keepalive);
                }
                if (policy.m_MaxPacket)
                {
                    Require(*policy.m_MaxPacket >= 2, "Invalid maximum packet size");
                    writer.U32(*policy.m_MaxPacket);
                }
            }
            if (properties.m_Quic)
            {
                const auto &policy = *properties.m_Quic;
                ValidateQuic(policy);
                writer.U8((policy.m_Strategy ? MMQP_QUIC_FLAGS_SAS : 0) | (policy.m_Bulk ? MMQP_QUIC_FLAGS_BTS : 0) |
                          (policy.m_Datagram ? MMQP_QUIC_FLAGS_QDG : 0));
                if (policy.m_Strategy)
                {
                    writer.U8(*policy.m_Strategy);
                }
                if (policy.m_Bulk)
                {
                    writer.U8(policy.m_Bulk->m_Streams);
                    writer.U32(policy.m_Bulk->m_Threshold);
                }
                if (policy.m_Datagram)
                {
                    writer.U32(*policy.m_Datagram);
                }
            }
            if (properties.m_Will)
            {
                const auto &will = *properties.m_Will;
                Qos(will.m_Qos);
                Require(will.m_Birth || will.m_Last, "Empty will block");
                writer.U8((will.m_Birth ? MMQP_STATUS_FLAGS_BWM : 0) | (will.m_Last ? MMQP_STATUS_FLAGS_LWM : 0) |
                          (will.m_Topic.empty() ? 0 : MMQP_STATUS_FLAGS_TOP) | will.m_Qos);
                if (will.m_Birth)
                {
                    writer.Blob(*will.m_Birth);
                }
                if (will.m_Last)
                {
                    writer.Blob(*will.m_Last);
                }
                if (!will.m_Topic.empty())
                {
                    writer.String(ParseTopic(will.m_Topic).m_Filter);
                }
            }
            if (properties.m_Profile)
            {
                Writer profile(options.m_MaxPacket);
                WriteProfiles(profile, *properties.m_Profile, options);
                auto length = static_cast<uint32_t>(profile.Data().size());
                auto width = LengthWidth(length);
                writer.U8(static_cast<uint8_t>((width - 1) << 2));
                writer.Number(length, width);
                writer.Bytes(profile.Data());
            }
        }

        SessionProperties ReadProperties(Reader &reader, CodecOptions options, bool ack)
        {
            SessionProperties properties;
            auto flags = reader.U8();
            Flags(flags, 0xF8);
            if (flags & MMQP_SESSION_FLAGS_SAF)
            {
                auto attrs = reader.U8();
                Flags(attrs, ack ? 0x80 : 0xC0);
                SessionAuth auth;
                if (attrs & MMQP_AUTHEN_FLAGS_SID)
                {
                    auth.m_Identifier = Text(reader);
                    Require(!auth.m_Identifier.empty(), "Empty session identifier");
                }
                if (attrs & MMQP_AUTHEN_FLAGS_TOK)
                {
                    auth.m_Token = reader.Blob();
                    Token(auth.m_Token);
                }
                properties.m_Auth = std::move(auth);
            }
            if (flags & MMQP_SESSION_FLAGS_SPF)
            {
                auto attrs = reader.U8();
                Flags(attrs, 0xF0);
                SessionPolicy policy;
                policy.m_Reasons = attrs & MMQP_POLICY_FLAGS_IRS;
                if (attrs & MMQP_POLICY_FLAGS_SEI)
                {
                    policy.m_Expiry = reader.U8();
                    DecodeInterval(*policy.m_Expiry);
                }
                if (attrs & MMQP_POLICY_FLAGS_KAI)
                {
                    policy.m_Keepalive = reader.U8();
                    DecodeInterval(*policy.m_Keepalive);
                }
                if (attrs & MMQP_POLICY_FLAGS_MPS)
                {
                    policy.m_MaxPacket = reader.U32();
                    Require(*policy.m_MaxPacket >= 2, "Invalid maximum packet size");
                }
                properties.m_Policy = policy;
            }
            if (flags & MMQP_SESSION_FLAGS_QPF)
            {
                auto attrs = reader.U8();
                Flags(attrs, 0xE0);
                QuicPolicy policy;
                if (attrs & MMQP_QUIC_FLAGS_SAS)
                {
                    policy.m_Strategy = reader.U8();
                }
                if (attrs & MMQP_QUIC_FLAGS_BTS)
                {
                    BulkPolicy bulk;
                    bulk.m_Streams = reader.U8();
                    bulk.m_Threshold = reader.U32();
                    policy.m_Bulk = bulk;
                }
                if (attrs & MMQP_QUIC_FLAGS_QDG)
                {
                    policy.m_Datagram = reader.U32();
                }
                ValidateQuic(policy);
                properties.m_Quic = policy;
            }
            if (flags & MMQP_SESSION_FLAGS_SWF)
            {
                auto attrs = reader.U8();
                Flags(attrs, 0xE3);
                SessionWill will;
                will.m_Qos = attrs & MMQP_QOS_LEVEL_MASK;
                Qos(will.m_Qos);
                if (attrs & MMQP_STATUS_FLAGS_BWM)
                {
                    will.m_Birth = reader.Blob();
                }
                if (attrs & MMQP_STATUS_FLAGS_LWM)
                {
                    will.m_Last = reader.Blob();
                }
                if (attrs & MMQP_STATUS_FLAGS_TOP)
                {
                    will.m_Topic = ParseTopic(Text(reader)).m_Filter;
                }
                Require(will.m_Birth || will.m_Last, "Empty will block");
                properties.m_Will = std::move(will);
            }
            if (flags & MMQP_SESSION_FLAGS_TPF)
            {
                auto attrs = reader.U8();
                Flags(attrs, MMQP_RLEN_FLAGS_MASK);
                auto width = static_cast<uint8_t>((attrs >> 2) + 1);
                auto length = static_cast<uint32_t>(reader.Number(width));
                Require(LengthWidth(length) == width, "Noncanonical embedded profile length");
                Reader profile(reader.Bytes(length));
                properties.m_Profile = ReadProfiles(profile, options);
            }
            return properties;
        }

        void WritePublication(Writer &writer, const PublishRequest &request, CodecOptions options,
                              std::span<const uint8_t> payload)
        {
            Require(request.m_Alias != 0, "Publication alias must be nonzero", MMQP_REASON_INVALID_ALIAS);
            Qos(request.m_Qos);
            Require(request.m_Qos || (!request.m_Duplicate && !request.m_PacketId), "QoS 0 cannot carry a PID or DUP");
            if (request.m_Qos)
            {
                Pid(request.m_PacketId);
            }
            bool correlated = request.m_Type == MMQP_PUBREQ_TYPE_REQ || request.m_Type == MMQP_PUBREQ_TYPE_RES;
            Require(correlated == (request.m_CorrelationId != 0),
                    "Requests and responses require a nonzero correlation ID; messages cannot carry one",
                    MMQP_REASON_INVALID_CORRELATION);
            writer.U16(request.m_Alias);
            writer.U8((request.m_Duplicate ? MMQP_PUBREQ_FLAGS_DUP : 0) |
                      (request.m_Encryption ? MMQP_PUBREQ_FLAGS_E2E : 0) |
                      (request.m_Topic.empty() ? 0 : MMQP_PUBREQ_FLAGS_TOP) | request.m_Qos);
            if (!request.m_Topic.empty())
            {
                writer.String(ParseTopic(request.m_Topic).m_Filter);
            }
            if (request.m_Qos)
            {
                writer.U16(request.m_PacketId);
            }
            if (correlated)
            {
                writer.U16(request.m_CorrelationId);
            }
            if (request.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                writer.U8(request.m_Responses);
            }
            if (request.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                WriteReason(writer, request.m_Reason, options);
            }
            if (request.m_Encryption)
            {
                writer.U32(request.m_Encryption->m_Epoch);
                writer.Bytes(request.m_Encryption->m_Nonce);
                writer.Bytes(request.m_Encryption->m_Tag);
            }
            writer.Bytes(payload);
        }

        PublishRequest ReadPublication(Reader &reader, uint8_t type, CodecOptions options)
        {
            PublishRequest request;
            request.m_Type = type;
            request.m_Alias = reader.U16();
            Require(request.m_Alias != 0, "Publication alias must be nonzero", MMQP_REASON_INVALID_ALIAS);
            auto flags = reader.U8();
            Flags(flags, 0xE3);
            bool correlated = type == MMQP_PUBREQ_TYPE_REQ || type == MMQP_PUBREQ_TYPE_RES;
            request.m_Qos = flags & MMQP_QOS_LEVEL_MASK;
            request.m_Duplicate = flags & MMQP_PUBREQ_FLAGS_DUP;
            Qos(request.m_Qos);
            Require(request.m_Qos || !request.m_Duplicate, "QoS 0 cannot set DUP");
            if (flags & MMQP_PUBREQ_FLAGS_TOP)
            {
                request.m_Topic = ParseTopic(Text(reader)).m_Filter;
            }
            if (request.m_Qos)
            {
                request.m_PacketId = reader.U16();
                Pid(request.m_PacketId);
            }
            if (correlated)
            {
                request.m_CorrelationId = reader.U16();
                Require(request.m_CorrelationId != 0, "Zero correlation ID", MMQP_REASON_INVALID_CORRELATION);
            }
            if (type == MMQP_PUBREQ_TYPE_REQ)
            {
                request.m_Responses = reader.U8();
            }
            if (type == MMQP_PUBREQ_TYPE_RES)
            {
                request.m_Reason = ReadReason(reader, options);
            }
            if (flags & MMQP_PUBREQ_FLAGS_E2E)
            {
                EncryptionHeader encryption;
                encryption.m_Epoch = reader.U32();
                auto nonce = reader.Bytes(encryption.m_Nonce.size());
                std::copy(nonce.begin(), nonce.end(), encryption.m_Nonce.begin());
                auto tag = reader.Bytes(encryption.m_Tag.size());
                std::copy(tag.begin(), tag.end(), encryption.m_Tag.begin());
                request.m_Encryption = encryption;
            }
            auto payload = reader.Bytes(reader.Remaining());
            request.m_Payload = SharedBytes::Copy(payload);
            return request;
        }
    } // namespace

    uint8_t Codec::Type(const Message &message)
    {
        return static_cast<uint8_t>((message.index() + 1) << 4);
    }

    Buffer Codec::Encode(const Message &message, CodecOptions options)
    {
        Writer writer(options.m_MaxPacket);
        uint8_t flags = 0;
        std::visit(
            [&](const auto &value)
            {
                using Type = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Type, ConnectionRequest>)
                {
                    Require(value.m_Version == MMQPv1, "Unsupported MMQP version", MMQP_REASON_UNSUPPORTED_VERSION);
                    Require(!value.m_Resume ||
                                (value.m_Properties.m_Auth && !value.m_Properties.m_Auth->m_Identifier.empty()),
                            "Resume requires a session identifier");
                    flags = value.m_Resume ? MMQP_CONREQ_FLAGS_RES : 0;
                    writer.String(MMQP_ALIAS);
                    writer.U8(value.m_Version);
                    WriteProperties(writer, value.m_Properties, options, false);
                }
                else if constexpr (std::is_same_v<Type, ConnectionAck>)
                {
                    WriteReason(writer, value.m_Reason, options);
                    WriteProperties(writer, value.m_Properties, options, true);
                }
                else if constexpr (std::is_same_v<Type, ConnectionSync>)
                {
                    if (value.m_Token)
                    {
                        Token(*value.m_Token);
                        flags = MMQP_CONSYN_FLAGS_TOK;
                        writer.Blob(*value.m_Token);
                    }
                }
                else if constexpr (std::is_same_v<Type, ConnectionEnd>)
                {
                    WriteReason(writer, value.m_Reason, options);
                }
                else if constexpr (std::is_same_v<Type, ProfileRequest>)
                {
                    WriteProfiles(writer, value, options);
                }
                else if constexpr (std::is_same_v<Type, ProfileSync>)
                {
                    Require(value.m_Alias != 0, "Alias mapping cannot use zero", MMQP_REASON_INVALID_ALIAS);
                    writer.U16(value.m_Alias);
                    writer.String(ParseTopic(value.m_Topic).m_Filter);
                    WriteReason(writer, value.m_Reason, options);
                }
                else if constexpr (std::is_same_v<Type, PublishRequest>)
                {
                    flags = value.m_Type;
                    writer.Reserve(value.m_Payload.Size() + 64 + value.m_Topic.size());
                    WritePublication(writer, value, options, value.m_Payload);
                }
                else
                {
                    Pid(value.m_PacketId);
                    writer.U16(value.m_PacketId);
                    WriteReason(writer, value.m_Reason, options);
                }
            },
            message);
        return Frame(Codec::Type(message), flags, writer.Data(), options.m_MaxPacket);
    }

    static PacketSummary PeekParts(std::span<const uint8_t> bytes, size_t totalSize, CodecOptions options)
    {
        auto header = ReadHeader(bytes, options.m_MaxPacket);
        Require(header && totalSize == uint64_t{header->m_Size} + header->m_Remaining,
                "Incomplete packet or trailing frame bytes");
        PacketSummary summary;
        summary.m_Type = header->m_Type;
        if (!summary.Publication())
        {
            return summary;
        }
        Reader reader(bytes.subspan(header->m_Size));
        reader.U16();
        auto flags = reader.U8();
        Flags(flags, 0xE3);
        summary.m_Qos = flags & MMQP_QOS_LEVEL_MASK;
        Qos(summary.m_Qos);
        if (flags & MMQP_PUBREQ_FLAGS_TOP)
        {
            auto topic = reader.Bytes(reader.U8());
            summary.m_Topic = {reinterpret_cast<const char *>(topic.data()), topic.size()};
        }
        if (summary.m_Qos)
        {
            reader.U16();
        }
        if (header->m_Flags != MMQP_PUBREQ_TYPE_MSG)
        {
            reader.U16();
        }
        if (header->m_Flags == MMQP_PUBREQ_TYPE_REQ)
        {
            reader.U8();
        }
        if (header->m_Flags == MMQP_PUBREQ_TYPE_RES && reader.U8() && options.m_Reasons)
        {
            reader.Bytes(reader.U8());
        }
        if (flags & MMQP_PUBREQ_FLAGS_E2E)
        {
            reader.Bytes(4 + 16 + 16);
        }
        summary.m_Payload = reader.Remaining() + totalSize - bytes.size();
        return summary;
    }

    PacketSummary Codec::Peek(std::span<const uint8_t> bytes, CodecOptions options)
    {
        return PeekParts(bytes, bytes.size(), options);
    }

    PacketSummary Codec::PeekFrame(const WireFrame &frame, CodecOptions options)
    {
        return PeekParts(frame.m_Head.Span(), frame.Size(), options);
    }

    WireFrame Codec::EncodeFrame(const PublishRequest &message, CodecOptions options)
    {
        Require(message.m_Type >= MMQP_PUBREQ_TYPE_MSG && message.m_Type <= MMQP_PUBREQ_TYPE_RES,
                "Invalid publication type");
        Writer body(options.m_MaxPacket);
        WritePublication(body, message, options, {});
        uint64_t length = body.Data().size() + message.m_Payload.Size();
        Require(length <= UINT32_MAX, "Publication is too large", MMQP_REASON_POLICY_VIOLATION);
        uint8_t width = RemainingLengthWidth(static_cast<uint32_t>(length));
        Require(length + width + 1 <= options.m_MaxPacket, "Publication exceeds negotiated MPS",
                MMQP_REASON_POLICY_VIOLATION);
        WireFrame frame;
        ByteWriter writer(frame.m_Head);
        writer.U8(MMQP_PACKET_PUBREQ | static_cast<uint8_t>((width - 1) << 2) | message.m_Type);
        writer.UN(length, width);
        writer.Bytes(body.Data());
        frame.m_Tail = message.m_Payload;
        return frame;
    }

    WireFrame Codec::EncodeFrame(const Message &message, CodecOptions options)
    {
        if (const auto *publication = std::get_if<PublishRequest>(&message))
        {
            return EncodeFrame(*publication, options);
        }
        return WireFrame(Encode(message, options));
    }

    Buffer Codec::Encode(const PublishRequest &header, CodecOptions options, std::span<const uint8_t> payload)
    {
        Require(header.m_Payload.Empty(), "External payload encoding requires an empty header payload");
        Writer writer(options.m_MaxPacket);
        writer.Reserve(payload.size() + 64 + header.m_Topic.size());
        WritePublication(writer, header, options, payload);
        return Frame(MMQP_PACKET_PUBREQ, header.m_Type, writer.Data(), options.m_MaxPacket);
    }

    Message Codec::Decode(std::span<const uint8_t> bytes, CodecOptions options)
    {
        auto header = ReadHeader(bytes, options.m_MaxPacket);
        Require(header && bytes.size() == uint64_t{header->m_Size} + header->m_Remaining,
                "Incomplete packet or trailing frame bytes");
        Reader reader(bytes.subspan(header->m_Size));
        Message message;
        switch (header->m_Type)
        {
        case MMQP_PACKET_CONREQ:
        {
            ConnectionRequest request;
            Require(Text(reader) == MMQP_ALIAS, "Invalid protocol alias");
            request.m_Version = reader.U8();
            Require(request.m_Version == MMQPv1, "Unsupported MMQP version", MMQP_REASON_UNSUPPORTED_VERSION);
            request.m_Resume = header->m_Flags & MMQP_CONREQ_FLAGS_RES;
            request.m_Properties = ReadProperties(reader, options, false);
            Require(!request.m_Resume ||
                        (request.m_Properties.m_Auth && !request.m_Properties.m_Auth->m_Identifier.empty()),
                    "Resume requires a session identifier");
            message = std::move(request);
            break;
        }
        case MMQP_PACKET_CONACK:
        {
            ConnectionAck ack;
            ack.m_Reason = ReadReason(reader, options);
            ack.m_Properties = ReadProperties(reader, options, true);
            message = std::move(ack);
            break;
        }
        case MMQP_PACKET_CONSYN:
        {
            ConnectionSync sync;
            if (header->m_Flags & MMQP_CONSYN_FLAGS_TOK)
            {
                sync.m_Token = reader.Blob();
                Token(*sync.m_Token);
            }
            message = std::move(sync);
            break;
        }
        case MMQP_PACKET_CONEND:
            message = ConnectionEnd{ReadReason(reader, options)};
            break;
        case MMQP_PACKET_PROREQ:
            message = ReadProfiles(reader, options);
            break;
        case MMQP_PACKET_PROACK:
        case MMQP_PACKET_PUBACK:
        {
            auto packetId = reader.U16();
            Pid(packetId);
            auto reason = ReadReason(reader, options);
            if (header->m_Type == MMQP_PACKET_PROACK)
            {
                message = ProfileAck{packetId, std::move(reason)};
            }
            else
            {
                message = PublishAck{packetId, std::move(reason)};
            }
            break;
        }
        case MMQP_PACKET_PROSYN:
        {
            ProfileSync sync;
            sync.m_Alias = reader.U16();
            Require(sync.m_Alias != 0, "Alias mapping cannot use zero", MMQP_REASON_INVALID_ALIAS);
            sync.m_Topic = ParseTopic(Text(reader)).m_Filter;
            sync.m_Reason = ReadReason(reader, options);
            message = std::move(sync);
            break;
        }
        case MMQP_PACKET_PUBREQ:
            message = ReadPublication(reader, header->m_Flags, options);
            break;
        default:
            throw ProtocolError("Unknown packet type");
        }
        reader.Finish();
        return message;
    }
} // namespace eCore::MMQP
