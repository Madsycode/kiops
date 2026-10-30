#include "Checks.h"
#include "Support/Protocol.h"
#include <eCore/Network/QuicPolicy.h>

using namespace eCore::MMQP;

namespace
{
    // The previous allocating matcher, kept as an independent reference.
    bool ReferenceMatch(const std::string &pattern, const std::string &name)
    {
        if (name.front() == '$' && pattern.front() != '$')
        {
            return false;
        }
        size_t left = 0;
        size_t right = 0;
        while (left < pattern.size())
        {
            auto stop = pattern.find('/', left);
            if (stop == std::string::npos)
            {
                stop = pattern.size();
            }
            auto level = std::string_view(pattern).substr(left, stop - left);
            if (level == "#")
            {
                return true;
            }
            if (right >= name.size())
            {
                return false;
            }
            auto next = name.find('/', right);
            if (next == std::string::npos)
            {
                next = name.size();
            }
            if (level != "+" && level != std::string_view(name).substr(right, next - right))
            {
                return false;
            }
            left = stop + 1;
            right = next + 1;
            if (next == name.size())
            {
                return stop == pattern.size() || std::string_view(pattern).substr(left) == "#";
            }
        }
        return right > name.size();
    }
} // namespace

EMPY_TEST(Protocol, ReferenceCases)
{
    // Independent golden frames follow the paper's field order and PRT-based request IDs.
    std::vector<std::pair<Message, Buffer>> golden{
        {ConnectionRequest{}, {0x10, 0x07, 0x04, 'M', 'M', 'Q', 'P', 0x01, 0x00}},
        {ConnectionAck{}, {0x20, 0x02, 0x00, 0x00}},
        {ConnectionSync{}, {0x30, 0x00}},
        {ConnectionEnd{}, {0x40, 0x01, 0x00}},
        {ProfileRequest{0x1234, {{MMQP_PROREQ_TYPE_SUB, 1, "a", 1}}},
         {0x50, 0x07, 0x12, 0x34, 0x61, 0x00, 0x01, 0x01, 'a'}},
        {ProfileAck{7, {}}, {0x60, 0x03, 0x00, 0x07, 0x00}},
        {ProfileSync{2, "x", {}}, {0x70, 0x05, 0x00, 0x02, 0x01, 'x', 0x00}},
        {PublishAck{7, {}}, {0x90, 0x03, 0x00, 0x07, 0x00}}};
    PublishRequest publication;
    publication.m_Alias = 2;
    publication.m_Topic = "x";
    publication.m_Qos = 1;
    publication.m_PacketId = 7;
    publication.m_Payload = {0xAA};
    golden.emplace_back(publication, Buffer{0x81, 0x08, 0x00, 0x02, 0x21, 0x01, 'x', 0x00, 0x07, 0xAA});
    publication.m_Type = MMQP_PUBREQ_TYPE_REQ;
    publication.m_CorrelationId = 0x1234;
    publication.m_Responses = 2;
    golden.emplace_back(publication,
                        Buffer{0x82, 0x0B, 0x00, 0x02, 0x21, 0x01, 'x', 0x00, 0x07, 0x12, 0x34, 0x02, 0xAA});
    publication.m_Type = MMQP_PUBREQ_TYPE_RES;
    golden.emplace_back(publication,
                        Buffer{0x83, 0x0B, 0x00, 0x02, 0x21, 0x01, 'x', 0x00, 0x07, 0x12, 0x34, 0x00, 0xAA});
    publication.m_Qos = 0;
    publication.m_PacketId = 0;
    publication.m_Topic.clear();
    golden.emplace_back(publication, Buffer{0x83, 0x07, 0x00, 0x02, 0x00, 0x12, 0x34, 0x00, 0xAA});
    publication.m_Type = MMQP_PUBREQ_TYPE_REQ;
    golden.emplace_back(publication, Buffer{0x82, 0x07, 0x00, 0x02, 0x00, 0x12, 0x34, 0x02, 0xAA});
    for (const auto &[message, bytes] : golden)
    {
        EMPY_CHECK(Codec::Encode(message) == bytes);
        EMPY_CHECK(Codec::Encode(Codec::Decode(bytes)) == bytes);
        if (auto expected = std::get_if<PublishRequest>(&message))
        {
            auto decoded = std::get<PublishRequest>(Codec::Decode(bytes));
            EMPY_CHECK(decoded.m_Type == expected->m_Type && decoded.m_Qos == expected->m_Qos);
            EMPY_CHECK(decoded.m_PacketId == expected->m_PacketId &&
                       decoded.m_CorrelationId == expected->m_CorrelationId);
            EMPY_CHECK(decoded.m_Payload == expected->m_Payload);
            for (auto reserved : {0x10, 0x08, 0x04})
            {
                auto invalid = bytes;
                invalid[4] |= reserved;
                Throws([&] { Codec::Decode(invalid); });
            }
        }
        for (size_t count = 0; count < bytes.size(); ++count)
        {
            Throws([&] { Codec::Decode(std::span(bytes).first(count)); });
        }
        auto extra = bytes;
        extra.push_back(0);
        Throws([&] { Codec::Decode(extra); });
    }

    for (auto length : {0u, 255u, 256u, 65535u, 65536u, 16777215u, 16777216u})
    {
        auto width = LengthWidth(length);
        Writer writer;
        writer.U8(0x80 | ((width - 1) << 2) | 1);
        writer.Number(length, width);
        auto header = ReadHeader(writer.Data(), UINT32_MAX).value();
        EMPY_CHECK(header.m_Remaining == length && header.m_Size == width + 1);
    }
    Throws([] { ReadHeader(Buffer{0x35, 0, 1}); });
    Throws([] { ReadHeader(Buffer{0x34, 0, 1}); });
    Throws([] { ReadHeader(Buffer{0xA0, 0}); });
    FrameStream limited(64);
    Throws([&] { limited.Feed(Buffer{0x8D, 0x7F, 0xFF, 0xFF, 0xFF}, [](auto) {}); });
    EMPY_CHECK(limited.Buffered() == 0);
    // Fragmented input buffers only the header before rejecting the advertised body.
    FrameStream fragmented(64);
    fragmented.Feed(Buffer{0x8D}, [](auto) {});
    Throws([&] { fragmented.Feed(Buffer{0x7F, 0xFF, 0xFF, 0xFF}, [](auto) {}); });
    EMPY_CHECK(fragmented.Buffered() == 5);
    FrameStream stream;
    size_t frames = 0;
    auto receive = [&](auto bytes)
    {
        Codec::Decode(bytes);
        ++frames;
    };
    for (const auto &[message, bytes] : golden)
    {
        for (auto byte : bytes)
        {
            stream.Feed(std::span(&byte, 1), receive);
        }
    }
    EMPY_CHECK(frames == golden.size() && stream.Buffered() == 0);
    Buffer batch;
    for (const auto &[message, bytes] : golden)
    {
        batch.insert(batch.end(), bytes.begin(), bytes.end());
    }
    stream.Feed(batch, receive);
    EMPY_CHECK(frames == golden.size() * 2);

    for (uint8_t type : {MMQP_PUBREQ_TYPE_REQ, MMQP_PUBREQ_TYPE_RES})
    {
        auto control = static_cast<uint8_t>(0x80 | type);
        // Complete frames with missing/truncated IDs, missing type-specific data, or zero IDs.
        for (const auto &invalid :
             {Buffer{control, 3, 0, 1, 0}, Buffer{control, 4, 0, 1, 0, 1}, Buffer{control, 5, 0, 1, 0, 0, 1},
              Buffer{control, 6, 0, 1, 0, 0, 0, 0}, Buffer{control, 5, 0, 1, 1, 0, 7},
              Buffer{control, 6, 0, 1, 1, 0, 7, 1}, Buffer{control, 7, 0, 1, 1, 0, 7, 0, 1},
              Buffer{control, 8, 0, 1, 1, 0, 7, 0, 0, 0}})
        {
            Throws([&] { Codec::Decode(invalid); });
        }
        publication.m_Type = type;
        publication.m_CorrelationId = 0;
        Throws([&] { Codec::Encode(publication); });
    }
    publication.m_Type = MMQP_PUBREQ_TYPE_MSG;
    publication.m_CorrelationId = 1;
    Throws([&] { Codec::Encode(publication); });

    ConnectionRequest connect;
    connect.m_Resume = true;
    connect.m_Properties.m_Auth = SessionAuth{"session", Buffer(32, 0xA5)};
    connect.m_Properties.m_Policy = SessionPolicy{true, EncodeInterval(3600), EncodeInterval(120), 65536};
    connect.m_Properties.m_Quic = QuicPolicy{MMQP_QUIC_STRATEGY_BULK, BulkPolicy{2, 4096}, 1200};
    connect.m_Properties.m_Will = SessionWill{Buffer{'o', 'n'}, Buffer{'o', 'f', 'f'}, "device/status", 1};
    connect.m_Properties.m_Profile = ProfileRequest{1, {{MMQP_PROREQ_TYPE_REG, 0, "room/#", 1, true, 10, 2}}};
    auto connectBytes = Codec::Encode(connect);
    EMPY_CHECK((connectBytes[0] & 0x03) == 0x02);
    EMPY_CHECK(Codec::Encode(Codec::Decode(connectBytes)) == connectBytes);
    for (auto type : {MMQP_PUBREQ_TYPE_MSG, MMQP_PUBREQ_TYPE_REQ, MMQP_PUBREQ_TYPE_RES})
    {
        publication.m_Type = type;
        publication.m_CorrelationId = type == MMQP_PUBREQ_TYPE_MSG ? 0 : UINT16_MAX;
        publication.m_Qos = 1;
        publication.m_PacketId = 7;
        publication.m_Encryption = EncryptionHeader{10, {}, {}};
        publication.m_Reason = {MMQP_REASON_NO_RESPONDER, "none"};
        publication.m_Duplicate = true;
        auto encoded = Codec::Encode(publication, {true});
        EMPY_CHECK((encoded[0] & 3) == type);
        EMPY_CHECK(Codec::Encode(Codec::Decode(encoded, {true}), {true}) == encoded);
    }
    Buffer token(32, 0x11);
    Buffer consyn{0x32, 33, 32};
    consyn.insert(consyn.end(), token.begin(), token.end());
    EMPY_CHECK(Codec::Encode(ConnectionSync{token}) == consyn);
    Throws([] { Codec::Encode(ConnectionSync{Buffer{0x11, 0x22}}); });
    Throws([] { Codec::Decode(Buffer{0x32, 3, 2, 0x11, 0x22}); });
    auto failure = Codec::Encode(ConnectionEnd{{0x80, "failure"}}, {true});
    EMPY_CHECK(std::get<ConnectionEnd>(Codec::Decode(failure, {true})).m_Reason.m_Text == "failure");

    EMPY_CHECK(CanonicalTopic("//home///room//") == "home/room");
    EMPY_CHECK(TopicMatches("room/#", "room"));
    EMPY_CHECK(TopicMatches("room/+", "room/temp"));
    EMPY_CHECK(!TopicMatches("room/+", "room"));
    EMPY_CHECK(!TopicMatches("room/+", "room/temp/value"));
    EMPY_CHECK(TopicMatches("$share/workers/room/#", "room/temp"));
    EMPY_CHECK(!TopicMatches("#", "$lwm/device"));
    EMPY_CHECK(TopicMatches("$lwm/#", "$lwm/device"));
    EMPY_CHECK(MatchTopic("a/+/c", "a/b/c") && !MatchTopic("a/+/c", "a/b") && !MatchTopic("a/b", "a/b/c"));
    EMPY_CHECK(MatchTopic("+/#", "a") && !MatchTopic("+", "$sys") && MatchTopic("$sys/+", "$sys/x"));
    EMPY_CHECK(SplitShared("$share/g/a/#") == std::pair<std::string_view, std::string_view>("g", "a/#"));
    {
        // The trie must report exactly the filters the linear matcher accepts.
        const std::vector<std::string> levels{"a", "b", "+", "#", "$s"};
        std::vector<std::string> filters;
        for (const auto &x : levels)
        {
            for (const auto &y : levels)
            {
                for (const auto &z : levels)
                {
                    for (const auto &candidate : {x, x + "/" + y, x + "/" + y + "/" + z})
                    {
                        try
                        {
                            if (ParseTopic(candidate, true).m_Filter == candidate)
                            {
                                filters.push_back(candidate);
                            }
                        }
                        catch (const ProtocolError &)
                        {
                        }
                    }
                }
            }
        }
        TopicTree<size_t> tree;
        for (size_t i = 0; i < filters.size(); ++i)
        {
            tree.Insert(filters[i], i);
        }
        for (const auto &topic : {"a", "b", "a/b", "a/b/a", "b/b/b", "$s", "$s/a", "a/$s", "$s/a/b"})
        {
            std::vector<size_t> found;
            tree.Match(topic, [&](size_t index) { found.push_back(index); });
            std::sort(found.begin(), found.end());
            std::vector<size_t> expected;
            for (size_t i = 0; i < filters.size(); ++i)
            {
                if (MatchTopic(filters[i], topic))
                {
                    expected.push_back(i);
                }
            }
            EMPY_CHECK(found == expected);
            for (size_t i = 0; i < filters.size(); ++i)
            {
                EMPY_CHECK(MatchTopic(filters[i], topic) == ReferenceMatch(filters[i], topic));
            }
        }
        for (size_t i = 0; i < filters.size(); ++i)
        {
            EMPY_CHECK(tree.Remove(filters[i], i));
        }
        EMPY_CHECK(tree.Size() == 0 && !tree.Remove("a", 0));
    }
    for (auto invalid : {"a/#/b", "a/x+", "$share/g", "$share/+/a"})
    {
        Throws([&] { ParseTopic(invalid, true); });
    }
    EMPY_CHECK(!ValidUtf8(std::string("\xC0\xAF", 2)));
    EMPY_CHECK(!ValidUtf8(std::string("\xED\xA0\x80", 3)));
    EMPY_CHECK(!ValidUtf8(std::string("\xF4\x90\x80\x80", 4)));
    EMPY_CHECK(ValidUtf8("temp\xC3\xA9rature"));
    Throws([] { ParseTopic(std::string(256, 'x')); });
    for (auto value : {1u, 35u, 63u, 60u, 3780u, 3600u, 226800u, 86400u, 5443200u})
    {
        EMPY_CHECK(DecodeInterval(EncodeInterval(value)) == value);
    }
    Throws([] { EncodeInterval(64); });
    Throws([] { EncodeInterval(0); });
    for (uint8_t value : {0x00, 0x40, 0x80, 0xC0})
    {
        Throws([&] { DecodeInterval(value); });
    }

    ReceiveWindow ordered;
    ordered.Commit(2);
    EMPY_CHECK(ordered.Inspect(1) == Reception::Duplicate);
    ReceiveWindow unordered(false, 64);
    unordered.Commit(3);
    unordered.Commit(1);
    EMPY_CHECK(unordered.Checkpoint() == 1 && unordered.Inspect(3) == Reception::Duplicate);
    unordered.Commit(2);
    EMPY_CHECK(unordered.Checkpoint() == 3);
    EMPY_CHECK(unordered.Inspect(68) == Reception::OutsideWindow);
    Throws([&] { unordered.Commit(68); });
    PacketSequence sequence;
    for (uint32_t i = 1; i <= UINT16_MAX; ++i)
    {
        EMPY_CHECK(sequence.Next() == i);
    }
    Throws([&] { sequence.Next(); });

    // Publication PIDs wrap with serial-number arithmetic instead of exhausting the session.
    EMPY_CHECK(WirePid(1) == 1 && WirePid(65535) == 65535 && WirePid(65536) == 1 && WirePid(131071) == 1);
    EMPY_CHECK(PidAhead(65535, 1) == 1 && PidAhead(65536, 1) == 0 && PidAhead(2, 1) == 65534);
    PublicationSequence outgoing(65534);
    EMPY_CHECK(outgoing.Peek() == 65535 && outgoing.Next() == 65535 && outgoing.Next() == 1 &&
               outgoing.Issued() == 65536);
    EMPY_CHECK(outgoing.Sent(1) && outgoing.Sent(65535) && !outgoing.Sent(2) && !outgoing.Sent(0));
    EMPY_CHECK(outgoing.Sequence(1) == 65536 && outgoing.Sequence(65535) == 65535);
    EMPY_CHECK(!PublicationSequence{}.Sent(1) && !PublicationSequence(3).Sent(4));
    Throws([&] { outgoing.Sequence(2); });

    ReceiveWindow wrapped;
    for (uint32_t i = 1; i <= 65535; ++i)
    {
        wrapped.Commit(static_cast<uint16_t>(i));
    }
    EMPY_CHECK(wrapped.Checkpoint() == 65535 && wrapped.Inspect(1) == Reception::New);
    EMPY_CHECK(wrapped.Inspect(65535) == Reception::Duplicate && wrapped.Inspect(40000) == Reception::Duplicate);
    EMPY_CHECK(wrapped.Sequence(1) == 65536 && wrapped.Sequence(65000) == 65000);
    wrapped.Commit(1);
    EMPY_CHECK(wrapped.Checkpoint() == 65536 && wrapped.Inspect(1) == Reception::Duplicate &&
               wrapped.Inspect(2) == Reception::New);
    Throws([] { ReceiveWindow{}.Sequence(65000); });

    // The 256-PID unordered window also spans the wrap, in any arrival order.
    ReceiveWindow multiplexed(false);
    for (uint32_t i = 1; i <= 65533; ++i)
    {
        multiplexed.Commit(static_cast<uint16_t>(i));
    }
    multiplexed.Commit(1);
    EMPY_CHECK(multiplexed.Checkpoint() == 65533 && multiplexed.Inspect(1) == Reception::Duplicate);
    EMPY_CHECK(multiplexed.Inspect(65534) == Reception::New && multiplexed.Sequence(1) == 65536);
    multiplexed.Commit(65535);
    multiplexed.Commit(65534);
    EMPY_CHECK(multiplexed.Checkpoint() == 65536);
    EMPY_CHECK(multiplexed.Inspect(257) == Reception::New && multiplexed.Inspect(258) == Reception::OutsideWindow);
    Throws([&] { multiplexed.Commit(258); });
    Writer savedWindow;
    multiplexed.Save(savedWindow);
    Reader loadedWindow(savedWindow.Data());
    EMPY_CHECK(ReceiveWindow::Load(loadedWindow).Checkpoint() == 65536);
    Writer legacyWindow;
    legacyWindow.U8(1);
    legacyWindow.U16(256);
    legacyWindow.U16(42);
    for (size_t i = 0; i < 32; ++i)
    {
        legacyWindow.U8(0);
    }
    Reader legacyReader(legacyWindow.Data());
    EMPY_CHECK(ReceiveWindow::Load(legacyReader, true).Checkpoint() == 42);

    // Pending publications stay in send order and admission stays bounded across the wrap.
    PendingQueue inflight;
    auto pending = [](uint64_t position)
    {
        PublishRequest message;
        message.m_Alias = 1;
        message.m_Qos = 1;
        message.m_PacketId = WirePid(position);
        return message;
    };
    for (uint64_t position : {65534ULL, 65535ULL, 65536ULL})
    {
        inflight.Add(pending(position), position, 1, TimePoint{}, TimePoint::max());
    }
    EMPY_CHECK(inflight.m_Entries.begin()->second.m_Message.m_PacketId == 65534 &&
               inflight.m_Entries.rbegin()->second.m_Message.m_PacketId == 1);
    EMPY_CHECK(inflight.CanAdd(1, 65534 + 255) && !inflight.CanAdd(1, 65534 + 256));
    EMPY_CHECK(inflight.Remove(1) && !inflight.Remove(1) && inflight.m_Entries.size() == 2 && inflight.Bytes() == 2);
    Throws([&] { inflight.Add(pending(2), 3, 1, TimePoint{}, TimePoint::max()); });

    CorrelationIds correlations;
    correlations.Claim(UINT16_MAX);
    correlations.Claim(2);
    EMPY_CHECK(correlations.Peek() == 1 && correlations.Contains(2));
    correlations.Claim(1);
    EMPY_CHECK(correlations.Peek() == 3);
    Throws([&] { correlations.Claim(0); });
    Throws([&] { correlations.Claim(2); });
    Writer stored;
    correlations.Save(stored);
    Reader restored(stored.Data());
    correlations = CorrelationIds::Load(restored);
    restored.Finish();
    EMPY_CHECK(correlations.Peek() == 3 && correlations.Contains(UINT16_MAX));
    for (uint32_t i = 3; i < UINT16_MAX; ++i)
    {
        correlations.Claim(static_cast<uint16_t>(i));
    }
    Throws([&] { correlations.Peek(); });
    Throws([&] { correlations.Claim(UINT16_MAX); });
    correlations.Release(2);
    EMPY_CHECK(correlations.Peek() == 2 && !correlations.Contains(2));
    correlations.Claim(2);

    ProfileTable profiles;
    profiles.Apply({1, {{MMQP_PROREQ_TYPE_REG, 1, "room/temp", 0}, {MMQP_PROREQ_TYPE_SUB, 0, "room/#", 1}}});
    EMPY_CHECK(profiles.Registration("room/temp", MMQP_PUBREQ_TYPE_MSG));
    EMPY_CHECK(!profiles.Registration("room/temp", MMQP_PUBREQ_TYPE_REQ));
    EMPY_CHECK(profiles.Subscriptions("room/temp", 1).size() == 1);
    Throws(
        [&]
        { profiles.Apply({2, {{MMQP_PROREQ_TYPE_SUB, 2, "valid", 1}, {MMQP_PROREQ_TYPE_REG, 1, "collision", 0}}}); });
    EMPY_CHECK(profiles.m_Entries.size() == 2 && !profiles.m_Subscribers.Find("valid"));
    profiles.Apply({3, {{MMQP_PROREQ_TYPE_UNS, 0, "room/#", 0}}});
    EMPY_CHECK(profiles.Subscriptions("room/temp", 1).empty());

    QuicPlan plan({MMQP_QUIC_STRATEGY_BULK, BulkPolicy{2, 1024}, 1200});
    PublishRequest telemetry;
    telemetry.m_Alias = 1;
    telemetry.m_Payload = Buffer(100);
    std::vector<Buffer> encoded;
    auto peek = [&](const Message &message)
    {
        encoded.push_back(Codec::Encode(message, {true}));
        return Codec::Peek(encoded.back(), {true});
    };
    EMPY_CHECK(plan.Select(peek(telemetry), 110, 1200, 0) == QuicChannel::Datagram);
    EMPY_CHECK(plan.Select(peek(telemetry), 110, 50, 0) == QuicChannel::BestEffort);
    telemetry.m_Qos = 1;
    telemetry.m_PacketId = 1;
    EMPY_CHECK(plan.Select(peek(telemetry), 112, 1200, 0) == QuicChannel::Reliable);
    Throws([&] { plan.Validate(QuicChannel::Datagram, peek(telemetry), 112); });
    telemetry.m_Payload = Buffer(2048);
    EMPY_CHECK(plan.Select(peek(telemetry), 2060, 1200, 0) == QuicChannel::Bulk);
    EMPY_CHECK(plan.Select(peek(telemetry), 2060, 1200, 2) == QuicChannel::Reliable);
    EMPY_CHECK(plan.Select(peek(PublishAck{1, {}}), 5, 1200, 0) == QuicChannel::Control);
    Throws([&] { plan.Validate(QuicChannel::Bulk, peek(PublishAck{1, {}}), 5); });
    // The peek reports the same routing fields as a full decode for every PRT block.
    for (auto type : {MMQP_PUBREQ_TYPE_MSG, MMQP_PUBREQ_TYPE_REQ, MMQP_PUBREQ_TYPE_RES})
    {
        PublishRequest routed;
        routed.m_Type = type;
        routed.m_Alias = 3;
        routed.m_Topic = "a/b";
        routed.m_Qos = 1;
        routed.m_PacketId = 9;
        routed.m_CorrelationId = type == MMQP_PUBREQ_TYPE_MSG ? 0 : 4;
        routed.m_Reason = {MMQP_REASON_TIMEOUT, "late"};
        routed.m_Encryption = EncryptionHeader{};
        routed.m_Payload = Buffer(77);
        auto packet = peek(routed);
        EMPY_CHECK(packet.Publication() && packet.m_Qos == 1 && packet.m_Payload == 77 && packet.m_Topic == "a/b");
    }
    Throws([&] { QuicPlan invalid({MMQP_QUIC_STRATEGY_CUSTOM, {}, {}}); });
    Throws([&] { QuicPlan invalid({MMQP_QUIC_STRATEGY_SEMANTIC, BulkPolicy{}, {}}); });
    auto negotiated = QuicPlan::Negotiate(plan.Policy(), {MMQP_QUIC_STRATEGY_BULK, BulkPolicy{1, 4096}, 512});
    EMPY_CHECK(negotiated.m_Bulk->m_Streams == 1 && negotiated.m_Bulk->m_Threshold == 4096 &&
               *negotiated.m_Datagram == 512);
    negotiated = QuicPlan::Negotiate(plan.Policy(), {MMQP_QUIC_STRATEGY_SINGLE, {}, {}});
    EMPY_CHECK(*negotiated.m_Strategy == MMQP_QUIC_STRATEGY_SINGLE && !negotiated.m_Bulk && !negotiated.m_Datagram);
}
