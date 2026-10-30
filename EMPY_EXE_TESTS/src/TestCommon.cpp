#include "Checks.h"
#include <eCore/API.h>

using namespace eCore;

namespace
{
    std::vector<uint8_t> Hex(std::string_view text)
    {
        std::vector<uint8_t> bytes;
        if (!FromHex(text, bytes))
        {
            throw std::runtime_error("bad hex in test vector");
        }
        return bytes;
    }

    std::string HexOf(ByteSpan bytes)
    {
        return ToHex(bytes);
    }
} // namespace

// -------------------------
// Crypto
// -------------------------

EMPY_TEST(Crypto, Sha256KnownAnswers)
{
    EMPY_CHECK_EQ(HexOf(Sha256(AsBytes(""))), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    EMPY_CHECK_EQ(HexOf(Sha256(AsBytes("abc"))), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    Sha256Stream stream;
    stream.Update(std::string_view("a"));
    stream.Update(std::string_view("bc"));
    EMPY_CHECK_EQ(HexOf(stream.Final()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // the stream resets after Final
    stream.Update(std::string_view("abc"));
    EMPY_CHECK_EQ(HexOf(stream.Final()), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

EMPY_TEST(Crypto, HmacSha256Rfc4231)
{
    // test case 1
    auto key1 = Hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    EMPY_CHECK_EQ(HexOf(HmacSha256(key1, AsBytes("Hi There"))),
                  "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    // test case 2
    EMPY_CHECK_EQ(HexOf(HmacSha256(AsBytes("Jefe"), AsBytes("what do ya want for nothing?"))),
                  "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    // test case 6: key longer than the block size
    std::vector<uint8_t> key6(131, 0xaa);
    EMPY_CHECK_EQ(HexOf(HmacSha256(key6, AsBytes("Test Using Larger Than Block-Size Key - Hash Key First"))),
                  "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    HmacStream stream(AsBytes("Jefe"));
    stream.Update(AsBytes("what do ya want "));
    stream.Update(AsBytes("for nothing?"));
    EMPY_CHECK_EQ(HexOf(stream.Final()), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
}

EMPY_TEST(Crypto, HkdfSha256Rfc5869)
{
    // test case 1
    auto ikm = Hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
    auto salt = Hex("000102030405060708090a0b0c");
    auto info = Hex("f0f1f2f3f4f5f6f7f8f9");
    std::vector<uint8_t> okm(42);
    EMPY_REQUIRE(HkdfSha256(ikm, salt, info, okm));
    EMPY_CHECK_EQ(HexOf(okm), "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865");
    // test case 3: empty salt and info
    std::vector<uint8_t> okm3(42);
    EMPY_REQUIRE(HkdfSha256(ikm, {}, {}, okm3));
    EMPY_CHECK_EQ(HexOf(okm3), "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8");
    // invalid output lengths are rejected
    std::vector<uint8_t> tooLong(255 * 32 + 1);
    EMPY_CHECK(!HkdfSha256(ikm, salt, info, tooLong));
}

EMPY_TEST(Crypto, HelpersAndHash64)
{
    Digest digest = Sha256(AsBytes("abc"));
    EMPY_CHECK_EQ(Hash64(AsBytes("abc")), LoadBE64(digest.data()));
    EMPY_CHECK_EQ(Hash64(AsBytes("abc")), 0xba7816bf8f01cfeaull);

    std::vector<uint8_t> bytes;
    EMPY_CHECK(FromHex("00ff10Ab", bytes));
    EMPY_CHECK_EQ(HexOf(bytes), "00ff10ab");
    EMPY_CHECK(!FromHex("0", bytes));
    EMPY_CHECK(!FromHex("zz", bytes));

    std::array<uint8_t, 32> a{}, b{};
    EMPY_CHECK(RandomBytes(a));
    EMPY_CHECK(RandomBytes(b));
    EMPY_CHECK(a != b);
    EMPY_CHECK(ConstantTimeEquals(a, a));
    EMPY_CHECK(!ConstantTimeEquals(a, b));
    EMPY_CHECK(!ConstantTimeEquals(ByteSpan(a.data(), 31), ByteSpan(a.data(), 32)));

    EMPY_CHECK_EQ(Fingerprint64(AsBytes("payload")), Fingerprint64(AsBytes("payload")));
    EMPY_CHECK(Fingerprint64(AsBytes("payload")) != Fingerprint64(AsBytes("payloae")));
    EMPY_CHECK(Fingerprint64(AsBytes("payload"), 1) != Fingerprint64(AsBytes("payload"), 2));
}

// -------------------------
// Buffers
// -------------------------

EMPY_TEST(Buffers, SharedBytesOwnership)
{
    SharedBytes empty;
    EMPY_CHECK(empty.Empty());
    EMPY_CHECK_EQ(empty.UseCount(), 0u);

    SharedBytes payload = SharedBytes::Copy(std::string_view("hello"));
    EMPY_CHECK_EQ(payload.Size(), 5u);
    EMPY_CHECK_EQ(payload.View(), std::string_view("hello"));
    {
        SharedBytes copy = payload;
        EMPY_CHECK_EQ(payload.UseCount(), 2u);
        EMPY_CHECK(copy.Data() == payload.Data());
        SharedBytes moved = std::move(copy);
        EMPY_CHECK(copy.Empty());
        EMPY_CHECK_EQ(payload.UseCount(), 2u);
    }
    EMPY_CHECK_EQ(payload.UseCount(), 1u);
    EMPY_CHECK(payload == SharedBytes::Copy(std::string_view("hello")));
    EMPY_CHECK(!(payload == SharedBytes::Copy(std::string_view("hellp"))));
    payload.Reset();
    EMPY_CHECK(payload.Empty());
}

EMPY_TEST(Buffers, InlineBufferGrowth)
{
    InlineBuffer<8> buffer;
    for (uint8_t i = 0; i < 100; ++i)
    {
        buffer.PushBack(i);
    }
    EMPY_CHECK_EQ(buffer.Size(), 100u);
    EMPY_CHECK(buffer.Capacity() >= 100u);
    bool ordered = true;
    for (uint8_t i = 0; i < 100; ++i)
    {
        ordered = ordered && buffer[i] == i;
    }
    EMPY_CHECK(ordered);

    InlineBuffer<8> moved = std::move(buffer);
    EMPY_CHECK_EQ(moved.Size(), 100u);
    EMPY_CHECK_EQ(buffer.Size(), 0u);

    InlineBuffer<8> small;
    small.Append(AsBytes("abc"));
    InlineBuffer<8> smallMoved = std::move(small);
    EMPY_CHECK_EQ(AsText(smallMoved.Span()), std::string_view("abc"));
    InlineBuffer<8> copied = smallMoved;
    EMPY_CHECK_EQ(AsText(copied.Span()), std::string_view("abc"));
}

EMPY_TEST(Buffers, ByteWriterReaderRoundTrip)
{
    std::vector<uint8_t> out;
    ByteWriter writer(out);
    writer.U8(0xAB);
    writer.U16(0x1234);
    writer.U32(0xDEADBEEF);
    writer.U64(0x0102030405060708ull);
    writer.UN(0x0A0B0C, 3);
    writer.Str8(std::string_view("topic"));
    writer.Bytes(std::string_view("rest"));
    EMPY_CHECK_EQ(out.size(), 1u + 2 + 4 + 8 + 3 + 6 + 4);
    EMPY_CHECK_EQ(out[1], 0x12);
    EMPY_CHECK_EQ(out[2], 0x34);

    ByteReader reader(out);
    uint8_t u8 = 0;
    uint16_t u16 = 0;
    uint32_t u32 = 0;
    uint64_t u64 = 0, un = 0;
    std::string_view text;
    EMPY_CHECK(reader.U8(u8) && u8 == 0xAB);
    EMPY_CHECK(reader.U16(u16) && u16 == 0x1234);
    EMPY_CHECK(reader.U32(u32) && u32 == 0xDEADBEEF);
    EMPY_CHECK(reader.U64(u64) && u64 == 0x0102030405060708ull);
    EMPY_CHECK(reader.UN(un, 3) && un == 0x0A0B0C);
    EMPY_CHECK(reader.Str8(text) && text == "topic");
    EMPY_CHECK_EQ(AsText(reader.Rest()), std::string_view("rest"));
    EMPY_CHECK(reader.AtEnd());

    // underflow is sticky and zeroes outputs
    ByteReader shortReader(ByteSpan(out.data(), 2));
    EMPY_CHECK(!shortReader.U32(u32));
    EMPY_CHECK_EQ(u32, 0u);
    EMPY_CHECK(!shortReader.U8(u8));
    EMPY_CHECK(!shortReader.Ok());
    EMPY_CHECK_EQ(shortReader.Remaining(), 0u);

    // a length prefix larger than the remaining data fails
    std::vector<uint8_t> bad = {5, 'a', 'b'};
    ByteReader badReader(bad);
    EMPY_CHECK(!badReader.Str8(text));
    EMPY_CHECK(text.empty());
}

EMPY_TEST(Buffers, AliasedGrowthAndInvalidWidths)
{
    InlineBuffer<4> small;
    small.Append(AsBytes("abcd"));
    small.Append(small.Span());
    EMPY_CHECK_EQ(AsText(small.Span()), std::string_view("abcdabcd"));
    ByteWriter inlineWriter(small);
    inlineWriter.Bytes(small.Span());
    EMPY_CHECK_EQ(AsText(small.Span()), std::string_view("abcdabcdabcdabcd"));
    std::vector<uint8_t> bytes = {'a', 'b', 'c', 'd'};
    bytes.shrink_to_fit();
    ByteWriter writer(bytes);
    writer.Bytes(ByteSpan(bytes));
    EMPY_CHECK_EQ(AsText(bytes), std::string_view("abcdabcd"));
    EMPY_CHECK_THROWS(writer.UN(1, 0));
    EMPY_CHECK_THROWS(writer.UN(1, 9));
    EMPY_CHECK_THROWS(writer.Str8(std::string(256, 'x')));
    EMPY_CHECK_EQ(bytes.size(), 8u);
}

// -------------------------
// YAML
// -------------------------

EMPY_TEST(Yaml, TypedAccessAndErrors)
{
    YamlNode root =
        YamlNode::Parse("broker:\n  id: b1\n  port: 5600\n  ratio: 0.5\n  tls: true\n  list: [a, b]\n", "test.yaml");
    YamlNode broker = root["broker"];
    EMPY_CHECK(broker.IsMap());
    EMPY_CHECK_EQ(broker.Require<std::string>("id"), std::string("b1"));
    EMPY_CHECK_EQ(broker.Get<uint16_t>("port", 0, 1, 65535), uint16_t(5600));
    EMPY_CHECK_EQ(broker.Get<uint32_t>("missing", 7u), 7u);
    EMPY_CHECK(broker.Get<bool>("tls", false));
    EMPY_CHECK_EQ(broker.Get<double>("ratio", 0.0), 0.5);
    EMPY_CHECK_EQ(broker["list"].Items().size(), 2u);
    EMPY_CHECK_EQ(broker["list"].Items()[1].As<std::string>(), std::string("b"));
    EMPY_CHECK_EQ(broker.Keys().size(), 5u);
    EMPY_CHECK_THROWS(broker.Get<uint16_t>("port", 0, 1, 1000));
    EMPY_CHECK_THROWS(broker.Require<std::string>("absent"));
    EMPY_CHECK_THROWS(broker.Get<uint8_t>("port", 0));
    EMPY_CHECK_THROWS(broker.Get<bool>("id", false));
    EMPY_CHECK_THROWS(YamlNode::Parse("a: b: c", "bad.yaml"));
    EMPY_CHECK_THROWS(YamlNode::Parse("key: [1, 2]]", "bad.yaml"));
    EMPY_CHECK_THROWS(YamlNode::Load("does/not/exist.yaml"));
    try
    {
        broker.Require<int32_t>("absent");
    }
    catch (const ConfigError &error)
    {
        EMPY_CHECK(std::string(error.what()).find("test.yaml.broker.absent") != std::string::npos);
    }
}
