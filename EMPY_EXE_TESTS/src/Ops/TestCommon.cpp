#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Common/Json.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eOps/Common/Process.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/HttpServer.h>
#include <eCore/Common/YAML.h>
#include <eCore/Common/Crypto.h>
#include <asio.hpp>
#include <asio/ssl.hpp>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <thread>

using namespace eOps;

namespace
{
    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsCommon")
        {
        }
    };

    using Clock = std::chrono::steady_clock;
    using asio::ip::tcp;

    int64_t ElapsedMs(Clock::time_point start)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }

    // one-connection raw TCP peer on an ephemeral loopback port; `session` runs on its thread
    class RawServer
    {
    public:
        explicit RawServer(std::function<void(tcp::socket &)> session)
            : m_Acceptor(m_Io, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0))
        {
            m_Port = m_Acceptor.local_endpoint().port();
            m_Thread = std::thread(
                [this, session = std::move(session)]
                {
                    std::error_code code;
                    tcp::socket socket(m_Io);
                    m_Acceptor.accept(socket, code);
                    m_Accepted = true;
                    if (!code)
                    {
                        try
                        {
                            session(socket);
                        }
                        catch (const std::exception &)
                        {
                        }
                    }
                });
        }

        ~RawServer()
        {
            if (!m_Accepted)
            {
                // unblock the pending accept with a throw-away connection
                asio::io_context io;
                tcp::socket poke(io);
                std::error_code ignored;
                poke.connect(tcp::endpoint(asio::ip::make_address("127.0.0.1"), m_Port), ignored);
            }
            m_Thread.join();
        }

        uint16_t Port() const
        {
            return m_Port;
        }

    private:
        asio::io_context m_Io;
        tcp::acceptor m_Acceptor;
        std::thread m_Thread;
        std::atomic<bool> m_Accepted{false};
        uint16_t m_Port = 0;
    };

    // read a request head (blocking) from a raw peer
    template <typename _Stream> std::string ReadHead(_Stream &stream)
    {
        std::string data;
        asio::read_until(stream, asio::dynamic_buffer(data), "\r\n\r\n");
        return data;
    }

    // wait until the peer closes the connection
    template <typename _Stream> void Drain(_Stream &stream)
    {
        char chunk[256];
        std::error_code code;
        while (!code)
        {
            stream.read_some(asio::buffer(chunk), code);
        }
    }

    // throw-away self-signed certificate with the given subjectAltName
    struct Certificate
    {
        TempDir m_Folder;
        std::string m_Certificate;
        std::string m_PrivateKey;

        explicit Certificate(const char *altNames)
        {
            m_Certificate = (m_Folder.m_Path / "certificate.pem").string();
            m_PrivateKey = (m_Folder.m_Path / "private-key.pem").string();
            std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
                EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "prime256v1"), EVP_PKEY_free);
            std::unique_ptr<X509, decltype(&X509_free)> cert(X509_new(), X509_free);
            EMPY_REQUIRE(key && cert);
            EMPY_REQUIRE(X509_set_version(cert.get(), 2) == 1);
            EMPY_REQUIRE(ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), 7) == 1);
            EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notBefore(cert.get()), -60));
            EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notAfter(cert.get()), 3600));
            EMPY_REQUIRE(X509_set_pubkey(cert.get(), key.get()) == 1);
            auto subject = X509_get_subject_name(cert.get());
            EMPY_REQUIRE(X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
                                                    reinterpret_cast<const unsigned char *>("EmpyOps Test"), -1, -1,
                                                    0) == 1);
            EMPY_REQUIRE(X509_set_issuer_name(cert.get(), subject) == 1);
            std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> san(
                X509V3_EXT_conf_nid(nullptr, nullptr, NID_subject_alt_name, const_cast<char *>(altNames)),
                X509_EXTENSION_free);
            EMPY_REQUIRE(san && X509_add_ext(cert.get(), san.get(), -1) == 1);
            std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> basic(
                X509V3_EXT_conf_nid(nullptr, nullptr, NID_basic_constraints, const_cast<char *>("critical,CA:TRUE")),
                X509_EXTENSION_free);
            EMPY_REQUIRE(basic && X509_add_ext(cert.get(), basic.get(), -1) == 1);
            EMPY_REQUIRE(X509_sign(cert.get(), key.get(), EVP_sha256()) > 0);
            auto certFile = std::fopen(m_Certificate.c_str(), "wb");
            EMPY_REQUIRE(certFile);
            EMPY_REQUIRE(PEM_write_X509(certFile, cert.get()) == 1);
            std::fclose(certFile);
            auto keyFile = std::fopen(m_PrivateKey.c_str(), "wb");
            EMPY_REQUIRE(keyFile);
            EMPY_REQUIRE(PEM_write_PrivateKey(keyFile, key.get(), nullptr, nullptr, 0, nullptr, nullptr) == 1);
            std::fclose(keyFile);
        }
    };

    // TLS peer serving `connections` sequential connections with a fixed response
    class TlsServer
    {
    public:
        TlsServer(const Certificate &certificate, size_t connections)
            : m_Context(asio::ssl::context::tls_server),
              m_Acceptor(m_Io, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0))
        {
            m_Context.use_certificate_chain_file(certificate.m_Certificate);
            m_Context.use_private_key_file(certificate.m_PrivateKey, asio::ssl::context::pem);
            m_Port = m_Acceptor.local_endpoint().port();
            m_Thread = std::thread(
                [this, connections]
                {
                    for (size_t i = 0; i < connections; ++i)
                    {
                        std::error_code code;
                        asio::ssl::stream<tcp::socket> stream(m_Io, m_Context);
                        m_Acceptor.accept(stream.lowest_layer(), code);
                        ++m_Accepted;
                        if (code)
                        {
                            continue;
                        }
                        stream.handshake(asio::ssl::stream_base::server, code);
                        if (code)
                        {
                            ++m_Failed;
                            continue;
                        }
                        try
                        {
                            std::string head = ReadHead(stream);
                            m_SawSni =
                                m_SawSni || SSL_get_servername(stream.native_handle(), TLSEXT_NAMETYPE_host_name);
                            std::string reply = "HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nsecure";
                            asio::write(stream, asio::buffer(reply));
                            stream.shutdown(code);
                            ++m_Served;
                        }
                        catch (const std::exception &)
                        {
                            ++m_Failed;
                        }
                    }
                });
        }

        ~TlsServer()
        {
            m_Thread.join();
        }

        uint16_t Port() const
        {
            return m_Port;
        }

        std::atomic<size_t> m_Accepted{0};
        std::atomic<size_t> m_Served{0};
        std::atomic<size_t> m_Failed{0};
        std::atomic<bool> m_SawSni{false};

    private:
        asio::io_context m_Io;
        asio::ssl::context m_Context;
        tcp::acceptor m_Acceptor;
        std::thread m_Thread;
        uint16_t m_Port = 0;
    };

    HttpRequest Get(std::string url, uint32_t timeoutMs = 5000)
    {
        HttpRequest request;
        request.m_Method = "GET";
        request.m_Url = std::move(url);
        request.m_TimeoutMs = timeoutMs;
        return request;
    }

    std::string Local(uint16_t port, std::string_view path = "/")
    {
        return "http://127.0.0.1:" + std::to_string(port) + std::string(path);
    }

    ProcessOptions Shell(std::vector<std::string> args)
    {
        ProcessOptions options;
#ifdef _WIN32
        options.m_Program = "cmd";
        args.insert(args.begin(), "/c");
#else
        options.m_Program = "/bin/sh";
        std::string script;
        for (const auto &arg : args)
        {
            script += (script.empty() ? "" : " ") + arg;
        }
        args = {"-c", script};
#endif
        options.m_Args = std::move(args);
        return options;
    }
} // namespace

// ---------------------------------------------------------------- Json

EMPY_TEST(OpsCommon, JsonRoundTrip)
{
    std::string text = R"({"name":"home-control","version":3,"ratio":0.25,"ok":true,"none":null,)"
                       R"("list":[1,-2,3.5,"x",[],{}],"nested":{"a":{"b":[{"c":"deep"}]}}})";
    Json value = Json::Parse(text);
    EMPY_CHECK_EQ(value.Dump(), text);
    EMPY_CHECK(Json::Parse(value.Dump()) == value);
    EMPY_CHECK(Json::Parse(value.Dump(2)) == value);
    EMPY_CHECK_EQ(value.GetString("name"), std::string("home-control"));
    EMPY_CHECK_EQ(value.GetInt("version"), 3);
    EMPY_CHECK_EQ(value.GetNumber("ratio"), 0.25);
    EMPY_CHECK(value.GetBool("ok"));
    EMPY_CHECK(value["none"].IsNull());
    EMPY_CHECK(value.Has("none"));
    EMPY_CHECK(!value.Has("missing"));
    EMPY_CHECK_EQ(value["list"].Size(), size_t(6));
    EMPY_CHECK_EQ(value["list"][2].AsNumber(), 3.5);
    EMPY_CHECK(value["list"][99].IsNull());
    EMPY_REQUIRE(value.At("nested.a.b.0.c"));
    EMPY_CHECK_EQ(value.At("nested.a.b.0.c")->AsString(), std::string("deep"));
    EMPY_CHECK(value.At("nested.a.b.1") == nullptr);
    EMPY_CHECK(value.At("list.x") == nullptr);
    EMPY_CHECK_EQ(Json::Parse(" \t\r\n[ 1 , 2 ]\n ").Dump(), std::string("[1,2]"));

    // pretty output layout
    Json small = Json::MakeObject({{"a", Json(1)}, {"b", Json::MakeArray({Json(true)})}, {"c", Json::MakeObject()}});
    EMPY_CHECK_EQ(small.Dump(2), std::string("{\n  \"a\": 1,\n  \"b\": [\n    true\n  ],\n  \"c\": {}\n}"));

    // mutation helpers
    Json built;
    built["x"] = Json(1);
    built.Set("y", Json("two")).Set("x", Json(3));
    EMPY_CHECK_EQ(built.Dump(), std::string(R"({"x":3,"y":"two"})"));
    EMPY_CHECK(built.Erase("x"));
    EMPY_CHECK(!built.Erase("x"));
    Json list;
    list.Push(Json(1)).Push(Json("a"));
    EMPY_CHECK_EQ(list.Dump(), std::string(R"([1,"a"])"));
    EMPY_CHECK_THROWS(list.Set("k", Json(1)));
    EMPY_CHECK_THROWS(built.Push(Json(1)));
    EMPY_CHECK_THROWS(Json(1).AsString());
    EMPY_CHECK_THROWS(Json("x").AsNumber());
    EMPY_CHECK_EQ(Json("x").GetString("k", "fallback"), std::string("fallback"));
    EMPY_CHECK_EQ(value.GetInt("name", -1), -1);

    // duplicate keys: last wins
    EMPY_CHECK_EQ(Json::Parse(R"({"a":1,"b":2,"a":3})").Dump(), std::string(R"({"a":3,"b":2})"));
    std::string many = "{";
    for (int32_t i = 0; i < 40; ++i)
    {
        many += "\"k" + std::to_string(i % 20) + "\":" + std::to_string(i) + (i == 39 ? "}" : ",");
    }
    Json duplicates = Json::Parse(many);
    EMPY_CHECK_EQ(duplicates.Size(), size_t(20));
    EMPY_CHECK_EQ(duplicates.GetInt("k0"), 20);
    EMPY_CHECK_EQ(duplicates.GetInt("k19"), 39);

    // bytes and YAML bridges
    EMPY_CHECK(FromBytes(ToBytes(value)) == value);
    auto yaml = eCore::YamlNode::Parse("a: 1\nb: [true, 2.5, text, \"7\"]\nc: {d: null, e: -4}\n");
    EMPY_CHECK_EQ(FromYaml(yaml).Dump(), std::string(R"({"a":1,"b":[true,2.5,"text","7"],"c":{"d":null,"e":-4}})"));
}

EMPY_TEST(OpsCommon, JsonStrings)
{
    Json escaped = Json::Parse(R"("q\" s\\ f\/ \b\f\n\r\t \u0001 \u00e9 \u20AC")");
    EMPY_CHECK_EQ(escaped.AsString(), std::string("q\" s\\ f/ \b\f\n\r\t \x01 \xC3\xA9 \xE2\x82\xAC"));
    // output escapes quotes, backslashes and controls, never '/'; UTF-8 stays raw
    EMPY_CHECK_EQ(escaped.Dump(), std::string("\"q\\\" s\\\\ f/ \\b\\f\\n\\r\\t \\u0001 \xC3\xA9 \xE2\x82\xAC\""));
    EMPY_CHECK_EQ(Json(std::string("\x1f\x7f", 2)).Dump(), std::string("\"\\u001f\x7f\""));
    EMPY_CHECK_EQ(std::string(Json::Parse(R"("\u0000")").AsString()), std::string(1, '\0'));

    // surrogate pairs
    EMPY_CHECK_EQ(std::string(Json::Parse(R"("\ud83d\ude00")").AsString()), std::string("\xF0\x9F\x98\x80"));
    EMPY_CHECK_EQ(std::string(Json::Parse(R"("\uD834\uDD1E")").AsString()), std::string("\xF0\x9D\x84\x9E"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\ud83d")"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\ud83dx")"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\ude00")"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\ud83d\u0041")"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\u12")"));
    EMPY_CHECK_THROWS(Json::Parse(R"("\x")"));

    // invalid UTF-8 and raw control characters are rejected on input
    EMPY_CHECK_THROWS(Json::Parse("\"\xC3\x28\""));
    EMPY_CHECK_THROWS(Json::Parse("\"\xC0\xAF\""));
    EMPY_CHECK_THROWS(Json::Parse("\"\xED\xA0\x80\""));
    EMPY_CHECK_THROWS(Json::Parse("\"\xF4\x90\x80\x80\""));
    EMPY_CHECK_THROWS(Json::Parse("\"\xE2\x82\""));
    EMPY_CHECK_THROWS(Json::Parse("\"tab\there\""));
    EMPY_CHECK_THROWS(Json::Parse("\"line\nbreak\""));
    EMPY_CHECK_THROWS(Json::Parse("\"unterminated"));
    EMPY_CHECK_EQ(std::string(Json::Parse("\"\xF0\x9F\x98\x80\"").AsString()), std::string("\xF0\x9F\x98\x80"));
    // ...and replaced on output so serialisation stays valid
    std::string replaced = Json(std::string("a\xFF"
                                            "b"))
                               .Dump();
    EMPY_CHECK_EQ(replaced, std::string("\"a\xEF\xBF\xBD"
                                        "b\""));
    EMPY_CHECK_EQ(std::string(Json::Parse(replaced).AsString()), std::string("a\xEF\xBF\xBD"
                                                                             "b"));
}

EMPY_TEST(OpsCommon, JsonNumbers)
{
    Json integer = Json::Parse("123");
    EMPY_CHECK(integer.IsInteger());
    EMPY_CHECK_EQ(integer.AsInt(), 123);
    EMPY_CHECK_EQ(Json::Parse("-9223372036854775808").AsInt(), std::numeric_limits<int64_t>::min());
    EMPY_CHECK_EQ(Json::Parse("9223372036854775807").AsInt(), std::numeric_limits<int64_t>::max());
    Json big = Json::Parse("18446744073709551615");
    EMPY_CHECK(big.IsInteger());
    EMPY_CHECK_EQ(big.AsUint(), std::numeric_limits<uint64_t>::max());
    EMPY_CHECK_EQ(big.Dump(), std::string("18446744073709551615"));
    EMPY_CHECK_THROWS(big.AsInt());
    Json beyond = Json::Parse("18446744073709551616");
    EMPY_CHECK(!beyond.IsInteger());
    EMPY_CHECK_EQ(beyond.AsNumber(), 18446744073709551616.0);
    EMPY_CHECK(!Json::Parse("-9223372036854775809").IsInteger());

    // integers print exactly, doubles shortest round-trip
    EMPY_CHECK_EQ(Json(int64_t(-42)).Dump(), std::string("-42"));
    EMPY_CHECK_EQ(Json(uint32_t(7)).Dump(), std::string("7"));
    EMPY_CHECK_EQ(Json(0.1).Dump(), std::string("0.1"));
    EMPY_CHECK_EQ(Json(1.5).Dump(), std::string("1.5"));
    EMPY_CHECK_EQ(Json(100.0).Dump(), std::string("100"));
    EMPY_CHECK_EQ(Json(1e21).Dump(), std::string("1e+21"));
    EMPY_CHECK_EQ(Json(-0.000001).Dump(), std::string("-1e-06"));
    EMPY_CHECK_EQ(Json::Parse("1e2").AsNumber(), 100.0);
    EMPY_CHECK(!Json::Parse("1e2").IsInteger());
    EMPY_CHECK_EQ(Json::Parse("1E-2").AsNumber(), 0.01);
    EMPY_CHECK_EQ(Json::Parse("-0.5e+1").AsNumber(), -5.0);
    EMPY_CHECK_EQ(Json::Parse("1e-400").AsNumber(), 0.0);
    double third = 1.0 / 3.0;
    EMPY_CHECK_EQ(Json::Parse(Json(third).Dump()).AsNumber(), third);
    long platformLong = 77;
    EMPY_CHECK_EQ(Json(platformLong).AsInt(), 77);
    EMPY_CHECK_EQ(Json(size_t(9)).AsUint(), uint64_t(9));
    EMPY_CHECK_EQ(Json(uint8_t(1)).AsInt(), 1);

    // conversions
    EMPY_CHECK_EQ(Json(5.0).AsInt(), 5);
    EMPY_CHECK_THROWS(Json(5.5).AsInt());
    EMPY_CHECK_THROWS(Json(int64_t(-1)).AsUint());
    EMPY_CHECK_EQ(Json(5.5).GetInt("x", 3), 3);
    EMPY_CHECK_THROWS(Json(std::nan("")).Dump());
    EMPY_CHECK_THROWS(Json(std::numeric_limits<double>::infinity()).Dump());

    // strict grammar
    for (const char *bad :
         {"01", "-", "1.", ".5", "+1", "1e", "1e+", "0x10", "NaN", "Infinity", "-01", "1.2.3", "1e999", "--1"})
    {
        bool threw = false;
        try
        {
            Json::Parse(bad);
        }
        catch (const JsonError &)
        {
            threw = true;
        }
        if (!threw)
        {
            eTest::Fail(__FILE__, __LINE__, std::string("accepted invalid number ") + bad);
        }
    }
}

EMPY_TEST(OpsCommon, JsonGrammarAndLimits)
{
    for (const char *bad : {"", " ", "nul", "tru", "[1,]", "[1 2]", "{\"a\" 1}", "{\"a\":1,}", "{a:1}", "[1]x",
                            "{\"a\":1}}", "[", "{", "\"a\" \"b\"", "\xEF\xBB\xBF{}", "/*c*/1", "'s'"})
    {
        bool threw = false;
        try
        {
            Json::Parse(bad);
        }
        catch (const JsonError &)
        {
            threw = true;
        }
        if (!threw)
        {
            eTest::Fail(__FILE__, __LINE__, std::string("accepted invalid document: ") + bad);
        }
    }
    EMPY_CHECK(Json::Parse("true").AsBool());
    EMPY_CHECK(!Json::Parse("false").AsBool());
    EMPY_CHECK(Json::Parse("null").IsNull());

    // depth: 64 nested containers pass, 65 fail; custom limits apply
    std::string deep(64, '[');
    deep += std::string(64, ']');
    EMPY_CHECK_EQ(Json::Parse(deep).Dump(), deep);
    std::string deeper(65, '[');
    deeper += std::string(65, ']');
    EMPY_CHECK_THROWS(Json::Parse(deeper));
    JsonLimits shallow;
    shallow.m_MaxDepth = 2;
    EMPY_CHECK(Json::Parse(R"({"a":[1]})", shallow).IsObject());
    EMPY_CHECK_THROWS(Json::Parse(R"({"a":[[1]]})", shallow));
    std::string hostile(100000, '[');
    EMPY_CHECK_THROWS(Json::Parse(hostile));

    // size limit
    JsonLimits tiny;
    tiny.m_MaxBytes = 8;
    EMPY_CHECK(Json::Parse("[1,2,3]", tiny).IsArray());
    EMPY_CHECK_THROWS(Json::Parse("[1,2,3,4,5]", tiny));
    EMPY_CHECK_THROWS(FromBytes(eCore::AsBytes("\"0123456789\""), tiny));
}

EMPY_TEST(OpsCommon, JsonEquality)
{
    EMPY_CHECK(Json::Parse(R"({"a":1,"b":[1,2],"c":{"x":null}})") ==
               Json::Parse(R"({"c":{"x":null},"b":[1,2],"a":1})"));
    EMPY_CHECK(!(Json::Parse(R"({"b":[1,2]})") == Json::Parse(R"({"b":[2,1]})")));
    EMPY_CHECK(!(Json::Parse(R"({"a":1})") == Json::Parse(R"({"a":1,"b":2})")));
    EMPY_CHECK(!(Json::Parse(R"({"a":1})") == Json::Parse(R"({"b":1})")));
    EMPY_CHECK(Json(1) == Json(1.0));
    EMPY_CHECK(Json(int64_t(5)) == Json(uint64_t(5)));
    EMPY_CHECK(!(Json(1) == Json("1")));
    EMPY_CHECK(!(Json() == Json(false)));
    EMPY_CHECK(!(Json(uint64_t(18446744073709551615ull)) == Json(int64_t(-1))));
    EMPY_CHECK(Json() == Json(nullptr));
    EMPY_CHECK(Json::MakeArray() == Json::Parse("[]"));
    EMPY_CHECK(!(Json::MakeArray() == Json::MakeObject()));
    // object construction keeps the first position of a duplicated key
    Json object = Json::MakeObject({{"k", Json(1)}, {"j", Json(2)}, {"k", Json(3)}});
    EMPY_CHECK_EQ(object.Dump(), std::string(R"({"k":3,"j":2})"));
}

// ---------------------------------------------------------------- Text

EMPY_TEST(OpsCommon, Text)
{
    EMPY_CHECK_EQ(Lower("MiXeD-123"), std::string("mixed-123"));
    EMPY_CHECK_EQ(Trim("  \t padded \r\n"), std::string("padded"));
    EMPY_CHECK_EQ(Trim("   "), std::string());
    EMPY_CHECK(Split("a,,b", ',') == (std::vector<std::string>{"a", "", "b"}));
    EMPY_CHECK(Split("", ',') == (std::vector<std::string>{""}));
    EMPY_CHECK_EQ(Join({"a", "b", "c"}, "/"), std::string("a/b/c"));
    EMPY_CHECK_EQ(Join({}, "/"), std::string());

    auto tokens = Tokenize("Keep the Temperature between 16 and 24 \xC2\xB0"
                           "C, p95 latency under 5ms; 90% accuracy "
                           "(0.9), every 24 hours on a GPU at home/temp, from -5 to 16-24.");
    std::vector<std::string> expected{"keep", "the",     "temperature", "between", "16", "and", "24",      "celsius",
                                      "p95",  "latency", "under",       "5",       "ms", "90",  "percent", "accuracy",
                                      "0.9",  "every",   "24",          "hours",   "on", "a",   "gpu",     "at",
                                      "home", "temp",    "from",        "-5",      "to", "16",  "24"};
    EMPY_CHECK_EQ(Join(tokens, " "), Join(expected, " "));
    EMPY_CHECK(Tokenize("20\xC2\xB0"
                        "f and 5\xC2\xB0") == (std::vector<std::string>{"20", "fahrenheit", "and", "5", "degree"}));
    EMPY_CHECK(Tokenize("caf\xC3\xA9 v1.2.3") == (std::vector<std::string>{"caf\xC3\xA9", "v1", "2.3"}));
    EMPY_CHECK(Tokenize("").empty());

    // one plural folding for the ontology and the retrieval engine
    for (const auto &[plural, singular] : std::vector<std::pair<std::string, std::string>>{{"gpus", "gpu"},
                                                                                           {"cpus", "cpu"},
                                                                                           {"npus", "npu"},
                                                                                           {"policies", "policy"},
                                                                                           {"switches", "switch"},
                                                                                           {"boxes", "box"},
                                                                                           {"classes", "class"},
                                                                                           {"nodes", "node"},
                                                                                           {"celsius", "celsius"},
                                                                                           {"status", "status"},
                                                                                           {"analysis", "analysis"},
                                                                                           {"glass", "glass"},
                                                                                           {"gps", "gps"},
                                                                                           {"ties", "tie"}})
    {
        EMPY_CHECK_EQ(Singular(plural), singular);
    }

    EMPY_CHECK(IsDnsLabel("home-control"));
    EMPY_CHECK(IsDnsLabel("a"));
    EMPY_CHECK(IsDnsLabel("x1-2y"));
    EMPY_CHECK(IsDnsLabel(std::string(63, 'a')));
    EMPY_CHECK(!IsDnsLabel(std::string(64, 'a')));
    EMPY_CHECK(!IsDnsLabel(""));
    EMPY_CHECK(!IsDnsLabel("-lead"));
    EMPY_CHECK(!IsDnsLabel("trail-"));
    EMPY_CHECK(!IsDnsLabel("Upper"));
    EMPY_CHECK(!IsDnsLabel("dot.ted"));
    EMPY_CHECK(!IsDnsLabel("under_score"));

    EMPY_CHECK_EQ(IsoTime(0), std::string("1970-01-01T00:00:00.000Z"));
    EMPY_CHECK_EQ(IsoTime(1700000000123ull), std::string("2023-11-14T22:13:20.123Z"));
    EMPY_CHECK_EQ(IsoTime(951782400000ull), std::string("2000-02-29T00:00:00.000Z"));
    uint64_t now = NowMillis();
    EMPY_CHECK(now > 1700000000000ull);
    EMPY_CHECK(IsoTime(now).starts_with("20"));

    std::string id = NewId("wf");
    EMPY_CHECK_EQ(id.size(), size_t(19));
    EMPY_CHECK(id.starts_with("wf-"));
    EMPY_CHECK(id.find_first_not_of("0123456789abcdef", 3) == std::string::npos);
    EMPY_CHECK(NewId("wf") != id);
    EMPY_CHECK_EQ(NewId({}).size(), size_t(16));

    EMPY_CHECK(ParseNumber("21.5") == std::optional<double>(21.5));
    EMPY_CHECK(ParseNumber("-3") == std::optional<double>(-3));
    EMPY_CHECK(ParseNumber("1e3") == std::optional<double>(1000));
    for (const char *bad : {"", " 1", "1 ", "+1", "1x", "inf", "nan", "0x10", "1,5"})
    {
        if (ParseNumber(bad))
        {
            eTest::Fail(__FILE__, __LINE__, std::string("ParseNumber accepted ") + bad);
        }
    }
}

EMPY_TEST(OpsCommon, TextFiles)
{
    TempDir directory;
    auto path = directory.m_Path / "nested" / "dir" / "state.json";
    WriteFileAtomic(path, "first");
    EMPY_CHECK_EQ(ReadFile(path), std::string("first"));
    std::string binary("a\0b\r\nc", 6);
    WriteFileAtomic(path, binary);
    EMPY_CHECK_EQ(ReadFile(path), binary);
    WriteFileAtomic(path, "");
    EMPY_CHECK_EQ(ReadFile(path), std::string());
    std::string large(200000, 'z');
    WriteFileAtomic(path, large);
    EMPY_CHECK_EQ(ReadFile(path).size(), large.size());
    EMPY_CHECK_THROWS(ReadFile(path, 1000));
    EMPY_CHECK_THROWS(ReadFile(directory.m_Path / "missing.txt"));
    // no temporary files are left behind
    size_t files = 0;
    for (const auto &entry : std::filesystem::directory_iterator(path.parent_path()))
    {
        files += entry.is_regular_file() ? 1 : 0;
    }
    EMPY_CHECK_EQ(files, size_t(1));
}

// ---------------------------------------------------------------- TopicAlgebra

EMPY_TEST(OpsCommon, TopicAlgebraOverlap)
{
    struct Row
    {
        const char *a;
        const char *b;
        bool overlap;
    };
    const Row rows[] = {
        {"home/temp", "home/temp", true},
        {"home/temp", "home/hum", false},
        {"home/+", "home/temp", true},
        {"home/+", "home", false},
        {"home/+", "home/a/b", false},
        {"home/#", "home", true},
        {"home/#", "home/a/b/c", true},
        {"home/#", "factory/x", false},
        {"#", "anything/at/all", true},
        {"+", "a", true},
        {"+", "a/b", false},
        {"+/+", "a/#", true},
        {"+/temp", "home/+", true},
        {"+/temp", "home/hum", false},
        {"a/+/c", "a/b/+", true},
        {"a/+/c", "a/b/d", false},
        {"a/#", "+/b/#", true},
        {"home/set/#", "home/set/temp", true},
        {"home/set/temp", "home/set/temp/x", false},
        {"$share/g/home/#", "home/temp", true},
        {"$share/g1/home/+", "$share/g2/+/temp", true},
        {"$SYS/broker", "#", false},
        {"$SYS/broker", "+/broker", false},
        {"$SYS/broker", "$SYS/#", true},
        {"$SYS/+", "$SYS/broker", true},
        {"$SYS/x", "$OTHER/x", false},
        {"a//b/", "a/b", true},
        {"bad/#/x", "bad/y", false},
        {"a/b+", "a/b", false},
    };
    for (const Row &row : rows)
    {
        bool forward = FiltersOverlap(row.a, row.b);
        bool backward = FiltersOverlap(row.b, row.a);
        if (forward != row.overlap || backward != row.overlap)
        {
            eTest::Fail(__FILE__, __LINE__,
                        std::string("FiltersOverlap(") + row.a + ", " + row.b + ") expected " +
                            (row.overlap ? "true" : "false"));
        }
    }
}

EMPY_TEST(OpsCommon, TopicAlgebraCover)
{
    struct Row
    {
        const char *outer;
        const char *inner;
        bool covers;
    };
    const Row rows[] = {
        {"home/#", "home/temp", true},
        {"home/#", "home", true},
        {"home/#", "home/+", true},
        {"home/#", "home/#", true},
        {"home/+", "home/temp", true},
        {"home/+", "home/#", false},
        {"home/temp", "home/+", false},
        {"home/temp", "home/temp", true},
        {"home/+", "home/+", true},
        {"#", "a/b", true},
        {"#", "+/x", true},
        {"#", "$SYS/x", false},
        {"+/x", "$SYS/x", false},
        {"$SYS/#", "$SYS/x", true},
        {"+/+", "a/b", true},
        {"+/+", "a/b/c", false},
        {"a/+/#", "a", false},
        {"a/+/#", "a/b", true},
        {"a/#", "a/+/c", true},
        {"home/set/#", "home/set/temp", true},
        {"home/set/temp", "home/set/#", false},
        {"$share/g/home/#", "home/x", true},
        {"home/#", "$share/g/home/x", true},
        {"x/#", "y/#", false},
        {"invalid/#/x", "invalid/a", false},
    };
    for (const Row &row : rows)
    {
        if (FilterCovers(row.outer, row.inner) != row.covers)
        {
            eTest::Fail(__FILE__, __LINE__,
                        std::string("FilterCovers(") + row.outer + ", " + row.inner + ") expected " +
                            (row.covers ? "true" : "false"));
        }
    }

    EMPY_CHECK(IsFilter("home/+"));
    EMPY_CHECK(IsFilter("home/#"));
    EMPY_CHECK(IsFilter("$share/g/home/temp"));
    EMPY_CHECK(!IsFilter("home/temp"));
    EMPY_CHECK(ValidTopic("home/temp", false));
    EMPY_CHECK(!ValidTopic("home/+", false));
    EMPY_CHECK(ValidTopic("home/+", true));
    EMPY_CHECK(ValidTopic("$share/g/home/#", true));
    EMPY_CHECK(!ValidTopic("$share/g/home/#", false));
    EMPY_CHECK(!ValidTopic("home/#/x", true));
    EMPY_CHECK(!ValidTopic("home/te+mp", true));
    EMPY_CHECK(!ValidTopic("", false));
    EMPY_CHECK(!ValidTopic(std::string(300, 'a'), false));
    EMPY_CHECK(!ValidTopic("bad\xFF", false));
    EMPY_CHECK_EQ(TopicLevel("mlops/event/home-control/drift", 2), std::string("home-control"));
    EMPY_CHECK_EQ(TopicLevel("mlops/event", 5), std::string());
    EMPY_CHECK_EQ(TopicLevel("/a//b/", 1), std::string("b"));
}

// ---------------------------------------------------------------- Process

EMPY_TEST(OpsCommon, ProcessPipesAndExit)
{
    Process child;
    ProcessOptions options = Shell({"echo", "one&", "echo", "two"});
#ifndef _WIN32
    options.m_Args = {"-c", "echo one; echo two"};
#endif
    options.m_PipeStdout = true;
    std::string error;
    EMPY_REQUIRE(child.Start(options, &error));
    EMPY_CHECK(child.Pid() != 0);
    EMPY_CHECK(child.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("one"));
    EMPY_CHECK(child.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("two"));
    auto start = Clock::now();
    EMPY_CHECK(!child.ReadLine(std::chrono::seconds(10)));
    EMPY_CHECK(ElapsedMs(start) < 5000);
    EMPY_CHECK(child.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(0));
    EMPY_CHECK(!child.Running());
    EMPY_CHECK(!child.Start(options, &error));
    EMPY_CHECK(!error.empty());

    Process failing;
    EMPY_REQUIRE(failing.Start(Shell({"exit", "7"})));
    EMPY_CHECK(failing.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(7));

    Process missing;
    ProcessOptions nothing;
    nothing.m_Program = "empy-ops-no-such-program-" + NewId({});
    error.clear();
    EMPY_CHECK(!missing.Start(nothing, &error));
    EMPY_CHECK(!error.empty());
    EMPY_CHECK(!missing.Running());
    EMPY_CHECK(!missing.Wait(std::chrono::milliseconds(1)));
}

EMPY_TEST(OpsCommon, ProcessStdinArgumentsEnvironment)
{
    // stdin pipe through sort
    Process sorter;
    ProcessOptions sort;
    sort.m_Program = "sort";
    sort.m_PipeStdin = true;
    sort.m_PipeStdout = true;
    EMPY_REQUIRE(sorter.Start(sort));
    EMPY_CHECK(sorter.Write("pear\napple\n"));
    sorter.CloseStdin();
    EMPY_CHECK(!sorter.Write("late"));
    EMPY_CHECK(sorter.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("apple"));
    EMPY_CHECK(sorter.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("pear"));
    EMPY_CHECK(sorter.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(0));

    // arguments with spaces and quotes arrive quoted (cmd's echo prints them verbatim)
    Process echo;
    ProcessOptions arguments = Shell({"echo", "a b", "x\"y", "plain"});
#ifndef _WIN32
    arguments.m_Args = {"-c", "printf '%s|' \"$@\"", "sh", "a b", "x\"y", "plain"};
#endif
    arguments.m_PipeStdout = true;
    EMPY_REQUIRE(echo.Start(arguments));
#ifdef _WIN32
    EMPY_CHECK(echo.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("\"a b\" \"x\\\"y\" plain"));
#else
    EMPY_CHECK(echo.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("a b|x\"y|plain|"));
#endif
    EMPY_CHECK(echo.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(0));

    // environment overrides and working directory
    TempDir directory;
    Process environment;
#ifdef _WIN32
    ProcessOptions variables = Shell({"echo", "%EMPY_OPS_TEST_VALUE%&", "cd"});
#else
    ProcessOptions variables = Shell({"echo $EMPY_OPS_TEST_VALUE; pwd"});
#endif
    variables.m_Environment = {{"EMPY_OPS_TEST_VALUE", "hello-env"}};
    variables.m_WorkingDirectory = directory.m_Path;
    variables.m_PipeStdout = true;
    EMPY_REQUIRE(environment.Start(variables));
    EMPY_CHECK(environment.ReadLine(std::chrono::seconds(10)) == std::optional<std::string>("hello-env"));
    auto cwd = environment.ReadLine(std::chrono::seconds(10));
    EMPY_REQUIRE(cwd);
    std::error_code code;
    EMPY_CHECK(std::filesystem::equivalent(std::filesystem::path(*cwd), directory.m_Path, code));
    EMPY_CHECK(environment.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(0));

    // stdout and stderr appended to one shared file
    auto log = directory.m_Path / "child.log";
    WriteFileAtomic(log, "previous\n");
    Process logged;
#ifdef _WIN32
    ProcessOptions output = Shell({"echo", "out&", "echo", "err", "1>&2"});
#else
    ProcessOptions output = Shell({"echo out; echo err 1>&2"});
#endif
    output.m_StdoutFile = log;
    output.m_StderrFile = log;
    EMPY_REQUIRE(logged.Start(output));
    EMPY_CHECK(logged.Wait(std::chrono::seconds(10)) == std::optional<int32_t>(0));
    std::string text = ReadFile(log);
    EMPY_CHECK(text.starts_with("previous\n"));
    EMPY_CHECK(text.find("out") != std::string::npos);
    EMPY_CHECK(text.find("err") != std::string::npos);
}

EMPY_TEST(OpsCommon, ProcessTimeoutAndKill)
{
    Process sleeper;
#ifdef _WIN32
    ProcessOptions options;
    options.m_Program = "ping";
    options.m_Args = {"-n", "60", "127.0.0.1"};
#else
    ProcessOptions options = Shell({"sleep 60"});
#endif
    options.m_PipeStdout = true;
    EMPY_REQUIRE(sleeper.Start(options));
    auto start = Clock::now();
    EMPY_CHECK(!sleeper.Wait(std::chrono::milliseconds(200)));
    EMPY_CHECK(ElapsedMs(start) >= 150);
    EMPY_CHECK(sleeper.Running());
    // drain any output; a timed-out read keeps the child running
    while (sleeper.ReadLine(std::chrono::milliseconds(50)))
    {
    }
    EMPY_CHECK(sleeper.Running());
    start = Clock::now();
    sleeper.Kill();
    EMPY_CHECK(!sleeper.Running());
    auto code = sleeper.Wait(std::chrono::milliseconds(0));
    EMPY_REQUIRE(code);
    EMPY_CHECK(*code != 0);
    EMPY_CHECK(ElapsedMs(start) < 5000);
    EMPY_CHECK(!sleeper.ReadLine(std::chrono::seconds(5)));

    // the destructor kills a running child
    start = Clock::now();
    {
        Process abandoned;
        EMPY_REQUIRE(abandoned.Start(options));
        EMPY_CHECK(abandoned.Running());
    }
    EMPY_CHECK(ElapsedMs(start) < 5000);

    // a child that never reads its input: a timed write gives up at its deadline instead of blocking
    Process deaf;
    ProcessOptions piped = options;
    piped.m_PipeStdin = true;
    EMPY_REQUIRE(deaf.Start(piped));
    std::string large(4 << 20, 'x');
    start = Clock::now();
    EMPY_CHECK(!deaf.Write(large, std::chrono::milliseconds(300)));
    EMPY_CHECK(ElapsedMs(start) >= 250);
    EMPY_CHECK(ElapsedMs(start) < 5000);
    EMPY_CHECK(deaf.Running());
    deaf.Kill();
    EMPY_CHECK(!deaf.Running());
}

// ---------------------------------------------------------------- Http

EMPY_TEST(OpsCommon, HttpUrl)
{
    Url url = ParseUrl("HTTPS://api.example.com/v1/messages?x=1#frag");
    EMPY_CHECK_EQ(url.m_Scheme, std::string("https"));
    EMPY_CHECK_EQ(url.m_Host, std::string("api.example.com"));
    EMPY_CHECK_EQ(url.m_Port, uint16_t(443));
    EMPY_CHECK_EQ(url.m_Target, std::string("/v1/messages?x=1"));
    url = ParseUrl("http://127.0.0.1:8080");
    EMPY_CHECK_EQ(url.m_Port, uint16_t(8080));
    EMPY_CHECK_EQ(url.m_Target, std::string("/"));
    url = ParseUrl("http://[::1]:9/a");
    EMPY_CHECK_EQ(url.m_Host, std::string("::1"));
    EMPY_CHECK_EQ(url.m_Port, uint16_t(9));
    url = ParseUrl("http://host?q");
    EMPY_CHECK_EQ(url.m_Target, std::string("/?q"));
    EMPY_CHECK_EQ(ParseUrl("http://h").m_Port, uint16_t(80));
    for (const char *bad :
         {"ftp://h/", "h/path", "http://", "http://:80/", "http://h:0/", "http://h:70000/", "http://user:pw@h/",
          "http://h:/", "http://[::1/", "http://h:8a/", "http://a b/", "https://api.example/v1\r\nX-Injected: 1",
          "http://h/a b", "http://h/\x7f", "http://h/?q=\"x\"", "http://h/caf\xc3\xa9", "http://h/\t"})
    {
        bool threw = false;
        try
        {
            ParseUrl(bad);
        }
        catch (const HttpError &)
        {
            threw = true;
        }
        if (!threw)
        {
            eTest::Fail(__FILE__, __LINE__, std::string("ParseUrl accepted ") + bad);
        }
    }
    EMPY_CHECK_EQ(ParseUrl("http://h/a%20b?x=1&y=(2)").m_Target, std::string("/a%20b?x=1&y=(2)"));
    for (const char *loopback : {"localhost", "LocalHost", "127.0.0.1", "127.255.0.9", "::1"})
    {
        EMPY_CHECK(IsLoopbackHost(loopback));
    }
    for (const char *remote : {"api.example.com", "128.0.0.1", "127.0.0", "127.0.0.256", "localhost.evil", "::2"})
    {
        EMPY_CHECK(!IsLoopbackHost(remote));
    }
}

EMPY_TEST(OpsCommon, HttpServerRoundTrip)
{
    std::atomic<size_t> calls{0};
    HttpServer server(
        "127.0.0.1", 0,
        [&](const HttpServerRequest &request)
        {
            ++calls;
            if (request.m_Target == "/boom")
            {
                throw std::runtime_error("handler exploded");
            }
            HttpResponse response;
            response.m_Status = request.m_Target == "/missing" ? 404 : 200;
            response.m_Headers = {{"Content-Type", "application/json"}, {"X-Echo", request.Header("x-test")}};
            Json body = Json::MakeObject({{"method", Json(request.m_Method)},
                                          {"target", Json(request.m_Target)},
                                          {"body", Json(request.m_Body)}});
            response.m_Body = body.Dump();
            return response;
        },
        4096);
    uint16_t port = server.Start();
    EMPY_REQUIRE(port != 0);
    EMPY_CHECK_EQ(server.Port(), port);
    EMPY_CHECK_THROWS(server.Start());

    HttpRequest post;
    post.m_Url = Local(port, "/api/echo?x=1");
    post.m_Body = R"({"hello":"world"})";
    post.m_Headers = {{"X-Test", "value-1"}, {"Content-Length", "999"}, {"Host", "ignored"}};
    post.m_TimeoutMs = 5000;
    HttpResponse response = SendHttp(post);
    EMPY_CHECK_EQ(response.m_Status, 200);
    EMPY_CHECK_EQ(response.Header("content-type"), std::string("application/json"));
    EMPY_CHECK_EQ(response.Header("X-ECHO"), std::string("value-1"));
    EMPY_CHECK_EQ(response.Header("Content-Length"), std::to_string(response.m_Body.size()));
    EMPY_CHECK_EQ(response.Header("absent"), std::string());
    Json echoed = Json::Parse(response.m_Body);
    EMPY_CHECK_EQ(echoed.GetString("method"), std::string("POST"));
    EMPY_CHECK_EQ(echoed.GetString("target"), std::string("/api/echo?x=1"));
    EMPY_CHECK_EQ(echoed.GetString("body"), post.m_Body);

    EMPY_CHECK_EQ(SendHttp(Get(Local(port, "/missing"))).m_Status, 404);
    HttpResponse failure = SendHttp(Get(Local(port, "/boom")));
    EMPY_CHECK_EQ(failure.m_Status, 500);
    EMPY_CHECK(failure.m_Body.find("handler exploded") != std::string::npos);
    HttpRequest head = Get(Local(port, "/head"));
    head.m_Method = "HEAD";
    HttpResponse headResponse = SendHttp(head);
    EMPY_CHECK_EQ(headResponse.m_Status, 200);
    EMPY_CHECK(headResponse.m_Body.empty());
    EMPY_CHECK(headResponse.Header("Content-Length") != "0");

    // oversize request → 413 without reaching the handler
    size_t before = calls;
    HttpRequest large;
    large.m_Url = Local(port, "/large");
    large.m_Body = std::string(10000, 'x');
    EMPY_CHECK_EQ(SendHttp(large).m_Status, 413);
    EMPY_CHECK_EQ(calls.load(), before);

    // header injection is refused client-side
    HttpRequest injected = Get(Local(port));
    injected.m_Headers = {{"X-Bad", "a\r\nX-Evil: 1"}};
    EMPY_CHECK_THROWS(SendHttp(injected));

    // the server is still healthy and stops promptly
    EMPY_CHECK_EQ(SendHttp(Get(Local(port, "/again"))).m_Status, 200);
    auto start = Clock::now();
    server.Stop();
    server.Stop();
    EMPY_CHECK(ElapsedMs(start) < 2000);
    EMPY_CHECK_THROWS(SendHttp(Get(Local(port), 2000)));
}

EMPY_TEST(OpsCommon, HttpServerReadTimeoutAndStop)
{
    HttpServer server("127.0.0.1", 0,
                      [](const HttpServerRequest &)
                      {
                          HttpResponse response;
                          response.m_Status = 200;
                          response.m_Body = "ok";
                          return response;
                      });
    uint16_t port = server.Start();
    // an idle client is dropped after the 5 s read timeout; the next request is served afterwards
    asio::io_context io;
    tcp::socket idle(io);
    idle.connect(tcp::endpoint(asio::ip::make_address("127.0.0.1"), port));
    auto start = Clock::now();
    HttpResponse response = SendHttp(Get(Local(port), 15000));
    EMPY_CHECK_EQ(response.m_Status, 200);
    EMPY_CHECK_EQ(response.m_Body, std::string("ok"));
    EMPY_CHECK(ElapsedMs(start) >= 4000);
    EMPY_CHECK(ElapsedMs(start) < 12000);
    char byte = 0;
    std::error_code code;
    idle.read_some(asio::buffer(&byte, 1), code);
    EMPY_CHECK(static_cast<bool>(code));

    // Stop aborts an active idle connection without waiting for its timeout
    tcp::socket second(io);
    second.connect(tcp::endpoint(asio::ip::make_address("127.0.0.1"), port));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    start = Clock::now();
    server.Stop();
    EMPY_CHECK(ElapsedMs(start) < 2000);

    EMPY_CHECK_THROWS(
        HttpServer("not-an-address", 0, [](const HttpServerRequest &) { return HttpResponse{}; }).Start());
}

EMPY_TEST(OpsCommon, HttpBodies)
{
    // chunked with extensions and trailers, after an interim 100 Continue
    {
        std::string seen;
        RawServer raw(
            [&](tcp::socket &socket)
            {
                seen = ReadHead(socket);
                asio::write(socket,
                            asio::buffer(std::string("HTTP/1.1 100 Continue\r\n\r\n"
                                                     "HTTP/1.1 201 Created\r\nTransfer-Encoding: chunked\r\n"
                                                     "X-Kind: chunks\r\n\r\n"
                                                     "5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\nX-Trailer: y\r\n\r\n")));
                Drain(socket);
            });
        HttpRequest request;
        request.m_Url = Local(raw.Port(), "/chunked");
        request.m_Body = "payload";
        HttpResponse response = SendHttp(request);
        EMPY_CHECK_EQ(response.m_Status, 201);
        EMPY_CHECK_EQ(response.m_Body, std::string("hello world"));
        EMPY_CHECK_EQ(response.Header("x-kind"), std::string("chunks"));
        // the request carried Host, Content-Length and Connection: close
        EMPY_CHECK(seen.starts_with("POST /chunked HTTP/1.1\r\n"));
        EMPY_CHECK(seen.find("Host: 127.0.0.1:" + std::to_string(raw.Port()) + "\r\n") != std::string::npos);
        EMPY_CHECK(seen.find("Content-Length: 7\r\n") != std::string::npos);
        EMPY_CHECK(seen.find("Connection: close\r\n") != std::string::npos);
    }
    // Content-Length body with trailing bytes ignored
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket, asio::buffer(std::string("HTTP/1.1 200 OK\r\ncontent-length: 4\r\n\r\nbodyEXTRA")));
                Drain(socket);
            });
        HttpResponse response = SendHttp(Get(Local(raw.Port())));
        EMPY_CHECK_EQ(response.m_Body, std::string("body"));
    }
    // close-delimited body
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket, asio::buffer(std::string("HTTP/1.1 200 OK\r\n\r\nuntil-close")));
                socket.shutdown(tcp::socket::shutdown_both);
            });
        EMPY_CHECK_EQ(SendHttp(Get(Local(raw.Port()))).m_Body, std::string("until-close"));
    }
    // truncated Content-Length body → transport error
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket, asio::buffer(std::string("HTTP/1.1 200 OK\r\nContent-Length: 50\r\n\r\nshort")));
                socket.shutdown(tcp::socket::shutdown_both);
            });
        EMPY_CHECK_THROWS(SendHttp(Get(Local(raw.Port()))));
    }
    // oversize responses are rejected (declared length and chunked growth)
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket, asio::buffer(std::string("HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n")));
                Drain(socket);
            });
        HttpRequest request = Get(Local(raw.Port()));
        request.m_MaxResponseBytes = 200;
        EMPY_CHECK_THROWS(SendHttp(request));
    }
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                std::string reply = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
                for (int32_t i = 0; i < 20; ++i)
                {
                    reply += "10\r\n0123456789abcdef\r\n";
                }
                reply += "0\r\n\r\n";
                asio::write(socket, asio::buffer(reply));
                Drain(socket);
            });
        HttpRequest request = Get(Local(raw.Port()));
        request.m_MaxResponseBytes = 200;
        EMPY_CHECK_THROWS(SendHttp(request));
    }
    // malformed status line and chunk size
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket, asio::buffer(std::string("SPDY/9 nonsense\r\n\r\n")));
                Drain(socket);
            });
        EMPY_CHECK_THROWS(SendHttp(Get(Local(raw.Port()))));
    }
    {
        RawServer raw(
            [](tcp::socket &socket)
            {
                ReadHead(socket);
                asio::write(socket,
                            asio::buffer(std::string("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n")));
                Drain(socket);
            });
        EMPY_CHECK_THROWS(SendHttp(Get(Local(raw.Port()))));
    }
}

EMPY_TEST(OpsCommon, HttpTimeouts)
{
    // a peer that accepts and never answers trips the overall deadline
    RawServer raw(
        [](tcp::socket &socket)
        {
            ReadHead(socket);
            Drain(socket);
        });
    auto start = Clock::now();
    bool timedOut = false;
    try
    {
        SendHttp(Get(Local(raw.Port()), 300));
    }
    catch (const HttpError &error)
    {
        timedOut = std::string(error.what()).find("timed out") != std::string::npos;
    }
    EMPY_CHECK(timedOut);
    EMPY_CHECK(ElapsedMs(start) >= 250);
    EMPY_CHECK(ElapsedMs(start) < 3000);

    // nothing listening → connection error, not a hang
    uint16_t closed = 0;
    {
        asio::io_context io;
        tcp::acceptor probe(io, tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0));
        closed = probe.local_endpoint().port();
    }
    start = Clock::now();
    EMPY_CHECK_THROWS(SendHttp(Get(Local(closed), 5000)));
    EMPY_CHECK(ElapsedMs(start) < 5000);
    EMPY_CHECK_THROWS(SendHttp(Get("http://empy-ops-unresolvable-host.invalid/", 5000)));
}

EMPY_TEST(OpsCommon, HttpsVerification)
{
    Certificate certificate("DNS:localhost,IP:127.0.0.1");
    {
        // untrusted self-signed certificate is rejected; trusted via the CA file it succeeds
        TlsServer server(certificate, 4);
        std::string base = "https://127.0.0.1:" + std::to_string(server.Port());
        bool rejected = false;
        try
        {
            SendHttp(Get(base + "/", 5000));
        }
        catch (const HttpError &error)
        {
            rejected = std::string(error.what()).find("verification") != std::string::npos;
        }
        EMPY_CHECK(rejected);

        HttpRequest trusted = Get(base + "/", 5000);
        trusted.m_CaFile = certificate.m_Certificate;
        HttpResponse response = SendHttp(trusted);
        EMPY_CHECK_EQ(response.m_Status, 200);
        EMPY_CHECK_EQ(response.m_Body, std::string("secure"));

        // DNS name with SNI
        HttpRequest named = Get("https://localhost:" + std::to_string(server.Port()) + "/", 5000);
        named.m_CaFile = certificate.m_Certificate;
        EMPY_CHECK_EQ(SendHttp(named).m_Body, std::string("secure"));
        EMPY_CHECK(server.m_SawSni.load());

        // verification disabled explicitly
        HttpRequest unverified = Get(base + "/", 5000);
        unverified.m_VerifyPeer = false;
        EMPY_CHECK_EQ(SendHttp(unverified).m_Body, std::string("secure"));

        HttpRequest missingCa = Get(base + "/", 5000);
        missingCa.m_CaFile = (certificate.m_Folder.m_Path / "absent.pem").string();
        EMPY_CHECK_THROWS(SendHttp(missingCa));
    }
    {
        // trusted certificate for another name fails hostname verification
        Certificate other("DNS:example.test");
        TlsServer server(other, 1);
        HttpRequest mismatch = Get("https://127.0.0.1:" + std::to_string(server.Port()) + "/", 5000);
        mismatch.m_CaFile = other.m_Certificate;
        bool rejected = false;
        try
        {
            SendHttp(mismatch);
        }
        catch (const HttpError &error)
        {
            rejected = std::string(error.what()).find("verification") != std::string::npos;
        }
        EMPY_CHECK(rejected);
    }
}
