#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Intelligence/Provider.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Intelligence/Anthropic.h>
#include <eOps/Intelligence/Command.h>
#include <eOps/Common/HttpServer.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <mutex>

using namespace eOps;
using namespace eOps::Intelligence;

namespace
{
    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsIntelligence")
        {
        }
    };

    using Clock = std::chrono::steady_clock;

    int64_t ElapsedMs(Clock::time_point start)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }

    bool Contains(std::string_view text, std::string_view part)
    {
        return text.find(part) != std::string_view::npos;
    }

    // process environment variable set for the scope of a test (unique names keep cases independent)
    struct ScopedEnv
    {
        std::string m_Name;
        explicit ScopedEnv(std::string prefix) : m_Name(std::move(prefix) + "_" + NewId({}))
        {
        }
        void Set(const std::string &value) const
        {
#ifdef _WIN32
            _putenv_s(m_Name.c_str(), value.c_str());
#else
            setenv(m_Name.c_str(), value.c_str(), 1);
#endif
        }
        void Clear() const
        {
#ifdef _WIN32
            _putenv_s(m_Name.c_str(), "");
#else
            unsetenv(m_Name.c_str());
#endif
        }
        ~ScopedEnv()
        {
            Clear();
        }
    };

    HttpResponse Reply(int32_t status, std::string body, std::vector<std::pair<std::string, std::string>> headers = {})
    {
        HttpResponse response;
        response.m_Status = status;
        response.m_Body = std::move(body);
        response.m_Headers = std::move(headers);
        response.m_Headers.emplace_back("content-type", "application/json");
        return response;
    }

    std::string Message(std::string_view stopReason, Json content, Json extra = Json::MakeObject())
    {
        Json message = Json::MakeObject({{"id", "msg_test"},
                                         {"type", "message"},
                                         {"role", "assistant"},
                                         {"model", "claude-opus-5-5"},
                                         {"content", std::move(content)},
                                         {"stop_reason", std::string(stopReason)},
                                         {"usage", Json::MakeObject({{"input_tokens", 42}, {"output_tokens", 7}})}});
        for (const auto &[key, value] : extra.AsObject())
        {
            message.Set(key, value);
        }
        return message.Dump();
    }

    Json TextBlock(std::string text)
    {
        return Json::MakeObject({{"type", "text"}, {"text", std::move(text)}});
    }

    // Messages API stand-in on 127.0.0.1: scripted replies, recorded requests
    class MockApi
    {
    public:
        MockApi()
            : m_Server("127.0.0.1", 0,
                       [this](const HttpServerRequest &request)
                       {
                           std::scoped_lock lock(m_Mutex);
                           m_Requests.push_back(request);
                           if (m_Replies.empty())
                           {
                               return Reply(500,
                                            R"({"type":"error","error":{"type":"api_error","message":"no reply"}})");
                           }
                           HttpResponse response = m_Replies.front();
                           m_Replies.pop_front();
                           return response;
                       })
        {
            m_Port = m_Server.Start();
        }

        void Push(HttpResponse response)
        {
            std::scoped_lock lock(m_Mutex);
            m_Replies.push_back(std::move(response));
        }

        std::vector<HttpServerRequest> Requests()
        {
            std::scoped_lock lock(m_Mutex);
            return m_Requests;
        }

        std::string BaseUrl() const
        {
            return "http://127.0.0.1:" + std::to_string(m_Port);
        }

    private:
        std::mutex m_Mutex;
        std::deque<HttpResponse> m_Replies;
        std::vector<HttpServerRequest> m_Requests;
        HttpServer m_Server;
        uint16_t m_Port = 0;
    };

    AnthropicOptions MockOptions(const MockApi &api, const ScopedEnv &key)
    {
        AnthropicOptions options;
        options.m_BaseUrl = api.BaseUrl() + "/";
        options.m_ApiKeyEnv = key.m_Name;
        options.m_TimeoutMs = 10000;
        return options;
    }

    // scripted in-process provider for AskJson
    struct FakeProvider : Provider
    {
        Completion m_Result;
        bool m_Available = true;
        Prompt m_Seen;
        std::string Name() const override
        {
            return "fake";
        }
        bool Available() const override
        {
            return m_Available;
        }
        Completion Complete(const Prompt &prompt) override
        {
            m_Seen = prompt;
            return m_Result;
        }
    };

    Json FrameSchema()
    {
        return SchemaObject({{"name", SchemaScalar("string")},
                             {"replicas", SchemaScalar("integer")},
                             {"regions", SchemaArray(SchemaEnum({"ran", "edge", "cloud"}))},
                             {"latency", SchemaNullable(SchemaScalar("number"))}});
    }

    std::vector<std::string> Shell(const std::string &command)
    {
#ifdef _WIN32
        return {"cmd", "/c", command};
#else
        return {"/bin/sh", "-c", command};
#endif
    }
} // namespace

EMPY_TEST(OpsIntelligence, CapabilitiesAndDeterministic)
{
    EMPY_CHECK_EQ(std::string(CapabilityName(Capability::Reasoning)), std::string("reasoning"));
    EMPY_CHECK_EQ(std::string(CapabilityName(Capability::Planning)), std::string("planning"));
    EMPY_CHECK_EQ(std::string(CapabilityName(Capability::Generation)), std::string("generation"));
    EMPY_CHECK(ParseCapability(" Planning ") == Capability::Planning);
    EMPY_CHECK(!ParseCapability("dreaming").has_value());

    DeterministicProvider deterministic;
    EMPY_CHECK_EQ(deterministic.Name(), std::string("deterministic"));
    EMPY_CHECK(!deterministic.Available());
    EMPY_CHECK(!deterministic.Complete(Prompt{}).m_Ok);

    ProviderSet set;
    EMPY_CHECK(set.Get(Capability::Reasoning) == nullptr);
    auto fake = std::make_shared<FakeProvider>();
    set.Set(Capability::Planning, fake);
    EMPY_CHECK(set.Get(Capability::Planning) == fake.get());
    fake->m_Available = false;
    EMPY_CHECK(set.Get(Capability::Planning) == nullptr); // unavailable → deterministic logic
    Json described = set.Describe();
    EMPY_CHECK_EQ(described["planning"].GetString("provider"), std::string("fake"));
    EMPY_CHECK(!described["planning"].GetBool("available", true));
    EMPY_CHECK_EQ(described["reasoning"].GetString("provider"), std::string("deterministic"));
}

EMPY_TEST(OpsIntelligence, AnthropicRequestContract)
{
    MockApi api;
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    key.Set("sk-test-secret-1234");
    AnthropicProvider provider(MockOptions(api, key));
    EMPY_CHECK(provider.Available());
    EMPY_CHECK_EQ(provider.Name(), std::string("anthropic:claude-opus-5-5"));

    api.Push(Reply(
        200, Message("end_turn",
                     Json::MakeArray({TextBlock("Hello, "),
                                      Json::MakeObject({{"type", "thinking"}, {"thinking", ""}, {"signature", "sig"}}),
                                      TextBlock("operator.")}))));
    Prompt prompt;
    prompt.m_Capability = Capability::Generation;
    prompt.m_System = "You write HITL questions.";
    prompt.m_User = "Ask about home/set/temp.";
    Completion completion = provider.Complete(prompt);
    EMPY_REQUIRE(completion.m_Ok);
    EMPY_CHECK_EQ(completion.m_Text, std::string("Hello, operator."));
    EMPY_CHECK(!completion.m_Json.has_value());
    EMPY_CHECK_EQ(completion.m_StopReason, std::string("end_turn"));
    EMPY_CHECK_EQ(completion.m_InputTokens, 42u);
    EMPY_CHECK_EQ(completion.m_OutputTokens, 7u);
    EMPY_CHECK_EQ(completion.m_Model, std::string("claude-opus-5-5"));
    EMPY_CHECK_EQ(completion.m_Provider, std::string("anthropic:claude-opus-5-5"));

    std::vector<HttpServerRequest> requests = api.Requests();
    EMPY_REQUIRE_EQ(requests.size(), size_t(1));
    const HttpServerRequest &request = requests.front();
    EMPY_CHECK_EQ(request.m_Method, std::string("POST"));
    EMPY_CHECK_EQ(request.m_Target, std::string("/v1/messages"));
    EMPY_CHECK_EQ(request.Header("x-api-key"), std::string("sk-test-secret-1234"));
    EMPY_CHECK_EQ(request.Header("anthropic-version"), std::string("2023-06-01"));
    EMPY_CHECK_EQ(request.Header("anthropic-beta"), std::string("server-side-fallback-2026-07-01"));
    EMPY_CHECK(Contains(request.Header("content-type"), "application/json"));

    const Json body = Json::Parse(request.m_Body);
    EMPY_CHECK_EQ(body.GetString("model"), std::string("claude-opus-5-5"));
    EMPY_CHECK_EQ(body.GetInt("max_tokens"), int64_t(16000));
    EMPY_CHECK_EQ(body.GetString("system"), std::string("You write HITL questions."));
    EMPY_REQUIRE_EQ(body["messages"].Size(), size_t(1));
    EMPY_CHECK_EQ(body["messages"][0].GetString("role"), std::string("user"));
    EMPY_CHECK_EQ(body["messages"][0].GetString("content"), std::string("Ask about home/set/temp."));
    EMPY_CHECK_EQ(body["output_config"].GetString("effort"), std::string("medium"));
    EMPY_CHECK(!body["output_config"].Has("format"));
    EMPY_CHECK_EQ(body.GetString("fallbacks"), std::string("default"));
    for (const char *forbidden : {"temperature", "top_p", "top_k", "thinking", "stop_sequences"})
    {
        EMPY_CHECK(!body.Has(forbidden));
    }
    EMPY_CHECK(!Contains(request.m_Body, "budget_tokens"));
    EMPY_CHECK(!Contains(request.m_Body, "assistant")); // no prefill turn
    EMPY_CHECK(!Contains(provider.RequestBody(prompt).Dump(), "sk-test-secret"));
}

EMPY_TEST(OpsIntelligence, AnthropicStructuredOutput)
{
    MockApi api;
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    key.Set("sk-test-structured");
    AnthropicOptions options = MockOptions(api, key);
    options.m_Fallbacks = false;
    AnthropicProvider provider(options);

    api.Push(
        Reply(200, Message("end_turn",
                           Json::MakeArray({TextBlock(
                               R"({"name":"home-control","replicas":2,"regions":["ran","edge"],"latency":5})")}))));
    Prompt prompt;
    prompt.m_User = "Interpret the intent.";
    prompt.m_Schema = FrameSchema();
    prompt.m_Effort = "high";
    prompt.m_MaxTokens = 0; // → options default
    Completion completion = provider.Complete(prompt);
    EMPY_REQUIRE(completion.m_Ok);
    EMPY_REQUIRE(completion.m_Json.has_value());
    EMPY_CHECK_EQ(completion.m_Json->GetString("name"), std::string("home-control"));
    EMPY_CHECK_EQ(completion.m_Json->GetInt("replicas"), int64_t(2));

    std::vector<HttpServerRequest> requests = api.Requests();
    EMPY_REQUIRE_EQ(requests.size(), size_t(1));
    EMPY_CHECK_EQ(requests[0].Header("anthropic-beta"), std::string()); // fallbacks disabled
    Json body = Json::Parse(requests[0].m_Body);
    EMPY_CHECK(!body.Has("fallbacks"));
    EMPY_CHECK(!body.Has("system")); // empty system prompt omitted
    EMPY_CHECK_EQ(body.GetInt("max_tokens"), int64_t(16000));
    EMPY_CHECK_EQ(body["output_config"].GetString("effort"), std::string("high"));
    EMPY_CHECK_EQ(body["output_config"]["format"].GetString("type"), std::string("json_schema"));
    EMPY_CHECK(body["output_config"]["format"]["schema"] == FrameSchema());
    EMPY_CHECK(!body.Has("temperature"));

    // the same exchange through AskJson validates the schema
    api.Push(Reply(200, Message("end_turn", Json::MakeArray({TextBlock(
                                                R"({"name":"x","replicas":1,"regions":["moon"],"latency":null})")}))));
    std::string error;
    EMPY_CHECK(!AskJson(&provider, Capability::Reasoning, "system", "user", FrameSchema(), &error).has_value());
    EMPY_CHECK(Contains(error, "regions.0"));
    api.Push(Reply(200, Message("end_turn", Json::MakeArray({TextBlock(
                                                R"({"name":"x","replicas":1,"regions":["cloud"],"latency":null})")}))));
    std::optional<Json> value = AskJson(&provider, Capability::Reasoning, "system", "user", FrameSchema(), &error);
    EMPY_REQUIRE(value.has_value());
    EMPY_CHECK_EQ(value->GetString("name"), std::string("x"));

    // structured text that is not JSON fails the completion
    api.Push(Reply(200, Message("end_turn", Json::MakeArray({TextBlock("I cannot produce that.")}))));
    Completion broken = provider.Complete(prompt);
    EMPY_CHECK(!broken.m_Ok);
    EMPY_CHECK(Contains(broken.m_Error, "not valid JSON"));
}

EMPY_TEST(OpsIntelligence, AnthropicRefusalAndTruncation)
{
    MockApi api;
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    key.Set("sk-test-refusal");
    AnthropicProvider provider(MockOptions(api, key));
    Prompt prompt;
    prompt.m_User = "something";

    api.Push(
        Reply(200, Message("refusal", Json::MakeArray(),
                           Json::MakeObject({{"stop_details", Json::MakeObject({{"type", "refusal"},
                                                                                {"category", "cyber"},
                                                                                {"explanation", "declined"}})}}))));
    Completion refused = provider.Complete(prompt);
    EMPY_CHECK(!refused.m_Ok);
    EMPY_CHECK_EQ(refused.m_StopReason, std::string("refusal"));
    EMPY_CHECK(Contains(refused.m_Error, "refus"));
    EMPY_CHECK(Contains(refused.m_Error, "cyber"));

    api.Push(Reply(200, Message("refusal", Json::MakeArray({TextBlock("partial")}),
                                Json::MakeObject({{"stop_details", Json()}}))));
    Completion unspecified = provider.Complete(prompt);
    EMPY_CHECK(!unspecified.m_Ok);
    EMPY_CHECK(Contains(unspecified.m_Error, "unspecified"));
    EMPY_CHECK(unspecified.m_Text.empty()); // content is never read after a refusal

    api.Push(Reply(200, Message("max_tokens", Json::MakeArray({TextBlock("{\"name\": \"trunc")}))));
    Completion truncated = provider.Complete(prompt);
    EMPY_CHECK(!truncated.m_Ok);
    EMPY_CHECK_EQ(truncated.m_StopReason, std::string("max_tokens"));
    EMPY_CHECK(Contains(truncated.m_Error, "truncated"));

    // ParseResponse is usable without HTTP
    EMPY_CHECK(!provider.ParseResponse("not json", false).m_Ok);
    EMPY_CHECK(provider.ParseResponse(Message("end_turn", Json::MakeArray({TextBlock("ok")})), false).m_Ok);
    EMPY_CHECK_EQ(api.Requests().size(), size_t(3)); // refusals and truncation are not retried
}

EMPY_TEST(OpsIntelligence, AnthropicRetries)
{
    MockApi api;
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    key.Set("sk-test-retry-secret");
    AnthropicOptions options = MockOptions(api, key);
    options.m_MaxRetries = 2;
    AnthropicProvider provider(options);
    Prompt prompt;
    prompt.m_User = "hello";

    // 429 with retry-after: 1 → waits ~1 s, then succeeds
    api.Push(Reply(429, R"({"type":"error","error":{"type":"rate_limit_error","message":"slow down"}})",
                   {{"retry-after", "1"}}));
    api.Push(Reply(200, Message("end_turn", Json::MakeArray({TextBlock("after retry")}))));
    auto start = Clock::now();
    Completion completion = provider.Complete(prompt);
    int64_t elapsed = ElapsedMs(start);
    EMPY_REQUIRE(completion.m_Ok);
    EMPY_CHECK_EQ(completion.m_Text, std::string("after retry"));
    EMPY_CHECK_EQ(api.Requests().size(), size_t(2));
    EMPY_CHECK(elapsed >= 900);
    EMPY_CHECK(elapsed < 8000);

    // 400 is not retried and surfaces error.message, never the key
    api.Push(Reply(400,
                   R"({"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: too large"}})",
                   {{"request-id", "req_123"}}));
    Completion invalid = provider.Complete(prompt);
    EMPY_CHECK(!invalid.m_Ok);
    EMPY_CHECK(Contains(invalid.m_Error, "400"));
    EMPY_CHECK(Contains(invalid.m_Error, "invalid_request_error"));
    EMPY_CHECK(Contains(invalid.m_Error, "max_tokens: too large"));
    EMPY_CHECK(Contains(invalid.m_Error, "req_123"));
    EMPY_CHECK(!Contains(invalid.m_Error, "sk-test-retry-secret"));
    EMPY_CHECK_EQ(api.Requests().size(), size_t(3));

    // 5xx exhausts the retry budget (retry-after: 0 keeps the case fast)
    for (int i = 0; i < 3; ++i)
    {
        api.Push(Reply(529, R"({"type":"error","error":{"type":"overloaded_error","message":"busy"}})",
                       {{"retry-after", "0"}}));
    }
    Completion overloaded = provider.Complete(prompt);
    EMPY_CHECK(!overloaded.m_Ok);
    EMPY_CHECK(Contains(overloaded.m_Error, "overloaded_error"));
    EMPY_CHECK(Contains(overloaded.m_Error, "3 attempts"));
    EMPY_CHECK_EQ(api.Requests().size(), size_t(6));

    // the server can veto a retry
    api.Push(Reply(503, R"({"type":"error","error":{"type":"api_error","message":"final"}})",
                   {{"x-should-retry", "false"}}));
    EMPY_CHECK(!provider.Complete(prompt).m_Ok);
    EMPY_CHECK_EQ(api.Requests().size(), size_t(7));
}

EMPY_TEST(OpsIntelligence, AnthropicTransportFailure)
{
    // a port that was just released: connection refused, no retries configured
    uint16_t port = 0;
    {
        HttpServer probe("127.0.0.1", 0, [](const HttpServerRequest &) { return HttpResponse{}; });
        port = probe.Start();
    }
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    key.Set("sk-test-transport");
    AnthropicOptions options;
    options.m_BaseUrl = "http://127.0.0.1:" + std::to_string(port);
    options.m_ApiKeyEnv = key.m_Name;
    options.m_MaxRetries = 0;
    options.m_TimeoutMs = 5000;
    AnthropicProvider provider(options);
    Prompt prompt;
    prompt.m_User = "hello";
    Completion completion = provider.Complete(prompt);
    EMPY_CHECK(!completion.m_Ok);
    EMPY_CHECK(Contains(completion.m_Error, "transport"));
    EMPY_CHECK(!Contains(completion.m_Error, "sk-test-transport"));
}

EMPY_TEST(OpsIntelligence, AnthropicMissingKey)
{
    MockApi api;
    ScopedEnv key("EMPY_TEST_ANTHROPIC_KEY");
    AnthropicProvider provider(MockOptions(api, key));
    EMPY_CHECK(!provider.Available());
    Prompt prompt;
    prompt.m_User = "hello";
    Completion completion = provider.Complete(prompt);
    EMPY_CHECK(!completion.m_Ok);
    EMPY_CHECK(Contains(completion.m_Error, "unavailable"));
    EMPY_CHECK(Contains(completion.m_Error, key.m_Name));
    EMPY_CHECK(api.Requests().empty());

    ProviderSet set;
    set.Set(Capability::Reasoning, std::make_shared<AnthropicProvider>(MockOptions(api, key)));
    EMPY_CHECK(set.Get(Capability::Reasoning) == nullptr);
    std::string error;
    EMPY_CHECK(!AskJson(set.Get(Capability::Reasoning), Capability::Reasoning, "s", "u", FrameSchema(), &error));
    EMPY_CHECK(!error.empty());
    key.Set("sk-now-present");
    EMPY_CHECK(set.Get(Capability::Reasoning) != nullptr);
    EMPY_CHECK(set.Describe()["reasoning"].GetBool("available"));
}

EMPY_TEST(OpsIntelligence, ProviderSetFromYaml)
{
    ScopedEnv key("EMPY_TEST_YAML_KEY");
    std::string configPath = (std::filesystem::temp_directory_path() / "mlops" / "Harness.yaml").string();
    eCore::YamlNode root = eCore::YamlNode::Parse("intelligence:\n"
                                                  "  reasoning:\n"
                                                  "    provider: anthropic\n"
                                                  "    model: claude-opus-5-5\n"
                                                  "    api_key_env: " +
                                                      key.m_Name +
                                                      "\n"
                                                      "    base_url: http://127.0.0.1:9\n"
                                                      "    effort: high\n"
                                                      "    max_tokens: 8000\n"
                                                      "    timeout_ms: 30000\n"
                                                      "    max_retries: 1\n"
                                                      "    fallbacks: false\n"
                                                      "    ca_file: certs/ca.pem\n"
                                                      "  planning:\n"
                                                      "    provider: command\n"
                                                      "    command: [tools/model.exe, --json]\n"
                                                      "    timeout_ms: 5000\n"
                                                      "    name: local-llm\n",
                                                  configPath);
    ProviderSet set = ProviderSet::FromYaml(root["intelligence"], configPath);
    Json described = set.Describe();
    EMPY_CHECK_EQ(described["reasoning"].GetString("provider"), std::string("anthropic:claude-opus-5-5"));
    EMPY_CHECK(!described["reasoning"].GetBool("available", true));
    EMPY_CHECK_EQ(described["planning"].GetString("provider"), std::string("local-llm"));
    EMPY_CHECK(described["planning"].GetBool("available"));
    EMPY_CHECK_EQ(described["generation"].GetString("provider"), std::string("deterministic"));
    EMPY_CHECK(set.Get(Capability::Generation) == nullptr);
    EMPY_CHECK(set.Get(Capability::Reasoning) == nullptr);
    key.Set("sk-yaml");
    auto *anthropic = dynamic_cast<AnthropicProvider *>(set.Get(Capability::Reasoning));
    EMPY_REQUIRE(anthropic != nullptr);
    const AnthropicOptions &options = anthropic->Options();
    EMPY_CHECK_EQ(options.m_Effort, std::string("high"));
    EMPY_CHECK_EQ(options.m_MaxTokens, 8000u);
    EMPY_CHECK_EQ(options.m_TimeoutMs, 30000u);
    EMPY_CHECK_EQ(options.m_MaxRetries, 1u);
    EMPY_CHECK(!options.m_Fallbacks);
    EMPY_CHECK_EQ(std::filesystem::path(options.m_CaFile),
                  (std::filesystem::path(configPath).parent_path() / "certs" / "ca.pem").lexically_normal());

    // empty / missing → all deterministic
    ProviderSet empty = ProviderSet::FromYaml(eCore::YamlNode::Parse("other: 1")["intelligence"], configPath);
    EMPY_CHECK_EQ(empty.Describe()["reasoning"].GetString("provider"), std::string("deterministic"));
    EMPY_CHECK_EQ(empty.Describe()["planning"].GetString("provider"), std::string("deterministic"));

    auto rejects = [&](const std::string &text)
    {
        bool threw = false;
        try
        {
            ProviderSet::FromYaml(eCore::YamlNode::Parse(text)["intelligence"], configPath);
        }
        catch (const eCore::ConfigError &)
        {
            threw = true;
        }
        if (!threw)
        {
            eTest::Fail(__FILE__, __LINE__, "accepted: " + text);
        }
    };
    rejects("intelligence: {reasoning: {provider: oracle}}");
    rejects("intelligence: {dreaming: {provider: deterministic}}");
    rejects("intelligence: {reasoning: {provider: anthropic, effort: extreme}}");
    rejects("intelligence: {reasoning: {provider: anthropic, base_url: 'ftp://x'}}");
    // the API key never travels over plain http to a remote host
    rejects("intelligence: {reasoning: {provider: anthropic, base_url: 'http://api.example.com'}}");
    rejects("intelligence: {reasoning: {provider: anthropic, max_retries: 99}}");
    rejects("intelligence: {reasoning: {provider: anthropic, temperature: 0.2}}");
    rejects("intelligence: {planning: {provider: command}}");
    rejects("intelligence: [1, 2]");
}

EMPY_TEST(OpsIntelligence, AskJsonConformance)
{
    FakeProvider fake;
    std::string error;
    EMPY_CHECK(!AskJson(nullptr, Capability::Reasoning, "s", "u", FrameSchema(), &error));
    EMPY_CHECK(Contains(error, "no intelligence provider"));

    fake.m_Result.m_Ok = true;
    fake.m_Result.m_Text = "Here you go:\n```json\n{\"name\": \"a\", \"replicas\": 3, \"regions\": [], \"latency\": "
                           "2.5}\n```";
    std::optional<Json> value = AskJson(&fake, Capability::Planning, "sys", "usr", FrameSchema(), &error);
    EMPY_REQUIRE(value.has_value());
    EMPY_CHECK_EQ(value->GetInt("replicas"), int64_t(3));
    EMPY_CHECK(fake.m_Seen.m_Capability == Capability::Planning);
    EMPY_CHECK_EQ(fake.m_Seen.m_System, std::string("sys"));
    EMPY_CHECK(fake.m_Seen.m_Schema.has_value());

    auto rejected = [&](std::string text, std::string_view expected)
    {
        fake.m_Result.m_Json.reset();
        fake.m_Result.m_Text = std::move(text);
        std::string reason;
        if (AskJson(&fake, Capability::Reasoning, "s", "u", FrameSchema(), &reason) || !Contains(reason, expected))
        {
            eTest::Fail(__FILE__, __LINE__, "expected rejection with '" + std::string(expected) + "', got: " + reason);
        }
    };
    rejected(R"({"replicas": 1, "regions": [], "latency": null})", "missing required member 'name'");
    rejected(R"({"name": 5, "replicas": 1, "regions": [], "latency": null})", "name: expected string");
    rejected(R"({"name": "a", "replicas": 1.5, "regions": [], "latency": null})", "replicas: expected integer");
    rejected(R"({"name": "a", "replicas": 1, "regions": "ran", "latency": null})", "regions: expected array");
    rejected(R"({"name": "a", "replicas": 1, "regions": [], "latency": "fast"})", "latency");
    rejected(R"({"name": "a", "replicas": 1, "regions": [], "latency": null, "extra": true})",
             "unexpected member 'extra'");
    rejected("no json here", "no JSON");

    fake.m_Result.m_Json = Json::Parse(R"({"name": "a", "replicas": 1, "regions": ["edge"], "latency": 4})");
    fake.m_Result.m_Text = "ignored";
    EMPY_CHECK(AskJson(&fake, Capability::Reasoning, "s", "u", FrameSchema(), &error).has_value());

    fake.m_Result.m_Ok = false;
    fake.m_Result.m_Error = "boom";
    EMPY_CHECK(!AskJson(&fake, Capability::Reasoning, "s", "u", FrameSchema(), &error));
    EMPY_CHECK(Contains(error, "boom"));
    fake.m_Available = false;
    EMPY_CHECK(!AskJson(&fake, Capability::Reasoning, "s", "u", FrameSchema(), &error));
    EMPY_CHECK(Contains(error, "not available"));
}

EMPY_TEST(OpsIntelligence, StructuredHelpers)
{
    Json schema = FrameSchema();
    EMPY_CHECK_EQ(schema.GetString("type"), std::string("object"));
    EMPY_CHECK(!schema.GetBool("additionalProperties", true));
    EMPY_CHECK_EQ(schema["required"].Size(), size_t(4));
    EMPY_CHECK_THROWS(SchemaScalar("date"));
    EMPY_CHECK_EQ(SchemaScalar("number", "ms").GetString("description"), std::string("ms"));
    EMPY_CHECK_EQ(SchemaNullable(SchemaScalar("string"))["anyOf"].Size(), size_t(2));

    std::string error;
    EMPY_CHECK(Conforms(Json::Parse("5.0"), SchemaScalar("integer"), &error));
    EMPY_CHECK(!Conforms(Json::Parse("true"), SchemaScalar("number"), &error));
    EMPY_CHECK_EQ(error, std::string("<root>: expected number"));
    EMPY_CHECK(Conforms(Json(), SchemaNullable(SchemaScalar("string"))));
    EMPY_CHECK(!Conforms(Json("z"), SchemaEnum({"a", "b"})));
    Json nested = SchemaObject({{"steps", SchemaArray(SchemaObject({{"agent", SchemaEnum({"placement"})}}))}});
    EMPY_CHECK(!Conforms(Json::Parse(R"({"steps": [{"agent": "placement"}, {"agent": "rogue"}]})"), nested, &error));
    EMPY_CHECK(Contains(error, "steps.1.agent"));

    EMPY_CHECK(ExtractJson(R"(  {"a": 1}  )").has_value());
    EMPY_CHECK(ExtractJson("[1, 2]").has_value());
    std::optional<Json> fenced = ExtractJson("text\n```json\n{\"a\": 2}\n```\nmore");
    EMPY_REQUIRE(fenced.has_value());
    EMPY_CHECK_EQ(fenced->GetInt("a"), int64_t(2));
    std::optional<Json> embedded = ExtractJson("The answer is {\"a\": {\"b\": 3}} as requested.");
    EMPY_REQUIRE(embedded.has_value());
    EMPY_CHECK_EQ(embedded->At("a.b")->AsInt(), int64_t(3));
    EMPY_CHECK(!ExtractJson("nothing to see").has_value());
    EMPY_CHECK(!ExtractJson("{broken").has_value());
    EMPY_CHECK(!ExtractJson("").has_value());
}

EMPY_TEST(OpsIntelligence, CommandProviderProcess)
{
    TempDir directory;
    std::filesystem::path reply = directory.m_Path / "reply.json";
    WriteFileAtomic(reply, R"({"text": "local answer", "json": {"name": "local", "replicas": 1, "regions": ["edge"],)"
                           R"( "latency": null}})");
    Prompt prompt;
    prompt.m_Capability = Capability::Planning;
    prompt.m_System = "plan";
    prompt.m_User = "place it";
    prompt.m_MaxTokens = 123;
    prompt.m_Schema = FrameSchema();

#ifdef _WIN32
    std::vector<std::string> cat = Shell("type " + reply.string());
    std::vector<std::string> echo = {"findstr", "/r", "."};
    std::vector<std::string> slow = {"ping", "-n", "30", "127.0.0.1"};
#else
    std::vector<std::string> cat = {"cat", reply.string()};
    std::vector<std::string> echo = {"cat"};
    std::vector<std::string> slow = {"sleep", "30"};
#endif
    // {text, json} object output
    CommandProvider object(CommandOptions{cat, 20000, "", 4 << 20});
    EMPY_CHECK(object.Available());
    EMPY_CHECK(Contains(object.Name(), "command:"));
    Completion completion = object.Complete(prompt);
    EMPY_REQUIRE(completion.m_Ok);
    EMPY_CHECK_EQ(completion.m_Text, std::string("local answer"));
    EMPY_REQUIRE(completion.m_Json.has_value());
    EMPY_CHECK_EQ(completion.m_Json->GetString("name"), std::string("local"));
    std::string error;
    EMPY_CHECK(AskJson(&object, Capability::Planning, "s", "u", FrameSchema(), &error).has_value());

    // the stdin request line comes back verbatim through an echoing child
    CommandProvider echoing(CommandOptions{echo, 20000, "echo", 4 << 20});
    EMPY_CHECK_EQ(echoing.Name(), std::string("echo"));
    Completion echoed = echoing.Complete(prompt);
    EMPY_REQUIRE(echoed.m_Ok);
    EMPY_REQUIRE(echoed.m_Json.has_value());
    EMPY_CHECK_EQ(echoed.m_Json->GetString("capability"), std::string("planning"));
    EMPY_CHECK_EQ(echoed.m_Json->GetString("system"), std::string("plan"));
    EMPY_CHECK_EQ(echoed.m_Json->GetString("user"), std::string("place it"));
    EMPY_CHECK_EQ(echoed.m_Json->GetInt("maxTokens"), int64_t(123));
    EMPY_CHECK((*echoed.m_Json)["schema"] == FrameSchema());

    // raw text output without a schema
    Prompt plain;
    plain.m_User = "say something";
    CommandProvider raw(CommandOptions{Shell("echo plain answer"), 20000, "", 4 << 20});
    Completion text = raw.Complete(plain);
    EMPY_REQUIRE(text.m_Ok);
    EMPY_CHECK_EQ(text.m_Text, std::string("plain answer"));
    EMPY_CHECK(!text.m_Json.has_value());
    // raw text that is not JSON fails a structured prompt
    EMPY_CHECK(!raw.Complete(prompt).m_Ok);

    // non-zero exit, timeout (the child is killed), launch failure, empty command
    CommandProvider failing(CommandOptions{Shell("exit 3"), 20000, "", 4 << 20});
    Completion failed = failing.Complete(plain);
    EMPY_CHECK(!failed.m_Ok);
    EMPY_CHECK(Contains(failed.m_Error, "status 3"));

    CommandProvider sleeper(CommandOptions{slow, 500, "", 4 << 20});
    auto start = Clock::now();
    Completion late = sleeper.Complete(plain);
    EMPY_CHECK(!late.m_Ok);
    EMPY_CHECK(Contains(late.m_Error, "timed out"));
    EMPY_CHECK(ElapsedMs(start) < 10000);
    // a request larger than the pipe buffer to a command that never reads its input: the write is bounded too
    Prompt large = plain;
    large.m_User.assign(512 * 1024, 'x');
    CommandProvider deaf(CommandOptions{slow, 800, "", 4 << 20});
    start = Clock::now();
    Completion blocked = deaf.Complete(large);
    EMPY_CHECK(!blocked.m_Ok);
    EMPY_CHECK(Contains(blocked.m_Error, "timed out"));
    EMPY_CHECK(ElapsedMs(start) < 10000);

    CommandProvider missing(CommandOptions{{(directory.m_Path / "no-such-model.exe").string()}, 2000, "", 4 << 20});
    Completion absent = missing.Complete(plain);
    EMPY_CHECK(!absent.m_Ok);
    EMPY_CHECK(Contains(absent.m_Error, "cannot launch"));

    CommandProvider none(CommandOptions{});
    EMPY_CHECK(!none.Available());
    EMPY_CHECK(!none.Complete(plain).m_Ok);

    // output bound
    CommandProvider chatty(CommandOptions{cat, 20000, "", 16});
    Completion bounded = chatty.Complete(plain);
    EMPY_CHECK(!bounded.m_Ok);
    EMPY_CHECK(Contains(bounded.m_Error, "exceeds"));
}
