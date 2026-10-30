#pragma once
#include <string>
#include <cstdint>
#include "Provider.h"

// remote provider over the Claude Messages API (raw HTTPS through eOps::SendHttp; C++ has no
// official SDK). Request: POST {base}/v1/messages with content-type: application/json,
// x-api-key: $<m_ApiKeyEnv>, anthropic-version: 2023-06-01 and, with fallbacks,
// anthropic-beta: server-side-fallback-2026-07-01 plus body "fallbacks": "default". Body:
// {model, max_tokens, system, messages: [{role: "user", content}], output_config: {effort,
// format?: {type: "json_schema", schema}}}; never thinking budgets, temperature/top_p/top_k or an
// assistant prefill (rejected by claude-opus-5-5, whose thinking is always adaptive).
// Response: stop_reason "refusal" → failure naming stop_details.category; "max_tokens" →
// failure "truncated"; text = concatenated content[] blocks of type "text" (thinking blocks are
// ignored); with a schema the text is parsed into m_Json. 429/5xx/connection errors are retried
// with exponential backoff honouring retry-after (≤ m_MaxRetries); other 4xx are not retried and
// surface error.message. The API key is never logged or placed in error text.

namespace eOps::Intelligence
{
    struct AnthropicOptions
    {
        std::string m_BaseUrl = "https://api.anthropic.com";
        std::string m_ApiKeyEnv = "ANTHROPIC_API_KEY";
        std::string m_Model = "claude-opus-5-5";
        std::string m_Effort = "medium"; // claude-opus-5-5 defaults to medium; sent explicitly
        uint32_t m_MaxTokens = 16000;    // used when Prompt::m_MaxTokens is 0
        uint32_t m_TimeoutMs = 120000;   // per HTTP attempt
        uint32_t m_MaxRetries = 2;
        bool m_Fallbacks = true;
        std::string m_CaFile; // extra trust anchors (tests, proxies); "" → system trust
    };

    class EMPY_OPS_API AnthropicProvider : public Provider
    {
    public:
        explicit AnthropicProvider(AnthropicOptions options = {});

        std::string Name() const override; // "anthropic:<model>"
        bool Available() const override;   // API key environment variable is non-empty
        Completion Complete(const Prompt &prompt) override;

        // the JSON body Complete would send for `prompt` (no credentials inside)
        Json RequestBody(const Prompt &prompt) const;
        // interpret one HTTP 200 response body (stop reason, text blocks, structured JSON, usage)
        Completion ParseResponse(const std::string &body, bool structured) const;
        const AnthropicOptions &Options() const noexcept;

    private:
        AnthropicOptions m_Options;
    };
} // namespace eOps::Intelligence
