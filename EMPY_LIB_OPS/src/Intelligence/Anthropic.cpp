#include <eOps/Intelligence/Anthropic.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <thread>

namespace eOps::Intelligence
{
    namespace
    {
        constexpr const char *API_VERSION = "2023-06-01";
        constexpr const char *FALLBACK_BETA = "server-side-fallback-2026-07-01";
        constexpr uint32_t BACKOFF_BASE_MS = 500;
        constexpr uint32_t BACKOFF_CAP_MS = 8000;
        constexpr uint32_t RETRY_AFTER_CAP_MS = 60000;
        constexpr size_t MAX_ERROR_TEXT = 512;

        std::string ApiKey(const std::string &variable)
        {
            if (variable.empty())
            {
                return {};
            }
            const char *value = std::getenv(variable.c_str());
            return value ? Trim(value) : std::string();
        }

        uint32_t Tokens(const Json &usage, std::string_view key)
        {
            const Json *value = usage.Find(key);
            if (!value || !value->IsNumber() || value->AsNumber() < 0)
            {
                return 0;
            }
            double number = value->AsNumber();
            return number >= std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                                  : static_cast<uint32_t>(number);
        }

        std::string Bounded(std::string text)
        {
            if (text.size() > MAX_ERROR_TEXT)
            {
                text.resize(MAX_ERROR_TEXT);
                text += "...";
            }
            return text;
        }

        // "<status> <error.type>: <error.message>" from an API error body
        std::string DescribeFailure(const HttpResponse &response)
        {
            std::string text = "HTTP " + std::to_string(response.m_Status);
            try
            {
                Json body = Json::Parse(response.m_Body);
                if (const Json *error = body.Find("error"); error && error->IsObject())
                {
                    std::string type = error->GetString("type");
                    std::string message = error->GetString("message");
                    text += type.empty() ? "" : " " + type;
                    text += message.empty() ? "" : ": " + Bounded(message);
                }
            }
            catch (const JsonError &)
            {
                if (!response.m_Body.empty())
                {
                    text += ": " + Bounded(response.m_Body);
                }
            }
            std::string request = response.Header("request-id");
            if (!request.empty())
            {
                text += " (request-id " + Bounded(request) + ")";
            }
            return text;
        }

        bool RetryableStatus(int32_t status)
        {
            return status == 408 || status == 409 || status == 429 || status >= 500;
        }

        // server hint: retry-after-ms, else retry-after seconds; nullopt when absent or unparsable
        std::optional<uint32_t> RetryAfter(const HttpResponse &response)
        {
            if (std::optional<double> ms = ParseNumber(Trim(response.Header("retry-after-ms"))); ms && *ms >= 0)
            {
                return static_cast<uint32_t>(std::min<double>(*ms, RETRY_AFTER_CAP_MS));
            }
            if (std::optional<double> seconds = ParseNumber(Trim(response.Header("retry-after")));
                seconds && *seconds >= 0)
            {
                return static_cast<uint32_t>(std::min<double>(*seconds * 1000.0, RETRY_AFTER_CAP_MS));
            }
            return std::nullopt;
        }
    } // namespace

    AnthropicProvider::AnthropicProvider(AnthropicOptions options) : m_Options(std::move(options))
    {
    }

    std::string AnthropicProvider::Name() const
    {
        return "anthropic:" + m_Options.m_Model;
    }

    bool AnthropicProvider::Available() const
    {
        return !ApiKey(m_Options.m_ApiKeyEnv).empty();
    }

    const AnthropicOptions &AnthropicProvider::Options() const noexcept
    {
        return m_Options;
    }

    Json AnthropicProvider::RequestBody(const Prompt &prompt) const
    {
        Json body = Json::MakeObject();
        body.Set("model", m_Options.m_Model);
        body.Set("max_tokens", prompt.m_MaxTokens > 0 ? prompt.m_MaxTokens : m_Options.m_MaxTokens);
        if (!prompt.m_System.empty())
        {
            body.Set("system", prompt.m_System);
        }
        body.Set("messages", Json::MakeArray({Json::MakeObject({{"role", "user"}, {"content", prompt.m_User}})}));
        // thinking is adaptive on the model; depth is steered by effort only (no budgets, no sampling knobs)
        Json output = Json::MakeObject();
        std::string effort = prompt.m_Effort.empty() ? m_Options.m_Effort : prompt.m_Effort;
        if (!effort.empty())
        {
            output.Set("effort", effort);
        }
        if (prompt.m_Schema)
        {
            output.Set("format", Json::MakeObject({{"type", "json_schema"}, {"schema", *prompt.m_Schema}}));
        }
        if (output.Size() > 0)
        {
            body.Set("output_config", std::move(output));
        }
        if (m_Options.m_Fallbacks)
        {
            body.Set("fallbacks", "default");
        }
        return body;
    }

    Completion AnthropicProvider::ParseResponse(const std::string &body, bool structured) const
    {
        Completion completion;
        completion.m_Provider = Name();
        completion.m_Model = m_Options.m_Model;
        Json message;
        try
        {
            message = Json::Parse(body);
        }
        catch (const JsonError &error)
        {
            completion.m_Error = std::string("malformed response: ") + error.what();
            return completion;
        }
        if (!message.IsObject())
        {
            completion.m_Error = "malformed response: expected a message object";
            return completion;
        }
        if (const Json *error = message.Find("error"); error && error->IsObject())
        {
            completion.m_Error = "API error " + error->GetString("type") + ": " + Bounded(error->GetString("message"));
            return completion;
        }
        completion.m_Model = message.GetString("model", m_Options.m_Model);
        completion.m_StopReason = message.GetString("stop_reason");
        const Json &usage = message["usage"];
        completion.m_InputTokens = Tokens(usage, "input_tokens");
        completion.m_OutputTokens = Tokens(usage, "output_tokens");

        // the stop reason decides before any content is read
        if (completion.m_StopReason == "refusal")
        {
            const Json *category = message.At("stop_details.category");
            completion.m_Error =
                "refused by the model (category: " +
                (category && category->IsString() ? category->AsString() : std::string("unspecified")) + ")";
            return completion;
        }
        if (completion.m_StopReason == "max_tokens")
        {
            completion.m_Error = "truncated: the response reached max_tokens";
            return completion;
        }
        if (completion.m_StopReason == "model_context_window_exceeded")
        {
            completion.m_Error = "truncated: the context window was exceeded";
            return completion;
        }
        const Json *content = message.Find("content");
        if (!content || !content->IsArray())
        {
            completion.m_Error = "malformed response: no content list";
            return completion;
        }
        for (const Json &block : content->AsArray())
        {
            // thinking, fallback and other block types carry no answer text
            if (block.GetString("type") == "text")
            {
                completion.m_Text += block.GetString("text");
            }
        }
        if (structured)
        {
            completion.m_Json = ExtractJson(completion.m_Text);
            if (!completion.m_Json)
            {
                completion.m_Error = "structured output is not valid JSON";
                return completion;
            }
        }
        else if (completion.m_Text.empty())
        {
            completion.m_Error = "the response holds no text";
            return completion;
        }
        completion.m_Ok = true;
        return completion;
    }

    Completion AnthropicProvider::Complete(const Prompt &prompt)
    {
        Completion failure;
        failure.m_Provider = Name();
        failure.m_Model = m_Options.m_Model;
        std::string key = ApiKey(m_Options.m_ApiKeyEnv);
        if (key.empty())
        {
            failure.m_Error = "unavailable: environment variable " + m_Options.m_ApiKeyEnv + " is not set";
            return failure;
        }
        if (prompt.m_User.empty())
        {
            failure.m_Error = "the prompt has no user content";
            return failure;
        }

        std::string base = m_Options.m_BaseUrl;
        while (!base.empty() && base.back() == '/')
        {
            base.pop_back();
        }
        HttpRequest request;
        request.m_Method = "POST";
        request.m_Url = base + "/v1/messages";
        try
        {
            // the API key travels in a header: plaintext http only to a loopback endpoint (local proxies, tests)
            Url url = ParseUrl(request.m_Url);
            if (url.m_Scheme != "https" && !IsLoopbackHost(url.m_Host))
            {
                failure.m_Error = "refusing to send the API key over plain http to " + url.m_Host + " (use https)";
                return failure;
            }
        }
        catch (const HttpError &error)
        {
            failure.m_Error = std::string("invalid base_url: ") + error.what();
            return failure;
        }
        request.m_Headers = {
            {"content-type", "application/json"}, {"x-api-key", key}, {"anthropic-version", API_VERSION}};
        if (m_Options.m_Fallbacks)
        {
            request.m_Headers.emplace_back("anthropic-beta", FALLBACK_BETA);
        }
        request.m_Body = RequestBody(prompt).Dump();
        request.m_TimeoutMs = m_Options.m_TimeoutMs;
        request.m_CaFile = m_Options.m_CaFile;

        std::string last;
        uint32_t attempts = m_Options.m_MaxRetries + 1;
        uint32_t used = 0;
        for (uint32_t attempt = 0; attempt < attempts; ++attempt)
        {
            ++used;
            uint32_t backoff = std::min<uint32_t>(BACKOFF_CAP_MS, BACKOFF_BASE_MS << std::min<uint32_t>(attempt, 4));
            bool retry = false;
            try
            {
                HttpResponse response = SendHttp(request);
                if (response.m_Status == 200)
                {
                    return ParseResponse(response.m_Body, prompt.m_Schema.has_value());
                }
                last = DescribeFailure(response);
                std::string hint = Lower(Trim(response.Header("x-should-retry")));
                retry = hint == "true" || (hint != "false" && RetryableStatus(response.m_Status));
                if (std::optional<uint32_t> after = RetryAfter(response))
                {
                    backoff = *after;
                }
            }
            catch (const HttpError &error)
            {
                // transport failures never carry header values (eOps::HttpError contract)
                last = std::string("transport: ") + error.what();
                retry = true;
            }
            if (!retry || attempt + 1 >= attempts)
            {
                break;
            }
            EMPY_DEBUG("anthropic: attempt {} failed ({}), retrying in {} ms", attempt + 1, last, backoff);
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
        }
        failure.m_Error = used > 1 ? last + " (after " + std::to_string(used) + " attempts)" : last;
        return failure;
    }
} // namespace eOps::Intelligence
