#pragma once
#include <array>
#include <memory>
#include <string>
#include <cstdint>
#include <optional>
#include "../Common/Json.h"

// Intelligence Service Providers (thesis Fig. 23): contextual reasoning | planning | generation
// components, remote or local. LLM output is advisory, never authoritative: every caller has a
// deterministic fallback and validates what a provider returns before using it.

namespace eOps::Intelligence
{
    enum class Capability : uint8_t
    {
        Reasoning,
        Planning,
        Generation
    };

    EMPY_OPS_API const char *CapabilityName(Capability capability); // "reasoning" | "planning" | "generation"
    EMPY_OPS_API std::optional<Capability> ParseCapability(std::string_view name);

    struct Prompt
    {
        Capability m_Capability = Capability::Reasoning;
        std::string m_System;
        std::string m_User;
        std::optional<Json> m_Schema; // JSON schema for structured output (see Structured.h)
        uint32_t m_MaxTokens = 16000;
        std::string m_Effort; // "" → provider default
    };

    struct Completion
    {
        bool m_Ok = false;
        std::string m_Text;
        std::optional<Json> m_Json; // parsed structured output when a schema was requested
        std::string m_Error;
        std::string m_Provider;
        std::string m_Model;
        std::string m_StopReason;
        uint32_t m_InputTokens = 0;
        uint32_t m_OutputTokens = 0;
    };

    struct EMPY_OPS_API Provider
    {
        virtual ~Provider();
        virtual std::string Name() const = 0;
        virtual bool Available() const = 0;
        // blocking; called from worker threads; implementations must be thread-safe
        virtual Completion Complete(const Prompt &prompt) = 0;
    };

    // placeholder for "deterministic": never available, so agents use their ontology/rule logic
    class EMPY_OPS_API DeterministicProvider : public Provider
    {
    public:
        std::string Name() const override;                  // "deterministic"
        bool Available() const override;                    // false
        Completion Complete(const Prompt &prompt) override; // m_Ok = false
    };

    // capability → provider routing
    class EMPY_OPS_API ProviderSet
    {
    public:
        void Set(Capability capability, std::shared_ptr<Provider> provider);
        // nullptr when unset or !Available() → caller uses deterministic logic
        Provider *Get(Capability capability) const;
        // {reasoning: {provider, available}, planning: {...}, generation: {...}}
        Json Describe() const;
        // `intelligence: {reasoning: {provider: anthropic|command|deterministic, model, api_key_env, base_url, effort,
        // max_tokens, timeout_ms, max_retries, fallbacks, ca_file, command: [..], name}, planning: {...},
        // generation: {...}}`; a missing capability → deterministic; paths relative to `configPath`;
        // throws eCore::ConfigError
        static ProviderSet FromYaml(const eCore::YamlNode &intelligence, const std::string &configPath);

    private:
        std::array<std::shared_ptr<Provider>, 3> m_Providers;
    };

    // one structured-output exchange: builds a Prompt with `schema`, completes it and returns the
    // JSON only when it parses and conforms to the schema (Structured.h Conforms); nullopt on any
    // failure, including a null or unavailable provider, with `error` describing why
    EMPY_OPS_API std::optional<Json> AskJson(Provider *provider, Capability capability, std::string system,
                                             std::string user, const Json &schema, std::string *error = nullptr);
} // namespace eOps::Intelligence
