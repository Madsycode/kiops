#include <eOps/Intelligence/Provider.h>
#include <eOps/Intelligence/Anthropic.h>
#include <eOps/Intelligence/Command.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Http.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <algorithm>

namespace eOps::Intelligence
{
    namespace
    {
        constexpr std::array<Capability, 3> CAPABILITIES = {Capability::Reasoning, Capability::Planning,
                                                            Capability::Generation};

        size_t IndexOf(Capability capability)
        {
            size_t index = static_cast<size_t>(capability);
            if (index >= CAPABILITIES.size())
            {
                throw std::invalid_argument("unknown intelligence capability");
            }
            return index;
        }

        std::shared_ptr<Provider> MakeAnthropic(const eCore::YamlNode &node, const std::string &configPath)
        {
            AnthropicOptions options;
            options.m_Model = node.Get<std::string>("model", options.m_Model);
            options.m_ApiKeyEnv = node.Get<std::string>("api_key_env", options.m_ApiKeyEnv);
            options.m_BaseUrl = node.Get<std::string>("base_url", options.m_BaseUrl);
            options.m_Effort = node.Get<std::string>("effort", options.m_Effort);
            options.m_MaxTokens = node.Get<uint32_t>("max_tokens", options.m_MaxTokens, 1, 128000);
            options.m_TimeoutMs = node.Get<uint32_t>("timeout_ms", options.m_TimeoutMs, 1000, 3600000);
            options.m_MaxRetries = node.Get<uint32_t>("max_retries", options.m_MaxRetries, 0, 10);
            options.m_Fallbacks = node.Get<bool>("fallbacks", options.m_Fallbacks);
            options.m_CaFile = eCore::ConfigPath(configPath.c_str(), node.Get<std::string>("ca_file", ""));
            if (Trim(options.m_Model).empty())
            {
                throw eCore::ConfigError(node.Path() + ".model: must not be empty");
            }
            if (Trim(options.m_ApiKeyEnv).empty())
            {
                throw eCore::ConfigError(node.Path() + ".api_key_env: must name an environment variable");
            }
            static const std::vector<std::string> s_Efforts = {"", "low", "medium", "high", "xhigh", "max"};
            if (std::find(s_Efforts.begin(), s_Efforts.end(), options.m_Effort) == s_Efforts.end())
            {
                throw eCore::ConfigError(node.Path() + ".effort: expected low, medium, high, xhigh or max");
            }
            Url base;
            try
            {
                base = ParseUrl(options.m_BaseUrl);
            }
            catch (const HttpError &error)
            {
                throw eCore::ConfigError(node.Path() + ".base_url: " + error.what());
            }
            if (base.m_Scheme != "https" && !IsLoopbackHost(base.m_Host))
            {
                throw eCore::ConfigError(node.Path() + ".base_url: the API key is only sent over https (plain http is "
                                                       "accepted for loopback hosts only)");
            }
            return std::make_shared<AnthropicProvider>(std::move(options));
        }

        std::shared_ptr<Provider> MakeCommand(const eCore::YamlNode &node, const std::string &configPath)
        {
            CommandOptions options;
            for (const eCore::YamlNode &argument : node["command"].Items())
            {
                options.m_Command.push_back(argument.As<std::string>());
            }
            if (options.m_Command.empty() || options.m_Command.front().empty())
            {
                throw eCore::ConfigError(node.Path() + ".command: a command provider needs a program");
            }
            // a relative program path (not a bare name searched on PATH) is relative to the configuration
            std::string &program = options.m_Command.front();
            if (program.find_first_of("/\\") != std::string::npos && std::filesystem::path(program).is_relative())
            {
                program = eCore::ConfigPath(configPath.c_str(), program);
            }
            options.m_TimeoutMs = node.Get<uint32_t>("timeout_ms", options.m_TimeoutMs, 100, 3600000);
            options.m_Name = node.Get<std::string>("name", "");
            options.m_MaxOutputBytes =
                node.Get<size_t>("max_output_bytes", options.m_MaxOutputBytes, 1024, size_t(256) << 20);
            return std::make_shared<CommandProvider>(std::move(options));
        }

        std::shared_ptr<Provider> MakeProvider(const eCore::YamlNode &node, const std::string &configPath)
        {
            if (node.IsNull())
            {
                return std::make_shared<DeterministicProvider>();
            }
            if (!node.IsMap())
            {
                throw eCore::ConfigError(node.Path() + ": expected a mapping");
            }
            static const std::vector<std::string> s_Keys = {
                "provider",    "model",     "api_key_env", "base_url", "effort", "max_tokens",      "timeout_ms",
                "max_retries", "fallbacks", "ca_file",     "command",  "name",   "max_output_bytes"};
            for (const std::string &key : node.Keys())
            {
                if (std::find(s_Keys.begin(), s_Keys.end(), key) == s_Keys.end())
                {
                    throw eCore::ConfigError(node.Path() + "." + key + ": unknown intelligence provider option");
                }
            }
            std::string provider = Lower(Trim(node.Get<std::string>("provider", "deterministic")));
            if (provider == "deterministic")
            {
                return std::make_shared<DeterministicProvider>();
            }
            if (provider == "anthropic")
            {
                return MakeAnthropic(node, configPath);
            }
            if (provider == "command")
            {
                return MakeCommand(node, configPath);
            }
            throw eCore::ConfigError(node.Path() + ".provider: expected anthropic, command or deterministic, got '" +
                                     provider + "'");
        }
    } // namespace

    const char *CapabilityName(Capability capability)
    {
        switch (capability)
        {
        case Capability::Reasoning:
            return "reasoning";
        case Capability::Planning:
            return "planning";
        case Capability::Generation:
            return "generation";
        }
        return "unknown";
    }

    std::optional<Capability> ParseCapability(std::string_view name)
    {
        std::string lowered = Lower(Trim(name));
        for (Capability capability : CAPABILITIES)
        {
            if (lowered == CapabilityName(capability))
            {
                return capability;
            }
        }
        return std::nullopt;
    }

    Provider::~Provider() = default;

    std::string DeterministicProvider::Name() const
    {
        return "deterministic";
    }

    bool DeterministicProvider::Available() const
    {
        return false;
    }

    Completion DeterministicProvider::Complete(const Prompt &)
    {
        Completion completion;
        completion.m_Provider = Name();
        completion.m_Error = "deterministic provider: agents use their built-in ontology and rule logic";
        return completion;
    }

    void ProviderSet::Set(Capability capability, std::shared_ptr<Provider> provider)
    {
        m_Providers[IndexOf(capability)] = std::move(provider);
    }

    Provider *ProviderSet::Get(Capability capability) const
    {
        const std::shared_ptr<Provider> &provider = m_Providers[IndexOf(capability)];
        return provider && provider->Available() ? provider.get() : nullptr;
    }

    Json ProviderSet::Describe() const
    {
        Json description = Json::MakeObject();
        for (Capability capability : CAPABILITIES)
        {
            const std::shared_ptr<Provider> &provider = m_Providers[IndexOf(capability)];
            description.Set(CapabilityName(capability),
                            Json::MakeObject({{"provider", provider ? provider->Name() : std::string("deterministic")},
                                              {"available", provider ? provider->Available() : false}}));
        }
        return description;
    }

    ProviderSet ProviderSet::FromYaml(const eCore::YamlNode &intelligence, const std::string &configPath)
    {
        ProviderSet set;
        for (Capability capability : CAPABILITIES)
        {
            set.Set(capability, std::make_shared<DeterministicProvider>());
        }
        if (intelligence.IsNull())
        {
            return set;
        }
        if (!intelligence.IsMap())
        {
            throw eCore::ConfigError(intelligence.Path() + ": expected a mapping of capabilities");
        }
        for (const std::string &key : intelligence.Keys())
        {
            std::optional<Capability> capability = ParseCapability(key);
            if (!capability)
            {
                throw eCore::ConfigError(intelligence.Path() + "." + key +
                                         ": unknown capability (expected reasoning, planning or generation)");
            }
            set.Set(*capability, MakeProvider(intelligence[key], configPath));
        }
        return set;
    }

    std::optional<Json> AskJson(Provider *provider, Capability capability, std::string system, std::string user,
                                const Json &schema, std::string *error)
    {
        auto fail = [&](std::string message) -> std::optional<Json>
        {
            if (error)
            {
                *error = std::move(message);
            }
            return std::nullopt;
        };
        if (!provider)
        {
            return fail("no intelligence provider for " + std::string(CapabilityName(capability)));
        }
        if (!provider->Available())
        {
            return fail(provider->Name() + " is not available");
        }
        Prompt prompt;
        prompt.m_Capability = capability;
        prompt.m_System = std::move(system);
        prompt.m_User = std::move(user);
        prompt.m_Schema = schema;
        Completion completion;
        try
        {
            completion = provider->Complete(prompt);
        }
        catch (const std::exception &exception)
        {
            return fail(provider->Name() + " failed: " + exception.what());
        }
        if (!completion.m_Ok)
        {
            return fail(provider->Name() + ": " + completion.m_Error);
        }
        std::optional<Json> value = completion.m_Json ? completion.m_Json : ExtractJson(completion.m_Text);
        if (!value)
        {
            return fail(provider->Name() + ": the completion holds no JSON document");
        }
        std::string violation;
        if (!Conforms(*value, schema, &violation))
        {
            return fail(provider->Name() + ": the completion does not conform to the schema: " + violation);
        }
        return value;
    }
} // namespace eOps::Intelligence
