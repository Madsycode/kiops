#include <eOps/Substrate/Substrate.h>
#include <eCore/Common/YAML.h>
#include <algorithm>
#include <stdexcept>

namespace eOps::Substrate
{
    namespace
    {
        constexpr std::string_view KEYS[] = {
            "type",           "output_directory",   "namespace",  "image",           "apply_command",
            "delete_command", "command_timeout_ms", "executable", "state_directory", "log_directory",
            "client_config",  "environment"};

        std::vector<std::string> Command(const eCore::YamlNode &node, std::string_view key)
        {
            std::vector<std::string> command;
            for (const eCore::YamlNode &item : node[key].Items())
            {
                command.push_back(item.As<std::string>());
                if (command.back().empty())
                {
                    throw eCore::ConfigError(item.Path() + ": empty command argument");
                }
            }
            return command;
        }

        std::filesystem::path Path(const eCore::YamlNode &node, std::string_view key, const std::string &configPath)
        {
            return eCore::ConfigPath(configPath.c_str(), node.Get<std::string>(key, ""));
        }
    } // namespace

    std::shared_ptr<ExecutionSubstrate> MakeSubstrate(const eCore::YamlNode &substrate, const std::string &configPath)
    {
        if (substrate.IsNull())
        {
            return std::make_shared<SimulatedSubstrate>();
        }
        if (!substrate.IsMap())
        {
            throw eCore::ConfigError(substrate.Path() + ": expected a mapping");
        }
        for (const std::string &key : substrate.Keys())
        {
            if (std::find(std::begin(KEYS), std::end(KEYS), key) == std::end(KEYS))
            {
                throw eCore::ConfigError(substrate.Path() + "." + key + ": unknown substrate setting");
            }
        }
        std::string type = substrate.Get<std::string>("type", "simulated");
        try
        {
            if (type == "simulated")
            {
                return std::make_shared<SimulatedSubstrate>();
            }
            if (type == "kubernetes")
            {
                KubernetesOptions options;
                if (!substrate.Has("output_directory"))
                {
                    throw eCore::ConfigError(substrate.Path() + ".output_directory: required for kubernetes");
                }
                options.m_OutputDirectory = Path(substrate, "output_directory", configPath);
                options.m_Namespace = substrate.Get<std::string>("namespace", options.m_Namespace);
                options.m_Image = substrate.Get<std::string>("image", options.m_Image);
                options.m_ApplyCommand = Command(substrate, "apply_command");
                options.m_DeleteCommand = Command(substrate, "delete_command");
                options.m_CommandTimeoutMs =
                    substrate.Get<uint32_t>("command_timeout_ms", options.m_CommandTimeoutMs, 1, 3600000);
                return std::make_shared<KubernetesSubstrate>(std::move(options));
            }
            if (type == "process")
            {
                ProcessSubstrateOptions options;
                options.m_Executable = Path(substrate, "executable", configPath);
                options.m_StateDirectory = Path(substrate, "state_directory", configPath);
                options.m_LogDirectory = Path(substrate, "log_directory", configPath);
                options.m_ClientConfig = Path(substrate, "client_config", configPath);
                eCore::YamlNode environment = substrate["environment"];
                for (const std::string &name : environment.Keys())
                {
                    options.m_Environment.emplace_back(name, environment[name].As<std::string>());
                }
                return std::make_shared<ProcessSubstrate>(std::move(options));
            }
        }
        catch (const std::invalid_argument &error)
        {
            throw eCore::ConfigError(substrate.Path() + ": " + error.what());
        }
        throw eCore::ConfigError(substrate.Path() + ".type: unknown substrate '" + type +
                                 "' (simulated | kubernetes | process)");
    }
} // namespace eOps::Substrate
