#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <filesystem>
#include "Globals.h"
#include "../Vendors/Yaml.h"

// typed YAML configuration access: every lookup carries its dotted path so a rejected
// configuration names the offending key, and numeric values are range-checked on read

namespace eCore
{
    inline std::string ConfigPath(const char *configuration, std::string value)
    {
        if (value.empty())
        {
            return {};
        }
        std::filesystem::path path(value);
        if (path.is_relative())
        {
            path = std::filesystem::path(configuration).parent_path() / path;
        }
        return std::filesystem::absolute(path).lexically_normal().string();
    }
    // raised for unreadable, malformed, missing or out-of-range configuration values
    struct ConfigError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    class YamlNode
    {
    public:
        EMPY_INLINE YamlNode() = default;
        EMPY_INLINE YamlNode(fkyaml::node node, std::string path) : m_Node(std::move(node)), m_Path(std::move(path))
        {
        }

        // parse a document from text
        static YamlNode Parse(std::string_view text, std::string origin = "<config>")
        {
            try
            {
                return YamlNode(fkyaml::node::deserialize(std::string(text)), std::move(origin));
            }
            catch (const std::exception &error)
            {
                throw ConfigError(origin + ": invalid YAML: " + error.what());
            }
        }

        // parse a document from a file
        static YamlNode Load(const std::string &path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                throw ConfigError("cannot open configuration file '" + path + "'");
            }
            std::ostringstream text;
            text << file.rdbuf();
            return Parse(text.str(), path);
        }

        EMPY_INLINE bool IsNull() const
        {
            return m_Node.is_null();
        }
        EMPY_INLINE bool IsMap() const
        {
            return m_Node.is_mapping();
        }
        EMPY_INLINE bool IsList() const
        {
            return m_Node.is_sequence();
        }
        EMPY_INLINE bool IsScalar() const
        {
            return m_Node.is_scalar();
        }
        EMPY_INLINE const std::string &Path() const noexcept
        {
            return m_Path;
        }

        EMPY_INLINE bool Has(std::string_view key) const
        {
            return m_Node.is_mapping() && m_Node.contains(std::string(key)) && !m_Node[std::string(key)].is_null();
        }

        // child of a mapping; a missing key yields a null node carrying the would-be path
        EMPY_INLINE YamlNode operator[](std::string_view key) const
        {
            std::string path = m_Path + "." + std::string(key);
            if (!m_Node.is_mapping() || !m_Node.contains(std::string(key)))
            {
                return YamlNode(fkyaml::node(), path);
            }
            return YamlNode(m_Node[std::string(key)], path);
        }

        // elements of a sequence (empty for null)
        EMPY_INLINE std::vector<YamlNode> Items() const
        {
            std::vector<YamlNode> items;
            if (m_Node.is_null())
            {
                return items;
            }
            if (!m_Node.is_sequence())
            {
                throw ConfigError(m_Path + ": expected a list");
            }
            size_t index = 0;
            for (const fkyaml::node &item : m_Node.as_seq())
            {
                items.emplace_back(item, m_Path + "[" + std::to_string(index++) + "]");
            }
            return items;
        }

        // keys of a mapping in document order (empty for null)
        EMPY_INLINE std::vector<std::string> Keys() const
        {
            std::vector<std::string> keys;
            if (m_Node.is_null())
            {
                return keys;
            }
            if (!m_Node.is_mapping())
            {
                throw ConfigError(m_Path + ": expected a mapping");
            }
            for (const auto &entry : m_Node.as_map())
            {
                keys.push_back(entry.first.get_value<std::string>());
            }
            return keys;
        }

        // required scalar conversion
        template <typename _Type> EMPY_INLINE _Type As() const
        {
            if (m_Node.is_null())
            {
                throw ConfigError(m_Path + ": required value is missing");
            }
            return Convert<_Type>();
        }

        // optional scalar with default
        template <typename _Type> EMPY_INLINE _Type Get(std::string_view key, _Type fallback) const
        {
            YamlNode child = (*this)[key];
            return child.IsNull() ? fallback : child.template Convert<_Type>();
        }

        // optional integral/floating scalar with default and inclusive range check
        template <typename _Type>
        EMPY_INLINE _Type Get(std::string_view key, _Type fallback, _Type minimum, _Type maximum) const
        {
            _Type value = Get<_Type>(key, fallback);
            if (value < minimum || value > maximum)
            {
                throw ConfigError(m_Path + "." + std::string(key) + ": value " + std::to_string(value) + " outside [" +
                                  std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
            }
            return value;
        }

        // required scalar from a mapping key
        template <typename _Type> EMPY_INLINE _Type Require(std::string_view key) const
        {
            return (*this)[key].template As<_Type>();
        }

    private:
        template <typename _Type> EMPY_INLINE _Type Convert() const
        {
            try
            {
                if constexpr (std::is_same_v<_Type, std::string>)
                {
                    if (m_Node.is_string())
                    {
                        return m_Node.get_value<std::string>();
                    }
                    if (m_Node.is_integer())
                    {
                        return std::to_string(m_Node.get_value<int64_t>());
                    }
                    if (m_Node.is_boolean())
                    {
                        return m_Node.get_value<bool>() ? "true" : "false";
                    }
                    if (m_Node.is_float_number())
                    {
                        return std::to_string(m_Node.get_value<double>());
                    }
                    throw ConfigError(m_Path + ": expected a string");
                }
                else if constexpr (std::is_same_v<_Type, bool>)
                {
                    if (!m_Node.is_boolean())
                    {
                        throw ConfigError(m_Path + ": expected true or false");
                    }
                    return m_Node.get_value<bool>();
                }
                else if constexpr (std::is_integral_v<_Type>)
                {
                    if (!m_Node.is_integer())
                    {
                        throw ConfigError(m_Path + ": expected an integer");
                    }
                    int64_t value = m_Node.get_value<int64_t>();
                    if constexpr (std::is_unsigned_v<_Type>)
                    {
                        if (value < 0 ||
                            static_cast<uint64_t>(value) > static_cast<uint64_t>(std::numeric_limits<_Type>::max()))
                        {
                            throw ConfigError(m_Path + ": integer out of range");
                        }
                    }
                    else if (value < static_cast<int64_t>(std::numeric_limits<_Type>::min()) ||
                             value > static_cast<int64_t>(std::numeric_limits<_Type>::max()))
                    {
                        throw ConfigError(m_Path + ": integer out of range");
                    }
                    return static_cast<_Type>(value);
                }
                else if constexpr (std::is_floating_point_v<_Type>)
                {
                    if (m_Node.is_integer())
                    {
                        return static_cast<_Type>(m_Node.get_value<int64_t>());
                    }
                    if (!m_Node.is_float_number())
                    {
                        throw ConfigError(m_Path + ": expected a number");
                    }
                    return static_cast<_Type>(m_Node.get_value<double>());
                }
                else
                {
                    EMPY_STATIC(sizeof(_Type) == 0, "unsupported YAML conversion");
                }
            }
            catch (const ConfigError &)
            {
                throw;
            }
            catch (const std::exception &error)
            {
                throw ConfigError(m_Path + ": " + error.what());
            }
        }

        fkyaml::node m_Node;
        std::string m_Path;
    };
} // namespace eCore
