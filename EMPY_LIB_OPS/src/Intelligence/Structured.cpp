#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Text.h>
#include <cmath>
#include <stdexcept>

namespace eOps::Intelligence
{
    namespace
    {
        Json Described(Json schema, std::string_view description)
        {
            if (!description.empty())
            {
                schema.Set("description", std::string(description));
            }
            return schema;
        }

        bool Fail(std::string *error, const std::string &path, std::string message)
        {
            if (error)
            {
                *error = (path.empty() ? std::string("<root>") : path) + ": " + std::move(message);
            }
            return false;
        }

        std::string Child(const std::string &path, std::string_view member)
        {
            return path.empty() ? std::string(member) : path + "." + std::string(member);
        }

        bool IsIntegral(const Json &value)
        {
            if (!value.IsNumber())
            {
                return false;
            }
            if (value.IsInteger())
            {
                return true;
            }
            double number = value.AsNumber();
            return std::isfinite(number) && std::floor(number) == number;
        }

        // nullopt for an unsupported type name
        std::optional<bool> TypeMatches(const Json &value, std::string_view type)
        {
            if (type == "null")
            {
                return value.IsNull();
            }
            if (type == "boolean")
            {
                return value.IsBool();
            }
            if (type == "string")
            {
                return value.IsString();
            }
            if (type == "number")
            {
                return value.IsNumber();
            }
            if (type == "integer")
            {
                return IsIntegral(value);
            }
            if (type == "object")
            {
                return value.IsObject();
            }
            if (type == "array")
            {
                return value.IsArray();
            }
            return std::nullopt;
        }

        bool Check(const Json &value, const Json &schema, const std::string &path, std::string *error)
        {
            if (schema.IsBool())
            {
                return schema.AsBool() || Fail(error, path, "no value is allowed here");
            }
            if (!schema.IsObject())
            {
                return Fail(error, path, "the schema is not an object");
            }
            if (const Json *anyOf = schema.Find("anyOf"))
            {
                if (!anyOf->IsArray() || anyOf->Size() == 0)
                {
                    return Fail(error, path, "'anyOf' must be a non-empty list");
                }
                bool matched = false;
                std::string first;
                for (const Json &option : anyOf->AsArray())
                {
                    std::string reason;
                    if (Check(value, option, path, &reason))
                    {
                        matched = true;
                        break;
                    }
                    if (first.empty())
                    {
                        first = reason;
                    }
                }
                if (!matched)
                {
                    return Fail(error, path, "matches none of the 'anyOf' alternatives (" + first + ")");
                }
            }
            if (const Json *type = schema.Find("type"))
            {
                std::vector<std::string> types;
                if (type->IsString())
                {
                    types.push_back(type->AsString());
                }
                else if (type->IsArray())
                {
                    for (const Json &item : type->AsArray())
                    {
                        if (!item.IsString())
                        {
                            return Fail(error, path, "'type' entries must be strings");
                        }
                        types.push_back(item.AsString());
                    }
                }
                else
                {
                    return Fail(error, path, "'type' must be a string or a list");
                }
                bool matched = false;
                for (const std::string &name : types)
                {
                    std::optional<bool> result = TypeMatches(value, name);
                    if (!result)
                    {
                        return Fail(error, path, "unsupported schema type '" + name + "'");
                    }
                    matched = matched || *result;
                }
                if (!matched)
                {
                    return Fail(error, path, "expected " + Join(types, " or "));
                }
            }
            if (const Json *values = schema.Find("enum"))
            {
                bool matched = false;
                if (values->IsArray())
                {
                    for (const Json &option : values->AsArray())
                    {
                        matched = matched || option == value;
                    }
                }
                if (!matched)
                {
                    return Fail(error, path, "value is not one of the allowed 'enum' values");
                }
            }
            if (const Json *constant = schema.Find("const"); constant && !(*constant == value))
            {
                return Fail(error, path, "value differs from 'const'");
            }
            if (value.IsObject())
            {
                const Json *properties = schema.Find("properties");
                if (const Json *required = schema.Find("required"); required && required->IsArray())
                {
                    for (const Json &key : required->AsArray())
                    {
                        if (key.IsString() && !value.Has(key.AsString()))
                        {
                            return Fail(error, path, "missing required member '" + key.AsString() + "'");
                        }
                    }
                }
                const Json *additional = schema.Find("additionalProperties");
                for (const auto &[key, member] : value.AsObject())
                {
                    const Json *property = properties && properties->IsObject() ? properties->Find(key) : nullptr;
                    if (property)
                    {
                        if (!Check(member, *property, Child(path, key), error))
                        {
                            return false;
                        }
                        continue;
                    }
                    if (additional && additional->IsBool() && !additional->AsBool())
                    {
                        return Fail(error, path, "unexpected member '" + key + "'");
                    }
                    if (additional && additional->IsObject() && !Check(member, *additional, Child(path, key), error))
                    {
                        return false;
                    }
                }
            }
            if (value.IsArray())
            {
                if (const Json *items = schema.Find("items"))
                {
                    size_t index = 0;
                    for (const Json &item : value.AsArray())
                    {
                        if (!Check(item, *items, Child(path, std::to_string(index++)), error))
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }

        std::optional<Json> TryParse(std::string_view text)
        {
            std::string trimmed = Trim(text);
            if (trimmed.empty())
            {
                return std::nullopt;
            }
            try
            {
                return Json::Parse(trimmed);
            }
            catch (const JsonError &)
            {
                return std::nullopt;
            }
        }
    } // namespace

    Json SchemaScalar(std::string_view kind, std::string_view description)
    {
        if (kind != "string" && kind != "number" && kind != "integer" && kind != "boolean")
        {
            throw std::invalid_argument("SchemaScalar: unsupported kind '" + std::string(kind) + "'");
        }
        return Described(Json::MakeObject({{"type", std::string(kind)}}), description);
    }

    Json SchemaEnum(const std::vector<std::string> &values, std::string_view description)
    {
        Json options = Json::MakeArray();
        for (const std::string &value : values)
        {
            options.Push(value);
        }
        return Described(Json::MakeObject({{"type", "string"}, {"enum", std::move(options)}}), description);
    }

    Json SchemaArray(Json items, std::string_view description)
    {
        return Described(Json::MakeObject({{"type", "array"}, {"items", std::move(items)}}), description);
    }

    Json SchemaObject(std::vector<std::pair<std::string, Json>> properties, std::string_view description)
    {
        Json members = Json::MakeObject();
        Json required = Json::MakeArray();
        for (auto &[name, schema] : properties)
        {
            required.Push(name);
            members.Set(std::move(name), std::move(schema));
        }
        return Described(Json::MakeObject({{"type", "object"},
                                           {"properties", std::move(members)},
                                           {"required", std::move(required)},
                                           {"additionalProperties", false}}),
                         description);
    }

    Json SchemaNullable(Json schema)
    {
        return Json::MakeObject(
            {{"anyOf", Json::MakeArray({std::move(schema), Json::MakeObject({{"type", "null"}})})}});
    }

    bool Conforms(const Json &value, const Json &schema, std::string *error)
    {
        return Check(value, schema, {}, error);
    }

    std::optional<Json> ExtractJson(std::string_view text)
    {
        if (std::optional<Json> whole = TryParse(text))
        {
            return whole;
        }
        // fenced blocks: ```json ... ``` (or a bare ``` fence)
        size_t cursor = 0;
        while ((cursor = text.find("```", cursor)) != std::string_view::npos)
        {
            size_t body = text.find('\n', cursor + 3);
            if (body == std::string_view::npos)
            {
                break;
            }
            size_t close = text.find("```", body + 1);
            if (close == std::string_view::npos)
            {
                break;
            }
            if (std::optional<Json> fenced = TryParse(text.substr(body + 1, close - body - 1)))
            {
                return fenced;
            }
            cursor = close + 3;
        }
        size_t open = text.find('{');
        size_t end = text.rfind('}');
        if (open != std::string_view::npos && end != std::string_view::npos && end > open)
        {
            return TryParse(text.substr(open, end - open + 1));
        }
        return std::nullopt;
    }
} // namespace eOps::Intelligence
