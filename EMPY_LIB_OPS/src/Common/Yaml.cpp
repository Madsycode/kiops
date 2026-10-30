#include "Yaml.h"
#include <eOps/Common/Text.h>
#include <array>

namespace eOps::Detail
{
    namespace
    {
        constexpr size_t MAX_DEPTH = 64;

        bool Letter(char c) noexcept
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        bool Digit(char c) noexcept
        {
            return c >= '0' && c <= '9';
        }

        bool Scalar(const Json &value) noexcept
        {
            return !value.IsArray() && !value.IsObject();
        }

        // literal block style keeps the text readable; anything it cannot represent exactly is quoted
        bool LiteralBlock(std::string_view text) noexcept
        {
            if (text.find('\n') == std::string_view::npos || text.empty() || text.front() == ' ' ||
                text.front() == '\t' || text.front() == '\n')
            {
                return false;
            }
            size_t newlines = 0;
            for (size_t i = text.size(); i > 0 && text[i - 1] == '\n'; --i)
            {
                ++newlines;
            }
            if (newlines > 1)
            {
                return false;
            }
            for (size_t i = 0; i < text.size(); ++i)
            {
                unsigned char c = static_cast<unsigned char>(text[i]);
                if ((c < 0x20 && c != '\n') || c == 0x7f)
                {
                    return false;
                }
                // trailing blanks (also blank-only lines) are invisible and interact with indentation detection
                if (c == '\n' && i > 0 && (text[i - 1] == ' ' || text[i - 1] == '\t'))
                {
                    return false;
                }
            }
            // the first line fixes the block indentation, so it must not start with a blank; later lines may
            return text.back() != ' ' && text.back() != '\t';
        }

        void Indent(std::string &out, size_t indent)
        {
            out.append(indent, ' ');
        }

        void WriteLiteral(std::string &out, std::string_view text, size_t indent)
        {
            bool trailing = !text.empty() && text.back() == '\n';
            out += trailing ? "|\n" : "|-\n";
            std::string_view body = trailing ? text.substr(0, text.size() - 1) : text;
            size_t start = 0;
            while (start <= body.size())
            {
                size_t end = body.find('\n', start);
                std::string_view line =
                    body.substr(start, end == std::string_view::npos ? body.size() - start : end - start);
                if (!line.empty())
                {
                    Indent(out, indent);
                    out += line;
                }
                out += '\n';
                if (end == std::string_view::npos)
                {
                    break;
                }
                start = end + 1;
            }
        }

        bool FlowList(const Json &array)
        {
            for (const Json &item : array.AsArray())
            {
                if (!Scalar(item) || (item.IsString() && item.AsString().find('\n') != std::string::npos))
                {
                    return false;
                }
            }
            return true;
        }

        void WriteMapping(std::string &out, const Json &object, size_t indent, size_t depth);
        void WriteSequence(std::string &out, const Json &array, size_t indent, size_t depth);

        // the value part after "key:" or "-" (writes the leading space or line break itself)
        void WriteValue(std::string &out, const Json &value, size_t indent, size_t depth)
        {
            if (depth > MAX_DEPTH)
            {
                throw JsonError("YAML document nesting too deep");
            }
            if (value.IsString() && LiteralBlock(value.AsString()))
            {
                out += ' ';
                WriteLiteral(out, value.AsString(), indent);
            }
            else if (Scalar(value))
            {
                out += ' ';
                out += YamlScalar(value);
                out += '\n';
            }
            else if (value.Size() == 0)
            {
                out += value.IsObject() ? " {}\n" : " []\n";
            }
            else if (value.IsArray() && FlowList(value))
            {
                out += " [";
                bool first = true;
                for (const Json &item : value.AsArray())
                {
                    out += first ? "" : ", ";
                    out += YamlScalar(item);
                    first = false;
                }
                out += "]\n";
            }
            else if (value.IsObject())
            {
                out += '\n';
                WriteMapping(out, value, indent, depth + 1);
            }
            else
            {
                out += '\n';
                WriteSequence(out, value, indent, depth + 1);
            }
        }

        std::string Key(const std::string &key)
        {
            return PlainScalar(key) ? key : Json(key).Dump();
        }

        void WriteMapping(std::string &out, const Json &object, size_t indent, size_t depth)
        {
            for (const auto &[key, value] : object.AsObject())
            {
                Indent(out, indent);
                out += Key(key);
                out += ':';
                WriteValue(out, value, indent + 2, depth);
            }
        }

        void WriteSequence(std::string &out, const Json &array, size_t indent, size_t depth)
        {
            for (const Json &item : array.AsArray())
            {
                if ((item.IsObject() || item.IsArray()) && item.Size() > 0 && !(item.IsArray() && FlowList(item)))
                {
                    // nested block written two columns deeper, then its first indentation becomes "- "
                    std::string nested;
                    if (item.IsObject())
                    {
                        WriteMapping(nested, item, indent + 2, depth + 1);
                    }
                    else
                    {
                        WriteSequence(nested, item, indent + 2, depth + 1);
                    }
                    nested.replace(0, indent + 2, std::string(indent, ' ') + "- ");
                    out += nested;
                }
                else
                {
                    Indent(out, indent);
                    out += '-';
                    WriteValue(out, item, indent + 2, depth);
                }
            }
        }
    } // namespace

    bool PlainScalar(std::string_view text)
    {
        if (text.empty() || !(Letter(text.front()) || text.front() == '_' || text.front() == '/'))
        {
            return false;
        }
        for (char c : text)
        {
            if (!(Letter(c) || Digit(c) || c == '_' || c == '.' || c == '/' || c == '-'))
            {
                return false;
            }
        }
        static const std::array<std::string_view, 10> reserved = {"true", "false", "yes", "no", "on",
                                                                  "off",  "null",  "y",   "n",  "nan"};
        std::string lower = Lower(text);
        for (std::string_view word : reserved)
        {
            if (lower == word)
            {
                return false;
            }
        }
        return true;
    }

    std::string YamlScalar(const Json &value)
    {
        switch (value.GetType())
        {
        case Json::Type::Null:
            return "null";
        case Json::Type::Bool:
            return value.AsBool() ? "true" : "false";
        case Json::Type::Number:
            return value.Dump();
        case Json::Type::String:
            return PlainScalar(value.AsString()) ? value.AsString() : value.Dump();
        default:
            throw JsonError("YAML scalar expected");
        }
    }

    std::string WriteYaml(const Json &document)
    {
        std::string out;
        if (document.IsObject() && document.Size() > 0)
        {
            WriteMapping(out, document, 0, 0);
        }
        else if (document.IsArray() && document.Size() > 0)
        {
            WriteSequence(out, document, 0, 0);
        }
        else if (document.IsObject())
        {
            out = "{}\n";
        }
        else if (document.IsArray())
        {
            out = "[]\n";
        }
        else
        {
            out = YamlScalar(document) + "\n";
        }
        return out;
    }
} // namespace eOps::Detail
