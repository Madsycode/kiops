#include <eOps/Common/Json.h>
#include <eCore/Common/YAML.h>
#include <cmath>
#include <format>
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <unordered_map>

namespace eOps
{
    namespace
    {
        const Json &StaticNull()
        {
            static const Json s_Null;
            return s_Null;
        }

        const char *TypeName(Json::Type type)
        {
            switch (type)
            {
            case Json::Type::Null:
                return "null";
            case Json::Type::Bool:
                return "boolean";
            case Json::Type::Number:
                return "number";
            case Json::Type::String:
                return "string";
            case Json::Type::Array:
                return "array";
            case Json::Type::Object:
                return "object";
            }
            return "unknown";
        }

        [[noreturn]] void Mismatch(Json::Type actual, const char *expected)
        {
            throw JsonError(std::format("JSON: expected {} but found {}", expected, TypeName(actual)));
        }

        // decode one UTF-8 sequence at `text[index]`; returns its length or 0 when invalid
        size_t Utf8Sequence(std::string_view text, size_t index)
        {
            auto first = static_cast<uint8_t>(text[index]);
            if (first < 0x80)
            {
                return 1;
            }
            size_t count = 0;
            uint32_t code = 0;
            uint32_t minimum = 0;
            if (first >= 0xC2 && first <= 0xDF)
            {
                count = 1;
                code = first & 0x1F;
                minimum = 0x80;
            }
            else if (first >= 0xE0 && first <= 0xEF)
            {
                count = 2;
                code = first & 0x0F;
                minimum = 0x800;
            }
            else if (first >= 0xF0 && first <= 0xF4)
            {
                count = 3;
                code = first & 0x07;
                minimum = 0x10000;
            }
            else
            {
                return 0;
            }
            if (count > text.size() - index - 1)
            {
                return 0;
            }
            for (size_t i = 1; i <= count; ++i)
            {
                auto byte = static_cast<uint8_t>(text[index + i]);
                if ((byte & 0xC0) != 0x80)
                {
                    return 0;
                }
                code = (code << 6) | (byte & 0x3F);
            }
            if (code < minimum || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF))
            {
                return 0;
            }
            return count + 1;
        }

        void AppendUtf8(std::string &out, uint32_t code)
        {
            if (code < 0x80)
            {
                out.push_back(static_cast<char>(code));
            }
            else if (code < 0x800)
            {
                out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            else if (code < 0x10000)
            {
                out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            else
            {
                out.push_back(static_cast<char>(0xF0 | (code >> 18)));
                out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
        }

        void WriteString(std::string &out, std::string_view text)
        {
            static constexpr char s_Hex[] = "0123456789abcdef";
            out.push_back('"');
            for (size_t i = 0; i < text.size();)
            {
                auto c = static_cast<uint8_t>(text[i]);
                if (c >= 0x80)
                {
                    size_t length = Utf8Sequence(text, i);
                    if (length == 0)
                    {
                        out += "\xEF\xBF\xBD";
                        ++i;
                    }
                    else
                    {
                        out.append(text.substr(i, length));
                        i += length;
                    }
                    continue;
                }
                ++i;
                switch (c)
                {
                case '"':
                    out += "\\\"";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\b':
                    out += "\\b";
                    break;
                case '\f':
                    out += "\\f";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\t':
                    out += "\\t";
                    break;
                default:
                    if (c < 0x20)
                    {
                        out += "\\u00";
                        out.push_back(s_Hex[c >> 4]);
                        out.push_back(s_Hex[c & 0x0F]);
                    }
                    else
                    {
                        out.push_back(static_cast<char>(c));
                    }
                }
            }
            out.push_back('"');
        }

        template <typename _Type> void WriteInteger(std::string &out, _Type value)
        {
            char buffer[32];
            auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
            out.append(buffer, result.ptr);
        }

        void WriteDouble(std::string &out, double value)
        {
            if (!std::isfinite(value))
            {
                throw JsonError("JSON: NaN and infinity cannot be serialised");
            }
            char buffer[64];
            auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
            out.append(buffer, result.ptr);
        }

        void Newline(std::string &out, int32_t indent, size_t level)
        {
            out.push_back('\n');
            out.append(static_cast<size_t>(indent) * level, ' ');
        }

        class Parser
        {
        public:
            Parser(std::string_view text, const JsonLimits &limits) : m_Text(text), m_Limits(limits)
            {
            }

            Json Document()
            {
                if (m_Text.size() > m_Limits.m_MaxBytes)
                {
                    throw JsonError(std::format("JSON: document of {} bytes exceeds the {} byte limit", m_Text.size(),
                                                m_Limits.m_MaxBytes));
                }
                Json value = Value(0);
                Skip();
                if (m_Position != m_Text.size())
                {
                    Fail("unexpected trailing content");
                }
                return value;
            }

        private:
            [[noreturn]] void Fail(const char *what) const
            {
                throw JsonError(std::format("JSON: {} at offset {}", what, m_Position));
            }

            bool AtEnd() const noexcept
            {
                return m_Position >= m_Text.size();
            }

            char Peek() const noexcept
            {
                return m_Text[m_Position];
            }

            void Skip() noexcept
            {
                while (!AtEnd())
                {
                    char c = Peek();
                    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                    {
                        break;
                    }
                    ++m_Position;
                }
            }

            void Expect(char c, const char *what)
            {
                if (AtEnd() || Peek() != c)
                {
                    Fail(what);
                }
                ++m_Position;
            }

            Json Value(size_t depth)
            {
                Skip();
                if (AtEnd())
                {
                    Fail("unexpected end of input");
                }
                switch (Peek())
                {
                case '{':
                    return ParseObject(depth + 1);
                case '[':
                    return ParseArray(depth + 1);
                case '"':
                    return Json(ParseString());
                case 't':
                    Literal("true");
                    return Json(true);
                case 'f':
                    Literal("false");
                    return Json(false);
                case 'n':
                    Literal("null");
                    return Json();
                default:
                    if (Peek() == '-' || (Peek() >= '0' && Peek() <= '9'))
                    {
                        return ParseNumber();
                    }
                    Fail("unexpected character");
                }
            }

            void Literal(std::string_view word)
            {
                if (m_Text.substr(m_Position, word.size()) != word)
                {
                    Fail("invalid literal");
                }
                m_Position += word.size();
            }

            void Enter(size_t depth) const
            {
                if (depth > m_Limits.m_MaxDepth)
                {
                    Fail("nesting exceeds the depth limit");
                }
            }

            Json ParseObject(size_t depth)
            {
                Enter(depth);
                ++m_Position;
                Json::Object members;
                std::unordered_map<std::string, size_t> index;
                Skip();
                if (!AtEnd() && Peek() == '}')
                {
                    ++m_Position;
                    return Json(std::move(members));
                }
                while (true)
                {
                    Skip();
                    if (AtEnd() || Peek() != '"')
                    {
                        Fail("expected an object key");
                    }
                    std::string key = ParseString();
                    Skip();
                    Expect(':', "expected ':' after an object key");
                    Json value = Value(depth);
                    // duplicate keys: the last value wins in the first key's position
                    if (members.size() < 16)
                    {
                        auto found = std::find_if(members.begin(), members.end(),
                                                  [&](const auto &member) { return member.first == key; });
                        if (found != members.end())
                        {
                            found->second = std::move(value);
                        }
                        else
                        {
                            members.emplace_back(std::move(key), std::move(value));
                        }
                    }
                    else
                    {
                        if (index.empty())
                        {
                            for (size_t i = 0; i < members.size(); ++i)
                            {
                                index.emplace(members[i].first, i);
                            }
                        }
                        auto [found, inserted] = index.emplace(key, members.size());
                        if (inserted)
                        {
                            members.emplace_back(std::move(key), std::move(value));
                        }
                        else
                        {
                            members[found->second].second = std::move(value);
                        }
                    }
                    Skip();
                    if (AtEnd())
                    {
                        Fail("unterminated object");
                    }
                    if (Peek() == ',')
                    {
                        ++m_Position;
                        continue;
                    }
                    Expect('}', "expected ',' or '}' in an object");
                    return Json(std::move(members));
                }
            }

            Json ParseArray(size_t depth)
            {
                Enter(depth);
                ++m_Position;
                Json::Array items;
                Skip();
                if (!AtEnd() && Peek() == ']')
                {
                    ++m_Position;
                    return Json(std::move(items));
                }
                while (true)
                {
                    items.push_back(Value(depth));
                    Skip();
                    if (AtEnd())
                    {
                        Fail("unterminated array");
                    }
                    if (Peek() == ',')
                    {
                        ++m_Position;
                        continue;
                    }
                    Expect(']', "expected ',' or ']' in an array");
                    return Json(std::move(items));
                }
            }

            uint32_t Hex4()
            {
                if (m_Text.size() - m_Position < 4)
                {
                    Fail("truncated \\u escape");
                }
                uint32_t value = 0;
                for (size_t i = 0; i < 4; ++i)
                {
                    char c = m_Text[m_Position++];
                    value <<= 4;
                    if (c >= '0' && c <= '9')
                    {
                        value |= static_cast<uint32_t>(c - '0');
                    }
                    else if (c >= 'a' && c <= 'f')
                    {
                        value |= static_cast<uint32_t>(c - 'a' + 10);
                    }
                    else if (c >= 'A' && c <= 'F')
                    {
                        value |= static_cast<uint32_t>(c - 'A' + 10);
                    }
                    else
                    {
                        Fail("invalid hexadecimal digit in \\u escape");
                    }
                }
                return value;
            }

            std::string ParseString()
            {
                ++m_Position;
                std::string out;
                while (true)
                {
                    if (AtEnd())
                    {
                        Fail("unterminated string");
                    }
                    auto c = static_cast<uint8_t>(Peek());
                    if (c == '"')
                    {
                        ++m_Position;
                        return out;
                    }
                    if (c < 0x20)
                    {
                        Fail("unescaped control character in string");
                    }
                    if (c >= 0x80)
                    {
                        size_t length = Utf8Sequence(m_Text, m_Position);
                        if (length == 0)
                        {
                            Fail("invalid UTF-8 in string");
                        }
                        out.append(m_Text.substr(m_Position, length));
                        m_Position += length;
                        continue;
                    }
                    ++m_Position;
                    if (c != '\\')
                    {
                        out.push_back(static_cast<char>(c));
                        continue;
                    }
                    if (AtEnd())
                    {
                        Fail("unterminated escape");
                    }
                    char escape = m_Text[m_Position++];
                    switch (escape)
                    {
                    case '"':
                        out.push_back('"');
                        break;
                    case '\\':
                        out.push_back('\\');
                        break;
                    case '/':
                        out.push_back('/');
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u':
                    {
                        uint32_t code = Hex4();
                        if (code >= 0xDC00 && code <= 0xDFFF)
                        {
                            Fail("unpaired low surrogate");
                        }
                        if (code >= 0xD800 && code <= 0xDBFF)
                        {
                            if (m_Text.substr(m_Position, 2) != "\\u")
                            {
                                Fail("unpaired high surrogate");
                            }
                            m_Position += 2;
                            uint32_t low = Hex4();
                            if (low < 0xDC00 || low > 0xDFFF)
                            {
                                Fail("invalid low surrogate");
                            }
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        }
                        AppendUtf8(out, code);
                        break;
                    }
                    default:
                        Fail("invalid escape sequence");
                    }
                }
            }

            Json ParseNumber()
            {
                size_t start = m_Position;
                bool negative = false;
                bool fraction = false;
                if (Peek() == '-')
                {
                    negative = true;
                    ++m_Position;
                }
                auto digits = [this]
                {
                    size_t begin = m_Position;
                    while (!AtEnd() && Peek() >= '0' && Peek() <= '9')
                    {
                        ++m_Position;
                    }
                    return m_Position - begin;
                };
                if (AtEnd() || Peek() < '0' || Peek() > '9')
                {
                    Fail("expected a digit");
                }
                if (Peek() == '0')
                {
                    ++m_Position;
                    if (!AtEnd() && Peek() >= '0' && Peek() <= '9')
                    {
                        Fail("leading zeros are not allowed");
                    }
                }
                else
                {
                    digits();
                }
                if (!AtEnd() && Peek() == '.')
                {
                    fraction = true;
                    ++m_Position;
                    if (digits() == 0)
                    {
                        Fail("expected a digit after the decimal point");
                    }
                }
                if (!AtEnd() && (Peek() == 'e' || Peek() == 'E'))
                {
                    fraction = true;
                    ++m_Position;
                    if (!AtEnd() && (Peek() == '+' || Peek() == '-'))
                    {
                        ++m_Position;
                    }
                    if (digits() == 0)
                    {
                        Fail("expected a digit in the exponent");
                    }
                }
                std::string_view text = m_Text.substr(start, m_Position - start);
                const char *first = text.data();
                const char *last = text.data() + text.size();
                if (!fraction)
                {
                    if (negative)
                    {
                        int64_t value = 0;
                        auto result = std::from_chars(first, last, value);
                        if (result.ec == std::errc{} && result.ptr == last)
                        {
                            return Json(value);
                        }
                    }
                    else
                    {
                        uint64_t value = 0;
                        auto result = std::from_chars(first, last, value);
                        if (result.ec == std::errc{} && result.ptr == last)
                        {
                            return value <= static_cast<uint64_t>(INT64_MAX) ? Json(static_cast<int64_t>(value))
                                                                             : Json(value);
                        }
                    }
                }
                double value = 0;
                auto result = std::from_chars(first, last, value);
                if (result.ec == std::errc::result_out_of_range)
                {
                    // underflow rounds to (signed) zero; overflow is unrepresentable
                    value = std::strtod(std::string(text).c_str(), nullptr);
                    if (!std::isfinite(value))
                    {
                        Fail("number out of range");
                    }
                }
                else if (result.ec != std::errc{} || result.ptr != last)
                {
                    Fail("invalid number");
                }
                return Json(value);
            }

            std::string_view m_Text;
            const JsonLimits &m_Limits;
            size_t m_Position = 0;
        };

        Json YamlValue(const eCore::YamlNode &node, size_t depth)
        {
            if (depth > 256)
            {
                throw JsonError(node.Path() + ": YAML nesting too deep");
            }
            if (node.IsNull())
            {
                return Json();
            }
            if (node.IsMap())
            {
                Json object = Json::MakeObject();
                for (const std::string &key : node.Keys())
                {
                    object.Set(key, YamlValue(node[key], depth + 1));
                }
                return object;
            }
            if (node.IsList())
            {
                Json array = Json::MakeArray();
                for (const eCore::YamlNode &item : node.Items())
                {
                    array.Push(YamlValue(item, depth + 1));
                }
                return array;
            }
            try
            {
                return Json(node.As<bool>());
            }
            catch (const eCore::ConfigError &)
            {
            }
            try
            {
                return Json(node.As<int64_t>());
            }
            catch (const eCore::ConfigError &)
            {
            }
            try
            {
                return Json(node.As<double>());
            }
            catch (const eCore::ConfigError &)
            {
            }
            return Json(node.As<std::string>());
        }
    } // namespace

    Json::Json() noexcept = default;

    Json::Json(std::nullptr_t) noexcept
    {
    }

    Json::Json(bool value) : m_Type(Type::Bool), m_Bool(value)
    {
    }

    Json::Json(int32_t value) : Json(static_cast<int64_t>(value))
    {
    }

    Json::Json(uint32_t value) : Json(static_cast<int64_t>(value))
    {
    }

    Json::Json(int64_t value) : m_Type(Type::Number), m_Number(Number::Signed), m_Signed(value)
    {
    }

    Json::Json(uint64_t value) : m_Type(Type::Number)
    {
        if (value <= static_cast<uint64_t>(INT64_MAX))
        {
            m_Number = Number::Signed;
            m_Signed = static_cast<int64_t>(value);
        }
        else
        {
            m_Number = Number::Unsigned;
            m_Unsigned = value;
        }
    }

    Json::Json(double value) : m_Type(Type::Number), m_Number(Number::Double), m_Double(value)
    {
    }

    Json::Json(const char *value) : m_Type(Type::String), m_String(value ? value : "")
    {
    }

    Json::Json(std::string value) : m_Type(Type::String), m_String(std::move(value))
    {
    }

    Json::Json(std::string_view value) : m_Type(Type::String), m_String(value)
    {
    }

    Json::Json(Array items) : m_Type(Type::Array), m_Array(std::move(items))
    {
    }

    Json::Json(Object members) : m_Type(Type::Object)
    {
        // enforce unique keys: a later duplicate replaces the earlier value
        for (auto &member : members)
        {
            Set(std::move(member.first), std::move(member.second));
        }
    }

    Json Json::MakeArray(Array items)
    {
        return Json(std::move(items));
    }

    Json Json::MakeObject(Object members)
    {
        return Json(std::move(members));
    }

    Json::Type Json::GetType() const noexcept
    {
        return m_Type;
    }

    bool Json::IsNull() const noexcept
    {
        return m_Type == Type::Null;
    }

    bool Json::IsBool() const noexcept
    {
        return m_Type == Type::Bool;
    }

    bool Json::IsNumber() const noexcept
    {
        return m_Type == Type::Number;
    }

    bool Json::IsInteger() const noexcept
    {
        return m_Type == Type::Number && m_Number != Number::Double;
    }

    bool Json::IsString() const noexcept
    {
        return m_Type == Type::String;
    }

    bool Json::IsArray() const noexcept
    {
        return m_Type == Type::Array;
    }

    bool Json::IsObject() const noexcept
    {
        return m_Type == Type::Object;
    }

    bool Json::AsBool() const
    {
        if (m_Type != Type::Bool)
        {
            Mismatch(m_Type, "boolean");
        }
        return m_Bool;
    }

    double Json::AsNumber() const
    {
        if (m_Type != Type::Number)
        {
            Mismatch(m_Type, "number");
        }
        switch (m_Number)
        {
        case Number::Signed:
            return static_cast<double>(m_Signed);
        case Number::Unsigned:
            return static_cast<double>(m_Unsigned);
        default:
            return m_Double;
        }
    }

    int64_t Json::AsInt() const
    {
        if (m_Type != Type::Number)
        {
            Mismatch(m_Type, "integer");
        }
        if (m_Number == Number::Signed)
        {
            return m_Signed;
        }
        if (m_Number == Number::Unsigned)
        {
            throw JsonError("JSON: integer exceeds the int64 range");
        }
        // 2^63 is exactly representable; the valid range is [-2^63, 2^63)
        if (std::trunc(m_Double) != m_Double || m_Double < -9223372036854775808.0 || m_Double >= 9223372036854775808.0)
        {
            throw JsonError("JSON: number is not an int64 integer");
        }
        return static_cast<int64_t>(m_Double);
    }

    uint64_t Json::AsUint() const
    {
        if (m_Type != Type::Number)
        {
            Mismatch(m_Type, "integer");
        }
        if (m_Number == Number::Unsigned)
        {
            return m_Unsigned;
        }
        if (m_Number == Number::Signed)
        {
            if (m_Signed < 0)
            {
                throw JsonError("JSON: integer is negative");
            }
            return static_cast<uint64_t>(m_Signed);
        }
        if (std::trunc(m_Double) != m_Double || m_Double < 0 || m_Double >= 18446744073709551616.0)
        {
            throw JsonError("JSON: number is not a uint64 integer");
        }
        return static_cast<uint64_t>(m_Double);
    }

    const std::string &Json::AsString() const
    {
        if (m_Type != Type::String)
        {
            Mismatch(m_Type, "string");
        }
        return m_String;
    }

    const Json::Array &Json::AsArray() const
    {
        if (m_Type != Type::Array)
        {
            Mismatch(m_Type, "array");
        }
        return m_Array;
    }

    Json::Array &Json::AsArray()
    {
        if (m_Type != Type::Array)
        {
            Mismatch(m_Type, "array");
        }
        return m_Array;
    }

    const Json::Object &Json::AsObject() const
    {
        if (m_Type != Type::Object)
        {
            Mismatch(m_Type, "object");
        }
        return m_Object;
    }

    Json::Object &Json::AsObject()
    {
        if (m_Type != Type::Object)
        {
            Mismatch(m_Type, "object");
        }
        return m_Object;
    }

    size_t Json::Size() const noexcept
    {
        if (m_Type == Type::Array)
        {
            return m_Array.size();
        }
        if (m_Type == Type::Object)
        {
            return m_Object.size();
        }
        return 0;
    }

    bool Json::Has(std::string_view key) const
    {
        return Find(key) != nullptr;
    }

    const Json *Json::Find(std::string_view key) const
    {
        if (m_Type != Type::Object)
        {
            return nullptr;
        }
        for (const auto &member : m_Object)
        {
            if (member.first == key)
            {
                return &member.second;
            }
        }
        return nullptr;
    }

    Json *Json::Find(std::string_view key)
    {
        return const_cast<Json *>(static_cast<const Json *>(this)->Find(key));
    }

    const Json &Json::operator[](std::string_view key) const
    {
        const Json *found = Find(key);
        return found ? *found : StaticNull();
    }

    Json &Json::operator[](std::string_view key)
    {
        if (m_Type == Type::Null)
        {
            m_Type = Type::Object;
        }
        if (m_Type != Type::Object)
        {
            Mismatch(m_Type, "object");
        }
        if (Json *found = Find(key))
        {
            return *found;
        }
        m_Object.emplace_back(std::string(key), Json());
        return m_Object.back().second;
    }

    const Json &Json::operator[](size_t index) const
    {
        if (m_Type != Type::Array || index >= m_Array.size())
        {
            return StaticNull();
        }
        return m_Array[index];
    }

    Json &Json::Set(std::string key, Json value)
    {
        if (m_Type == Type::Null)
        {
            m_Type = Type::Object;
        }
        if (m_Type != Type::Object)
        {
            Mismatch(m_Type, "object");
        }
        if (Json *found = Find(key))
        {
            *found = std::move(value);
        }
        else
        {
            m_Object.emplace_back(std::move(key), std::move(value));
        }
        return *this;
    }

    bool Json::Erase(std::string_view key)
    {
        if (m_Type != Type::Object)
        {
            return false;
        }
        auto found =
            std::find_if(m_Object.begin(), m_Object.end(), [&](const auto &member) { return member.first == key; });
        if (found == m_Object.end())
        {
            return false;
        }
        m_Object.erase(found);
        return true;
    }

    Json &Json::Push(Json value)
    {
        if (m_Type == Type::Null)
        {
            m_Type = Type::Array;
        }
        if (m_Type != Type::Array)
        {
            Mismatch(m_Type, "array");
        }
        m_Array.push_back(std::move(value));
        return *this;
    }

    std::string Json::GetString(std::string_view key, std::string fallback) const
    {
        const Json *found = Find(key);
        return found && found->IsString() ? found->m_String : fallback;
    }

    double Json::GetNumber(std::string_view key, double fallback) const
    {
        const Json *found = Find(key);
        return found && found->IsNumber() ? found->AsNumber() : fallback;
    }

    int64_t Json::GetInt(std::string_view key, int64_t fallback) const
    {
        const Json *found = Find(key);
        if (!found || !found->IsNumber())
        {
            return fallback;
        }
        try
        {
            return found->AsInt();
        }
        catch (const JsonError &)
        {
            return fallback;
        }
    }

    bool Json::GetBool(std::string_view key, bool fallback) const
    {
        const Json *found = Find(key);
        return found && found->IsBool() ? found->m_Bool : fallback;
    }

    const Json *Json::At(std::string_view dotted) const
    {
        const Json *current = this;
        while (current)
        {
            auto end = dotted.find('.');
            std::string_view segment = dotted.substr(0, end);
            if (current->IsObject())
            {
                current = current->Find(segment);
            }
            else if (current->IsArray())
            {
                size_t index = 0;
                auto result = std::from_chars(segment.data(), segment.data() + segment.size(), index);
                if (segment.empty() || result.ec != std::errc{} || result.ptr != segment.data() + segment.size() ||
                    index >= current->m_Array.size())
                {
                    return nullptr;
                }
                current = &current->m_Array[index];
            }
            else
            {
                return nullptr;
            }
            if (end == std::string_view::npos)
            {
                return current;
            }
            dotted.remove_prefix(end + 1);
        }
        return nullptr;
    }

    bool Json::operator==(const Json &other) const
    {
        if (m_Type != other.m_Type)
        {
            return false;
        }
        switch (m_Type)
        {
        case Type::Null:
            return true;
        case Type::Bool:
            return m_Bool == other.m_Bool;
        case Type::Number:
            if (m_Number != Number::Double && other.m_Number != Number::Double)
            {
                // exact integer comparison (a Signed value is never above INT64_MAX by construction)
                if (m_Number != other.m_Number)
                {
                    return false;
                }
                return m_Number == Number::Signed ? m_Signed == other.m_Signed : m_Unsigned == other.m_Unsigned;
            }
            return AsNumber() == other.AsNumber();
        case Type::String:
            return m_String == other.m_String;
        case Type::Array:
            return m_Array == other.m_Array;
        case Type::Object:
            if (m_Object.size() != other.m_Object.size())
            {
                return false;
            }
            for (const auto &member : m_Object)
            {
                const Json *match = other.Find(member.first);
                if (!match || !(member.second == *match))
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    std::string Json::Dump(int32_t indent) const
    {
        std::string out;
        // recursion depth equals the document nesting depth
        auto write = [&](auto &&self, const Json &value, size_t level) -> void
        {
            switch (value.m_Type)
            {
            case Type::Null:
                out += "null";
                break;
            case Type::Bool:
                out += value.m_Bool ? "true" : "false";
                break;
            case Type::Number:
                if (value.m_Number == Number::Signed)
                {
                    WriteInteger(out, value.m_Signed);
                }
                else if (value.m_Number == Number::Unsigned)
                {
                    WriteInteger(out, value.m_Unsigned);
                }
                else
                {
                    WriteDouble(out, value.m_Double);
                }
                break;
            case Type::String:
                WriteString(out, value.m_String);
                break;
            case Type::Array:
                out.push_back('[');
                for (size_t i = 0; i < value.m_Array.size(); ++i)
                {
                    if (i > 0)
                    {
                        out.push_back(',');
                    }
                    if (indent >= 0)
                    {
                        Newline(out, indent, level + 1);
                    }
                    self(self, value.m_Array[i], level + 1);
                }
                if (indent >= 0 && !value.m_Array.empty())
                {
                    Newline(out, indent, level);
                }
                out.push_back(']');
                break;
            case Type::Object:
                out.push_back('{');
                for (size_t i = 0; i < value.m_Object.size(); ++i)
                {
                    if (i > 0)
                    {
                        out.push_back(',');
                    }
                    if (indent >= 0)
                    {
                        Newline(out, indent, level + 1);
                    }
                    WriteString(out, value.m_Object[i].first);
                    out += indent >= 0 ? ": " : ":";
                    self(self, value.m_Object[i].second, level + 1);
                }
                if (indent >= 0 && !value.m_Object.empty())
                {
                    Newline(out, indent, level);
                }
                out.push_back('}');
                break;
            }
        };
        write(write, *this, 0);
        return out;
    }

    Json Json::Parse(std::string_view text, const JsonLimits &limits)
    {
        return Parser(text, limits).Document();
    }

    eCore::SharedBytes ToBytes(const Json &value)
    {
        return eCore::SharedBytes::Copy(value.Dump());
    }

    Json FromBytes(eCore::ByteSpan bytes, const JsonLimits &limits)
    {
        return Json::Parse(eCore::AsText(bytes), limits);
    }

    Json FromYaml(const eCore::YamlNode &node)
    {
        return YamlValue(node, 0);
    }
} // namespace eOps
