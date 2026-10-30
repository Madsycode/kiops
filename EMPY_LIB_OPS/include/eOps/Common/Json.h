#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <eCore/Common/Buffers.h>
#include "../API.h"

// JSON document model for MLOps envelopes, descriptors and knowledge entries: strict RFC 8259
// parsing with depth/size limits, exact integers, insertion-ordered objects and deterministic output

namespace eCore
{
    class YamlNode;
}

namespace eOps
{
    // parser resource limits
    struct JsonLimits
    {
        size_t m_MaxDepth = 64;
        size_t m_MaxBytes = 16 * 1024 * 1024;
    };

    // raised for malformed text, violated limits, type mismatches and unrepresentable values
    struct JsonError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    class EMPY_OPS_API Json
    {
    public:
        enum class Type : uint8_t
        {
            Null,
            Bool,
            Number,
            String,
            Array,
            Object
        };
        using Array = std::vector<Json>;
        using Object = std::vector<std::pair<std::string, Json>>; // insertion ordered, unique keys

        Json() noexcept;
        Json(std::nullptr_t) noexcept;
        Json(bool value);
        Json(int32_t value);
        Json(uint32_t value);
        Json(int64_t value);
        Json(uint64_t value);
        Json(double value);
        Json(const char *value);
        Json(std::string value);
        Json(std::string_view value);
        Json(Array items);
        Json(Object members);

        // remaining integral types (long, short, ...) keep their exact value
        template <typename _Type>
            requires(std::is_integral_v<_Type> && !std::is_same_v<_Type, bool> && !std::is_same_v<_Type, char> &&
                     !std::is_same_v<_Type, int32_t> && !std::is_same_v<_Type, uint32_t> &&
                     !std::is_same_v<_Type, int64_t> && !std::is_same_v<_Type, uint64_t>)
        Json(_Type value)
            : Json(std::is_signed_v<_Type> ? Json(static_cast<int64_t>(value)) : Json(static_cast<uint64_t>(value)))
        {
        }

        static Json MakeArray(Array items = {});
        static Json MakeObject(Object members = {});

        Type GetType() const noexcept;
        bool IsNull() const noexcept;
        bool IsBool() const noexcept;
        bool IsNumber() const noexcept;
        bool IsInteger() const noexcept; // number held exactly as an int64/uint64 value
        bool IsString() const noexcept;
        bool IsArray() const noexcept;
        bool IsObject() const noexcept;

        // typed access; throws JsonError on type mismatch
        bool AsBool() const;
        double AsNumber() const;
        int64_t AsInt() const;   // integers in int64 range, or integral-valued doubles in range
        uint64_t AsUint() const; // non-negative integers, or integral-valued doubles in range
        const std::string &AsString() const;
        const Array &AsArray() const;
        Array &AsArray();
        const Object &AsObject() const;
        Object &AsObject();

        size_t Size() const noexcept;         // array/object element count, else 0
        bool Has(std::string_view key) const; // object member present (any value)
        const Json *Find(std::string_view key) const;
        Json *Find(std::string_view key);
        const Json &operator[](std::string_view key) const; // missing → shared static null
        Json &operator[](std::string_view key);             // null converts to object; inserts null member
        const Json &operator[](size_t index) const;         // out of range → static null
        Json &Set(std::string key, Json value);             // insert or replace, returns *this
        bool Erase(std::string_view key);
        Json &Push(Json value); // null converts to array, returns *this

        // lenient member access: absent or mismatched members yield the fallback
        std::string GetString(std::string_view key, std::string fallback = {}) const;
        double GetNumber(std::string_view key, double fallback = 0) const;
        int64_t GetInt(std::string_view key, int64_t fallback = 0) const;
        bool GetBool(std::string_view key, bool fallback = false) const;

        const Json *At(std::string_view dotted) const; // "a.b.2.c" path; nullptr when absent

        bool operator==(const Json &other) const; // deep, object order-insensitive, numbers by value

        // deterministic text: -1 compact, otherwise pretty with `indent` spaces; throws on NaN/inf.
        // Invalid UTF-8 inside strings is written as U+FFFD so the output is always valid JSON.
        std::string Dump(int32_t indent = -1) const;

        // strict RFC 8259 document; duplicate keys keep the last value; throws JsonError
        static Json Parse(std::string_view text, const JsonLimits &limits = {});

    private:
        enum class Number : uint8_t
        {
            Double,
            Signed,
            Unsigned
        };

        Type m_Type = Type::Null;
        Number m_Number = Number::Double;
        bool m_Bool = false;
        int64_t m_Signed = 0;
        uint64_t m_Unsigned = 0;
        double m_Double = 0;
        std::string m_String;
        Array m_Array;
        Object m_Object;
    };

    EMPY_OPS_API eCore::SharedBytes ToBytes(const Json &value);
    EMPY_OPS_API Json FromBytes(eCore::ByteSpan bytes, const JsonLimits &limits = {});
    // scalars: bool → int64 → double → string; maps/lists recursive; null → null
    EMPY_OPS_API Json FromYaml(const eCore::YamlNode &node);
} // namespace eOps
