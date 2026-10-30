#include <eOps/Fabric/Envelope.h>
#include <eOps/Common/Text.h>
#include <cmath>

namespace eOps::Fabric
{
    namespace
    {
        // data-plane values are small; a deep or large document is not a value
        constexpr JsonLimits VALUE_LIMITS{8, 64 * 1024};

        std::optional<double> Finite(double value)
        {
            if (!std::isfinite(value))
            {
                return std::nullopt;
            }
            return value;
        }
    } // namespace

    Envelope Envelope::Make(std::string kind, Json body, std::string source)
    {
        Envelope envelope;
        envelope.m_Id = NewId(kind);
        envelope.m_Kind = std::move(kind);
        envelope.m_Time = NowMillis();
        envelope.m_Source = std::move(source);
        envelope.m_Body = std::move(body);
        return envelope;
    }

    Json Envelope::ToJson() const
    {
        if (m_Kind.empty())
        {
            throw JsonError("Envelope kind must not be empty");
        }
        return Json::Object{{"kind", m_Kind},
                            {"id", m_Id.empty() ? NewId(m_Kind) : m_Id},
                            {"time", m_Time ? m_Time : NowMillis()},
                            {"source", m_Source},
                            {"body", m_Body}};
    }

    Envelope Envelope::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("Envelope must be a JSON object");
        }
        const Json *kind = json.Find("kind");
        if (!kind || !kind->IsString() || kind->AsString().empty())
        {
            throw JsonError("Envelope requires a non-empty string \"kind\"");
        }
        Envelope envelope;
        envelope.m_Kind = kind->AsString();
        if (const Json *id = json.Find("id"); id && !id->IsNull())
        {
            if (!id->IsString())
            {
                throw JsonError("Envelope \"id\" must be a string");
            }
            envelope.m_Id = id->AsString();
        }
        if (const Json *time = json.Find("time"); time && !time->IsNull())
        {
            if (!time->IsNumber())
            {
                throw JsonError("Envelope \"time\" must be a number");
            }
            envelope.m_Time = time->AsUint(); // throws for negative or fractional values
        }
        if (const Json *source = json.Find("source"); source && !source->IsNull())
        {
            if (!source->IsString())
            {
                throw JsonError("Envelope \"source\" must be a string");
            }
            envelope.m_Source = source->AsString();
        }
        if (const Json *body = json.Find("body"))
        {
            envelope.m_Body = *body;
        }
        return envelope;
    }

    eCore::SharedBytes Envelope::Encode() const
    {
        return ToBytes(ToJson());
    }

    Envelope Envelope::Decode(eCore::ByteSpan bytes, const JsonLimits &limits)
    {
        return FromJson(FromBytes(bytes, limits));
    }

    std::optional<double> ParseValue(eCore::ByteSpan bytes)
    {
        if (bytes.empty() || bytes.size() > VALUE_LIMITS.m_MaxBytes)
        {
            return std::nullopt;
        }
        std::string_view text(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        if (auto number = ParseNumber(Trim(text)))
        {
            return Finite(*number);
        }
        try
        {
            Json json = Json::Parse(text, VALUE_LIMITS);
            if (json.IsNumber())
            {
                return Finite(json.AsNumber());
            }
            if (const Json *value = json.IsObject() ? json.Find("value") : nullptr; value && value->IsNumber())
            {
                return Finite(value->AsNumber());
            }
        }
        catch (const JsonError &)
        {
        }
        return std::nullopt;
    }

    eCore::SharedBytes EncodeAction(double value, std::string_view unit, std::string_view service,
                                    std::string_view instance, uint32_t model, uint64_t time)
    {
        return ToBytes(Json::Object{{"value", value},
                                    {"unit", unit},
                                    {"service", service},
                                    {"instance", instance},
                                    {"model", model},
                                    {"time", time ? time : NowMillis()}});
    }
} // namespace eOps::Fabric
