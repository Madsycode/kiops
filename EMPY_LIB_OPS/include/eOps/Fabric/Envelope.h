#pragma once
#include <string>
#include <cstdint>
#include <optional>
#include <string_view>
#include <eCore/Common/Buffers.h>
#include "../Common/Json.h"

// control-plane message envelope: {"kind", "id", "time", "source", "body"} as compact JSON.
// Data-plane payloads (observables, actionables, labels) are not envelopes.

namespace eOps::Fabric
{
    struct EMPY_OPS_API Envelope
    {
        std::string m_Kind;
        std::string m_Id;    // "" → NewId(kind) when encoded
        uint64_t m_Time = 0; // Unix ms; 0 → NowMillis() when encoded
        std::string m_Source;
        Json m_Body;

        // kind + body with a fresh id and the current time
        static Envelope Make(std::string kind, Json body, std::string source = {});

        Json ToJson() const;
        // requires a non-empty string "kind"; id/source default to "", time to 0, body to null; throws JsonError
        static Envelope FromJson(const Json &json);
        eCore::SharedBytes Encode() const;
        static Envelope Decode(eCore::ByteSpan bytes, const JsonLimits &limits = {}); // throws JsonError
    };

    // plain number text ("21.5") or a JSON object with a numeric "value"; nullopt otherwise or when non-finite
    EMPY_OPS_API std::optional<double> ParseValue(eCore::ByteSpan bytes);

    // {"value":x,"unit":u,"service":s,"instance":i,"model":v,"time":ms}; time 0 → NowMillis()
    EMPY_OPS_API eCore::SharedBytes EncodeAction(double value, std::string_view unit, std::string_view service,
                                                 std::string_view instance, uint32_t model, uint64_t time = 0);
} // namespace eOps::Fabric
