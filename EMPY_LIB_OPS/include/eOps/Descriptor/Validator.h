#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "Model.h"
#include "../Common/Json.h"

// formal execution-boundary checks applied to every descriptor before deployment (by the
// Executive after generation and again, independently, by the harness Guard)

namespace eOps::Descriptor
{
    enum class Severity : uint8_t
    {
        Info,
        Warning,
        Error
    };

    EMPY_OPS_API const char *SeverityName(Severity severity); // "info" | "warning" | "error"

    struct Issue
    {
        std::string m_Path; // dotted descriptor path, e.g. "specs.actionables[0].max"
        std::string m_Message;
        Severity m_Severity = Severity::Error;
    };

    struct ValidationOptions
    {
        std::vector<std::string> m_Regions; // known regions; empty = skip the membership check
        std::vector<std::string> m_Accelerators = {"gpu", "npu", "tpu"};
        std::string m_TopicRoot = "mlops"; // control-plane root no service topic may use
    };

    EMPY_OPS_API std::vector<Issue> Validate(const RichMLService &service, const ValidationOptions &options = {});
    // Validate plus a Warning per unknown key of the raw document (UnknownKeys in Parser.h)
    EMPY_OPS_API std::vector<Issue> ValidateDocument(const Json &document, const RichMLService &service,
                                                     const ValidationOptions &options = {});
    EMPY_OPS_API bool HasErrors(const std::vector<Issue> &issues);
    // [{path, message, severity}]
    EMPY_OPS_API Json ToJson(const std::vector<Issue> &issues);
} // namespace eOps::Descriptor
