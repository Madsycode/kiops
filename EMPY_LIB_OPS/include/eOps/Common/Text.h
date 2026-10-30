#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <filesystem>
#include <string_view>
#include "../API.h"

// text, identifier, time and file helpers shared by every MLOps module

namespace eOps
{
    // ASCII lowercase
    EMPY_OPS_API std::string Lower(std::string_view text);

    // strip leading/trailing ASCII whitespace
    EMPY_OPS_API std::string Trim(std::string_view text);

    // split on `separator`, keeping empty fields ("a,,b" → a, "", b; "" → one empty field)
    EMPY_OPS_API std::vector<std::string> Split(std::string_view text, char separator);

    EMPY_OPS_API std::string Join(const std::vector<std::string> &parts, std::string_view separator);

    // lowercase word/number tokens for semantic parsing: numbers keep decimals ("0.9", "-5"),
    // a number prefix splits from its unit ("5ms" → "5", "ms"), words keep inner digits ("p95"),
    // "%" → "percent", "°c" → "celsius", "°f" → "fahrenheit", lone "°" → "degree"
    EMPY_OPS_API std::vector<std::string> Tokenize(std::string_view text);

    // English plural folding of one lowercase word (the single routine behind ontology and retrieval matching):
    // words of up to 3 characters and "ss"/"us"/"is" endings are kept (celsius) except the accelerator acronyms
    // gpus/cpus/npus/tpus; "ies" → "y"; sibilant "sses/xes/zes/ches/shes" lose "es"; other plurals lose "s"
    EMPY_OPS_API std::string Singular(std::string_view word);

    // RFC 1123 label as used by Kubernetes: 1..63 of [a-z0-9-], alphanumeric first and last
    EMPY_OPS_API bool IsDnsLabel(std::string_view text);

    // system clock in Unix milliseconds
    EMPY_OPS_API uint64_t NowMillis();

    // UTC "YYYY-MM-DDTHH:MM:SS.mmmZ"
    EMPY_OPS_API std::string IsoTime(uint64_t unixMillis);

    // prefix + "-" + 16 lowercase hex characters of system randomness (prefix empty → hex only)
    EMPY_OPS_API std::string NewId(std::string_view prefix);

    // strict finite decimal number (whole text, no whitespace, no '+', no inf/nan)
    EMPY_OPS_API std::optional<double> ParseNumber(std::string_view text);

    // whole file; throws std::runtime_error when unreadable or larger than `maxBytes`
    EMPY_OPS_API std::string ReadFile(const std::filesystem::path &path, size_t maxBytes = 16 * 1024 * 1024);

    // write a sibling temporary, flush it to the device and atomically replace `path`;
    // creates missing parent directories; throws std::runtime_error on failure
    EMPY_OPS_API void WriteFileAtomic(const std::filesystem::path &path, std::string_view content);
} // namespace eOps
