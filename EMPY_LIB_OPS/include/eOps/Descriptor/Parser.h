#pragma once
#include <string>
#include <stdexcept>
#include <filesystem>
#include <string_view>
#include "Model.h"
#include "../Common/Json.h"

// descriptor YAML/JSON codecs. Parsing accepts "specs" (Listing 2.1) and "spec"; emission uses
// "specs" in Listing 2.1 order and round-trips through ParseDescriptor. Unknown keys are not
// errors here (the Validator reports them as warnings when given the raw document).

namespace eOps::Descriptor
{
    // malformed descriptor; the message names the origin and the dotted path ("specs.inference.resources.cpu")
    struct DescriptorError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    EMPY_OPS_API RichMLService ParseDescriptor(std::string_view yaml, const std::string &origin = "<descriptor>");
    EMPY_OPS_API RichMLService LoadDescriptor(const std::filesystem::path &path);
    EMPY_OPS_API std::string EmitDescriptor(const RichMLService &service);

    // same field names as the YAML form (cpu/memory as quantity strings, e.g. "500m", "1Gi")
    EMPY_OPS_API Json ToJson(const RichMLService &service);
    EMPY_OPS_API RichMLService FromJson(const Json &json); // throws DescriptorError

    // keys present in a YAML/JSON descriptor document that the model does not know, as dotted paths
    EMPY_OPS_API std::vector<std::string> UnknownKeys(const Json &document);
} // namespace eOps::Descriptor
