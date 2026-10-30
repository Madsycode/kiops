#pragma once
#include <string>
#include <optional>
#include <filesystem>
#include <eOps/Substrate/Substrate.h>

// helpers shared by the substrate implementations (private to EmpyOps)

namespace eOps::Substrate::Detail
{
    // spec with m_Instance and m_Version filled in; an error message when the spec cannot be enacted
    // (service/role/instance must be DNS labels, a node is required)
    std::optional<std::string> Resolve(const DeploymentSpec &spec, DeploymentSpec &resolved);

    // failed EnactResult for `instance`
    EnactResult Failure(std::string instance, std::string detail);

    // first line of every rendered manifest: "# empy-mlops: {instance, service, role, node}" (compact JSON)
    inline constexpr std::string_view MANIFEST_MARKER = "# empy-mlops: ";

    // up to `maxBytes` from the end of a text file ("" when unreadable)
    std::string Tail(const std::filesystem::path &path, size_t maxBytes = 4096);
} // namespace eOps::Substrate::Detail
