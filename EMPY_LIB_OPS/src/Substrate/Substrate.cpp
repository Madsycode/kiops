#include <eOps/Substrate/Substrate.h>
#include <eOps/Common/Text.h>
#include "Detail.h"
#include <cstdio>
#include <fstream>

namespace eOps::Substrate
{
    namespace
    {
        constexpr size_t MAX_LABEL = 63;

        uint64_t Fnv1a(std::string_view text)
        {
            uint64_t hash = 14695981039346656037ull;
            for (char c : text)
            {
                hash ^= static_cast<unsigned char>(c);
                hash *= 1099511628211ull;
            }
            return hash;
        }

        // lowercase alphanumerics with single '-' separators, no leading/trailing '-'
        std::string Sanitise(std::string_view text)
        {
            std::string out;
            for (char c : Lower(text))
            {
                bool keep = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
                if (keep)
                {
                    out += c;
                }
                else if (!out.empty() && out.back() != '-')
                {
                    out += '-';
                }
            }
            while (!out.empty() && out.back() == '-')
            {
                out.pop_back();
            }
            return out;
        }
    } // namespace

    std::string InstanceName(const std::string &service, const std::string &role, const std::string &node,
                             uint32_t replica)
    {
        std::string full = service + "-" + role + "-" + node + "-" + std::to_string(replica);
        std::string name = Sanitise(full);
        if (name.empty())
        {
            name = "instance";
        }
        if (name.size() > MAX_LABEL)
        {
            // keep names unique after truncation: 54 characters + '-' + 8 hex digits of the full name
            char suffix[9];
            std::snprintf(suffix, sizeof(suffix), "%08x", static_cast<uint32_t>(Fnv1a(full)));
            name.resize(MAX_LABEL - 9);
            while (!name.empty() && name.back() == '-')
            {
                name.pop_back();
            }
            name += "-";
            name += suffix;
        }
        return name;
    }

    Json EnactResult::ToJson() const
    {
        Json artifacts = Json::MakeArray();
        for (const std::string &artifact : m_Artifacts)
        {
            artifacts.Push(Json(artifact));
        }
        return Json(Json::Object{
            {"ok", m_Ok}, {"instance", m_Instance}, {"detail", m_Detail}, {"artifacts", std::move(artifacts)}});
    }

    Json InstanceStatus::ToJson() const
    {
        return Json(Json::Object{
            {"instance", m_Instance}, {"service", m_Service}, {"role", m_Role}, {"node", m_Node}, {"state", m_State}});
    }

    ExecutionSubstrate::~ExecutionSubstrate() = default;

    Json ExecutionSubstrate::Describe()
    {
        Json instances = Json::MakeArray();
        for (const InstanceStatus &status : List())
        {
            instances.Push(status.ToJson());
        }
        return Json(Json::Object{{"type", Name()}, {"instances", std::move(instances)}});
    }

    std::vector<std::string> WorkloadArgs(const DeploymentSpec &spec, const std::string &descriptorPath)
    {
        std::vector<std::string> args = {spec.m_Role,
                                         "--descriptor",
                                         descriptorPath,
                                         "--instance",
                                         spec.m_Instance,
                                         "--node",
                                         spec.m_Node,
                                         "--region",
                                         spec.m_Region,
                                         "--host",
                                         spec.m_BrokerHost,
                                         "--port",
                                         std::to_string(spec.m_BrokerPort),
                                         "--root",
                                         spec.m_TopicRoot};
        args.insert(args.end(), spec.m_ExtraArgs.begin(), spec.m_ExtraArgs.end());
        return args;
    }

    std::filesystem::path DefaultWorkloadExecutable()
    {
#ifdef _WIN32
        std::wstring buffer(MAX_PATH, L'\0');
        while (true)
        {
            DWORD size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (size == 0)
            {
                return std::filesystem::path("EmpyWorkload.exe");
            }
            if (size < buffer.size())
            {
                buffer.resize(size);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }
        return std::filesystem::path(buffer).parent_path() / "EmpyWorkload.exe";
#else
        std::error_code error;
        std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
        if (error)
        {
            return std::filesystem::path("EmpyWorkload");
        }
        return self.parent_path() / "EmpyWorkload";
#endif
    }

    namespace Detail
    {
        std::optional<std::string> Resolve(const DeploymentSpec &spec, DeploymentSpec &resolved)
        {
            resolved = spec;
            if (!IsDnsLabel(spec.m_Service))
            {
                return "service '" + spec.m_Service + "' is not a DNS label";
            }
            if (!IsDnsLabel(spec.m_Role))
            {
                return "role '" + spec.m_Role + "' is not a DNS label";
            }
            if (spec.m_Node.empty())
            {
                return "no node selected for " + spec.m_Service + "/" + spec.m_Role;
            }
            if (resolved.m_Instance.empty())
            {
                resolved.m_Instance = InstanceName(spec.m_Service, spec.m_Role, spec.m_Node, spec.m_Replica);
            }
            else if (!IsDnsLabel(resolved.m_Instance))
            {
                return "instance '" + resolved.m_Instance + "' is not a DNS label";
            }
            if (resolved.m_Version.empty())
            {
                resolved.m_Version = spec.m_Descriptor.m_Metadata.m_Version;
            }
            if (resolved.m_TopicRoot.empty())
            {
                return "empty topic root";
            }
            return std::nullopt;
        }

        EnactResult Failure(std::string instance, std::string detail)
        {
            EnactResult result;
            result.m_Ok = false;
            result.m_Instance = std::move(instance);
            result.m_Detail = std::move(detail);
            return result;
        }

        std::string Tail(const std::filesystem::path &path, size_t maxBytes)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                return {};
            }
            file.seekg(0, std::ios::end);
            std::streamoff size = file.tellg();
            if (size <= 0)
            {
                return {};
            }
            std::streamoff start =
                size > static_cast<std::streamoff>(maxBytes) ? size - static_cast<std::streamoff>(maxBytes) : 0;
            file.seekg(start);
            std::string text(static_cast<size_t>(size - start), '\0');
            file.read(text.data(), static_cast<std::streamsize>(text.size()));
            text.resize(static_cast<size_t>(file.gcount()));
            return Trim(text);
        }
    } // namespace Detail
} // namespace eOps::Substrate
