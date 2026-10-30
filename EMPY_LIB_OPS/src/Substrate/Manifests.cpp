#include <eOps/Substrate/Substrate.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Text.h>
#include "../Common/Yaml.h"
#include "Detail.h"
#include <stdexcept>
#include <algorithm>
#include "../Common/JsonRead.h"

namespace eOps::Substrate
{
    namespace
    {
        constexpr const char *DESCRIPTOR_DIRECTORY = "/etc/mlops";
        constexpr const char *DESCRIPTOR_KEY = "descriptor.yaml";

        // Kubernetes label value: ≤ 63 of [A-Za-z0-9-_.], alphanumeric first and last (or empty)
        bool LabelValue(std::string_view text)
        {
            if (text.size() > 63)
            {
                return false;
            }
            auto alnum = [](char c)
            { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); };
            for (char c : text)
            {
                if (!alnum(c) && c != '-' && c != '_' && c != '.')
                {
                    return false;
                }
            }
            return text.empty() || (alnum(text.front()) && alnum(text.back()));
        }

        std::string AcceleratorResource(const std::string &kind)
        {
            if (kind == "gpu")
            {
                return "nvidia.com/gpu";
            }
            if (kind == "npu")
            {
                return "empy.io/npu";
            }
            if (kind == "tpu")
            {
                return "google.com/tpu";
            }
            return "empy.io/" + kind;
        }

        Json Labels(const DeploymentSpec &spec)
        {
            return Json(Json::Object{{"app.kubernetes.io/name", spec.m_Service},
                                     {"app.kubernetes.io/component", spec.m_Role},
                                     {"app.kubernetes.io/instance", spec.m_Instance},
                                     {"app.kubernetes.io/managed-by", "empy-mlops"},
                                     {"mlops.empy.io/service", spec.m_Service},
                                     {"mlops.empy.io/version", spec.m_Version}});
        }

        Json Metadata(const std::string &name, const std::string &nameSpace, const DeploymentSpec &spec)
        {
            return Json(Json::Object{{"name", name}, {"namespace", nameSpace}, {"labels", Labels(spec)}});
        }

        using eOps::Detail::Strings;
    } // namespace

    std::string RenderManifests(const DeploymentSpec &spec, const std::string &nameSpace, const std::string &image)
    {
        DeploymentSpec resolved;
        if (std::optional<std::string> error = Detail::Resolve(spec, resolved))
        {
            throw std::invalid_argument(*error);
        }
        if (!IsDnsLabel(nameSpace))
        {
            throw std::invalid_argument("namespace '" + nameSpace + "' is not a DNS label");
        }
        if (!LabelValue(resolved.m_Version) || !LabelValue(resolved.m_Node))
        {
            throw std::invalid_argument("version '" + resolved.m_Version + "' or node '" + resolved.m_Node +
                                        "' is not a valid Kubernetes label value");
        }
        const Descriptor::RichMLService &descriptor = resolved.m_Descriptor;
        const Descriptor::Resources *resources = nullptr;
        std::string containerImage;
        if (resolved.m_Role == "inference" && descriptor.m_Inference)
        {
            resources = &descriptor.m_Inference->m_Resources;
            containerImage = descriptor.m_Inference->m_Image;
        }
        else if (resolved.m_Role == "training" && descriptor.m_Training)
        {
            resources = &descriptor.m_Training->m_Resources;
            containerImage = descriptor.m_Training->m_Image;
        }
        else
        {
            throw std::invalid_argument("descriptor '" + descriptor.m_Metadata.m_Name + "' declares no " +
                                        resolved.m_Role + " workload");
        }
        if (containerImage.empty())
        {
            containerImage = image;
        }
        if (containerImage.empty())
        {
            throw std::invalid_argument("no container image for " + resolved.m_Instance);
        }

        const std::string &name = resolved.m_Instance;
        const std::string configMap = name + "-descriptor";
        std::string header = std::string(Detail::MANIFEST_MARKER) +
                             Json(Json::Object{{"instance", name},
                                               {"service", resolved.m_Service},
                                               {"role", resolved.m_Role},
                                               {"node", resolved.m_Node}})
                                 .Dump() +
                             "\n";

        Json configDocument(Json::Object{
            {"apiVersion", "v1"},
            {"kind", "ConfigMap"},
            {"metadata", Metadata(configMap, nameSpace, resolved)},
            {"data", Json(Json::Object{{DESCRIPTOR_KEY, Descriptor::EmitDescriptor(descriptor)}})},
        });

        Json quantities(Json::Object{{"cpu", Descriptor::FormatCpu(resources->m_Cpu)},
                                     {"memory", Descriptor::FormatMemory(resources->m_Memory)}});
        Json limits = quantities;
        if (!resources->m_Accelerator.empty())
        {
            // extended resources are whole devices and only set as limits (requests default to them)
            limits.Set(AcceleratorResource(resources->m_Accelerator),
                       std::max<uint32_t>(1, resources->m_AcceleratorCount));
        }
        std::string descriptorPath = std::string(DESCRIPTOR_DIRECTORY) + "/" + DESCRIPTOR_KEY;
        Json container(Json::Object{
            {"name", resolved.m_Role},
            {"image", containerImage},
            {"args", Strings(WorkloadArgs(resolved, descriptorPath))},
            {"resources", Json(Json::Object{{"requests", std::move(quantities)}, {"limits", std::move(limits)}})},
            {"volumeMounts", Json(Json::Array{Json(Json::Object{
                                 {"name", "descriptor"}, {"mountPath", DESCRIPTOR_DIRECTORY}, {"readOnly", true}})})},
        });
        Json podSpec(Json::Object{
            {"nodeSelector", Json(Json::Object{{"kubernetes.io/hostname", resolved.m_Node}})},
            {"containers", Json(Json::Array{std::move(container)})},
            {"volumes", Json(Json::Array{Json(Json::Object{{"name", "descriptor"},
                                                           {"configMap", Json(Json::Object{{"name", configMap}})}})})},
        });
        Json deployment(Json::Object{
            {"apiVersion", "apps/v1"},
            {"kind", "Deployment"},
            {"metadata", Metadata(name, nameSpace, resolved)},
            {"spec", Json(Json::Object{
                         {"replicas", 1},
                         {"selector", Json(Json::Object{
                                          {"matchLabels", Json(Json::Object{{"app.kubernetes.io/instance", name}})}})},
                         {"template", Json(Json::Object{{"metadata", Json(Json::Object{{"labels", Labels(resolved)}})},
                                                        {"spec", std::move(podSpec)}})},
                     })},
        });

        Json specs = Descriptor::ToJson(descriptor)["specs"];
        Json resource(Json::Object{
            {"apiVersion", "mlops.empy.io/v1"},
            {"kind", "RichMLService"},
            {"metadata", Metadata(name, nameSpace, resolved)},
            {"spec", std::move(specs)},
        });

        return header + eOps::Detail::WriteYaml(configDocument) + "---\n" + eOps::Detail::WriteYaml(deployment) +
               "---\n" + eOps::Detail::WriteYaml(resource);
    }
} // namespace eOps::Substrate
