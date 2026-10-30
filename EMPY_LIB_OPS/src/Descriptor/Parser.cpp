#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include "../Common/Yaml.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <initializer_list>
#include "../Common/JsonRead.h"

namespace eOps::Descriptor
{
    namespace
    {
        std::string Child(const std::string &path, std::string_view key)
        {
            return path.empty() ? std::string(key) : path + "." + std::string(key);
        }

        std::string Index(const std::string &path, size_t index)
        {
            return path + "[" + std::to_string(index) + "]";
        }

        // typed access to the JSON form of a descriptor; every failure names origin and dotted path
        class Reader
        {
        public:
            explicit Reader(std::string origin) : m_Origin(std::move(origin))
            {
            }

            [[noreturn]] void Fail(const std::string &path, const std::string &message) const
            {
                throw DescriptorError(m_Origin + ": " + (path.empty() ? std::string("<root>") : path) + ": " + message);
            }

            // member value, nullptr when absent or null
            const Json *Member(const Json &object, std::string_view key) const
            {
                const Json *value = object.Find(key);
                return value && !value->IsNull() ? value : nullptr;
            }

            const Json &Mapping(const Json &value, const std::string &path) const
            {
                if (!value.IsObject())
                {
                    Fail(path, "expected a mapping");
                }
                return value;
            }

            std::string String(const Json &value, const std::string &path) const
            {
                if (value.IsString())
                {
                    return value.AsString();
                }
                if (value.IsNumber())
                {
                    return value.Dump();
                }
                if (value.IsBool())
                {
                    return value.AsBool() ? "true" : "false";
                }
                Fail(path, "expected a string");
            }

            double Number(const Json &value, const std::string &path) const
            {
                if (value.IsNumber())
                {
                    return value.AsNumber();
                }
                if (value.IsString())
                {
                    if (std::optional<double> number = ParseNumber(Trim(value.AsString())))
                    {
                        return *number;
                    }
                }
                Fail(path, "expected a number");
            }

            bool Bool(const Json &value, const std::string &path) const
            {
                if (!value.IsBool())
                {
                    Fail(path, "expected true or false");
                }
                return value.AsBool();
            }

            int64_t Integer(const Json &value, const std::string &path, int64_t minimum, int64_t maximum) const
            {
                double number = Number(value, path);
                if (number != std::floor(number))
                {
                    Fail(path, "expected an integer");
                }
                if (number < static_cast<double>(minimum) || number > static_cast<double>(maximum))
                {
                    Fail(path, "integer outside [" + std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
                }
                return static_cast<int64_t>(number);
            }

            // a list of scalars; a single scalar is accepted as a one-element list
            std::vector<std::string> Strings(const Json &value, const std::string &path) const
            {
                std::vector<std::string> items;
                if (!value.IsArray())
                {
                    items.push_back(String(value, path));
                    return items;
                }
                for (size_t i = 0; i < value.Size(); ++i)
                {
                    items.push_back(String(value[i], Index(path, i)));
                }
                return items;
            }

            // read optional member `key` of `object` through `read` into `target`
            template <typename _Read>
            void Optional(const Json &object, std::string_view key, const std::string &path, _Read &&read) const
            {
                if (const Json *value = Member(object, key))
                {
                    read(*value, Child(path, key));
                }
            }

        private:
            std::string m_Origin;
        };

        Resources DecodeResources(const Reader &reader, const Json &value, const std::string &path)
        {
            Resources resources;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "cpu", path,
                            [&](const Json &cpu, const std::string &at)
                            {
                                try
                                {
                                    resources.m_Cpu =
                                        cpu.IsString() ? ParseCpu(cpu.AsString()) : reader.Number(cpu, at);
                                }
                                catch (const std::invalid_argument &error)
                                {
                                    reader.Fail(at, error.what());
                                }
                                if (!std::isfinite(resources.m_Cpu) || resources.m_Cpu < 0)
                                {
                                    reader.Fail(at, "cpu must be a non-negative quantity");
                                }
                            });
            reader.Optional(object, "memory", path,
                            [&](const Json &memory, const std::string &at)
                            {
                                try
                                {
                                    resources.m_Memory = memory.IsString()
                                                             ? ParseMemory(memory.AsString())
                                                             : static_cast<uint64_t>(reader.Integer(
                                                                   memory, at, 0, std::numeric_limits<int64_t>::max()));
                                }
                                catch (const std::invalid_argument &error)
                                {
                                    reader.Fail(at, error.what());
                                }
                            });
            reader.Optional(object, "accelerator", path, [&](const Json &accelerator, const std::string &at)
                            { resources.m_Accelerator = Lower(Trim(reader.String(accelerator, at))); });
            reader.Optional(
                object, "acceleratorCount", path, [&](const Json &count, const std::string &at)
                { resources.m_AcceleratorCount = static_cast<uint32_t>(reader.Integer(count, at, 0, 1024)); });
            if (!resources.m_Accelerator.empty() && !reader.Member(object, "acceleratorCount"))
            {
                resources.m_AcceleratorCount = 1;
            }
            return resources;
        }

        uint8_t DecodeQos(const Reader &reader, const Json &value, const std::string &path)
        {
            return static_cast<uint8_t>(reader.Integer(value, path, 0, 255));
        }

        Observable DecodeObservable(const Reader &reader, const Json &value, const std::string &path)
        {
            Observable observable;
            if (value.IsString())
            {
                observable.m_Name = value.AsString();
                return observable;
            }
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "name", path,
                            [&](const Json &v, const std::string &at) { observable.m_Name = reader.String(v, at); });
            reader.Optional(object, "unit", path,
                            [&](const Json &v, const std::string &at) { observable.m_Unit = reader.String(v, at); });
            reader.Optional(object, "description", path, [&](const Json &v, const std::string &at)
                            { observable.m_Description = reader.String(v, at); });
            reader.Optional(object, "qos", path,
                            [&](const Json &v, const std::string &at) { observable.m_Qos = DecodeQos(reader, v, at); });
            return observable;
        }

        Actionable DecodeActionable(const Reader &reader, const Json &value, const std::string &path)
        {
            Actionable actionable;
            if (value.IsString())
            {
                actionable.m_Name = value.AsString();
                return actionable;
            }
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "name", path,
                            [&](const Json &v, const std::string &at) { actionable.m_Name = reader.String(v, at); });
            reader.Optional(object, "unit", path,
                            [&](const Json &v, const std::string &at) { actionable.m_Unit = reader.String(v, at); });
            reader.Optional(object, "min", path,
                            [&](const Json &v, const std::string &at) { actionable.m_Min = reader.Number(v, at); });
            reader.Optional(object, "max", path,
                            [&](const Json &v, const std::string &at) { actionable.m_Max = reader.Number(v, at); });
            reader.Optional(
                object, "maxRatePerMinute", path, [&](const Json &v, const std::string &at)
                { actionable.m_MaxRatePerMinute = static_cast<uint32_t>(reader.Integer(v, at, 0, UINT32_MAX)); });
            reader.Optional(object, "description", path, [&](const Json &v, const std::string &at)
                            { actionable.m_Description = reader.String(v, at); });
            reader.Optional(object, "qos", path,
                            [&](const Json &v, const std::string &at) { actionable.m_Qos = DecodeQos(reader, v, at); });
            return actionable;
        }

        LatencyObjective DecodeLatency(const Reader &reader, const Json &value, const std::string &path)
        {
            LatencyObjective latency;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "observable", path,
                            [&](const Json &v, const std::string &at) { latency.m_Observable = reader.String(v, at); });
            reader.Optional(object, "path", path,
                            [&](const Json &v, const std::string &at) { latency.m_Path = reader.String(v, at); });
            reader.Optional(object, "statistic", path, [&](const Json &v, const std::string &at)
                            { latency.m_Statistic = Lower(Trim(reader.String(v, at))); });
            reader.Optional(object, "windowSeconds", path, [&](const Json &v, const std::string &at)
                            { latency.m_WindowSeconds = static_cast<uint32_t>(reader.Integer(v, at, 0, UINT32_MAX)); });
            reader.Optional(object, "maxMs", path,
                            [&](const Json &v, const std::string &at) { latency.m_MaxMs = reader.Number(v, at); });
            return latency;
        }

        InferenceSpec DecodeInference(const Reader &reader, const Json &value, const std::string &path)
        {
            InferenceSpec inference;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "resources", path, [&](const Json &v, const std::string &at)
                            { inference.m_Resources = DecodeResources(reader, v, at); });
            reader.Optional(object, "endpoint", path,
                            [&](const Json &v, const std::string &at) { inference.m_Endpoint = reader.String(v, at); });
            reader.Optional(object, "regions", path,
                            [&](const Json &v, const std::string &at) { inference.m_Regions = reader.Strings(v, at); });
            reader.Optional(object, "replicas", path, [&](const Json &v, const std::string &at)
                            { inference.m_Replicas = static_cast<uint32_t>(reader.Integer(v, at, 0, 1024)); });
            reader.Optional(object, "image", path,
                            [&](const Json &v, const std::string &at) { inference.m_Image = reader.String(v, at); });
            reader.Optional(object, "actuation", path, [&](const Json &v, const std::string &at)
                            { inference.m_Actuation = reader.String(v, at); });
            reader.Optional(object, "latency", path, [&](const Json &v, const std::string &at)
                            { inference.m_Latency = DecodeLatency(reader, v, at); });
            return inference;
        }

        TrainingSpec DecodeTraining(const Reader &reader, const Json &value, const std::string &path)
        {
            TrainingSpec training;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "resources", path, [&](const Json &v, const std::string &at)
                            { training.m_Resources = DecodeResources(reader, v, at); });
            reader.Optional(object, "regions", path,
                            [&](const Json &v, const std::string &at) { training.m_Regions = reader.Strings(v, at); });
            reader.Optional(object, "schedule", path, [&](const Json &v, const std::string &at)
                            { training.m_Schedule = Trim(reader.String(v, at)); });
            reader.Optional(object, "dataset", path,
                            [&](const Json &v, const std::string &at) { training.m_Dataset = reader.String(v, at); });
            reader.Optional(object, "label", path,
                            [&](const Json &v, const std::string &at) { training.m_Label = reader.String(v, at); });
            reader.Optional(object, "window", path, [&](const Json &v, const std::string &at)
                            { training.m_Window = static_cast<uint32_t>(reader.Integer(v, at, 0, 1 << 24)); });
            reader.Optional(object, "minSamples", path, [&](const Json &v, const std::string &at)
                            { training.m_MinSamples = static_cast<uint32_t>(reader.Integer(v, at, 0, 1 << 24)); });
            reader.Optional(object, "holdout", path,
                            [&](const Json &v, const std::string &at) { training.m_Holdout = reader.Number(v, at); });
            reader.Optional(object, "image", path,
                            [&](const Json &v, const std::string &at) { training.m_Image = reader.String(v, at); });
            return training;
        }

        Constraints DecodeConstraints(const Reader &reader, const Json &value, const std::string &path)
        {
            Constraints constraints;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(
                object, "priority", path, [&](const Json &v, const std::string &at)
                { constraints.m_Priority = static_cast<int32_t>(reader.Integer(v, at, INT32_MIN, INT32_MAX)); });
            reader.Optional(object, "owner", path,
                            [&](const Json &v, const std::string &at) { constraints.m_Owner = reader.String(v, at); });
            reader.Optional(object, "objectives", path, [&](const Json &v, const std::string &at)
                            { constraints.m_Objectives = reader.Strings(v, at); });
            reader.Optional(object, "security", path,
                            [&](const Json &v, const std::string &at)
                            {
                                const Json &security = reader.Mapping(v, at);
                                SecurityConstraints &out = constraints.m_Security;
                                reader.Optional(security, "encryption", at, [&](const Json &e, const std::string &p)
                                                { out.m_Encryption = reader.Bool(e, p); });
                                reader.Optional(security, "tenant", at, [&](const Json &e, const std::string &p)
                                                { out.m_Tenant = reader.String(e, p); });
                                reader.Optional(security, "identity", at, [&](const Json &e, const std::string &p)
                                                { out.m_Identity = reader.String(e, p); });
                            });
            reader.Optional(object, "placement", path,
                            [&](const Json &v, const std::string &at)
                            {
                                const Json &placement = reader.Mapping(v, at);
                                PlacementConstraints &out = constraints.m_Placement;
                                reader.Optional(placement, "prefer", at, [&](const Json &e, const std::string &p)
                                                { out.m_Prefer = reader.Strings(e, p); });
                                reader.Optional(placement, "avoid", at, [&](const Json &e, const std::string &p)
                                                { out.m_Avoid = reader.Strings(e, p); });
                                reader.Optional(placement, "nodes", at, [&](const Json &e, const std::string &p)
                                                { out.m_Nodes = reader.Strings(e, p); });
                                reader.Optional(placement, "allowRelocation", at,
                                                [&](const Json &e, const std::string &p)
                                                { out.m_AllowRelocation = reader.Bool(e, p); });
                                reader.Optional(placement, "hysteresis", at, [&](const Json &e, const std::string &p)
                                                { out.m_Hysteresis = reader.Number(e, p); });
                            });
            return constraints;
        }

        LifecyclePolicy DecodeLifecycle(const Reader &reader, const Json &value, const std::string &path)
        {
            LifecyclePolicy policy;
            const Json &object = reader.Mapping(value, path);
            reader.Optional(object, "accuracyThreshold", path, [&](const Json &v, const std::string &at)
                            { policy.m_AccuracyThreshold = reader.Number(v, at); });
            reader.Optional(object, "minImprovement", path, [&](const Json &v, const std::string &at)
                            { policy.m_MinImprovement = reader.Number(v, at); });
            reader.Optional(
                object, "cooldownSeconds", path, [&](const Json &v, const std::string &at)
                { policy.m_CooldownSeconds = static_cast<uint32_t>(reader.Integer(v, at, 0, UINT32_MAX)); });
            reader.Optional(object, "evaluationWindow", path, [&](const Json &v, const std::string &at)
                            { policy.m_EvaluationWindow = static_cast<uint32_t>(reader.Integer(v, at, 0, 1 << 24)); });
            reader.Optional(object, "minSamples", path, [&](const Json &v, const std::string &at)
                            { policy.m_MinSamples = static_cast<uint32_t>(reader.Integer(v, at, 0, 1 << 24)); });
            reader.Optional(object, "tolerance", path,
                            [&](const Json &v, const std::string &at) { policy.m_Tolerance = reader.Number(v, at); });
            reader.Optional(object, "autoRollback", path,
                            [&](const Json &v, const std::string &at) { policy.m_AutoRollback = reader.Bool(v, at); });
            return policy;
        }

        RichMLService Decode(const Json &document, const std::string &origin)
        {
            Reader reader(origin);
            const Json &root = reader.Mapping(document, "");
            RichMLService service;
            // identity fields have no default when parsing: an undeclared apiVersion/kind fails validation
            service.m_ApiVersion.clear();
            service.m_Kind.clear();
            reader.Optional(root, "apiVersion", "",
                            [&](const Json &v, const std::string &at) { service.m_ApiVersion = reader.String(v, at); });
            reader.Optional(root, "kind", "",
                            [&](const Json &v, const std::string &at) { service.m_Kind = reader.String(v, at); });
            reader.Optional(root, "metadata", "",
                            [&](const Json &v, const std::string &at)
                            {
                                const Json &metadata = reader.Mapping(v, at);
                                Metadata &out = service.m_Metadata;
                                reader.Optional(metadata, "name", at, [&](const Json &e, const std::string &p)
                                                { out.m_Name = reader.String(e, p); });
                                reader.Optional(metadata, "version", at, [&](const Json &e, const std::string &p)
                                                { out.m_Version = reader.String(e, p); });
                                reader.Optional(metadata, "description", at, [&](const Json &e, const std::string &p)
                                                { out.m_Description = reader.String(e, p); });
                                reader.Optional(metadata, "intent", at, [&](const Json &e, const std::string &p)
                                                { out.m_Intent = reader.String(e, p); });
                                reader.Optional(metadata, "labels", at,
                                                [&](const Json &e, const std::string &p)
                                                {
                                                    for (const auto &[key, label] : reader.Mapping(e, p).AsObject())
                                                    {
                                                        out.m_Labels.emplace_back(key,
                                                                                  reader.String(label, Child(p, key)));
                                                    }
                                                    // YAML mappings do not keep document order: labels are ordered by
                                                    // key
                                                    std::sort(out.m_Labels.begin(), out.m_Labels.end());
                                                });
                            });

            const Json *specs = reader.Member(root, "specs");
            std::string specsPath = "specs";
            if (const Json *spec = reader.Member(root, "spec"))
            {
                if (specs)
                {
                    reader.Fail("spec", "both 'specs' and 'spec' are present");
                }
                specs = spec;
                specsPath = "spec";
            }
            if (!specs)
            {
                return service;
            }
            const Json &body = reader.Mapping(*specs, specsPath);
            reader.Optional(body, "observables", specsPath,
                            [&](const Json &v, const std::string &at)
                            {
                                if (!v.IsArray())
                                {
                                    reader.Fail(at, "expected a list");
                                }
                                for (size_t i = 0; i < v.Size(); ++i)
                                {
                                    service.m_Observables.push_back(DecodeObservable(reader, v[i], Index(at, i)));
                                }
                            });
            reader.Optional(body, "actionables", specsPath,
                            [&](const Json &v, const std::string &at)
                            {
                                if (!v.IsArray())
                                {
                                    reader.Fail(at, "expected a list");
                                }
                                for (size_t i = 0; i < v.Size(); ++i)
                                {
                                    service.m_Actionables.push_back(DecodeActionable(reader, v[i], Index(at, i)));
                                }
                            });
            reader.Optional(body, "inference", specsPath, [&](const Json &v, const std::string &at)
                            { service.m_Inference = DecodeInference(reader, v, at); });
            reader.Optional(body, "training", specsPath, [&](const Json &v, const std::string &at)
                            { service.m_Training = DecodeTraining(reader, v, at); });
            reader.Optional(body, "constraints", specsPath, [&](const Json &v, const std::string &at)
                            { service.m_Constraints = DecodeConstraints(reader, v, at); });
            reader.Optional(body, "lifecycle", specsPath, [&](const Json &v, const std::string &at)
                            { service.m_Lifecycle = DecodeLifecycle(reader, v, at); });
            return service;
        }

        // ---- encoding (compact = omit extension fields at their defaults, used for YAML emission)

        using eOps::Detail::Strings;

        Json EncodeResources(const Resources &resources, bool compact)
        {
            Json object = Json::MakeObject();
            object.Set("cpu", FormatCpu(resources.m_Cpu));
            object.Set("memory", FormatMemory(resources.m_Memory));
            if (!compact || !resources.m_Accelerator.empty())
            {
                object.Set("accelerator", resources.m_Accelerator);
            }
            uint32_t implied = resources.m_Accelerator.empty() ? 0 : 1;
            if (!compact || resources.m_AcceleratorCount != implied)
            {
                object.Set("acceleratorCount", resources.m_AcceleratorCount);
            }
            return object;
        }

        Json Encode(const RichMLService &service, bool compact)
        {
            const RichMLService defaults;
            const InferenceSpec inferenceDefaults;
            const TrainingSpec trainingDefaults;
            Json root = Json::MakeObject();
            root.Set("apiVersion", service.m_ApiVersion);
            root.Set("kind", service.m_Kind);

            const Metadata &metadata = service.m_Metadata;
            Json meta = Json::MakeObject();
            meta.Set("name", metadata.m_Name);
            meta.Set("version", metadata.m_Version);
            if (!compact || !metadata.m_Description.empty())
            {
                meta.Set("description", metadata.m_Description);
            }
            if (!compact || !metadata.m_Intent.empty())
            {
                meta.Set("intent", metadata.m_Intent);
            }
            if (!compact || !metadata.m_Labels.empty())
            {
                std::vector<std::pair<std::string, std::string>> sorted = metadata.m_Labels;
                std::sort(sorted.begin(), sorted.end());
                Json labels = Json::MakeObject();
                for (const auto &[key, value] : sorted)
                {
                    labels.Set(key, value);
                }
                meta.Set("labels", std::move(labels));
            }
            root.Set("metadata", std::move(meta));

            Json specs = Json::MakeObject();
            Json observables = Json::MakeArray();
            for (const Observable &observable : service.m_Observables)
            {
                Json item = Json::MakeObject();
                item.Set("name", observable.m_Name);
                item.Set("unit", observable.m_Unit);
                if (!compact || !observable.m_Description.empty())
                {
                    item.Set("description", observable.m_Description);
                }
                if (!compact || observable.m_Qos != 1)
                {
                    item.Set("qos", static_cast<uint32_t>(observable.m_Qos));
                }
                observables.Push(std::move(item));
            }
            specs.Set("observables", std::move(observables));

            Json actionables = Json::MakeArray();
            for (const Actionable &actionable : service.m_Actionables)
            {
                Json item = Json::MakeObject();
                item.Set("name", actionable.m_Name);
                item.Set("unit", actionable.m_Unit);
                if (actionable.m_Min)
                {
                    item.Set("min", *actionable.m_Min);
                }
                if (actionable.m_Max)
                {
                    item.Set("max", *actionable.m_Max);
                }
                if (!compact || actionable.m_MaxRatePerMinute != 0)
                {
                    item.Set("maxRatePerMinute", actionable.m_MaxRatePerMinute);
                }
                if (!compact || !actionable.m_Description.empty())
                {
                    item.Set("description", actionable.m_Description);
                }
                if (!compact || actionable.m_Qos != 1)
                {
                    item.Set("qos", static_cast<uint32_t>(actionable.m_Qos));
                }
                actionables.Push(std::move(item));
            }
            specs.Set("actionables", std::move(actionables));

            if (const std::optional<InferenceSpec> &inference = service.m_Inference)
            {
                Json item = Json::MakeObject();
                item.Set("resources", EncodeResources(inference->m_Resources, compact));
                item.Set("endpoint", inference->m_Endpoint);
                item.Set("regions", Strings(inference->m_Regions));
                if (inference->m_Latency)
                {
                    const LatencyObjective &latency = *inference->m_Latency;
                    item.Set("latency", Json(Json::Object{{"observable", latency.m_Observable},
                                                          {"path", latency.m_Path},
                                                          {"statistic", latency.m_Statistic},
                                                          {"windowSeconds", latency.m_WindowSeconds},
                                                          {"maxMs", latency.m_MaxMs}}));
                }
                if (!compact || inference->m_Replicas != inferenceDefaults.m_Replicas)
                {
                    item.Set("replicas", inference->m_Replicas);
                }
                if (!compact || !inference->m_Image.empty())
                {
                    item.Set("image", inference->m_Image);
                }
                if (!compact || !inference->m_Actuation.empty())
                {
                    item.Set("actuation", inference->m_Actuation);
                }
                specs.Set("inference", std::move(item));
            }

            if (const std::optional<TrainingSpec> &training = service.m_Training)
            {
                Json item = Json::MakeObject();
                item.Set("resources", EncodeResources(training->m_Resources, compact));
                item.Set("regions", Strings(training->m_Regions));
                if (!compact || !training->m_Schedule.empty())
                {
                    item.Set("schedule", training->m_Schedule);
                }
                item.Set("dataset", training->m_Dataset);
                if (!compact || !training->m_Label.empty())
                {
                    item.Set("label", training->m_Label);
                }
                if (!compact || training->m_Window != trainingDefaults.m_Window)
                {
                    item.Set("window", training->m_Window);
                }
                if (!compact || training->m_MinSamples != trainingDefaults.m_MinSamples)
                {
                    item.Set("minSamples", training->m_MinSamples);
                }
                if (!compact || training->m_Holdout != trainingDefaults.m_Holdout)
                {
                    item.Set("holdout", training->m_Holdout);
                }
                if (!compact || !training->m_Image.empty())
                {
                    item.Set("image", training->m_Image);
                }
                specs.Set("training", std::move(item));
            }

            const Constraints &constraints = service.m_Constraints;
            const Constraints &noConstraints = defaults.m_Constraints;
            Json limits = Json::MakeObject();
            if (!compact || constraints.m_Priority != noConstraints.m_Priority)
            {
                limits.Set("priority", constraints.m_Priority);
            }
            if (!compact || !constraints.m_Owner.empty())
            {
                limits.Set("owner", constraints.m_Owner);
            }
            if (!compact || !constraints.m_Objectives.empty())
            {
                limits.Set("objectives", Strings(constraints.m_Objectives));
            }
            const SecurityConstraints &security = constraints.m_Security;
            Json secure = Json::MakeObject();
            if (!compact || security.m_Encryption)
            {
                secure.Set("encryption", security.m_Encryption);
            }
            if (!compact || !security.m_Tenant.empty())
            {
                secure.Set("tenant", security.m_Tenant);
            }
            if (!compact || !security.m_Identity.empty())
            {
                secure.Set("identity", security.m_Identity);
            }
            if (!compact || secure.Size() > 0)
            {
                limits.Set("security", std::move(secure));
            }
            const PlacementConstraints &placement = constraints.m_Placement;
            const PlacementConstraints &noPlacement = noConstraints.m_Placement;
            Json place = Json::MakeObject();
            if (!compact || !placement.m_Prefer.empty())
            {
                place.Set("prefer", Strings(placement.m_Prefer));
            }
            if (!compact || !placement.m_Avoid.empty())
            {
                place.Set("avoid", Strings(placement.m_Avoid));
            }
            if (!compact || !placement.m_Nodes.empty())
            {
                place.Set("nodes", Strings(placement.m_Nodes));
            }
            if (!compact || placement.m_AllowRelocation != noPlacement.m_AllowRelocation)
            {
                place.Set("allowRelocation", placement.m_AllowRelocation);
            }
            if (!compact || placement.m_Hysteresis != noPlacement.m_Hysteresis)
            {
                place.Set("hysteresis", placement.m_Hysteresis);
            }
            if (!compact || place.Size() > 0)
            {
                limits.Set("placement", std::move(place));
            }
            if (!compact || limits.Size() > 0)
            {
                specs.Set("constraints", std::move(limits));
            }

            const LifecyclePolicy &policy = service.m_Lifecycle;
            const LifecyclePolicy &noPolicy = defaults.m_Lifecycle;
            Json lifecycle = Json::MakeObject();
            lifecycle.Set("accuracyThreshold", policy.m_AccuracyThreshold);
            lifecycle.Set("minImprovement", policy.m_MinImprovement);
            lifecycle.Set("cooldownSeconds", policy.m_CooldownSeconds);
            if (!compact || policy.m_EvaluationWindow != noPolicy.m_EvaluationWindow)
            {
                lifecycle.Set("evaluationWindow", policy.m_EvaluationWindow);
            }
            if (!compact || policy.m_MinSamples != noPolicy.m_MinSamples)
            {
                lifecycle.Set("minSamples", policy.m_MinSamples);
            }
            if (!compact || policy.m_Tolerance != noPolicy.m_Tolerance)
            {
                lifecycle.Set("tolerance", policy.m_Tolerance);
            }
            if (!compact || policy.m_AutoRollback != noPolicy.m_AutoRollback)
            {
                lifecycle.Set("autoRollback", policy.m_AutoRollback);
            }
            specs.Set("lifecycle", std::move(lifecycle));
            root.Set("specs", std::move(specs));
            return root;
        }

        // ---- unknown keys

        using Keys = std::initializer_list<std::string_view>;

        // report members of `object` outside `known`; returns false when `object` is not a mapping
        bool Unknown(const Json *object, const std::string &path, Keys known, std::vector<std::string> &out)
        {
            if (!object || !object->IsObject())
            {
                return false;
            }
            for (const auto &member : object->AsObject())
            {
                bool found = false;
                for (std::string_view key : known)
                {
                    found = found || key == member.first;
                }
                if (!found)
                {
                    out.push_back(Child(path, member.first));
                }
            }
            return true;
        }

        void UnknownResources(const Json *object, const std::string &path, std::vector<std::string> &out)
        {
            Unknown(object, path, {"cpu", "memory", "accelerator", "acceleratorCount"}, out);
        }

        template <typename _Item>
        void UnknownItems(const Json *list, const std::string &path, std::vector<std::string> &out, _Item &&item)
        {
            if (list && list->IsArray())
            {
                for (size_t i = 0; i < list->Size(); ++i)
                {
                    item(&(*list)[i], Index(path, i));
                }
            }
        }
    } // namespace

    RichMLService ParseDescriptor(std::string_view yaml, const std::string &origin)
    {
        Json document;
        try
        {
            document = FromYaml(eCore::YamlNode::Parse(yaml, origin));
        }
        catch (const std::exception &error)
        {
            throw DescriptorError(error.what());
        }
        return Decode(document, origin);
    }

    RichMLService LoadDescriptor(const std::filesystem::path &path)
    {
        std::string text;
        try
        {
            text = ReadFile(path);
        }
        catch (const std::exception &error)
        {
            throw DescriptorError(error.what());
        }
        return ParseDescriptor(text, path.string());
    }

    std::string EmitDescriptor(const RichMLService &service)
    {
        return eOps::Detail::WriteYaml(Encode(service, true));
    }

    Json ToJson(const RichMLService &service)
    {
        return Encode(service, false);
    }

    RichMLService FromJson(const Json &json)
    {
        return Decode(json, "<json>");
    }

    std::vector<std::string> UnknownKeys(const Json &document)
    {
        std::vector<std::string> out;
        if (!Unknown(&document, "", {"apiVersion", "kind", "metadata", "specs", "spec"}, out))
        {
            return out;
        }
        Unknown(document.Find("metadata"), "metadata", {"name", "version", "description", "intent", "labels"}, out);
        for (std::string_view name : {"specs", "spec"})
        {
            const Json *specs = document.Find(name);
            std::string path(name);
            if (!Unknown(specs, path,
                         {"observables", "actionables", "inference", "training", "constraints", "lifecycle"}, out))
            {
                continue;
            }
            UnknownItems(specs->Find("observables"), Child(path, "observables"), out,
                         [&](const Json *item, const std::string &at)
                         { Unknown(item, at, {"name", "unit", "description", "qos"}, out); });
            UnknownItems(
                specs->Find("actionables"), Child(path, "actionables"), out,
                [&](const Json *item, const std::string &at)
                { Unknown(item, at, {"name", "unit", "min", "max", "maxRatePerMinute", "description", "qos"}, out); });
            const Json *inference = specs->Find("inference");
            std::string inferencePath = Child(path, "inference");
            if (Unknown(inference, inferencePath,
                        {"resources", "endpoint", "regions", "replicas", "image", "actuation", "latency"}, out))
            {
                UnknownResources(inference->Find("resources"), Child(inferencePath, "resources"), out);
                Unknown(inference->Find("latency"), Child(inferencePath, "latency"),
                        {"observable", "path", "statistic", "windowSeconds", "maxMs"}, out);
            }
            const Json *training = specs->Find("training");
            std::string trainingPath = Child(path, "training");
            if (Unknown(training, trainingPath,
                        {"resources", "regions", "schedule", "dataset", "label", "window", "minSamples", "holdout",
                         "image"},
                        out))
            {
                UnknownResources(training->Find("resources"), Child(trainingPath, "resources"), out);
            }
            const Json *constraints = specs->Find("constraints");
            std::string constraintsPath = Child(path, "constraints");
            if (Unknown(constraints, constraintsPath, {"priority", "owner", "objectives", "security", "placement"},
                        out))
            {
                Unknown(constraints->Find("security"), Child(constraintsPath, "security"),
                        {"encryption", "tenant", "identity"}, out);
                Unknown(constraints->Find("placement"), Child(constraintsPath, "placement"),
                        {"prefer", "avoid", "nodes", "allowRelocation", "hysteresis"}, out);
            }
            Unknown(specs->Find("lifecycle"), Child(path, "lifecycle"),
                    {"accuracyThreshold", "minImprovement", "cooldownSeconds", "evaluationWindow", "minSamples",
                     "tolerance", "autoRollback"},
                    out);
        }
        return out;
    }
} // namespace eOps::Descriptor
