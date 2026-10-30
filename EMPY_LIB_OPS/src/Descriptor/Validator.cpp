#include <eOps/Descriptor/Validator.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Descriptor/Cron.h>
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <cmath>
#include <algorithm>

namespace eOps::Descriptor
{
    namespace
    {
        constexpr const char *STATISTICS[] = {"p50", "p90", "p95", "p99", "mean", "max"};

        std::string Number(double value)
        {
            return std::isfinite(value) ? Json(value).Dump() : std::string("non-finite");
        }

        bool Contains(const std::vector<std::string> &items, const std::string &item)
        {
            return std::find(items.begin(), items.end(), item) != items.end();
        }

        // MAJOR.MINOR.PATCH, decimal without leading zeros
        bool SemanticVersion(std::string_view text)
        {
            std::vector<std::string> parts = Split(text, '.');
            if (parts.size() != 3)
            {
                return false;
            }
            for (const std::string &part : parts)
            {
                if (part.empty() || part.size() > 9 || (part.size() > 1 && part.front() == '0'))
                {
                    return false;
                }
                for (char c : part)
                {
                    if (c < '0' || c > '9')
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        class Checker
        {
        public:
            Checker(const RichMLService &service, const ValidationOptions &options)
                : m_Service(service), m_Options(options)
            {
            }

            std::vector<Issue> Run()
            {
                Identity();
                Topics();
                if (m_Service.m_Inference)
                {
                    Inference(*m_Service.m_Inference);
                }
                if (m_Service.m_Training)
                {
                    Training(*m_Service.m_Training);
                }
                if (!m_Service.m_Inference && !m_Service.m_Training)
                {
                    Error("specs", "at least one of inference or training must be declared");
                }
                Constraints();
                Lifecycle();
                return std::move(m_Issues);
            }

        private:
            void Add(std::string path, std::string message, Severity severity)
            {
                m_Issues.push_back(Issue{std::move(path), std::move(message), severity});
            }

            void Error(std::string path, std::string message)
            {
                Add(std::move(path), std::move(message), Severity::Error);
            }

            void Warning(std::string path, std::string message)
            {
                Add(std::move(path), std::move(message), Severity::Warning);
            }

            void Identity()
            {
                if (m_Service.m_ApiVersion != "mlops/v1")
                {
                    Error("apiVersion", "must be 'mlops/v1' (got '" + m_Service.m_ApiVersion + "')");
                }
                if (m_Service.m_Kind != "RichMLService")
                {
                    Error("kind", "must be 'RichMLService' (got '" + m_Service.m_Kind + "')");
                }
                if (!IsDnsLabel(m_Service.m_Metadata.m_Name))
                {
                    Error("metadata.name", "'" + m_Service.m_Metadata.m_Name +
                                               "' is not a DNS label (1-63 lowercase alphanumerics or '-')");
                }
                if (!SemanticVersion(m_Service.m_Metadata.m_Version))
                {
                    Error("metadata.version",
                          "'" + m_Service.m_Metadata.m_Version + "' is not a MAJOR.MINOR.PATCH version");
                }
            }

            // rules shared by every service topic: no reserved '$' space, no MLOps control root, tenant prefix
            void TopicPlacement(const std::string &path, const std::string &topic, bool publication)
            {
                std::string first = TopicLevel(topic, 0);
                if (!topic.empty() && topic.front() == '$' && (publication || !topic.starts_with("$share/")))
                {
                    Error(path, "'" + topic + "' is under the reserved '$' topic space");
                }
                else if (!first.empty() && first.front() == '$')
                {
                    Error(path, "'" + topic + "' is under the reserved '$' topic space");
                }
                if (first == m_Options.m_TopicRoot && !m_Options.m_TopicRoot.empty())
                {
                    Error(path, "'" + topic + "' is under the MLOps control root '" + m_Options.m_TopicRoot + "/'");
                }
                else if ((first == "+" || first == "#") && !m_Options.m_TopicRoot.empty())
                {
                    std::string message =
                        "'" + topic + "' also matches the MLOps control root '" + m_Options.m_TopicRoot + "/'";
                    if (publication)
                    {
                        Error(path, message);
                    }
                    else
                    {
                        Warning(path, message);
                    }
                }
                // a shared subscription is judged by the filter after "$share/<group>/"
                std::string_view filter = topic;
                if (!publication && filter.starts_with("$share/"))
                {
                    size_t group = filter.find('/', 7);
                    filter = group == std::string_view::npos ? std::string_view() : filter.substr(group + 1);
                }
                const std::string &tenant = m_Service.m_Constraints.m_Security.m_Tenant;
                if (!tenant.empty() && !filter.starts_with(tenant + "/"))
                {
                    Error(path, "'" + topic + "' is outside the tenant root '" + tenant + "/'");
                }
            }

            void Qos(const std::string &path, uint8_t qos)
            {
                if (qos > 1)
                {
                    Error(path, "qos must be 0 or 1");
                }
            }

            void Topics()
            {
                const auto &observables = m_Service.m_Observables;
                const auto &actionables = m_Service.m_Actionables;
                if (m_Service.m_Inference && observables.empty())
                {
                    Error("specs.observables", "inference needs at least one observable");
                }
                for (size_t i = 0; i < observables.size(); ++i)
                {
                    std::string path = "specs.observables[" + std::to_string(i) + "]";
                    const std::string &name = observables[i].m_Name;
                    if (!ValidTopic(name, true))
                    {
                        Error(path + ".name", "'" + name + "' is not a valid MMQP topic filter");
                        continue;
                    }
                    TopicPlacement(path + ".name", name, false);
                    Qos(path + ".qos", observables[i].m_Qos);
                    for (size_t j = 0; j < i; ++j)
                    {
                        if (observables[j].m_Name == name)
                        {
                            Error(path + ".name", "duplicate observable '" + name + "'");
                        }
                    }
                }
                for (size_t i = 0; i < actionables.size(); ++i)
                {
                    std::string path = "specs.actionables[" + std::to_string(i) + "]";
                    const Actionable &actionable = actionables[i];
                    const std::string &name = actionable.m_Name;
                    if (!ValidTopic(name, true) || name.starts_with("$share/"))
                    {
                        Error(path + ".name", "'" + name + "' is not a valid MMQP publication topic or domain");
                        continue;
                    }
                    TopicPlacement(path + ".name", name, true);
                    Qos(path + ".qos", actionable.m_Qos);
                    for (size_t j = 0; j < i; ++j)
                    {
                        if (actionables[j].m_Name == name)
                        {
                            Error(path + ".name", "duplicate actionable '" + name + "'");
                        }
                    }
                    if (actionable.m_Min && !std::isfinite(*actionable.m_Min))
                    {
                        Error(path + ".min", "must be finite");
                    }
                    if (actionable.m_Max && !std::isfinite(*actionable.m_Max))
                    {
                        Error(path + ".max", "must be finite");
                    }
                    if (actionable.m_Min && actionable.m_Max && *actionable.m_Min > *actionable.m_Max)
                    {
                        Error(path, "min " + Number(*actionable.m_Min) + " exceeds max " + Number(*actionable.m_Max));
                    }
                    for (size_t j = 0; j < observables.size(); ++j)
                    {
                        if (FiltersOverlap(observables[j].m_Name, name))
                        {
                            Warning(path + ".name", "actionable '" + name + "' overlaps observable '" +
                                                        observables[j].m_Name + "' (feedback loop)");
                        }
                    }
                }
            }

            void Resources(const std::string &path, const Descriptor::Resources &resources)
            {
                if (!(resources.m_Cpu > 0) || !std::isfinite(resources.m_Cpu))
                {
                    Error(path + ".cpu", "cpu must be greater than zero");
                }
                if (resources.m_Memory == 0)
                {
                    Error(path + ".memory", "memory must be greater than zero");
                }
                if (!resources.m_Accelerator.empty())
                {
                    if (!Contains(m_Options.m_Accelerators, resources.m_Accelerator))
                    {
                        Error(path + ".accelerator", "unknown accelerator '" + resources.m_Accelerator + "'");
                    }
                    if (resources.m_AcceleratorCount < 1)
                    {
                        Error(path + ".acceleratorCount", "must be at least 1 when an accelerator is requested");
                    }
                }
                else if (resources.m_AcceleratorCount > 0)
                {
                    Warning(path + ".acceleratorCount", "ignored without an accelerator kind");
                }
            }

            void Regions(const std::string &path, const std::vector<std::string> &regions)
            {
                if (regions.empty())
                {
                    Error(path, "at least one region is required");
                }
                for (size_t i = 0; i < regions.size(); ++i)
                {
                    std::string at = path + "[" + std::to_string(i) + "]";
                    if (!m_Options.m_Regions.empty() && !Contains(m_Options.m_Regions, regions[i]))
                    {
                        Error(at, "unknown region '" + regions[i] + "'");
                    }
                    if (std::find(regions.begin(), regions.begin() + static_cast<std::ptrdiff_t>(i), regions[i]) !=
                        regions.begin() + static_cast<std::ptrdiff_t>(i))
                    {
                        Warning(at, "duplicate region '" + regions[i] + "'");
                    }
                }
            }

            void Inference(const InferenceSpec &inference)
            {
                Resources("specs.inference.resources", inference.m_Resources);
                Regions("specs.inference.regions", inference.m_Regions);
                if (inference.m_Replicas < 1)
                {
                    Error("specs.inference.replicas", "must be at least 1");
                }
                if (inference.m_Endpoint.empty() || inference.m_Endpoint.front() != '/')
                {
                    Warning("specs.inference.endpoint", "endpoint should be an absolute path such as /v1/predict");
                }
                if (inference.m_Latency)
                {
                    const LatencyObjective &latency = *inference.m_Latency;
                    bool known = false;
                    for (const char *statistic : STATISTICS)
                    {
                        known = known || latency.m_Statistic == statistic;
                    }
                    if (!known)
                    {
                        Error("specs.inference.latency.statistic",
                              "'" + latency.m_Statistic + "' is not one of p50, p90, p95, p99, mean, max");
                    }
                    if (latency.m_WindowSeconds < 1)
                    {
                        Error("specs.inference.latency.windowSeconds", "must be at least 1");
                    }
                    if (!(latency.m_MaxMs > 0) || !std::isfinite(latency.m_MaxMs))
                    {
                        Error("specs.inference.latency.maxMs", "must be greater than zero");
                    }
                }
                // the actuation topic is what predictions drive; it must be a declared, concrete actionable
                bool concrete = std::any_of(m_Service.m_Actionables.begin(), m_Service.m_Actionables.end(),
                                            [](const Actionable &a) { return !IsFilter(a.m_Name); });
                if (!m_Service.m_Actionables.empty() || !inference.m_Actuation.empty())
                {
                    std::string actuation = m_Service.ActuationTopic();
                    std::string path = "specs.inference.actuation";
                    if (actuation.empty())
                    {
                        Error(path,
                              concrete ? "no actuation topic" : "wildcard actionables need an explicit actuation");
                    }
                    else if (!ValidTopic(actuation, false) || IsFilter(actuation))
                    {
                        Error(path, "'" + actuation + "' must be a concrete topic");
                    }
                    else if (!m_Service.FindActionable(actuation))
                    {
                        Error(path, "'" + actuation + "' is not covered by a declared actionable");
                    }
                }
            }

            void Training(const TrainingSpec &training)
            {
                Resources("specs.training.resources", training.m_Resources);
                Regions("specs.training.regions", training.m_Regions);
                if (!training.m_Schedule.empty())
                {
                    try
                    {
                        CronSchedule::Parse(training.m_Schedule);
                    }
                    catch (const std::exception &error)
                    {
                        Error("specs.training.schedule", error.what());
                    }
                }
                if (Trim(training.m_Dataset).empty())
                {
                    Error("specs.training.dataset", "a dataset is required for training");
                }
                if (!training.m_Label.empty())
                {
                    if (!ValidTopic(training.m_Label, false) || IsFilter(training.m_Label))
                    {
                        Error("specs.training.label", "'" + training.m_Label + "' must be a concrete MMQP topic");
                    }
                    else
                    {
                        TopicPlacement("specs.training.label", training.m_Label, false);
                    }
                }
                if (!(training.m_Holdout > 0 && training.m_Holdout < 1))
                {
                    Error("specs.training.holdout", "must lie strictly between 0 and 1");
                }
                if (training.m_Window < 1)
                {
                    Error("specs.training.window", "must be at least 1");
                }
                if (training.m_MinSamples > training.m_Window)
                {
                    Error("specs.training.minSamples", "exceeds the training window");
                }
                size_t features = m_Service.m_Observables.size();
                if (training.m_MinSamples < features + 1)
                {
                    Warning("specs.training.minSamples",
                            "fewer than features + 1 samples (" + std::to_string(features + 1) + ") cannot be fitted");
                }
            }

            void Constraints()
            {
                const Descriptor::Constraints &constraints = m_Service.m_Constraints;
                const std::string &tenant = constraints.m_Security.m_Tenant;
                if (!tenant.empty() && (!ValidTopic(tenant, false) || IsFilter(tenant) ||
                                        tenant.find('/') != std::string::npos || tenant.front() == '$'))
                {
                    Error("specs.constraints.security.tenant", "'" + tenant + "' must be a single topic level");
                }
                for (const std::string &region : constraints.m_Placement.m_Prefer)
                {
                    if (Contains(constraints.m_Placement.m_Avoid, region))
                    {
                        Warning("specs.constraints.placement", "region '" + region + "' is both preferred and avoided");
                    }
                }
                double hysteresis = constraints.m_Placement.m_Hysteresis;
                if (!(hysteresis >= 0 && hysteresis <= 1))
                {
                    Error("specs.constraints.placement.hysteresis", "must lie in [0, 1]");
                }
            }

            void Fraction(const std::string &path, double value)
            {
                if (!(value >= 0 && value <= 1))
                {
                    Error(path, "must lie in [0, 1] (got " + Number(value) + ")");
                }
            }

            void Lifecycle()
            {
                const LifecyclePolicy &policy = m_Service.m_Lifecycle;
                Fraction("specs.lifecycle.accuracyThreshold", policy.m_AccuracyThreshold);
                Fraction("specs.lifecycle.minImprovement", policy.m_MinImprovement);
                if (policy.m_EvaluationWindow < 1)
                {
                    Error("specs.lifecycle.evaluationWindow", "must be at least 1");
                }
                if (policy.m_MinSamples > policy.m_EvaluationWindow)
                {
                    Warning("specs.lifecycle.minSamples", "exceeds the evaluation window, drift is never detected");
                }
                if (!(policy.m_Tolerance >= 0) || !std::isfinite(policy.m_Tolerance))
                {
                    Error("specs.lifecycle.tolerance", "must be a finite non-negative number");
                }
            }

            const RichMLService &m_Service;
            const ValidationOptions &m_Options;
            std::vector<Issue> m_Issues;
        };
    } // namespace

    const char *SeverityName(Severity severity)
    {
        switch (severity)
        {
        case Severity::Info:
            return "info";
        case Severity::Warning:
            return "warning";
        default:
            return "error";
        }
    }

    std::vector<Issue> Validate(const RichMLService &service, const ValidationOptions &options)
    {
        return Checker(service, options).Run();
    }

    std::vector<Issue> ValidateDocument(const Json &document, const RichMLService &service,
                                        const ValidationOptions &options)
    {
        std::vector<Issue> issues = Validate(service, options);
        for (std::string &path : UnknownKeys(document))
        {
            issues.push_back(Issue{std::move(path), "unknown key (ignored)", Severity::Warning});
        }
        return issues;
    }

    bool HasErrors(const std::vector<Issue> &issues)
    {
        return std::any_of(issues.begin(), issues.end(),
                           [](const Issue &issue) { return issue.m_Severity == Severity::Error; });
    }

    Json ToJson(const std::vector<Issue> &issues)
    {
        Json array = Json::MakeArray();
        for (const Issue &issue : issues)
        {
            array.Push(Json(Json::Object{
                {"path", issue.m_Path}, {"message", issue.m_Message}, {"severity", SeverityName(issue.m_Severity)}}));
        }
        return array;
    }
} // namespace eOps::Descriptor
