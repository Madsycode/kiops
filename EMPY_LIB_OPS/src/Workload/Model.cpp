#include <eOps/Workload/Model.h>
#include <eOps/Common/Text.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace eOps::Workload
{
    namespace
    {
        // solve A·w = b in place (A row-major n×n); throws when A is singular
        std::vector<double> Solve(std::vector<double> a, std::vector<double> b, size_t n)
        {
            double scale = 0;
            for (double value : a)
            {
                scale = std::max(scale, std::fabs(value));
            }
            double threshold = std::numeric_limits<double>::epsilon() * scale * static_cast<double>(n);
            for (size_t column = 0; column < n; ++column)
            {
                size_t pivot = column;
                for (size_t row = column + 1; row < n; ++row)
                {
                    if (std::fabs(a[row * n + column]) > std::fabs(a[pivot * n + column]))
                    {
                        pivot = row;
                    }
                }
                double lead = a[pivot * n + column];
                if (!std::isfinite(lead) || std::fabs(lead) <= threshold)
                {
                    throw std::invalid_argument("linear model: singular system (collinear or constant features)");
                }
                if (pivot != column)
                {
                    for (size_t k = 0; k < n; ++k)
                    {
                        std::swap(a[pivot * n + k], a[column * n + k]);
                    }
                    std::swap(b[pivot], b[column]);
                }
                for (size_t row = column + 1; row < n; ++row)
                {
                    double factor = a[row * n + column] / lead;
                    if (factor == 0)
                    {
                        continue;
                    }
                    for (size_t k = column; k < n; ++k)
                    {
                        a[row * n + k] -= factor * a[column * n + k];
                    }
                    b[row] -= factor * b[column];
                }
            }
            std::vector<double> w(n);
            for (size_t row = n; row-- > 0;)
            {
                double sum = b[row];
                for (size_t k = row + 1; k < n; ++k)
                {
                    sum -= a[row * n + k] * w[k];
                }
                w[row] = sum / a[row * n + row];
            }
            return w;
        }

        bool Finite(const std::vector<double> &values)
        {
            return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
        }
    } // namespace

    Json ModelArtifact::ToJson() const
    {
        Json features = Json::MakeArray();
        for (const std::string &feature : m_Features)
        {
            features.Push(Json(feature));
        }
        Json weights = Json::MakeArray();
        for (double weight : m_Weights)
        {
            weights.Push(Json(weight));
        }
        return Json(Json::Object{
            {"id", m_Id.empty() ? m_Service + "-v" + std::to_string(m_Version) : m_Id},
            {"service", m_Service},
            {"version", m_Version},
            {"features", std::move(features)},
            {"weights", std::move(weights)},
            {"accuracy", m_Accuracy},
            {"samples", m_Samples},
            {"trainedAt", m_TrainedAt},
            {"dataset", m_Dataset},
        });
    }

    ModelArtifact ModelArtifact::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("model artifact must be an object");
        }
        ModelArtifact artifact;
        artifact.m_Service = json.GetString("service");
        uint64_t version = json["version"].IsNull() ? 0 : json["version"].AsUint();
        uint64_t samples = json["samples"].IsNull() ? 0 : json["samples"].AsUint();
        if (version > std::numeric_limits<uint32_t>::max() || samples > std::numeric_limits<uint32_t>::max())
        {
            throw JsonError("model artifact version or samples out of range");
        }
        artifact.m_Version = static_cast<uint32_t>(version);
        artifact.m_Samples = static_cast<uint32_t>(samples);
        const Json &id = json["id"];
        artifact.m_Id = id.IsNull() ? artifact.m_Service + "-v" + std::to_string(artifact.m_Version) : id.AsString();
        const Json &features = json["features"];
        if (!features.IsNull())
        {
            for (const Json &feature : features.AsArray())
            {
                artifact.m_Features.push_back(feature.AsString());
            }
        }
        const Json &weights = json["weights"];
        if (!weights.IsNull())
        {
            for (const Json &weight : weights.AsArray())
            {
                artifact.m_Weights.push_back(weight.AsNumber());
            }
        }
        if (!json["accuracy"].IsNull())
        {
            artifact.m_Accuracy = json["accuracy"].AsNumber();
        }
        if (!json["trainedAt"].IsNull())
        {
            artifact.m_TrainedAt = json["trainedAt"].AsUint();
        }
        artifact.m_Dataset = json.GetString("dataset");
        return artifact;
    }

    ModelArtifact LinearModel::Fit(const std::vector<std::vector<double>> &x, const std::vector<double> &y,
                                   double ridge)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument("linear model: feature rows and labels differ in count");
        }
        if (x.empty())
        {
            throw std::invalid_argument("linear model: no samples");
        }
        if (!std::isfinite(ridge) || ridge < 0)
        {
            throw std::invalid_argument("linear model: ridge must be finite and non-negative");
        }
        size_t features = x.front().size();
        size_t n = features + 1;
        if (x.size() < n)
        {
            throw std::invalid_argument("linear model: needs at least " + std::to_string(n) + " samples, got " +
                                        std::to_string(x.size()));
        }
        // normal equations over the augmented row [1, x...]
        std::vector<double> a(n * n, 0.0);
        std::vector<double> b(n, 0.0);
        std::vector<double> row(n);
        for (size_t i = 0; i < x.size(); ++i)
        {
            if (x[i].size() != features)
            {
                throw std::invalid_argument("linear model: sample " + std::to_string(i) + " has " +
                                            std::to_string(x[i].size()) + " features, expected " +
                                            std::to_string(features));
            }
            if (!Finite(x[i]) || !std::isfinite(y[i]))
            {
                throw std::invalid_argument("linear model: sample " + std::to_string(i) + " is not finite");
            }
            row[0] = 1.0;
            std::copy(x[i].begin(), x[i].end(), row.begin() + 1);
            for (size_t r = 0; r < n; ++r)
            {
                for (size_t c = 0; c < n; ++c)
                {
                    a[r * n + c] += row[r] * row[c];
                }
                b[r] += row[r] * y[i];
            }
        }
        for (size_t d = 1; d < n; ++d)
        {
            a[d * n + d] += ridge; // the bias is not regularised
        }
        ModelArtifact artifact;
        artifact.m_Weights = Solve(std::move(a), std::move(b), n);
        if (!Finite(artifact.m_Weights))
        {
            throw std::invalid_argument("linear model: solution is not finite");
        }
        artifact.m_Samples = static_cast<uint32_t>(std::min<size_t>(x.size(), std::numeric_limits<uint32_t>::max()));
        artifact.m_TrainedAt = NowMillis();
        return artifact;
    }

    double LinearModel::Predict(const ModelArtifact &model, const std::vector<double> &features)
    {
        if (model.m_Weights.size() != features.size() + 1)
        {
            throw std::invalid_argument("linear model: " + std::to_string(features.size()) +
                                        " features for a model with " + std::to_string(model.m_Weights.size()) +
                                        " weights");
        }
        double value = model.m_Weights[0];
        for (size_t i = 0; i < features.size(); ++i)
        {
            value += model.m_Weights[i + 1] * features[i];
        }
        return value;
    }

    double LinearModel::Accuracy(const ModelArtifact &model, const std::vector<std::vector<double>> &x,
                                 const std::vector<double> &y, double tolerance)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument("linear model: feature rows and labels differ in count");
        }
        if (x.empty())
        {
            return 0;
        }
        size_t hits = 0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            double error = std::fabs(Predict(model, x[i]) - y[i]);
            hits += error <= tolerance ? 1 : 0;
        }
        return static_cast<double>(hits) / static_cast<double>(x.size());
    }

    // -------------------------
    // artefact contract
    // -------------------------

    std::vector<std::string> FeatureNames(const Descriptor::RichMLService &service)
    {
        std::vector<std::string> names;
        names.reserve(service.m_Observables.size());
        for (const auto &observable : service.m_Observables)
        {
            names.push_back(observable.m_Name);
        }
        return names;
    }

    std::string CheckArtifact(const ModelArtifact &artifact, const Descriptor::RichMLService &service)
    {
        if (artifact.m_Version == 0)
        {
            return "artifact version must be positive";
        }
        if (!artifact.m_Service.empty() && artifact.m_Service != service.m_Metadata.m_Name)
        {
            return "artifact belongs to service '" + artifact.m_Service + "', not '" + service.m_Metadata.m_Name + "'";
        }
        auto features = FeatureNames(service);
        if (artifact.m_Features != features)
        {
            std::string declared;
            for (const auto &feature : features)
            {
                declared += (declared.empty() ? "" : ", ") + feature;
            }
            return "artifact features do not match the declared observables [" + declared + "]";
        }
        if (artifact.m_Weights.size() != features.size() + 1)
        {
            return "artifact needs " + std::to_string(features.size() + 1) + " weights (bias first), has " +
                   std::to_string(artifact.m_Weights.size());
        }
        for (double weight : artifact.m_Weights)
        {
            if (!std::isfinite(weight))
            {
                return "artifact weights must be finite";
            }
        }
        if (!std::isfinite(artifact.m_Accuracy))
        {
            return "artifact accuracy must be finite";
        }
        return {};
    }
} // namespace eOps::Workload
