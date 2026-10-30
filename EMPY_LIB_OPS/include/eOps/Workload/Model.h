#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "../Common/Json.h"
#include "../Descriptor/Model.h"

// model artefact exchanged between training and inference workloads, and the linear model used
// by the reference workloads (ordinary least squares with ridge, no external libraries)

namespace eOps::Workload
{
    struct EMPY_OPS_API ModelArtifact
    {
        std::string m_Id; // "<service>-v<version>" unless set
        std::string m_Service;
        uint32_t m_Version = 0;
        std::vector<std::string> m_Features; // observable names in declaration order
        std::vector<double> m_Weights;       // bias first, then one per feature
        double m_Accuracy = 0;               // holdout accuracy at training time
        uint32_t m_Samples = 0;
        uint64_t m_TrainedAt = 0; // Unix ms
        std::string m_Dataset;

        // {id, service, version, features[], weights[], accuracy, samples, trainedAt, dataset}
        Json ToJson() const;
        static ModelArtifact FromJson(const Json &json); // throws JsonError
    };

    class EMPY_OPS_API LinearModel
    {
    public:
        // normal equations (XᵀX + ridge·I)w = Xᵀy solved by Gaussian elimination with partial pivoting
        // (the bias is not regularised); needs ≥ features + 1 samples of equal width, else
        // std::invalid_argument; m_Features/m_Service are left for the caller
        static ModelArtifact Fit(const std::vector<std::vector<double>> &x, const std::vector<double> &y,
                                 double ridge = 1e-6);
        // throws std::invalid_argument when the feature count does not match the weights
        static double Predict(const ModelArtifact &model, const std::vector<double> &features);
        // fraction of samples with |prediction − label| ≤ tolerance (0 for no samples)
        static double Accuracy(const ModelArtifact &model, const std::vector<std::vector<double>> &x,
                               const std::vector<double> &y, double tolerance);
    };

    // the artefact contract of a RichML-Service, shared by the runtimes (load) and the harness Guard
    // (promotion, rollback): observable names in declaration order are the model features
    EMPY_OPS_API std::vector<std::string> FeatureNames(const Descriptor::RichMLService &service);
    // "" when `artifact` can drive `service`: version > 0, same service (or unnamed), features equal to the
    // observable names, bias + one finite weight per feature, finite accuracy; else the first problem
    EMPY_OPS_API std::string CheckArtifact(const ModelArtifact &artifact, const Descriptor::RichMLService &service);
} // namespace eOps::Workload
