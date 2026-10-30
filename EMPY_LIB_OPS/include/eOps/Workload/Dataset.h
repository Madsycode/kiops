#pragma once
#include <deque>
#include <vector>
#include <cstdint>
#include "../API.h"

// bounded sliding training window (training.window samples); owner-thread use

namespace eOps::Workload
{
    struct Sample
    {
        std::vector<double> m_Features;
        double m_Label = 0;
        uint64_t m_Time = 0; // Unix ms
    };

    class EMPY_OPS_API Dataset
    {
    public:
        explicit Dataset(size_t capacity = 512); // capacity ≥ 1
        void Add(Sample sample);                 // evicts the oldest sample when full
        size_t Size() const;
        size_t Capacity() const;
        // deterministic: with k = round(1 / holdout) (holdout in (0,1)), every k-th sample (indices
        // k-1, 2k-1, ...) goes to `test`, the rest to `train`; outputs are cleared first
        void Split(double holdout, std::vector<Sample> &train, std::vector<Sample> &test) const;
        std::vector<Sample> Samples() const; // oldest first
        void Clear();

    private:
        size_t m_Capacity = 512;
        std::deque<Sample> m_Samples;
    };
} // namespace eOps::Workload
