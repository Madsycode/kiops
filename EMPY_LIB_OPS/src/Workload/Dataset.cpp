#include <eOps/Workload/Dataset.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace eOps::Workload
{
    Dataset::Dataset(size_t capacity) : m_Capacity(capacity)
    {
        if (capacity == 0)
        {
            throw std::invalid_argument("dataset capacity must be at least 1");
        }
    }

    void Dataset::Add(Sample sample)
    {
        if (m_Samples.size() >= m_Capacity)
        {
            m_Samples.pop_front();
        }
        m_Samples.push_back(std::move(sample));
    }

    size_t Dataset::Size() const
    {
        return m_Samples.size();
    }

    size_t Dataset::Capacity() const
    {
        return m_Capacity;
    }

    void Dataset::Split(double holdout, std::vector<Sample> &train, std::vector<Sample> &test) const
    {
        if (!(holdout > 0 && holdout < 1))
        {
            throw std::invalid_argument("dataset holdout must lie strictly between 0 and 1");
        }
        train.clear();
        test.clear();
        size_t k = static_cast<size_t>(std::max(1.0, std::round(1.0 / holdout)));
        for (size_t i = 0; i < m_Samples.size(); ++i)
        {
            ((i + 1) % k == 0 ? test : train).push_back(m_Samples[i]);
        }
    }

    std::vector<Sample> Dataset::Samples() const
    {
        return std::vector<Sample>(m_Samples.begin(), m_Samples.end());
    }

    void Dataset::Clear()
    {
        m_Samples.clear();
    }
} // namespace eOps::Workload
