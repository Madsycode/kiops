#pragma once
#include <map>
#include <deque>
#include <chrono>
#include <string>
#include <vector>
#include <string_view>
#include "../Descriptor/Model.h"

// formalised execution boundary (thesis Fig. 24) enforced inside the workload itself: a
// service may only act on its declared actionables, within their bounds and rate limits

namespace eOps::Workload
{
    struct ActionDecision
    {
        bool m_Allowed = false;
        double m_Value = 0; // value to publish (clamped when m_Clamped)
        bool m_Clamped = false;
        std::string m_Reason; // why rejected or clamped ("" when allowed unchanged)
        std::string m_Rule;   // the rule that rejected it: "domain" (topic), "value" (not finite) or "rate"
    };

    class EMPY_OPS_API ActionBoundary
    {
    public:
        explicit ActionBoundary(const Descriptor::RichMLService &service);

        // undeclared or invalid topic → rejected; non-finite value → rejected; out of [min, max] →
        // clamped (m_Clamped); more than maxRatePerMinute allowed actions in the trailing 60 s for that
        // concrete topic → rejected. Allowed actions are counted toward the rate.
        ActionDecision Check(std::string_view topic, double value,
                             std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

    private:
        std::vector<Descriptor::Actionable> m_Actionables;
        std::map<std::string, std::deque<std::chrono::steady_clock::time_point>, std::less<>> m_Recent;
    };
} // namespace eOps::Workload
