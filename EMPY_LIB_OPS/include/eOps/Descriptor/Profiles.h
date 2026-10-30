#pragma once
#include <vector>
#include <eCore/Network/Codecs.h>
#include "Model.h"

// MMQP topic-profiles derived from a descriptor: what a workload subscribes to and registers

namespace eOps::Descriptor
{
    // SUB entries for every observable plus training.label, QoS from the declaration
    EMPY_OPS_API std::vector<eCore::MMQP::ProfileEntry> SubscriptionProfiles(const RichMLService &service);

    // REG entries for each concrete actionable; wildcard actionables are permission domains whose
    // concrete topics are registered lazily on first publication
    EMPY_OPS_API std::vector<eCore::MMQP::ProfileEntry> PublicationProfiles(const RichMLService &service);
} // namespace eOps::Descriptor
