#pragma once
#include <map>
#include <string>
#include <vector>
#include <string_view>
#include "../Common/Json.h"

// Network Knowledge model used for communication-aware placement: inter-region latency,
// telemetry (observable) density, actionable consumer density and profile-shard ownership

namespace eOps::Topology
{
    struct EMPY_OPS_API NetworkModel
    {
        std::vector<std::string> m_Regions;
        std::map<std::string, std::map<std::string, double>> m_LatencyMs;    // region → region → ms (symmetric lookup)
        std::map<std::string, std::map<std::string, double>> m_TopicDensity; // topic filter → region → msgs/s
        std::map<std::string, std::map<std::string, double>>
            m_ConsumerDensity;                                         // actionable filter → region → subscribers
        std::map<std::string, std::vector<std::string>> m_ShardOwners; // region → owned topic filters

        // a→b or b→a entry; same region defaults to 0.5 ms, unknown pairs to 50 ms
        double Latency(std::string_view a, std::string_view b) const;
        // sum over m_TopicDensity entries whose filter overlaps `topic` (a topic or filter)
        double Density(std::string_view topic, std::string_view region) const;
        // same over m_ConsumerDensity
        double Consumers(std::string_view topic, std::string_view region) const;
        // `region` owns a shard filter overlapping `topic`
        bool OwnsShard(std::string_view region, std::string_view topic) const;
        // locality report {"home/temp": 12.5, ...}: sets m_TopicDensity[topic][region]; adds unknown regions
        void ApplyLocality(std::string_view region, const Json &topics);

        // {regions[], latency{a:{b:ms}}, topics{filter:{region:rate}}, consumers{...}, shards{region:[filters]}}
        // (the Knowledge.yaml "network" section format)
        Json ToJson() const;
        static NetworkModel FromJson(const Json &json); // throws JsonError
    };
} // namespace eOps::Topology
