#pragma once
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include <string_view>
#include "../Common/Json.h"
#include "../Descriptor/Model.h"

// Resource Registry model: compute nodes with capacity, accelerators, labels, the broker they
// attach to, and the reservations currently held against them

namespace eOps::Topology
{
    struct EMPY_OPS_API Node
    {
        std::string m_Id;
        std::string m_Region;
        double m_Cpu = 0;
        uint64_t m_Memory = 0;
        std::map<std::string, uint32_t> m_Accelerators; // kind → count
        double m_UsedCpu = 0;
        uint64_t m_UsedMemory = 0;
        std::map<std::string, uint32_t> m_UsedAccelerators;
        std::map<std::string, std::string> m_Labels;
        std::string m_Broker; // "host:port" of the node-local broker ("" = default)

        // free capacity covers cpu, memory and accelerator kind/count
        bool Fits(const Descriptor::Resources &resources) const;
        void Reserve(const Descriptor::Resources &resources); // adds to the used amounts
        void Release(const Descriptor::Resources &resources); // subtracts, clamped at zero
        double FreeCpuFraction() const;                       // (cpu − used) / cpu, 0 when cpu is 0
        uint32_t FreeAccelerators(const std::string &kind) const;

        // {id, region, cpu, memory ("8Gi" form), accelerators{}, used{cpu, memory, accelerators{}}, labels{}, broker}
        Json ToJson() const;
        // also accepts the Knowledge.yaml resource form (memory as a quantity string or bytes)
        static Node FromJson(const Json &json); // throws JsonError / std::invalid_argument
    };

    struct EMPY_OPS_API Inventory
    {
        std::vector<Node> m_Nodes;

        Node *Find(std::string_view id);
        const Node *Find(std::string_view id) const;

        Json ToJson() const;                         // {nodes: [...]}
        static Inventory FromJson(const Json &json); // {nodes: [...]} or a bare array
    };
} // namespace eOps::Topology
