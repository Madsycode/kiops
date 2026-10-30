#include <eOps/Topology/Inventory.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace eOps::Topology
{
    namespace
    {
        // tolerance for accumulated floating-point cpu reservations
        constexpr double CPU_EPSILON = 1e-9;

        uint32_t RequiredAccelerators(const Descriptor::Resources &resources)
        {
            if (resources.m_Accelerator.empty())
            {
                return 0;
            }
            return std::max<uint32_t>(1, resources.m_AcceleratorCount);
        }

        double Cpu(const Json &value, const std::string &field)
        {
            double cpu = value.IsString() ? Descriptor::ParseCpu(value.AsString()) : value.AsNumber();
            if (!std::isfinite(cpu) || cpu < 0)
            {
                throw std::invalid_argument("node " + field + " must be a non-negative quantity");
            }
            return cpu;
        }

        uint64_t Memory(const Json &value, const std::string &field)
        {
            if (value.IsString())
            {
                return Descriptor::ParseMemory(value.AsString());
            }
            if (value.IsNumber() && value.AsNumber() < 0)
            {
                throw std::invalid_argument("node " + field + " must be a non-negative quantity");
            }
            return value.AsUint();
        }

        std::map<std::string, uint32_t> Counts(const Json &value, const std::string &field)
        {
            std::map<std::string, uint32_t> counts;
            if (value.IsNull())
            {
                return counts;
            }
            if (!value.IsObject())
            {
                throw JsonError("node " + field + " must be a mapping of kind to count");
            }
            for (const auto &[kind, count] : value.AsObject())
            {
                uint64_t number = count.AsUint();
                if (number > std::numeric_limits<uint32_t>::max())
                {
                    throw std::invalid_argument("node " + field + "." + kind + " is out of range");
                }
                if (number > 0)
                {
                    counts[kind] = static_cast<uint32_t>(number);
                }
            }
            return counts;
        }

        Json CountsJson(const std::map<std::string, uint32_t> &counts)
        {
            Json object = Json::MakeObject();
            for (const auto &[kind, count] : counts)
            {
                object.Set(kind, count);
            }
            return object;
        }

        std::string Scalar(const Json &value)
        {
            return value.IsString() ? value.AsString() : value.Dump();
        }
    } // namespace

    bool Node::Fits(const Descriptor::Resources &resources) const
    {
        if (!std::isfinite(resources.m_Cpu) || resources.m_Cpu < 0)
        {
            return false;
        }
        if (m_Cpu - m_UsedCpu + CPU_EPSILON < resources.m_Cpu)
        {
            return false;
        }
        if (m_UsedMemory > m_Memory || m_Memory - m_UsedMemory < resources.m_Memory)
        {
            return false;
        }
        uint32_t accelerators = RequiredAccelerators(resources);
        return accelerators == 0 || FreeAccelerators(resources.m_Accelerator) >= accelerators;
    }

    void Node::Reserve(const Descriptor::Resources &resources)
    {
        m_UsedCpu += std::max(0.0, resources.m_Cpu);
        m_UsedMemory = resources.m_Memory > std::numeric_limits<uint64_t>::max() - m_UsedMemory
                           ? std::numeric_limits<uint64_t>::max()
                           : m_UsedMemory + resources.m_Memory;
        if (uint32_t accelerators = RequiredAccelerators(resources))
        {
            m_UsedAccelerators[resources.m_Accelerator] += accelerators;
        }
    }

    void Node::Release(const Descriptor::Resources &resources)
    {
        m_UsedCpu = std::max(0.0, m_UsedCpu - std::max(0.0, resources.m_Cpu));
        if (m_UsedCpu < CPU_EPSILON)
        {
            m_UsedCpu = 0;
        }
        m_UsedMemory = resources.m_Memory > m_UsedMemory ? 0 : m_UsedMemory - resources.m_Memory;
        if (uint32_t accelerators = RequiredAccelerators(resources))
        {
            auto used = m_UsedAccelerators.find(resources.m_Accelerator);
            if (used != m_UsedAccelerators.end())
            {
                used->second = accelerators >= used->second ? 0 : used->second - accelerators;
                if (used->second == 0)
                {
                    m_UsedAccelerators.erase(used);
                }
            }
        }
    }

    double Node::FreeCpuFraction() const
    {
        if (!(m_Cpu > 0))
        {
            return 0;
        }
        return std::clamp((m_Cpu - m_UsedCpu) / m_Cpu, 0.0, 1.0);
    }

    uint32_t Node::FreeAccelerators(const std::string &kind) const
    {
        auto total = m_Accelerators.find(kind);
        if (total == m_Accelerators.end())
        {
            return 0;
        }
        auto used = m_UsedAccelerators.find(kind);
        uint32_t taken = used == m_UsedAccelerators.end() ? 0 : used->second;
        return taken >= total->second ? 0 : total->second - taken;
    }

    Json Node::ToJson() const
    {
        Json labels = Json::MakeObject();
        for (const auto &[key, value] : m_Labels)
        {
            labels.Set(key, value);
        }
        return Json(Json::Object{
            {"id", m_Id},
            {"region", m_Region},
            {"cpu", m_Cpu},
            {"memory", Descriptor::FormatMemory(m_Memory)},
            {"accelerators", CountsJson(m_Accelerators)},
            {"used", Json(Json::Object{{"cpu", m_UsedCpu},
                                       {"memory", Descriptor::FormatMemory(m_UsedMemory)},
                                       {"accelerators", CountsJson(m_UsedAccelerators)}})},
            {"labels", std::move(labels)},
            {"broker", m_Broker},
        });
    }

    Node Node::FromJson(const Json &json)
    {
        if (!json.IsObject())
        {
            throw JsonError("node must be a mapping");
        }
        Node node;
        node.m_Id = json["id"].AsString();
        if (node.m_Id.empty())
        {
            throw std::invalid_argument("node id must not be empty");
        }
        if (const Json *region = json.Find("region"); region && !region->IsNull())
        {
            node.m_Region = region->AsString();
        }
        if (const Json *cpu = json.Find("cpu"); cpu && !cpu->IsNull())
        {
            node.m_Cpu = Cpu(*cpu, "cpu");
        }
        if (const Json *memory = json.Find("memory"); memory && !memory->IsNull())
        {
            node.m_Memory = Memory(*memory, "memory");
        }
        node.m_Accelerators = Counts(json["accelerators"], "accelerators");
        if (const Json *used = json.Find("used"); used && !used->IsNull())
        {
            if (!used->IsObject())
            {
                throw JsonError("node used must be a mapping");
            }
            if (const Json *cpu = used->Find("cpu"); cpu && !cpu->IsNull())
            {
                node.m_UsedCpu = Cpu(*cpu, "used.cpu");
            }
            if (const Json *memory = used->Find("memory"); memory && !memory->IsNull())
            {
                node.m_UsedMemory = Memory(*memory, "used.memory");
            }
            node.m_UsedAccelerators = Counts((*used)["accelerators"], "used.accelerators");
        }
        if (const Json *labels = json.Find("labels"); labels && !labels->IsNull())
        {
            if (!labels->IsObject())
            {
                throw JsonError("node labels must be a mapping");
            }
            for (const auto &[key, value] : labels->AsObject())
            {
                node.m_Labels[key] = Scalar(value);
            }
        }
        if (const Json *broker = json.Find("broker"); broker && !broker->IsNull())
        {
            node.m_Broker = broker->AsString();
        }
        return node;
    }

    Node *Inventory::Find(std::string_view id)
    {
        for (Node &node : m_Nodes)
        {
            if (node.m_Id == id)
            {
                return &node;
            }
        }
        return nullptr;
    }

    const Node *Inventory::Find(std::string_view id) const
    {
        for (const Node &node : m_Nodes)
        {
            if (node.m_Id == id)
            {
                return &node;
            }
        }
        return nullptr;
    }

    Json Inventory::ToJson() const
    {
        Json nodes = Json::MakeArray();
        for (const Node &node : m_Nodes)
        {
            nodes.Push(node.ToJson());
        }
        return Json(Json::Object{{"nodes", std::move(nodes)}});
    }

    Inventory Inventory::FromJson(const Json &json)
    {
        const Json *nodes = json.IsArray() ? &json : json.Find("nodes");
        if (!nodes || !nodes->IsArray())
        {
            throw JsonError("inventory must be a list of nodes or {nodes: [...]}");
        }
        Inventory inventory;
        for (const Json &item : nodes->AsArray())
        {
            Node node = Node::FromJson(item);
            if (inventory.Find(node.m_Id))
            {
                throw std::invalid_argument("duplicate node id '" + node.m_Id + "'");
            }
            inventory.m_Nodes.push_back(std::move(node));
        }
        return inventory;
    }
} // namespace eOps::Topology
