#include "Checks.h"
#include <eOps/Topology/Inventory.h>
#include <eOps/Topology/Network.h>
#include <eCore/Common/YAML.h>

using namespace eOps;
using namespace eOps::Topology;

namespace
{
    // the Knowledge.yaml resource registry and network sections (spec §5)
    const char *KNOWLEDGE = R"(resources:
  - { id: ran-1, region: ran, cpu: 4, memory: 8Gi, accelerators: {npu: 1}, broker: "127.0.0.1:5600", labels: {site: tower-a, floor: 3} }
  - { id: edge-1, region: edge, cpu: 16, memory: 32Gi, accelerators: {gpu: 1} }
  - { id: cloud-1, region: cloud, cpu: 64, memory: 256Gi, accelerators: {gpu: 8} }
network:
  regions: [ran, edge, cloud]
  latency: { ran: {edge: 2, cloud: 25}, edge: {cloud: 15} }
  topics: { "home/#": {ran: 40, edge: 5}, "home/temp": {ran: 2} }
  consumers: { "home/set/#": {ran: 10} }
  shards: { ran: ["home/#"], cloud: ["factory/#"] }
)";

    Json Knowledge()
    {
        return FromYaml(eCore::YamlNode::Parse(KNOWLEDGE));
    }

    Descriptor::Resources Need(double cpu, uint64_t memory, std::string accelerator = {}, uint32_t count = 0)
    {
        Descriptor::Resources resources;
        resources.m_Cpu = cpu;
        resources.m_Memory = memory;
        resources.m_Accelerator = std::move(accelerator);
        resources.m_AcceleratorCount = count;
        return resources;
    }

    constexpr uint64_t GI = uint64_t{1} << 30;
} // namespace

EMPY_TEST(OpsTopology, NodeCapacity)
{
    Node node;
    node.m_Id = "edge-1";
    node.m_Cpu = 4;
    node.m_Memory = 8 * GI;
    node.m_Accelerators = {{"gpu", 2}};

    EMPY_CHECK(node.Fits(Need(4, 8 * GI)));
    EMPY_CHECK(!node.Fits(Need(4.5, GI)));
    EMPY_CHECK(!node.Fits(Need(1, 9 * GI)));
    EMPY_CHECK(node.Fits(Need(1, GI, "gpu", 2)));
    EMPY_CHECK(!node.Fits(Need(1, GI, "gpu", 3)));
    EMPY_CHECK(!node.Fits(Need(1, GI, "npu", 1)));
    EMPY_CHECK(node.Fits(Need(1, GI, "gpu", 0))); // a named accelerator needs at least one
    EMPY_CHECK(!node.Fits(Need(-1, GI)));
    EMPY_CHECK_EQ(node.FreeCpuFraction(), 1.0);

    node.Reserve(Need(1, 2 * GI, "gpu", 0));
    EMPY_CHECK_EQ(node.m_UsedCpu, 1.0);
    EMPY_CHECK_EQ(node.m_UsedMemory, 2 * GI);
    EMPY_CHECK_EQ(node.FreeAccelerators("gpu"), 1u);
    EMPY_CHECK_EQ(node.FreeCpuFraction(), 0.75);
    node.Reserve(Need(1, 2 * GI, "gpu", 1));
    EMPY_CHECK_EQ(node.FreeAccelerators("gpu"), 0u);
    EMPY_CHECK(!node.Fits(Need(0.5, GI, "gpu", 1)));
    EMPY_CHECK(node.Fits(Need(2, 4 * GI)));
    EMPY_CHECK(!node.Fits(Need(2.1, GI)));

    // fractional reservations accumulate without drifting below the free amount
    Node cores;
    cores.m_Cpu = 1;
    cores.m_Memory = GI;
    for (int32_t i = 0; i < 10; ++i)
    {
        EMPY_CHECK(cores.Fits(Need(0.1, 1)));
        cores.Reserve(Need(0.1, 1));
    }
    EMPY_CHECK(!cores.Fits(Need(0.1, 1)));
    for (int32_t i = 0; i < 10; ++i)
    {
        cores.Release(Need(0.1, 1));
    }
    EMPY_CHECK_EQ(cores.m_UsedCpu, 0.0);
    EMPY_CHECK_EQ(cores.m_UsedMemory, uint64_t{0});

    node.Release(Need(1, 2 * GI, "gpu", 1));
    node.Release(Need(5, 100 * GI, "gpu", 5)); // clamped at zero
    EMPY_CHECK_EQ(node.m_UsedCpu, 0.0);
    EMPY_CHECK_EQ(node.m_UsedMemory, uint64_t{0});
    EMPY_CHECK(node.m_UsedAccelerators.empty());
    EMPY_CHECK_EQ(node.FreeAccelerators("gpu"), 2u);
    EMPY_CHECK_EQ(node.FreeAccelerators("tpu"), 0u);

    Node empty;
    EMPY_CHECK_EQ(empty.FreeCpuFraction(), 0.0);
    EMPY_CHECK(empty.Fits(Need(0, 0)));
    EMPY_CHECK(!empty.Fits(Need(0.1, 0)));
}

EMPY_TEST(OpsTopology, NodeJson)
{
    Json knowledge = Knowledge();
    Inventory inventory = Inventory::FromJson(knowledge["resources"]);
    EMPY_REQUIRE_EQ(inventory.m_Nodes.size(), size_t{3});
    const Node &ran = inventory.m_Nodes[0];
    EMPY_CHECK_EQ(ran.m_Id, std::string("ran-1"));
    EMPY_CHECK_EQ(ran.m_Region, std::string("ran"));
    EMPY_CHECK_EQ(ran.m_Cpu, 4.0);
    EMPY_CHECK_EQ(ran.m_Memory, 8 * GI);
    EMPY_CHECK_EQ(ran.FreeAccelerators("npu"), 1u);
    EMPY_CHECK_EQ(ran.m_Broker, std::string("127.0.0.1:5600"));
    EMPY_CHECK_EQ(ran.m_Labels.at("site"), std::string("tower-a"));
    EMPY_CHECK_EQ(ran.m_Labels.at("floor"), std::string("3"));
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->FreeAccelerators("gpu"), 8u);
    EMPY_CHECK(inventory.Find("fog-1") == nullptr);

    Node *edge = inventory.Find("edge-1");
    EMPY_REQUIRE(edge != nullptr);
    edge->Reserve(Need(2.5, 3 * GI, "gpu", 1));
    Json json = edge->ToJson();
    EMPY_CHECK_EQ(json.GetString("memory"), std::string("32Gi"));
    EMPY_CHECK_EQ(json.At("used.memory")->AsString(), std::string("3Gi"));
    EMPY_CHECK_EQ(json.At("used.cpu")->AsNumber(), 2.5);
    EMPY_CHECK_EQ(json.At("used.accelerators.gpu")->AsInt(), int64_t{1});
    Node back = Node::FromJson(json);
    EMPY_CHECK(back.ToJson() == json);
    EMPY_CHECK_EQ(back.FreeAccelerators("gpu"), 0u);

    const Inventory &constant = inventory;
    Inventory copy = Inventory::FromJson(constant.ToJson());
    EMPY_CHECK(copy.ToJson() == inventory.ToJson());
    EMPY_CHECK_EQ(copy.Find("edge-1")->m_UsedCpu, 2.5);
    EMPY_CHECK(constant.Find("edge-1") == &inventory.m_Nodes[1]);

    // plain byte counts and string cpu quantities are accepted
    Node bytes = Node::FromJson(Json::Parse(R"({"id":"n","cpu":"500m","memory":1024})"));
    EMPY_CHECK_EQ(bytes.m_Cpu, 0.5);
    EMPY_CHECK_EQ(bytes.m_Memory, uint64_t{1024});

    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"region":"ran"})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":""})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":"n","cpu":-1})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":"n","memory":"8GB"})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":"n","memory":-8})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":"n","accelerators":["gpu"]})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"({"id":"n","accelerators":{"gpu":-1}})")));
    EMPY_CHECK_THROWS(Node::FromJson(Json::Parse(R"(["n"])")));
    EMPY_CHECK_THROWS(Inventory::FromJson(Json::Parse(R"([{"id":"a"},{"id":"a"}])")));
    EMPY_CHECK_THROWS(Inventory::FromJson(Json::Parse(R"({"nodes":{}})")));
    EMPY_CHECK(Inventory::FromJson(Json::Parse(R"({"nodes":[]})")).m_Nodes.empty());
}

EMPY_TEST(OpsTopology, NetworkModel)
{
    NetworkModel network = NetworkModel::FromJson(Knowledge()["network"]);
    EMPY_CHECK(network.m_Regions == std::vector<std::string>({"ran", "edge", "cloud"}));

    // symmetric lookup, same-region and unknown defaults
    EMPY_CHECK_EQ(network.Latency("ran", "edge"), 2.0);
    EMPY_CHECK_EQ(network.Latency("edge", "ran"), 2.0);
    EMPY_CHECK_EQ(network.Latency("cloud", "edge"), 15.0);
    EMPY_CHECK_EQ(network.Latency("ran", "ran"), 0.5);
    EMPY_CHECK_EQ(network.Latency("ran", "mars"), 50.0);

    // densities sum every overlapping filter entry
    EMPY_CHECK_EQ(network.Density("home/temp", "ran"), 42.0);
    EMPY_CHECK_EQ(network.Density("home/hum", "ran"), 40.0);
    EMPY_CHECK_EQ(network.Density("home/+", "edge"), 5.0);
    EMPY_CHECK_EQ(network.Density("home/temp", "cloud"), 0.0);
    EMPY_CHECK_EQ(network.Density("factory/line1", "ran"), 0.0);
    EMPY_CHECK_EQ(network.Consumers("home/set/temp", "ran"), 10.0);
    EMPY_CHECK_EQ(network.Consumers("home/temp", "ran"), 0.0);
    EMPY_CHECK(network.OwnsShard("ran", "home/set/temp"));
    EMPY_CHECK(network.OwnsShard("cloud", "factory/+/motor"));
    EMPY_CHECK(!network.OwnsShard("edge", "home/temp"));
    EMPY_CHECK(!network.OwnsShard("cloud", "home/temp"));

    // locality reports replace the density for that topic and region, and add new regions
    network.ApplyLocality("fog", Json::Parse(R"({"home/temp": 12.5, "home/hum": 3})"));
    EMPY_CHECK_EQ(network.Density("home/temp", "fog"), 12.5);
    EMPY_CHECK_EQ(network.Density("home/#", "fog"), 15.5);
    EMPY_CHECK(network.m_Regions.back() == "fog");
    network.ApplyLocality("fog", Json::Parse(R"({"home/temp": 1})"));
    EMPY_CHECK_EQ(network.m_TopicDensity["home/temp"]["fog"], 1.0);
    EMPY_CHECK_EQ(network.m_Regions.size(), size_t{4});
    // malformed reports change nothing
    EMPY_CHECK_THROWS(network.ApplyLocality("fog", Json::Parse(R"({"home/temp": 9, "home/#/x": 1})")));
    EMPY_CHECK_THROWS(network.ApplyLocality("fog", Json::Parse(R"({"home/temp": 9, "home/hum": -1})")));
    EMPY_CHECK_THROWS(network.ApplyLocality("fog", Json::Parse(R"({"home/temp": "9"})")));
    EMPY_CHECK_THROWS(network.ApplyLocality("fog", Json::Parse(R"([1])")));
    EMPY_CHECK_THROWS(network.ApplyLocality("", Json::Parse(R"({})")));
    EMPY_CHECK_EQ(network.m_TopicDensity["home/temp"]["fog"], 1.0);

    Json json = network.ToJson();
    EMPY_CHECK_EQ(json.At("latency.ran.cloud")->AsNumber(), 25.0);
    EMPY_CHECK_EQ(json.At("shards.cloud.0")->AsString(), std::string("factory/#"));
    NetworkModel back = NetworkModel::FromJson(json);
    EMPY_CHECK(back.ToJson() == json);
    EMPY_CHECK_EQ(back.Density("home/temp", "fog"), 1.0); // home/hum does not overlap home/temp

    NetworkModel empty = NetworkModel::FromJson(Json::MakeObject());
    EMPY_CHECK(empty.m_Regions.empty());
    EMPY_CHECK_EQ(empty.Latency("a", "b"), 50.0);
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"({"latency":{"ran":{"edge":"far"}}})")));
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"({"latency":{"ran":3}})")));
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"({"topics":{"a/#/b":{"ran":1}}})")));
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"({"shards":{"ran":"home/#"}})")));
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"({"regions":"ran"})")));
    EMPY_CHECK_THROWS(NetworkModel::FromJson(Json::Parse(R"([])")));
}
