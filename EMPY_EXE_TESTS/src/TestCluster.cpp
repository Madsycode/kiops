#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <Cluster/Replication.h>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    ProfileAssertion Assertion(std::string origin, uint64_t version, std::string broker = "b")
    {
        ProfileEntry profile;
        profile.m_Type = MMQP_PROREQ_TYPE_SUB;
        profile.m_Topic = "tenant/+/temperature";
        profile.m_Qos = 1;
        profile.m_Request = true;
        return {{std::move(origin), version}, std::move(broker), "sensor", profile};
    }

    void Placement()
    {
        FabricOptions options;
        for (size_t i = 0; i < 31; ++i)
        {
            options.m_Brokers.push_back("broker-" + std::to_string(i));
        }
        Fabric fabric(options);
        std::reverse(options.m_Brokers.begin(), options.m_Brokers.end());
        Fabric shuffled(options);
        EMPY_REQUIRE(fabric.Shard("//tenant/room//temperature/") == fabric.Shard("tenant/room/temperature"));
        for (uint32_t shard = 0; shard < options.m_Shards; ++shard)
        {
            auto replicas = fabric.Replicas(shard);
            EMPY_REQUIRE(replicas == shuffled.Replicas(shard));
            EMPY_REQUIRE(replicas.size() == 3 && std::set<std::string>(replicas.begin(), replicas.end()).size() == 3);
        }
        EMPY_REQUIRE(fabric.SessionReplicas("SID") == shuffled.SessionReplicas("SID"));
        EMPY_REQUIRE(fabric.ProfileShards("tenant/#").size() == options.m_Shards);
        EMPY_REQUIRE(fabric.ProfileShards("$share/workers/tenant/+/temperature").size() == options.m_Shards);
        EMPY_REQUIRE(fabric.ProfileShards("tenant/room/temperature") ==
                     std::vector<uint32_t>{fabric.Shard("tenant/room/temperature")});
        Throws([&] { fabric.Replicas(options.m_Shards); });
        auto tree = fabric.Tree("tenant/room/temperature", "broker-15", options.m_Brokers);
        EMPY_REQUIRE(tree.m_Brokers.front() == "broker-15" && tree.m_Brokers.size() == options.m_Brokers.size());
        EMPY_REQUIRE(tree.m_Brokers ==
                     shuffled.Tree("tenant/room/temperature", "broker-15", options.m_Brokers).m_Brokers);
        EMPY_REQUIRE(!tree.Parent(0));
        EMPY_REQUIRE(tree.Subtree(0).size() == options.m_Brokers.size());
        for (size_t rank = 1; rank < tree.m_Brokers.size(); ++rank)
        {
            auto parent = tree.Parent(rank);
            EMPY_REQUIRE(parent && *parent < rank && tree.Edge(tree.m_Brokers[*parent], tree.m_Brokers[rank]));
            EMPY_REQUIRE(!tree.Edge(tree.m_Brokers[rank], tree.m_Brokers[*parent]));
            auto children = tree.Children(*parent);
            EMPY_REQUIRE(std::find(children.begin(), children.end(), rank) != children.end());
            EMPY_REQUIRE(children.size() <= options.m_Fanout);
        }
        auto duplicates = options.m_Brokers;
        duplicates.push_back(duplicates.back());
        EMPY_REQUIRE(fabric.Tree("tenant/t", "broker-15", duplicates).m_Brokers.size() == options.m_Brokers.size());
        Throws([&] { fabric.Tree("tenant/t", "unknown", options.m_Brokers); });
        Throws([&] { fabric.Tree("tenant/t", "broker-15", {"unknown"}); });
        options.m_Brokers.push_back(options.m_Brokers.front());
        Throws([&] { Fabric invalid(options); });

        TenantPolicy tenant{"tenant", {fabric.Shard("tenant/t")}};
        EMPY_REQUIRE(tenant.Allows(fabric, "tenant/t"));
        EMPY_REQUIRE(!tenant.Allows(fabric, "tenant-other/t"));
        EMPY_REQUIRE(!tenant.Allows(fabric, "other/t"));
        // A wildcard needs every shard it can match; an empty allowlist is unrestricted.
        EMPY_REQUIRE(!tenant.Allows(fabric, "tenant/#") && !tenant.Allows(fabric, "tenant/+/temperature"));
        std::set<uint32_t> all;
        for (uint32_t shard = 0; shard < options.m_Shards; ++shard)
        {
            all.insert(shard);
        }
        TenantPolicy everywhere{"tenant", all};
        EMPY_REQUIRE(everywhere.Allows(fabric, "tenant/#") && everywhere.Allows(fabric, "tenant/+/temperature"));
        EMPY_REQUIRE(!everywhere.Allows(fabric, "other/#"));
        TenantPolicy unrestricted{"tenant", {}};
        EMPY_REQUIRE(unrestricted.Allows(fabric, "tenant/#") && unrestricted.Allows(fabric, "$bwm/tenant/device"));
        EMPY_REQUIRE(!unrestricted.Allows(fabric, "other/#") && !unrestricted.Allows(fabric, "$bwm/other/device"));
    }

    void Forwarding()
    {
        FabricOptions options;
        options.m_Brokers = {"a", "b", "c", "d", "e"};
        Fabric fabric(options);
        Buffer key(32, 0x5a);
        ForwardRecord record;
        record.m_AcceptedEpoch = record.m_RoutingEpoch = 1;
        record.m_Sequence = 1;
        record.m_Expires = 1000;
        record.m_Session = "SID";
        record.m_Ingress = "a";
        record.m_Destinations = {"b", "c", "d", "e"};
        record.m_Publication.m_Topic = "tenant/t";
        record.m_Publication.m_Alias = 1;
        record.m_Publication.m_Payload = {1, 2, 3};
        record.m_Identity = MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                            record.m_AcceptedEpoch, record.m_Publication.m_Payload);
        auto bytes = FabricCodec::Encode(record, key);
        auto decoded = FabricCodec::Decode(bytes, key);
        EMPY_REQUIRE(decoded.m_Identity == record.m_Identity &&
                     decoded.m_Publication.m_Payload == record.m_Publication.m_Payload);
        auto tree = fabric.Tree(record.m_Publication.m_Topic, record.m_Ingress, record.m_Destinations);
        for (size_t rank = 1; rank < tree.m_Brokers.size(); ++rank)
        {
            decoded.m_SenderRank = static_cast<uint32_t>(*tree.Parent(rank));
            auto validated =
                FabricCodec::Validate(decoded, fabric, tree.m_Brokers[decoded.m_SenderRank], tree.m_Brokers[rank], 1);
            EMPY_REQUIRE(validated.m_Brokers == tree.m_Brokers);
            Throws([&] { FabricCodec::Validate(decoded, fabric, "unknown", tree.m_Brokers[rank], 1); });
            Throws(
                [&]
                {
                    FabricCodec::Validate(decoded, fabric, tree.m_Brokers[decoded.m_SenderRank], tree.m_Brokers[rank],
                                          1000);
                });
        }
        // Relays rewrite only the authenticated sender rank, byte-identical to a re-encoding.
        auto relayed = FabricCodec::Encode(record, key);
        auto parsed = FabricCodec::Decode(relayed, key);
        FabricCodec::Rebind(relayed, parsed, 2, key);
        auto ranked = record;
        ranked.m_SenderRank = 2;
        EMPY_REQUIRE(relayed == FabricCodec::Encode(ranked, key) &&
                     FabricCodec::Decode(relayed, key).m_SenderRank == 2);
        Throws([&] { FabricCodec::Rebind(relayed, parsed, 1, key); });
        bytes[8] ^= 1;
        Throws([&] { FabricCodec::Decode(bytes, key); });
        auto wrongKey = key;
        wrongKey[0] ^= 1;
        Throws([&] { FabricCodec::Decode(FabricCodec::Encode(record, key), wrongKey); });
        Throws([&] { FabricCodec::Decode(FabricCodec::Encode(record, key), key, 64); });
        Throws([&] { FabricCodec::Decode(FabricCodec::Encode(record, key), key, 10000, 2); });
        EpochFilter filter(1, 2);
        EMPY_REQUIRE(filter.Accept(1, record.m_Identity) && !filter.Accept(1, record.m_Identity));
        filter.Advance(2);
        record.m_RoutingEpoch = 2;
        options.m_Epoch = 2;
        Fabric next(options);
        EMPY_REQUIRE(!filter.Accept(2, record.m_Identity));
        record.m_Sequence = 2;
        record.m_Identity = MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                            record.m_AcceptedEpoch, record.m_Publication.m_Payload);
        EMPY_REQUIRE(filter.Accept(2, record.m_Identity));
        record.m_Sequence = 3;
        record.m_Identity = MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                            record.m_AcceptedEpoch, record.m_Publication.m_Payload);
        EMPY_REQUIRE(filter.Accept(2, record.m_Identity));
        Throws([&] { filter.Accept(2, Digest{}); });
        Throws([&] { filter.Advance(1); });
        filter.Advance(3);
        Throws([&] { filter.Accept(1, Digest{}); });
    }

    void Convergence()
    {
        std::set<std::string> replicas{"a", "b", "c"};
        ObservedProfiles a("a", replicas), b("b", replicas), c("c", replicas);
        auto first = Assertion("a", 1);
        a.Add(first);
        auto stale = a;
        b.Merge(a);
        auto seen = b.Observe("b", "sensor", "tenant/+/temperature");
        EMPY_REQUIRE(seen.size() == 1);
        b.Remove({{"b", 1}, seen});
        // Removal can arrive before the assertion it names.
        c.Remove({{"b", 1}, seen});
        c.Add(first);
        EMPY_REQUIRE(c.Destinations("tenant/room/temperature", 1).empty());
        // A concurrent assertion survives an observed-remove of the first dot.
        auto concurrent = Assertion("a", 2, "c");
        a.Add(concurrent);
        a.Merge(b);
        b.Merge(a);
        c.Merge(b);
        for (auto *replica : {&a, &b, &c})
        {
            EMPY_REQUIRE(replica->Destinations("tenant/room/temperature", 1) == std::vector<std::string>{"c"});
        }
        a.Compact();
        EMPY_REQUIRE(a.Tombstones() == 1);
        a.Acknowledge("b", b.Versions());
        a.Acknowledge("c", VersionVector{{"a", 2}});
        a.Compact();
        EMPY_REQUIRE(a.Tombstones() == 1); // c has not acknowledged b's removal.
        a.Acknowledge("c", c.Versions());
        a.Compact();
        EMPY_REQUIRE(a.Tombstones() == 0);
        a.Merge(stale);
        EMPY_REQUIRE(a.Destinations("tenant/room/temperature", 1) == std::vector<std::string>{"c"});
        stale.Merge(a);
        EMPY_REQUIRE(stale.Destinations("tenant/room/temperature", 1) == std::vector<std::string>{"c"});
        a.Add(first);
        EMPY_REQUIRE(a.Observe("b", "sensor", "tenant/+/temperature").empty());
        Throws([&] { a.Acknowledge("c", {}); });
        Throws([&] { a.Acknowledge("unknown", {}); });

        ObservedProfiles gaps("a", replicas);
        gaps.Add(Assertion("a", 3));
        EMPY_REQUIRE(gaps.Versions().at("a") == 0);
        gaps.Add(first);
        EMPY_REQUIRE(gaps.Versions().at("a") == 1);
        ObservedProfiles middle("b", replicas);
        middle.Add(first);
        middle.Add(concurrent);
        gaps.Merge(middle);
        EMPY_REQUIRE(gaps.Versions().at("a") == 3);
        auto conflict = first;
        conflict.m_Broker = "different";
        Throws([&] { gaps.Add(conflict); });
        Throws([&] { gaps.Remove({first.m_Dot, {concurrent.m_Dot}}); });
        Throws([&] { gaps.Add(Assertion("a", 0)); });
        ObservedProfiles limited("a", replicas, 2);
        limited.Add(first);
        Throws([&] { limited.Remove({{"b", 1}, {first.m_Dot}}); });
        limited.Add(concurrent);
        Throws([&] { limited.Add(Assertion("a", 3)); });
        EMPY_REQUIRE(limited.Destinations("tenant/room/temperature", 1).size() == 2);
    }

    void QuorumsAndBudgets()
    {
        SessionQuorum quorum(7, {"a", "b", "c"}, 64);
        EMPY_REQUIRE(quorum.Required() == 2 && quorum.CommittedVersion() == 0);
        Throws([&] { quorum.Snapshot(); });
        auto hash = quorum.Propose(1, {1, 2, 3});
        EMPY_REQUIRE(!quorum.Acknowledge("a", 7, 1, hash));
        EMPY_REQUIRE(!quorum.Acknowledge("a", 7, 1, hash));
        Throws([&] { quorum.Acknowledge("b", 6, 1, hash); });
        Throws([&] { quorum.Acknowledge("unknown", 7, 1, hash); });
        Throws([&] { quorum.Acknowledge("b", 7, 1, Digest{}); });
        Throws([&] { quorum.Propose(1, {2, 3, 4}); });
        EMPY_REQUIRE(quorum.Acknowledge("b", 7, 1, hash));
        EMPY_REQUIRE(quorum.CommittedVersion() == 1 && quorum.Snapshot() == Buffer({1, 2, 3}));
        auto second = quorum.Propose(2, {4, 5});
        EMPY_REQUIRE(!quorum.Acknowledge("c", 7, 2, second));
        EMPY_REQUIRE(quorum.Snapshot() == Buffer({1, 2, 3}));
        EMPY_REQUIRE(quorum.Acknowledge("b", 7, 2, second));
        EMPY_REQUIRE(quorum.Snapshot() == Buffer({4, 5}));

        AdmissionBudget budget;
        budget.m_MaxRecords = 1;
        budget.m_MaxBytes = 10;
        EMPY_REQUIRE(budget.Reserve(1, 7, true) == Admission::Accept);
        EMPY_REQUIRE(budget.Reserve(0, 1, true) == Admission::Drop);
        EMPY_REQUIRE(budget.Reserve(1, 1, true) == Admission::Reject);
        EMPY_REQUIRE(budget.Reserve(1, 1, true, true) == Admission::Defer);
        Throws([&] { budget.Release(8); });
        budget.Release(7);
        EMPY_REQUIRE(budget.Bytes() == 0 && budget.Records() == 0);
        EMPY_REQUIRE(budget.Reserve(1, 1, false) == Admission::Reject);
        EMPY_REQUIRE(budget.Reserve(0, 11, true) == Admission::Drop);
    }
} // namespace

EMPY_TEST(Cluster, ReferenceCases)
{
    Placement();
    Forwarding();
    Convergence();
    QuorumsAndBudgets();
}
