#include "Checks.h"
#include "Support/Process.h"
#include "Support/Ops.h"
#include <eOps/Context/Knowledge.h>
#include <eOps/Context/Query.h>
#include <eOps/Context/Comas.h>
#include <eOps/Context/Mcp.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Storage.h>
#include <eCore/Common/YAML.h>
#include <set>
#include <atomic>
#include <thread>
#include <sstream>
#include <filesystem>

using namespace eOps;
using namespace eOps::Context;

namespace
{
    using eTest::Ops::ConfigFile;

    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsContext")
        {
        }
    };

    using Clock = std::chrono::steady_clock;

    std::filesystem::path KnowledgeFile()
    {
        return ConfigFile("Knowledge.yaml");
    }

    void SeedShipped(KnowledgePlane &plane)
    {
        plane.Seed(eCore::YamlNode::Load(KnowledgeFile().string()));
    }

    Entry Make(std::string collection, std::string id, Json data, std::vector<std::string> tags = {},
               uint64_t updated = 0)
    {
        Entry entry;
        entry.m_Collection = std::move(collection);
        entry.m_Id = std::move(id);
        entry.m_Data = std::move(data);
        entry.m_Tags = std::move(tags);
        entry.m_Updated = updated;
        return entry;
    }

    std::vector<std::string> Ids(const std::vector<Entry> &entries)
    {
        std::vector<std::string> ids;
        for (const Entry &entry : entries)
        {
            ids.push_back(entry.m_Id);
        }
        return ids;
    }

    std::vector<Entry> Run(const KnowledgePlane &plane, const Json &query)
    {
        LocalSource source(plane);
        return source.Execute(Query::FromJson(query));
    }

    Query One(std::string collection, std::string field, Op op, Json value)
    {
        Query query;
        query.m_Collections = {std::move(collection)};
        query.m_Where.push_back(Condition{std::move(field), op, std::move(value)});
        return query;
    }

    // keys a structured-output schema must never use (no numeric or length constraints)
    bool StrictSchema(const Json &schema, std::string &problem, const std::string &path = "<root>")
    {
        if (schema.IsArray())
        {
            for (size_t i = 0; i < schema.Size(); ++i)
            {
                if (!StrictSchema(schema[i], problem, path + "." + std::to_string(i)))
                {
                    return false;
                }
            }
            return true;
        }
        if (!schema.IsObject())
        {
            return true;
        }
        for (const char *forbidden : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                                      "minLength", "maxLength", "minItems", "maxItems", "pattern"})
        {
            if (schema.Has(forbidden))
            {
                problem = path + " uses " + forbidden;
                return false;
            }
        }
        if (schema.GetString("type") == "object")
        {
            const Json *additional = schema.Find("additionalProperties");
            if (!additional || !additional->IsBool() || additional->AsBool())
            {
                problem = path + " lacks additionalProperties:false";
                return false;
            }
            size_t properties = schema["properties"].Size();
            if (schema["required"].Size() != properties)
            {
                problem = path + " does not require every property";
                return false;
            }
        }
        for (const auto &[key, member] : schema.AsObject())
        {
            if (!StrictSchema(member, problem, path + "." + key))
            {
                return false;
            }
        }
        return true;
    }

    // scripted Reasoning provider
    struct FakeProvider : Intelligence::Provider
    {
        Intelligence::Completion m_Result;
        bool m_Available = true;
        std::atomic<int32_t> m_Calls{0};
        Intelligence::Prompt m_Seen;
        std::string Name() const override
        {
            return "fake";
        }
        bool Available() const override
        {
            return m_Available;
        }
        Intelligence::Completion Complete(const Intelligence::Prompt &prompt) override
        {
            ++m_Calls;
            m_Seen = prompt;
            return m_Result;
        }
    };

    Json Rpc(int64_t id, std::string method, Json params = Json())
    {
        Json message = Json(Json::Object{{"jsonrpc", "2.0"}, {"id", id}, {"method", std::move(method)}});
        if (!params.IsNull())
        {
            message.Set("params", std::move(params));
        }
        return message;
    }

    int64_t ErrorCode(const Json &response)
    {
        const Json *error = response.Find("error");
        return error ? error->GetInt("code") : 0;
    }

    Json Initialize(McpServer &server)
    {
        return server.Handle(Rpc(1, "initialize",
                                 Json(Json::Object{{"protocolVersion", "2025-06-18"},
                                                   {"capabilities", Json::MakeObject()},
                                                   {"clientInfo", Json(Json::Object{{"name", "test"}})}})));
    }

    Json Call(McpServer &server, int64_t id, std::string tool, Json arguments)
    {
        return server.Handle(
            Rpc(id, "tools/call", Json(Json::Object{{"name", std::move(tool)}, {"arguments", std::move(arguments)}})));
    }

    std::vector<std::string> Shell(std::vector<std::string> windows, std::vector<std::string> posix)
    {
#ifdef _WIN32
        (void)posix;
        return windows;
#else
        (void)windows;
        return posix;
#endif
    }

    ProcessOptions Command(std::vector<std::string> argv)
    {
        ProcessOptions options;
        options.m_Program = argv.front();
        options.m_Args.assign(argv.begin() + 1, argv.end());
        return options;
    }
} // namespace

EMPY_TEST(OpsContext, EntryJson)
{
    Entry entry = Make(MEMORY, "a-1", Json(Json::Object{{"service", "home-control"}, {"value", 3}}), {"x", "y"}, 77);
    entry.m_Created = 70;
    entry.m_Score = 0.5;
    Json json = entry.ToJson();
    EMPY_CHECK_EQ(std::string(json["id"].AsString()), "a-1");
    EMPY_CHECK_EQ(std::string(json["collection"].AsString()), "memory");
    EMPY_CHECK_EQ(json["updated"].AsInt(), 77);
    Entry back = Entry::FromJson(Json::Parse(json.Dump()));
    EMPY_CHECK_EQ(back.m_Id, "a-1");
    EMPY_CHECK(back.m_Data == entry.m_Data);
    EMPY_CHECK_EQ(back.m_Tags.size(), size_t(2));
    EMPY_CHECK_EQ(back.m_Created, uint64_t(70));
    EMPY_CHECK_EQ(back.m_Updated, uint64_t(77));
    EMPY_CHECK_EQ(back.m_Score, 0.5);
    EMPY_CHECK_THROWS(Entry::FromJson(Json(1)));
    EMPY_CHECK_THROWS(Entry::FromJson(Json::Parse(R"({"id": 1, "collection": "memory"})")));
    EMPY_CHECK_THROWS(Entry::FromJson(Json::Parse(R"({"id": "a", "collection": "memory", "tags": [1]})")));
    EMPY_CHECK_THROWS(Entry::FromJson(Json::Parse(R"({"id": "a", "collection": "memory", "updated": -1})")));
}

EMPY_TEST(OpsContext, Crud)
{
    KnowledgePlane plane;
    EMPY_CHECK(plane.Collections().empty());
    plane.Put(Make(TOOLS, "b", Json(Json::Object{{"description", "second"}})));
    plane.Put(Make(TOOLS, "a", Json(Json::Object{{"description", "first"}}), {"deploy"}));
    plane.Put(Make(OPERATOR, "p", Json(Json::Object{{"text", "prefer ran"}})));
    EMPY_CHECK_EQ(plane.Size(TOOLS), size_t(2));
    EMPY_CHECK_EQ(plane.Size("missing"), size_t(0));
    EMPY_CHECK(plane.Collections() == std::vector<std::string>({"operator", "tools"}));
    EMPY_CHECK(Ids(plane.Scan(TOOLS)) == std::vector<std::string>({"a", "b"}));

    std::optional<Entry> a = plane.Get(TOOLS, "a");
    EMPY_REQUIRE(a.has_value());
    EMPY_CHECK(a->m_Created > 0 && a->m_Updated >= a->m_Created);
    EMPY_CHECK(a->m_Tags == std::vector<std::string>({"deploy"}));
    EMPY_CHECK(!plane.Get(TOOLS, "zzz"));

    // replacing keeps the original creation time; an explicit update time is kept
    Entry replacement = Make(TOOLS, "a", Json(Json::Object{{"description", "replaced"}}), {}, a->m_Updated + 5000);
    plane.Put(replacement);
    std::optional<Entry> replaced = plane.Get(TOOLS, "a");
    EMPY_REQUIRE(replaced.has_value());
    EMPY_CHECK_EQ(replaced->m_Created, a->m_Created);
    EMPY_CHECK_EQ(replaced->m_Updated, a->m_Updated + 5000);
    EMPY_CHECK_EQ(replaced->m_Data.GetString("description"), "replaced");
    EMPY_CHECK_EQ(plane.Size(TOOLS), size_t(2));

    EMPY_CHECK(plane.Erase(TOOLS, "a"));
    EMPY_CHECK(!plane.Erase(TOOLS, "a"));
    EMPY_CHECK(plane.Erase(OPERATOR, "p"));
    EMPY_CHECK(plane.Collections() == std::vector<std::string>({"tools"}));

    // invalid keys and unrepresentable data are rejected without side effects
    EMPY_CHECK_THROWS(plane.Put(Make("", "x", Json(1))));
    EMPY_CHECK_THROWS(plane.Put(Make("Bad/Name", "x", Json(1))));
    EMPY_CHECK_THROWS(plane.Put(Make(TOOLS, "", Json(1))));
    EMPY_CHECK_THROWS(plane.Put(Make(TOOLS, "line\nbreak", Json(1))));
    EMPY_CHECK_THROWS(plane.Put(Make(TOOLS, std::string(300, 'x'), Json(1))));
    EMPY_CHECK_THROWS(plane.Put(Make(TOOLS, "nan", Json(std::nan("")))));
    EMPY_CHECK_EQ(plane.Size(TOOLS), size_t(1));
}

EMPY_TEST(OpsContext, BoundsAndEviction)
{
    KnowledgeOptions options;
    options.m_MaxEntries = 3;
    KnowledgePlane plane(options);
    for (int32_t i = 0; i < 5; ++i)
    {
        plane.Put(Make(OBSERVATIONS, "o" + std::to_string(i), Json(i), {}, 1000 + i));
    }
    // the oldest m_Updated entries were evicted, per collection
    EMPY_CHECK(Ids(plane.Scan(OBSERVATIONS)) == std::vector<std::string>({"o2", "o3", "o4"}));
    plane.Put(Make(MEMORY, "m", Json(1), {}, 1));
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(1)); // another collection's budget is independent
    // refreshing an entry protects it: o2 becomes the newest
    plane.Put(Make(OBSERVATIONS, "o2", Json(2), {}, 2000));
    plane.Put(Make(OBSERVATIONS, "o5", Json(5), {}, 1500));
    EMPY_CHECK(Ids(plane.Scan(OBSERVATIONS)) == std::vector<std::string>({"o2", "o4", "o5"}));
    // a new entry older than everything still stays (the newest insert is never its own victim)
    plane.Put(Make(OBSERVATIONS, "old", Json(0), {}, 1));
    EMPY_CHECK_EQ(plane.Size(OBSERVATIONS), size_t(3));
    EMPY_CHECK(plane.Get(OBSERVATIONS, "old").has_value());
    EMPY_CHECK(!plane.Get(OBSERVATIONS, "o4").has_value());

    // byte budget: the globally oldest entries make room; an entry above the budget is refused
    KnowledgeOptions budget;
    budget.m_MaxBytes = 600;
    KnowledgePlane bytes(budget);
    std::string payload(100, 'p');
    for (int32_t i = 0; i < 8; ++i)
    {
        bytes.Put(Make(i % 2 ? TOOLS : MEMORY, "e" + std::to_string(i), Json(payload), {}, 100 + i));
    }
    size_t total = bytes.Size(TOOLS) + bytes.Size(MEMORY);
    EMPY_CHECK(total >= 2 && total < 8);
    EMPY_CHECK(bytes.Get(TOOLS, "e7").has_value());
    EMPY_CHECK(!bytes.Get(MEMORY, "e0").has_value());
    EMPY_CHECK_THROWS(bytes.Put(Make(TOOLS, "huge", Json(std::string(1000, 'h')))));
    EMPY_CHECK(bytes.Get(TOOLS, "e7").has_value());

    // the collection count is bounded
    KnowledgePlane many;
    for (int32_t i = 0; i < 64; ++i)
    {
        many.Put(Make("c" + std::to_string(i), "x", Json(i)));
    }
    EMPY_CHECK_THROWS(many.Put(Make("overflow", "x", Json(1))));
    EMPY_CHECK_EQ(many.Collections().size(), size_t(64));
    KnowledgeOptions unbounded;
    unbounded.m_MaxEntries = 0;
    EMPY_CHECK_THROWS(KnowledgePlane(unbounded));
}

EMPY_TEST(OpsContext, PersistenceReload)
{
    TempDir temp;
    KnowledgeOptions options;
    options.m_Directory = temp.m_Path / "knowledge";
    {
        KnowledgePlane plane(options);
        plane.Put(Make(MEMORY, "m1", Json(Json::Object{{"service", "home-control"}}), {"lifecycle"}, 10));
        plane.Put(Make(MEMORY, "m2", Json(2), {}, 20));
        plane.Put(Make(MEMORY, "m3", Json(3), {}, 30));
        plane.Put(Make(TOOLS, "kubectl", Json(Json::Object{{"kind", "tool"}})));
        EMPY_CHECK(plane.Erase(MEMORY, "m2"));
        // the store is exclusive to one plane at a time
        EMPY_CHECK_THROWS(KnowledgePlane(options));
    }
    {
        KnowledgePlane plane(options);
        EMPY_CHECK(Ids(plane.Scan(MEMORY)) == std::vector<std::string>({"m1", "m3"}));
        std::optional<Entry> m1 = plane.Get(MEMORY, "m1");
        EMPY_REQUIRE(m1.has_value());
        EMPY_CHECK_EQ(m1->m_Updated, uint64_t(10));
        EMPY_CHECK_EQ(m1->m_Data.GetString("service"), "home-control");
        EMPY_CHECK(m1->m_Tags == std::vector<std::string>({"lifecycle"}));
        EMPY_CHECK_EQ(plane.Size(TOOLS), size_t(1));
        plane.Put(Make(MEMORY, "m4", Json(4), {}, 40));
    }
    {
        // tighter bounds on reload keep the newest entries and persist the eviction
        KnowledgeOptions tight = options;
        tight.m_MaxEntries = 2;
        KnowledgePlane plane(tight);
        EMPY_CHECK(Ids(plane.Scan(MEMORY)) == std::vector<std::string>({"m3", "m4"}));
    }
    {
        KnowledgePlane plane(options);
        EMPY_CHECK(Ids(plane.Scan(MEMORY)) == std::vector<std::string>({"m3", "m4"}));
    }
    // a record that is not a knowledge entry fails the reload closed
    std::filesystem::path corrupt = temp.m_Path / "corrupt";
    {
        eCore::MMQP::FileStore store(corrupt, 1 << 20, 64, true);
        std::string text = "{not json";
        store.Put("memory/bad", eCore::MMQP::Buffer(text.begin(), text.end()));
    }
    KnowledgeOptions broken;
    broken.m_Directory = corrupt;
    EMPY_CHECK_THROWS(KnowledgePlane(broken));
    std::filesystem::path mismatched = temp.m_Path / "mismatched";
    {
        eCore::MMQP::FileStore store(mismatched, 1 << 20, 64, true);
        std::string text = Make(MEMORY, "other", Json(1), {}, 5).ToJson().Dump();
        store.Put("memory/bad", eCore::MMQP::Buffer(text.begin(), text.end()));
    }
    broken.m_Directory = mismatched;
    EMPY_CHECK_THROWS(KnowledgePlane(broken));
}

EMPY_TEST(OpsContext, QueryOps)
{
    KnowledgePlane plane;
    plane.Put(
        Make(RESOURCES, "edge-1",
             Json::Parse(R"({"region": "Edge", "cpu": 16, "accelerators": {"gpu": 1}, "labels": {"site": "metro"}})"),
             {"node", "edge"}, 100));
    plane.Put(
        Make(RESOURCES, "ran-1",
             Json::Parse(R"({"region": "ran", "cpu": 4, "accelerators": {"npu": 1}, "labels": {"site": "tower-a"}})"),
             {"node", "ran"}, 200));
    plane.Put(Make(RESOURCES, "cloud-1", Json::Parse(R"({"region": "cloud", "cpu": 64, "accelerators": {}})"), {"node"},
                   300));
    plane.Put(Make(NETWORK, "latency/ran/cloud",
                   Json::Parse(R"({"kind": "latency", "region": ["ran", "cloud"], "ms": 25})"), {}, 400));
    plane.Put(Make(NETWORK, "topics/home/#", Json::Parse(R"({"kind": "density", "topic": "home/#"})"), {}, 500));
    plane.Put(Make(NETWORK, "shard/cloud", Json::Parse(R"({"kind": "shard", "topics": ["factory/#"]})"), {}, 600));
    LocalSource source(plane);
    auto ids = [&](const Query &query) { return Ids(source.Execute(query)); };

    // eq: case-insensitive strings, numbers by value, any array element
    EMPY_CHECK(ids(One(RESOURCES, "region", Op::Eq, Json("edge"))) == std::vector<std::string>({"edge-1"}));
    EMPY_CHECK(ids(One(RESOURCES, "cpu", Op::Eq, Json(4.0))) == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(ids(One(NETWORK, "region", Op::Eq, Json("cloud"))) == std::vector<std::string>({"latency/ran/cloud"}));
    // ne: missing fields qualify
    EMPY_CHECK(ids(One(RESOURCES, "region", Op::Ne, Json("ran"))) == std::vector<std::string>({"cloud-1", "edge-1"}));
    EMPY_CHECK_EQ(ids(One(NETWORK, "region", Op::Ne, Json("ran"))).size(), size_t(2));
    // lt/le/gt/ge on numbers and strings
    EMPY_CHECK(ids(One(RESOURCES, "cpu", Op::Lt, Json(16))) == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK_EQ(ids(One(RESOURCES, "cpu", Op::Le, Json(16))).size(), size_t(2));
    EMPY_CHECK(ids(One(RESOURCES, "cpu", Op::Gt, Json(16))) == std::vector<std::string>({"cloud-1"}));
    EMPY_CHECK_EQ(ids(One(RESOURCES, "cpu", Op::Ge, Json(16))).size(), size_t(2));
    EMPY_CHECK(ids(One(RESOURCES, "id", Op::Ge, Json("ran"))) == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(ids(One(RESOURCES, "cpu", Op::Gt, Json("a"))).empty()); // incomparable types never match
    // contains: substring, array element, object key
    EMPY_CHECK(ids(One(RESOURCES, "labels.site", Op::Contains, Json("TOWER"))) == std::vector<std::string>({"ran-1"}));
    EMPY_CHECK(ids(One(RESOURCES, "tags", Op::Contains, Json("edge"))) == std::vector<std::string>({"edge-1"}));
    EMPY_CHECK(ids(One(RESOURCES, "accelerators", Op::Contains, Json("npu"))) == std::vector<std::string>({"ran-1"}));
    // in
    EMPY_CHECK_EQ(ids(One(RESOURCES, "region", Op::In, Json::Parse(R"(["ran", "cloud"])"))).size(), size_t(2));
    EMPY_CHECK(ids(One(RESOURCES, "id", Op::In, Json("cloud-1"))) == std::vector<std::string>({"cloud-1"}));
    // exists / exists false
    EMPY_CHECK(ids(One(RESOURCES, "accelerators.gpu", Op::Exists, Json(true))) == std::vector<std::string>({"edge-1"}));
    EMPY_CHECK(ids(One(RESOURCES, "accelerators.gpu", Op::Exists, Json())) == std::vector<std::string>({"edge-1"}));
    EMPY_CHECK_EQ(ids(One(RESOURCES, "accelerators.gpu", Op::Exists, Json(false))).size(), size_t(2));
    // matches: MMQP filter overlap, "topic" also consults topics/filter(s)
    EMPY_CHECK(ids(One(NETWORK, "topic", Op::Matches, Json("home/temp"))) ==
               std::vector<std::string>({"topics/home/#"}));
    EMPY_CHECK(ids(One(NETWORK, "topic", Op::Matches, Json("factory/+/speed"))) ==
               std::vector<std::string>({"shard/cloud"}));
    EMPY_CHECK(ids(One(NETWORK, "topic", Op::Matches, Json("vehicle/#"))).empty());
    // entry fields
    EMPY_CHECK(ids(One(RESOURCES, "updated", Op::Gt, Json(150))) == std::vector<std::string>({"cloud-1", "ran-1"}));
    EMPY_CHECK_EQ(ids(One(NETWORK, "collection", Op::Eq, Json("network"))).size(), size_t(3));
    EMPY_CHECK(ids(One(RESOURCES, "data.cpu", Op::Eq, Json(64))) == std::vector<std::string>({"cloud-1"}));

    // several conditions all hold; no collections = all
    std::vector<Entry> both = Run(plane, Json::Parse(R"({"where": [{"field": "cpu", "op": ">=", "value": 4},
        {"field": "cpu", "op": "<", "value": 64}]})"));
    EMPY_CHECK(Ids(both) == std::vector<std::string>({"ran-1", "edge-1"}));

    // ordering, limit and projection
    std::vector<Entry> ordered =
        Run(plane, Json::Parse(R"({"collections": ["resources"], "orderBy": "cpu", "descending": true, "limit": 2,
            "select": ["cpu", "labels.site"]})"));
    EMPY_REQUIRE_EQ(ordered.size(), size_t(2));
    EMPY_CHECK_EQ(ordered[0].m_Id, "cloud-1");
    EMPY_CHECK_EQ(ordered[1].m_Id, "edge-1");
    EMPY_CHECK(ordered[0].m_Data == Json::Parse(R"({"cpu": 64})"));
    EMPY_CHECK(ordered[1].m_Data == Json::Parse(R"({"cpu": 16, "labels.site": "metro"})"));
    std::vector<Entry> ascending =
        Run(plane, Json::Parse(R"({"collections": ["resources"], "orderBy": "labels.site"})"));
    EMPY_CHECK(Ids(ascending) == std::vector<std::string>({"edge-1", "ran-1", "cloud-1"})); // missing last
    // default order: score, then most recent first
    EMPY_CHECK(Ids(Run(plane, Json::Parse(R"({"collections": ["resources"]})"))) ==
               std::vector<std::string>({"cloud-1", "ran-1", "edge-1"}));

    // op names, aliases and the JSON form
    for (Op op : {Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge, Op::Contains, Op::In, Op::Exists, Op::Matches})
    {
        EMPY_CHECK(ParseOp(OpName(op)) == op);
    }
    EMPY_CHECK(ParseOp("!=") == Op::Ne);
    EMPY_CHECK(ParseOp(" EQ ") == Op::Eq);
    EMPY_CHECK(!ParseOp("like"));
    Query query;
    query.m_Collections = {RESOURCES, NETWORK};
    query.m_Where = {Condition{"region", Op::In, Json::Parse(R"(["ran"])")},
                     Condition{"topic", Op::Matches, Json("home/#")}};
    query.m_Text = {"gpu", "capacity"};
    query.m_OrderBy = "cpu";
    query.m_Descending = true;
    query.m_Limit = 7;
    query.m_Select = {"region"};
    Query back = Query::FromJson(Json::Parse(query.ToJson().Dump()));
    EMPY_CHECK(back.ToJson() == query.ToJson());
    EMPY_CHECK_EQ(back.m_Where.size(), size_t(2));
    EMPY_CHECK(back.m_Where[1].m_Op == Op::Matches);
    EMPY_CHECK_EQ(size_t(Query::FromJson(Json::Parse(R"({"limit": 100000})")).m_Limit), size_t(1000));
    EMPY_CHECK(Query::FromJson(Json::Parse(R"({"text": "gpu nodes"})")).m_Text ==
               std::vector<std::string>({"gpu nodes"}));
    EMPY_CHECK_THROWS(Query::FromJson(Json(1)));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"from": "resources"})")));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"where": [{"field": "a", "op": "like", "value": 1}]})")));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"where": [{"field": "", "op": "eq"}]})")));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"where": [{"field": "a", "op": "eq", "x": 1}]})")));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"limit": -1})")));
    EMPY_CHECK_THROWS(Query::FromJson(Json::Parse(R"({"collections": [3]})")));

    // the structured-output schema is strict and accepts the query documents FromJson reads
    Json schema = Query::Schema();
    std::string problem;
    EMPY_CHECK(StrictSchema(schema, problem));
    EMPY_CHECK_EQ(problem, "");
    std::string violation;
    EMPY_CHECK(Intelligence::Conforms(query.ToJson(), schema, &violation));
    EMPY_CHECK(Intelligence::Conforms(Query().ToJson(), schema, &violation));
    EMPY_CHECK(!Intelligence::Conforms(Json::Parse(R"({"collections": ["galaxy"], "where": [], "text": [],
        "orderBy": "", "descending": false, "limit": 5, "select": []})"),
                                       schema));
}

EMPY_TEST(OpsContext, TextRelevance)
{
    KnowledgePlane plane;
    plane.Put(Make(MEMORY, "a", Json::Parse(R"({"note": "retrained the thermostat model after drift"})"), {}, 10));
    plane.Put(Make(MEMORY, "b", Json::Parse(R"({"note": "thermostat setpoint changed"})"), {"comfort"}, 20));
    plane.Put(Make(MEMORY, "c", Json::Parse(R"({"note": "unrelated"})"), {}, 30));
    plane.Put(Make(MEMORY, "d", Json::Parse(R"({"note": "thermodynamics"})"), {}, 40));
    Query query;
    query.m_Text = {"Thermostat retraining"};
    std::vector<Entry> results = LocalSource(plane).Execute(query);
    EMPY_REQUIRE_EQ(results.size(), size_t(4));
    EMPY_CHECK_EQ(results[0].m_Id, "a"); // both keywords (stemmed "retrain")
    EMPY_CHECK_EQ(results[0].m_Score, 1.0);
    EMPY_CHECK_EQ(results[1].m_Id, "b");
    EMPY_CHECK_EQ(results[1].m_Score, 0.5);
    EMPY_CHECK_EQ(results[2].m_Id, "d"); // "thermo..." prefix of a longer token is not a match; tie by recency
    EMPY_CHECK_EQ(results[2].m_Score, 0.0);
    EMPY_CHECK_EQ(results[3].m_Score, 0.0);
    // tags and ids count as entry text; a keyword prefix earns half credit
    Query tagged;
    tagged.m_Text = {"comfort"};
    EMPY_CHECK_EQ(std::string(Ids(LocalSource(plane).Execute(tagged)).front()), "b");
    Query prefix;
    prefix.m_Text = {"thermodyn"};
    std::vector<Entry> partial = LocalSource(plane).Execute(prefix);
    EMPY_CHECK_EQ(partial[0].m_Id, "d");
    EMPY_CHECK_EQ(partial[0].m_Score, 0.5);
    // no keywords: every entry scores 1
    std::vector<Entry> all = LocalSource(plane).Execute(Query());
    EMPY_CHECK_EQ(all[0].m_Score, 1.0);
}

EMPY_TEST(OpsContext, SeedShippedKnowledge)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    Topology::Inventory inventory = plane.Inventory();
    EMPY_REQUIRE_EQ(inventory.m_Nodes.size(), size_t(6));
    const Topology::Node *ran = inventory.Find("ran-1");
    EMPY_REQUIRE(ran != nullptr);
    EMPY_CHECK_EQ(ran->m_Region, "ran");
    EMPY_CHECK_EQ(ran->m_Cpu, 4.0);
    EMPY_CHECK_EQ(ran->m_Memory, uint64_t(8) << 30);
    EMPY_CHECK_EQ(ran->FreeAccelerators("npu"), uint32_t(1));
    EMPY_CHECK(ran->m_Broker.empty()); // no node-local broker: instances join the harness broker
    EMPY_CHECK_EQ(ran->m_Labels.at("site"), "tower-a");
    EMPY_CHECK_EQ(inventory.Find("cloud-1")->FreeAccelerators("gpu"), uint32_t(8));
    EMPY_CHECK_EQ(inventory.Find("cloud-2")->FreeAccelerators("tpu"), uint32_t(4));
    std::set<std::string> regions;
    for (const Topology::Node &node : inventory.m_Nodes)
    {
        regions.insert(node.m_Region);
    }
    EMPY_CHECK(regions == std::set<std::string>({"ran", "edge", "cloud"}));

    Topology::NetworkModel network = plane.Network();
    EMPY_CHECK(network.m_Regions == std::vector<std::string>({"ran", "edge", "cloud"}));
    EMPY_CHECK_EQ(network.Latency("ran", "cloud"), 25.0);
    EMPY_CHECK_EQ(network.Latency("cloud", "edge"), 15.0);
    EMPY_CHECK_EQ(network.Latency("edge", "edge"), 0.5);
    EMPY_CHECK_EQ(network.Density("home/temp", "ran"), 40.0);
    EMPY_CHECK_EQ(network.Density("home/temp", "edge"), 5.0);
    EMPY_CHECK_EQ(network.Consumers("home/set/temp", "ran"), 10.0);
    EMPY_CHECK(network.OwnsShard("ran", "home/temp"));
    EMPY_CHECK(network.OwnsShard("cloud", "factory/line1/speed"));
    EMPY_CHECK(!network.OwnsShard("edge", "home/temp"));

    EMPY_CHECK_EQ(plane.Size(TOOLS), size_t(5));
    EMPY_CHECK_EQ(plane.Size(OPERATOR), size_t(5));
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(0));
    std::optional<Entry> kubectl = plane.Get(TOOLS, "kubectl");
    EMPY_REQUIRE(kubectl.has_value());
    EMPY_CHECK(std::find(kubectl->m_Tags.begin(), kubectl->m_Tags.end(), "deploy") != kubectl->m_Tags.end());
    std::optional<Entry> policy = plane.Get(OPERATOR, "site-policy");
    EMPY_REQUIRE(policy.has_value());
    EMPY_CHECK_EQ(policy->m_Data.GetString("text"), "Prefer RAN placement for closed-loop home control.");
    EMPY_CHECK(policy->m_Tags == std::vector<std::string>({"home", "placement", "ran"}));
    // derived network entries make links, densities and shards retrievable
    std::optional<Entry> link = plane.Get(NETWORK, "latency/ran/cloud");
    EMPY_REQUIRE(link.has_value());
    EMPY_CHECK_EQ(link->m_Data.GetNumber("ms"), 25.0);
    EMPY_CHECK(plane.Get(NETWORK, "topics/home/#").has_value());
    EMPY_CHECK(plane.Get(NETWORK, "consumers/home/set/#").has_value());
    EMPY_CHECK(plane.Get(NETWORK, "shard/ran").has_value());
    std::optional<Entry> node = plane.Get(RESOURCES, "edge-2");
    EMPY_REQUIRE(node.has_value());
    EMPY_CHECK_EQ(node->m_Data.GetInt("memoryBytes"), int64_t(64) << 30);

    // seeding again is idempotent
    size_t networkEntries = plane.Size(NETWORK);
    SeedShipped(plane);
    EMPY_CHECK_EQ(plane.Size(RESOURCES), size_t(6));
    EMPY_CHECK_EQ(plane.Size(NETWORK), networkEntries);

    // invalid seeds are rejected as a whole
    KnowledgePlane fresh;
    auto seed = [&fresh](std::string_view yaml) { fresh.Seed(eCore::YamlNode::Parse(yaml, "<seed>")); };
    EMPY_CHECK_THROWS(seed("resources: []\nservices: []\n"));
    EMPY_CHECK_THROWS(seed("resources:\n  - {id: a, cpu: 1}\n  - {id: a, cpu: 2}\n"));
    EMPY_CHECK_THROWS(seed("resources:\n  - {id: a, cpu: -1}\n"));
    EMPY_CHECK_THROWS(seed("tools:\n  - {kind: tool}\n"));
    EMPY_CHECK_THROWS(seed("resources:\n  - {id: ok, cpu: 1}\noperator:\n  - {id: p, tags: [x]}\n"));
    EMPY_CHECK_THROWS(seed("network: {latency: {ran: {edge: fast}}}\n"));
    EMPY_CHECK(fresh.Collections().empty());
    try
    {
        seed("tools:\n  - {id: t, capabilities: deploy}\n");
        EMPY_CHECK(false);
    }
    catch (const eCore::ConfigError &error)
    {
        EMPY_CHECK(std::string(error.what()).find("<seed>.tools[0].capabilities") != std::string::npos);
    }
    seed("memory:\n  - {agent: lifecycle, kind: decision, service: home-control, data: {action: retrain}}\n");
    std::vector<Entry> memory = fresh.Scan(MEMORY);
    EMPY_REQUIRE_EQ(memory.size(), size_t(1));
    EMPY_CHECK(memory[0].m_Tags == std::vector<std::string>({"lifecycle", "decision", "home-control"}));
}

EMPY_TEST(OpsContext, UpdateNetwork)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    uint64_t linkUpdated = plane.Get(NETWORK, "latency/ran/cloud")->m_Updated;
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    plane.UpdateNetwork([](Topology::NetworkModel &model)
                        { model.ApplyLocality("ran", Json::Parse(R"({"home/temp": 12.5, "home/hum": 3})")); });
    Topology::NetworkModel network = plane.Network();
    EMPY_CHECK_EQ(network.Density("home/temp", "ran"), 52.5); // home/# 40 + home/temp 12.5
    std::optional<Entry> density = plane.Get(NETWORK, "topics/home/temp");
    EMPY_REQUIRE(density.has_value());
    EMPY_CHECK_EQ(density->m_Data["rates"].GetNumber("ran"), 12.5);
    EMPY_CHECK_EQ(uint64_t(plane.Get(NETWORK, "latency/ran/cloud")->m_Updated), linkUpdated); // unchanged entries kept

    // removed filters drop their derived entries
    plane.UpdateNetwork([](Topology::NetworkModel &model) { model.m_TopicDensity.erase("home/hum"); });
    EMPY_CHECK(!plane.Get(NETWORK, "topics/home/hum").has_value());
    EMPY_CHECK(plane.Get(NETWORK, "topics/home/temp").has_value());

    // an exception inside the update leaves the plane unchanged
    size_t before = plane.Size(NETWORK);
    EMPY_CHECK_THROWS(plane.UpdateNetwork(
        [](Topology::NetworkModel &model)
        {
            model.m_LatencyMs["ran"]["cloud"] = 99;
            throw std::runtime_error("abort");
        }));
    EMPY_CHECK_EQ(plane.Network().Latency("ran", "cloud"), 25.0);
    EMPY_CHECK_EQ(plane.Size(NETWORK), before);

    // an empty plane starts from an empty model
    KnowledgePlane empty;
    EMPY_CHECK(empty.Network().m_Regions.empty());
    empty.UpdateNetwork([](Topology::NetworkModel &model)
                        { model.ApplyLocality("edge", Json::Parse(R"({"factory/line1/vibration": 7})")); });
    EMPY_CHECK_EQ(empty.Network().Density("factory/#", "edge"), 7.0);
    EMPY_CHECK(empty.Network().m_Regions == std::vector<std::string>({"edge"}));
}

EMPY_TEST(OpsContext, DeterministicTranslation)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    Comas comas(plane);
    Json context = comas.Augment("");

    // "which edge nodes have a gpu" → resources, accelerator and region conditions, no residue
    Query gpu = comas.TranslateDeterministic("which edge nodes have a gpu", context);
    EMPY_CHECK(gpu.m_Collections == std::vector<std::string>({"resources"}));
    EMPY_REQUIRE_EQ(gpu.m_Where.size(), size_t(2));
    EMPY_CHECK_EQ(gpu.m_Where[0].m_Field, "accelerators.gpu");
    EMPY_CHECK(gpu.m_Where[0].m_Op == Op::Exists);
    EMPY_CHECK_EQ(gpu.m_Where[1].m_Field, "region");
    EMPY_CHECK(gpu.m_Where[1].m_Op == Op::Eq);
    EMPY_CHECK(gpu.m_Where[1].m_Value == Json("edge"));
    EMPY_CHECK(gpu.m_Text.empty());
    EMPY_CHECK(Ids(comas.Execute(gpu)) == std::vector<std::string>({"edge-2", "edge-1"}) ||
               Ids(comas.Execute(gpu)) == std::vector<std::string>({"edge-1", "edge-2"}));
    // plural accelerator acronyms fold like the ontology terms: "gpus" is the gpu condition, not residual text
    Query plural = comas.TranslateDeterministic("which nodes have GPUs", context);
    EMPY_CHECK(plural.m_Collections == std::vector<std::string>({"resources"}));
    EMPY_REQUIRE_EQ(plural.m_Where.size(), size_t(1));
    EMPY_CHECK_EQ(plural.m_Where[0].m_Field, "accelerators.gpu");
    EMPY_CHECK(plural.m_Where[0].m_Op == Op::Exists);
    EMPY_CHECK(plural.m_Text.empty());
    EMPY_CHECK(!comas.Execute(plural).empty());

    // "latency between ran and cloud" → the network link touching both regions
    Query latency = comas.TranslateDeterministic("latency between ran and cloud", context);
    EMPY_CHECK(latency.m_Collections == std::vector<std::string>({"network"}));
    EMPY_REQUIRE_EQ(latency.m_Where.size(), size_t(2));
    EMPY_CHECK(latency.m_Where[0].m_Value == Json("ran"));
    EMPY_CHECK(latency.m_Where[1].m_Value == Json("cloud"));
    std::vector<Entry> links = comas.Execute(latency);
    EMPY_REQUIRE_EQ(links.size(), size_t(1));
    EMPY_CHECK_EQ(links[0].m_Id, "latency/ran/cloud");

    // "previous retraining decisions for home-control" → agents memory filtered by service
    Query history = comas.TranslateDeterministic("previous retraining decisions for home-control", context);
    EMPY_CHECK(history.m_Collections == std::vector<std::string>({"memory", "consolidated"}));
    EMPY_REQUIRE_EQ(history.m_Where.size(), size_t(1));
    EMPY_CHECK_EQ(history.m_Where[0].m_Field, "service");
    EMPY_CHECK(history.m_Where[0].m_Value == Json("home-control"));
    EMPY_CHECK(history.m_Text == std::vector<std::string>({"retrain", "decision"}));

    // more vocabulary: tools, operator context, observations, topics, region sets, ordering
    Query tools = comas.TranslateDeterministic("Which tools can deploy?", context);
    EMPY_CHECK(tools.m_Collections == std::vector<std::string>({"tools"}));
    EMPY_CHECK(tools.m_Text == std::vector<std::string>({"deploy"}));
    Query policy = comas.TranslateDeterministic("operator placement policy", context);
    EMPY_CHECK(policy.m_Collections == std::vector<std::string>({"operator"}));
    EMPY_CHECK(policy.m_Text == std::vector<std::string>({"placement", "policy"}));
    Query density = comas.TranslateDeterministic("traffic density of home/temp", context);
    EMPY_CHECK(density.m_Collections == std::vector<std::string>({"network"}));
    EMPY_REQUIRE_EQ(density.m_Where.size(), size_t(1));
    EMPY_CHECK(density.m_Where[0].m_Op == Op::Matches);
    EMPY_CHECK(density.m_Where[0].m_Value == Json("home/temp"));
    std::vector<std::string> dense = Ids(comas.Execute(density));
    std::sort(dense.begin(), dense.end());
    EMPY_CHECK(dense == std::vector<std::string>({"shard/ran", "topics/home/#"})); // density and shard owner
    Query quoted = comas.TranslateDeterministic("status of 'energy-saver'", context);
    EMPY_CHECK(quoted.m_Collections == std::vector<std::string>({"observations"}));
    EMPY_REQUIRE_EQ(quoted.m_Where.size(), size_t(1));
    EMPY_CHECK(quoted.m_Where[0].m_Value == Json("energy-saver"));
    Query set = comas.TranslateDeterministic("nodes in the radio access network or the data center", context);
    EMPY_CHECK(set.m_Collections == std::vector<std::string>({"resources", "network"}));
    EMPY_REQUIRE_EQ(set.m_Where.size(), size_t(1));
    EMPY_CHECK(set.m_Where[0].m_Op == Op::In);
    EMPY_CHECK(set.m_Where[0].m_Value == Json::Parse(R"(["ran", "cloud"])"));
    Query most = comas.TranslateDeterministic("node with the most cpu", context);
    EMPY_CHECK_EQ(most.m_OrderBy, "cpu");
    EMPY_CHECK(most.m_Descending);
    EMPY_CHECK_EQ(std::string(Ids(comas.Execute(most)).front()), "cloud-2");
    // mixed collections keep region words as keywords rather than excluding entries
    Query mixed = comas.TranslateDeterministic("gpu placement policy at the edge", context);
    EMPY_CHECK(mixed.m_Collections == std::vector<std::string>({"resources", "operator"}));
    EMPY_CHECK(mixed.m_Where.empty());
    EMPY_CHECK(std::find(mixed.m_Text.begin(), mixed.m_Text.end(), "edge") != mixed.m_Text.end());
    EMPY_CHECK(std::find(mixed.m_Text.begin(), mixed.m_Text.end(), "gpu") != mixed.m_Text.end());
    // no vocabulary: all collections with residual keywords
    Query residue = comas.TranslateDeterministic("thermostat", context);
    EMPY_CHECK(residue.m_Collections.empty());
    EMPY_CHECK(residue.m_Text == std::vector<std::string>({"thermostat"}));

    std::string translator;
    comas.Translate("which edge nodes have a gpu", context, &translator);
    EMPY_CHECK_EQ(translator, "deterministic");
}

EMPY_TEST(OpsContext, LlmTranslation)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    auto fake = std::make_shared<FakeProvider>();
    Intelligence::ProviderSet providers;
    providers.Set(Intelligence::Capability::Reasoning, fake);
    Comas comas(plane, &providers);
    Json context = comas.Augment("cloud tpu");

    // a conforming proposal is used as x2
    fake->m_Result.m_Ok = true;
    fake->m_Result.m_Json = Json::Parse(R"({"collections": ["resources"],
        "where": [{"field": "accelerators.tpu", "op": "exists", "value": true}], "text": [], "orderBy": "",
        "descending": false, "limit": 5, "select": []})");
    std::string translator;
    Query query = comas.Translate("where can I train on a tpu", context, &translator);
    EMPY_CHECK_EQ(translator, "llm:fake");
    EMPY_CHECK_EQ(fake->m_Calls.load(), 1);
    EMPY_REQUIRE_EQ(query.m_Where.size(), size_t(1));
    EMPY_CHECK_EQ(query.m_Where[0].m_Field, "accelerators.tpu");
    EMPY_CHECK_EQ(query.m_Limit, size_t(5));
    EMPY_CHECK(fake->m_Seen.m_Capability == Intelligence::Capability::Reasoning);
    EMPY_REQUIRE(fake->m_Seen.m_Schema.has_value());
    EMPY_CHECK(*fake->m_Seen.m_Schema == Query::Schema());
    Json user = Json::Parse(fake->m_Seen.m_User);
    EMPY_CHECK_EQ(user.GetString("intent"), "where can I train on a tpu");
    EMPY_CHECK(user["context"] == context);
    EMPY_CHECK(!fake->m_Seen.m_System.empty());
    RetrievalResult result = comas.Retrieve({"placement", "where can I train on a tpu", 20});
    EMPY_CHECK_EQ(result.m_Translator, "llm:fake");
    EMPY_CHECK(Ids(result.m_Entries) == std::vector<std::string>({"cloud-2"}));

    // invalid output, unknown collections, empty queries and failures fall back to the deterministic translator
    for (const char *proposal :
         {R"({"collections": ["resources"]})",
          R"({"collections": ["galaxy"], "where": [], "text": [], "orderBy": "", "descending": false, "limit": 5,
              "select": []})",
          R"({"collections": [], "where": [], "text": [], "orderBy": "", "descending": false, "limit": 5,
              "select": []})",
          R"({"collections": ["resources"], "where": [{"field": "", "op": "eq", "value": null}], "text": [],
              "orderBy": "", "descending": false, "limit": 5, "select": []})"})
    {
        fake->m_Result.m_Json = Json::Parse(proposal);
        Query fallback = comas.Translate("which edge nodes have a gpu", context, &translator);
        EMPY_CHECK_EQ(translator, "deterministic(fallback)");
        EMPY_CHECK(fallback.m_Collections == std::vector<std::string>({"resources"}));
        EMPY_CHECK_EQ(fallback.m_Where.size(), size_t(2));
    }
    fake->m_Result.m_Json.reset();
    fake->m_Result.m_Text = "I cannot produce JSON today.";
    comas.Translate("which edge nodes have a gpu", context, &translator);
    EMPY_CHECK_EQ(translator, "deterministic(fallback)");
    fake->m_Result.m_Ok = false;
    fake->m_Result.m_Error = "overloaded";
    comas.Translate("which edge nodes have a gpu", context, &translator);
    EMPY_CHECK_EQ(translator, "deterministic(fallback)");
    // an unavailable provider is never asked
    int32_t calls = fake->m_Calls.load();
    fake->m_Available = false;
    comas.Translate("which edge nodes have a gpu", context, &translator);
    EMPY_CHECK_EQ(translator, "deterministic");
    EMPY_CHECK_EQ(fake->m_Calls.load(), calls);
}

EMPY_TEST(OpsContext, RetrieveFlow)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    Comas comas(plane);
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"action": "retrain", "reason": "drift"})"));
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"action": "promote", "version": 2})"));
    comas.Remember("lifecycle", "decision", "energy-saver", Json::Parse(R"({"action": "retrain"})"));
    comas.Remember("placement", "placement", "home-control", Json::Parse(R"({"nodes": ["ran-1"]})"));

    // c: collections, regions, services, operator context, recent memory
    const Json context = comas.Augment("home placement", 3);
    EMPY_CHECK_EQ(context["collections"].GetInt("resources"), 6);
    EMPY_CHECK(context["regions"] == Json::Parse(R"(["cloud", "edge", "ran"])"));
    EMPY_CHECK(context["accelerators"] == Json::Parse(R"(["gpu", "npu", "tpu"])"));
    EMPY_CHECK(context["services"] == Json::Parse(R"(["energy-saver", "home-control"])"));
    EMPY_REQUIRE_EQ(context["operator"].Size(), size_t(3));
    EMPY_CHECK_EQ(context["operator"][size_t(0)].GetString("id"), "site-policy"); // home + placement
    EMPY_REQUIRE_EQ(context["recent"].Size(), size_t(3));
    EMPY_CHECK_EQ(context["recent"][size_t(0)].GetString("agent"), "placement");

    // R for "which edge nodes have a gpu"
    RetrievalResult gpu = comas.Retrieve({"placement", "which edge nodes have a gpu", 20});
    EMPY_CHECK_EQ(gpu.m_Translator, "deterministic");
    EMPY_CHECK_EQ(gpu.m_Entries.size(), size_t(2));
    EMPY_CHECK_EQ(gpu.m_Aggregates.GetInt("total"), 2);
    EMPY_CHECK(gpu.m_Aggregates["count"] == Json::Parse(R"({"resources": 2})"));
    EMPY_CHECK(gpu.m_Aggregates["regions"] == Json::Parse(R"({"edge": 2})"));
    EMPY_CHECK(gpu.m_Aggregates["accelerators"] == Json::Parse(R"({"gpu": 3})"));
    EMPY_CHECK(gpu.m_Summary.find("resources/edge-1: region=edge cpu=16 memory=32Gi accelerators=gpu:1\n") !=
               std::string::npos);
    EMPY_CHECK_EQ(gpu.m_Context.GetString("agent"), "placement");
    Json json = gpu.ToJson();
    for (const char *member : {"context", "query", "entries", "aggregates", "summary", "translator"})
    {
        EMPY_CHECK(json.Has(member));
    }
    EMPY_CHECK_EQ(json["entries"].Size(), size_t(2));
    EMPY_CHECK(Query::FromJson(json["query"]).ToJson() == gpu.m_Query.ToJson());

    RetrievalResult latency = comas.Retrieve({"placement", "latency between ran and cloud", 20});
    EMPY_REQUIRE_EQ(latency.m_Entries.size(), size_t(1));
    EMPY_CHECK_EQ(latency.m_Summary, "network/latency/ran/cloud: latency ran-cloud 25 ms\n");

    RetrievalResult history = comas.Retrieve({"lifecycle", "previous retraining decisions for home-control", 20});
    EMPY_REQUIRE_EQ(history.m_Entries.size(), size_t(3)); // home-control only, conditions keep zero scores
    EMPY_CHECK_EQ(history.m_Entries[0].m_Data["data"].GetString("action"), "retrain");
    for (const Entry &entry : history.m_Entries)
    {
        EMPY_CHECK_EQ(entry.m_Data.GetString("service"), "home-control");
    }
    EMPY_CHECK(history.m_Summary.find("lifecycle decision home-control: ") != std::string::npos);

    // request limit caps the query
    RetrievalResult capped = comas.Retrieve({"executive", "nodes", 2});
    EMPY_CHECK_EQ(capped.m_Entries.size(), size_t(2));
    EMPY_CHECK_EQ(capped.m_Query.m_Limit, size_t(2));

    // Contextualise: dedupe, drop zero scores for keyword-only queries, bound the summary
    Query keywords;
    keywords.m_Text = {"thermostat"};
    Entry hit = Make(MEMORY, "hit", Json("thermostat"));
    hit.m_Score = 1;
    Entry miss = Make(MEMORY, "miss", Json("other"));
    RetrievalResult filtered =
        comas.Contextualise({"x", "y", 10}, Json::MakeObject(), keywords, {hit, hit, miss}, "deterministic");
    EMPY_CHECK(Ids(filtered.m_Entries) == std::vector<std::string>({"hit"}));
    keywords.m_Where.push_back(Condition{"id", Op::Exists, Json(true)});
    EMPY_CHECK_EQ(comas.Contextualise({"x", "y", 10}, Json(), keywords, {hit, miss}, "d").m_Entries.size(), size_t(2));
    std::vector<Entry> many;
    for (int32_t i = 0; i < 100; ++i)
    {
        many.push_back(Make(OBSERVATIONS, "o" + std::to_string(i),
                            Json(Json::Object{{"source", "w"}, {"kind", "metrics"}, {"data", std::string(300, 'v')}})));
    }
    RetrievalResult bounded = comas.Contextualise({"x", "y", 100}, Json(), Query(), many, "deterministic");
    EMPY_CHECK_EQ(bounded.m_Entries.size(), size_t(100));
    EMPY_CHECK(bounded.m_Summary.size() <= 4096);
    EMPY_CHECK(bounded.m_Summary.find(" more)\n") != std::string::npos);
    EMPY_CHECK_EQ(bounded.m_Aggregates["count"].GetInt("observations"), 100);
}

EMPY_TEST(OpsContext, MemoryAndConsolidation)
{
    KnowledgePlane plane;
    Comas comas(plane);
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"action": "retrain"})"));
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"outcome": "rejected"})"));
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"status": "promoted", "version": 3})"));
    comas.Remember("compliance", "review", "energy-saver", Json::Parse(R"({"verdict": "Approved"})"));
    comas.Remember("executive", "interpretation", "", Json::Parse(R"({"ok": false})"));
    std::vector<Entry> memory = plane.Scan(MEMORY);
    EMPY_REQUIRE_EQ(memory.size(), size_t(5));
    EMPY_CHECK(memory[0].m_Id < memory[4].m_Id || memory[0].m_Updated <= memory[4].m_Updated);
    EMPY_CHECK_EQ(memory[0].m_Data.GetString("agent"), "lifecycle");
    EMPY_CHECK_EQ(memory[0].m_Data.GetString("outcome"), "retrain");
    EMPY_CHECK(memory[0].m_Tags == std::vector<std::string>({"lifecycle", "decision", "home-control"}));

    comas.Observe("placement", "locality",
                  Json::Parse(R"({"region": "ran", "service": "home-control", "topics": {"home/temp": 3}})"));
    std::vector<Entry> observations = plane.Scan(OBSERVATIONS);
    EMPY_REQUIRE_EQ(observations.size(), size_t(1));
    EMPY_CHECK_EQ(observations[0].m_Data.GetString("region"), "ran");
    EMPY_CHECK_EQ(observations[0].m_Data.GetString("service"), "home-control");
    EMPY_CHECK_EQ(observations[0].m_Data["data"]["topics"].GetNumber("home/temp"), 3.0);
    EMPY_CHECK(observations[0].m_Tags == std::vector<std::string>({"placement", "locality"}));

    // nothing is old enough yet
    uint64_t now = NowMillis();
    EMPY_CHECK_EQ(comas.Consolidate(now, 3600000), size_t(0));
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(5));

    // two hours later every episodic entry is grouped by (service, kind)
    EMPY_CHECK_EQ(comas.Consolidate(now + 7200000, 3600000), size_t(3));
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(0));
    std::optional<Entry> decisions = plane.Get(CONSOLIDATED, "home-control/decision");
    EMPY_REQUIRE(decisions.has_value());
    EMPY_CHECK_EQ(decisions->m_Data.GetInt("count"), 3);
    EMPY_CHECK_EQ(decisions->m_Data.GetInt("successes"), 1);
    EMPY_CHECK_EQ(decisions->m_Data.GetInt("failures"), 1);
    EMPY_CHECK_EQ(decisions->m_Data.GetString("lastOutcome"), "promoted");
    EMPY_CHECK_EQ(decisions->m_Data["last"].GetInt("version"), 3);
    EMPY_CHECK(decisions->m_Data["agents"] == Json::Parse(R"(["lifecycle"])"));
    EMPY_CHECK(decisions->m_Data.GetInt("lastAt") >= decisions->m_Data.GetInt("firstAt"));
    EMPY_CHECK(decisions->m_Tags == std::vector<std::string>({"home-control", "decision", "consolidated"}));
    std::optional<Entry> review = plane.Get(CONSOLIDATED, "energy-saver/review");
    EMPY_REQUIRE(review.has_value());
    EMPY_CHECK_EQ(review->m_Data.GetInt("successes"), 1);
    std::optional<Entry> anonymous = plane.Get(CONSOLIDATED, "-/interpretation");
    EMPY_REQUIRE(anonymous.has_value());
    EMPY_CHECK_EQ(anonymous->m_Data.GetInt("failures"), 1);

    // later episodes merge into the existing summaries; recent ones stay episodic
    plane.Put(Make(MEMORY, "old", Json::Parse(R"({"agent": "lifecycle", "kind": "decision", "service": "home-control",
                                   "data": {"action": "rollback"}, "outcome": "rollback"})"),
                   {}, now - 3600000 - 10));
    comas.Remember("lifecycle", "decision", "home-control", Json::Parse(R"({"action": "retrain"})"));
    EMPY_CHECK_EQ(comas.Consolidate(NowMillis(), 3600000), size_t(1));
    decisions = plane.Get(CONSOLIDATED, "home-control/decision");
    EMPY_CHECK_EQ(decisions->m_Data.GetInt("count"), 4);
    EMPY_CHECK_EQ(decisions->m_Data.GetInt("failures"), 2);
    EMPY_CHECK_EQ(decisions->m_Data.GetString("lastOutcome"), "promoted"); // the merged episode is older
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(1));

    // consolidated knowledge answers history questions
    RetrievalResult history = comas.Retrieve({"lifecycle", "previous decisions for home-control", 20});
    EMPY_CHECK(std::any_of(history.m_Entries.begin(), history.m_Entries.end(),
                           [](const Entry &entry) { return entry.m_Collection == CONSOLIDATED; }));
    EMPY_CHECK(history.m_Summary.find("home-control decision count=4 successes=1 failures=2 last=promoted") !=
               std::string::npos);

    // concurrent memory, observation and retrieval traffic stays consistent
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < 4; ++t)
    {
        threads.emplace_back(
            [&comas, t]
            {
                for (int32_t i = 0; i < 50; ++i)
                {
                    comas.Remember("agent" + std::to_string(t), "kind", "svc", Json(i));
                    comas.Observe("w", "metrics", Json(Json::Object{{"value", i}}));
                    comas.Retrieve({"x", "recent decisions for svc", 5});
                }
            });
    }
    for (std::thread &thread : threads)
    {
        thread.join();
    }
    EMPY_CHECK_EQ(plane.Size(MEMORY), size_t(201));
    EMPY_CHECK_EQ(plane.Size(OBSERVATIONS), size_t(201));
}

EMPY_TEST(OpsContext, McpServerHandle)
{
    KnowledgePlane plane;
    SeedShipped(plane);
    Comas comas(plane);
    McpServer server(comas);

    // tools are refused before initialize; ping is always answered
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Rpc(9, "tools/list"))), -32600);
    Json pong = server.Handle(Rpc(10, "ping"));
    EMPY_CHECK(pong["result"] == Json::MakeObject());
    EMPY_CHECK_EQ(pong["id"].AsInt(), 10);

    Json init = Initialize(server);
    EMPY_CHECK_EQ(std::string(init["jsonrpc"].AsString()), "2.0");
    EMPY_CHECK_EQ(init["id"].AsInt(), 1);
    EMPY_CHECK_EQ(init["result"].GetString("protocolVersion"), "2025-06-18");
    EMPY_CHECK(init["result"]["capabilities"] == Json::Parse(R"({"tools": {}})"));
    EMPY_CHECK_EQ(init["result"]["serverInfo"].GetString("name"), "empy-comas");
    EMPY_CHECK(!init["result"]["serverInfo"].GetString("version").empty());
    // a supported older version is echoed, an unknown one gets ours
    Json older = server.Handle(Rpc(2, "initialize", Json::Parse(R"({"protocolVersion": "2025-03-26"})")));
    EMPY_CHECK_EQ(older["result"].GetString("protocolVersion"), "2025-03-26");
    Json future = server.Handle(Rpc(3, "initialize", Json::Parse(R"({"protocolVersion": "2099-01-01"})")));
    EMPY_CHECK_EQ(future["result"].GetString("protocolVersion"), "2025-06-18");
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Rpc(4, "initialize", Json::MakeObject()))), -32602);

    // notifications and responses produce no reply
    EMPY_CHECK(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "method": "notifications/initialized"})")).IsNull());
    EMPY_CHECK(server
                   .Handle(Json::Parse(R"({"jsonrpc": "2.0", "method": "notifications/cancelled",
        "params": {"requestId": 3}})"))
                   .IsNull());
    EMPY_CHECK(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "method": "no/such"})")).IsNull());
    EMPY_CHECK(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "id": 5, "result": {}})")).IsNull());

    Json list = server.Handle(Rpc(11, "tools/list"));
    const Json &tools = list["result"]["tools"];
    EMPY_REQUIRE_EQ(tools.Size(), size_t(4));
    std::vector<std::string> names;
    for (const Json &tool : tools.AsArray())
    {
        names.push_back(tool.GetString("name"));
        EMPY_CHECK_EQ(tool["inputSchema"].GetString("type"), "object");
        EMPY_CHECK(!tool.GetString("description").empty());
    }
    EMPY_CHECK(names == std::vector<std::string>(
                            {"knowledge_query", "knowledge_retrieve", "knowledge_put", "knowledge_collections"}));
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Rpc(12, "tools/list", Json::Parse(R"({"cursor": 5})")))), -32602);

    // knowledge_query
    Json query = Call(server, 20, "knowledge_query",
                      Json::Parse(R"({"collections": ["resources"], "where": [{"field": "region", "op": "eq",
                                      "value": "cloud"}], "orderBy": "id"})"));
    const Json &queried = query["result"];
    EMPY_CHECK(!queried.GetBool("isError"));
    EMPY_CHECK_EQ(queried["structuredContent"].GetInt("count"), 2);
    EMPY_CHECK_EQ(queried["structuredContent"]["entries"][size_t(0)].GetString("id"), "cloud-1");
    EMPY_CHECK_EQ(queried["content"][size_t(0)].GetString("type"), "text");
    EMPY_CHECK(Json::Parse(queried["content"][size_t(0)].GetString("text")) == queried["structuredContent"]);
    Json badQuery = Call(server, 21, "knowledge_query", Json::Parse(R"({"where": [{"field": "a", "op": "like"}]})"));
    EMPY_CHECK(badQuery["result"].GetBool("isError"));
    EMPY_CHECK(badQuery["result"]["structuredContent"].GetString("error").find("like") != std::string::npos);

    // knowledge_retrieve
    Json retrieve = Call(server, 22, "knowledge_retrieve",
                         Json::Parse(R"({"intent": "which edge nodes have a gpu", "agent": "placement", "limit": 5})"));
    const Json &retrieved = retrieve["result"]["structuredContent"];
    EMPY_CHECK(!retrieve["result"].GetBool("isError"));
    EMPY_CHECK_EQ(retrieved["entries"].Size(), size_t(2));
    EMPY_CHECK_EQ(retrieved.GetString("translator"), "deterministic");
    EMPY_CHECK_EQ(retrieved["context"].GetString("agent"), "placement");
    EMPY_CHECK(Call(server, 23, "knowledge_retrieve", Json::Parse(R"({"intent": ""})"))["result"].GetBool("isError"));
    EMPY_CHECK(Call(server, 24, "knowledge_retrieve", Json::Parse(R"({"intent": "x", "limit": 0})"))["result"].GetBool(
        "isError"));
    EMPY_CHECK(Call(server, 25, "knowledge_retrieve", Json::Parse(R"({"intent": "x", "extra": 1})"))["result"].GetBool(
        "isError"));

    // knowledge_put: writable collections only
    Json put = Call(server, 30, "knowledge_put",
                    Json::Parse(R"({"collection": "operator", "id": "night", "data": {"text": "Quiet hours"},
                                    "tags": ["schedule"]})"));
    EMPY_CHECK(!put["result"].GetBool("isError"));
    EMPY_CHECK(put["result"]["structuredContent"].GetBool("stored"));
    std::optional<Entry> stored = plane.Get(OPERATOR, "night");
    EMPY_REQUIRE(stored.has_value());
    EMPY_CHECK(stored->m_Tags == std::vector<std::string>({"schedule"}));
    Json generated = Call(server, 31, "knowledge_put", Json::Parse(R"({"collection": "memory", "data": {"a": 1}})"));
    EMPY_CHECK(plane.Get(MEMORY, generated["result"]["structuredContent"].GetString("id")).has_value());
    for (const char *arguments :
         {R"({"collection": "resources", "id": "x", "data": {}})", R"({"collection": "memory", "id": "x"})",
          R"({"collection": "memory", "id": "", "data": 1})",
          R"({"collection": "memory", "id": "x", "data": 1, "tags": "a"})",
          R"({"collection": "memory", "id": "x", "data": 1, "owner": "me"})"})
    {
        EMPY_CHECK(Call(server, 32, "knowledge_put", Json::Parse(arguments))["result"].GetBool("isError"));
    }
    EMPY_CHECK(!plane.Get(MEMORY, "x").has_value());

    // knowledge_collections
    Json collections = Call(server, 40, "knowledge_collections", Json());
    const Json &listed = collections["result"]["structuredContent"]["collections"];
    EMPY_CHECK(listed.Size() >= 5);
    EMPY_CHECK_EQ(listed[size_t(0)].GetString("name"), "memory");
    EMPY_CHECK(Call(server, 41, "knowledge_collections", Json::Parse(R"({"x": 1})"))["result"].GetBool("isError"));

    // protocol errors
    EMPY_CHECK_EQ(ErrorCode(Call(server, 50, "knowledge_delete", Json::MakeObject())), -32602);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Rpc(51, "tools/call", Json::Parse(R"({"arguments": {}})")))), -32602);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(
                      Rpc(52, "tools/call", Json::Parse(R"({"name": "knowledge_query", "arguments": [1]})")))),
                  -32602);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "id": 53, "method": "ping",
        "params": [1]})"))),
                  -32602);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Rpc(54, "resources/list"))), -32601);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Json::Parse(R"({"jsonrpc": "1.0", "id": 55, "method": "ping"})"))), -32600);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "id": 56})"))), -32600);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "id": 1.5, "method": "ping"})"))), -32600);
    EMPY_CHECK_EQ(ErrorCode(server.Handle(Json::Parse(R"([1, 2])"))), -32600);
    Json stringId = server.Handle(Json::Parse(R"({"jsonrpc": "2.0", "id": "abc", "method": "ping"})"));
    EMPY_CHECK_EQ(stringId.GetString("id"), "abc");

    // stdio loop: one response per request line, parse errors, blank lines, oversized lines, EOF
    std::string longLine(1100000, ' ');
    std::istringstream in(Rpc(1, "ping").Dump() + "\r\n\n   \n{broken\n" +
                          R"({"jsonrpc": "2.0", "method": "notifications/initialized"})" + "\n" + longLine + "x\n" +
                          Rpc(2, "tools/list").Dump());
    std::ostringstream out;
    McpServer loop(comas);
    EMPY_CHECK_EQ(loop.Run(in, out), 0);
    std::vector<std::string> lines;
    for (const std::string &line : Split(out.str(), '\n'))
    {
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }
    EMPY_REQUIRE_EQ(lines.size(), size_t(4));
    EMPY_CHECK_EQ(Json::Parse(lines[0])["id"].AsInt(), 1);
    EMPY_CHECK_EQ(ErrorCode(Json::Parse(lines[1])), -32700);
    EMPY_CHECK(Json::Parse(lines[1])["id"].IsNull());
    EMPY_CHECK_EQ(ErrorCode(Json::Parse(lines[2])), -32700);
    EMPY_CHECK_EQ(Json::Parse(lines[3])["result"]["tools"].Size(), size_t(4)); // initialized by the notification
}

EMPY_TEST(OpsContext, McpClientFailures)
{
    McpClient client;
    std::string error;
    ProcessOptions missing;
    missing.m_Program = std::filesystem::temp_directory_path() / ("no-such-server-" + NewId({}));
    EMPY_CHECK(!client.Start(missing, &error, 2000));
    EMPY_CHECK(error.find("cannot launch") != std::string::npos);
    EMPY_CHECK(!client.Running());
    EMPY_CHECK_THROWS(client.Tools(100));

    // a program that prints non-protocol output and exits
    auto start = Clock::now();
    EMPY_CHECK(!client.Start(Command(Shell({"cmd", "/c", "echo", "not-json"}, {"/bin/sh", "-c", "echo not-json"})),
                             &error, 5000));
    EMPY_CHECK(error.find("handshake") != std::string::npos);
    EMPY_CHECK(Clock::now() - start < std::chrono::seconds(4));

    // a silent program times out within the requested bound and is stopped
    start = Clock::now();
    EMPY_CHECK(
        !client.Start(Command(Shell({"ping", "-n", "30", "127.0.0.1"}, {"/bin/sh", "-c", "sleep 30"})), &error, 500));
    EMPY_CHECK(error.find("timed out") != std::string::npos);
    if (error.find("timed out") == std::string::npos)
    {
        std::cout << "unexpected error: " << error << std::endl;
    }
    EMPY_CHECK(Clock::now() - start < std::chrono::seconds(6));
    EMPY_CHECK(!client.Running());

    // McpSource without a usable client yields nothing
    McpSource none(nullptr);
    EMPY_CHECK(none.Execute(Query()).empty());
    McpSource stopped(std::make_shared<McpClient>(), 200);
    EMPY_CHECK(stopped.Execute(Query()).empty());
}

// the MCP stdio server of the harness CLI ("EmpyHarness mcp-server --config FILE") in a child process
EMPY_TEST(OpsContext, McpClientSubprocess)
{
    TempDir temp;
#ifdef _WIN32
    std::filesystem::path harness = eTest::BinaryDirectory() / "EmpyHarness.exe";
#else
    std::filesystem::path harness = eTest::BinaryDirectory() / "EmpyHarness";
#endif
    EMPY_REQUIRE(std::filesystem::exists(harness));
    std::filesystem::path config = temp.m_Path / "Harness.yaml";
    std::string knowledge = KnowledgeFile().generic_string();
    WriteFileAtomic(config, "identity: mcp-test\nknowledge: \"" + knowledge +
                                "\"\nintelligence: {reasoning: {provider: deterministic}}\nsubstrate: {type: "
                                "simulated}\n");
    ProcessOptions options;
    options.m_Program = harness;
    options.m_Args = {"mcp-server", "--config", config.string()};
    options.m_WorkingDirectory = temp.m_Path;
    options.m_StderrFile = temp.m_Path / "mcp-server.log";

    auto client = std::make_shared<McpClient>();
    std::string error;
    EMPY_REQUIRE(client->Start(options, &error, 20000));
    EMPY_CHECK_EQ(error, "");
    EMPY_CHECK(client->Running());
    EMPY_CHECK_EQ(client->ServerInfo().GetString("protocolVersion"), MCP_PROTOCOL_VERSION);
    EMPY_CHECK_EQ(client->ServerInfo()["serverInfo"].GetString("name"), "empy-comas");
    EMPY_CHECK(client->Tools() == std::vector<std::string>({"knowledge_query", "knowledge_retrieve", "knowledge_put",
                                                            "knowledge_collections"}));

    Json collections = client->CallTool("knowledge_collections", Json());
    EMPY_CHECK(!collections.GetBool("isError"));
    EMPY_CHECK(collections["structuredContent"].GetInt("total") > 0);

    Json put = client->CallTool("knowledge_put",
                                Json::Parse(R"({"collection": "memory", "id": "mcp-1", "data": {"agent": "lifecycle",
            "kind": "decision", "service": "home-control", "data": {"action": "retrain"}}})"));
    EMPY_CHECK(!put.GetBool("isError"));
    Json rejected = client->CallTool("knowledge_put", Json::Parse(R"({"collection": "resources", "data": {}})"));
    EMPY_CHECK(rejected.GetBool("isError"));
    EMPY_CHECK_THROWS(client->CallTool("knowledge_drop", Json()));

    // the knowledge plane behind the MCP server is a CoMaS knowledge source
    KnowledgePlane local;
    Comas comas(local, nullptr, std::make_shared<McpSource>(client, 10000));
    std::vector<Entry> nodes = comas.Execute(One(RESOURCES, "accelerators.gpu", Op::Exists, Json(true)));
    std::vector<std::string> ids = Ids(nodes);
    std::sort(ids.begin(), ids.end());
    EMPY_CHECK(ids == std::vector<std::string>({"cloud-1", "edge-1", "edge-2"}));
    RetrievalResult history = comas.Retrieve({"lifecycle", "previous retraining decisions for home-control", 10});
    EMPY_REQUIRE_EQ(history.m_Entries.size(), size_t(1));
    EMPY_CHECK_EQ(history.m_Entries[0].m_Id, "mcp-1");
    const Json retrieve =
        client->CallTool("knowledge_retrieve", Json::Parse(R"({"intent": "latency between ran and cloud"})"));
    EMPY_CHECK_EQ(retrieve["structuredContent"]["entries"][size_t(0)].GetString("id"), "latency/ran/cloud");

    client->Stop();
    EMPY_CHECK(!client->Running());
    EMPY_CHECK_THROWS(client->Tools(100));
    EMPY_CHECK(comas.Execute(Query()).empty()); // a stopped server yields no entries
}

EMPY_TEST(OpsContext, ExternalSourceMergesLiveExperience)
{
    // with an external (MCP) knowledge source, what CoMaS accumulates at run time (agents memory, observations,
    // consolidated experience, network knowledge) is merged into R', the local copy of an entry winning
    struct External : KnowledgeSource
    {
        std::vector<Query> m_Queries;
        bool m_Fail = false;
        std::vector<Entry> Execute(const Query &query) override
        {
            m_Queries.push_back(query);
            if (m_Fail)
            {
                throw std::runtime_error("MCP server input is closed");
            }
            Entry node;
            node.m_Collection = RESOURCES;
            node.m_Id = "edge-9";
            node.m_Data = Json::MakeObject({{"region", "edge"}, {"cpu", 8}});
            node.m_Updated = 1;
            Entry stale;
            stale.m_Collection = MEMORY;
            stale.m_Id = "shared";
            stale.m_Data = Json::MakeObject({{"agent", "lifecycle"}, {"kind", "decision"}, {"outcome", "stale"}});
            stale.m_Updated = 1;
            return Evaluate(query, {node, stale});
        }
    };
    KnowledgePlane plane;
    auto external = std::make_shared<External>();
    Comas comas(plane, nullptr, external);
    comas.Remember("lifecycle", "decision", "home-control", Json::MakeObject({{"action", "retrain"}}));
    comas.Observe("harness", "failure", Json::MakeObject({{"instance", "home-control-inference-ran-1-0"}}));
    Entry shared;
    shared.m_Collection = MEMORY;
    shared.m_Id = "shared";
    shared.m_Data = Json::MakeObject({{"agent", "lifecycle"}, {"kind", "decision"}, {"outcome", "fresh"}});
    plane.Put(shared);

    Query memory;
    memory.m_Collections = {MEMORY};
    std::vector<Entry> remembered = comas.Execute(memory);
    EMPY_REQUIRE_EQ(remembered.size(), size_t(2));
    for (const Entry &entry : remembered)
    {
        EMPY_CHECK_EQ(entry.m_Collection, MEMORY);
        EMPY_CHECK(entry.m_Data.GetString("outcome") != "stale");
    }
    // every collection: the external static knowledge and the live local experience
    std::vector<Entry> all = comas.Execute(Query());
    std::set<std::string> collections;
    for (const Entry &entry : all)
    {
        collections.insert(entry.m_Collection);
    }
    EMPY_CHECK(collections == std::set<std::string>({RESOURCES, MEMORY, OBSERVATIONS}));
    EMPY_CHECK_EQ(all.size(), size_t(4));
    // projection and limit apply once, to the merged result
    Query projected;
    projected.m_Collections = {RESOURCES, OBSERVATIONS};
    projected.m_Select = {"region"};
    projected.m_Limit = 1;
    size_t queries = external->m_Queries.size();
    std::vector<Entry> one = comas.Execute(projected);
    EMPY_REQUIRE_EQ(one.size(), size_t(1));
    EMPY_CHECK_EQ(external->m_Queries.size(), queries + 1);
    EMPY_CHECK(external->m_Queries.back().m_Select.empty());
    Query nodes = One(RESOURCES, "region", Op::Eq, Json("edge"));
    nodes.m_Select = {"cpu"};
    std::vector<Entry> projectedNodes = comas.Execute(nodes);
    EMPY_REQUIRE_EQ(projectedNodes.size(), size_t(1));
    EMPY_CHECK(!projectedNodes[0].m_Data.Has("region"));
    EMPY_CHECK(external->m_Queries.back().m_Select == std::vector<std::string>({"cpu"})); // no live collection
    // a failing external source still yields the live experience
    external->m_Fail = true;
    EMPY_CHECK_EQ(comas.Execute(memory).size(), size_t(2));
    EMPY_CHECK(comas.Execute(One(RESOURCES, "region", Op::Eq, Json("edge"))).empty());
}
