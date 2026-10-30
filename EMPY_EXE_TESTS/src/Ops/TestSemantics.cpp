#include "Checks.h"
#include "Support/Ops.h"
#include <eOps/Semantics/Ontology.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/YAML.h>
#include <filesystem>
#include <stdexcept>

using namespace eOps;
using namespace eOps::Semantics;

namespace
{
    struct TempDir : eTest::Ops::TempDir
    {
        TempDir() : eTest::Ops::TempDir("EmpyOpsSemantics")
        {
        }
    };

    std::filesystem::path SeedPath()
    {
        return std::filesystem::path(EMPY_TEST_DATA_DIR) / ".." / ".." / "Configs" / "MLOps" / "Ontology.yaml";
    }

    const Ontology &Seed()
    {
        static const Ontology s_Seed = Ontology::Load(SeedPath());
        return s_Seed;
    }

    std::string IdOf(const Concept *entry)
    {
        return entry ? entry->m_Id : std::string("<none>");
    }

    Concept Make(std::string id, std::string kind, std::vector<std::string> synonyms = {}, Json attributes = {},
                 std::vector<Relation> relations = {})
    {
        Concept entry;
        entry.m_Id = std::move(id);
        entry.m_Kind = std::move(kind);
        entry.m_Synonyms = std::move(synonyms);
        entry.m_Attributes = std::move(attributes);
        entry.m_Relations = std::move(relations);
        return entry;
    }
} // namespace

EMPY_TEST(OpsSemantics, Normalize)
{
    EMPY_CHECK_EQ(Normalize("  Sensors "), std::string("sensor"));
    EMPY_CHECK_EQ(Normalize("degrees"), std::string("degree"));
    EMPY_CHECK_EQ(Normalize("boxes"), std::string("box"));
    EMPY_CHECK_EQ(Normalize("switches"), std::string("switch"));
    EMPY_CHECK_EQ(Normalize("batteries"), std::string("battery"));
    EMPY_CHECK_EQ(Normalize("gpus"), std::string("gpu"));
    EMPY_CHECK_EQ(Normalize("celsius"), std::string("celsius"));
    EMPY_CHECK_EQ(Normalize("access"), std::string("access"));
    EMPY_CHECK_EQ(Normalize("analysis"), std::string("analysis"));
    EMPY_CHECK_EQ(Normalize("ms"), std::string("ms"));
    EMPY_CHECK_EQ(Normalize("has"), std::string("has"));
    EMPY_CHECK_EQ(Normalize("Base   Stations"), std::string("base station"));
    EMPY_CHECK_EQ(Normalize(""), std::string());
}

EMPY_TEST(OpsSemantics, SeedCoverage)
{
    const Ontology &seed = Seed();
    EMPY_CHECK(seed.Size() >= 40);
    for (const char *kind :
         {"domain", "quantity", "actuator", "pattern", "objective", "region", "accelerator", "unit", "schedule"})
    {
        EMPY_CHECK(!seed.OfKind(kind).empty());
    }
    for (const char *id :
         {"home",       "factory",   "vehicle",  "building", "temperature", "humidity",    "occupancy", "power",
          "traffic",    "vibration", "speed",    "cooling",  "heating",     "ventilation", "lighting",  "motor",
          "controller", "predictor", "detector", "comfort",  "energy",      "performance", "safety",    "ran",
          "edge",       "cloud",     "gpu",      "npu",      "tpu",         "celsius",     "percent",   "ms",
          "second",     "daily",     "hourly",   "weekly"})
    {
        if (!seed.Find(id))
        {
            eTest::Fail(__FILE__, __LINE__, std::string("seed concept missing: ") + id);
        }
    }

    const Concept *temperature = seed.Find("temperature");
    EMPY_REQUIRE(temperature);
    EMPY_CHECK_EQ(temperature->m_Kind, std::string("quantity"));
    EMPY_CHECK_EQ(temperature->m_Attributes.GetString("unit"), std::string("celsius"));
    EMPY_CHECK_EQ(temperature->m_Attributes.GetString("topic"), std::string("temp"));
    EMPY_CHECK_EQ(seed.Find("humidity")->m_Attributes.GetString("unit"), std::string("percent"));
    EMPY_CHECK_EQ(seed.Find("humidity")->m_Attributes.GetString("topic"), std::string("hum"));

    const Concept *cooling = seed.Find("cooling");
    EMPY_REQUIRE(cooling);
    EMPY_CHECK_EQ(cooling->m_Attributes.GetString("topic"), std::string("set/temp"));
    EMPY_CHECK_EQ(cooling->m_Attributes.GetNumber("min"), 16.0);
    EMPY_CHECK_EQ(cooling->m_Attributes.GetNumber("max"), 24.0);
    // the setpoint actuator of temperature is the first actuator relating to it
    std::vector<const Concept *> actuators = seed.RelatedFrom("temperature", "actuates");
    EMPY_REQUIRE(!actuators.empty());
    EMPY_CHECK_EQ(IdOf(actuators.front()), std::string("cooling"));
    EMPY_CHECK_EQ(IdOf(seed.Related("cooling", "actuates").at(0)), std::string("temperature"));

    const Concept *controller = seed.Find("controller");
    EMPY_REQUIRE(controller);
    EMPY_CHECK(controller->m_Attributes.GetBool("actionable"));
    EMPY_CHECK_EQ(controller->m_Attributes.GetNumber("latencyMs"), 5.0);
    EMPY_CHECK_EQ(controller->m_Attributes.GetString("statistic"), std::string("p95"));
    EMPY_CHECK_EQ(controller->m_Attributes.GetString("suffix"), std::string("control"));
    EMPY_CHECK_EQ(seed.Find("predictor")->m_Attributes.GetString("suffix"), std::string("forecast"));
    EMPY_CHECK_EQ(seed.Find("detector")->m_Attributes.GetString("suffix"), std::string("detector"));
    EMPY_CHECK_EQ(seed.Find("home")->m_Attributes.GetString("root"), std::string("home"));
    EMPY_CHECK_EQ(seed.Find("daily")->m_Attributes.GetString("cron"), std::string("0 0 * * *"));
    EMPY_CHECK_EQ(seed.Find("hourly")->m_Attributes.GetString("cron"), std::string("0 * * * *"));
    EMPY_CHECK_EQ(seed.Find("weekly")->m_Attributes.GetString("cron"), std::string("0 0 * * 0"));

    std::vector<const Concept *> conflicts = seed.Related("energy", "conflictsWith");
    EMPY_CHECK(std::any_of(conflicts.begin(), conflicts.end(),
                           [](const Concept *entry) { return entry->m_Id == "performance"; }));
    std::vector<const Concept *> contained = seed.Related("home", "contains");
    EMPY_CHECK(std::any_of(contained.begin(), contained.end(),
                           [](const Concept *entry) { return entry->m_Id == "temperature"; }));
}

EMPY_TEST(OpsSemantics, LookupSynonyms)
{
    const Ontology &seed = Seed();
    EMPY_CHECK_EQ(IdOf(seed.Lookup("celcius")), std::string("celsius"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("°C")), std::string("celsius"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("Degrees")), std::string("celsius"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("%")), std::string("percent"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("temp")), std::string("temperature"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("thermal")), std::string("temperature"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("hum")), std::string("humidity"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("thermostat")), std::string("cooling"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("fan")), std::string("ventilation"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("control")), std::string("controller"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("forecast")), std::string("predictor"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("anomaly")), std::string("detector"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("radio access")), std::string("ran"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("Far-Edge")), std::string("ran"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("base station")), std::string("ran"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("data center")), std::string("cloud"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("datacenter")), std::string("cloud"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("central")), std::string("cloud"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("nightly")), std::string("daily"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("every day")), std::string("daily"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("every 24 hours")), std::string("daily"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("24h")), std::string("daily"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("millisecond")), std::string("ms"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("GPU")), std::string("gpu"));
    // plural folding
    EMPY_CHECK_EQ(IdOf(seed.Lookup("thermostats")), std::string("cooling"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("base stations")), std::string("ran"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("controllers")), std::string("controller"));
    EMPY_CHECK_EQ(IdOf(seed.Lookup("detectors")), std::string("detector"));
    EMPY_CHECK(seed.Lookup("banana") == nullptr);
    EMPY_CHECK(seed.Lookup("   ") == nullptr);
    EMPY_CHECK(seed.Find("Temperature") == nullptr); // Find is by exact id
}

EMPY_TEST(OpsSemantics, MatchListing21Intent)
{
    const Ontology &seed = Seed();
    std::vector<std::string> tokens =
        Tokenize("Deploy a home temperature controller that keeps the temperature between 16 and 24 degrees using "
                 "temperature and humidity sensors, with a p95 latency under 5 ms, retrained daily on a GPU");
    std::vector<ConceptMatch> matches = seed.Match(tokens);
    std::vector<std::string> ids;
    for (const ConceptMatch &match : matches)
    {
        ids.push_back(match.m_Concept->m_Id);
    }
    std::vector<std::string> expected = {"home",        "temperature", "controller",  "controller",
                                         "temperature", "celsius",     "temperature", "humidity",
                                         "ms",          "daily",       "gpu"};
    EMPY_CHECK_EQ(Join(ids, ","), Join(expected, ","));
    for (size_t i = 1; i < matches.size(); ++i)
    {
        EMPY_CHECK(matches[i - 1].m_Token + matches[i - 1].m_Length <= matches[i].m_Token);
    }
    // "keeps" and "sensors"-style plurals match through normalisation with a lower score
    auto keeps =
        std::find_if(matches.begin(), matches.end(), [](const ConceptMatch &m) { return m.m_Term == "keeps"; });
    EMPY_REQUIRE(keeps != matches.end());
    EMPY_CHECK_EQ(keeps->m_Score, 1.0); // "keeps" is a listed synonym
    EMPY_CHECK_EQ(matches.front().m_Score, 1.0);
}

EMPY_TEST(OpsSemantics, MatchMultiwordLongestFirst)
{
    const Ontology &seed = Seed();
    std::vector<ConceptMatch> matches =
        seed.Match(Tokenize("place the inference in the radio access network near base stations, train in the data "
                            "center on gpus"));
    EMPY_REQUIRE_EQ(matches.size(), size_t(4));
    EMPY_CHECK_EQ(IdOf(matches[0].m_Concept), std::string("ran"));
    EMPY_CHECK_EQ(matches[0].m_Length, size_t(3)); // "radio access network" beats "radio access"
    EMPY_CHECK_EQ(matches[0].m_Term, std::string("radio access network"));
    EMPY_CHECK_EQ(IdOf(matches[1].m_Concept), std::string("ran"));
    EMPY_CHECK_EQ(matches[1].m_Term, std::string("base stations"));
    EMPY_CHECK_EQ(matches[1].m_Score, 1.0); // listed plural
    EMPY_CHECK_EQ(IdOf(matches[2].m_Concept), std::string("cloud"));
    EMPY_CHECK_EQ(matches[2].m_Length, size_t(2));
    EMPY_CHECK_EQ(IdOf(matches[3].m_Concept), std::string("gpu"));

    // "far edge" (ran) wins over the single word "edge"; "energy consumption" (power) over "energy" (objective)
    matches = seed.Match(Tokenize("far-edge energy consumption forecast"));
    EMPY_REQUIRE_EQ(matches.size(), size_t(3));
    EMPY_CHECK_EQ(IdOf(matches[0].m_Concept), std::string("ran"));
    EMPY_CHECK_EQ(IdOf(matches[1].m_Concept), std::string("power"));
    EMPY_CHECK_EQ(IdOf(matches[2].m_Concept), std::string("predictor"));

    // plural normalisation gives a sub-1 score and keeps the written term
    matches = seed.Match(Tokenize("two controllers with humidities"));
    EMPY_REQUIRE_EQ(matches.size(), size_t(2));
    EMPY_CHECK_EQ(IdOf(matches[0].m_Concept), std::string("controller"));
    EMPY_CHECK(matches[0].m_Score < 1.0);
    EMPY_CHECK_EQ(matches[0].m_Term, std::string("controllers"));
    EMPY_CHECK_EQ(matches[0].m_Token, size_t(1));
    EMPY_CHECK_EQ(IdOf(matches[1].m_Concept), std::string("humidity"));
    EMPY_CHECK(seed.Match({}).empty());
}

EMPY_TEST(OpsSemantics, ExtendAddsAndMerges)
{
    Ontology ontology = Seed();
    uint64_t revision = ontology.Revision();
    size_t size = ontology.Size();
    const Concept *temperature = ontology.Find("temperature");

    std::string reason;
    EMPY_CHECK(ontology.Extend(Make("pressure", "quantity", {"barometric pressure", "air pressure"},
                                    Json::MakeObject({{"unit", "hpa"}, {"topic", "pressure"}})),
                               &reason));
    EMPY_CHECK_EQ(reason, std::string());
    EMPY_CHECK_EQ(ontology.Size(), size + 1);
    EMPY_CHECK_EQ(ontology.Revision(), revision + 1);
    EMPY_CHECK_EQ(IdOf(ontology.Lookup("barometric pressure")), std::string("pressure"));
    EMPY_CHECK_EQ(IdOf(ontology.Lookup("pressures")), std::string("pressure"));
    EMPY_CHECK(ontology.Find("temperature") == temperature); // pointers stay valid across Extend

    // an actuator relating to the new quantity
    EMPY_CHECK(ontology.Extend(
        Make("valve", "actuator", {"pressure valve"},
             Json::MakeObject(
                 {{"quantity", "pressure"}, {"topic", "set/pressure"}, {"unit", "percent"}, {"min", 0}, {"max", 100}}),
             {{"actuates", "pressure"}}),
        &reason));
    EMPY_CHECK_EQ(IdOf(ontology.RelatedFrom("pressure", "actuates").at(0)), std::string("valve"));

    // merge into an existing same-kind concept: new synonyms, new attributes, new relations
    EMPY_CHECK(ontology.Extend(Make("temperature", "quantity", {"warmness", "temp"},
                                    Json::MakeObject({{"unit", "celsius"}, {"precision", 0.1}})),
                               &reason));
    EMPY_CHECK(ontology.Find("temperature") == temperature);
    EMPY_CHECK_EQ(IdOf(ontology.Lookup("warmness")), std::string("temperature"));
    EMPY_CHECK_EQ(temperature->m_Attributes.GetNumber("precision"), 0.1);
    EMPY_CHECK_EQ(std::count(temperature->m_Synonyms.begin(), temperature->m_Synonyms.end(), "temp"), 1);
    EMPY_CHECK_EQ(ontology.Size(), size + 2);
    EMPY_CHECK_EQ(ontology.Revision(), revision + 3);

    // a same-kind synonym already owned by another concept is accepted; the first binding stays
    EMPY_CHECK(ontology.Extend(
        Make("boiler-valve", "actuator", {"heater"}, Json::MakeObject({{"topic", "set/boiler"}})), &reason));
    EMPY_CHECK_EQ(IdOf(ontology.Lookup("heater")), std::string("heating"));
    EMPY_CHECK_EQ(IdOf(ontology.Lookup("boiler valve")), std::string("boiler-valve"));

    // the seed copy is untouched
    EMPY_CHECK(Seed().Find("pressure") == nullptr);
    EMPY_CHECK(Seed().Lookup("warmness") == nullptr);
}

EMPY_TEST(OpsSemantics, ExtendRejections)
{
    Ontology ontology = Seed();
    uint64_t revision = ontology.Revision();
    size_t size = ontology.Size();
    auto rejected = [&](Concept entry, std::string_view expected)
    {
        std::string reason;
        bool accepted = ontology.Extend(std::move(entry), &reason);
        if (accepted || reason.find(expected) == std::string::npos)
        {
            eTest::Fail(__FILE__, __LINE__,
                        "expected rejection containing '" + std::string(expected) +
                            "', got: " + (accepted ? std::string("accepted") : reason));
        }
    };
    rejected(Make("", "quantity"), "concept id");
    rejected(Make("two words", "unit"), "whitespace");
    rejected(Make("gizmo", "widget"), "unknown kind");
    rejected(Make("temperature", "unit"), "already exists with kind 'quantity'");
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/knob"}}), {{"actuates", "missing-thing"}}),
             "does not exist");
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/knob"}}), {{"tickles", "temperature"}}),
             "unknown relation");
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/knob"}}), {{"actuates", "celsius"}}),
             "cannot link actuator to unit");
    // a synonym bound to a concept of another kind (temp → quantity temperature)
    rejected(Make("tempo", "unit", {"temp"}), "bound to temperature (quantity)");
    // an id colliding with another kind's synonym (thermostat → actuator cooling)
    rejected(Make("thermostat", "pattern"), "bound to cooling (actuator)");
    // plural-folded collisions count too ("humidities" folds onto humidity)
    rejected(Make("humidities", "objective"), "bound to humidity (quantity)");
    // attribute consistency
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/knob"}, {"min", 5}, {"max", 1}})),
             "min exceeds max");
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/+"}})), "concrete topic");
    rejected(Make("knob", "actuator"), "attribute 'topic'");
    rejected(Make("knob", "actuator", {}, Json::MakeObject({{"topic", "set/knob"}, {"quantity", "celsius"}})),
             "known quantity");
    rejected(Make("sometimes", "schedule", {}, Json::MakeObject({{"cron", "every now and then"}})), "cron");
    rejected(Make("fast", "pattern", {}, Json::MakeObject({{"statistic", "p42"}})), "statistic");
    rejected(Make("cooling", "actuator", {}, Json::MakeObject({{"min", 10}})), "conflicts");
    rejected(Make("cooling", "actuator", {"   "}), "not a usable term");
    rejected(Make("weird", "unit", {}, Json(5)), "attributes must be an object");
    EMPY_CHECK_EQ(ontology.Size(), size);
    EMPY_CHECK_EQ(ontology.Revision(), revision);
    EMPY_CHECK(ontology.Find("knob") == nullptr);
}

EMPY_TEST(OpsSemantics, ConceptJson)
{
    const Concept *cooling = Seed().Find("cooling");
    EMPY_REQUIRE(cooling);
    Json json = cooling->ToJson();
    EMPY_CHECK_EQ(json.GetString("id"), std::string("cooling"));
    EMPY_CHECK_EQ(json["relations"].Size(), cooling->m_Relations.size());
    Concept back = Concept::FromJson(json);
    EMPY_CHECK_EQ(back.m_Id, cooling->m_Id);
    EMPY_CHECK_EQ(back.m_Kind, cooling->m_Kind);
    EMPY_CHECK(back.m_Synonyms == cooling->m_Synonyms);
    EMPY_CHECK(back.m_Attributes == cooling->m_Attributes);
    EMPY_CHECK_EQ(back.m_Relations.size(), cooling->m_Relations.size());
    EMPY_CHECK_THROWS(Concept::FromJson(Json::Parse(R"({"id": 1, "kind": "unit"})")));
    EMPY_CHECK_THROWS(Concept::FromJson(Json::Parse(R"({"id": "a", "kind": "unit", "synonyms": [1]})")));
    EMPY_CHECK_THROWS(Concept::FromJson(Json::Parse(R"({"id": "a", "kind": "unit", "relations": [{"type": 1}]})")));
    EMPY_CHECK_THROWS(Concept::FromJson(Json::Parse("[]")));
}

EMPY_TEST(OpsSemantics, SaveLoadRoundTrip)
{
    TempDir directory;
    Ontology ontology = Seed();
    std::string reason;
    EMPY_REQUIRE(ontology.Extend(Make("pressure", "quantity", {"air \"pressure\"", "°bar"},
                                      Json::MakeObject({{"unit", "hpa"}, {"topic", "pressure"}, {"ratio", 0.25}})),
                                 &reason));
    std::filesystem::path path = directory.m_Path / "nested" / "Ontology.yaml";
    ontology.Save(path);
    EMPY_REQUIRE(std::filesystem::exists(path));

    Ontology loaded = Ontology::Load(path);
    EMPY_CHECK_EQ(loaded.Size(), ontology.Size());
    EMPY_CHECK_EQ(loaded.Revision(), ontology.Revision());
    // YAML mappings come back key-sorted (fkYAML), so compare values and the reloaded text
    EMPY_CHECK_EQ(Ontology::FromYaml(eCore::YamlNode::Parse(loaded.ToYaml())).ToYaml(), loaded.ToYaml());
    for (const char *kind : {"domain", "quantity", "actuator", "pattern", "unit", "schedule"})
    {
        std::vector<const Concept *> before = ontology.OfKind(kind);
        std::vector<const Concept *> after = loaded.OfKind(kind);
        EMPY_REQUIRE_EQ(before.size(), after.size());
        for (size_t i = 0; i < before.size(); ++i)
        {
            EMPY_CHECK(before[i]->ToJson() == after[i]->ToJson());
        }
    }
    EMPY_CHECK_EQ(IdOf(loaded.Lookup("air \"pressure\"")), std::string("pressure"));
    EMPY_CHECK_EQ(loaded.Find("pressure")->m_Attributes.GetNumber("ratio"), 0.25);
    EMPY_CHECK_EQ(IdOf(loaded.Lookup("celcius")), std::string("celsius"));

    // copies and moves are independent
    Ontology copy = loaded;
    EMPY_CHECK(copy.Extend(Make("lumen", "unit"), &reason));
    EMPY_CHECK(loaded.Find("lumen") == nullptr);
    Ontology moved = std::move(copy);
    EMPY_CHECK(moved.Find("lumen") != nullptr);
}

EMPY_TEST(OpsSemantics, FromYamlForwardReferencesAndErrors)
{
    // relations may point at concepts declared later in the document
    Ontology ontology = Ontology::FromYaml(eCore::YamlNode::Parse(R"(
concepts:
  - {id: knob, kind: actuator, attributes: {quantity: level, topic: set/level}, relations: [{type: actuates, target: level}]}
  - {id: level, kind: quantity, synonyms: [fill level], attributes: {topic: level}}
)"));
    EMPY_CHECK_EQ(ontology.Size(), size_t(2));
    EMPY_CHECK_EQ(IdOf(ontology.Related("knob", "actuates").at(0)), std::string("level"));
    EMPY_CHECK_EQ(ontology.Revision(), uint64_t(0));

    EMPY_CHECK_THROWS(Ontology::FromYaml(eCore::YamlNode::Parse("other: 1")));
    EMPY_CHECK_THROWS(Ontology::FromYaml(eCore::YamlNode::Parse("concepts: [{id: a, kind: nonsense}]")));
    EMPY_CHECK_THROWS(Ontology::FromYaml(eCore::YamlNode::Parse(
        "concepts: [{id: a, kind: unit, synonyms: [x]}, {id: b, kind: region, synonyms: [x]}]")));
    EMPY_CHECK_THROWS(Ontology::FromYaml(
        eCore::YamlNode::Parse("concepts: [{id: a, kind: objective, relations: [{type: conflictsWith, target: z}]}]")));
    EMPY_CHECK_THROWS(Ontology::Load(std::filesystem::temp_directory_path() / ("missing-" + NewId({}) + ".yaml")));
}
