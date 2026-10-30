#include "Checks.h"
#include "Support/Ops.h"
#include "Support/Process.h"
#include <eOps/Workload/Model.h>
#include <eOps/Workload/Dataset.h>
#include <eOps/Workload/Stats.h>
#include <eOps/Workload/Boundary.h>
#include <eOps/Workload/Runtime.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Common/Process.h>
#include <eOps/Common/Text.h>
#include <Runtime/Runtime.h>
#include <filesystem>
#include <cmath>
#include <limits>

// model primitives (spec §8) and the MMQP workload runtimes (spec §12) against an in-process broker,
// plus the EmpyWorkload executable (spec §14) as a child process

using namespace eOps;
using namespace eOps::Workload;

namespace
{
    using Clock = std::chrono::steady_clock;

    // the simulator's ground truth y = 0.5·temperature + 0.05·humidity + 8 over deterministic signals
    void Signals(size_t count, std::vector<std::vector<double>> &x, std::vector<double> &y, double noise = 0)
    {
        x.clear();
        y.clear();
        for (size_t i = 0; i < count; ++i)
        {
            double t = static_cast<double>(i);
            double temperature = 20 + 3 * std::sin(t * 0.3);
            double humidity = 50 + 10 * std::sin(t * 0.17 + 1);
            x.push_back({temperature, humidity});
            y.push_back(0.5 * temperature + 0.05 * humidity + 8 + noise * std::sin(t * 1.7));
        }
    }

    bool Near(double a, double b, double tolerance)
    {
        return std::fabs(a - b) <= tolerance;
    }

    Descriptor::RichMLService Service()
    {
        Descriptor::RichMLService service;
        service.m_Metadata.m_Name = "home-control";
        service.m_Actionables.push_back(Descriptor::Actionable{"home/set/temp", "celsius", 16.0, 24.0, 0, "", 1});
        service.m_Actionables.push_back(
            Descriptor::Actionable{"home/cooling/#", "percent", 0.0, std::nullopt, 2, "", 1});
        service.m_Actionables.push_back(
            Descriptor::Actionable{"home/cooling/fan", "percent", std::nullopt, 50.0, 0, "", 1});
        return service;
    }
} // namespace

EMPY_TEST(OpsWorkload, LinearFit)
{
    std::vector<std::vector<double>> x;
    std::vector<double> y;
    Signals(64, x, y);
    ModelArtifact model = LinearModel::Fit(x, y);
    EMPY_REQUIRE_EQ(model.m_Weights.size(), size_t{3});
    EMPY_CHECK(Near(model.m_Weights[0], 8.0, 1e-3));
    EMPY_CHECK(Near(model.m_Weights[1], 0.5, 1e-4));
    EMPY_CHECK(Near(model.m_Weights[2], 0.05, 1e-4));
    EMPY_CHECK_EQ(model.m_Samples, 64u);
    EMPY_CHECK(model.m_TrainedAt > 0);
    EMPY_CHECK(model.m_Features.empty() && model.m_Service.empty()); // left for the caller
    EMPY_CHECK(Near(LinearModel::Predict(model, {22, 55}), 0.5 * 22 + 0.05 * 55 + 8, 1e-4));
    EMPY_CHECK_EQ(LinearModel::Accuracy(model, x, y, 1e-3), 1.0);

    // noisy labels: accuracy is the fraction within tolerance
    Signals(200, x, y, 0.8);
    ModelArtifact noisy = LinearModel::Fit(x, y);
    double tight = LinearModel::Accuracy(noisy, x, y, 0.1);
    double loose = LinearModel::Accuracy(noisy, x, y, 1.0);
    EMPY_CHECK(tight > 0 && tight < 1);
    EMPY_CHECK_EQ(loose, 1.0);
    EMPY_CHECK_EQ(LinearModel::Accuracy(noisy, {}, {}, 1.0), 0.0);

    // exactly features + 1 samples suffice; a constant feature stays solvable through the ridge term
    EMPY_CHECK_EQ(LinearModel::Fit({{1}, {2}}, {3, 5}).m_Weights.size(), size_t{2});
    ModelArtifact constant = LinearModel::Fit({{4}, {4}, {4}}, {1, 2, 3});
    EMPY_CHECK(Near(LinearModel::Predict(constant, {4}), 2.0, 1e-6));

    EMPY_CHECK_THROWS(LinearModel::Fit({{1, 2}, {3, 4}}, {1, 2}));               // fewer than features + 1
    EMPY_CHECK_THROWS(LinearModel::Fit({{1}, {2, 3}, {4}}, {1, 2, 3}));          // ragged rows
    EMPY_CHECK_THROWS(LinearModel::Fit({{1}, {2}, {3}}, {1, 2}));                // label count
    EMPY_CHECK_THROWS(LinearModel::Fit({}, {}));                                 // no samples
    EMPY_CHECK_THROWS(LinearModel::Fit({{1}, {2}}, {1, std::nan("")}));          // non-finite
    EMPY_CHECK_THROWS(LinearModel::Fit({{1}, {2}}, {1, 2}, -1));                 // ridge
    EMPY_CHECK_THROWS(LinearModel::Fit({{1, 1}, {2, 2}, {3, 3}}, {1, 2, 3}, 0)); // collinear without ridge
    EMPY_CHECK_THROWS(LinearModel::Predict(model, {1}));
    EMPY_CHECK_THROWS(LinearModel::Accuracy(model, x, {1.0}, 1.0));
}

EMPY_TEST(OpsWorkload, ArtifactJson)
{
    ModelArtifact artifact;
    artifact.m_Service = "home-control";
    artifact.m_Version = 3;
    artifact.m_Features = {"home/temp", "home/hum"};
    artifact.m_Weights = {8, 0.5, 0.05};
    artifact.m_Accuracy = 0.97;
    artifact.m_Samples = 384;
    artifact.m_TrainedAt = 1792000000123;
    artifact.m_Dataset = "home-telemetry";
    Json json = artifact.ToJson();
    EMPY_CHECK_EQ(json.GetString("id"), std::string("home-control-v3"));
    EMPY_CHECK_EQ(json.GetInt("trainedAt"), int64_t{1792000000123});
    EMPY_CHECK_EQ(json["weights"].Size(), size_t{3});
    ModelArtifact back = ModelArtifact::FromJson(Json::Parse(json.Dump()));
    EMPY_CHECK_EQ(back.m_Id, std::string("home-control-v3"));
    EMPY_CHECK(back.ToJson() == json);
    EMPY_CHECK(back.m_Features == artifact.m_Features);
    EMPY_CHECK(back.m_Weights == artifact.m_Weights);
    EMPY_CHECK_EQ(back.m_Version, 3u);

    artifact.m_Id = "custom";
    EMPY_CHECK_EQ(ModelArtifact::FromJson(artifact.ToJson()).m_Id, std::string("custom"));
    ModelArtifact minimal = ModelArtifact::FromJson(Json::Parse(R"({"service":"s","version":1})"));
    EMPY_CHECK_EQ(minimal.m_Id, std::string("s-v1"));
    EMPY_CHECK(minimal.m_Weights.empty());

    EMPY_CHECK_THROWS(ModelArtifact::FromJson(Json::Parse(R"([1])")));
    EMPY_CHECK_THROWS(ModelArtifact::FromJson(Json::Parse(R"({"version":-1})")));
    EMPY_CHECK_THROWS(ModelArtifact::FromJson(Json::Parse(R"({"version":4294967296})")));
    EMPY_CHECK_THROWS(ModelArtifact::FromJson(Json::Parse(R"({"weights":["a"]})")));
    EMPY_CHECK_THROWS(ModelArtifact::FromJson(Json::Parse(R"({"features":[1]})")));
}

EMPY_TEST(OpsWorkload, DatasetSplit)
{
    EMPY_CHECK_THROWS(Dataset(0));
    Dataset dataset(8);
    EMPY_CHECK_EQ(dataset.Capacity(), size_t{8});
    for (uint64_t i = 0; i < 10; ++i)
    {
        dataset.Add(Sample{{static_cast<double>(i)}, static_cast<double>(i) * 2, i});
    }
    EMPY_CHECK_EQ(dataset.Size(), size_t{8}); // the two oldest were evicted
    std::vector<Sample> samples = dataset.Samples();
    EMPY_CHECK_EQ(samples.front().m_Time, uint64_t{2});
    EMPY_CHECK_EQ(samples.back().m_Time, uint64_t{9});

    std::vector<Sample> train{Sample{}};
    std::vector<Sample> test{Sample{}, Sample{}};
    dataset.Split(0.25, train, test); // every 4th sample: indices 3 and 7
    EMPY_REQUIRE_EQ(test.size(), size_t{2});
    EMPY_CHECK_EQ(train.size(), size_t{6});
    EMPY_CHECK_EQ(test[0].m_Time, uint64_t{5});
    EMPY_CHECK_EQ(test[1].m_Time, uint64_t{9});
    EMPY_CHECK_EQ(train[0].m_Time, uint64_t{2});

    dataset.Split(0.5, train, test);
    EMPY_CHECK_EQ(test.size(), size_t{4});
    EMPY_CHECK_EQ(train.size(), size_t{4});
    dataset.Split(0.3, train, test); // k = round(3.33) = 3: indices 2 and 5
    EMPY_CHECK_EQ(test.size(), size_t{2});
    std::vector<Sample> again;
    std::vector<Sample> againTest;
    dataset.Split(0.3, again, againTest);
    EMPY_CHECK_EQ(againTest[0].m_Time, test[0].m_Time); // deterministic

    EMPY_CHECK_THROWS(dataset.Split(0, train, test));
    EMPY_CHECK_THROWS(dataset.Split(1, train, test));
    EMPY_CHECK_THROWS(dataset.Split(std::nan(""), train, test));
    dataset.Clear();
    EMPY_CHECK_EQ(dataset.Size(), size_t{0});
    dataset.Split(0.25, train, test);
    EMPY_CHECK(train.empty() && test.empty());
}

EMPY_TEST(OpsWorkload, WindowStatistics)
{
    EMPY_CHECK_THROWS(WindowStats(0));
    EMPY_CHECK_THROWS(WindowStats(1, 0));
    Clock::time_point t0 = Clock::now();
    WindowStats stats(60);
    EMPY_CHECK_EQ(stats.WindowSeconds(), 60u);
    EMPY_CHECK(!stats.Statistic("p95", t0).has_value());
    for (int32_t i = 100; i >= 1; --i)
    {
        stats.Add(i, t0);
    }
    stats.Add(std::numeric_limits<double>::infinity(), t0); // ignored
    EMPY_CHECK_EQ(stats.Count(t0), size_t{100});
    EMPY_CHECK_EQ(stats.Statistic("p50", t0).value_or(-1), 50.0);
    EMPY_CHECK_EQ(stats.Statistic("p90", t0).value_or(-1), 90.0);
    EMPY_CHECK_EQ(stats.Statistic("p95", t0).value_or(-1), 95.0);
    EMPY_CHECK_EQ(stats.Statistic("p99", t0).value_or(-1), 99.0);
    EMPY_CHECK_EQ(stats.Statistic("mean", t0).value_or(-1), 50.5);
    EMPY_CHECK_EQ(stats.Statistic("max", t0).value_or(-1), 100.0);
    EMPY_CHECK(!stats.Statistic("p42", t0).has_value());
    EMPY_CHECK(WindowStats::KnownStatistic("mean"));
    EMPY_CHECK(!WindowStats::KnownStatistic("median"));

    // samples older than the window leave it
    stats.Add(7, t0 + std::chrono::seconds(30));
    EMPY_CHECK_EQ(stats.Count(t0 + std::chrono::seconds(60)), size_t{101});
    EMPY_CHECK_EQ(stats.Count(t0 + std::chrono::seconds(61)), size_t{1});
    EMPY_CHECK_EQ(stats.Statistic("p95", t0 + std::chrono::seconds(61)).value_or(-1), 7.0);
    EMPY_CHECK(!stats.Statistic("mean", t0 + std::chrono::seconds(200)).has_value());

    // nearest rank on small samples
    WindowStats small(10);
    small.Add(3, t0);
    small.Add(1, t0);
    small.Add(2, t0);
    EMPY_CHECK_EQ(small.Statistic("p50", t0).value_or(-1), 2.0);
    EMPY_CHECK_EQ(small.Statistic("p95", t0).value_or(-1), 3.0);

    // bounded memory: beyond maxSamples the oldest is dropped
    WindowStats bounded(60, 4);
    for (int32_t i = 1; i <= 6; ++i)
    {
        bounded.Add(i, t0);
    }
    EMPY_CHECK_EQ(bounded.Count(t0), size_t{4});
    EMPY_CHECK_EQ(bounded.Statistic("mean", t0).value_or(-1), 4.5);
}

EMPY_TEST(OpsWorkload, ActionBoundary)
{
    ActionBoundary boundary(Service());
    Clock::time_point t0 = Clock::now();

    ActionDecision inside = boundary.Check("home/set/temp", 20.5, t0);
    EMPY_CHECK(inside.m_Allowed && !inside.m_Clamped);
    EMPY_CHECK_EQ(inside.m_Value, 20.5);
    EMPY_CHECK(inside.m_Reason.empty());

    ActionDecision high = boundary.Check("home/set/temp", 30, t0);
    EMPY_CHECK(high.m_Allowed && high.m_Clamped);
    EMPY_CHECK_EQ(high.m_Value, 24.0);
    EMPY_CHECK(high.m_Reason.find("clamped") != std::string::npos);
    ActionDecision low = boundary.Check("home/set/temp", -4, t0);
    EMPY_CHECK(low.m_Allowed && low.m_Clamped);
    EMPY_CHECK_EQ(low.m_Value, 16.0);

    ActionDecision undeclared = boundary.Check("home/set/humidity", 40, t0);
    EMPY_CHECK(!undeclared.m_Allowed);
    EMPY_CHECK(undeclared.m_Reason.find("not a declared actionable") != std::string::npos);
    EMPY_CHECK_EQ(undeclared.m_Rule, std::string("domain"));
    EMPY_CHECK_EQ(boundary.Check("home/set/+", 20, t0).m_Rule, std::string("domain"));
    EMPY_CHECK_EQ(boundary.Check("home/set/temp", std::nan(""), t0).m_Rule, std::string("value"));
    EMPY_CHECK(high.m_Rule.empty() && inside.m_Rule.empty());
    EMPY_CHECK(!boundary.Check("home/set/+", 20, t0).m_Allowed);
    EMPY_CHECK(!boundary.Check("home//#", 20, t0).m_Allowed);
    EMPY_CHECK(!boundary.Check("", 20, t0).m_Allowed);
    EMPY_CHECK(!boundary.Check("home/set/temp", std::nan(""), t0).m_Allowed);

    // the exact declaration wins over the covering domain (max 50, no rate limit)
    for (int32_t i = 0; i < 5; ++i)
    {
        ActionDecision fan = boundary.Check("home/cooling/fan", 80, t0);
        EMPY_CHECK(fan.m_Allowed && fan.m_Clamped);
        EMPY_CHECK_EQ(fan.m_Value, 50.0);
    }

    // the wildcard domain applies its rate limit per concrete topic over a trailing minute
    EMPY_CHECK(boundary.Check("home/cooling/pump", 10, t0).m_Allowed);
    EMPY_CHECK(boundary.Check("home/cooling/pump", -5, t0 + std::chrono::seconds(10)).m_Clamped);
    ActionDecision limited = boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(20));
    EMPY_CHECK(!limited.m_Allowed && !limited.m_Clamped);
    EMPY_CHECK(limited.m_Reason.find("rate limit") != std::string::npos);
    EMPY_CHECK_EQ(limited.m_Rule, std::string("rate"));
    EMPY_CHECK(boundary.Check("home/cooling/vent", 10, t0 + std::chrono::seconds(20)).m_Allowed);
    EMPY_CHECK(boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(60)).m_Allowed);
    EMPY_CHECK(!boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(65)).m_Allowed);
    EMPY_CHECK(boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(71)).m_Allowed);
    // rejected actions do not consume the budget
    EMPY_CHECK(!boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(72)).m_Allowed);
    EMPY_CHECK(!boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(119)).m_Allowed);
    EMPY_CHECK(boundary.Check("home/cooling/pump", 10, t0 + std::chrono::seconds(120)).m_Allowed);

    ActionBoundary none(Descriptor::RichMLService{});
    EMPY_CHECK(!none.Check("home/set/temp", 20, t0).m_Allowed);
}

// -- MMQP runtimes (spec §12)

namespace
{
    using eTest::Ops::ConfigFile;
    using eTest::Ops::LocalBroker;
    using Fabric::Bus;
    using Fabric::BusModule;
    using Fabric::BusOptions;
    using Fabric::Envelope;
    using Fabric::Topics;

    // runtimes attach before Start (the first Poll runs OnAttach of every module in attachment order)
    struct Peer : eTest::Ops::Peer
    {
        Peer(uint16_t port, std::string source) : eTest::Ops::Peer(port, std::move(source), {}, false)
        {
        }
    };

    using Peers = std::vector<Peer *>;

    template <typename _Predicate> bool Pump(const Peers &peers, _Predicate &&done, uint32_t timeoutMs = 5000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        while (true)
        {
            for (auto *peer : peers)
            {
                peer->m_Client->Poll(1);
            }
            if (done())
            {
                return true;
            }
            if (Clock::now() >= deadline)
            {
                return false;
            }
        }
    }

    void Settle(const Peers &peers, uint32_t ms)
    {
        Pump(peers, [] { return false; }, ms);
    }

    struct Answer
    {
        bool m_Done = false;
        eMQ::Reason m_Reason;
        std::optional<Envelope> m_Envelope;

        bool Ok() const
        {
            return m_Done && m_Reason.m_Code < 0x80 && m_Envelope.has_value();
        }
    };

    // Bus::Call from `caller` while pumping every peer
    Answer Ask(Peer &caller, const Peers &peers, const std::string &topic, Json body, std::string kind = "model")
    {
        auto answer = std::make_shared<Answer>();
        caller.Get().Call(
            topic, Envelope::Make(std::move(kind), std::move(body)),
            [answer](eMQ::Reason reason, std::optional<Envelope> envelope)
            {
                answer->m_Done = true;
                answer->m_Reason = std::move(reason);
                answer->m_Envelope = std::move(envelope);
            },
            4000);
        Pump(peers, [&] { return answer->m_Done; }, 6000);
        return *answer;
    }

    std::filesystem::path HomeControlFile()
    {
        return ConfigFile("HomeControl.yaml");
    }

    // Listing 2.1 with test-sized windows: quick training and a short evaluation window
    Descriptor::RichMLService HomeControl()
    {
        auto service = Descriptor::LoadDescriptor(HomeControlFile());
        service.m_Training->m_MinSamples = 24;
        service.m_Training->m_Window = 40;
        service.m_Lifecycle.m_MinSamples = 10;
        service.m_Lifecycle.m_EvaluationWindow = 20;
        return service;
    }

    // the simulator's ground truth before the concept drift (bias, temperature, humidity)
    ModelArtifact Exact(uint32_t version, std::vector<double> weights = {8, 0.5, 0.05})
    {
        ModelArtifact artifact;
        artifact.m_Service = "home-control";
        artifact.m_Version = version;
        artifact.m_Id = "home-control-v" + std::to_string(version);
        artifact.m_Features = {"home/temp", "home/hum"};
        artifact.m_Weights = std::move(weights);
        artifact.m_Accuracy = 1;
        return artifact;
    }

    WorkloadOptions Options(Descriptor::RichMLService service, std::string instance, uint32_t periodMs = 100)
    {
        WorkloadOptions options;
        options.m_Service = std::move(service);
        options.m_Instance = std::move(instance);
        options.m_Node = "edge-1";
        options.m_Region = "edge";
        options.m_HeartbeatMs = periodMs;
        options.m_MetricsMs = periodMs;
        options.m_DriftDebounceMs = 600;
        return options;
    }

    SimulatorOptions SimulatorFor(Descriptor::RichMLService service, uint32_t periodMs = 10, uint32_t driftAfterMs = 0)
    {
        SimulatorOptions options;
        options.m_Service = std::move(service);
        options.m_PeriodMs = periodMs;
        options.m_DriftAfterMs = driftAfterMs;
        options.m_Region = "edge";
        options.m_LocalityRate = 12.5;
        options.m_LocalityMs = 200;
        return options;
    }

    // a data-plane publication observed by a test peer
    struct Sighting
    {
        std::string m_Topic;
        Json m_Body;
    };

    void ListenRaw(Peer &peer, const std::string &filter, std::vector<Sighting> &seen)
    {
        EMPY_REQUIRE(peer.Get()
                         .ListenRaw(
                             filter, [&seen](const eMQ::PublishRequest &message)
                             { seen.push_back(Sighting{message.m_Topic, FromBytes(message.m_Payload)}); }, 1)
                         .m_Code < 0x80);
    }

    struct Received
    {
        Clock::time_point m_At;
        std::string m_Topic;
        Envelope m_Envelope;
    };

    void ListenEnvelopes(Peer &peer, const std::string &filter, std::vector<Received> &seen)
    {
        EMPY_REQUIRE(peer.Get()
                         .Listen(filter, [&seen](const std::string &topic, const Envelope &envelope)
                                 { seen.push_back(Received{Clock::now(), topic, envelope}); })
                         .m_Code < 0x80);
    }

    // one observation pair (and optionally its label) on behalf of a sensor gateway
    void Observe(Peer &gateway, double temperature, double humidity, std::optional<double> label = std::nullopt)
    {
        auto &bus = gateway.Get();
        (void)bus.PublishRaw("home/temp", eCore::SharedBytes::Copy(Json(temperature).Dump()), 1);
        (void)bus.PublishRaw("home/hum", eCore::SharedBytes::Copy(Json(humidity).Dump()), 1);
        if (label)
        {
            (void)bus.PublishRaw("home/label/temp", eCore::SharedBytes::Copy(Json(*label).Dump()), 1);
        }
    }

    Json Load(const ModelArtifact &artifact)
    {
        return Json::Object{{"action", "load"}, {"artifact", artifact.ToJson()}};
    }
} // namespace

EMPY_TEST(OpsWorkload, RuntimeOptions)
{
    Descriptor::RichMLService service = HomeControl();

    // the default instance names the service, role and node
    InferenceRuntime inference(Options(service, ""));
    Json status = inference.Status();
    EMPY_CHECK_EQ(status.GetString("instance"), std::string("home-control-inference-edge-1-0"));
    EMPY_CHECK_EQ(status.GetString("role"), std::string("inference"));
    EMPY_CHECK_EQ(status.GetString("state"), std::string("waiting-model"));
    EMPY_CHECK_EQ(status.GetInt("modelVersion"), int64_t{0});
    EMPY_CHECK(status["accuracy"].IsNull());
    EMPY_CHECK_EQ(status.GetString("actuation"), std::string("home/set/temp"));
    EMPY_CHECK_EQ(status["latency"].GetString("statistic"), std::string("p95"));
    EMPY_CHECK_EQ(status["latency"].GetInt("windowSeconds"), int64_t{60});
    EMPY_CHECK_EQ(status["latency"].GetNumber("maxMs"), 5.0);
    EMPY_CHECK(!inference.Active().has_value());

    WorkloadOptions seeded = Options(service, "inf-0");
    seeded.m_InitialModel = Exact(3);
    InferenceRuntime ready(seeded);
    EMPY_CHECK_EQ(ready.Status().GetString("state"), std::string("ready"));
    EMPY_REQUIRE(ready.Active().has_value());
    EMPY_CHECK_EQ(uint32_t(ready.Active()->m_Version), 3u);

    TrainingRuntime training(Options(service, ""));
    EMPY_CHECK_EQ(training.Status().GetString("instance"), std::string("home-control-training-edge-1-0"));
    EMPY_CHECK_EQ(training.Status().GetString("state"), std::string("collecting"));
    EMPY_CHECK_EQ(training.Status().GetInt("minSamples"), int64_t{24});
    EMPY_CHECK_EQ(training.Samples(), size_t{0});

    // initial models must fit the declared observables
    for (const ModelArtifact &bad : {Exact(0), Exact(1, {8, 0.5}), Exact(1, {8, 0.5, std::nan("")})})
    {
        WorkloadOptions options = Options(service, "inf-0");
        options.m_InitialModel = bad;
        EMPY_CHECK_THROWS(InferenceRuntime{options});
    }
    WorkloadOptions reversed = Options(service, "inf-0");
    reversed.m_InitialModel = Exact(1);
    std::swap(reversed.m_InitialModel->m_Features[0], reversed.m_InitialModel->m_Features[1]);
    EMPY_CHECK_THROWS(InferenceRuntime{reversed});
    WorkloadOptions foreign = Options(service, "inf-0");
    foreign.m_InitialModel = Exact(1);
    foreign.m_InitialModel->m_Service = "energy-saver";
    EMPY_CHECK_THROWS(InferenceRuntime{foreign});

    // descriptor sections each role needs
    Descriptor::RichMLService noInference = service;
    noInference.m_Inference.reset();
    EMPY_CHECK_THROWS(InferenceRuntime{Options(noInference, "inf-0")});
    Descriptor::RichMLService noTraining = service;
    noTraining.m_Training.reset();
    EMPY_CHECK_THROWS(TrainingRuntime{Options(noTraining, "trn-0")});
    Descriptor::RichMLService noLabel = service;
    noLabel.m_Training->m_Label.clear();
    EMPY_CHECK_THROWS(TrainingRuntime{Options(noLabel, "trn-0")});
    Descriptor::RichMLService noObservables = service;
    noObservables.m_Observables.clear();
    EMPY_CHECK_THROWS(InferenceRuntime{Options(noObservables, "inf-0")});
    EMPY_CHECK_THROWS(TrainingRuntime{Options(noObservables, "trn-0")});
    EMPY_CHECK_THROWS(InferenceRuntime{Options(service, "bad/instance")});
    Descriptor::RichMLService unnamed = service;
    unnamed.m_Metadata.m_Name.clear();
    EMPY_CHECK_THROWS(InferenceRuntime{Options(unnamed, "inf-0")});

    // the simulator needs a concrete observable, a positive period and a valid region level
    EMPY_CHECK_THROWS(Simulator{SimulatorFor(service, 0)});
    Descriptor::RichMLService wildcard = service;
    wildcard.m_Observables = {Descriptor::Observable{"home/+/temp", "celsius", "", 1}};
    EMPY_CHECK_THROWS(Simulator{SimulatorFor(wildcard)});
    SimulatorOptions region = SimulatorFor(service);
    region.m_Region = "a/b";
    EMPY_CHECK_THROWS(Simulator{region});
    Simulator simulator(SimulatorFor(service));
    EMPY_CHECK_EQ(simulator.Published(), size_t{0});
    EMPY_CHECK(!simulator.Drifted());
}

// training collects labelled samples and answers train; inference validates and loads the artefact,
// then publishes bounded actions and reports metrics, latency and a retained heartbeat
EMPY_TEST(OpsWorkload, RuntimeTrainAndServe)
{
    LocalBroker fabric;
    Topics topics;
    Descriptor::RichMLService service = HomeControl();
    Peer trainer(fabric.m_Port, "trainer"), predictor(fabric.m_Port, "predictor"), ops(fabric.m_Port, "harness");
    auto *training = trainer.Attach<TrainingRuntime>(Options(service, "trn-0"));
    auto *inference = predictor.Attach<InferenceRuntime>(Options(service, "inf-0"));
    trainer.Start();
    predictor.Start();
    ops.Start();
    Peers all = {&trainer, &predictor, &ops};
    const std::string train = topics.Workload("home-control", "train");
    const std::string model = topics.Workload("home-control", "model");

    // no samples yet: training refuses with an error reason
    Answer early = Ask(ops, all, train, Json::Object{{"version", 1}}, "train");
    EMPY_REQUIRE(early.m_Done);
    EMPY_CHECK_EQ(early.m_Reason.m_Code, eMQ::MMQP_REASON_UNSPECIFIED);
    EMPY_CHECK(early.m_Reason.m_Text.find("Insufficient samples") != std::string::npos);
    EMPY_CHECK_EQ(Ask(ops, all, train, Json::Object{{"version", -2}}, "train").m_Reason.m_Code,
                  eMQ::MMQP_REASON_MALFORMED_PACKET);
    EMPY_CHECK_EQ(Ask(ops, all, train, Json("now"), "train").m_Reason.m_Code, eMQ::MMQP_REASON_MALFORMED_PACKET);

    std::vector<Sighting> actions;
    std::vector<Received> candidates, metrics, drifts;
    ListenRaw(ops, "home/set/#", actions);
    ListenEnvelopes(ops, topics.CandidateFilter(), candidates);
    ListenEnvelopes(ops, topics.EventFilter("metrics"), metrics);
    ListenEnvelopes(ops, topics.EventFilter("drift"), drifts);

    Peer gateway(fabric.m_Port, "gateway");
    auto *simulator = gateway.Attach<Simulator>(SimulatorFor(service));
    gateway.Start();
    all.push_back(&gateway);
    EMPY_REQUIRE(Pump(all, [&] { return training->Samples() >= 32; }, 10000));
    EMPY_CHECK(simulator->Published() > 0);
    EMPY_CHECK_EQ(training->Status().GetString("state"), std::string("ready"));

    Answer trained = Ask(ops, all, train, Json::Object{{"version", 1}, {"workflowId", "wf-1"}}, "train");
    EMPY_REQUIRE(trained.Ok());
    const Json &result = trained.m_Envelope->m_Body;
    EMPY_CHECK_EQ(trained.m_Envelope->m_Kind, std::string("artifact"));
    EMPY_CHECK_EQ(result.GetString("workflowId"), std::string("wf-1"));
    EMPY_CHECK_EQ(result.GetString("instance"), std::string("trn-0"));
    EMPY_CHECK(result.GetInt("samples") >= 24 && result.GetInt("testSamples") >= 8);
    ModelArtifact artifact = ModelArtifact::FromJson(result["artifact"]);
    EMPY_CHECK_EQ(artifact.m_Version, 1u);
    EMPY_CHECK_EQ(artifact.m_Id, std::string("home-control-v1"));
    EMPY_CHECK_EQ(artifact.m_Service, std::string("home-control"));
    EMPY_CHECK_EQ(artifact.m_Dataset, std::string("home-telemetry"));
    EMPY_CHECK(artifact.m_Features == std::vector<std::string>({"home/temp", "home/hum"}));
    EMPY_REQUIRE_EQ(artifact.m_Weights.size(), size_t{3});
    EMPY_CHECK(Near(artifact.m_Weights[0], 8, 1e-3));
    EMPY_CHECK(Near(artifact.m_Weights[1], 0.5, 1e-4));
    EMPY_CHECK(Near(artifact.m_Weights[2], 0.05, 1e-4));
    EMPY_CHECK_EQ(artifact.m_Accuracy, 1.0);
    EMPY_CHECK_EQ(int64_t(artifact.m_Samples), result.GetInt("samples"));
    EMPY_CHECK(artifact.m_TrainedAt > 0);
    // the candidate artefact is published on the train/model observable
    EMPY_REQUIRE(Pump(all, [&] { return !candidates.empty(); }));
    EMPY_CHECK_EQ(candidates[0].m_Topic, topics.Candidate("home-control"));
    EMPY_CHECK_EQ(candidates[0].m_Envelope.m_Kind, std::string("artifact"));
    EMPY_CHECK_EQ(candidates[0].m_Envelope.m_Body["artifact"].GetString("id"), std::string("home-control-v1"));
    // without an explicit version the trainer numbers its artefacts itself
    Answer local = Ask(ops, all, train, Json(), "train");
    EMPY_REQUIRE(local.Ok());
    EMPY_CHECK_EQ(local.m_Envelope->m_Body["artifact"].GetInt("version"), int64_t{2});
    EMPY_CHECK_EQ(training->Status().GetInt("trained"), int64_t{2});
    EMPY_CHECK_EQ(training->Status().GetInt("lastVersion"), int64_t{2});
    EMPY_CHECK_EQ(training->Status().GetNumber("lastAccuracy"), 1.0);

    // mismatched artefacts and malformed requests are refused; nothing is acted upon
    ModelArtifact reversed = artifact;
    std::swap(reversed.m_Features[0], reversed.m_Features[1]);
    ModelArtifact foreign = artifact;
    foreign.m_Service = "energy-saver";
    ModelArtifact unversioned = artifact;
    unversioned.m_Version = 0;
    ModelArtifact narrow = artifact;
    narrow.m_Weights.pop_back();
    for (const ModelArtifact &bad : {reversed, foreign, unversioned, narrow})
    {
        Answer refused = Ask(ops, all, model, Load(bad));
        EMPY_CHECK(refused.m_Done);
        EMPY_CHECK_EQ(refused.m_Reason.m_Code, eMQ::MMQP_REASON_POLICY_VIOLATION);
        EMPY_CHECK(refused.m_Reason.m_Text.find("Artifact rejected") != std::string::npos);
    }
    EMPY_CHECK_EQ(Ask(ops, all, model, Json::Object{{"action", "load"}}).m_Reason.m_Code,
                  eMQ::MMQP_REASON_MALFORMED_PACKET);
    EMPY_CHECK_EQ(Ask(ops, all, model, Json::Object{{"action", "load"}, {"artifact", Json::Object{{"version", "x"}}}})
                      .m_Reason.m_Code,
                  eMQ::MMQP_REASON_MALFORMED_PACKET);
    EMPY_CHECK_EQ(Ask(ops, all, model, Json::Object{{"action", "explode"}}).m_Reason.m_Code,
                  eMQ::MMQP_REASON_MALFORMED_PACKET);
    EMPY_CHECK_EQ(Ask(ops, all, model, Json("load")).m_Reason.m_Code, eMQ::MMQP_REASON_MALFORMED_PACKET);
    EMPY_CHECK_EQ(Ask(ops, all, model, Json::Object{{"action", "rollback"}}).m_Reason.m_Code,
                  eMQ::MMQP_REASON_UNSPECIFIED); // nothing to roll back to
    Answer waiting = Ask(ops, all, model, Json::Object{{"action", "status"}});
    EMPY_REQUIRE(waiting.Ok());
    EMPY_CHECK_EQ(waiting.m_Envelope->m_Body.GetString("state"), std::string("waiting-model"));
    EMPY_CHECK(!inference->Active().has_value());
    EMPY_CHECK(actions.empty());

    // a valid artefact activates and drives bounded actions on the declared actuation topic only
    Answer loaded = Ask(ops, all, model, Load(artifact));
    EMPY_REQUIRE(loaded.Ok());
    EMPY_CHECK_EQ(loaded.m_Envelope->m_Kind, std::string("model"));
    EMPY_CHECK_EQ(loaded.m_Envelope->m_Body.GetInt("loaded"), int64_t{1});
    EMPY_CHECK(loaded.m_Envelope->m_Body["previous"].IsNull());
    EMPY_REQUIRE(Pump(
        all,
        [&]
        {
            return actions.size() >= 30 && inference->Status().GetInt("samples") >= 10 && !metrics.empty() &&
                   metrics.back().m_Envelope.m_Body.GetInt("samples") >= 10;
        },
        10000));
    for (const Sighting &action : actions)
    {
        EMPY_CHECK_EQ(action.m_Topic, std::string("home/set/temp"));
        double value = action.m_Body.GetNumber("value", -1);
        EMPY_CHECK(value >= 16 && value <= 24);
        EMPY_CHECK_EQ(action.m_Body.GetString("unit"), std::string("celsius"));
        EMPY_CHECK_EQ(action.m_Body.GetString("service"), std::string("home-control"));
        EMPY_CHECK_EQ(action.m_Body.GetString("instance"), std::string("inf-0"));
        EMPY_CHECK_EQ(action.m_Body.GetInt("model"), int64_t{1});
    }
    Json status = inference->Status();
    EMPY_CHECK_EQ(status.GetString("state"), std::string("ready"));
    EMPY_CHECK_EQ(status.GetInt("modelVersion"), int64_t{1});
    EMPY_CHECK_EQ(status.GetNumber("accuracy"), 1.0);
    EMPY_CHECK_EQ(status.GetInt("rejected"), int64_t{0});
    EMPY_CHECK(status.GetInt("actions") >= 30);
    EMPY_CHECK(status["latency"]["value"].IsNumber() && status["latency"].GetNumber("value") >= 0);
    EMPY_CHECK(status["latency"].GetInt("samples") >= 30);

    const Json &report = metrics.back().m_Envelope.m_Body;
    EMPY_CHECK_EQ(metrics.back().m_Topic, topics.Event("home-control", "metrics"));
    EMPY_CHECK_EQ(report.GetString("instance"), std::string("inf-0"));
    EMPY_CHECK_EQ(report.GetString("node"), std::string("edge-1"));
    EMPY_CHECK_EQ(report.GetString("region"), std::string("edge"));
    EMPY_CHECK_EQ(report.GetInt("modelVersion"), int64_t{1});
    EMPY_CHECK_EQ(report.GetNumber("accuracy"), 1.0);
    EMPY_CHECK_EQ(report["latency"].GetString("statistic"), std::string("p95"));
    EMPY_CHECK_EQ(report["latency"].GetInt("windowSeconds"), int64_t{60});
    EMPY_CHECK(report["latency"]["value"].IsNumber());
    EMPY_CHECK(report.Has("actions") && report.Has("rejected"));
    EMPY_CHECK(drifts.empty());

    // a late subscriber receives the retained heartbeats
    Peer late(fabric.m_Port, "late");
    late.Start();
    all.push_back(&late);
    std::vector<Received> beats, trainerBeats;
    ListenEnvelopes(late, topics.Heartbeat("home-control", "inf-0"), beats);
    ListenEnvelopes(late, topics.Heartbeat("home-control", "trn-0"), trainerBeats);
    EMPY_REQUIRE(Pump(all, [&] { return !beats.empty() && !trainerBeats.empty(); }));
    EMPY_CHECK_EQ(beats[0].m_Envelope.m_Kind, std::string("status"));
    EMPY_CHECK_EQ(beats[0].m_Envelope.m_Source, std::string("predictor"));
    EMPY_CHECK_EQ(beats[0].m_Envelope.m_Body.GetString("role"), std::string("inference"));
    EMPY_CHECK_EQ(beats[0].m_Envelope.m_Body.GetString("state"), std::string("ready"));
    EMPY_CHECK_EQ(beats[0].m_Envelope.m_Body.GetInt("modelVersion"), int64_t{1});
    EMPY_CHECK_EQ(trainerBeats[0].m_Envelope.m_Body.GetString("role"), std::string("training"));
    EMPY_CHECK(trainerBeats[0].m_Envelope.m_Body.GetInt("samples") > 0);

    // detaching publishes a final retained "stopped" heartbeat and stops acting
    predictor.m_Client->DetachModule<InferenceRuntime>();
    EMPY_REQUIRE(
        Pump(all, [&] { return beats.back().m_Envelope.m_Body.GetString("state") == std::string("stopped"); }));
    Settle(all, 100);
    size_t final = actions.size();
    Settle(all, 300);
    EMPY_CHECK_EQ(actions.size(), final);
    EMPY_CHECK_EQ(Ask(ops, all, model, Json::Object{{"action", "status"}}).m_Reason.m_Code,
                  eMQ::MMQP_REASON_NO_RESPONDER);
}

// the execution boundary inside the inference workload: clamped values, rate limits and undeclared
// actuation topics (driven by explicit observations, no simulator)
EMPY_TEST(OpsWorkload, RuntimeBoundary)
{
    LocalBroker fabric;
    Topics topics;
    Descriptor::RichMLService service = HomeControl();
    Peer gateway(fabric.m_Port, "gateway"), ops(fabric.m_Port, "harness");
    gateway.Start();
    ops.Start();
    std::vector<Sighting> actions;
    ListenRaw(ops, "home/set/#", actions);
    // rejected actions are policy violations on Event(service, "violation"), debounced per boundary rule
    std::vector<Received> violations;
    ListenEnvelopes(ops, topics.EventFilter("violation"), violations);

    struct Scenario
    {
        const char *m_Name;
        Descriptor::RichMLService m_Service;
        std::vector<double> m_Weights;
    };
    Descriptor::RichMLService limited = service;
    limited.m_Actionables[0].m_MaxRatePerMinute = 3;
    Descriptor::RichMLService undeclared = service;
    undeclared.m_Inference->m_Actuation = "home/set/fan"; // not a declared actionable
    std::vector<Scenario> scenarios = {{"high", service, {100, 0, 0}},
                                       {"low", service, {-100, 0, 0}},
                                       {"limited", limited, {8, 0.5, 0.05}},
                                       {"undeclared", undeclared, {8, 0.5, 0.05}}};
    for (auto &scenario : scenarios)
    {
        actions.clear();
        violations.clear();
        WorkloadOptions options = Options(scenario.m_Service, std::string("inf-") + scenario.m_Name);
        options.m_InitialModel = Exact(1, scenario.m_Weights);
        options.m_DriftDebounceMs = 60000; // one violation event per rule for the whole scenario
        Peer predictor(fabric.m_Port, "predictor");
        auto *inference = predictor.Attach<InferenceRuntime>(options);
        predictor.Start();
        Peers all = {&gateway, &ops, &predictor};
        // ten observation pairs; every complete feature vector yields one prediction
        for (int32_t i = 0; i < 10; ++i)
        {
            Observe(gateway, 20 + 0.1 * i, 50);
            Settle(all, 15);
        }
        EMPY_REQUIRE(Pump(all, [&] { return inference->Status().GetInt("predictions") >= 19; }));
        Settle(all, 200);
        Json status = inference->Status();
        std::string name = scenario.m_Name;
        if (name == "high" || name == "low")
        {
            double bound = name == "high" ? 24 : 16;
            EMPY_CHECK_EQ(int64_t(actions.size()), status.GetInt("actions"));
            EMPY_CHECK(actions.size() >= 19);
            for (const Sighting &action : actions)
            {
                EMPY_CHECK_EQ(action.m_Topic, std::string("home/set/temp"));
                EMPY_CHECK_EQ(action.m_Body.GetNumber("value"), bound);
            }
            EMPY_CHECK_EQ(status.GetInt("clamped"), status.GetInt("actions"));
            EMPY_CHECK_EQ(status.GetInt("rejected"), int64_t{0});
            EMPY_CHECK(violations.empty()); // clamping keeps the action inside the boundary
        }
        else if (name == "limited")
        {
            // at most maxRatePerMinute actions; the rest are rejected inside the workload
            EMPY_CHECK_EQ(actions.size(), size_t{3});
            EMPY_CHECK_EQ(status.GetInt("actions"), int64_t{3});
            EMPY_CHECK(status.GetInt("rejected") >= 16);
            for (const Sighting &action : actions)
            {
                double value = action.m_Body.GetNumber("value");
                EMPY_CHECK(value >= 16 && value <= 24);
            }
            EMPY_REQUIRE_EQ(violations.size(), size_t{1}); // debounced
            const Json &body = violations[0].m_Envelope.m_Body;
            EMPY_CHECK_EQ(body.GetString("rule"), std::string("rate"));
            EMPY_CHECK_EQ(body.GetString("instance"), std::string("inf-limited"));
            EMPY_CHECK_EQ(body.GetString("topic"), std::string("home/set/temp"));
            EMPY_CHECK(body.GetString("reason").find("rate limit") != std::string::npos);
            EMPY_CHECK_EQ(body.GetInt("modelVersion"), int64_t{1});
        }
        else
        {
            // an undeclared actuation topic is never published
            EMPY_CHECK(actions.empty());
            EMPY_CHECK_EQ(status.GetInt("actions"), int64_t{0});
            EMPY_CHECK(status.GetInt("rejected") >= 19);
            EMPY_CHECK_EQ(status.GetString("actuation"), std::string("home/set/fan"));
            EMPY_REQUIRE_EQ(violations.size(), size_t{1});
            EMPY_CHECK_EQ(violations[0].m_Envelope.m_Body.GetString("rule"), std::string("domain"));
            EMPY_CHECK_EQ(violations[0].m_Envelope.m_Body.GetString("topic"), std::string("home/set/fan"));
        }
    }
}

// a latency objective no real action can meet: the observation-to-action statistic raises a
// debounced latency drift event (labels are absent, so accuracy never drifts)
EMPY_TEST(OpsWorkload, RuntimeLatencyDrift)
{
    LocalBroker fabric;
    Topics topics;
    Descriptor::RichMLService service = HomeControl();
    service.m_Inference->m_Latency->m_MaxMs = 1e-6;
    WorkloadOptions options = Options(service, "inf-0");
    options.m_InitialModel = Exact(1);
    options.m_DriftDebounceMs = 60000;
    Peer predictor(fabric.m_Port, "predictor"), gateway(fabric.m_Port, "gateway"), ops(fabric.m_Port, "harness");
    auto *inference = predictor.Attach<InferenceRuntime>(options);
    predictor.Start();
    gateway.Start();
    ops.Start();
    std::vector<Received> drifts;
    ListenEnvelopes(ops, topics.EventFilter("drift"), drifts);
    Peers all = {&predictor, &gateway, &ops};
    for (int32_t i = 0; i < 12; ++i)
    {
        Observe(gateway, 20 + 0.1 * i, 50);
        Settle(all, 15);
    }
    EMPY_REQUIRE(Pump(all, [&] { return !drifts.empty(); }));
    const Json &body = drifts[0].m_Envelope.m_Body;
    EMPY_CHECK_EQ(body.GetString("kind"), std::string("latency"));
    EMPY_CHECK(body.GetNumber("value") > 1e-6);
    EMPY_CHECK_EQ(body.GetNumber("threshold"), 1e-6);
    EMPY_CHECK(body.GetInt("samples") >= 10);
    EMPY_CHECK_EQ(body.GetInt("modelVersion"), int64_t{1});
    Json latency = inference->Status()["latency"];
    EMPY_CHECK_EQ(latency.GetNumber("maxMs"), 1e-6);
    EMPY_CHECK(latency.GetNumber("value") > 1e-6);
    // the long debounce keeps the persisting violation from repeating
    for (int32_t i = 0; i < 6; ++i)
    {
        Observe(gateway, 21, 51 + i);
        Settle(all, 15);
    }
    Settle(all, 300);
    EMPY_CHECK_EQ(drifts.size(), size_t{1});
    EMPY_CHECK_EQ(inference->Status().GetInt("drifts"), int64_t{1});
}

// accuracy drift after the simulator changes its coefficients: debounced drift events, then a
// retrained artefact restores the accuracy and the events stop
EMPY_TEST(OpsWorkload, RuntimeDrift)
{
    LocalBroker fabric;
    Topics topics;
    Descriptor::RichMLService service = HomeControl();
    WorkloadOptions options = Options(service, "inf-0");
    options.m_InitialModel = Exact(1);
    Peer predictor(fabric.m_Port, "predictor"), trainer(fabric.m_Port, "trainer"), ops(fabric.m_Port, "harness");
    auto *inference = predictor.Attach<InferenceRuntime>(options);
    auto *training = trainer.Attach<TrainingRuntime>(Options(service, "trn-0"));
    predictor.Start();
    trainer.Start();
    ops.Start();
    std::vector<Received> drifts;
    ListenEnvelopes(ops, topics.EventFilter("drift"), drifts);
    Peer gateway(fabric.m_Port, "gateway");
    auto *simulator = gateway.Attach<Simulator>(SimulatorFor(service, 10, 1500));
    gateway.Start();
    Peers all = {&predictor, &trainer, &ops, &gateway};

    // before the drift the exact model is always within tolerance
    EMPY_REQUIRE(Pump(all, [&] { return inference->Status().GetInt("samples") >= 20; }, 10000));
    EMPY_CHECK(!simulator->Drifted());
    EMPY_CHECK_EQ(inference->Status().GetNumber("accuracy"), 1.0);
    EMPY_CHECK(drifts.empty());

    EMPY_REQUIRE(Pump(all, [&] { return !drifts.empty(); }, 10000));
    EMPY_CHECK(simulator->Drifted());
    const Received first = drifts.front();
    const Json &body = first.m_Envelope.m_Body;
    EMPY_CHECK_EQ(first.m_Topic, topics.Event("home-control", "drift"));
    EMPY_CHECK_EQ(first.m_Envelope.m_Kind, std::string("event"));
    EMPY_CHECK_EQ(body.GetString("kind"), std::string("accuracy"));
    EMPY_CHECK_EQ(body.GetString("instance"), std::string("inf-0"));
    EMPY_CHECK(body.GetNumber("value", 1) < 0.9);
    EMPY_CHECK_EQ(body.GetNumber("threshold"), 0.9);
    EMPY_CHECK_EQ(body.GetInt("modelVersion"), int64_t{1});
    EMPY_CHECK(body.GetInt("samples") >= 10);

    // while the drift persists, events repeat no faster than the debounce interval (600 ms)
    Settle(all, 2000);
    EMPY_CHECK(drifts.size() >= 3 && drifts.size() <= 5);
    for (size_t i = 1; i < drifts.size(); ++i)
    {
        auto gap = std::chrono::duration_cast<std::chrono::milliseconds>(drifts[i].m_At - drifts[i - 1].m_At);
        EMPY_CHECK(gap.count() >= 450);
        EMPY_CHECK_EQ(drifts[i].m_Envelope.m_Body.GetString("kind"), std::string("accuracy"));
    }
    EMPY_CHECK_EQ(inference->Status().GetInt("drifts"), int64_t(drifts.size()));

    // the training window now holds post-drift samples only: the retrained model fits the new concept
    EMPY_CHECK_EQ(training->Samples(), size_t{40});
    Answer trained = Ask(ops, all, topics.Workload("home-control", "train"), Json::Object{{"version", 2}}, "train");
    EMPY_REQUIRE(trained.Ok());
    ModelArtifact retrained = ModelArtifact::FromJson(trained.m_Envelope->m_Body["artifact"]);
    EMPY_CHECK(Near(retrained.m_Weights[0], 12, 1e-3));
    EMPY_CHECK(Near(retrained.m_Weights[1], 0.2, 1e-4));
    EMPY_CHECK(Near(retrained.m_Weights[2], 0.1, 1e-4));
    EMPY_CHECK_EQ(retrained.m_Accuracy, 1.0);

    Answer loaded = Ask(ops, all, topics.Workload("home-control", "model"), Load(retrained));
    EMPY_REQUIRE(loaded.Ok());
    EMPY_CHECK_EQ(loaded.m_Envelope->m_Body.GetInt("previous"), int64_t{1});
    EMPY_REQUIRE(Pump(all, [&] { return inference->Status().GetInt("samples") >= 15; }));
    Settle(all, 100);
    size_t settled = drifts.size();
    Settle(all, 800);
    EMPY_CHECK_EQ(drifts.size(), settled);
    EMPY_CHECK_EQ(inference->Status().GetNumber("accuracy"), 1.0);
    EMPY_CHECK_EQ(inference->Status().GetInt("modelVersion"), int64_t{2});
}

// rollback to the previous artefact restores it; a bad model drifts with its own version
EMPY_TEST(OpsWorkload, RuntimeRollback)
{
    LocalBroker fabric;
    Topics topics;
    Descriptor::RichMLService service = HomeControl();
    WorkloadOptions options = Options(service, "inf-0");
    options.m_InitialModel = Exact(1);
    Peer predictor(fabric.m_Port, "predictor"), ops(fabric.m_Port, "harness"), gateway(fabric.m_Port, "gateway");
    auto *inference = predictor.Attach<InferenceRuntime>(options);
    gateway.Attach<Simulator>(SimulatorFor(service));
    predictor.Start();
    ops.Start();
    std::vector<Received> drifts, beats;
    ListenEnvelopes(ops, topics.EventFilter("drift"), drifts);
    ListenEnvelopes(ops, topics.Heartbeat("home-control", "inf-0"), beats);
    gateway.Start();
    Peers all = {&predictor, &ops, &gateway};
    const std::string model = topics.Workload("home-control", "model");

    // a degraded version 2 (constant prediction at the lower bound)
    Answer loaded = Ask(ops, all, model, Load(Exact(2, {0, 0, 0})));
    EMPY_REQUIRE(loaded.Ok());
    EMPY_CHECK_EQ(loaded.m_Envelope->m_Body.GetInt("loaded"), int64_t{2});
    EMPY_CHECK_EQ(loaded.m_Envelope->m_Body.GetInt("previous"), int64_t{1});
    EMPY_CHECK_EQ(inference->Status().GetInt("samples"), int64_t{0}); // the accuracy window restarts
    EMPY_REQUIRE(Pump(all, [&] { return !drifts.empty(); }));
    EMPY_CHECK_EQ(drifts[0].m_Envelope.m_Body.GetInt("modelVersion"), int64_t{2});
    EMPY_CHECK_EQ(drifts[0].m_Envelope.m_Body.GetNumber("value"), 0.0);
    EMPY_CHECK_EQ(inference->Status().GetInt("previousVersion"), int64_t{1});

    // only the previous version is a valid rollback target
    Answer wrong = Ask(ops, all, model, Json::Object{{"action", "rollback"}, {"target", 5}});
    EMPY_CHECK_EQ(wrong.m_Reason.m_Code, eMQ::MMQP_REASON_UNSPECIFIED);
    EMPY_REQUIRE(inference->Active().has_value());
    EMPY_CHECK_EQ(uint32_t(inference->Active()->m_Version), 2u);
    Answer rolled = Ask(ops, all, model, Json::Object{{"action", "rollback"}, {"target", 1}});
    EMPY_REQUIRE(rolled.Ok());
    EMPY_CHECK_EQ(rolled.m_Envelope->m_Body.GetInt("loaded"), int64_t{1});
    EMPY_CHECK_EQ(rolled.m_Envelope->m_Body.GetInt("previous"), int64_t{2});
    EMPY_CHECK(rolled.m_Envelope->m_Body.GetBool("rolledBack"));
    EMPY_REQUIRE(inference->Active().has_value());
    EMPY_CHECK_EQ(uint32_t(inference->Active()->m_Version), 1u);
    EMPY_REQUIRE(Pump(all,
                      [&]
                      {
                          return !beats.empty() && beats.back().m_Envelope.m_Body.GetInt("modelVersion") == 1 &&
                                 inference->Status().GetInt("samples") >= 15;
                      }));
    EMPY_CHECK_EQ(inference->Status().GetNumber("accuracy"), 1.0);
    size_t settled = drifts.size();
    Settle(all, 700);
    EMPY_CHECK_EQ(drifts.size(), settled);

    // reloading the same version keeps the previous one; a newer version shifts it
    EMPY_REQUIRE(Ask(ops, all, model, Load(Exact(1))).Ok());
    EMPY_CHECK_EQ(inference->Status().GetInt("previousVersion"), int64_t{2});
    Answer third = Ask(ops, all, model, Load(Exact(3)));
    EMPY_REQUIRE(third.Ok());
    EMPY_CHECK_EQ(third.m_Envelope->m_Body.GetInt("previous"), int64_t{1});
    Answer status = Ask(ops, all, model, Json::Object{{"action", "status"}});
    EMPY_REQUIRE(status.Ok());
    EMPY_CHECK_EQ(status.m_Envelope->m_Kind, std::string("status"));
    EMPY_CHECK_EQ(status.m_Envelope->m_Body.GetInt("modelVersion"), int64_t{3});
}

// the synthetic gateway: bounded deterministic signals, the ground-truth label, locality reports
EMPY_TEST(OpsWorkload, SimulatorSignals)
{
    Descriptor::RichMLService service = HomeControl();
    Topics topics;
    std::vector<std::vector<double>> runs;
    for (int32_t run = 0; run < 2; ++run)
    {
        LocalBroker fabric;
        Peer ops(fabric.m_Port, "harness"), gateway(fabric.m_Port, "gateway");
        ops.Start();
        std::vector<Sighting> data;
        std::vector<Received> locality;
        ListenRaw(ops, "home/#", data);
        ListenEnvelopes(ops, topics.LocalityFilter(), locality);
        auto *simulator = gateway.Attach<Simulator>(SimulatorFor(service, 10, run == 0 ? 400 : 0));
        gateway.Start();
        Peers all = {&ops, &gateway};
        EMPY_REQUIRE(Pump(all, [&] { return data.size() >= 180 && !locality.empty(); }, 10000));
        EMPY_CHECK(simulator->Published() > 0);

        // label = a·temp + b·hum + c of the same tick, before and after the induced drift
        std::optional<double> temperature, humidity;
        size_t labels = 0, drifted = 0;
        std::vector<double> sequence;
        for (const Sighting &sighting : data)
        {
            double value = sighting.m_Body.GetNumber("value", std::nan(""));
            EMPY_REQUIRE(std::isfinite(value));
            EMPY_CHECK(sighting.m_Body.GetInt("time") > 0);
            if (sighting.m_Topic == "home/temp")
            {
                EMPY_CHECK(value >= 17 && value <= 23);
                EMPY_CHECK_EQ(sighting.m_Body.GetString("unit"), std::string("celsius"));
                temperature = value;
                sequence.push_back(value);
            }
            else if (sighting.m_Topic == "home/hum")
            {
                EMPY_CHECK(value >= 40 && value <= 60);
                humidity = value;
            }
            else
            {
                EMPY_REQUIRE_EQ(sighting.m_Topic, std::string("home/label/temp"));
                EMPY_REQUIRE(temperature && humidity);
                double base = 0.5 * *temperature + 0.05 * *humidity + 8;
                double shifted = 0.2 * *temperature + 0.1 * *humidity + 12;
                EMPY_CHECK(Near(value, base, 1e-9) || Near(value, shifted, 1e-9));
                EMPY_CHECK(value >= 16 && value <= 24);
                ++labels;
                drifted += Near(value, shifted, 1e-9) && !Near(value, base, 1e-9);
            }
        }
        EMPY_CHECK(labels >= 50);
        EMPY_CHECK_EQ(simulator->Drifted(), run == 0);
        EMPY_CHECK(run == 0 ? drifted > 0 : drifted == 0);
        runs.push_back(std::vector<double>(sequence.begin(), sequence.begin() + 20));

        const Received &report = locality.front();
        EMPY_CHECK_EQ(report.m_Topic, topics.Locality("edge"));
        EMPY_CHECK_EQ(report.m_Envelope.m_Kind, std::string("locality"));
        EMPY_CHECK_EQ(report.m_Envelope.m_Body.GetString("region"), std::string("edge"));
        EMPY_CHECK_EQ(report.m_Envelope.m_Body["topics"].GetNumber("home/temp"), 12.5);
        EMPY_CHECK_EQ(report.m_Envelope.m_Body["topics"].GetNumber("home/hum"), 12.5);
        EMPY_CHECK_EQ(report.m_Envelope.m_Body["topics"].Size(), size_t{2});
    }
    // the same seed yields the same signal sequence
    EMPY_CHECK(runs[0] == runs[1]);
}

// EmpyWorkload as a child process against an in-process broker: simulate + training + inference
EMPY_TEST(OpsWorkload, ExecutableRoles)
{
#ifdef _WIN32
    const std::filesystem::path program = eTest::BinaryDirectory() / "EmpyWorkload.exe";
#else
    const std::filesystem::path program = eTest::BinaryDirectory() / "EmpyWorkload";
#endif
    EMPY_REQUIRE(std::filesystem::exists(program));
    const std::filesystem::path temp = std::filesystem::temp_directory_path() / ("EmpyOpsWorkload-" + NewId({}));
    std::filesystem::create_directories(temp);
    struct Cleanup
    {
        std::filesystem::path m_Path;
        ~Cleanup()
        {
            std::error_code ignored;
            std::filesystem::remove_all(m_Path, ignored);
        }
    } cleanup{temp};

    auto run = [&](std::vector<std::string> args, std::string log) -> std::unique_ptr<Process>
    {
        ProcessOptions options;
        options.m_Program = program;
        options.m_Args = std::move(args);
        options.m_WorkingDirectory = temp;
        options.m_StdoutFile = temp / log;
        options.m_StderrFile = temp / log;
        auto process = std::make_unique<Process>();
        std::string error;
        EMPY_REQUIRE(process->Start(options, &error));
        return process;
    };
    auto exitCode = [&](std::vector<std::string> args) -> int32_t
    {
        auto process = run(std::move(args), "usage.log");
        auto code = process->Wait(std::chrono::seconds(20));
        EMPY_REQUIRE(code.has_value());
        return *code;
    };
    EMPY_CHECK_EQ(exitCode({"--help"}), 0);
    EMPY_CHECK_EQ(exitCode({}), 2);
    EMPY_CHECK_EQ(exitCode({"predict", "--descriptor", "x.yaml"}), 2);
    EMPY_CHECK_EQ(exitCode({"inference"}), 2);                                 // no descriptor
    EMPY_CHECK_EQ(exitCode({"inference", "--descriptor", "x", "--bogus"}), 2); // unknown option
    EMPY_CHECK_EQ(exitCode({"training", "--descriptor", "x", "--port", "70000"}), 2);
    EMPY_CHECK_EQ(exitCode({"simulate", "--descriptor", (temp / "missing.yaml").string()}), 1);

    LocalBroker fabric;
    const std::string port = std::to_string(fabric.m_Port);
    const std::string descriptor = HomeControlFile().string();
    const std::filesystem::path modelFile = temp / "model.json";
    WriteFileAtomic(modelFile, Exact(1).ToJson().Dump(2));
    WriteFileAtomic(temp / "bad-model.json", Exact(1, {1, 2}).ToJson().Dump());
    // a mismatched initial model is refused before connecting
    EMPY_CHECK_EQ(exitCode({"inference", "--descriptor", descriptor, "--port", port, "--model",
                            (temp / "bad-model.json").string()}),
                  1);

    Topics topics;
    Peer ops(fabric.m_Port, "harness");
    ops.Start();
    std::vector<Sighting> actions;
    std::vector<Received> beats;
    ListenRaw(ops, "home/set/#", actions);
    ListenEnvelopes(ops, topics.HeartbeatFilter(), beats);
    const std::vector<std::string> common = {"--descriptor", descriptor, "--host",    "127.0.0.1",
                                             "--port",       port,       "--seconds", "8"};
    auto with = [&](std::vector<std::string> head, std::vector<std::string> tail)
    {
        head.insert(head.end(), common.begin(), common.end());
        head.insert(head.end(), tail.begin(), tail.end());
        return head;
    };
    auto inference = run(with({"inference"}, {"--instance", "exe-inf", "--node", "edge-1", "--region", "edge",
                                              "--model", modelFile.string(), "--heartbeat-ms", "200"}),
                         "inference.log");
    auto training =
        run(with({"training"}, {"--instance", "exe-trn", "--node", "cloud-1", "--region", "cloud"}), "training.log");
    auto simulate = run(with({"simulate"}, {"--period-ms", "20", "--region", "ran"}), "simulate.log");
    Peers all = {&ops};

    auto beat = [&](const std::string &instance, const std::string &state)
    {
        for (const Received &received : beats)
        {
            if (received.m_Envelope.m_Body.GetString("instance") == instance &&
                received.m_Envelope.m_Body.GetString("state") == state)
            {
                return true;
            }
        }
        return false;
    };
    EMPY_REQUIRE(Pump(all, [&] { return actions.size() >= 20 && beat("exe-inf", "ready"); }, 7000));
    for (const Sighting &action : actions)
    {
        EMPY_CHECK_EQ(action.m_Topic, std::string("home/set/temp"));
        EMPY_CHECK_EQ(action.m_Body.GetString("instance"), std::string("exe-inf"));
        EMPY_CHECK_EQ(action.m_Body.GetInt("model"), int64_t{1});
        double value = action.m_Body.GetNumber("value");
        EMPY_CHECK(value >= 16 && value <= 24);
    }
    // the training child has collected Listing 2.1's minimum of 32 samples within a second or two
    Answer trained;
    EMPY_REQUIRE(Pump(
        all,
        [&]
        {
            trained = Ask(ops, all, topics.Workload("home-control", "train"), Json::Object{{"version", 7}}, "train");
            if (!trained.Ok())
            {
                Settle(all, 100); // not subscribed yet or still collecting samples
            }
            return trained.Ok();
        },
        5000));
    EMPY_CHECK_EQ(trained.m_Envelope->m_Body["artifact"].GetInt("version"), int64_t{7});
    EMPY_CHECK_EQ(trained.m_Envelope->m_Body.GetString("instance"), std::string("exe-trn"));
    EMPY_CHECK_EQ(trained.m_Envelope->m_Body["artifact"].GetNumber("accuracy"), 1.0);
    EMPY_CHECK(beat("exe-trn", "ready") || beat("exe-trn", "collecting"));

    // every role exits cleanly after --seconds and leaves a "stopped" heartbeat
    for (auto *process : {inference.get(), training.get(), simulate.get()})
    {
        std::optional<int32_t> code;
        EMPY_REQUIRE(
            Pump(all, [&] { return (code = process->Wait(std::chrono::milliseconds(0))).has_value(); }, 15000));
        EMPY_CHECK_EQ(*code, 0);
    }
    EMPY_REQUIRE(Pump(all, [&] { return beat("exe-inf", "stopped") && beat("exe-trn", "stopped"); }));
}
