#include "Cli.h"
#include <thread>
#include <algorithm>

// EmpyHarness online commands: run (harness runtime and agent swarm) and the operator commands that talk to a
// running harness over MMQP (intent, deploy, status, approve)

namespace eHarness
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        constexpr auto STATUS_POLL = std::chrono::milliseconds(250);

        std::optional<std::filesystem::path> Discover()
        {
            std::vector<std::filesystem::path> starts = {std::filesystem::current_path()};
            std::filesystem::path executable = eOps::Substrate::DefaultWorkloadExecutable().parent_path();
            if (!executable.empty())
            {
                starts.push_back(executable);
            }
            for (std::filesystem::path directory : starts)
            {
                for (int32_t level = 0; level < 8 && !directory.empty(); ++level)
                {
                    std::filesystem::path candidate = directory / "Configs" / "MLOps" / "Harness.yaml";
                    std::error_code error;
                    if (std::filesystem::is_regular_file(candidate, error))
                    {
                        return candidate;
                    }
                    if (directory == directory.parent_path())
                    {
                        break;
                    }
                    directory = directory.parent_path();
                }
            }
            return std::nullopt;
        }

        // one MMQP session of an operator command
        struct Session
        {
            std::unique_ptr<eMQ::Client> m_Client;

            Session(const eOps::Harness::HarnessConfig &config, uint32_t responseMs, const std::string &source)
                : m_Client(std::make_unique<eMQ::Client>(ClientOptions(config, responseMs)))
            {
                m_Client->Connect(config.m_BrokerHost, config.m_BrokerPort);
                eOps::Fabric::BusOptions bus;
                bus.m_Source = source;
                bus.m_DefaultTimeoutMs = responseMs;
                m_Client->AttachModule<eOps::Fabric::BusModule>(bus);
                m_Client->Poll();
            }

            ~Session()
            {
                m_Client->Shutdown();
            }

            eOps::Fabric::Bus &Get()
            {
                return eOps::Fabric::BusOf(*m_Client);
            }

            eOps::Fabric::Envelope Call(const std::string &topic, eOps::Fabric::Envelope request, uint32_t timeoutMs)
            {
                auto [reason, reply] = Get().CallSync(
                    topic, request,
                    [this]
                    {
                        m_Client->Poll(5);
                        if (g_Interrupted)
                        {
                            throw std::runtime_error("interrupted");
                        }
                    },
                    timeoutMs);
                if (reason)
                {
                    std::string text =
                        reason.m_Text.empty() ? "reason " + std::to_string(reason.m_Code) : reason.m_Text;
                    if (reason.m_Code == eMQ::MMQP_REASON_NO_RESPONDER)
                    {
                        text += " (is `EmpyHarness run` connected to this broker?)";
                    }
                    throw std::runtime_error(topic + ": " + text);
                }
                if (!reply)
                {
                    throw std::runtime_error(topic + ": empty response");
                }
                return std::move(*reply);
            }
        };

        uint32_t WorkflowTimeout(const Options &options, const eOps::Harness::HarnessConfig &config)
        {
            if (options.m_TimeoutMs)
            {
                return *options.m_TimeoutMs;
            }
            uint64_t timeout = static_cast<uint64_t>(config.m_AgentTimeoutMs) * 4 + config.m_HitlTimeoutMs +
                               config.m_ReadyTimeoutMs + 10000;
            return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(timeout, 120000), UINT32_MAX));
        }

        // submit with wait=false, then follow the workflow through Status() until it finishes; prints the record
        int32_t Submit(const Options &options, const std::string &topic, const char *kind, eOps::Json body)
        {
            eOps::Harness::HarnessConfig config = Effective(options);
            eOps::Fabric::Topics topics(config.m_TopicRoot);
            uint32_t timeout = WorkflowTimeout(options, config);
            Session session(config, std::max<uint32_t>(timeout, 10000),
                            options.m_Operator.empty() ? "operator" : options.m_Operator);
            body.Set("wait", false);
            eOps::Fabric::Envelope reply = session.Call(topic, eOps::Fabric::Envelope::Make(kind, std::move(body)),
                                                        std::min<uint32_t>(timeout, 30000));
            eOps::Json record = reply.m_Body;
            auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
            auto finished = [](const eOps::Json &workflow)
            {
                std::string status = workflow.GetString("status");
                return status == "succeeded" || status == "failed" || status == "rejected";
            };
            while (!options.m_NoWait && !finished(record))
            {
                if (Clock::now() >= deadline)
                {
                    Print(record);
                    std::cerr << "EmpyHarness: workflow " << record.GetString("id") << " still "
                              << record.GetString("status") << " after " << timeout << " ms\n";
                    return EXIT_FAILURE;
                }
                std::this_thread::sleep_for(STATUS_POLL);
                eOps::Fabric::Envelope snapshot =
                    session.Call(topics.Status(), eOps::Fabric::Envelope::Make("status-request", eOps::Json()), 10000);
                if (const eOps::Json *workflows = snapshot.m_Body.Find("workflows"); workflows && workflows->IsArray())
                {
                    for (const eOps::Json &workflow : workflows->AsArray())
                    {
                        if (workflow.GetString("id") == record.GetString("id"))
                        {
                            record = workflow;
                        }
                    }
                }
            }
            Print(record);
            std::string status = record.GetString("status");
            return status == "succeeded" || (options.m_NoWait && (status == "running" || status == "waiting-hitl"))
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE;
        }
    } // namespace

    void InstallSignals()
    {
        std::signal(SIGINT, [](int) { g_Interrupted = 1; });
        std::signal(SIGTERM, [](int) { g_Interrupted = 1; });
    }

    eOps::Harness::HarnessConfig LoadConfig(const Options &options)
    {
        std::string path = options.m_Config;
        if (path.empty())
        {
            if (const char *variable = std::getenv("EMPY_MLOPS_CONFIG"); variable && *variable)
            {
                path = variable;
            }
            else if (std::optional<std::filesystem::path> found = Discover())
            {
                path = found->string();
            }
        }
        if (path.empty())
        {
            std::cerr << "EmpyHarness: no configuration found; using built-in defaults (no ontology or knowledge)\n";
            return eOps::Harness::ParseHarnessConfig("", "");
        }
        return eOps::Harness::LoadHarnessConfig(path);
    }

    eOps::Harness::HarnessConfig Effective(const Options &options)
    {
        eOps::Harness::HarnessConfig config = LoadConfig(options);
        if (options.m_Host)
        {
            config.m_BrokerHost = *options.m_Host;
        }
        if (options.m_Port)
        {
            config.m_BrokerPort = *options.m_Port;
        }
        if (options.m_Root)
        {
            config.m_TopicRoot = eOps::Fabric::Topics(*options.m_Root).m_Root;
        }
        if (options.m_Roles)
        {
            for (const std::string &role : *options.m_Roles)
            {
                if (role != "harness" && role != "executive" && role != "placement" && role != "lifecycle" &&
                    role != "compliance")
                {
                    throw UsageError("--roles: unknown role '" + role + "'");
                }
            }
            config.m_Roles = *options.m_Roles;
        }
        if (options.m_StatusPort)
        {
            config.m_StatusPort = *options.m_StatusPort;
            config.m_StatusEnabled = true;
        }
        return config;
    }

    eOps::Harness::HarnessConfig Offline(eOps::Harness::HarnessConfig config, bool durableKnowledge)
    {
        config.m_McpCommand.clear();
        if (!durableKnowledge)
        {
            config.m_KnowledgeDirectory.clear();
        }
        config.m_Substrate = eCore::YamlNode();
        return config;
    }

    eMQ::Configs ClientOptions(const eOps::Harness::HarnessConfig &config, uint32_t responseMs)
    {
        eMQ::Configs options =
            config.m_ClientConfig.empty() ? eMQ::Configs{} : eMQ::LoadConfigs(config.m_ClientConfig.c_str());
        options.m_ResponseTimeout = std::max(options.m_ResponseTimeout, responseMs);
        return options;
    }

    namespace
    {
        std::streambuf *g_Result = nullptr; // the process stdout while logging is routed to stderr
    }

    StdoutRoute::StdoutRoute()
    {
        g_Result = std::cout.rdbuf();
        std::cout.rdbuf(std::cerr.rdbuf());
    }

    StdoutRoute::~StdoutRoute()
    {
        Out().flush();
        std::cout.rdbuf(g_Result);
        g_Result = nullptr;
    }

    std::ostream &Out()
    {
        static std::ostream stream(nullptr);
        stream.rdbuf(g_Result ? g_Result : std::cout.rdbuf());
        return stream;
    }

    void Print(const eOps::Json &json)
    {
        Out() << json.Dump(2) << std::endl;
    }

    // -------------------------
    // run
    // -------------------------

    int32_t Run(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Info);
        eOps::Harness::HarnessConfig config = Effective(options);
        InstallSignals();
        eOps::Harness::HarnessComponents components = eOps::Harness::BuildComponents(config);
        eOps::Harness::AgentHostOptions agents = eOps::Harness::MakeAgentHostOptions(config, components);
        bool harnessRole = config.HasRole("harness");
        if (!harnessRole && agents.m_Roles.empty())
        {
            throw UsageError("no role selected");
        }
        // proposals wait for whole workflows; HITL questions for operators
        uint64_t response = std::max<uint64_t>(agents.m_HarnessTimeoutMs, config.m_HitlTimeoutMs) + 1000;
        std::unique_ptr<eOps::Harness::StatusServer> status;
        eMQ::Client client(ClientOptions(config, static_cast<uint32_t>(std::min<uint64_t>(response, UINT32_MAX))));
        client.Connect(config.m_BrokerHost, config.m_BrokerPort);
        eOps::Fabric::BusOptions bus;
        bus.m_Source = config.m_Identity;
        client.AttachModule<eOps::Fabric::BusModule>(bus);
        if (!agents.m_Roles.empty())
        {
            client.AttachModule<eOps::Harness::AgentHost>(agents, components.Environment());
        }
        uint16_t statusPort = 0;
        if (harnessRole)
        {
            auto *harness = client.AttachModule<eOps::Harness::ExecutionHarness>(
                eOps::Harness::MakeHarnessOptions(config, components), components.m_Substrate, components.m_Comas);
            if (config.m_StatusEnabled)
            {
                status = std::make_unique<eOps::Harness::StatusServer>(config.m_StatusAddress, config.m_StatusPort);
                statusPort = status->Start();
                eOps::Harness::StatusServer *server = status.get();
                harness->SetSnapshotSink([server](const eOps::Json &snapshot) { server->Update(snapshot); });
            }
        }
        client.Poll();
        std::cout << "EmpyHarness running: identity " << config.m_Identity << ", roles "
                  << eOps::Join(config.m_Roles, ",") << ", broker " << config.m_BrokerHost << ":" << config.m_BrokerPort
                  << ", root " << config.m_TopicRoot << ", substrate " << components.m_Substrate->Name();
        if (statusPort)
        {
            std::cout << ", status http://" << config.m_StatusAddress << ":" << statusPort << "/api/mlops/status";
        }
        std::cout << std::endl;
        auto deadline =
            options.m_Seconds ? Clock::now() + std::chrono::seconds(*options.m_Seconds) : Clock::time_point::max();
        // a stop file is a portable graceful stop (a supervisor without signals, the integration tests)
        auto stopRequested = [&options]
        {
            std::error_code ignored;
            return !options.m_StopFile.empty() && std::filesystem::exists(options.m_StopFile, ignored);
        };
        while (!g_Interrupted && !client.IsTerminated() && Clock::now() < deadline && !stopRequested())
        {
            client.Poll(50);
        }
        client.Shutdown();
        if (status)
        {
            status->Stop();
        }
        std::cout << "EmpyHarness stopped" << std::endl;
        return EXIT_SUCCESS;
    }

    // -------------------------
    // operator commands
    // -------------------------

    int32_t Intent(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        InstallSignals();
        eOps::Json body =
            eOps::Json::MakeObject({{"intent", options.m_Positional.front()}, {"dryRun", options.m_DryRun}});
        if (!options.m_Operator.empty())
        {
            body.Set("operator", options.m_Operator);
        }
        eOps::Harness::HarnessConfig config = Effective(options);
        return Submit(options, eOps::Fabric::Topics(config.m_TopicRoot).Intent(), "intent", std::move(body));
    }

    int32_t Deploy(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        InstallSignals();
        std::string descriptor = eOps::ReadFile(options.m_Positional.front(), 1 << 20);
        eOps::Json body = eOps::Json::MakeObject({{"descriptor", descriptor}, {"dryRun", options.m_DryRun}});
        eOps::Harness::HarnessConfig config = Effective(options);
        return Submit(options, eOps::Fabric::Topics(config.m_TopicRoot).Deploy(), "deploy", std::move(body));
    }

    int32_t Status(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        InstallSignals();
        eOps::Harness::HarnessConfig config = Effective(options);
        uint32_t timeout = options.m_TimeoutMs.value_or(10000);
        Session session(config, std::max<uint32_t>(timeout, 1000), "operator");
        eOps::Fabric::Envelope snapshot =
            session.Call(eOps::Fabric::Topics(config.m_TopicRoot).Status(),
                         eOps::Fabric::Envelope::Make("status-request", eOps::Json()), timeout);
        Print(snapshot.m_Body);
        return EXIT_SUCCESS;
    }

    int32_t Approve(const Options &options)
    {
        eCore::SetLevel(options.m_Verbose ? eCore::LogLevel::Debug : eCore::LogLevel::Warn);
        InstallSignals();
        eOps::Harness::HarnessConfig config = Effective(options);
        std::string identity = options.m_Operator.empty() ? "operator" : options.m_Operator;
        Session session(config, 10000, identity);
        std::string decision = options.m_Deny ? "reject" : "approve";
        uint32_t answered = 0;
        eMQ::Reason reason = session.Get().Serve(
            eOps::Fabric::Topics(config.m_TopicRoot).Hitl(),
            [&](const eOps::Fabric::Inbound &request, eOps::Fabric::Bus::Responder respond)
            {
                eOps::Json question =
                    eOps::Json::MakeObject({{"workflowId", request.m_Envelope.m_Body.GetString("workflowId")},
                                            {"service", request.m_Envelope.m_Body.GetString("service")},
                                            {"question", request.m_Envelope.m_Body.GetString("question")},
                                            {"findings", request.m_Envelope.m_Body["findings"]},
                                            {"decision", decision}});
                Out() << question.Dump() << std::endl;
                respond(eOps::Fabric::Envelope::Make(
                            "hitl-decision",
                            eOps::Json::MakeObject(
                                {{"decision", decision}, {"operator", identity}, {"note", options.m_Note}}),
                            identity),
                        {});
                ++answered;
            });
        if (reason)
        {
            throw std::runtime_error("cannot serve HITL approvals: " + reason.m_Text);
        }
        std::cerr << "EmpyHarness approve: answering HITL questions with '" << decision << "' as " << identity << "\n";
        auto deadline =
            options.m_Seconds ? Clock::now() + std::chrono::seconds(*options.m_Seconds) : Clock::time_point::max();
        while (!g_Interrupted && Clock::now() < deadline && (!options.m_Count || answered < *options.m_Count))
        {
            session.m_Client->Poll(50);
        }
        // let the last answer leave before the session closes
        for (int32_t i = 0; i < 10; ++i)
        {
            session.m_Client->Poll(10);
        }
        std::cerr << "EmpyHarness approve: answered " << answered << " question(s)\n";
        return EXIT_SUCCESS;
    }
} // namespace eHarness
