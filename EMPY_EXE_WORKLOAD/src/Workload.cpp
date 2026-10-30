#include <eOps/API.h>
#include <eOps/Common/Text.h>
#include <eOps/Descriptor/Parser.h>
#include <eOps/Descriptor/Validator.h>
#include <eOps/Workload/Runtime.h>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <filesystem>
#include <functional>

// EmpyWorkload: the reference ML workloads of a RichML-Service as an MMQP client process
//   inference  subscribes the observables, acts on the actuation topic within the declared bounds
//   training   collects labelled samples and answers train requests with model artefacts
//   simulate   synthetic IoT gateway publishing the observables and their ground-truth label
// Launched by the harness substrates (ProcessSubstrate, Kubernetes manifests) or by hand.

namespace
{
    // Signal handlers may only write sig_atomic_t; shutdown happens on the owner thread.
    volatile std::sig_atomic_t g_Interrupted = 0;

    // command-line mistakes: usage and exit status 2
    struct UsageError : std::invalid_argument
    {
        using std::invalid_argument::invalid_argument;
    };

    void Usage(std::ostream &out)
    {
        out << "EmpyWorkload inference|training|simulate --descriptor FILE   (" << eOps::EmpyOpsTag() << ")\n"
            << "  [--instance ID] [--node N] [--region R] [--host H] [--port P] [--root R] [--config CLIENT.yaml]\n"
            << "  [--model FILE.json] [--seconds N] [--period-ms N] [--drift-after-ms N] [--verbose]\n"
            << "  [--heartbeat-ms N] [--metrics-ms N] [--identity ID] [--stop-file PATH]\n"
            << "  inference  --model loads an initial artefact (JSON); --heartbeat-ms/--metrics-ms set the\n"
            << "             status and metrics periods (default 2000)\n"
            << "  training   answers <root>/workload/<service>/train with artefacts\n"
            << "  simulate   --period-ms sets the signal period (default 100), --drift-after-ms induces\n"
            << "             concept drift, --region names the reported locality region (default ran)\n"
            << "  --identity authenticates the MMQP session as ID (with the auth key of --config)\n"
            << "  --seconds  runs for N seconds (default: until SIGINT/SIGTERM or until --stop-file exists)\n";
    }

    uint32_t Number(std::string_view text, uint32_t maximum)
    {
        uint32_t value = 0;
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > maximum)
        {
            throw UsageError("Invalid numeric argument '" + std::string(text) + "'");
        }
        return value;
    }

    struct Arguments
    {
        std::string m_Command;
        std::string m_Descriptor;
        std::string m_Instance;
        std::string m_Node;
        std::optional<std::string> m_Region;
        std::string m_Host = "127.0.0.1";
        uint16_t m_Port = 5600;
        std::string m_Root = "mlops";
        std::string m_Config;
        std::string m_Identity;
        std::string m_StopFile;
        std::string m_Model;
        uint32_t m_Seconds = 0;
        std::optional<uint32_t> m_PeriodMs;
        std::optional<uint32_t> m_DriftAfterMs;
        std::optional<uint32_t> m_HeartbeatMs;
        std::optional<uint32_t> m_MetricsMs;
        bool m_Verbose = false;
        bool m_Help = false;
    };

    Arguments Parse(int32_t argc, char **argv)
    {
        Arguments arguments;
        for (int32_t i = 1; i < argc; ++i)
        {
            const std::string_view argument(argv[i]);
            if (argument == "--help" || argument == "help")
            {
                arguments.m_Help = true;
                return arguments;
            }
            if (!argument.starts_with("--"))
            {
                if (!arguments.m_Command.empty())
                {
                    throw UsageError("Unexpected argument '" + std::string(argument) + "'");
                }
                arguments.m_Command = argument;
                continue;
            }
            if (argument == "--verbose")
            {
                arguments.m_Verbose = true;
                continue;
            }
            if (++i == argc)
            {
                throw UsageError("Missing value for " + std::string(argument));
            }
            const std::string value(argv[i]);
            if (argument == "--descriptor")
            {
                arguments.m_Descriptor = value;
            }
            else if (argument == "--instance")
            {
                arguments.m_Instance = value;
            }
            else if (argument == "--node")
            {
                arguments.m_Node = value;
            }
            else if (argument == "--region")
            {
                arguments.m_Region = value;
            }
            else if (argument == "--host")
            {
                arguments.m_Host = value;
            }
            else if (argument == "--port")
            {
                arguments.m_Port = static_cast<uint16_t>(Number(value, UINT16_MAX));
            }
            else if (argument == "--root")
            {
                arguments.m_Root = value;
            }
            else if (argument == "--config")
            {
                arguments.m_Config = value;
            }
            else if (argument == "--identity")
            {
                if (value.empty())
                {
                    throw UsageError("--identity must not be empty");
                }
                arguments.m_Identity = value;
            }
            else if (argument == "--stop-file")
            {
                arguments.m_StopFile = value;
            }
            else if (argument == "--model")
            {
                arguments.m_Model = value;
            }
            else if (argument == "--seconds")
            {
                arguments.m_Seconds = Number(value, 31 * 86400);
            }
            else if (argument == "--period-ms")
            {
                arguments.m_PeriodMs = Number(value, 3600000);
            }
            else if (argument == "--drift-after-ms")
            {
                arguments.m_DriftAfterMs = Number(value, UINT32_MAX);
            }
            else if (argument == "--heartbeat-ms")
            {
                arguments.m_HeartbeatMs = Number(value, 3600000);
            }
            else if (argument == "--metrics-ms")
            {
                arguments.m_MetricsMs = Number(value, 3600000);
            }
            else
            {
                throw UsageError("Unknown option " + std::string(argument));
            }
        }
        const auto &command = arguments.m_Command;
        if (command != "inference" && command != "training" && command != "simulate")
        {
            throw UsageError(command.empty() ? "Missing command" : "Unknown command '" + command + "'");
        }
        if (arguments.m_Descriptor.empty())
        {
            throw UsageError("--descriptor FILE is required");
        }
        if (!arguments.m_Model.empty() && command != "inference")
        {
            throw UsageError("--model applies to inference only");
        }
        if ((arguments.m_PeriodMs || arguments.m_DriftAfterMs) && command != "simulate")
        {
            throw UsageError("--period-ms and --drift-after-ms apply to simulate only");
        }
        if (arguments.m_MetricsMs && command != "inference")
        {
            throw UsageError("--metrics-ms applies to inference only");
        }
        if (arguments.m_HeartbeatMs && command == "simulate")
        {
            throw UsageError("--heartbeat-ms applies to inference and training");
        }
        if (arguments.m_PeriodMs && !*arguments.m_PeriodMs)
        {
            throw UsageError("--period-ms must be at least 1");
        }
        return arguments;
    }

    // a descriptor the workload may run: parsed and free of validation errors
    eOps::Descriptor::RichMLService LoadService(const std::string &path)
    {
        auto service = eOps::Descriptor::LoadDescriptor(path);
        auto issues = eOps::Descriptor::Validate(service);
        if (eOps::Descriptor::HasErrors(issues))
        {
            std::string text;
            for (const auto &issue : issues)
            {
                if (issue.m_Severity == eOps::Descriptor::Severity::Error)
                {
                    text += "\n  " + issue.m_Path + ": " + issue.m_Message;
                }
            }
            throw std::runtime_error("Descriptor " + path + " is invalid:" + text);
        }
        return service;
    }

    eOps::Workload::WorkloadOptions MakeOptions(const Arguments &arguments, eOps::Descriptor::RichMLService service)
    {
        eOps::Workload::WorkloadOptions options;
        options.m_Service = std::move(service);
        options.m_Instance = arguments.m_Instance;
        options.m_Node = arguments.m_Node;
        options.m_Region = arguments.m_Region.value_or("");
        options.m_Topics = eOps::Fabric::Topics(arguments.m_Root);
        if (arguments.m_HeartbeatMs)
        {
            options.m_HeartbeatMs = *arguments.m_HeartbeatMs;
        }
        if (arguments.m_MetricsMs)
        {
            options.m_MetricsMs = *arguments.m_MetricsMs;
        }
        if (!arguments.m_Model.empty())
        {
            options.m_InitialModel =
                eOps::Workload::ModelArtifact::FromJson(eOps::Json::Parse(eOps::ReadFile(arguments.m_Model)));
        }
        return options;
    }

    // the instance identity: explicit, else "<service>-<role>-<node|local>-0" ("<service>-simulator")
    std::string Instance(const Arguments &arguments, const eOps::Descriptor::RichMLService &service)
    {
        if (!arguments.m_Instance.empty())
        {
            return arguments.m_Instance;
        }
        const auto &name = service.m_Metadata.m_Name;
        if (arguments.m_Command == "simulate")
        {
            return name + "-simulator";
        }
        return name + "-" + arguments.m_Command + "-" + (arguments.m_Node.empty() ? "local" : arguments.m_Node) + "-0";
    }

    // attaches the role's runtime (its constructor validates the options) and returns how to detach it
    std::function<void()> AttachRole(eMQ::Client &client, const Arguments &arguments,
                                     eOps::Descriptor::RichMLService service, const std::string &instance)
    {
        using namespace eOps::Workload;
        if (arguments.m_Command == "simulate")
        {
            SimulatorOptions options;
            options.m_Service = std::move(service);
            options.m_Topics = eOps::Fabric::Topics(arguments.m_Root);
            options.m_Region = arguments.m_Region.value_or(options.m_Region);
            if (arguments.m_PeriodMs)
            {
                options.m_PeriodMs = *arguments.m_PeriodMs;
            }
            if (arguments.m_DriftAfterMs)
            {
                options.m_DriftAfterMs = *arguments.m_DriftAfterMs;
            }
            client.AttachModule<Simulator>(std::move(options));
            return [&client] { client.DetachModule<Simulator>(); };
        }
        auto options = MakeOptions(arguments, std::move(service));
        options.m_Instance = instance;
        if (arguments.m_Command == "training")
        {
            client.AttachModule<TrainingRuntime>(std::move(options));
            return [&client] { client.DetachModule<TrainingRuntime>(); };
        }
        client.AttachModule<InferenceRuntime>(std::move(options));
        return [&client] { client.DetachModule<InferenceRuntime>(); };
    }
} // namespace

int32_t main(int32_t argc, char **argv)
{
    Arguments arguments;
    try
    {
        arguments = Parse(argc, argv);
    }
    catch (const UsageError &error)
    {
        std::cerr << "EmpyWorkload: " << error.what() << '\n';
        Usage(std::cerr);
        return 2;
    }
    if (arguments.m_Help)
    {
        Usage(std::cout);
        return EXIT_SUCCESS;
    }
    try
    {
        eCore::SetLevel(arguments.m_Verbose ? eCore::LogLevel::Info : eCore::LogLevel::Warn);
        auto service = LoadService(arguments.m_Descriptor);
        eMQ::Configs configs =
            arguments.m_Config.empty() ? eMQ::Configs{} : eMQ::LoadConfigs(arguments.m_Config.c_str());
        if (!arguments.m_Identity.empty())
        {
            configs.m_AuthIdentity = arguments.m_Identity; // the descriptor's security.identity (harness-assigned)
        }
        std::signal(SIGINT, [](int) { g_Interrupted = 1; });
        std::signal(SIGTERM, [](int) { g_Interrupted = 1; });

        eMQ::Client client(std::move(configs));
        std::string failure;
        bool stopping = false;
        client.On<eMQ::Disconnect>(
            [&](const eMQ::Disconnect &event)
            {
                if (!event.m_Reconnecting && !stopping)
                {
                    failure = "MMQP connection closed: " + std::to_string(event.m_Reason.m_Code) + " " +
                              event.m_Reason.m_Text;
                    g_Interrupted = 1;
                }
            });
        // modules start on the first Poll (after Connect); constructing them validates the options first
        const std::string instance = Instance(arguments, service);
        eOps::Fabric::BusOptions bus;
        bus.m_Source = instance;
        client.AttachModule<eOps::Fabric::BusModule>(bus);
        auto detach = AttachRole(client, arguments, std::move(service), instance);

        client.Connect(arguments.m_Host, arguments.m_Port);
        client.Poll(); // OnAttach: profiles, subscriptions and responders are in place after this
        std::cout << "EmpyWorkload " << arguments.m_Command << " " << instance << " connected to " << arguments.m_Host
                  << ":" << arguments.m_Port << std::endl;
        const auto deadline = arguments.m_Seconds ? eCore::Clock::now() + std::chrono::seconds(arguments.m_Seconds)
                                                  : eCore::Clock::time_point::max();
        auto stopRequested = [&arguments]
        {
            std::error_code ignored;
            return !arguments.m_StopFile.empty() && std::filesystem::exists(arguments.m_StopFile, ignored);
        };
        while (!g_Interrupted && !client.IsTerminated() && eCore::Clock::now() < deadline && !stopRequested())
        {
            client.Poll(50);
        }

        // detach the role first so its final retained heartbeat goes out before the session closes
        stopping = true;
        detach();
        client.Poll(0);
        client.Poll(0);
        if (client.Connected())
        {
            client.Disconnect();
        }
        client.Shutdown();
        if (!failure.empty())
        {
            throw std::runtime_error(failure);
        }
        return EXIT_SUCCESS;
    }
    catch (const std::exception &error)
    {
        std::cerr << "EmpyWorkload: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
