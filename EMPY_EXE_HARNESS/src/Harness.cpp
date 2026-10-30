#include "Cli.h"
#include <charconv>
#include <cstdlib>
#include <cstring>

// EmpyHarness entry point: argument parsing and command dispatch (commands in Commands.cpp and Offline.cpp)

namespace
{
    void Usage(std::ostream &out)
    {
        out << "EmpyHarness <command> [options]   (" << eOps::EmpyOpsTag() << ")\n"
            << "  run [--config FILE] [--host H] [--port P] [--seconds N] [--roles a,b] [--status-port P] [--verbose]\n"
            << "      [--stop-file PATH]   (stops gracefully once PATH exists)\n"
            << "  intent \"TEXT\" [--dry-run] [--timeout-ms N] [--operator ID] [--no-wait]\n"
            << "  deploy FILE.yaml [--dry-run] [--timeout-ms N] [--no-wait]\n"
            << "  status [--timeout-ms N]\n"
            << "  approve [--deny] [--count N] [--seconds N] [--operator ID] [--note TEXT]\n"
            << "  validate FILE.yaml\n"
            << "  interpret \"TEXT\"\n"
            << "  render FILE.yaml --out DIR [--node N] [--namespace NS] [--image IMAGE]\n"
            << "  query \"TEXT\" [--limit N]\n"
            << "  mcp-server\n"
            << "  common options: --config FILE --host H --port P --root R --verbose --help\n"
            << "  without --config: $EMPY_MLOPS_CONFIG, else Configs/MLOps/Harness.yaml above the working or\n"
            << "  executable directory\n";
    }

    uint32_t Number(std::string_view option, std::string_view text, uint32_t maximum)
    {
        uint32_t value = 0;
        auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value > maximum)
        {
            throw eHarness::UsageError(std::string(option) + ": invalid number '" + std::string(text) + "'");
        }
        return value;
    }

    eHarness::Options Parse(int32_t argc, char **argv)
    {
        eHarness::Options options;
        options.m_Command = argv[1];
        for (int32_t i = 2; i < argc; ++i)
        {
            std::string_view argument(argv[i]);
            if (!argument.starts_with("--"))
            {
                options.m_Positional.emplace_back(argument);
                continue;
            }
            if (argument == "--dry-run")
            {
                options.m_DryRun = true;
                continue;
            }
            if (argument == "--deny")
            {
                options.m_Deny = true;
                continue;
            }
            if (argument == "--no-wait")
            {
                options.m_NoWait = true;
                continue;
            }
            if (argument == "--verbose")
            {
                options.m_Verbose = true;
                continue;
            }
            if (i + 1 >= argc)
            {
                throw eHarness::UsageError(std::string(argument) + ": missing value");
            }
            std::string value = argv[++i];
            if (argument == "--config")
            {
                options.m_Config = value;
            }
            else if (argument == "--host")
            {
                options.m_Host = value;
            }
            else if (argument == "--port")
            {
                options.m_Port = static_cast<uint16_t>(Number(argument, value, UINT16_MAX));
            }
            else if (argument == "--root")
            {
                options.m_Root = value;
            }
            else if (argument == "--seconds")
            {
                options.m_Seconds = Number(argument, value, 7 * 86400);
            }
            else if (argument == "--roles")
            {
                options.m_Roles = eOps::Split(value, ',');
            }
            else if (argument == "--status-port")
            {
                options.m_StatusPort = static_cast<uint16_t>(Number(argument, value, UINT16_MAX));
            }
            else if (argument == "--timeout-ms")
            {
                options.m_TimeoutMs = Number(argument, value, 86400000);
            }
            else if (argument == "--count")
            {
                options.m_Count = Number(argument, value, 1000000);
            }
            else if (argument == "--limit")
            {
                options.m_Limit = Number(argument, value, 1000);
            }
            else if (argument == "--out")
            {
                options.m_Out = value;
            }
            else if (argument == "--stop-file")
            {
                options.m_StopFile = value;
            }
            else if (argument == "--node")
            {
                options.m_Node = value;
            }
            else if (argument == "--namespace")
            {
                options.m_Namespace = value;
            }
            else if (argument == "--image")
            {
                options.m_Image = value;
            }
            else if (argument == "--operator")
            {
                options.m_Operator = value;
            }
            else if (argument == "--note")
            {
                options.m_Note = value;
            }
            else
            {
                throw eHarness::UsageError("unknown option " + std::string(argument));
            }
        }
        return options;
    }

    // commands taking exactly `count` positional arguments
    void Expect(const eHarness::Options &options, size_t count, const char *what)
    {
        if (options.m_Positional.size() != count)
        {
            throw eHarness::UsageError(options.m_Command + " expects " + what);
        }
    }
} // namespace

int32_t main(int32_t argc, char **argv)
{
    if (argc < 2)
    {
        Usage(std::cerr);
        return 2;
    }
    std::string_view command(argv[1]);
    if (command == "--help" || command == "help" || command == "-h")
    {
        Usage(std::cout);
        return EXIT_SUCCESS;
    }
    for (int32_t i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0)
        {
            Usage(std::cout);
            return EXIT_SUCCESS;
        }
    }
    // every command but run reports its result on stdout and its log on stderr
    std::optional<eHarness::StdoutRoute> route;
    if (command != "run")
    {
        route.emplace();
    }
    try
    {
        eHarness::Options options = Parse(argc, argv);
        if (command == "run")
        {
            Expect(options, 0, "no positional argument");
            return eHarness::Run(options);
        }
        if (command == "intent")
        {
            Expect(options, 1, "the intent text");
            return eHarness::Intent(options);
        }
        if (command == "deploy")
        {
            Expect(options, 1, "a descriptor file");
            return eHarness::Deploy(options);
        }
        if (command == "status")
        {
            Expect(options, 0, "no positional argument");
            return eHarness::Status(options);
        }
        if (command == "approve")
        {
            Expect(options, 0, "no positional argument");
            return eHarness::Approve(options);
        }
        if (command == "validate")
        {
            Expect(options, 1, "a descriptor file");
            return eHarness::Validate(options);
        }
        if (command == "interpret")
        {
            Expect(options, 1, "the intent text");
            return eHarness::Interpret(options);
        }
        if (command == "render")
        {
            Expect(options, 1, "a descriptor file");
            if (options.m_Out.empty())
            {
                throw eHarness::UsageError("render needs --out DIR");
            }
            return eHarness::Render(options);
        }
        if (command == "query")
        {
            Expect(options, 1, "the query text");
            return eHarness::Query(options);
        }
        if (command == "mcp-server")
        {
            Expect(options, 0, "no positional argument");
            return eHarness::McpServe(options);
        }
        throw eHarness::UsageError("unknown command '" + std::string(command) + "'");
    }
    catch (const eHarness::UsageError &error)
    {
        std::cerr << "EmpyHarness: " << error.what() << "\n";
        Usage(std::cerr);
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "EmpyHarness: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
