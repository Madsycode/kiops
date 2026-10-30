#pragma once
#include <eOps/MLOps.h>
#include <eMQ/Client.h>
#include <csignal>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <iostream>
#include <stdexcept>
#include <filesystem>

// EmpyHarness command-line plumbing shared by the command implementations

namespace eHarness
{
    // a command-line mistake (exit status 2)
    struct UsageError : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct Options
    {
        std::string m_Command;
        std::vector<std::string> m_Positional;
        std::string m_Config; // --config ("" = discovered)
        std::optional<std::string> m_Host;
        std::optional<uint16_t> m_Port;
        std::optional<std::string> m_Root;
        std::optional<uint32_t> m_Seconds;
        std::optional<std::vector<std::string>> m_Roles;
        std::optional<uint16_t> m_StatusPort;
        std::optional<uint32_t> m_TimeoutMs;
        std::optional<uint32_t> m_Count;
        std::optional<uint32_t> m_Limit;
        std::string m_Out;
        std::string m_StopFile; // run: stop gracefully once this file exists
        std::string m_Node;
        std::string m_Namespace;
        std::string m_Image;
        std::string m_Operator;
        std::string m_Note;
        bool m_DryRun = false;
        bool m_Deny = false;
        bool m_NoWait = false;
        bool m_Verbose = false;
    };

    // set by SIGINT/SIGTERM; the owner loop performs the cleanup
    inline volatile std::sig_atomic_t g_Interrupted = 0;
    void InstallSignals();

    // --config, else $EMPY_MLOPS_CONFIG, else Configs/MLOps/Harness.yaml above the working or executable directory;
    // a configuration without a file (all defaults) when none is found
    eOps::Harness::HarnessConfig LoadConfig(const Options &options);
    // the configuration with the command-line overrides (--host, --port, --root, --roles, --status-port) applied
    eOps::Harness::HarnessConfig Effective(const Options &options);
    // configuration for offline commands: no external MCP server, no substrate side effects and, unless
    // `durableKnowledge`, no durable CoMaS knowledge (a running harness holds its store lock)
    eOps::Harness::HarnessConfig Offline(eOps::Harness::HarnessConfig config, bool durableKnowledge = false);

    // eMQ client options (client_config when set) with a response timeout of at least `responseMs`
    eMQ::Configs ClientOptions(const eOps::Harness::HarnessConfig &config, uint32_t responseMs);

    // result stream of a command: for every command but run, eCore logging (std::cout) is routed to stderr
    // while a StdoutRoute exists, so stdout carries only the result (JSON, YAML or the MCP protocol)
    std::ostream &Out();
    struct StdoutRoute
    {
        StdoutRoute();
        ~StdoutRoute();
        StdoutRoute(const StdoutRoute &) = delete;
        StdoutRoute &operator=(const StdoutRoute &) = delete;
    };

    void Print(const eOps::Json &json); // pretty JSON on Out()

    int32_t Run(const Options &options);       // run
    int32_t Intent(const Options &options);    // intent
    int32_t Deploy(const Options &options);    // deploy
    int32_t Status(const Options &options);    // status
    int32_t Approve(const Options &options);   // approve
    int32_t Validate(const Options &options);  // validate
    int32_t Interpret(const Options &options); // interpret
    int32_t Render(const Options &options);    // render
    int32_t Query(const Options &options);     // query
    int32_t McpServe(const Options &options);  // mcp-server
} // namespace eHarness
