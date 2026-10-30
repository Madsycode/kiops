#pragma once
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <istream>
#include <ostream>
#include "Comas.h"
#include "../Common/Process.h"

// Model Context Protocol binding of the knowledge plane: JSON-RPC 2.0 over stdio, one message
// per line (MCP stdio transport), lines bounded to 1 MiB.
// Server methods: initialize → {protocolVersion: "2025-06-18" (or the client's when supported),
// capabilities: {tools: {}}, serverInfo: {name: "empy-comas", version}}, notifications/initialized,
// ping, tools/list, tools/call. Tools: knowledge_query (arguments = Query JSON) → entries,
// knowledge_retrieve ({intent, agent, limit}) → RetrievalResult, knowledge_put ({collection, id,
// data, tags}; collections memory|observations|operator|tools only), knowledge_collections.
// Tool results: {content: [{type: "text", text: <compact JSON>}], structuredContent: <object>,
// isError}. Errors: -32700 parse, -32600 invalid request, -32601 unknown method, -32602 invalid
// params; tool failures are isError results.

namespace eOps::Context
{
    inline constexpr const char *MCP_PROTOCOL_VERSION = "2025-06-18";

    class EMPY_OPS_API McpServer
    {
    public:
        explicit McpServer(Comas &comas);
        Json Handle(const Json &message);                 // response, or null for notifications
        int32_t Run(std::istream &in, std::ostream &out); // until EOF; returns 0 (1 on stream failure)

    private:
        Comas &m_Comas;
        bool m_Initialized = false;
    };

    // not thread-safe: one owner calls Tools/CallTool
    class EMPY_OPS_API McpClient
    {
    public:
        McpClient();
        ~McpClient(); // Stop
        McpClient(const McpClient &) = delete;
        McpClient &operator=(const McpClient &) = delete;

        // launches the server with piped stdio (stdin/stdout forced on), then initialize +
        // notifications/initialized; false with `error` on launch/handshake failure
        bool Start(const ProcessOptions &server, std::string *error = nullptr, uint32_t timeoutMs = 10000);
        std::vector<std::string> Tools(uint32_t timeoutMs = 10000); // throws std::runtime_error
        // the tool result object; throws std::runtime_error on transport errors, JSON-RPC errors or timeout
        Json CallTool(const std::string &name, const Json &arguments, uint32_t timeoutMs = 10000);
        void Stop();
        bool Running();
        const Json &ServerInfo() const noexcept; // initialize result

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    // knowledge source backed by an MCP server's knowledge_query tool
    class EMPY_OPS_API McpSource : public KnowledgeSource
    {
    public:
        explicit McpSource(std::shared_ptr<McpClient> client, uint32_t timeoutMs = 10000);
        std::vector<Entry> Execute(const Query &query) override; // serialised; failures → empty + EMPY_WARN

    private:
        std::shared_ptr<McpClient> m_Client;
        uint32_t m_TimeoutMs = 10000;
        std::mutex m_Mutex;
    };
} // namespace eOps::Context
