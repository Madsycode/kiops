#include <eOps/Context/Mcp.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include <array>
#include <chrono>
#include <utility>
#include <algorithm>
#include <stdexcept>

namespace eOps::Context
{
    namespace
    {
        constexpr size_t MAX_LINE = 1 << 20;
        constexpr size_t MAX_REPLY = 16 * MAX_LINE; // client-side bound of one response line
        constexpr size_t MAX_TOOL_PAGES = 16;
        constexpr uint32_t SEND_TIMEOUT_MS = 10000; // notifications and answers to server requests
        constexpr int32_t PARSE_ERROR = -32700;
        constexpr int32_t INVALID_REQUEST = -32600;
        constexpr int32_t METHOD_NOT_FOUND = -32601;
        constexpr int32_t INVALID_PARAMS = -32602;
        constexpr int32_t INTERNAL_ERROR = -32603;
        constexpr std::array<std::string_view, 3> VERSIONS = {"2025-06-18", "2025-03-26", "2024-11-05"};
        constexpr std::array<std::string_view, 4> WRITABLE = {"memory", "observations", "operator", "tools"};

        // JSON-RPC failure carried to the response builder
        struct RpcError : std::runtime_error
        {
            int32_t m_Code;
            RpcError(int32_t code, const std::string &message) : std::runtime_error(message), m_Code(code)
            {
            }
        };

        // tool argument rejected: reported as an isError tool result
        struct ToolError : std::runtime_error
        {
            using std::runtime_error::runtime_error;
        };

        Json ErrorResponse(const Json &id, int32_t code, const std::string &message)
        {
            return Json(Json::Object{
                {"jsonrpc", "2.0"}, {"id", id}, {"error", Json(Json::Object{{"code", code}, {"message", message}})}});
        }

        Json ResultResponse(const Json &id, Json result)
        {
            return Json(Json::Object{{"jsonrpc", "2.0"}, {"id", id}, {"result", std::move(result)}});
        }

        Json ToolResult(Json structured, bool isError)
        {
            Json text = Json(Json::Object{{"type", "text"}, {"text", structured.Dump()}});
            return Json(Json::Object{{"content", Json::MakeArray({std::move(text)})},
                                     {"structuredContent", std::move(structured)},
                                     {"isError", isError}});
        }

        Json Type(std::string_view type)
        {
            return Json::MakeObject({{"type", std::string(type)}});
        }

        Json StringList()
        {
            return Json::MakeObject({{"type", "array"}, {"items", Type("string")}});
        }

        Json Tool(std::string_view name, std::string_view title, std::string_view description, Json properties,
                  std::vector<std::string> required)
        {
            Json schema = Json::MakeObject({{"type", "object"}, {"properties", std::move(properties)}});
            if (!required.empty())
            {
                Json list = Json::MakeArray();
                for (std::string &member : required)
                {
                    list.Push(Json(std::move(member)));
                }
                schema.Set("required", std::move(list));
            }
            schema.Set("additionalProperties", false);
            return Json(Json::Object{{"name", std::string(name)},
                                     {"title", std::string(title)},
                                     {"description", std::string(description)},
                                     {"inputSchema", std::move(schema)}});
        }

        Json ToolList()
        {
            Json ops = Json::MakeArray();
            for (Op op :
                 {Op::Eq, Op::Ne, Op::Lt, Op::Le, Op::Gt, Op::Ge, Op::Contains, Op::In, Op::Exists, Op::Matches})
            {
                ops.Push(Json(OpName(op)));
            }
            Json condition = Json::MakeObject(
                {{"type", "object"},
                 {"properties", Json::MakeObject({{"field", Type("string")},
                                                  {"op", Json::MakeObject({{"type", "string"}, {"enum", ops}})},
                                                  {"value", Json::MakeObject()}})},
                 {"required", Json::MakeArray({Json("field"), Json("op")})},
                 {"additionalProperties", false}});
            Json writable = Json::MakeArray();
            for (std::string_view collection : WRITABLE)
            {
                writable.Push(Json(std::string(collection)));
            }
            Json tools = Json::MakeArray();
            tools.Push(Tool("knowledge_query", "Query the knowledge plane",
                            "Executes a structured CoMaS query (collections, where conditions, text keywords, "
                            "orderBy, descending, limit, select) and returns the matching entries.",
                            Json::MakeObject({{"collections", StringList()},
                                              {"where", Json::MakeObject({{"type", "array"}, {"items", condition}})},
                                              {"text", StringList()},
                                              {"orderBy", Type("string")},
                                              {"descending", Type("boolean")},
                                              {"limit", Type("integer")},
                                              {"select", StringList()}}),
                            {}));
            tools.Push(Tool(
                "knowledge_retrieve", "Retrieve context for an intent",
                "Runs the CoMaS context retrieval flow (augment, translate, execute, contextualise) "
                "for a natural-language intent and returns the structured result.",
                Json::MakeObject({{"intent", Type("string")}, {"agent", Type("string")}, {"limit", Type("integer")}}),
                {"intent"}));
            tools.Push(
                Tool("knowledge_put", "Store a knowledge entry",
                     "Inserts or replaces an entry in the memory, observations, operator or tools "
                     "collection.",
                     Json::MakeObject({{"collection", Json::MakeObject({{"type", "string"}, {"enum", writable}})},
                                       {"id", Type("string")},
                                       {"data", Json::MakeObject()},
                                       {"tags", StringList()}}),
                     {"collection", "data"}));
            tools.Push(Tool("knowledge_collections", "List knowledge collections",
                            "Lists the non-empty knowledge collections with their entry counts.", Json::MakeObject(),
                            {}));
            return tools;
        }

        Json EntryList(const std::vector<Entry> &entries)
        {
            Json list = Json::MakeArray();
            for (const Entry &entry : entries)
            {
                list.Push(entry.ToJson());
            }
            return list;
        }

        std::vector<std::string> Tags(const Json &value)
        {
            std::vector<std::string> tags;
            if (value.IsNull())
            {
                return tags;
            }
            if (!value.IsArray())
            {
                throw ToolError("'tags' must be a list of strings");
            }
            for (const Json &tag : value.AsArray())
            {
                if (!tag.IsString())
                {
                    throw ToolError("'tags' must be a list of strings");
                }
                tags.push_back(tag.AsString());
            }
            return tags;
        }

        void Allow(const Json &arguments, std::initializer_list<std::string_view> members)
        {
            for (const auto &[key, value] : arguments.AsObject())
            {
                if (std::find(members.begin(), members.end(), key) == members.end())
                {
                    throw ToolError("unknown argument '" + key + "'");
                }
            }
        }
    } // namespace

    McpServer::McpServer(Comas &comas) : m_Comas(comas)
    {
    }

    Json McpServer::Handle(const Json &message)
    {
        if (!message.IsObject())
        {
            return ErrorResponse(Json(), INVALID_REQUEST, "Invalid Request: expected a JSON-RPC 2.0 object");
        }
        const Json *id = message.Find("id");
        bool notification = id == nullptr;
        if (id && !id->IsString() && !id->IsInteger())
        {
            return ErrorResponse(Json(), INVALID_REQUEST, "Invalid Request: id must be a string or an integer");
        }
        Json requestId = id ? *id : Json();
        const Json *version = message.Find("jsonrpc");
        const Json *method = message.Find("method");
        if (!method && (message.Has("result") || message.Has("error")))
        {
            return Json(); // a response: this server issues no requests
        }
        if (!version || !version->IsString() || version->AsString() != "2.0" || !method || !method->IsString())
        {
            return notification ? Json()
                                : ErrorResponse(requestId, INVALID_REQUEST,
                                                "Invalid Request: needs jsonrpc \"2.0\" and a string method");
        }
        const std::string &name = method->AsString();
        if (notification)
        {
            if (name == "notifications/initialized")
            {
                m_Initialized = true;
            }
            return Json(); // notifications/cancelled and unknown notifications are ignored
        }
        const Json &params = message["params"];
        if (!params.IsNull() && !params.IsObject())
        {
            return ErrorResponse(requestId, INVALID_PARAMS, "Invalid params: params must be an object");
        }
        try
        {
            if (name == "initialize")
            {
                const Json &requested = params["protocolVersion"];
                if (!requested.IsString())
                {
                    throw RpcError(INVALID_PARAMS, "Invalid params: initialize needs a string protocolVersion");
                }
                std::string chosen = std::find(VERSIONS.begin(), VERSIONS.end(), requested.AsString()) != VERSIONS.end()
                                         ? requested.AsString()
                                         : std::string(MCP_PROTOCOL_VERSION);
                m_Initialized = true;
                return ResultResponse(
                    requestId,
                    Json(Json::Object{{"protocolVersion", chosen},
                                      {"capabilities", Json::MakeObject({{"tools", Json::MakeObject()}})},
                                      {"serverInfo", Json(Json::Object{{"name", "empy-comas"}, {"version", "1.0.0"}})},
                                      {"instructions",
                                       "CoMaS knowledge plane of the EMPY Agentic-MLOps harness: query, retrieve "
                                       "and record operational knowledge."}}));
            }
            if (name == "ping")
            {
                return ResultResponse(requestId, Json::MakeObject());
            }
            if (name != "tools/list" && name != "tools/call")
            {
                throw RpcError(METHOD_NOT_FOUND, "Method not found: " + name);
            }
            if (!m_Initialized)
            {
                throw RpcError(INVALID_REQUEST, "Invalid Request: the session is not initialized");
            }
            if (name == "tools/list")
            {
                if (const Json *cursor = params.Find("cursor"); cursor && !cursor->IsNull() && !cursor->IsString())
                {
                    throw RpcError(INVALID_PARAMS, "Invalid params: cursor must be a string");
                }
                return ResultResponse(requestId, Json(Json::Object{{"tools", ToolList()}}));
            }
            const Json &tool = params["name"];
            if (!tool.IsString())
            {
                throw RpcError(INVALID_PARAMS, "Invalid params: tools/call needs a string name");
            }
            const Json arguments = params["arguments"].IsNull() ? Json::MakeObject() : params["arguments"];
            if (!arguments.IsObject())
            {
                throw RpcError(INVALID_PARAMS, "Invalid params: arguments must be an object");
            }
            const std::string &toolName = tool.AsString();
            if (toolName != "knowledge_query" && toolName != "knowledge_retrieve" && toolName != "knowledge_put" &&
                toolName != "knowledge_collections")
            {
                throw RpcError(INVALID_PARAMS, "Unknown tool: " + toolName);
            }
            try
            {
                if (toolName == "knowledge_query")
                {
                    Query query;
                    try
                    {
                        query = Query::FromJson(arguments);
                    }
                    catch (const JsonError &error)
                    {
                        throw ToolError(error.what());
                    }
                    std::vector<Entry> entries = m_Comas.Execute(query);
                    return ResultResponse(requestId,
                                          ToolResult(Json(Json::Object{{"count", static_cast<uint64_t>(entries.size())},
                                                                       {"entries", EntryList(entries)}}),
                                                     false));
                }
                if (toolName == "knowledge_retrieve")
                {
                    Allow(arguments, {"intent", "agent", "limit"});
                    RetrievalRequest request;
                    const Json &intent = arguments["intent"];
                    if (!intent.IsString() || Trim(intent.AsString()).empty())
                    {
                        throw ToolError("'intent' must be a non-empty string");
                    }
                    request.m_Intent = intent.AsString();
                    request.m_Agent = "mcp";
                    if (const Json *agent = arguments.Find("agent"); agent && !agent->IsNull())
                    {
                        if (!agent->IsString())
                        {
                            throw ToolError("'agent' must be a string");
                        }
                        request.m_Agent = agent->AsString();
                    }
                    if (const Json *limit = arguments.Find("limit"); limit && !limit->IsNull())
                    {
                        if (!limit->IsInteger() || limit->AsInt() < 1 || limit->AsInt() > 1000)
                        {
                            throw ToolError("'limit' must be an integer in [1, 1000]");
                        }
                        request.m_Limit = static_cast<size_t>(limit->AsInt());
                    }
                    return ResultResponse(requestId, ToolResult(m_Comas.Retrieve(request).ToJson(), false));
                }
                if (toolName == "knowledge_put")
                {
                    Allow(arguments, {"collection", "id", "data", "tags"});
                    const Json &collection = arguments["collection"];
                    if (!collection.IsString() ||
                        std::find(WRITABLE.begin(), WRITABLE.end(), collection.AsString()) == WRITABLE.end())
                    {
                        throw ToolError("'collection' must be one of memory, observations, operator, tools");
                    }
                    const Json *data = arguments.Find("data");
                    if (!data || data->IsNull())
                    {
                        throw ToolError("'data' is required");
                    }
                    Entry entry;
                    entry.m_Collection = collection.AsString();
                    if (const Json *entryId = arguments.Find("id"); entryId && !entryId->IsNull())
                    {
                        if (!entryId->IsString() || entryId->AsString().empty())
                        {
                            throw ToolError("'id' must be a non-empty string");
                        }
                        entry.m_Id = entryId->AsString();
                    }
                    else
                    {
                        entry.m_Id = NewId(entry.m_Collection);
                    }
                    entry.m_Data = *data;
                    entry.m_Tags = Tags(arguments["tags"]);
                    try
                    {
                        m_Comas.Plane().Put(entry);
                    }
                    catch (const std::invalid_argument &error)
                    {
                        throw ToolError(error.what());
                    }
                    std::optional<Entry> stored = m_Comas.Plane().Get(entry.m_Collection, entry.m_Id);
                    return ResultResponse(requestId,
                                          ToolResult(Json(Json::Object{{"collection", entry.m_Collection},
                                                                       {"id", entry.m_Id},
                                                                       {"stored", stored.has_value()},
                                                                       {"updated", stored ? stored->m_Updated : 0}}),
                                                     false));
                }
                Allow(arguments, {});
                Json collections = Json::MakeArray();
                uint64_t total = 0;
                for (const std::string &collection : m_Comas.Plane().Collections())
                {
                    uint64_t size = m_Comas.Plane().Size(collection);
                    total += size;
                    collections.Push(Json(Json::Object{{"name", collection}, {"size", size}}));
                }
                return ResultResponse(
                    requestId,
                    ToolResult(Json(Json::Object{{"collections", std::move(collections)}, {"total", total}}), false));
            }
            catch (const ToolError &error)
            {
                return ResultResponse(requestId, ToolResult(Json(Json::Object{{"error", error.what()}}), true));
            }
            catch (const std::length_error &error)
            {
                return ResultResponse(requestId, ToolResult(Json(Json::Object{{"error", error.what()}}), true));
            }
            catch (const JsonError &error)
            {
                return ResultResponse(requestId, ToolResult(Json(Json::Object{{"error", error.what()}}), true));
            }
        }
        catch (const RpcError &error)
        {
            return ErrorResponse(requestId, error.m_Code, error.what());
        }
        catch (const std::exception &error)
        {
            return ErrorResponse(requestId, INTERNAL_ERROR, std::string("Internal error: ") + error.what());
        }
    }

    int32_t McpServer::Run(std::istream &in, std::ostream &out)
    {
        std::streambuf *buffer = in.rdbuf();
        if (!buffer)
        {
            return 1;
        }
        std::string line;
        bool overflow = false;
        auto process = [&]() -> bool
        {
            Json response;
            if (overflow)
            {
                response = ErrorResponse(Json(), PARSE_ERROR, "Parse error: message exceeds 1 MiB");
            }
            else
            {
                if (!line.empty() && line.back() == '\r')
                {
                    line.pop_back();
                }
                if (Trim(line).empty())
                {
                    line.clear();
                    return true;
                }
                try
                {
                    response = Handle(Json::Parse(line, JsonLimits{64, MAX_LINE}));
                }
                catch (const JsonError &error)
                {
                    response = ErrorResponse(Json(), PARSE_ERROR, std::string("Parse error: ") + error.what());
                }
            }
            line.clear();
            overflow = false;
            if (response.IsNull())
            {
                return true;
            }
            out << response.Dump() << '\n';
            out.flush();
            return static_cast<bool>(out);
        };
        while (true)
        {
            int32_t c = buffer->sbumpc();
            if (c == std::char_traits<char>::eof())
            {
                if ((!line.empty() || overflow) && !process())
                {
                    return 1;
                }
                return 0;
            }
            if (c == '\n')
            {
                if (!process())
                {
                    return 1;
                }
                continue;
            }
            if (line.size() < MAX_LINE)
            {
                line.push_back(static_cast<char>(c));
            }
            else
            {
                overflow = true;
            }
        }
    }

    struct McpClient::Impl
    {
        std::unique_ptr<Process> m_Process;
        int64_t m_NextId = 1;
        Json m_ServerInfo;

        // one newline-delimited message within `timeoutMs`; a server that stops reading its input is killed, since
        // a partly written line leaves the stream unusable
        bool Send(const Json &message, uint32_t timeoutMs = SEND_TIMEOUT_MS)
        {
            if (!m_Process)
            {
                return false;
            }
            if (m_Process->Write(message.Dump() + "\n", std::chrono::milliseconds(timeoutMs)))
            {
                return true;
            }
            if (m_Process->Running())
            {
                EMPY_WARN("comas: MCP server stopped reading its input; stopping it");
                m_Process->Kill();
            }
            return false;
        }

        Json Request(const std::string &method, Json params, uint32_t timeoutMs)
        {
            if (!m_Process)
            {
                throw std::runtime_error("MCP client is not started");
            }
            using Clock = std::chrono::steady_clock;
            auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
            int64_t id = m_NextId++;
            if (!Send(Json(Json::Object{
                          {"jsonrpc", "2.0"}, {"id", id}, {"method", method}, {"params", std::move(params)}}),
                      std::max<uint32_t>(timeoutMs, 1)))
            {
                throw std::runtime_error("MCP server input is closed or not read within " + std::to_string(timeoutMs) +
                                         " ms ('" + method + "')");
            }
            int32_t immediate = 0;
            while (true)
            {
                auto start = Clock::now();
                if (start >= deadline)
                {
                    throw std::runtime_error("MCP request '" + method + "' timed out after " +
                                             std::to_string(timeoutMs) + " ms");
                }
                auto wait = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - start) +
                            std::chrono::milliseconds(1);
                std::optional<std::string> line = m_Process->ReadLine(wait);
                if (!line)
                {
                    if (!m_Process->Running())
                    {
                        throw std::runtime_error("MCP server exited during '" + method + "'");
                    }
                    // ReadLine returns at once after end of output while the server keeps running; short
                    // waits near the deadline may legitimately return early and are not counted
                    if (wait >= std::chrono::milliseconds(20) && Clock::now() - start < std::chrono::milliseconds(1) &&
                        ++immediate > 2)
                    {
                        throw std::runtime_error("MCP server closed its output during '" + method + "'");
                    }
                    continue;
                }
                immediate = 0;
                Json reply;
                try
                {
                    reply = Json::Parse(*line, JsonLimits{64, MAX_REPLY});
                }
                catch (const JsonError &)
                {
                    continue; // not protocol output
                }
                if (!reply.IsObject())
                {
                    continue;
                }
                if (const Json *request = reply.Find("method"))
                {
                    // server-initiated request: answer ping, refuse anything else; notifications are skipped
                    if (const Json *requestId = reply.Find("id"))
                    {
                        Send(request->IsString() && request->AsString() == "ping"
                                 ? ResultResponse(*requestId, Json::MakeObject())
                                 : ErrorResponse(*requestId, METHOD_NOT_FOUND, "Method not supported by the client"));
                    }
                    continue;
                }
                const Json *replyId = reply.Find("id");
                if (!replyId || !(*replyId == Json(id)))
                {
                    continue; // stale reply of an abandoned request
                }
                if (const Json *error = reply.Find("error"); error && !error->IsNull())
                {
                    throw std::runtime_error("MCP error " + std::to_string((*error).GetInt("code")) + " on '" + method +
                                             "': " + error->GetString("message"));
                }
                return std::as_const(reply)["result"];
            }
        }

        void Stop()
        {
            if (!m_Process)
            {
                return;
            }
            m_Process->CloseStdin(); // the server ends at end of input
            if (!m_Process->Wait(std::chrono::milliseconds(2000)))
            {
                m_Process->Kill();
                m_Process->Wait(std::chrono::milliseconds(2000));
            }
            m_Process.reset();
            m_ServerInfo = Json();
        }
    };

    McpClient::McpClient() : m_Impl(std::make_unique<Impl>())
    {
    }

    McpClient::~McpClient()
    {
        try
        {
            m_Impl->Stop();
        }
        catch (...)
        {
        }
    }

    bool McpClient::Start(const ProcessOptions &server, std::string *error, uint32_t timeoutMs)
    {
        auto fail = [&](std::string message)
        {
            m_Impl->Stop();
            if (error)
            {
                *error = std::move(message);
            }
            return false;
        };
        m_Impl->Stop();
        ProcessOptions options = server;
        options.m_PipeStdin = true;
        options.m_PipeStdout = true;
        options.m_MaxLineBytes = std::max(options.m_MaxLineBytes, MAX_REPLY); // large tool results stay one line
        auto process = std::make_unique<Process>();
        std::string launch;
        if (!process->Start(options, &launch))
        {
            return fail("cannot launch the MCP server: " + launch);
        }
        m_Impl->m_Process = std::move(process);
        m_Impl->m_NextId = 1;
        try
        {
            const Json result = m_Impl->Request(
                "initialize",
                Json(Json::Object{
                    {"protocolVersion", MCP_PROTOCOL_VERSION},
                    {"capabilities", Json::MakeObject()},
                    {"clientInfo", Json(Json::Object{{"name", "empy-mcp-client"}, {"version", "1.0.0"}})}}),
                timeoutMs);
            const Json &version = result["protocolVersion"];
            if (!result.IsObject() || !version.IsString())
            {
                return fail("the MCP server sent an invalid initialize result");
            }
            if (std::find(VERSIONS.begin(), VERSIONS.end(), version.AsString()) == VERSIONS.end())
            {
                return fail("the MCP server speaks the unsupported protocol version " + version.AsString());
            }
            if (!m_Impl->Send(Json(Json::Object{{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}})))
            {
                return fail("the MCP server input is closed");
            }
            m_Impl->m_ServerInfo = result;
        }
        catch (const std::exception &exception)
        {
            return fail(std::string("MCP handshake failed: ") + exception.what());
        }
        return true;
    }

    std::vector<std::string> McpClient::Tools(uint32_t timeoutMs)
    {
        std::vector<std::string> names;
        std::string cursor;
        for (size_t page = 0; page < MAX_TOOL_PAGES; ++page)
        {
            Json params = Json::MakeObject();
            if (!cursor.empty())
            {
                params.Set("cursor", cursor);
            }
            const Json result = m_Impl->Request("tools/list", std::move(params), timeoutMs);
            const Json &tools = result["tools"];
            if (!tools.IsArray())
            {
                throw std::runtime_error("MCP tools/list returned no tool list");
            }
            for (const Json &tool : tools.AsArray())
            {
                if (tool["name"].IsString())
                {
                    names.push_back(tool["name"].AsString());
                }
            }
            const Json &next = result["nextCursor"];
            if (!next.IsString() || next.AsString().empty())
            {
                break;
            }
            cursor = next.AsString();
        }
        return names;
    }

    Json McpClient::CallTool(const std::string &name, const Json &arguments, uint32_t timeoutMs)
    {
        Json result = m_Impl->Request(
            "tools/call",
            Json(Json::Object{{"name", name}, {"arguments", arguments.IsNull() ? Json::MakeObject() : arguments}}),
            timeoutMs);
        if (!result.IsObject())
        {
            throw std::runtime_error("MCP tools/call '" + name + "' returned no result object");
        }
        return result;
    }

    void McpClient::Stop()
    {
        m_Impl->Stop();
    }

    bool McpClient::Running()
    {
        return m_Impl->m_Process && m_Impl->m_Process->Running();
    }

    const Json &McpClient::ServerInfo() const noexcept
    {
        return m_Impl->m_ServerInfo;
    }

    McpSource::McpSource(std::shared_ptr<McpClient> client, uint32_t timeoutMs)
        : m_Client(std::move(client)), m_TimeoutMs(timeoutMs)
    {
    }

    std::vector<Entry> McpSource::Execute(const Query &query)
    {
        std::scoped_lock lock(m_Mutex);
        std::vector<Entry> entries;
        if (!m_Client)
        {
            EMPY_WARN("comas: MCP knowledge source has no client");
            return entries;
        }
        try
        {
            const Json result = m_Client->CallTool("knowledge_query", query.ToJson(), m_TimeoutMs);
            Json content = result["structuredContent"];
            if (!content.IsObject())
            {
                // servers without structured output carry the JSON in the first text block
                const Json &text = result["content"][size_t(0)]["text"];
                content = text.IsString() ? Json::Parse(text.AsString()) : Json();
            }
            if (result.GetBool("isError"))
            {
                EMPY_WARN("comas: MCP knowledge_query failed: {}", content.GetString("error", "tool error"));
                return entries;
            }
            const Json &list = content["entries"];
            if (!list.IsArray())
            {
                EMPY_WARN("comas: MCP knowledge_query returned no entry list");
                return entries;
            }
            for (const Json &item : list.AsArray())
            {
                try
                {
                    entries.push_back(Entry::FromJson(item));
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("comas: MCP knowledge_query returned an invalid entry: {}", error.what());
                }
            }
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("comas: MCP knowledge source failed: {}", error.what());
            entries.clear();
        }
        return entries;
    }
} // namespace eOps::Context
