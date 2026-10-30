#include <eOps/Intelligence/Command.h>
#include <eOps/Intelligence/Structured.h>
#include <eOps/Common/Process.h>
#include <eOps/Common/Text.h>
#include <algorithm>
#include <chrono>
#include <filesystem>

namespace eOps::Intelligence
{
    namespace
    {
        using Clock = std::chrono::steady_clock;
        constexpr std::chrono::milliseconds READ_SLICE{100};
    } // namespace

    CommandProvider::CommandProvider(CommandOptions options) : m_Options(std::move(options))
    {
    }

    std::string CommandProvider::Name() const
    {
        if (!m_Options.m_Name.empty())
        {
            return m_Options.m_Name;
        }
        if (m_Options.m_Command.empty())
        {
            return "command";
        }
        return "command:" + std::filesystem::path(m_Options.m_Command.front()).stem().string();
    }

    bool CommandProvider::Available() const
    {
        return !m_Options.m_Command.empty() && !m_Options.m_Command.front().empty();
    }

    Completion CommandProvider::Complete(const Prompt &prompt)
    {
        Completion completion;
        completion.m_Provider = Name();
        completion.m_Model = completion.m_Provider;
        auto fail = [&](std::string message)
        {
            completion.m_Error = std::move(message);
            return completion;
        };
        if (!Available())
        {
            return fail("unavailable: no command configured");
        }

        // one Process per completion keeps concurrent callers independent (Process has a single owner)
        ProcessOptions options;
        options.m_Program = m_Options.m_Command.front();
        options.m_Args.assign(m_Options.m_Command.begin() + 1, m_Options.m_Command.end());
        options.m_PipeStdin = true;
        options.m_PipeStdout = true;
        Process child;
        std::string error;
        if (!child.Start(options, &error))
        {
            return fail("cannot launch the command: " + error);
        }
        Json request = Json::MakeObject({{"capability", CapabilityName(prompt.m_Capability)},
                                         {"system", prompt.m_System},
                                         {"user", prompt.m_User},
                                         {"schema", prompt.m_Schema ? *prompt.m_Schema : Json()},
                                         {"maxTokens", prompt.m_MaxTokens}});
        // the request is fed within the provider timeout: a command that stops reading its input is killed. A
        // command that ignores stdin may exit before reading it (broken pipe); its output still counts
        auto deadline = Clock::now() + std::chrono::milliseconds(m_Options.m_TimeoutMs);
        if (!child.Write(request.Dump() + "\n", std::chrono::milliseconds(m_Options.m_TimeoutMs)) &&
            Clock::now() >= deadline)
        {
            child.Kill();
            return fail("timed out after " + std::to_string(m_Options.m_TimeoutMs) +
                        " ms while writing the request (the command does not read its input)");
        }
        child.CloseStdin();

        std::string output;
        bool exited = false;
        while (true)
        {
            auto now = Clock::now();
            if (now >= deadline)
            {
                child.Kill();
                return fail("timed out after " + std::to_string(m_Options.m_TimeoutMs) + " ms");
            }
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            std::optional<std::string> line = child.ReadLine(std::min(remaining, READ_SLICE));
            if (line)
            {
                if (output.size() + line->size() + 1 > m_Options.m_MaxOutputBytes)
                {
                    child.Kill();
                    return fail("output exceeds " + std::to_string(m_Options.m_MaxOutputBytes) + " bytes");
                }
                output += *line;
                output.push_back('\n');
                continue;
            }
            // nothing to read: once the child has exited, one more empty read means the pipe is drained
            if (exited)
            {
                break;
            }
            exited = !child.Running();
        }
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
        std::optional<int32_t> status = child.Wait(std::max(remaining, std::chrono::milliseconds(1)));
        if (!status)
        {
            child.Kill();
            return fail("timed out after " + std::to_string(m_Options.m_TimeoutMs) + " ms");
        }
        if (*status != 0)
        {
            return fail("the command exited with status " + std::to_string(*status));
        }

        // {text, json?} object, or raw text
        std::string text = Trim(output);
        std::optional<Json> document;
        try
        {
            document = Json::Parse(text);
        }
        catch (const JsonError &)
        {
        }
        if (document && document->IsObject() && (document->Has("text") || document->Has("json")))
        {
            const Json *body = document->Find("text");
            if (body && !body->IsNull() && !body->IsString())
            {
                return fail("the command output field 'text' is not a string");
            }
            completion.m_Text = body && body->IsString() ? body->AsString() : std::string();
            if (const Json *json = document->Find("json"); json && !json->IsNull())
            {
                completion.m_Json = *json;
            }
        }
        else if (document && document->IsObject() && document->Has("error"))
        {
            return fail("the command reported: " + document->GetString("error", document->Find("error")->Dump()));
        }
        else
        {
            completion.m_Text = text;
        }
        if (prompt.m_Schema && !completion.m_Json)
        {
            completion.m_Json = ExtractJson(completion.m_Text);
            if (!completion.m_Json)
            {
                return fail("structured output is not valid JSON");
            }
        }
        if (!prompt.m_Schema && completion.m_Text.empty())
        {
            return fail("the command produced no output");
        }
        completion.m_StopReason = "end_turn";
        completion.m_Ok = true;
        return completion;
    }
} // namespace eOps::Intelligence
