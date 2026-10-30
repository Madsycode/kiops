#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "Provider.h"

// local model wrapper: per completion launches m_Command (eOps::Process, no shell), writes one
// JSON line {capability, system, user, schema, maxTokens} to stdin, closes stdin and reads stdout
// until exit or timeout. Stdout is either a JSON object {text, json?} or raw text. A non-zero exit
// status or a timeout (the child is killed) fails the completion.

namespace eOps::Intelligence
{
    struct CommandOptions
    {
        std::vector<std::string> m_Command; // argv; [0] is the program
        uint32_t m_TimeoutMs = 60000;
        std::string m_Name; // reported provider name; "" → "command:<program stem>"
        size_t m_MaxOutputBytes = 4 * 1024 * 1024;
    };

    class EMPY_OPS_API CommandProvider : public Provider
    {
    public:
        explicit CommandProvider(CommandOptions options);

        std::string Name() const override;
        bool Available() const override; // command non-empty
        Completion Complete(const Prompt &prompt) override;

    private:
        CommandOptions m_Options;
    };
} // namespace eOps::Intelligence
