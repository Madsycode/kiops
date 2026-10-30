#include <eOps/Common/Text.h>
#include <eCore/Common/Crypto.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <charconv>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#endif

namespace eOps
{
    namespace
    {
        bool IsDigit(char c) noexcept
        {
            return c >= '0' && c <= '9';
        }

        bool IsAlpha(char c) noexcept
        {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        }

        bool IsAlnum(char c) noexcept
        {
            return IsDigit(c) || IsAlpha(c);
        }

        bool IsSpace(char c) noexcept
        {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        }

        char ToLower(char c) noexcept
        {
            return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
        }

        // the UTF-8 encoding of U+00B0 DEGREE SIGN
        bool IsDegree(std::string_view text, size_t index) noexcept
        {
            return index + 1 < text.size() && static_cast<uint8_t>(text[index]) == 0xC2 &&
                   static_cast<uint8_t>(text[index + 1]) == 0xB0;
        }
    } // namespace

    std::string Lower(std::string_view text)
    {
        std::string out(text);
        for (char &c : out)
        {
            c = ToLower(c);
        }
        return out;
    }

    std::string Trim(std::string_view text)
    {
        size_t begin = 0;
        size_t end = text.size();
        while (begin < end && IsSpace(text[begin]))
        {
            ++begin;
        }
        while (end > begin && IsSpace(text[end - 1]))
        {
            --end;
        }
        return std::string(text.substr(begin, end - begin));
    }

    std::vector<std::string> Split(std::string_view text, char separator)
    {
        std::vector<std::string> parts;
        while (true)
        {
            size_t end = text.find(separator);
            parts.emplace_back(text.substr(0, end));
            if (end == std::string_view::npos)
            {
                return parts;
            }
            text.remove_prefix(end + 1);
        }
    }

    std::string Join(const std::vector<std::string> &parts, std::string_view separator)
    {
        std::string out;
        for (size_t i = 0; i < parts.size(); ++i)
        {
            if (i > 0)
            {
                out.append(separator);
            }
            out += parts[i];
        }
        return out;
    }

    std::string Singular(std::string_view text)
    {
        std::string word(text);
        size_t size = word.size();
        auto ends = [&word](std::string_view suffix) { return word.ends_with(suffix); };
        // accelerator acronyms (gpus, cpus, npus, tpus) are plurals despite the "us" ending
        bool acronym = size == 4 && word[1] == 'p' && word[2] == 'u' && word[3] == 's';
        if (size <= 3 || ends("ss") || (ends("us") && !acronym) || ends("is"))
        {
            return word;
        }
        if (ends("ies") && size > 4)
        {
            word.resize(size - 3);
            word.push_back('y');
        }
        else if (ends("sses") || ends("xes") || ends("zes") || ends("ches") || ends("shes"))
        {
            word.resize(size - 2);
        }
        else if (word.back() == 's')
        {
            word.pop_back();
        }
        return word;
    }

    std::vector<std::string> Tokenize(std::string_view text)
    {
        std::vector<std::string> tokens;
        size_t i = 0;
        while (i < text.size())
        {
            char c = text[i];
            if (c == '%')
            {
                tokens.emplace_back("percent");
                ++i;
                continue;
            }
            if (IsDegree(text, i))
            {
                i += 2;
                char unit = i < text.size() ? ToLower(text[i]) : '\0';
                bool alone = i + 1 >= text.size() || !IsAlnum(text[i + 1]);
                if (unit == 'c' && alone)
                {
                    tokens.emplace_back("celsius");
                    ++i;
                }
                else if (unit == 'f' && alone)
                {
                    tokens.emplace_back("fahrenheit");
                    ++i;
                }
                else
                {
                    tokens.emplace_back("degree");
                }
                continue;
            }
            bool negative =
                c == '-' && i + 1 < text.size() && IsDigit(text[i + 1]) && (i == 0 || !IsAlnum(text[i - 1]));
            if (IsDigit(c) || negative)
            {
                size_t start = i;
                if (negative)
                {
                    ++i;
                }
                while (i < text.size() && IsDigit(text[i]))
                {
                    ++i;
                }
                if (i + 1 < text.size() && text[i] == '.' && IsDigit(text[i + 1]))
                {
                    ++i;
                    while (i < text.size() && IsDigit(text[i]))
                    {
                        ++i;
                    }
                }
                tokens.emplace_back(text.substr(start, i - start));
                continue;
            }
            if (IsAlpha(c) || static_cast<uint8_t>(c) >= 0x80)
            {
                std::string word;
                while (i < text.size() && (IsAlnum(text[i]) || static_cast<uint8_t>(text[i]) >= 0x80) &&
                       !IsDegree(text, i))
                {
                    word.push_back(ToLower(text[i]));
                    ++i;
                }
                if (!word.empty())
                {
                    tokens.push_back(std::move(word));
                    continue;
                }
            }
            ++i;
        }
        return tokens;
    }

    bool IsDnsLabel(std::string_view text)
    {
        if (text.empty() || text.size() > 63)
        {
            return false;
        }
        for (char c : text)
        {
            if (!((c >= 'a' && c <= 'z') || IsDigit(c) || c == '-'))
            {
                return false;
            }
        }
        return text.front() != '-' && text.back() != '-';
    }

    uint64_t NowMillis()
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
    }

    std::string IsoTime(uint64_t unixMillis)
    {
        using namespace std::chrono;
        sys_time<milliseconds> time{milliseconds(static_cast<int64_t>(unixMillis))};
        auto day = floor<days>(time);
        year_month_day date(day);
        hh_mm_ss clock(time - day);
        return std::format("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", static_cast<int32_t>(date.year()),
                           static_cast<uint32_t>(date.month()), static_cast<uint32_t>(date.day()),
                           clock.hours().count(), clock.minutes().count(), clock.seconds().count(),
                           clock.subseconds().count());
    }

    std::string NewId(std::string_view prefix)
    {
        std::array<uint8_t, 8> random{};
        if (!eCore::RandomBytes(random))
        {
            throw std::runtime_error("system random number generator failed");
        }
        std::string hex = eCore::ToHex(random);
        return prefix.empty() ? hex : std::string(prefix) + "-" + hex;
    }

    std::optional<double> ParseNumber(std::string_view text)
    {
        if (text.empty())
        {
            return std::nullopt;
        }
        double value = 0;
        auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(value))
        {
            return std::nullopt;
        }
        return value;
    }

    std::string ReadFile(const std::filesystem::path &path, size_t maxBytes)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("cannot open '" + path.string() + "'");
        }
        std::string content;
        char buffer[64 * 1024];
        while (file)
        {
            file.read(buffer, sizeof(buffer));
            auto count = static_cast<size_t>(file.gcount());
            if (content.size() + count > maxBytes)
            {
                throw std::runtime_error("'" + path.string() + "' exceeds " + std::to_string(maxBytes) + " bytes");
            }
            content.append(buffer, count);
        }
        if (file.bad())
        {
            throw std::runtime_error("cannot read '" + path.string() + "'");
        }
        return content;
    }

    void WriteFileAtomic(const std::filesystem::path &path, std::string_view content)
    {
        std::filesystem::path target = std::filesystem::absolute(path);
        if (target.has_parent_path())
        {
            std::filesystem::create_directories(target.parent_path());
        }
        std::filesystem::path temporary = target;
        temporary += ".tmp-" + NewId({});
        auto failure = [&](const std::string &what)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return std::runtime_error("cannot write '" + target.string() + "': " + what);
        };
#ifdef _WIN32
        HANDLE file = CreateFileW(temporary.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
        {
            throw failure("create failed (" + std::to_string(GetLastError()) + ")");
        }
        bool okay = true;
        size_t written = 0;
        while (okay && written < content.size())
        {
            DWORD chunk = static_cast<DWORD>(std::min<size_t>(content.size() - written, 1u << 30));
            DWORD done = 0;
            okay = ::WriteFile(file, content.data() + written, chunk, &done, nullptr) && done > 0;
            written += done;
        }
        okay = okay && FlushFileBuffers(file);
        CloseHandle(file);
        if (!okay)
        {
            throw failure("write failed (" + std::to_string(GetLastError()) + ")");
        }
        if (!MoveFileExW(temporary.wstring().c_str(), target.wstring().c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        {
            throw failure("replace failed (" + std::to_string(GetLastError()) + ")");
        }
#else
        int file = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        if (file < 0)
        {
            throw failure("create failed");
        }
        bool okay = true;
        size_t written = 0;
        while (okay && written < content.size())
        {
            auto done = ::write(file, content.data() + written, content.size() - written);
            okay = done > 0;
            written += okay ? static_cast<size_t>(done) : 0;
        }
        okay = okay && ::fsync(file) == 0;
        okay = ::close(file) == 0 && okay;
        if (!okay)
        {
            throw failure("write failed");
        }
        if (std::rename(temporary.c_str(), target.c_str()) != 0)
        {
            throw failure("replace failed");
        }
        int directory = ::open(target.parent_path().c_str(), O_RDONLY | O_CLOEXEC);
        if (directory >= 0)
        {
            ::fsync(directory);
            ::close(directory);
        }
#endif
    }
} // namespace eOps
