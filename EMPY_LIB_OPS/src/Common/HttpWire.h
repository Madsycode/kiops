#pragma once
#include <eOps/Common/Http.h>
#include <asio.hpp>
#include <asio/ssl.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <charconv>
#include <functional>

// private HTTP/1.1 wire helpers shared by the client (Http.cpp) and the loopback server
// (HttpServer.cpp): deadline-bounded asynchronous steps on a private io_context, message
// heads, header fields and Content-Length / chunked / close-delimited bodies

namespace eOps::Detail
{
    using Clock = std::chrono::steady_clock;
    using Fields = std::vector<std::pair<std::string, std::string>>;

    // a message exceeded its byte budget (the server answers 413)
    struct TooLarge : HttpError
    {
        using HttpError::HttpError;
    };

    // a deadline expired
    struct Expired : HttpError
    {
        using HttpError::HttpError;
    };

    inline char LowerChar(char c) noexcept
    {
        return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
    }

    inline bool SameText(std::string_view a, std::string_view b) noexcept
    {
        if (a.size() != b.size())
        {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (LowerChar(a[i]) != LowerChar(b[i]))
            {
                return false;
            }
        }
        return true;
    }

    inline std::string FindField(const Fields &fields, std::string_view name)
    {
        for (const auto &[key, value] : fields)
        {
            if (SameText(key, name))
            {
                return value;
            }
        }
        return {};
    }

    inline bool HasField(const Fields &fields, std::string_view name)
    {
        for (const auto &field : fields)
        {
            if (SameText(field.first, name))
            {
                return true;
            }
        }
        return false;
    }

    // RFC 9110 token characters (methods, field names)
    inline bool IsToken(std::string_view text) noexcept
    {
        if (text.empty())
        {
            return false;
        }
        for (char c : text)
        {
            bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
            if (!alnum && std::string_view("!#$%&'*+-.^_`|~").find(c) == std::string_view::npos)
            {
                return false;
            }
        }
        return true;
    }

    inline bool SafeValue(std::string_view text) noexcept
    {
        return text.find_first_of("\r\n", 0) == std::string_view::npos && text.find('\0') == std::string_view::npos;
    }

    // run one asynchronous step until it completes or `deadline` passes; on expiry `cancel`
    // aborts the I/O object, the aborted handler is drained and Expired is thrown
    template <typename _Start, typename _Cancel>
    std::error_code Await(asio::io_context &io, Clock::time_point deadline, _Start &&start, _Cancel &&cancel)
    {
        std::error_code result;
        bool done = false;
        start(
            [&result, &done](std::error_code code)
            {
                result = code;
                done = true;
            });
        io.restart();
        while (!done)
        {
            if (Clock::now() >= deadline || io.run_one_until(deadline) == 0)
            {
                if (done)
                {
                    break;
                }
                cancel();
                io.restart();
                while (!done && io.run_one() > 0)
                {
                }
                throw Expired("HTTP operation timed out");
            }
        }
        return result;
    }

    // buffered reader over a TCP socket or TLS stream
    template <typename _Stream> struct Reader
    {
        Reader(asio::io_context &io, _Stream &stream, Clock::time_point deadline, std::function<void()> cancel)
            : m_Io(io), m_Stream(stream), m_Deadline(deadline), m_Cancel(std::move(cancel))
        {
        }

        asio::io_context &m_Io;
        _Stream &m_Stream;
        Clock::time_point m_Deadline;
        std::function<void()> m_Cancel;
        std::string m_Buffer;
        bool m_Eof = false;

        // append more bytes; false at end of stream
        bool More()
        {
            if (m_Eof)
            {
                return false;
            }
            char chunk[16 * 1024];
            size_t count = 0;
            std::error_code code = Await(
                m_Io, m_Deadline,
                [&](auto done)
                {
                    m_Stream.async_read_some(asio::buffer(chunk),
                                             [&count, done](std::error_code error, size_t bytes)
                                             {
                                                 count = bytes;
                                                 done(error);
                                             });
                },
                m_Cancel);
            m_Buffer.append(chunk, count);
            if (code == asio::error::eof || code == asio::ssl::error::stream_truncated)
            {
                m_Eof = true;
                return count > 0;
            }
            if (code)
            {
                throw HttpError("HTTP read failed: " + code.message());
            }
            return true;
        }

        // the head up to and including the blank line, removed from the buffer
        std::string Head(size_t limit)
        {
            size_t searched = 0;
            while (true)
            {
                size_t end = m_Buffer.find("\r\n\r\n", searched);
                if (end != std::string::npos)
                {
                    if (end + 4 > limit)
                    {
                        throw TooLarge("HTTP head exceeds its size limit");
                    }
                    std::string head = m_Buffer.substr(0, end + 4);
                    m_Buffer.erase(0, end + 4);
                    return head;
                }
                if (m_Buffer.size() > limit)
                {
                    throw TooLarge("HTTP head exceeds its size limit");
                }
                searched = m_Buffer.size() >= 3 ? m_Buffer.size() - 3 : 0;
                if (!More())
                {
                    throw HttpError("connection closed inside the HTTP head");
                }
            }
        }

        // one CRLF-terminated line (chunk sizes and trailers), removed from the buffer
        std::string Line(size_t limit)
        {
            while (true)
            {
                size_t end = m_Buffer.find("\r\n");
                if (end != std::string::npos)
                {
                    std::string line = m_Buffer.substr(0, end);
                    m_Buffer.erase(0, end + 2);
                    return line;
                }
                if (m_Buffer.size() > limit)
                {
                    throw TooLarge("HTTP chunk line exceeds its size limit");
                }
                if (!More())
                {
                    throw HttpError("connection closed inside a chunked body");
                }
            }
        }

        void Need(size_t bytes, const char *what)
        {
            while (m_Buffer.size() < bytes)
            {
                if (!More())
                {
                    throw HttpError(what);
                }
            }
        }

        // decode the body selected by the fields; `closeDelimited` permits read-until-EOF
        std::string Body(const Fields &fields, size_t limit, bool closeDelimited)
        {
            std::string encoding = FindField(fields, "Transfer-Encoding");
            if (!encoding.empty())
            {
                std::string lowered;
                for (char c : encoding)
                {
                    lowered.push_back(LowerChar(c));
                }
                while (!lowered.empty() && (lowered.back() == ' ' || lowered.back() == '\t'))
                {
                    lowered.pop_back();
                }
                if (lowered.size() < 7 || lowered.compare(lowered.size() - 7, 7, "chunked") != 0)
                {
                    throw HttpError("unsupported HTTP transfer coding");
                }
                return Chunked(limit);
            }
            std::string length = FindField(fields, "Content-Length");
            if (!length.empty())
            {
                size_t size = 0;
                auto result = std::from_chars(length.data(), length.data() + length.size(), size);
                if (result.ec != std::errc{} || result.ptr != length.data() + length.size())
                {
                    throw HttpError("invalid HTTP Content-Length");
                }
                if (size > limit)
                {
                    throw TooLarge("HTTP body exceeds its size limit");
                }
                Need(size, "connection closed before the HTTP body was complete");
                std::string body = m_Buffer.substr(0, size);
                m_Buffer.erase(0, size);
                return body;
            }
            if (!closeDelimited)
            {
                return {};
            }
            while (More())
            {
                if (m_Buffer.size() > limit)
                {
                    throw TooLarge("HTTP body exceeds its size limit");
                }
            }
            if (m_Buffer.size() > limit)
            {
                throw TooLarge("HTTP body exceeds its size limit");
            }
            return std::move(m_Buffer);
        }

        std::string Chunked(size_t limit)
        {
            std::string body;
            while (true)
            {
                std::string line = Line(1024);
                std::string_view digits(line);
                digits = digits.substr(0, digits.find(';'));
                while (!digits.empty() && (digits.back() == ' ' || digits.back() == '\t'))
                {
                    digits.remove_suffix(1);
                }
                size_t size = 0;
                auto result = std::from_chars(digits.data(), digits.data() + digits.size(), size, 16);
                if (digits.empty() || result.ec != std::errc{} || result.ptr != digits.data() + digits.size())
                {
                    throw HttpError("invalid HTTP chunk size");
                }
                if (size == 0)
                {
                    // trailer fields until the blank line
                    size_t trailers = 0;
                    while (!Line(limit).empty())
                    {
                        if (++trailers > 64)
                        {
                            throw HttpError("too many HTTP trailer fields");
                        }
                    }
                    return body;
                }
                if (size > limit - body.size())
                {
                    throw TooLarge("HTTP body exceeds its size limit");
                }
                Need(size + 2, "connection closed inside an HTTP chunk");
                if (m_Buffer.compare(size, 2, "\r\n") != 0)
                {
                    throw HttpError("HTTP chunk is not CRLF terminated");
                }
                body.append(m_Buffer, 0, size);
                m_Buffer.erase(0, size + 2);
            }
        }
    };

    // split a head into its start line and header fields
    inline std::string ParseHead(const std::string &head, Fields &fields)
    {
        size_t end = head.find("\r\n");
        std::string start = head.substr(0, end);
        size_t position = end + 2;
        while (position + 2 <= head.size())
        {
            size_t next = head.find("\r\n", position);
            if (next == position)
            {
                break;
            }
            std::string_view line(head.data() + position, next - position);
            size_t colon = line.find(':');
            if (colon == std::string_view::npos || colon == 0 || line.front() == ' ' || line.front() == '\t' ||
                !IsToken(line.substr(0, colon)))
            {
                throw HttpError("malformed HTTP header field");
            }
            std::string_view value = line.substr(colon + 1);
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
            {
                value.remove_prefix(1);
            }
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t'))
            {
                value.remove_suffix(1);
            }
            fields.emplace_back(std::string(line.substr(0, colon)), std::string(value));
            if (fields.size() > 256)
            {
                throw HttpError("too many HTTP header fields");
            }
            position = next + 2;
        }
        return start;
    }

    inline const char *ReasonPhrase(int32_t status)
    {
        switch (status)
        {
        case 200:
            return "OK";
        case 201:
            return "Created";
        case 202:
            return "Accepted";
        case 204:
            return "No Content";
        case 400:
            return "Bad Request";
        case 401:
            return "Unauthorized";
        case 403:
            return "Forbidden";
        case 404:
            return "Not Found";
        case 405:
            return "Method Not Allowed";
        case 408:
            return "Request Timeout";
        case 413:
            return "Content Too Large";
        case 429:
            return "Too Many Requests";
        case 500:
            return "Internal Server Error";
        case 501:
            return "Not Implemented";
        case 502:
            return "Bad Gateway";
        case 503:
            return "Service Unavailable";
        case 504:
            return "Gateway Timeout";
        default:
            return "Status";
        }
    }
} // namespace eOps::Detail
