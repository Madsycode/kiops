#pragma once
#include <array>
#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <eCore/Common/Buffers.h>
#include "MMQP.h"

namespace eCore::MMQP
{
    using Buffer = std::vector<uint8_t>;

    struct ProtocolError : std::runtime_error
    {
        uint8_t m_Reason;

        ProtocolError(const char *text, uint8_t code = MMQP_REASON_MALFORMED_PACKET)
            : std::runtime_error(text), m_Reason(code)
        {
        }
    };

    struct Reader
    {
        EMPY_INLINE explicit Reader(std::span<const uint8_t> data) : m_Reader(data)
        {
        }

        EMPY_INLINE size_t Remaining() const
        {
            return m_Reader.Remaining();
        }
        EMPY_INLINE bool Empty() const
        {
            return this->Remaining() == 0;
        }

        std::span<const uint8_t> Bytes(size_t count)
        {
            ByteSpan bytes;
            if (!m_Reader.Bytes(count, bytes))
            {
                throw ProtocolError("Truncated packet body");
            }
            return bytes;
        }

        uint64_t Number(uint8_t count)
        {
            if (!count || count > 8)
            {
                throw ProtocolError("Invalid integer width");
            }
            uint64_t value = 0;
            if (!m_Reader.UN(value, count))
            {
                throw ProtocolError("Truncated integer");
            }
            return value;
        }

        EMPY_INLINE uint8_t U8()
        {
            return static_cast<uint8_t>(this->Number(1));
        }
        EMPY_INLINE uint16_t U16()
        {
            return static_cast<uint16_t>(this->Number(2));
        }
        EMPY_INLINE uint32_t U32()
        {
            return static_cast<uint32_t>(this->Number(4));
        }
        EMPY_INLINE uint64_t U64()
        {
            return this->Number(8);
        }

        std::string String()
        {
            auto bytes = this->Bytes(this->U8());
            return std::string(bytes.begin(), bytes.end());
        }

        Buffer Blob()
        {
            auto bytes = this->Bytes(this->U8());
            return Buffer(bytes.begin(), bytes.end());
        }

        void Finish() const
        {
            if (!this->Empty())
            {
                throw ProtocolError("Unexpected trailing packet data");
            }
        }

    private:
        ByteReader m_Reader;
    };

    struct Writer
    {
        EMPY_INLINE explicit Writer(size_t limit = 1024 * 1024) : m_Limit(limit)
        {
        }

        void Bytes(std::span<const uint8_t> bytes)
        {
            if (bytes.size() > m_Limit - m_Data.Size())
            {
                throw ProtocolError("Packet exceeds configured size", MMQP_REASON_POLICY_VIOLATION);
            }
            m_Data.Append(bytes);
        }

        void Number(uint64_t value, uint8_t count)
        {
            if (!count || count > 8 || (count < 8 && value >= (uint64_t{1} << (count * 8))))
            {
                throw ProtocolError("Integer does not fit its wire field");
            }
            if (count > m_Limit - m_Data.Size())
            {
                throw ProtocolError("Packet exceeds configured size", MMQP_REASON_POLICY_VIOLATION);
            }
            ByteWriter(m_Data).UN(value, count);
        }

        EMPY_INLINE void U8(uint8_t value)
        {
            this->Number(value, 1);
        }
        EMPY_INLINE void U16(uint16_t value)
        {
            this->Number(value, 2);
        }
        EMPY_INLINE void U32(uint32_t value)
        {
            this->Number(value, 4);
        }
        EMPY_INLINE void U64(uint64_t value)
        {
            this->Number(value, 8);
        }

        void Blob(std::span<const uint8_t> bytes)
        {
            if (bytes.size() > UINT8_MAX)
            {
                throw ProtocolError("One-octet string length exceeded");
            }
            this->U8(static_cast<uint8_t>(bytes.size()));
            this->Bytes(bytes);
        }

        void String(std::string_view text)
        {
            this->Blob({reinterpret_cast<const uint8_t *>(text.data()), text.size()});
        }

        EMPY_INLINE void Reserve(size_t bytes)
        {
            m_Data.Reserve(std::min(bytes, m_Limit));
        }
        EMPY_INLINE ByteSpan Data() const
        {
            return m_Data.Span();
        }
        EMPY_INLINE Buffer Take()
        {
            return Buffer(m_Data.Data(), m_Data.Data() + m_Data.Size());
        }

    private:
        InlineBuffer<512> m_Data;
        size_t m_Limit;
    };

    struct PacketHeader
    {
        uint8_t m_Type = 0;
        uint8_t m_Flags = 0;
        uint32_t m_Remaining = 0;
        uint8_t m_Size = 0;
    };

    EMPY_CORE_API uint8_t LengthWidth(uint32_t length);
    EMPY_CORE_API std::optional<PacketHeader> ReadHeader(std::span<const uint8_t> bytes, uint32_t limit = 1024 * 1024);
    EMPY_CORE_API Buffer Frame(uint8_t type, uint8_t flags, std::span<const uint8_t> body,
                               uint32_t limit = 1024 * 1024);

    struct EMPY_CORE_API FrameStream
    {
        explicit FrameStream(uint32_t limit = 1024 * 1024);
        void Feed(std::span<const uint8_t> bytes, const std::function<void(std::span<const uint8_t>)> &receive);
        void Reset();
        void SetLimit(uint32_t limit);
        size_t Buffered() const;

    private:
        Buffer m_Buffer;
        uint32_t m_Limit;
        size_t m_Target = 1;
        bool m_Body = false;
    };
} // namespace eCore::MMQP
