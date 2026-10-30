#pragma once
#include <span>
#include <atomic>
#include <string>
#include <vector>
#include <cstring>
#include <utility>
#include <algorithm>
#include <string_view>
#include <limits>
#include <stdexcept>
#include "Globals.h"

// buffer utilities: shared immutable payloads, inline small buffers, big-endian codecs

namespace eCore
{
    using Byte = uint8_t;
    using ByteSpan = std::span<const uint8_t>;
    using MutableByteSpan = std::span<uint8_t>;

    // -------------------------
    // big-endian load/store
    // -------------------------

    EMPY_INLINE constexpr uint16_t LoadBE16(const uint8_t *p) noexcept
    {
        return static_cast<uint16_t>((uint16_t(p[0]) << 8) | uint16_t(p[1]));
    }

    EMPY_INLINE constexpr uint32_t LoadBE32(const uint8_t *p) noexcept
    {
        return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
    }

    EMPY_INLINE constexpr uint64_t LoadBE64(const uint8_t *p) noexcept
    {
        return (uint64_t(LoadBE32(p)) << 32) | uint64_t(LoadBE32(p + 4));
    }

    // load an unsigned big-endian integer of 1..8 bytes
    EMPY_INLINE constexpr uint64_t LoadBEN(const uint8_t *p, size_t width) noexcept
    {
        uint64_t value = 0;
        for (size_t i = 0; i < width; ++i)
        {
            value = (value << 8) | p[i];
        }
        return value;
    }

    EMPY_INLINE constexpr void StoreBE16(uint8_t *p, uint16_t v) noexcept
    {
        p[0] = uint8_t(v >> 8);
        p[1] = uint8_t(v);
    }

    EMPY_INLINE constexpr void StoreBE32(uint8_t *p, uint32_t v) noexcept
    {
        p[0] = uint8_t(v >> 24);
        p[1] = uint8_t(v >> 16);
        p[2] = uint8_t(v >> 8);
        p[3] = uint8_t(v);
    }

    EMPY_INLINE constexpr void StoreBE64(uint8_t *p, uint64_t v) noexcept
    {
        StoreBE32(p, uint32_t(v >> 32));
        StoreBE32(p + 4, uint32_t(v));
    }

    // store the low `width` (1..8) bytes of an unsigned value in big-endian order
    EMPY_INLINE constexpr void StoreBEN(uint8_t *p, uint64_t v, size_t width) noexcept
    {
        for (size_t i = width; i > 0; --i)
        {
            p[i - 1] = uint8_t(v);
            v >>= 8;
        }
    }

    // -------------------------
    // SharedBytes
    // -------------------------

    // Immutable, reference-counted byte block allocated in one piece (control word + data).
    // Publication payloads are copied out of network buffers exactly once into a SharedBytes and
    // then referenced by every recipient, retained queue and retry without further copies.
    // The data is writable only through MutableData() while the block is uniquely owned.
    class SharedBytes
    {
    public:
        EMPY_INLINE SharedBytes() noexcept = default;

        SharedBytes(ByteSpan bytes) : SharedBytes(Copy(bytes))
        {
        }
        SharedBytes(const std::vector<uint8_t> &bytes) : SharedBytes(Copy(ByteSpan(bytes)))
        {
        }
        SharedBytes(std::initializer_list<uint8_t> bytes) : SharedBytes(Copy(ByteSpan(bytes.begin(), bytes.size())))
        {
        }

        EMPY_INLINE ~SharedBytes()
        {
            Release();
        }

        EMPY_INLINE SharedBytes(const SharedBytes &other) noexcept : m_Block(other.m_Block)
        {
            if (m_Block)
            {
                m_Block->m_Refs.fetch_add(1, std::memory_order_relaxed);
            }
        }

        EMPY_INLINE SharedBytes(SharedBytes &&other) noexcept : m_Block(std::exchange(other.m_Block, nullptr))
        {
        }

        EMPY_INLINE SharedBytes &operator=(const SharedBytes &other) noexcept
        {
            if (this != &other)
            {
                SharedBytes copy(other);
                Swap(copy);
            }
            return *this;
        }

        EMPY_INLINE SharedBytes &operator=(SharedBytes &&other) noexcept
        {
            if (this != &other)
            {
                Release();
                m_Block = std::exchange(other.m_Block, nullptr);
            }
            return *this;
        }

        // allocate an uninitialised block of `size` bytes (size 0 yields an empty handle)
        EMPY_NODISCARD static SharedBytes Allocate(size_t size)
        {
            SharedBytes bytes;
            if (size == 0)
            {
                return bytes;
            }
            if (size > UINT32_MAX || size > SIZE_MAX - sizeof(Block))
            {
                throw std::length_error("Shared payload exceeds its size representation");
            }
            void *memory = ::operator new(sizeof(Block) + size);
            bytes.m_Block = new (memory) Block{};
            bytes.m_Block->m_Size = static_cast<uint32_t>(size);
            return bytes;
        }

        EMPY_NODISCARD static SharedBytes Copy(ByteSpan source)
        {
            SharedBytes bytes = Allocate(source.size());
            if (!source.empty())
            {
                std::memcpy(bytes.MutableData(), source.data(), source.size());
            }
            return bytes;
        }

        EMPY_NODISCARD static SharedBytes Copy(std::string_view source)
        {
            return Copy(ByteSpan(reinterpret_cast<const uint8_t *>(source.data()), source.size()));
        }

        EMPY_INLINE const uint8_t *Data() const noexcept
        {
            return m_Block ? reinterpret_cast<const uint8_t *>(m_Block + 1) : nullptr;
        }

        // writable access, permitted only while this handle is the sole owner
        EMPY_INLINE uint8_t *MutableData() noexcept
        {
            EMPY_ASSERT(!m_Block || m_Block->m_Refs.load(std::memory_order_relaxed) == 1);
            return m_Block ? reinterpret_cast<uint8_t *>(m_Block + 1) : nullptr;
        }

        EMPY_INLINE size_t Size() const noexcept
        {
            return m_Block ? m_Block->m_Size : 0;
        }
        EMPY_INLINE bool Empty() const noexcept
        {
            return Size() == 0;
        }
        EMPY_INLINE ByteSpan Span() const noexcept
        {
            return ByteSpan(Data(), Size());
        }
        EMPY_INLINE operator ByteSpan() const noexcept
        {
            return Span();
        }
        EMPY_INLINE uint8_t operator[](size_t index) const noexcept
        {
            return Data()[index];
        }

        EMPY_INLINE std::string_view View() const noexcept
        {
            return std::string_view(reinterpret_cast<const char *>(Data()), Size());
        }

        EMPY_INLINE uint32_t UseCount() const noexcept
        {
            return m_Block ? m_Block->m_Refs.load(std::memory_order_relaxed) : 0;
        }

        EMPY_INLINE void Swap(SharedBytes &other) noexcept
        {
            std::swap(m_Block, other.m_Block);
        }

        EMPY_INLINE void Reset() noexcept
        {
            Release();
        }

        EMPY_INLINE friend bool operator==(const SharedBytes &a, const SharedBytes &b) noexcept
        {
            return a.Size() == b.Size() && (a.m_Block == b.m_Block || std::memcmp(a.Data(), b.Data(), a.Size()) == 0);
        }

    private:
        struct Block
        {
            std::atomic<uint32_t> m_Refs{1};
            uint32_t m_Size = 0;
        };
        EMPY_STATIC(sizeof(Block) == 8, "SharedBytes control block must stay compact");

        EMPY_INLINE void Release() noexcept
        {
            if (m_Block && m_Block->m_Refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                m_Block->~Block();
                ::operator delete(static_cast<void *>(m_Block));
            }
            m_Block = nullptr;
        }

        Block *m_Block = nullptr;
    };

    // -------------------------
    // InlineBuffer
    // -------------------------

    // Growable contiguous byte buffer that keeps up to `_Inline` bytes without touching the heap.
    // Used for per-recipient packet headers, which are small and created on every delivery.
    template <size_t _Inline> class InlineBuffer
    {
    public:
        EMPY_INLINE InlineBuffer() noexcept = default;

        EMPY_INLINE ~InlineBuffer()
        {
            if (m_Data != m_Inline)
            {
                ::operator delete(m_Data);
            }
        }

        EMPY_INLINE InlineBuffer(const InlineBuffer &other)
        {
            Append(other.Span());
        }

        EMPY_INLINE InlineBuffer(InlineBuffer &&other) noexcept
        {
            MoveFrom(other);
        }

        EMPY_INLINE InlineBuffer &operator=(const InlineBuffer &other)
        {
            if (this != &other)
            {
                Clear();
                Append(other.Span());
            }
            return *this;
        }

        EMPY_INLINE InlineBuffer &operator=(InlineBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (m_Data != m_Inline)
                {
                    ::operator delete(m_Data);
                }
                m_Data = m_Inline;
                m_Size = 0;
                m_Capacity = _Inline;
                MoveFrom(other);
            }
            return *this;
        }

        EMPY_INLINE uint8_t *Data() noexcept
        {
            return m_Data;
        }
        EMPY_INLINE const uint8_t *Data() const noexcept
        {
            return m_Data;
        }
        EMPY_INLINE size_t Size() const noexcept
        {
            return m_Size;
        }
        EMPY_INLINE size_t Capacity() const noexcept
        {
            return m_Capacity;
        }
        EMPY_INLINE bool Empty() const noexcept
        {
            return m_Size == 0;
        }
        EMPY_INLINE ByteSpan Span() const noexcept
        {
            return ByteSpan(m_Data, m_Size);
        }
        EMPY_INLINE uint8_t &operator[](size_t i) noexcept
        {
            return m_Data[i];
        }
        EMPY_INLINE uint8_t operator[](size_t i) const noexcept
        {
            return m_Data[i];
        }
        EMPY_INLINE void Clear() noexcept
        {
            m_Size = 0;
        }

        EMPY_INLINE void Reserve(size_t capacity)
        {
            if (capacity <= m_Capacity)
            {
                return;
            }
            size_t next = std::max(capacity, m_Capacity <= SIZE_MAX / 2 ? m_Capacity * 2 : m_Capacity);
            auto *memory = static_cast<uint8_t *>(::operator new(next));
            if (m_Size)
            {
                std::memcpy(memory, m_Data, m_Size);
            }
            if (m_Data != m_Inline)
            {
                ::operator delete(m_Data);
            }
            m_Data = memory;
            m_Capacity = next;
        }

        // extend by `count` bytes and return a pointer to the (uninitialised) new region
        EMPY_INLINE uint8_t *Grow(size_t count)
        {
            if (count > SIZE_MAX - m_Size)
            {
                throw std::length_error("Buffer size overflow");
            }
            Reserve(m_Size + count);
            uint8_t *tail = m_Data + m_Size;
            m_Size += count;
            return tail;
        }

        EMPY_INLINE void Resize(size_t size)
        {
            Reserve(size);
            m_Size = size;
        }

        EMPY_INLINE void PushBack(uint8_t value)
        {
            *Grow(1) = value;
        }

        EMPY_INLINE void Append(ByteSpan bytes)
        {
            if (bytes.empty())
            {
                return;
            }
            const auto address = reinterpret_cast<uintptr_t>(bytes.data());
            const auto base = reinterpret_cast<uintptr_t>(m_Data);
            const bool internal = address >= base && address - base < m_Size;
            const auto offset = internal ? address - base : 0;
            auto *tail = Grow(bytes.size());
            std::memmove(tail, internal ? m_Data + offset : bytes.data(), bytes.size());
        }

    private:
        EMPY_INLINE void MoveFrom(InlineBuffer &other) noexcept
        {
            if (other.m_Data == other.m_Inline)
            {
                std::memcpy(m_Inline, other.m_Inline, other.m_Size);
                m_Size = other.m_Size;
            }
            else
            {
                m_Data = other.m_Data;
                m_Size = other.m_Size;
                m_Capacity = other.m_Capacity;
                other.m_Data = other.m_Inline;
                other.m_Capacity = _Inline;
            }
            other.m_Size = 0;
        }

        uint8_t *m_Data = m_Inline;
        size_t m_Size = 0;
        size_t m_Capacity = _Inline;
        uint8_t m_Inline[_Inline];
    };

    // -------------------------
    // WireFrame
    // -------------------------

    // inline capacity for encoded control packets and publication headers (topic, ids, envelope)
    inline constexpr size_t EMPY_FRAME_INLINE = 160;

    // One encoded packet ready for transmission: a small per-recipient head followed by an
    // optional shared tail (typically the publication payload). Transports emit Head then Tail
    // with a gathered write, so fan-out never copies the payload.
    struct WireFrame
    {
        WireFrame() = default;
        WireFrame(const std::vector<uint8_t> &bytes)
        {
            m_Head.Append(bytes);
        }

        InlineBuffer<EMPY_FRAME_INLINE> m_Head;
        SharedBytes m_Tail;

        EMPY_INLINE size_t Size() const noexcept
        {
            return m_Head.Size() + m_Tail.Size();
        }

        std::vector<uint8_t> Flatten() const
        {
            std::vector<uint8_t> result(Size());
            if (!m_Head.Empty())
            {
                std::memcpy(result.data(), m_Head.Data(), m_Head.Size());
            }
            if (!m_Tail.Empty())
            {
                std::memcpy(result.data() + m_Head.Size(), m_Tail.Data(), m_Tail.Size());
            }
            return result;
        }
    };

    // -------------------------
    // ByteWriter
    // -------------------------

    // Appends big-endian fields to a byte container exposing Grow(n) (InlineBuffer) or
    // resize/data (std::vector<uint8_t>). Callers validate field ranges before writing.
    template <typename _Buffer> class ByteWriter
    {
    public:
        EMPY_INLINE explicit ByteWriter(_Buffer &out) noexcept : m_Out(out)
        {
        }

        EMPY_INLINE void U8(uint8_t v)
        {
            *Grow(1) = v;
        }
        EMPY_INLINE void U16(uint16_t v)
        {
            StoreBE16(Grow(2), v);
        }
        EMPY_INLINE void U32(uint32_t v)
        {
            StoreBE32(Grow(4), v);
        }
        EMPY_INLINE void U64(uint64_t v)
        {
            StoreBE64(Grow(8), v);
        }

        // unsigned integer occupying exactly `width` (1..8) bytes
        EMPY_INLINE void UN(uint64_t v, size_t width)
        {
            if (!width || width > 8)
            {
                throw std::invalid_argument("Integer width must be in 1..8");
            }
            StoreBEN(Grow(width), v, width);
        }

        EMPY_INLINE void Bytes(ByteSpan bytes)
        {
            if (bytes.empty())
            {
                return;
            }
            const auto address = reinterpret_cast<uintptr_t>(bytes.data());
            const auto base = reinterpret_cast<uintptr_t>(DataOf(m_Out));
            const bool internal = address >= base && address - base < SizeOf(m_Out);
            const auto offset = internal ? address - base : 0;
            auto *tail = Grow(bytes.size());
            std::memmove(tail, internal ? DataOf(m_Out) + offset : bytes.data(), bytes.size());
        }

        EMPY_INLINE void Bytes(std::string_view text)
        {
            Bytes(ByteSpan(reinterpret_cast<const uint8_t *>(text.data()), text.size()));
        }

        // Checked one-octet length prefix followed by the bytes.
        EMPY_INLINE void Str8(std::string_view text)
        {
            if (text.size() > 255)
            {
                throw std::length_error("String exceeds one-octet length");
            }
            U8(static_cast<uint8_t>(text.size()));
            Bytes(text);
        }

        EMPY_INLINE void Str8(ByteSpan bytes)
        {
            if (bytes.size() > 255)
            {
                throw std::length_error("Blob exceeds one-octet length");
            }
            U8(static_cast<uint8_t>(bytes.size()));
            Bytes(bytes);
        }

        EMPY_INLINE size_t Position() const noexcept
        {
            return SizeOf(m_Out);
        }
        EMPY_INLINE uint8_t *At(size_t offset) noexcept
        {
            return DataOf(m_Out) + offset;
        }

    private:
        EMPY_INLINE uint8_t *Grow(size_t count)
        {
            if constexpr (requires(_Buffer &b) { b.Grow(count); })
            {
                return m_Out.Grow(count);
            }
            else
            {
                size_t offset = m_Out.size();
                if (count > m_Out.max_size() - offset)
                {
                    throw std::length_error("Buffer size overflow");
                }
                m_Out.resize(offset + count);
                return m_Out.data() + offset;
            }
        }

        static EMPY_INLINE size_t SizeOf(const _Buffer &b) noexcept
        {
            if constexpr (requires { b.Size(); })
            {
                return b.Size();
            }
            else
            {
                return b.size();
            }
        }

        static EMPY_INLINE uint8_t *DataOf(_Buffer &b) noexcept
        {
            if constexpr (requires { b.Data(); })
            {
                return b.Data();
            }
            else
            {
                return b.data();
            }
        }

        _Buffer &m_Out;
    };

    // -------------------------
    // ByteReader
    // -------------------------

    // Bounds-checked big-endian reader over a byte span. Every accessor returns false on
    // underflow and the failure is sticky, so decoders may test Ok() once after a sequence.
    // Outputs are zeroed on failure, so lengths read from a failed reader are always 0.
    class ByteReader
    {
    public:
        EMPY_INLINE explicit ByteReader(ByteSpan data) noexcept : m_Data(data)
        {
        }

        EMPY_INLINE bool U8(uint8_t &v) noexcept
        {
            if (!Need(1))
            {
                v = 0;
                return false;
            }
            v = m_Data[m_Offset];
            m_Offset += 1;
            return true;
        }

        EMPY_INLINE bool U16(uint16_t &v) noexcept
        {
            if (!Need(2))
            {
                v = 0;
                return false;
            }
            v = LoadBE16(m_Data.data() + m_Offset);
            m_Offset += 2;
            return true;
        }

        EMPY_INLINE bool U32(uint32_t &v) noexcept
        {
            if (!Need(4))
            {
                v = 0;
                return false;
            }
            v = LoadBE32(m_Data.data() + m_Offset);
            m_Offset += 4;
            return true;
        }

        EMPY_INLINE bool U64(uint64_t &v) noexcept
        {
            if (!Need(8))
            {
                v = 0;
                return false;
            }
            v = LoadBE64(m_Data.data() + m_Offset);
            m_Offset += 8;
            return true;
        }

        // unsigned integer occupying exactly `width` (1..8) bytes
        EMPY_INLINE bool UN(uint64_t &v, size_t width) noexcept
        {
            if (width == 0 || width > 8)
            {
                m_Ok = false;
                v = 0;
                return false;
            }
            if (!Need(width))
            {
                v = 0;
                return false;
            }
            v = LoadBEN(m_Data.data() + m_Offset, width);
            m_Offset += width;
            return true;
        }

        // view of the next `count` bytes (valid as long as the underlying buffer)
        EMPY_INLINE bool Bytes(size_t count, ByteSpan &out) noexcept
        {
            if (!Need(count))
            {
                out = {};
                return false;
            }
            out = m_Data.subspan(m_Offset, count);
            m_Offset += count;
            return true;
        }

        // copy the next `out.size()` bytes into a fixed-size destination
        EMPY_INLINE bool Copy(MutableByteSpan out) noexcept
        {
            if (!Need(out.size()))
            {
                return false;
            }
            if (!out.empty())
            {
                std::memcpy(out.data(), m_Data.data() + m_Offset, out.size());
            }
            m_Offset += out.size();
            return true;
        }

        // one-octet length prefix followed by that many bytes
        EMPY_INLINE bool Str8(std::string_view &out) noexcept
        {
            uint8_t length = 0;
            ByteSpan bytes;
            if (!U8(length) || !Bytes(length, bytes))
            {
                out = {};
                return false;
            }
            out = std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            return true;
        }

        EMPY_INLINE bool Skip(size_t count) noexcept
        {
            if (!Need(count))
            {
                return false;
            }
            m_Offset += count;
            return true;
        }

        // remaining unread bytes (consumes them)
        EMPY_INLINE ByteSpan Rest() noexcept
        {
            ByteSpan rest = m_Ok ? m_Data.subspan(m_Offset) : ByteSpan{};
            m_Offset = m_Data.size();
            return rest;
        }

        EMPY_INLINE size_t Remaining() const noexcept
        {
            return m_Ok ? m_Data.size() - m_Offset : 0;
        }
        EMPY_INLINE size_t Offset() const noexcept
        {
            return m_Offset;
        }
        EMPY_INLINE bool AtEnd() const noexcept
        {
            return m_Ok && m_Offset == m_Data.size();
        }
        EMPY_INLINE bool Ok() const noexcept
        {
            return m_Ok;
        }

    private:
        EMPY_INLINE bool Need(size_t count) noexcept
        {
            if (!m_Ok || m_Data.size() - m_Offset < count)
            {
                EMPY_UNLIKELY
                {
                    m_Ok = false;
                    return false;
                }
            }
            return true;
        }

        ByteSpan m_Data;
        size_t m_Offset = 0;
        bool m_Ok = true;
    };

    // view a string as bytes without copying
    EMPY_INLINE ByteSpan AsBytes(std::string_view text) noexcept
    {
        return ByteSpan(reinterpret_cast<const uint8_t *>(text.data()), text.size());
    }

    // view bytes as a string without copying
    EMPY_INLINE std::string_view AsText(ByteSpan bytes) noexcept
    {
        return std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
} // namespace eCore
