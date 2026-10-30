#pragma once
#include <array>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "Buffers.h"

// crypto utilities: hashing, MACs, key derivation and randomness (backed by OpenSSL, which
// stays an implementation detail of EmpyCore; no OpenSSL type appears in this interface)

namespace eCore
{
    // SHA-256 / HMAC-SHA256 output
    using Digest = std::array<uint8_t, 32>;

    // SHA-256 of a byte string
    EMPY_CORE_API Digest Sha256(ByteSpan data);

    // HMAC-SHA256 under `key`
    EMPY_CORE_API Digest HmacSha256(ByteSpan key, ByteSpan data);

    // HKDF-SHA256 (RFC 5869) extract-then-expand; fills `out` (at most 255*32 bytes)
    EMPY_CORE_API bool HkdfSha256(ByteSpan ikm, ByteSpan salt, ByteSpan info, MutableByteSpan out);

    // cryptographically secure random bytes; false if the system RNG failed
    EMPY_CORE_API bool RandomBytes(MutableByteSpan out);

    // constant-time equality (length is not secret; differing lengths compare unequal)
    EMPY_CORE_API bool ConstantTimeEquals(ByteSpan a, ByteSpan b) noexcept;

    // best-effort zeroisation that the optimiser may not elide
    EMPY_CORE_API void SecureZero(void *data, size_t size) noexcept;

    // H64(x): the first eight bytes of SHA-256(x) as an unsigned big-endian integer
    // (the specified placement hash; replaces implementation-dependent library hashes)
    EMPY_CORE_API uint64_t Hash64(ByteSpan data);

    // fast non-cryptographic 64-bit fingerprint (receipt/content identity, never security)
    EMPY_CORE_API uint64_t Fingerprint64(ByteSpan data, uint64_t seed = 0) noexcept;

    // lowercase hexadecimal encoding / strict decoding (even length, [0-9a-fA-F])
    EMPY_CORE_API std::string ToHex(ByteSpan data);
    EMPY_CORE_API bool FromHex(std::string_view text, std::vector<uint8_t> &out);

    // incremental SHA-256 for multi-part inputs (UMIDs, manifests)
    class EMPY_CORE_API Sha256Stream
    {
    public:
        Sha256Stream();
        ~Sha256Stream();
        Sha256Stream(Sha256Stream &&) noexcept;
        Sha256Stream &operator=(Sha256Stream &&) noexcept;
        Sha256Stream(const Sha256Stream &) = delete;
        Sha256Stream &operator=(const Sha256Stream &) = delete;

        void Update(ByteSpan data);
        void Update(std::string_view text)
        {
            Update(AsBytes(text));
        }
        Digest Final();

    private:
        struct State;
        std::unique_ptr<State> m_State;
    };

    // incremental HMAC-SHA256 for envelopes assembled from several parts
    class EMPY_CORE_API HmacStream
    {
    public:
        explicit HmacStream(ByteSpan key);
        ~HmacStream();
        HmacStream(HmacStream &&) noexcept;
        HmacStream &operator=(HmacStream &&) noexcept;
        HmacStream(const HmacStream &) = delete;
        HmacStream &operator=(const HmacStream &) = delete;

        void Update(ByteSpan data);
        Digest Final();

    private:
        struct State;
        std::unique_ptr<State> m_State;
    };
} // namespace eCore
