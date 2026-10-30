#pragma once
#include "Stream.h"
#include <eCore/Common/Crypto.h>

namespace eCore::MMQP
{
    using Key128 = std::array<uint8_t, 16>;
    using eCore::Hash64;
    using eCore::Sha256;
    using eCore::ToHex;

    EMPY_CORE_API Buffer RandomBytes(size_t count);
    EMPY_CORE_API Buffer FromHex(std::string_view text);
    EMPY_CORE_API Buffer Hkdf256(ByteSpan key, ByteSpan salt, ByteSpan info, size_t count);
    EMPY_CORE_API std::array<uint8_t, 8> AuthenticationIdentity(std::string_view name);

    inline Digest Hmac256(ByteSpan key, ByteSpan bytes)
    {
        return eCore::HmacSha256(key, bytes);
    }
    inline Digest FrameDigest(const WireFrame &frame)
    {
        eCore::Sha256Stream digest;
        digest.Update(frame.m_Head.Span());
        digest.Update(frame.m_Tail.Span());
        return digest.Final();
    }
    inline bool SecureEqual(ByteSpan a, ByteSpan b) noexcept
    {
        return eCore::ConstantTimeEquals(a, b);
    }
    inline void SecureErase(MutableByteSpan bytes) noexcept
    {
        eCore::SecureZero(bytes.data(), bytes.size());
    }
} // namespace eCore::MMQP
