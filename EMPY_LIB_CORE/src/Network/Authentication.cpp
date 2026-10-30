#include <eCore/Network/Authentication.h>
#include <eCore/Network/Topic.h>

namespace eCore::MMQP
{
    Buffer RandomBytes(size_t count)
    {
        if (count > INT_MAX)
        {
            throw std::length_error("Random byte request exceeds provider bounds");
        }
        Buffer bytes(count);
        if (!eCore::RandomBytes(bytes))
        {
            throw std::runtime_error("Secure random generator failed");
        }
        return bytes;
    }

    Buffer FromHex(std::string_view text)
    {
        Buffer bytes;
        if (!eCore::FromHex(text, bytes))
        {
            throw std::invalid_argument("Invalid hexadecimal data");
        }
        return bytes;
    }

    Buffer Hkdf256(ByteSpan key, ByteSpan salt, ByteSpan info, size_t count)
    {
        if (count == 0 || count > 255 * 32)
        {
            throw std::invalid_argument("Invalid HKDF output size");
        }
        Buffer bytes(count);
        if (!eCore::HkdfSha256(key, salt, info, bytes))
        {
            SecureErase(bytes);
            throw std::runtime_error("HKDF-SHA256 failed");
        }
        return bytes;
    }

    std::array<uint8_t, 8> AuthenticationIdentity(std::string_view name)
    {
        if (name.empty() || name.size() > 128 || !ValidUtf8(name))
        {
            throw std::invalid_argument("Authentication identity must contain 1..128 UTF-8 bytes");
        }
        Sha256Stream hash;
        hash.Update(std::string_view("MMQP/AUTH/"));
        hash.Update(name);
        auto digest = hash.Final();
        std::array<uint8_t, 8> identity;
        std::copy_n(digest.begin(), identity.size(), identity.begin());
        return identity;
    }
} // namespace eCore::MMQP
