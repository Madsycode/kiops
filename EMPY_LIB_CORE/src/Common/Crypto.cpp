#include <eCore/Common/Crypto.h>
#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#include <stdexcept>

namespace eCore
{
    namespace
    {
        // algorithm objects are fetched once: implicit per-call fetching is a measurable cost
        // in OpenSSL 3 for the small inputs typical of key derivation and envelope MACs
        EVP_MD *Sha256Algorithm()
        {
            static EVP_MD *s_Md = EVP_MD_fetch(nullptr, "SHA256", nullptr);
            return s_Md;
        }

        EVP_MAC *HmacAlgorithm()
        {
            static EVP_MAC *s_Mac = EVP_MAC_fetch(nullptr, "HMAC", nullptr);
            return s_Mac;
        }

        EVP_KDF *HkdfAlgorithm()
        {
            static EVP_KDF *s_Kdf = EVP_KDF_fetch(nullptr, "HKDF", nullptr);
            return s_Kdf;
        }

        EVP_MAC_CTX *NewHmacContext(ByteSpan key)
        {
            EVP_MAC_CTX *ctx = EVP_MAC_CTX_new(HmacAlgorithm());
            if (!ctx)
            {
                return nullptr;
            }
            char digest[] = "SHA256";
            OSSL_PARAM params[] = {OSSL_PARAM_construct_utf8_string(OSSL_MAC_PARAM_DIGEST, digest, 0),
                                   OSSL_PARAM_construct_end()};
            // HMAC accepts empty keys; OpenSSL requires a non-null pointer for them
            static const uint8_t s_Empty = 0;
            const uint8_t *keyData = key.empty() ? &s_Empty : key.data();
            if (EVP_MAC_init(ctx, keyData, key.size(), params) != 1)
            {
                EVP_MAC_CTX_free(ctx);
                return nullptr;
            }
            return ctx;
        }
    } // namespace

    Digest Sha256(ByteSpan data)
    {
        Digest digest{};
        unsigned int length = 0;
        if (EVP_Digest(data.data(), data.size(), digest.data(), &length, Sha256Algorithm(), nullptr) != 1)
        {
            throw std::runtime_error("SHA-256 failed");
        }
        return digest;
    }

    Digest HmacSha256(ByteSpan key, ByteSpan data)
    {
        HmacStream stream(key);
        stream.Update(data);
        return stream.Final();
    }

    bool HkdfSha256(ByteSpan ikm, ByteSpan salt, ByteSpan info, MutableByteSpan out)
    {
        if (out.empty() || out.size() > 255 * 32)
        {
            return false;
        }
        EVP_KDF_CTX *ctx = EVP_KDF_CTX_new(HkdfAlgorithm());
        if (!ctx)
        {
            return false;
        }
        static uint8_t s_Empty = 0;
        char digest[] = "SHA256";
        OSSL_PARAM params[5];
        size_t n = 0;
        params[n++] = OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, digest, 0);
        params[n++] = OSSL_PARAM_construct_octet_string(
            OSSL_KDF_PARAM_KEY, ikm.empty() ? &s_Empty : const_cast<uint8_t *>(ikm.data()), ikm.size());
        params[n++] = OSSL_PARAM_construct_octet_string(
            OSSL_KDF_PARAM_SALT, salt.empty() ? &s_Empty : const_cast<uint8_t *>(salt.data()), salt.size());
        params[n++] = OSSL_PARAM_construct_octet_string(
            OSSL_KDF_PARAM_INFO, info.empty() ? &s_Empty : const_cast<uint8_t *>(info.data()), info.size());
        params[n++] = OSSL_PARAM_construct_end();
        bool ok = EVP_KDF_derive(ctx, out.data(), out.size(), params) == 1;
        EVP_KDF_CTX_free(ctx);
        return ok;
    }

    bool RandomBytes(MutableByteSpan out)
    {
        if (out.empty())
        {
            return true;
        }
        return RAND_bytes(out.data(), static_cast<int>(out.size())) == 1;
    }

    bool ConstantTimeEquals(ByteSpan a, ByteSpan b) noexcept
    {
        if (a.size() != b.size())
        {
            return false;
        }
        if (a.empty())
        {
            return true;
        }
        return CRYPTO_memcmp(a.data(), b.data(), a.size()) == 0;
    }

    void SecureZero(void *data, size_t size) noexcept
    {
        if (data && size)
        {
            OPENSSL_cleanse(data, size);
        }
    }

    uint64_t Hash64(ByteSpan data)
    {
        Digest digest = Sha256(data);
        return LoadBE64(digest.data());
    }

    uint64_t Fingerprint64(ByteSpan data, uint64_t seed) noexcept
    {
        // 64-bit FNV-1a over 8-byte lanes folded with a multiply-xorshift finaliser;
        // adequate for detecting conflicting retransmissions, not adversarial collisions
        uint64_t hash = 14695981039346656037ull ^ (seed * 0x9E3779B97F4A7C15ull);
        const uint8_t *p = data.data();
        size_t size = data.size();
        while (size >= 8)
        {
            uint64_t lane = 0;
            std::memcpy(&lane, p, 8);
            hash = (hash ^ lane) * 1099511628211ull;
            hash ^= hash >> 29;
            p += 8;
            size -= 8;
        }
        while (size--)
        {
            hash = (hash ^ *p++) * 1099511628211ull;
        }
        hash ^= data.size();
        hash ^= hash >> 33;
        hash *= 0xFF51AFD7ED558CCDull;
        hash ^= hash >> 33;
        hash *= 0xC4CEB9FE1A85EC53ull;
        hash ^= hash >> 33;
        return hash;
    }

    std::string ToHex(ByteSpan data)
    {
        static constexpr char s_Digits[] = "0123456789abcdef";
        std::string text(data.size() * 2, '\0');
        for (size_t i = 0; i < data.size(); ++i)
        {
            text[2 * i] = s_Digits[data[i] >> 4];
            text[2 * i + 1] = s_Digits[data[i] & 0x0F];
        }
        return text;
    }

    bool FromHex(std::string_view text, std::vector<uint8_t> &out)
    {
        auto nibble = [](char c) -> int
        {
            if (c >= '0' && c <= '9')
            {
                return c - '0';
            }
            if (c >= 'a' && c <= 'f')
            {
                return c - 'a' + 10;
            }
            if (c >= 'A' && c <= 'F')
            {
                return c - 'A' + 10;
            }
            return -1;
        };
        if (text.size() % 2 != 0)
        {
            return false;
        }
        std::vector<uint8_t> bytes(text.size() / 2);
        for (size_t i = 0; i < bytes.size(); ++i)
        {
            int hi = nibble(text[2 * i]);
            int lo = nibble(text[2 * i + 1]);
            if (hi < 0 || lo < 0)
            {
                return false;
            }
            bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
        }
        out = std::move(bytes);
        return true;
    }

    // -------------------------
    // Sha256Stream
    // -------------------------

    struct Sha256Stream::State
    {
        EVP_MD_CTX *m_Ctx = nullptr;
        ~State()
        {
            EVP_MD_CTX_free(m_Ctx);
        }
    };

    Sha256Stream::Sha256Stream() : m_State(std::make_unique<State>())
    {
        m_State->m_Ctx = EVP_MD_CTX_new();
        if (!m_State->m_Ctx || EVP_DigestInit_ex2(m_State->m_Ctx, Sha256Algorithm(), nullptr) != 1)
        {
            throw std::runtime_error("SHA-256 initialisation failed");
        }
    }

    Sha256Stream::~Sha256Stream() = default;
    Sha256Stream::Sha256Stream(Sha256Stream &&) noexcept = default;
    Sha256Stream &Sha256Stream::operator=(Sha256Stream &&) noexcept = default;

    void Sha256Stream::Update(ByteSpan data)
    {
        if (!data.empty())
        {
            EVP_DigestUpdate(m_State->m_Ctx, data.data(), data.size());
        }
    }

    Digest Sha256Stream::Final()
    {
        Digest digest{};
        unsigned int length = 0;
        EVP_DigestFinal_ex(m_State->m_Ctx, digest.data(), &length);
        EVP_DigestInit_ex2(m_State->m_Ctx, Sha256Algorithm(), nullptr);
        return digest;
    }

    // -------------------------
    // HmacStream
    // -------------------------

    struct HmacStream::State
    {
        EVP_MAC_CTX *m_Ctx = nullptr;
        ~State()
        {
            EVP_MAC_CTX_free(m_Ctx);
        }
    };

    HmacStream::HmacStream(ByteSpan key) : m_State(std::make_unique<State>())
    {
        m_State->m_Ctx = NewHmacContext(key);
        if (!m_State->m_Ctx)
        {
            throw std::runtime_error("HMAC-SHA256 initialisation failed");
        }
    }

    HmacStream::~HmacStream() = default;
    HmacStream::HmacStream(HmacStream &&) noexcept = default;
    HmacStream &HmacStream::operator=(HmacStream &&) noexcept = default;

    void HmacStream::Update(ByteSpan data)
    {
        if (!data.empty())
        {
            EVP_MAC_update(m_State->m_Ctx, data.data(), data.size());
        }
    }

    Digest HmacStream::Final()
    {
        Digest digest{};
        size_t length = 0;
        EVP_MAC_final(m_State->m_Ctx, digest.data(), &length, digest.size());
        return digest;
    }
} // namespace eCore
