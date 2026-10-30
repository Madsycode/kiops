#pragma once
#include <eCore/Network/Codecs.h>
#include <eCore/Network/Authentication.h>

namespace eMQ
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    struct PayloadReplay : ProtocolError
    {
        PayloadReplay() : ProtocolError("Replayed encrypted publication", MMQP_REASON_AUTH_FAILED)
        {
        }
    };

    struct EMPY_MQ_API Ascon
    {
        // NIST SP 800-232 Ascon-AEAD128; output is ciphertext followed by its tag.
        static Buffer Encrypt(std::span<const uint8_t, 16> key, std::span<const uint8_t, 16> nonce,
                              std::span<const uint8_t> plaintext, std::span<const uint8_t> associated);
        static Buffer Decrypt(std::span<const uint8_t, 16> key, std::span<const uint8_t, 16> nonce,
                              std::span<const uint8_t> ciphertext, std::span<const uint8_t> associated);
    };

    struct EMPY_MQ_API KeyGrant
    {
        Buffer m_Key;
        std::string m_Root;
        std::optional<uint32_t> m_Epoch;

        explicit KeyGrant(Buffer secret, std::string topic = {}, std::optional<uint32_t> period = {});
        KeyGrant(KeyGrant &&) noexcept;
        KeyGrant &operator=(KeyGrant &&) noexcept;
        KeyGrant(const KeyGrant &) = delete;
        KeyGrant &operator=(const KeyGrant &) = delete;
        ~KeyGrant();
    };

    struct EMPY_MQ_API NonceSequence
    {
        NonceSequence();
        NonceSequence(std::array<uint8_t, 8> prefix, uint64_t counter);
        std::array<uint8_t, 16> Next();
        void Save(Writer &writer) const;
        void Load(Reader &reader);
        EMPY_INLINE const std::array<uint8_t, 8> &Prefix() const
        {
            return m_Prefix;
        }
        EMPY_INLINE uint64_t Counter() const
        {
            return m_Counter;
        }
        EMPY_INLINE bool Exhausted() const
        {
            return m_Exhausted;
        }

    private:
        std::array<uint8_t, 8> m_Prefix{};
        uint64_t m_Counter = 0;
        bool m_Exhausted = false;
    };

    struct EMPY_MQ_API PayloadCipher
    {
        PayloadCipher(KeyGrant grant, uint32_t minimumEpoch, uint32_t maximumEpoch, size_t cacheLimit = 128,
                      size_t replayLimit = 4096, std::string stateDirectory = {}, uint64_t nonceReservation = 1024);
        ~PayloadCipher();
        PayloadCipher(const PayloadCipher &) = delete;
        PayloadCipher &operator=(const PayloadCipher &) = delete;

        Key128 Derive(std::string_view topic, uint32_t epoch);
        void Encrypt(PublishRequest &publication, uint32_t epoch);
        Buffer Decrypt(const PublishRequest &publication, std::string_view topic);
        void Rotate(uint32_t minimumEpoch, uint32_t maximumEpoch);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eMQ
