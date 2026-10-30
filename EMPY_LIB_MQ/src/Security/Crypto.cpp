#include <eMQ/Security/Crypto.h>
#include <eCore/Common/Storage.h>
#include <algorithm>
#include <bitset>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <tuple>

extern "C"
{
#include "crypto_aead.h"
}

namespace eMQ
{
    namespace
    {
        const uint8_t *Data(std::span<const uint8_t> data)
        {
            static constexpr uint8_t empty = 0;
            return data.empty() ? &empty : data.data();
        }

        std::span<const uint8_t> Associated(std::array<uint8_t, 259> &buffer, std::string_view topic, uint32_t epoch)
        {
            if (topic.size() > UINT8_MAX)
            {
                throw ProtocolError("Topic length is outside 1..255 bytes", MMQP_REASON_INVALID_TOPIC);
            }
            std::memcpy(buffer.data(), topic.data(), topic.size());
            for (size_t i = 0; i < 4; ++i)
            {
                buffer[topic.size() + i] = static_cast<uint8_t>(epoch >> (24 - i * 8));
            }
            return std::span<const uint8_t>(buffer).first(topic.size() + 4);
        }

        struct EraseKey
        {
            Key128 &m_Key;
            ~EraseKey()
            {
                SecureErase(m_Key);
            }
        };
    } // namespace

    Buffer Ascon::Encrypt(std::span<const uint8_t, 16> key, std::span<const uint8_t, 16> nonce,
                          std::span<const uint8_t> plaintext, std::span<const uint8_t> associated)
    {
        if (plaintext.size() > SIZE_MAX - 16)
        {
            throw std::length_error("Plaintext too large");
        }
        Buffer encrypted(plaintext.size() + 16);
        unsigned long long size = encrypted.size();
        if (crypto_aead_encrypt(encrypted.data(), &size, Data(plaintext), plaintext.size(), Data(associated),
                                associated.size(), nullptr, nonce.data(), key.data()) != 0 ||
            size != encrypted.size())
        {
            throw std::runtime_error("Ascon encryption failed");
        }
        return encrypted;
    }

    Buffer Ascon::Decrypt(std::span<const uint8_t, 16> key, std::span<const uint8_t, 16> nonce,
                          std::span<const uint8_t> ciphertext, std::span<const uint8_t> associated)
    {
        if (ciphertext.size() < 16)
        {
            throw ProtocolError("Missing authentication tag", MMQP_REASON_AUTH_FAILED);
        }
        Buffer plaintext(std::max<size_t>(1, ciphertext.size() - 16));
        unsigned long long size = 0;
        if (crypto_aead_decrypt(plaintext.data(), &size, nullptr, ciphertext.data(), ciphertext.size(),
                                Data(associated), associated.size(), nonce.data(), key.data()) != 0 ||
            size != ciphertext.size() - 16)
        {
            SecureErase(plaintext);
            throw ProtocolError("Payload authentication failed", MMQP_REASON_AUTH_FAILED);
        }
        plaintext.resize(static_cast<size_t>(size));
        return plaintext;
    }

    KeyGrant::KeyGrant(Buffer secret, std::string topic, std::optional<uint32_t> period)
        : m_Key(std::move(secret)), m_Root(std::move(topic)), m_Epoch(period)
    {
        if ((m_Key.size() != 16) || (!m_Root.empty() && (!m_Epoch || m_Key.size() != 16)))
        {
            SecureErase(m_Key);
            throw std::invalid_argument("Root keys require 16 bytes; subtree keys require 16 bytes and an epoch");
        }
        try
        {
            if (!m_Root.empty())
            {
                m_Root = ParseTopic(m_Root).m_Filter;
            }
        }
        catch (...)
        {
            SecureErase(m_Key);
            throw;
        }
    }

    KeyGrant::KeyGrant(KeyGrant &&) noexcept = default;
    KeyGrant &KeyGrant::operator=(KeyGrant &&other) noexcept
    {
        if (this != &other)
        {
            SecureErase(m_Key);
            m_Key = std::move(other.m_Key);
            m_Root = std::move(other.m_Root);
            m_Epoch = other.m_Epoch;
        }
        return *this;
    }
    KeyGrant::~KeyGrant()
    {
        SecureErase(m_Key);
    }

    NonceSequence::NonceSequence()
    {
        auto random = RandomBytes(m_Prefix.size());
        std::copy(random.begin(), random.end(), m_Prefix.begin());
    }

    NonceSequence::NonceSequence(std::array<uint8_t, 8> prefix, uint64_t counter) : m_Prefix(prefix), m_Counter(counter)
    {
    }

    std::array<uint8_t, 16> NonceSequence::Next()
    {
        if (m_Exhausted)
        {
            throw std::overflow_error("Nonce counter exhausted");
        }
        std::array<uint8_t, 16> nonce{};
        std::copy(m_Prefix.begin(), m_Prefix.end(), nonce.begin());
        for (size_t i = 0; i < 8; ++i)
        {
            nonce[15 - i] = static_cast<uint8_t>(m_Counter >> (i * 8));
        }
        if (m_Counter == UINT64_MAX)
        {
            m_Exhausted = true;
        }
        else
        {
            ++m_Counter;
        }
        return nonce;
    }

    void NonceSequence::Save(Writer &writer) const
    {
        writer.Bytes(m_Prefix);
        writer.U64(m_Counter);
        writer.U8(m_Exhausted);
    }

    void NonceSequence::Load(Reader &reader)
    {
        auto prefix = reader.Bytes(m_Prefix.size());
        std::copy(prefix.begin(), prefix.end(), m_Prefix.begin());
        m_Counter = reader.U64();
        auto exhausted = reader.U8();
        if (exhausted > 1 || (exhausted && m_Counter != UINT64_MAX))
        {
            throw ProtocolError("Invalid persistent nonce sequence");
        }
        m_Exhausted = exhausted != 0;
    }

    // Persistent state layout (FileStore records):
    //   "payload/policy"       U32 0x45504332 | grant identity[32] | minimum U32 | maximum U32
    //   "payload/nonce"        U32 0x45504333 | prefix[8] | reserved U64
    //   "payload/replay/<hex>" U32 0x45504334 | topic String | epoch U32 | prefix U64 | highest U64 | bitmap[32]
    // <hex> is Sha256(topic String | epoch U32 | prefix U64). Counters below `reserved`
    // may have been used; a restart resumes at `reserved`, so a crash skips counters
    // but never reuses one. Each replay tuple is checkpointed alone before plaintext
    // release. A legacy single "payload" record (0x45504331) is migrated, then erased.
    struct PayloadCipher::Impl
    {
        using CacheName = std::pair<std::string_view, uint32_t>;
        using ReplayName = std::tuple<std::string_view, uint32_t, uint64_t>;
        struct CachedKey
        {
            std::string m_Topic;
            uint32_t m_Epoch;
            Key128 m_Key;
        };
        struct ReplayState
        {
            std::string m_Topic;
            std::string m_Record;
            uint64_t m_Highest = 0;
            std::bitset<256> m_Bits;
        };

        KeyGrant m_Grant;
        uint32_t m_Minimum;
        uint32_t m_Maximum;
        size_t m_CacheLimit;
        size_t m_ReplayLimit;
        uint64_t m_Reservation;
        uint64_t m_Reserved = 0;
        std::list<CachedKey> m_Keys; // Most recently used first; index views point into its nodes.
        std::map<CacheName, std::list<CachedKey>::iterator> m_Index;
        std::map<ReplayName, std::unique_ptr<ReplayState>> m_Replays; // Name views point into the state.
        NonceSequence m_Nonces;
        std::mutex m_Mutex;
        std::unique_ptr<FileStore> m_Storage;
        Digest m_Identity{};
        bool m_Failed = false;

        Impl(KeyGrant secret, uint32_t first, uint32_t last, size_t cache, size_t replay, const std::string &directory,
             uint64_t block)
            : m_Grant(std::move(secret)), m_Minimum(first), m_Maximum(last), m_CacheLimit(cache), m_ReplayLimit(replay),
              m_Reservation(block)
        {
            if (first > last || !cache || !replay || !block || replay > UINT32_MAX ||
                replay > (SIZE_MAX / 2 - 1024) / 320)
            {
                throw std::invalid_argument("Invalid epoch, cryptographic cache or nonce reservation bounds");
            }
            if (directory.empty())
            {
                return;
            }
            Writer binding;
            binding.String("MMQP/PAYLOAD/STATE/1");
            binding.String(m_Grant.m_Root);
            binding.U8(m_Grant.m_Epoch.has_value());
            if (m_Grant.m_Epoch)
            {
                binding.U32(*m_Grant.m_Epoch);
            }
            m_Identity = Hmac256(m_Grant.m_Key, binding.Data());
            // Room for the per-tuple layout plus a legacy record during migration.
            m_Storage = std::make_unique<FileStore>(directory, 2 * (1024 + replay * 320), replay + 3);
            if (auto legacy = m_Storage->Get("payload"))
            {
                this->Legacy(*legacy);
                this->Policy();
                this->Persist(m_Nonces.Counter());
                for (const auto &[name, state] : m_Replays)
                {
                    this->Persist(*state, state->m_Highest, state->m_Bits, name);
                }
                m_Storage->Erase("payload");
                return;
            }
            this->Load();
            this->Policy();
        }

        ~Impl()
        {
            for (auto &entry : m_Keys)
            {
                SecureErase(entry.m_Key);
            }
        }

        void Check() const
        {
            if (m_Failed)
            {
                throw ProtocolError("Payload state storage failed; reopen the cipher before continuing",
                                    MMQP_REASON_BUSY);
            }
        }

        void Put(std::string key, Buffer value)
        {
            try
            {
                m_Storage->Put(std::move(key), std::move(value));
            }
            catch (...)
            {
                m_Failed = true;
                throw;
            }
        }

        void Policy()
        {
            Writer writer(64);
            writer.U32(0x45504332);
            writer.Bytes(m_Identity);
            writer.U32(m_Minimum);
            writer.U32(m_Maximum);
            this->Put("payload/policy", writer.Take());
        }

        void Persist(uint64_t limit)
        {
            Writer writer(32);
            writer.U32(0x45504333);
            writer.Bytes(m_Nonces.Prefix());
            writer.U64(limit);
            this->Put("payload/nonce", writer.Take());
        }

        void Persist(const ReplayState &state, uint64_t highest, const std::bitset<256> &bits, ReplayName name)
        {
            Writer writer(320);
            writer.U32(0x45504334);
            writer.String(state.m_Topic);
            writer.U32(std::get<1>(name));
            writer.U64(std::get<2>(name));
            writer.U64(highest);
            for (size_t i = 0; i < bits.size(); i += 8)
            {
                uint8_t byte = 0;
                for (size_t j = 0; j < 8; ++j)
                {
                    if (bits[i + j])
                    {
                        byte |= uint8_t(1U << j);
                    }
                }
                writer.U8(byte);
            }
            this->Put(state.m_Record, writer.Take());
        }

        static std::string Record(std::string_view topic, uint32_t epoch, uint64_t prefix)
        {
            Writer writer(272);
            writer.String(topic);
            writer.U32(epoch);
            writer.U64(prefix);
            return "payload/replay/" + ToHex(Sha256(writer.Data()));
        }

        static std::bitset<256> Bits(Reader &reader)
        {
            std::bitset<256> bits;
            for (size_t i = 0; i < bits.size(); i += 8)
            {
                auto byte = reader.U8();
                for (size_t j = 0; j < 8; ++j)
                {
                    bits[i + j] = (byte & (1U << j)) != 0;
                }
            }
            return bits;
        }

        void Track(std::string topic, uint32_t epoch, uint64_t prefix, uint64_t highest, std::bitset<256> bits,
                   uint32_t first, uint32_t last)
        {
            if (!bits[0] || epoch < first || epoch > last)
            {
                throw ProtocolError("Invalid persistent payload replay window");
            }
            if (epoch < m_Minimum || epoch > m_Maximum)
            {
                return;
            }
            if (m_Replays.size() >= m_ReplayLimit)
            {
                throw ProtocolError("Persistent payload replay budget exceeded");
            }
            auto record = Record(topic, epoch, prefix);
            auto state = std::make_unique<ReplayState>(ReplayState{std::move(topic), std::move(record), highest, bits});
            if (!m_Replays.emplace(ReplayName{state->m_Topic, epoch, prefix}, std::move(state)).second)
            {
                throw ProtocolError("Duplicate persistent replay entry");
            }
        }

        void Legacy(const Buffer &bytes)
        {
            Reader reader(bytes);
            if (reader.U32() != 0x45504331 || !SecureEqual(reader.Bytes(32), m_Identity))
            {
                throw ProtocolError("Payload state belongs to a different key grant or format");
            }
            auto first = reader.U32();
            auto last = reader.U32();
            if (first > last || m_Minimum < first || m_Maximum < last)
            {
                throw ProtocolError("Persistent payload epoch policy cannot move backwards");
            }
            m_Nonces.Load(reader);
            m_Reserved = m_Nonces.Counter();
            auto count = reader.U32();
            if (count > m_ReplayLimit)
            {
                throw ProtocolError("Persistent payload replay budget exceeded");
            }
            for (uint32_t i = 0; i < count; ++i)
            {
                auto topic = ParseTopic(reader.String()).m_Filter;
                auto epoch = reader.U32();
                auto prefix = reader.U64();
                auto highest = reader.U64();
                this->Track(std::move(topic), epoch, prefix, highest, Bits(reader), first, last);
            }
            reader.Finish();
        }

        void Load()
        {
            auto policy = m_Storage->Get("payload/policy");
            if (!policy)
            {
                if (!m_Storage->Keys("payload").empty())
                {
                    throw ProtocolError("Payload state lacks its epoch policy record");
                }
                return;
            }
            Reader reader(*policy);
            if (reader.U32() != 0x45504332 || !SecureEqual(reader.Bytes(32), m_Identity))
            {
                throw ProtocolError("Payload state belongs to a different key grant or format");
            }
            auto first = reader.U32();
            auto last = reader.U32();
            reader.Finish();
            if (first > last || m_Minimum < first || m_Maximum < last)
            {
                throw ProtocolError("Persistent payload epoch policy cannot move backwards");
            }
            if (auto saved = m_Storage->Get("payload/nonce"))
            {
                Reader nonce(*saved);
                if (nonce.U32() != 0x45504333)
                {
                    throw ProtocolError("Invalid persistent nonce reservation");
                }
                std::array<uint8_t, 8> prefix{};
                auto bytes = nonce.Bytes(prefix.size());
                std::copy(bytes.begin(), bytes.end(), prefix.begin());
                m_Reserved = nonce.U64();
                nonce.Finish();
                m_Nonces = NonceSequence(prefix, m_Reserved);
            }
            for (const auto &key : m_Storage->Keys("payload/replay/"))
            {
                auto bytes = m_Storage->Get(key);
                Reader entry(*bytes);
                if (entry.U32() != 0x45504334)
                {
                    throw ProtocolError("Invalid persistent payload replay record");
                }
                auto topic = entry.String();
                auto epoch = entry.U32();
                auto prefix = entry.U64();
                auto highest = entry.U64();
                auto bits = Bits(entry);
                entry.Finish();
                if (ParseTopic(topic).m_Filter != topic || Record(topic, epoch, prefix) != key)
                {
                    throw ProtocolError("Persistent payload replay record does not match its key");
                }
                if (epoch < m_Minimum || epoch > m_Maximum)
                {
                    if (!bits[0] || epoch < first || epoch > last)
                    {
                        throw ProtocolError("Invalid persistent payload replay window");
                    }
                    m_Storage->Erase(key);
                    continue;
                }
                this->Track(std::move(topic), epoch, prefix, highest, bits, first, last);
            }
        }

        std::array<uint8_t, 16> Nonce()
        {
            if (m_Storage && m_Nonces.Counter() >= m_Reserved)
            {
                // Reserve a block durably before releasing any ciphertext from it.
                auto counter = m_Nonces.Counter();
                if (m_Nonces.Exhausted() || counter == UINT64_MAX)
                {
                    throw std::overflow_error("Nonce counter exhausted");
                }
                auto limit = counter + std::min(m_Reservation, UINT64_MAX - counter);
                this->Persist(limit);
                m_Reserved = limit;
            }
            return m_Nonces.Next();
        }

        void Evict(std::list<CachedKey>::iterator entry)
        {
            m_Index.erase(CacheName{entry->m_Topic, entry->m_Epoch});
            SecureErase(entry->m_Key);
            m_Keys.erase(entry);
        }

        Key128 Derive(std::string_view topic, uint32_t epoch)
        {
            if (epoch < m_Minimum || epoch > m_Maximum || (m_Grant.m_Epoch && *m_Grant.m_Epoch != epoch))
            {
                throw ProtocolError("Key epoch is not authorized", MMQP_REASON_UNAUTHORIZED);
            }
            if (auto found = m_Index.find(CacheName{topic, epoch}); found != m_Index.end())
            {
                m_Keys.splice(m_Keys.begin(), m_Keys, found->second);
                return found->second->m_Key;
            }
            size_t start = 0;
            if (!m_Grant.m_Root.empty())
            {
                if (topic != m_Grant.m_Root &&
                    !(topic.starts_with(m_Grant.m_Root) && topic[m_Grant.m_Root.size()] == '/'))
                {
                    throw ProtocolError("Topic is outside the delegated key subtree", MMQP_REASON_UNAUTHORIZED);
                }
                start = m_Grant.m_Root.size() + 1;
            }
            Buffer secret = m_Grant.m_Key;
            try
            {
                Writer salt(4);
                salt.U32(epoch);
                while (start < topic.size())
                {
                    auto end = topic.find('/', start);
                    if (end == std::string_view::npos)
                    {
                        end = topic.size();
                    }
                    auto level = topic.substr(start, end - start);
                    auto derived = Hkdf256(secret, salt.Data(),
                                           {reinterpret_cast<const uint8_t *>(level.data()), level.size()}, 16);
                    SecureErase(secret);
                    secret = std::move(derived);
                    start = end + 1;
                }
                Key128 key{};
                EraseKey erase{key};
                std::copy_n(secret.begin(), key.size(), key.begin());
                SecureErase(secret);
                if (m_Keys.size() >= m_CacheLimit)
                {
                    this->Evict(std::prev(m_Keys.end()));
                }
                m_Keys.push_front(CachedKey{std::string(topic), epoch, key});
                m_Index.emplace(CacheName{m_Keys.front().m_Topic, epoch}, m_Keys.begin());
                return Key128(key);
            }
            catch (...)
            {
                SecureErase(secret);
                throw;
            }
        }

        void Replay(std::string_view topic, const EncryptionHeader &encryption)
        {
            // Rotation may have run since the key was derived outside this lock.
            if (encryption.m_Epoch < m_Minimum || encryption.m_Epoch > m_Maximum)
            {
                throw ProtocolError("Key epoch is not authorized", MMQP_REASON_UNAUTHORIZED);
            }
            Reader reader(encryption.m_Nonce);
            auto prefix = reader.U64();
            auto counter = reader.U64();
            ReplayName name{topic, encryption.m_Epoch, prefix};
            auto found = m_Replays.find(name);
            if (found == m_Replays.end())
            {
                if (m_Replays.size() >= m_ReplayLimit)
                {
                    throw ProtocolError("Replay state budget exhausted", MMQP_REASON_POLICY_VIOLATION);
                }
                auto state = std::make_unique<ReplayState>(ReplayState{
                    std::string(topic), m_Storage ? Record(topic, encryption.m_Epoch, prefix) : std::string{}, counter,
                    std::bitset<256>{1}});
                name = ReplayName{state->m_Topic, encryption.m_Epoch, prefix};
                if (m_Storage)
                {
                    this->Persist(*state, counter, state->m_Bits, name);
                }
                m_Replays.emplace(name, std::move(state));
                return;
            }
            auto &state = *found->second;
            auto highest = state.m_Highest;
            auto bits = state.m_Bits;
            if (counter > highest)
            {
                auto shift = counter - highest;
                if (shift >= bits.size())
                {
                    bits.reset();
                }
                else
                {
                    bits <<= static_cast<size_t>(shift);
                }
                bits.set(0);
                highest = counter;
            }
            else
            {
                auto shift = highest - counter;
                if (shift >= bits.size())
                {
                    throw ProtocolError("Encrypted nonce is outside the replay window", MMQP_REASON_POLICY_VIOLATION);
                }
                if (bits[static_cast<size_t>(shift)])
                {
                    throw PayloadReplay();
                }
                bits.set(static_cast<size_t>(shift));
            }
            if (m_Storage)
            {
                this->Persist(state, highest, bits, found->first);
            }
            state.m_Highest = highest;
            state.m_Bits = bits;
        }

        void Prune()
        {
            for (auto entry = m_Keys.begin(); entry != m_Keys.end();)
            {
                auto next = std::next(entry);
                if (entry->m_Epoch < m_Minimum || entry->m_Epoch > m_Maximum)
                {
                    this->Evict(entry);
                }
                entry = next;
            }
            for (auto entry = m_Replays.begin(); entry != m_Replays.end();)
            {
                auto epoch = std::get<1>(entry->first);
                if (epoch >= m_Minimum && epoch <= m_Maximum)
                {
                    ++entry;
                    continue;
                }
                if (m_Storage)
                {
                    try
                    {
                        m_Storage->Erase(entry->second->m_Record);
                    }
                    catch (...)
                    {
                        m_Failed = true;
                        throw;
                    }
                }
                entry = m_Replays.erase(entry);
            }
        }
    };

    PayloadCipher::PayloadCipher(KeyGrant grant, uint32_t minimumEpoch, uint32_t maximumEpoch, size_t cacheLimit,
                                 size_t replayLimit, std::string stateDirectory, uint64_t nonceReservation)
        : m_Impl(std::make_unique<Impl>(std::move(grant), minimumEpoch, maximumEpoch, cacheLimit, replayLimit,
                                        stateDirectory, nonceReservation))
    {
    }

    PayloadCipher::~PayloadCipher() = default;

    Key128 PayloadCipher::Derive(std::string_view topic, uint32_t epoch)
    {
        auto name = ParseTopic(topic).m_Filter;
        std::scoped_lock lock(m_Impl->m_Mutex);
        m_Impl->Check();
        return m_Impl->Derive(name, epoch);
    }

    void PayloadCipher::Encrypt(PublishRequest &publication, uint32_t epoch)
    {
        if (publication.m_Encryption)
        {
            throw std::invalid_argument("Publication is already encrypted");
        }
        auto topic = ParseTopic(publication.m_Topic).m_Filter;
        Key128 key{};
        EraseKey erase{key};
        EncryptionHeader encryption;
        encryption.m_Epoch = epoch;
        {
            std::scoped_lock lock(m_Impl->m_Mutex);
            m_Impl->Check();
            key = m_Impl->Derive(topic, epoch);
            encryption.m_Nonce = m_Impl->Nonce();
        }
        // The nonce is now exclusively ours, so the cipher itself runs without the lock.
        std::array<uint8_t, 259> associated;
        auto bytes =
            Ascon::Encrypt(key, encryption.m_Nonce, publication.m_Payload, Associated(associated, topic, epoch));
        std::copy(bytes.end() - 16, bytes.end(), encryption.m_Tag.begin());
        bytes.resize(bytes.size() - 16);
        if (publication.m_Payload.UseCount() == 1)
        {
            eCore::SecureZero(publication.m_Payload.MutableData(), publication.m_Payload.Size());
        }
        publication.m_Payload = std::move(bytes);
        publication.m_Topic = std::move(topic);
        publication.m_Encryption = encryption;
    }

    Buffer PayloadCipher::Decrypt(const PublishRequest &publication, std::string_view topic)
    {
        if (!publication.m_Encryption)
        {
            throw ProtocolError("An encrypted payload is required", MMQP_REASON_AUTH_FAILED);
        }
        auto name = ParseTopic(topic).m_Filter;
        const auto &encryption = *publication.m_Encryption;
        Key128 key{};
        EraseKey erase{key};
        {
            std::scoped_lock lock(m_Impl->m_Mutex);
            m_Impl->Check();
            key = m_Impl->Derive(name, encryption.m_Epoch);
        }
        Buffer bytes;
        bytes.reserve(publication.m_Payload.Size() + encryption.m_Tag.size());
        bytes.insert(bytes.end(), publication.m_Payload.Span().begin(), publication.m_Payload.Span().end());
        bytes.insert(bytes.end(), encryption.m_Tag.begin(), encryption.m_Tag.end());
        std::array<uint8_t, 259> associated;
        auto plaintext =
            Ascon::Decrypt(key, encryption.m_Nonce, bytes, Associated(associated, name, encryption.m_Epoch));
        try
        {
            // The tag is verified; checkpoint the replay tuple before releasing plaintext.
            std::scoped_lock lock(m_Impl->m_Mutex);
            m_Impl->Check();
            m_Impl->Replay(name, encryption);
        }
        catch (...)
        {
            SecureErase(plaintext);
            throw;
        }
        return plaintext;
    }

    void PayloadCipher::Rotate(uint32_t minimumEpoch, uint32_t maximumEpoch)
    {
        std::scoped_lock lock(m_Impl->m_Mutex);
        m_Impl->Check();
        if (minimumEpoch > maximumEpoch || minimumEpoch < m_Impl->m_Minimum || maximumEpoch < m_Impl->m_Maximum)
        {
            throw std::invalid_argument("Epoch policy cannot move backwards");
        }
        m_Impl->m_Minimum = minimumEpoch;
        m_Impl->m_Maximum = maximumEpoch;
        if (m_Impl->m_Storage)
        {
            m_Impl->Policy();
        }
        m_Impl->Prune();
    }
} // namespace eMQ
