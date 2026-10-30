#pragma once
#include "Storage.h"

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    inline FileDurabilityStore::FileDurabilityStore(std::filesystem::path path, size_t limit)
        : m_Store(std::move(path), limit)
    {
    }

    inline RetainedDescriptor FileDurabilityStore::Append(const ForwardRecord &record)
    {
        auto identity = MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                        record.m_AcceptedEpoch, record.m_Publication.m_Payload);
        if (identity != record.m_Identity || record.m_Expires <= UnixMilliseconds())
        {
            throw ProtocolError("Invalid durable publication");
        }
        auto payload = Codec::Encode(record.m_Publication, {true, UINT32_MAX});
        auto location = ToHex(identity);
        Writer writer(payload.size() + 16);
        writer.U64(record.m_Expires);
        WriteBuffer(writer, payload);
        if (auto previous = m_Store.Get(location); previous && !std::ranges::equal(*previous, writer.Data()))
        {
            throw ProtocolError("Durability identity conflicts with existing content");
        }
        m_Store.Put(location, writer.Take());
        return {identity, record.m_Publication.m_Topic, location, record.m_Expires, 0, Sha256(payload)};
    }

    inline Buffer FileDurabilityStore::Read(const RetainedDescriptor &descriptor)
    {
        if (descriptor.m_Location != ToHex(descriptor.m_Identity))
        {
            throw ProtocolError("Invalid durable location");
        }
        auto record = m_Store.Get(descriptor.m_Location);
        if (!record)
        {
            throw ProtocolError("Durable publication is unavailable");
        }
        Reader reader(*record);
        auto expiry = reader.U64();
        auto payload = ReadBuffer(reader);
        reader.Finish();
        if (expiry <= UnixMilliseconds() || expiry != descriptor.m_Expiry ||
            Sha256(payload) != descriptor.m_PayloadHash)
        {
            throw ProtocolError("Durable publication expired or descriptor differs");
        }
        return payload;
    }

    inline void FileDurabilityStore::Compact(uint64_t now)
    {
        for (const auto &key : m_Store.Keys())
        {
            auto value = m_Store.Get(key);
            Reader reader(*value);
            if (reader.U64() <= now)
            {
                m_Store.Erase(key);
            }
        }
    }
} // namespace eMQ::Server
