#pragma once
#include "Replication.h"
#include <eCore/Common/Storage.h>

namespace eMQ::Server
{
    struct FileDurabilityStore : DurabilityStore
    {
        explicit FileDurabilityStore(std::filesystem::path directory, size_t maxBytes = 256 * 1024 * 1024);
        RetainedDescriptor Append(const ForwardRecord &record) override;
        Buffer Read(const RetainedDescriptor &descriptor) override;
        void Compact(uint64_t now);

    private:
        FileStore m_Store;
    };
} // namespace eMQ::Server

#include "Storage.inl"
