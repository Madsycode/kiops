#pragma once
#include "QuicPolicy.h"

namespace eCore::MMQP
{
    struct TransportOptions
    {
        uint32_t m_MaxPacket = 1024 * 1024;
        size_t m_MaxQueue = 4 * 1024 * 1024;
        size_t m_MaxQueueFrames = 4096;
        uint32_t m_Timeout = 5000;
        bool m_Tls = false;
        bool m_Quic = false;
        QuicPolicy m_QuicPolicy{MMQP_QUIC_STRATEGY_BULK, BulkPolicy{2, 65536}, 1200};
        bool m_Mutual = false;
        bool m_Backplane = false;
        std::string m_Certificate;
        std::string m_CertificateHash;
        std::string m_PrivateKey;
        std::string m_Authority;
        std::string m_ServerName;
    };

    struct EMPY_CORE_API Connection : std::enable_shared_from_this<Connection>
    {
        using Pointer = std::shared_ptr<Connection>;
        using Receive = std::function<void(std::span<const uint8_t>)>;
        using Closed = std::function<void(const std::string &)>;
        using Flushed = std::function<void()>;

        Receive m_OnReceive;
        Closed m_OnClose;
        // Runs after the frames of one transport read were delivered through onReceive,
        // so a receiver can commit them as one group before releasing their effects.
        Flushed m_OnFlush;
        virtual ~Connection() = default;
        virtual bool Send(WireFrame bytes) = 0;
        virtual void Close(bool drain = false) = 0;
        virtual void SetLimit(uint32_t limit) = 0;
        virtual size_t Queued() const = 0;
        virtual size_t Capacity() const = 0;
        virtual bool Secure() const = 0;
        virtual std::string PeerFingerprint() const
        {
            return {};
        }
        virtual bool Quic() const
        {
            return false;
        }
        virtual void Configure(const QuicPolicy &, bool)
        {
            throw ProtocolError("QUIC policy is not supported by this transport", MMQP_REASON_POLICY_VIOLATION);
        }
    };
} // namespace eCore::MMQP
