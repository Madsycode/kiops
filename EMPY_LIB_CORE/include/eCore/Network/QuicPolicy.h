#pragma once
#include <eCore/Network/Codecs.h>

namespace eCore::MMQP
{
    enum class QuicChannel : uint8_t
    {
        Control = 0,
        BestEffort = 1,
        Reliable = 2,
        Bulk = 3,
        Datagram = 4
    };

    struct EMPY_CORE_API QuicPlan
    {
        explicit QuicPlan(QuicPolicy policy = {MMQP_QUIC_STRATEGY_SINGLE, {}, {}});
        static QuicPolicy Negotiate(const QuicPolicy &requested, const QuicPolicy &limits);
        QuicChannel Select(const PacketSummary &packet, size_t frameSize, uint16_t datagramLimit,
                           size_t activeBulk) const;
        void Validate(QuicChannel channel, const PacketSummary &packet, size_t frameSize) const;
        const QuicPolicy &Policy() const;

    private:
        QuicPolicy m_Policy;
    };
} // namespace eCore::MMQP
