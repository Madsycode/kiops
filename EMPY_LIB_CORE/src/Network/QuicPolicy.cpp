#include <eCore/Network/QuicPolicy.h>

namespace eCore::MMQP
{
    QuicPlan::QuicPlan(QuicPolicy policy) : m_Policy(std::move(policy))
    {
        if (!m_Policy.m_Strategy)
        {
            m_Policy.m_Strategy = MMQP_QUIC_STRATEGY_SINGLE;
        }
        auto strategy = *m_Policy.m_Strategy;
        if (strategy != MMQP_QUIC_STRATEGY_SINGLE && strategy != MMQP_QUIC_STRATEGY_SEMANTIC &&
            strategy != MMQP_QUIC_STRATEGY_BULK)
        {
            throw ProtocolError("This transport does not provide an application-defined stream allocation strategy",
                                MMQP_REASON_POLICY_VIOLATION);
        }
        if (m_Policy.m_Bulk && (strategy != MMQP_QUIC_STRATEGY_BULK || !m_Policy.m_Bulk->m_Streams ||
                                m_Policy.m_Bulk->m_Streams > 32 || !m_Policy.m_Bulk->m_Threshold))
        {
            throw ProtocolError("Bulk policy requires SAS 3, 1..32 streams and a positive threshold",
                                MMQP_REASON_POLICY_VIOLATION);
        }
        if (strategy == MMQP_QUIC_STRATEGY_BULK && !m_Policy.m_Bulk)
        {
            m_Policy.m_Bulk = BulkPolicy{};
        }
        if (m_Policy.m_Datagram && !*m_Policy.m_Datagram)
        {
            throw ProtocolError("Datagram payload threshold must be positive", MMQP_REASON_POLICY_VIOLATION);
        }
    }

    QuicPolicy QuicPlan::Negotiate(const QuicPolicy &requested, const QuicPolicy &limits)
    {
        auto result = QuicPlan(requested).Policy();
        auto maximum = QuicPlan(limits).Policy();
        if (*maximum.m_Strategy == MMQP_QUIC_STRATEGY_SINGLE)
        {
            result.m_Strategy = MMQP_QUIC_STRATEGY_SINGLE;
        }
        else if (*result.m_Strategy == MMQP_QUIC_STRATEGY_BULK && *maximum.m_Strategy != MMQP_QUIC_STRATEGY_BULK)
        {
            result.m_Strategy = MMQP_QUIC_STRATEGY_SEMANTIC;
        }
        if (*result.m_Strategy != MMQP_QUIC_STRATEGY_BULK)
        {
            result.m_Bulk.reset();
        }
        else
        {
            result.m_Bulk->m_Streams = std::min(result.m_Bulk->m_Streams, maximum.m_Bulk->m_Streams);
            result.m_Bulk->m_Threshold = std::max(result.m_Bulk->m_Threshold, maximum.m_Bulk->m_Threshold);
        }
        if (result.m_Datagram && maximum.m_Datagram)
        {
            result.m_Datagram = std::min(*result.m_Datagram, *maximum.m_Datagram);
        }
        else
        {
            result.m_Datagram.reset();
        }
        return result;
    }

    QuicChannel QuicPlan::Select(const PacketSummary &packet, size_t frameSize, uint16_t datagramLimit,
                                 size_t activeBulk) const
    {
        if (!packet.Publication())
        {
            return QuicChannel::Control;
        }
        if (!packet.m_Qos && m_Policy.m_Datagram && packet.m_Payload <= *m_Policy.m_Datagram &&
            frameSize <= datagramLimit)
        {
            return QuicChannel::Datagram;
        }
        if (*m_Policy.m_Strategy == MMQP_QUIC_STRATEGY_SINGLE)
        {
            return QuicChannel::Control;
        }
        if (m_Policy.m_Bulk && packet.m_Payload >= m_Policy.m_Bulk->m_Threshold &&
            activeBulk < m_Policy.m_Bulk->m_Streams)
        {
            return QuicChannel::Bulk;
        }
        return packet.m_Qos ? QuicChannel::Reliable : QuicChannel::BestEffort;
    }

    void QuicPlan::Validate(QuicChannel channel, const PacketSummary &packet, size_t frameSize) const
    {
        auto publication = packet.Publication() ? &packet : nullptr;
        bool valid = false;
        switch (channel)
        {
        case QuicChannel::Control:
            valid = !publication || *m_Policy.m_Strategy == MMQP_QUIC_STRATEGY_SINGLE;
            break;
        case QuicChannel::BestEffort:
        case QuicChannel::Reliable:
            valid = publication && *m_Policy.m_Strategy != MMQP_QUIC_STRATEGY_SINGLE &&
                    publication->m_Qos == (channel == QuicChannel::Reliable ? 1 : 0);
            break;
        case QuicChannel::Bulk:
            valid = publication && m_Policy.m_Bulk && publication->m_Payload >= m_Policy.m_Bulk->m_Threshold;
            break;
        case QuicChannel::Datagram:
            valid = publication && !publication->m_Qos && m_Policy.m_Datagram &&
                    publication->m_Payload <= *m_Policy.m_Datagram && frameSize <= UINT16_MAX;
            break;
        }
        if (!valid)
        {
            throw ProtocolError("Packet arrived on an incompatible QUIC channel", MMQP_REASON_POLICY_VIOLATION);
        }
    }

    const QuicPolicy &QuicPlan::Policy() const
    {
        return m_Policy;
    }
} // namespace eCore::MMQP
