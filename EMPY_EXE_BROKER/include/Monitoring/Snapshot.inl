#pragma once
#include <Monitoring/Snapshot.h>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    namespace SnapshotDetail
    {
        // This writer only produces JSON; it is not a general parser or dependency.
        struct Json
        {
            std::string m_Value;
            std::vector<bool> m_First;

            void Separator()
            {
                if (!m_First.back())
                {
                    m_Value += ',';
                }
                m_First.back() = false;
            }

            void Quote(std::string_view text)
            {
                constexpr char hex[] = "0123456789abcdef";
                m_Value += '"';
                for (auto byte : text)
                {
                    auto c = static_cast<uint8_t>(byte);
                    if (c == '"' || c == '\\')
                    {
                        m_Value += '\\';
                        m_Value += byte;
                    }
                    else if (c < 32)
                    {
                        m_Value += "\\u00";
                        m_Value += hex[c >> 4];
                        m_Value += hex[c & 15];
                    }
                    else
                    {
                        m_Value += byte;
                    }
                }
                m_Value += '"';
            }

            void Key(std::string_view key)
            {
                this->Separator();
                this->Quote(key);
                m_Value += ':';
            }
            void Begin(char bracket)
            {
                m_Value += bracket;
                m_First.push_back(true);
            }
            void End(char bracket)
            {
                m_Value += bracket;
                m_First.pop_back();
            }
            void Object()
            {
                this->Separator();
                this->Begin('{');
            }
            void Object(std::string_view key)
            {
                this->Key(key);
                this->Begin('{');
            }
            void Array(std::string_view key)
            {
                this->Key(key);
                this->Begin('[');
            }
            void Number(std::string_view key, uint64_t number)
            {
                this->Key(key);
                m_Value += std::to_string(number);
            }
            void String(std::string_view key, std::string_view text)
            {
                this->Key(key);
                this->Quote(text);
            }
            void Boolean(std::string_view key, bool boolean)
            {
                this->Key(key);
                m_Value += boolean ? "true" : "false";
            }
            void Optional(std::string_view key, const std::optional<uint64_t> &number)
            {
                this->Key(key);
                m_Value += number ? std::to_string(*number) : "null";
            }

            void Traffic(const TrafficMetrics &metrics)
            {
                this->Number("receivedPackets", metrics.m_ReceivedPackets);
                this->Number("receivedBytes", metrics.m_ReceivedBytes);
                this->Number("sentPackets", metrics.m_SentPackets);
                this->Number("sentBytes", metrics.m_SentBytes);
            }
        };
    } // namespace SnapshotDetail
    using namespace SnapshotDetail;

    inline std::string SnapshotJson(const BrokerSnapshot &snapshot)
    {
        Json json;
        json.Begin('{');
        json.Number("schema", 1);
        json.String("identifier", snapshot.m_Identifier);
        json.String("instance", snapshot.m_Instance);
        json.String("address", snapshot.m_Address);
        json.Number("port", snapshot.m_Port);
        json.String("transport", snapshot.m_Transport);
        json.Number("collectedAt", snapshot.m_CollectedAt);
        json.Number("uptimeMs", snapshot.m_UptimeMs);
        json.Boolean("durable", snapshot.m_Durable);
        json.Boolean("authenticationRequired", snapshot.m_AuthenticationRequired);
        json.Optional("storedBytes", snapshot.m_StoredBytes);
        json.Object("metrics");
        json.Traffic(snapshot.m_Metrics);
        json.Number("accepted", snapshot.m_Metrics.m_Accepted);
        json.Number("publications", snapshot.m_Metrics.m_Publications);
        json.Number("duplicates", snapshot.m_Metrics.m_Duplicates);
        json.Number("rejected", snapshot.m_Metrics.m_Rejected);
        json.Number("forwarded", snapshot.m_Metrics.m_Forwarded);
        json.Number("retransmitted", snapshot.m_Metrics.m_Retransmitted);
        json.Number("sessions", snapshot.m_Metrics.m_Sessions);
        json.Number("online", snapshot.m_Metrics.m_Online);
        json.Number("retainedBytes", snapshot.m_Metrics.m_RetainedBytes);
        json.Number("pending", snapshot.m_Metrics.m_Pending);
        json.End('}');
        json.Object("limits");
        json.Number("sessions", snapshot.m_MaxSessions);
        json.Number("inflightPerSession", snapshot.m_MaxInflight);
        json.Number("retainedBytes", snapshot.m_MaxRetainedBytes);
        json.Number("storedBytes", snapshot.m_MaxStateBytes);
        json.End('}');
        json.Number("omittedSessions", snapshot.m_OmittedSessions);
        json.Array("sessions");
        for (const auto &session : snapshot.m_Sessions)
        {
            json.Object();
            json.String("identifier", session.m_Identifier);
            json.String("realm", session.m_Realm);
            json.String("transport", session.m_Transport);
            json.Boolean("online", session.m_Online);
            json.Boolean("authenticated", session.m_Authenticated);
            json.Boolean("resumable", session.m_Resumable);
            json.Boolean("owner", session.m_Owner);
            json.Number("observedAt", session.m_ObservedAt);
            json.Number("publications", session.m_Publications);
            json.Number("deliveries", session.m_Deliveries);
            json.Optional("idleMs", session.m_IdleMs);
            json.Optional("expiresInMs", session.m_ExpiresInMs);
            json.Number("keepaliveSeconds", session.m_KeepaliveSeconds);
            json.Number("subscriptions", session.m_Subscriptions);
            json.Number("registrations", session.m_Registrations);
            json.Number("responders", session.m_Responders);
            json.Number("requesters", session.m_Requesters);
            json.Number("pending", session.m_Pending);
            json.Number("pendingBytes", session.m_PendingBytes);
            json.Number("queuedBytes", session.m_QueuedBytes);
            json.Number("replay", session.m_Replay);
            json.Number("replayBytes", session.m_ReplayBytes);
            json.Number("requests", session.m_Requests);
            json.Number("responseRoutes", session.m_ResponseRoutes);
            json.Traffic(session);
            if (session.m_Lease)
            {
                json.Object("lease");
                json.String("fence", std::to_string(session.m_Lease->m_Fence));
                json.Number("remainingMs", session.m_Lease->m_RemainingMs);
                json.End('}');
            }
            else
            {
                json.Key("lease");
                json.m_Value += "null";
            }
            json.End('}');
        }
        json.End(']');
        if (snapshot.m_Cluster)
        {
            const auto &cluster = *snapshot.m_Cluster;
            json.Object("cluster");
            json.String("identifier", cluster.m_Identifier);
            json.String("epoch", std::to_string(cluster.m_Epoch));
            json.Number("shards", cluster.m_Shards);
            json.Number("replicas", cluster.m_Replicas);
            json.Number("fanout", cluster.m_Fanout);
            json.Number("pendingOperations", cluster.m_PendingOperations);
            json.Number("activeLeases", cluster.m_ActiveLeases);
            json.Number("quarantineMs", cluster.m_QuarantineMs);
            json.Array("peers");
            for (const auto &peer : cluster.m_Peers)
            {
                json.Object();
                json.String("identifier", peer.m_Identifier);
                json.String("host", peer.m_Host);
                json.Number("port", peer.m_Port);
                json.Boolean("connected", peer.m_Connected);
                json.Number("queuedBytes", peer.m_QueuedBytes);
                json.End('}');
            }
            json.End(']');
            json.End('}');
        }
        else
        {
            json.Key("cluster");
            json.m_Value += "null";
        }
        json.End('}');
        if (json.m_Value.size() > 16 * 1024 * 1024)
        {
            throw std::length_error("Telemetry snapshot exceeds 16 MiB");
        }
        return std::move(json.m_Value);
    }
} // namespace eMQ::Server
