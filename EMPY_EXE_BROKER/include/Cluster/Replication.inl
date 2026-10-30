#pragma once
#include <Cluster/Replication.h>

namespace eMQ::Server
{
    using namespace eCore::MMQP;
    using eCore::Digest;
    using eCore::WireFrame;
    namespace ReplicationDetail
    {
        inline uint64_t Version(const VersionVector &vector, const std::string &origin)
        {
            auto found = vector.find(origin);
            return found == vector.end() ? 0 : found->second;
        }

        inline bool Same(const ProfileAssertion &left, const ProfileAssertion &right)
        {
            return left.m_Broker == right.m_Broker && left.m_Client == right.m_Client &&
                   Codec::Encode(ProfileRequest{1, {left.m_Profile}}) ==
                       Codec::Encode(ProfileRequest{1, {right.m_Profile}});
        }

        inline void CheckName(const std::string &name)
        {
            if (name.empty() || name.size() > 255 || !ValidUtf8(name))
            {
                throw std::invalid_argument("Profile identities must be 1..255 UTF-8 bytes");
            }
        }

        inline void CheckDot(const VersionDot &dot)
        {
            CheckName(dot.m_Origin);
            if (!dot.m_Version)
            {
                throw std::invalid_argument("Profile operation versions start at one");
            }
        }
    } // namespace ReplicationDetail
    using namespace ReplicationDetail;

    inline ObservedProfiles::ObservedProfiles(std::string replica, std::set<std::string> replicas, size_t budget)
        : m_Replica(std::move(replica)), m_Replicas(std::move(replicas)), m_Budget(budget)
    {
        if (!budget || !m_Replicas.contains(m_Replica))
        {
            throw std::invalid_argument("Invalid profile replica set");
        }
        for (const auto &member : m_Replicas)
        {
            CheckName(member);
        }
        m_Acknowledged.emplace(m_Replica, VersionVector{});
    }

    inline void ObservedProfiles::Seen(const VersionDot &dot)
    {
        CheckDot(dot);
        if (!m_Versions.contains(dot.m_Origin) && m_Versions.size() >= m_Budget)
        {
            throw ProtocolError("Profile origin budget exhausted", MMQP_REASON_BUSY);
        }
        auto &version = m_Versions[dot.m_Origin];
        if (dot.m_Version > version)
        {
            auto &gaps = m_Gaps[dot.m_Origin];
            if (gaps.size() >= m_Budget && !gaps.contains(dot.m_Version))
            {
                throw ProtocolError("Out-of-order profile budget exhausted", MMQP_REASON_BUSY);
            }
            gaps.insert(dot.m_Version);
            while (version != UINT64_MAX && gaps.erase(version + 1))
            {
                ++version;
            }
        }
        m_Acknowledged[m_Replica] = m_Versions;
    }

    inline bool ObservedProfiles::Retired(const VersionDot &dot) const
    {
        return dot.m_Version <= Version(m_Stable, dot.m_Origin);
    }

    inline bool ObservedProfiles::Removed(const VersionDot &dot) const
    {
        return std::any_of(m_Removes.begin(), m_Removes.end(),
                           [&](const auto &removal) { return removal.second.m_Observed.contains(dot); });
    }

    inline void ObservedProfiles::Add(const ProfileAssertion &assertion)
    {
        CheckDot(assertion.m_Dot);
        CheckName(assertion.m_Broker);
        CheckName(assertion.m_Client);
        if (assertion.m_Broker.empty() || assertion.m_Client.empty() || assertion.m_Profile.m_Topic.empty() ||
            assertion.m_Profile.m_Type == MMQP_PROREQ_TYPE_UNS)
        {
            throw std::invalid_argument("Invalid profile assertion");
        }
        Codec::Encode(ProfileRequest{1, {assertion.m_Profile}});
        if (m_Removes.contains(assertion.m_Dot))
        {
            throw ProtocolError("An operation dot cannot identify both an add and a removal");
        }
        if (auto found = m_Adds.find(assertion.m_Dot); found != m_Adds.end())
        {
            if (!Same(found->second, assertion))
            {
                throw ProtocolError("Conflicting profile operation identity");
            }
            return;
        }
        if (this->Retired(assertion.m_Dot))
        {
            return;
        }
        if (m_Entries >= m_Budget)
        {
            throw ProtocolError("Profile shard operation budget exhausted", MMQP_REASON_BUSY);
        }
        this->Seen(assertion.m_Dot);
        m_Adds.emplace(assertion.m_Dot, assertion);
        ++m_Entries;
    }

    inline void ObservedProfiles::Remove(const ProfileRemoval &removal)
    {
        CheckDot(removal.m_Dot);
        if (removal.m_Observed.empty() || removal.m_Observed.size() > m_Budget)
        {
            throw std::invalid_argument("Removal must name its observed assertion dots");
        }
        for (const auto &dot : removal.m_Observed)
        {
            CheckDot(dot);
        }
        if (removal.m_Observed.contains(removal.m_Dot) || m_Adds.contains(removal.m_Dot))
        {
            throw ProtocolError("An operation dot cannot identify both an add and a removal");
        }
        if (auto found = m_Removes.find(removal.m_Dot); found != m_Removes.end())
        {
            if (found->second.m_Observed != removal.m_Observed)
            {
                throw ProtocolError("Conflicting removal identity");
            }
            return;
        }
        if (this->Retired(removal.m_Dot))
        {
            return;
        }
        if (m_Entries >= m_Budget || removal.m_Observed.size() >= m_Budget - m_Entries)
        {
            throw ProtocolError("Profile shard operation budget exhausted", MMQP_REASON_BUSY);
        }
        this->Seen(removal.m_Dot);
        m_Removes.emplace(removal.m_Dot, removal);
        m_Entries += removal.m_Observed.size() + 1;
    }

    inline void ObservedProfiles::Merge(const ObservedProfiles &other)
    {
        if (m_Replicas != other.m_Replicas)
        {
            throw std::invalid_argument("Profile epochs must share a replica set");
        }
        auto merged = *this;
        for (auto it = merged.m_Adds.begin(); it != merged.m_Adds.end();)
        {
            if (other.Retired(it->first) && !other.m_Adds.contains(it->first))
            {
                it = merged.m_Adds.erase(it);
                --merged.m_Entries;
            }
            else
            {
                ++it;
            }
        }
        for (const auto &[dot, assertion] : other.m_Adds)
        {
            merged.Add(assertion);
        }
        for (const auto &[dot, removal] : other.m_Removes)
        {
            merged.Remove(removal);
        }
        for (const auto &[origin, version] : other.m_Stable)
        {
            merged.m_Stable[origin] = std::max(merged.m_Stable[origin], version);
        }
        for (const auto &[origin, version] : other.m_Versions)
        {
            merged.m_Versions[origin] = std::max(merged.m_Versions[origin], version);
        }
        for (auto &[origin, gaps] : merged.m_Gaps)
        {
            std::erase_if(gaps, [&](auto value) { return value <= merged.m_Versions[origin]; });
            auto &version = merged.m_Versions[origin];
            while (version != UINT64_MAX && gaps.erase(version + 1))
            {
                ++version;
            }
        }
        if (merged.m_Versions.size() > merged.m_Budget)
        {
            throw ProtocolError("Profile origin budget exhausted", MMQP_REASON_BUSY);
        }
        merged.m_Acknowledged[merged.m_Replica] = merged.m_Versions;
        *this = std::move(merged);
    }

    inline std::set<VersionDot> ObservedProfiles::Observe(std::string_view broker, std::string_view client,
                                                          std::string_view topic) const
    {
        std::set<VersionDot> observed;
        auto canonical = CanonicalTopic(topic);
        for (const auto &[dot, assertion] : m_Adds)
        {
            if (!this->Removed(dot) && assertion.m_Broker == broker && assertion.m_Client == client &&
                CanonicalTopic(assertion.m_Profile.m_Topic) == canonical)
            {
                observed.insert(dot);
            }
        }
        return observed;
    }

    inline std::vector<std::string> ObservedProfiles::Destinations(std::string_view topic, uint8_t qos,
                                                                   bool responders) const
    {
        if (qos > 1)
        {
            throw std::invalid_argument("Unsupported publication QoS");
        }
        ParseTopic(topic);
        std::set<std::string> brokers;
        for (const auto &[dot, assertion] : m_Adds)
        {
            const auto &profile = assertion.m_Profile;
            if (!this->Removed(dot) && profile.m_Type == MMQP_PROREQ_TYPE_SUB && profile.m_Qos >= qos &&
                (!responders || profile.m_Request) && TopicMatches(profile.m_Topic, topic))
            {
                brokers.insert(assertion.m_Broker);
            }
        }
        return {brokers.begin(), brokers.end()};
    }

    inline void ObservedProfiles::Acknowledge(std::string replica, VersionVector vector)
    {
        if (!m_Replicas.contains(replica) || replica == m_Replica)
        {
            throw std::invalid_argument("Only a remote member can acknowledge profile versions");
        }
        if (vector.size() > m_Budget)
        {
            throw std::invalid_argument("Replica vector exceeds the origin budget");
        }
        for (const auto &[origin, version] : vector)
        {
            CheckName(origin);
        }
        auto &acknowledged = m_Acknowledged[replica];
        for (const auto &[origin, version] : acknowledged)
        {
            if (Version(vector, origin) < version)
            {
                throw std::invalid_argument("Replica acknowledgement regressed");
            }
        }
        acknowledged = std::move(vector);
    }

    inline void ObservedProfiles::Compact()
    {
        if (m_Acknowledged.size() != m_Replicas.size())
        {
            return;
        }
        VersionVector stable;
        for (const auto &[origin, version] : m_Versions)
        {
            auto floor = version;
            for (const auto &[replica, vector] : m_Acknowledged)
            {
                floor = std::min(floor, Version(vector, origin));
            }
            stable[origin] = floor;
        }
        for (auto it = m_Removes.begin(); it != m_Removes.end();)
        {
            bool safe = it->first.m_Version <= Version(stable, it->first.m_Origin);
            for (const auto &dot : it->second.m_Observed)
            {
                safe = safe && dot.m_Version <= Version(stable, dot.m_Origin);
            }
            if (!safe)
            {
                ++it;
                continue;
            }
            for (const auto &dot : it->second.m_Observed)
            {
                m_Entries -= m_Adds.erase(dot);
            }
            m_Entries -= it->second.m_Observed.size() + 1;
            it = m_Removes.erase(it);
        }
        for (const auto &[origin, version] : stable)
        {
            m_Stable[origin] = std::max(m_Stable[origin], version);
        }
    }

    inline const VersionVector &ObservedProfiles::Versions() const
    {
        return m_Versions;
    }
    inline size_t ObservedProfiles::Tombstones() const
    {
        return m_Removes.size();
    }

    inline SessionQuorum::SessionQuorum(uint64_t epoch, std::set<std::string> replicas, size_t limit)
        : m_Epoch(epoch), m_Replicas(std::move(replicas)), m_Limit(limit)
    {
        if (!epoch || m_Replicas.empty() || !limit)
        {
            throw std::invalid_argument("Invalid session replica unit");
        }
    }

    inline Digest SessionQuorum::Propose(uint64_t version, Buffer snapshot)
    {
        if (!version || snapshot.size() > m_Limit || m_Committed == UINT64_MAX || version != m_Committed + 1)
        {
            throw ProtocolError("Invalid session snapshot version or size");
        }
        auto digest = Sha256(snapshot);
        if (m_Proposed > m_Committed)
        {
            if (version != m_Proposed || !SecureEqual(m_Digest, digest))
            {
                throw ProtocolError("An uncommitted snapshot is already pending");
            }
            return digest;
        }
        m_Proposed = version;
        m_Pending = std::move(snapshot);
        m_Digest = digest;
        m_Acks.clear();
        return digest;
    }

    inline bool SessionQuorum::Acknowledge(std::string_view replica, uint64_t epoch, uint64_t version,
                                           const Digest &digest)
    {
        if (epoch != m_Epoch || version != m_Proposed || !version || !m_Replicas.contains(std::string(replica)) ||
            !SecureEqual(digest, m_Digest))
        {
            throw ProtocolError("Invalid session replica acknowledgement", MMQP_REASON_UNAUTHORIZED);
        }
        m_Acks.emplace(replica);
        if (m_Acks.size() < this->Required())
        {
            return false;
        }
        if (m_Committed != m_Proposed)
        {
            m_Committed = m_Proposed;
            m_Snapshot = m_Pending;
        }
        return true;
    }

    inline uint64_t SessionQuorum::CommittedVersion() const
    {
        return m_Committed;
    }
    inline size_t SessionQuorum::Required() const
    {
        return m_Replicas.size() / 2 + 1;
    }
    inline const Buffer &SessionQuorum::Snapshot() const
    {
        if (!m_Committed)
        {
            throw std::logic_error("No session snapshot has committed");
        }
        return m_Snapshot;
    }
} // namespace eMQ::Server
