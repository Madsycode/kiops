#include <eOps/Fabric/Bus.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eOps/Common/Text.h>
#include <eCore/Common/Stdout.h>
#include <algorithm>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace eOps::Fabric
{
    using eCore::MMQP::MMQP_PUBREQ_TYPE_REQ;
    using eCore::MMQP::MMQP_PUBREQ_TYPE_RES;
    using eCore::MMQP::MMQP_REASON_BUSY;
    using eCore::MMQP::MMQP_REASON_INTERNAL_ERROR;
    using eCore::MMQP::MMQP_REASON_INVALID_TOPIC;
    using eCore::MMQP::MMQP_REASON_MALFORMED_PACKET;
    using eCore::MMQP::MMQP_REASON_NO_RESPONDER;
    using eCore::MMQP::MMQP_REASON_POLICY_VIOLATION;
    using eCore::MMQP::MMQP_REASON_TIMEOUT;
    using eCore::MMQP::MMQP_REASON_UNAUTHORIZED;
    using eCore::MMQP::MMQP_REASON_UNSPECIFIED;

    namespace
    {
        using Clock = std::chrono::steady_clock;

        // broker retention: every QoS 1 message is kept (depth 1) for its profile expiry, so
        // non-retained topics use a short expiry; retained topics keep `retain` records for a day
        constexpr uint32_t TRANSIENT_EXPIRY = 10;
        constexpr uint32_t RETAINED_EXPIRY = 86400;
        // upper bound on a blocking profile operation (the client fails it earlier on its own)
        constexpr auto PROFILE_WAIT = std::chrono::seconds(30);
        constexpr auto SWEEP_PERIOD = eCore::Milliseconds(20);
        // abandoned correlation IDs stay reserved until the client releases them or this elapses
        constexpr auto DRAIN_LIMIT = std::chrono::minutes(5);

        bool Failed(const eMQ::Reason &reason)
        {
            return reason.m_Code >= 0x80;
        }

        // reason of a client operation future; local exceptions become error reasons
        eMQ::Reason Resolve(std::future<eMQ::Reason> &future, Clock::duration wait)
        {
            if (!future.valid())
            {
                return {MMQP_REASON_INTERNAL_ERROR, "Client operation has no result"};
            }
            if (future.wait_for(wait) != std::future_status::ready)
            {
                return {MMQP_REASON_TIMEOUT, "Client operation did not complete"};
            }
            try
            {
                return future.get();
            }
            catch (const eCore::MMQP::ProtocolError &error)
            {
                return {error.m_Reason >= 0x80 ? error.m_Reason : MMQP_REASON_UNSPECIFIED, error.what()};
            }
            catch (const std::exception &error)
            {
                return {MMQP_REASON_UNSPECIFIED, error.what()};
            }
        }

        std::future<eMQ::Reason> Ready(eMQ::Reason reason)
        {
            std::promise<eMQ::Reason> promise;
            promise.set_value(std::move(reason));
            return promise.get_future();
        }

        // canonical profile text, its matching filter (shared prefix removed) and share group
        struct FilterKey
        {
            std::string m_Key;
            std::string m_Match;
            std::string m_Group;
        };

        std::optional<FilterKey> ParseFilter(std::string_view filter)
        {
            try
            {
                auto canonical = eCore::MMQP::CanonicalTopic(filter);
                auto pattern = eCore::MMQP::ParseTopic(canonical, true);
                return FilterKey{std::move(canonical), std::move(pattern.m_Filter), std::move(pattern.m_Group)};
            }
            catch (const std::exception &)
            {
                return std::nullopt;
            }
        }

        // the most specific filter wins (same ordering as the client's profile table)
        size_t Specificity(const std::string &filter)
        {
            auto wildcard = filter.find_first_of("+#");
            return wildcard == std::string::npos ? filter.size() + 256 : wildcard;
        }

        Envelope ErrorEnvelope(const eMQ::Reason &reason)
        {
            Envelope envelope;
            envelope.m_Kind = "error";
            envelope.m_Body = Json::Object{{"reason", static_cast<uint32_t>(reason.m_Code)}, {"text", reason.m_Text}};
            return envelope;
        }

        // shared between the Bus and the Responders it hands out; Responders may be dropped on any thread
        struct Link
        {
            struct State;

            std::mutex m_Mutex;
            eMQ::Service *m_Client = nullptr; // null once the Bus is closed
            std::string m_Source;
            std::set<State *> m_Open;

            // one inbound request awaiting its single answer
            struct State
            {
                std::shared_ptr<Link> m_Link;
                std::string m_Topic;
                uint16_t m_CorrelationId = 0;
                uint8_t m_Qos = 1;
                bool m_Done = false;

                State(std::shared_ptr<Link> link, std::string topic, uint16_t correlationId, uint8_t qos)
                    : m_Link(std::move(link)), m_Topic(std::move(topic)), m_CorrelationId(correlationId), m_Qos(qos)
                {
                    std::lock_guard lock(m_Link->m_Mutex);
                    m_Link->m_Open.insert(this);
                }

                ~State()
                {
                    this->Answer({}, {MMQP_REASON_INTERNAL_ERROR, "Request handler dropped its responder"});
                    std::lock_guard lock(m_Link->m_Mutex);
                    m_Link->m_Open.erase(this);
                }

                // false when already answered
                bool Answer(Envelope response, eMQ::Reason reason)
                {
                    std::lock_guard lock(m_Link->m_Mutex);
                    if (m_Done)
                    {
                        return false;
                    }
                    m_Done = true;
                    m_Link->m_Open.erase(this);
                    if (m_Link->m_Client)
                    {
                        m_Link->Send(*this, std::move(response), std::move(reason));
                    }
                    return true;
                }
            };

            // caller holds m_Mutex
            void Send(const State &state, Envelope response, eMQ::Reason reason)
            {
                if (response.m_Kind.empty())
                {
                    response = Failed(reason) ? ErrorEnvelope(reason) : Envelope{"response", {}, 0, {}, {}};
                }
                if (response.m_Source.empty())
                {
                    response.m_Source = m_Source;
                }
                eCore::SharedBytes payload;
                try
                {
                    payload = response.Encode();
                }
                catch (const std::exception &error)
                {
                    reason = {MMQP_REASON_INTERNAL_ERROR, std::string("Response is not encodable: ") + error.what()};
                    payload = ErrorEnvelope(reason).Encode();
                }
                try
                {
                    // an error answer always carries a payload, so requesters never mistake it for the
                    // broker's terminal (empty) NO_RESPONDER/TIMEOUT outcome of the whole exchange
                    (void)m_Client->Respond(state.m_Topic, state.m_CorrelationId, std::move(payload), std::move(reason),
                                            state.m_Qos);
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("Bus: response on {} failed: {}", state.m_Topic, error.what());
                }
            }

            // answers every open request and detaches from the client
            void Close()
            {
                std::lock_guard lock(m_Mutex);
                for (auto *state : m_Open)
                {
                    state->m_Done = true;
                    if (m_Client)
                    {
                        this->Send(*state, {}, {MMQP_REASON_UNSPECIFIED, "Responder bus closed"});
                    }
                }
                m_Open.clear();
                m_Client = nullptr;
            }
        };
    } // namespace

    struct Bus::Impl
    {
        struct Exposure
        {
            bool m_Requests = false;
            uint8_t m_Retain = 0;
        };

        // one broker subscription shared by every Listen/Serve on the same filter
        struct Subscription
        {
            std::string m_Match;
            std::string m_Group;
            uint8_t m_Qos = 0;
            bool m_Responder = false; // subscribed with responder=true
            size_t m_Listeners = 0;
            size_t m_Servers = 0;

            size_t Users() const noexcept
            {
                return m_Listeners + m_Servers;
            }
        };

        struct Listener
        {
            FilterKey m_Filter;
            std::shared_ptr<Handler> m_Handler;
            std::shared_ptr<RawHandler> m_Raw;
            const void *m_Owner = nullptr;
        };

        struct Server
        {
            FilterKey m_Filter;
            std::shared_ptr<RequestHandler> m_Handler;
            const void *m_Owner = nullptr;
            size_t m_Specificity = 0;
        };

        struct Exchange
        {
            std::string m_Topic;
            uint8_t m_Expected = 1;  // 0 = every responder until the deadline
            uint8_t m_Remaining = 1; // mirror of the client's own count (0 = until a terminal outcome)
            bool m_Single = false;
            std::vector<Envelope> m_Replies;
            std::optional<Envelope> m_Error;
            size_t m_Dropped = 0;
            size_t m_Failures = 0;                // gather: replies that are responder errors
            std::optional<eMQ::Reason> m_Failure; // gather: the last responder error
            Replies m_Gather;
            Reply m_Reply;
            Clock::time_point m_Deadline;
            uint32_t m_TimeoutMs = 0;
            const void *m_Owner = nullptr;
            std::future<eMQ::Reason> m_Admission;
        };

        struct Drain
        {
            uint8_t m_Remaining = 0;
            Clock::time_point m_Expiry;
        };

        eMQ::Service &m_Client;
        BusOptions m_Options;
        std::shared_ptr<Link> m_Link = std::make_shared<Link>();
        std::map<std::string, Exposure> m_Exposed;
        std::map<std::string, Subscription> m_Subscriptions;
        std::map<uint64_t, Listener> m_Listeners;
        std::map<uint64_t, Server> m_Servers;
        std::map<uint16_t, Exchange> m_Pending;
        std::map<uint16_t, Drain> m_Draining;
        std::vector<eCore::ListenerId> m_Events;
        eCore::TimerId m_Timer = 0;
        std::string m_Session;
        uint64_t m_NextHandler = 1;
        uint16_t m_NextId = 1;
        bool m_Closed = false;

        Impl(eMQ::Service &client, BusOptions options) : m_Client(client), m_Options(std::move(options))
        {
            if (!m_Options.m_DefaultTimeoutMs || !m_Options.m_MaxPending || m_Options.m_MaxPending > UINT16_MAX)
            {
                throw std::invalid_argument("Bus needs a positive default timeout and 1..65535 pending exchanges");
            }
            m_Link->m_Client = &m_Client;
            m_Link->m_Source = m_Options.m_Source;
        }

        void Start()
        {
            m_Session = m_Client.Connected() ? m_Client.Identifier() : std::string();
            m_Events.push_back(
                m_Client.On<eMQ::Receive>([this](const eMQ::Receive &event) { this->OnReceive(event); }));
            m_Events.push_back(
                m_Client.On<eMQ::Connect>([this](const eMQ::Connect &event) { this->OnConnect(event); }));
            m_Events.push_back(
                m_Client.On<eMQ::Disconnect>([this](const eMQ::Disconnect &event) { this->OnDisconnect(event); }));
            this->ScheduleSweep();
        }

        void ScheduleSweep()
        {
            m_Timer = m_Client.Schedule(SWEEP_PERIOD, eCore::Task(
                                                          [this]
                                                          {
                                                              m_Timer = 0;
                                                              this->Sweep();
                                                              if (!m_Closed)
                                                              {
                                                                  this->ScheduleSweep();
                                                              }
                                                          }));
        }

        Envelope Stamp(const Envelope &message) const
        {
            Envelope stamped = message;
            if (stamped.m_Source.empty())
            {
                stamped.m_Source = m_Options.m_Source;
            }
            return stamped;
        }

        // -- profiles

        bool Satisfied(const std::string &key, bool requests, uint8_t retain) const
        {
            auto fits = [&](const Exposure &exposure)
            { return (!requests || exposure.m_Requests) && (!retain || exposure.m_Retain == retain); };
            if (auto found = m_Exposed.find(key); found != m_Exposed.end() && fits(found->second))
            {
                return true;
            }
            // a wildcard registration (e.g. <r>/workload/<service>/+) authorises its concrete topics
            for (const auto &[filter, exposure] : m_Exposed)
            {
                if (fits(exposure) && IsFilter(filter) && FilterCovers(filter, key))
                {
                    return true;
                }
            }
            return false;
        }

        eMQ::Reason Expose(const std::string &topic, bool requests, uint8_t retain)
        {
            if (m_Closed)
            {
                return {MMQP_REASON_UNSPECIFIED, "Bus is closed"};
            }
            auto filter = ParseFilter(topic);
            if (!filter || !filter->m_Group.empty())
            {
                return {MMQP_REASON_INVALID_TOPIC, "Invalid publication profile topic: " + topic};
            }
            if (this->Satisfied(filter->m_Key, requests, retain))
            {
                return {};
            }
            Exposure merged{requests, retain};
            if (auto found = m_Exposed.find(filter->m_Key); found != m_Exposed.end())
            {
                merged.m_Requests = merged.m_Requests || found->second.m_Requests;
                merged.m_Retain = retain ? retain : found->second.m_Retain;
            }
            auto future = merged.m_Retain
                              ? m_Client.Register(filter->m_Key, 1, RETAINED_EXPIRY, merged.m_Retain, merged.m_Requests)
                              : m_Client.Register(filter->m_Key, 0, TRANSIENT_EXPIRY, 1, merged.m_Requests);
            auto reason = Resolve(future, PROFILE_WAIT);
            if (!Failed(reason))
            {
                m_Exposed[filter->m_Key] = merged;
            }
            return reason;
        }

        // the publication was rejected locally for a missing profile: forget cached exposures of `topic`
        void Forget(const std::string &topic)
        {
            std::erase_if(m_Exposed,
                          [&](const auto &entry) { return entry.first == topic || FilterCovers(entry.first, topic); });
        }

        static void Count(Subscription &subscription, bool responder)
        {
            ++(responder ? subscription.m_Servers : subscription.m_Listeners);
        }

        eMQ::Reason Acquire(const FilterKey &filter, uint8_t qos, bool responder)
        {
            auto found = m_Subscriptions.find(filter.m_Key);
            if (found != m_Subscriptions.end() && found->second.m_Qos >= qos &&
                (!responder || found->second.m_Responder))
            {
                Count(found->second, responder);
                return {};
            }
            uint8_t level = qos;
            bool responding = responder;
            if (found != m_Subscriptions.end())
            {
                level = std::max(level, found->second.m_Qos);
                responding = responding || found->second.m_Responder;
            }
            auto future = m_Client.Subscribe(filter.m_Key, level, responding);
            auto reason = Resolve(future, PROFILE_WAIT);
            if (Failed(reason))
            {
                return reason;
            }
            auto &subscription = m_Subscriptions[filter.m_Key];
            subscription.m_Match = filter.m_Match;
            subscription.m_Group = filter.m_Group;
            subscription.m_Qos = level;
            subscription.m_Responder = responding;
            Count(subscription, responder);
            return reason;
        }

        // drops one user; a filter left with listeners only stops being a responder, so the broker
        // no longer routes requests to this session for it
        void Release(const std::string &key, bool responder)
        {
            auto found = m_Subscriptions.find(key);
            if (found == m_Subscriptions.end())
            {
                return;
            }
            auto &subscription = found->second;
            auto &count = responder ? subscription.m_Servers : subscription.m_Listeners;
            if (count)
            {
                --count;
            }
            if (subscription.m_Responder && !subscription.m_Servers && subscription.m_Listeners)
            {
                try
                {
                    (void)m_Client.Subscribe(key, subscription.m_Qos, false);
                    subscription.m_Responder = false;
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("Bus: responder release of {} failed: {}", key, error.what());
                }
            }
        }

        // Unsubscribe idle filters. The broker also removes concrete subscriptions a wildcard
        // unsubscription covers, so an idle wildcard stays while it covers an active concrete one.
        void ReleaseIdle()
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (auto it = m_Subscriptions.begin(); it != m_Subscriptions.end(); ++it)
                {
                    if (it->second.Users())
                    {
                        continue;
                    }
                    const std::string key = it->first;
                    const Subscription idle = it->second;
                    // concrete subscriptions (other than this one) the broker would drop along with it
                    auto covered = [&](const std::pair<const std::string, Subscription> &entry)
                    {
                        return entry.first != key && !IsFilter(entry.second.m_Match) &&
                               entry.second.m_Group == idle.m_Group &&
                               eCore::MMQP::MatchTopic(idle.m_Match, entry.second.m_Match);
                    };
                    if (std::any_of(m_Subscriptions.begin(), m_Subscriptions.end(),
                                    [&](const auto &entry) { return entry.second.Users() && covered(entry); }))
                    {
                        continue;
                    }
                    try
                    {
                        (void)m_Client.Unsubscribe(key);
                    }
                    catch (const std::exception &error)
                    {
                        EMPY_WARN("Bus: unsubscribe {} failed: {}", key, error.what());
                    }
                    std::erase_if(m_Subscriptions,
                                  [&](const auto &entry) { return entry.first == key || covered(entry); });
                    changed = true;
                    break;
                }
            }
        }

        // -- inbound dispatch (owner thread, inside Poll)

        void OnReceive(const eMQ::Receive &event)
        {
            if (m_Closed)
            {
                return;
            }
            const auto &message = event.m_Message;
            if (message.m_Type == MMQP_PUBREQ_TYPE_RES)
            {
                this->OnResponse(message);
            }
            else if (message.m_Type == MMQP_PUBREQ_TYPE_REQ)
            {
                this->OnRequest(message);
            }
            else
            {
                this->OnMessage(message);
            }
        }

        void OnMessage(const eMQ::PublishRequest &message)
        {
            std::vector<uint64_t> targets;
            for (const auto &[id, listener] : m_Listeners)
            {
                if (eCore::MMQP::MatchTopic(listener.m_Filter.m_Match, message.m_Topic))
                {
                    targets.push_back(id);
                }
            }
            std::optional<Envelope> envelope;
            bool decoded = false;
            for (auto id : targets)
            {
                auto found = m_Listeners.find(id); // a handler may remove others (or Close)
                if (found == m_Listeners.end())
                {
                    continue;
                }
                try
                {
                    if (found->second.m_Raw)
                    {
                        auto handler = found->second.m_Raw;
                        (*handler)(message);
                        continue;
                    }
                    if (!decoded)
                    {
                        decoded = true;
                        try
                        {
                            envelope = Envelope::Decode(message.m_Payload, m_Options.m_Limits);
                        }
                        catch (const std::exception &error)
                        {
                            EMPY_WARN("Bus: dropped malformed envelope on {}: {}", message.m_Topic, error.what());
                        }
                    }
                    if (envelope)
                    {
                        auto handler = found->second.m_Handler;
                        (*handler)(message.m_Topic, *envelope);
                    }
                }
                catch (const std::exception &error)
                {
                    EMPY_WARN("Bus: listener for {} failed: {}", message.m_Topic, error.what());
                }
            }
        }

        void OnRequest(const eMQ::PublishRequest &message)
        {
            auto state = std::make_shared<Link::State>(m_Link, message.m_Topic, message.m_CorrelationId, message.m_Qos);
            const Server *best = nullptr;
            for (const auto &[id, server] : m_Servers)
            {
                if (eCore::MMQP::MatchTopic(server.m_Filter.m_Match, message.m_Topic) &&
                    (!best || server.m_Specificity > best->m_Specificity))
                {
                    best = &server;
                }
            }
            if (!best)
            {
                state->Answer({}, {MMQP_REASON_NO_RESPONDER, "No request handler for " + message.m_Topic});
                return;
            }
            auto handler = best->m_Handler;
            Inbound inbound;
            inbound.m_Topic = message.m_Topic;
            inbound.m_CorrelationId = message.m_CorrelationId;
            inbound.m_Qos = message.m_Qos;
            try
            {
                inbound.m_Envelope = Envelope::Decode(message.m_Payload, m_Options.m_Limits);
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("Bus: malformed request on {}: {}", message.m_Topic, error.what());
                state->Answer(
                    {}, {MMQP_REASON_MALFORMED_PACKET, std::string("Malformed request envelope: ") + error.what()});
                return;
            }
            Responder responder = [state](Envelope response, eMQ::Reason reason)
            {
                if (!state->Answer(std::move(response), std::move(reason)))
                {
                    EMPY_WARN("Bus: request on {} was already answered", state->m_Topic);
                }
            };
            try
            {
                (*handler)(inbound, std::move(responder));
            }
            catch (const std::exception &error)
            {
                state->Answer({}, {MMQP_REASON_INTERNAL_ERROR, std::string("Request handler failed: ") + error.what()});
            }
        }

        void OnResponse(const eMQ::PublishRequest &message)
        {
            const auto id = message.m_CorrelationId;
            const bool terminal = message.m_Payload.Empty() && (message.m_Reason.m_Code == MMQP_REASON_NO_RESPONDER ||
                                                                message.m_Reason.m_Code == MMQP_REASON_TIMEOUT);
            if (auto drain = m_Draining.find(id); drain != m_Draining.end())
            {
                if (terminal || (drain->second.m_Remaining && --drain->second.m_Remaining == 0))
                {
                    m_Draining.erase(drain);
                }
                return;
            }
            auto found = m_Pending.find(id);
            if (found == m_Pending.end() || found->second.m_Topic != message.m_Topic)
            {
                return; // late or foreign response
            }
            auto &exchange = found->second;
            const bool released = terminal || (exchange.m_Remaining && --exchange.m_Remaining == 0);
            std::optional<eMQ::Reason> outcome;
            if (terminal)
            {
                // a gather of every responder normally ends at the broker deadline
                bool complete = message.m_Reason.m_Code == MMQP_REASON_TIMEOUT && !exchange.m_Single &&
                                !exchange.m_Expected && !exchange.m_Replies.empty();
                outcome = complete ? Gathered(exchange) : message.m_Reason;
            }
            else
            {
                std::optional<Envelope> envelope;
                try
                {
                    envelope = Envelope::Decode(message.m_Payload, m_Options.m_Limits);
                }
                catch (const std::exception &error)
                {
                    ++exchange.m_Dropped;
                    EMPY_WARN("Bus: dropped malformed response on {}: {}", message.m_Topic, error.what());
                }
                if (Failed(message.m_Reason) && exchange.m_Single)
                {
                    auto reason = message.m_Reason;
                    if (reason.m_Text.empty() && envelope)
                    {
                        reason.m_Text = envelope->m_Body.GetString("text");
                    }
                    exchange.m_Error = std::move(envelope);
                    outcome = std::move(reason);
                }
                else if (Failed(message.m_Reason))
                {
                    // an application error is one responder's outcome (as in eMQ::Service): the gather goes on and
                    // reports it as that responder's "error" reply {code, text, reply}
                    auto reason = message.m_Reason;
                    if (reason.m_Text.empty() && envelope)
                    {
                        reason.m_Text = envelope->m_Body.GetString("text");
                    }
                    Envelope failure;
                    failure.m_Kind = "error";
                    failure.m_Source = envelope ? envelope->m_Source : std::string();
                    failure.m_Time = envelope ? envelope->m_Time : NowMillis();
                    failure.m_Body = Json::MakeObject({{"code", static_cast<uint32_t>(reason.m_Code)},
                                                       {"text", reason.m_Text},
                                                       {"reply", envelope ? envelope->m_Body : Json()}});
                    exchange.m_Replies.push_back(std::move(failure));
                    ++exchange.m_Failures;
                    exchange.m_Failure = std::move(reason);
                    if (exchange.m_Expected && exchange.m_Replies.size() >= exchange.m_Expected)
                    {
                        outcome = Gathered(exchange);
                    }
                }
                else if (envelope)
                {
                    exchange.m_Replies.push_back(std::move(*envelope));
                    if (exchange.m_Single || (exchange.m_Expected && exchange.m_Replies.size() >= exchange.m_Expected))
                    {
                        outcome = exchange.m_Single ? eMQ::Reason{} : Gathered(exchange);
                    }
                }
                if (!outcome && released)
                {
                    // the client closed the exchange: nothing else arrives for it
                    outcome = exchange.m_Dropped ? eMQ::Reason{MMQP_REASON_MALFORMED_PACKET, "Malformed response"}
                                                 : Gathered(exchange);
                }
            }
            if (outcome)
            {
                this->Complete(id, std::move(*outcome), released);
            }
        }

        void OnConnect(const eMQ::Connect &event)
        {
            if (event.m_Resumed || event.m_Identifier == m_Session)
            {
                return;
            }
            // a fresh session: the client dropped its profiles and outstanding requests
            m_Session = event.m_Identifier;
            m_Draining.clear();
            this->FailAll({MMQP_REASON_UNSPECIFIED, "MMQP session was replaced"});
            for (const auto &[topic, exposure] : m_Exposed)
            {
                (void)(exposure.m_Retain
                           ? m_Client.Register(topic, 1, RETAINED_EXPIRY, exposure.m_Retain, exposure.m_Requests)
                           : m_Client.Register(topic, 0, TRANSIENT_EXPIRY, 1, exposure.m_Requests));
            }
            std::erase_if(m_Subscriptions, [](const auto &entry) { return !entry.second.Users(); });
            for (const auto &[filter, subscription] : m_Subscriptions)
            {
                (void)m_Client.Subscribe(filter, subscription.m_Qos, subscription.m_Responder);
            }
        }

        void OnDisconnect(const eMQ::Disconnect &event)
        {
            if (event.m_Reconnecting)
            {
                return; // a resumed session keeps its exchanges
            }
            auto code = Failed(event.m_Reason) ? event.m_Reason.m_Code : MMQP_REASON_UNSPECIFIED;
            this->FailAll({code, "MMQP session closed: " + event.m_Reason.m_Text});
        }

        // -- exchanges

        uint16_t Allocate()
        {
            for (uint32_t attempt = 0; attempt < UINT16_MAX; ++attempt)
            {
                uint16_t id = m_NextId;
                m_NextId = m_NextId == UINT16_MAX ? 1 : static_cast<uint16_t>(m_NextId + 1);
                if (!m_Pending.contains(id) && !m_Draining.contains(id))
                {
                    return id;
                }
            }
            return 0;
        }

        void Start(const std::string &topic, const Envelope &request, uint8_t responses, bool single, Replies gather,
                   Reply reply, uint32_t timeoutMs, const void *owner)
        {
            Exchange exchange;
            exchange.m_Topic = topic;
            exchange.m_Expected = responses;
            exchange.m_Remaining = responses;
            exchange.m_Single = single;
            exchange.m_Gather = std::move(gather);
            exchange.m_Reply = std::move(reply);
            exchange.m_Owner = owner;
            exchange.m_TimeoutMs = timeoutMs ? timeoutMs : m_Options.m_DefaultTimeoutMs;
            auto fail = [&](eMQ::Reason reason) { this->Finish(exchange, std::move(reason)); };
            if (m_Closed)
            {
                return fail({MMQP_REASON_UNSPECIFIED, "Bus is closed"});
            }
            if (!ValidTopic(topic, false) || IsFilter(topic))
            {
                return fail({MMQP_REASON_INVALID_TOPIC, "Requests need a concrete topic: " + topic});
            }
            // responses carry the canonical topic ("a//b/" is answered on "a/b")
            exchange.m_Topic = eCore::MMQP::CanonicalTopic(topic);
            if (m_Pending.size() >= m_Options.m_MaxPending)
            {
                return fail({MMQP_REASON_BUSY, "Too many pending exchanges"});
            }
            auto exposed = this->Expose(exchange.m_Topic, true, 0);
            if (Failed(exposed))
            {
                return fail(std::move(exposed));
            }
            auto id = this->Allocate();
            if (!id)
            {
                return fail({MMQP_REASON_BUSY, "No free correlation ID"});
            }
            eCore::SharedBytes payload;
            try
            {
                payload = this->Stamp(request).Encode();
            }
            catch (const std::exception &error)
            {
                return fail({MMQP_REASON_MALFORMED_PACKET, std::string("Request is not encodable: ") + error.what()});
            }
            exchange.m_Admission = m_Client.Request(exchange.m_Topic, std::move(payload), responses, 1, id);
            exchange.m_Deadline = Clock::now() + std::chrono::milliseconds(exchange.m_TimeoutMs);
            m_Pending.emplace(id, std::move(exchange));
        }

        // a finished gather succeeds when any responder succeeded, else reports the last responder error
        static eMQ::Reason Gathered(const Exchange &exchange)
        {
            if (exchange.m_Failure && exchange.m_Failures >= exchange.m_Replies.size())
            {
                return *exchange.m_Failure;
            }
            return {};
        }

        eMQ::Reason Expired(const Exchange &exchange) const
        {
            if (!exchange.m_Single && !exchange.m_Expected && !exchange.m_Replies.empty())
            {
                return Gathered(exchange);
            }
            return {MMQP_REASON_TIMEOUT, "No response within " + std::to_string(exchange.m_TimeoutMs) + " ms"};
        }

        // removes the exchange, keeps its ID reserved while the client still tracks it, then reports
        void Complete(uint16_t id, eMQ::Reason reason, bool released)
        {
            auto found = m_Pending.find(id);
            if (found == m_Pending.end())
            {
                return;
            }
            Exchange exchange = std::move(found->second);
            m_Pending.erase(found);
            if (!released)
            {
                m_Draining[id] = Drain{exchange.m_Remaining, Clock::now() + DRAIN_LIMIT};
            }
            this->Finish(exchange, std::move(reason));
        }

        void Finish(Exchange &exchange, eMQ::Reason reason)
        {
            try
            {
                if (exchange.m_Single)
                {
                    std::optional<Envelope> envelope;
                    if (!exchange.m_Replies.empty())
                    {
                        envelope = std::move(exchange.m_Replies.front());
                    }
                    else
                    {
                        envelope = std::move(exchange.m_Error);
                    }
                    auto reply = std::move(exchange.m_Reply);
                    reply(std::move(reason), std::move(envelope));
                }
                else
                {
                    auto gather = std::move(exchange.m_Gather);
                    gather(std::move(reason), std::move(exchange.m_Replies));
                }
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("Bus: completion for {} failed: {}", exchange.m_Topic, error.what());
            }
        }

        void FailAll(const eMQ::Reason &reason)
        {
            std::vector<uint16_t> ids;
            for (const auto &[id, exchange] : m_Pending)
            {
                ids.push_back(id);
            }
            for (auto id : ids)
            {
                this->Complete(id, reason, true);
            }
        }

        void Sweep()
        {
            const auto now = Clock::now();
            std::vector<std::pair<uint16_t, eMQ::Reason>> rejected, expired;
            for (auto &[id, exchange] : m_Pending)
            {
                if (exchange.m_Admission.valid() &&
                    exchange.m_Admission.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
                {
                    auto admission = Resolve(exchange.m_Admission, std::chrono::seconds(0));
                    if (Failed(admission))
                    {
                        if (admission.m_Code == MMQP_REASON_UNAUTHORIZED)
                        {
                            this->Forget(exchange.m_Topic);
                        }
                        rejected.emplace_back(id, std::move(admission));
                        continue;
                    }
                }
                if (now >= exchange.m_Deadline)
                {
                    expired.emplace_back(id, this->Expired(exchange));
                }
            }
            for (auto &[id, reason] : rejected)
            {
                this->Complete(id, std::move(reason), true); // the client never tracked (or dropped) it
            }
            for (auto &[id, reason] : expired)
            {
                this->Complete(id, std::move(reason), false);
            }
            std::erase_if(m_Draining, [&](const auto &entry) { return entry.second.m_Expiry <= now; });
        }

        void Remove(const void *owner)
        {
            if (!owner)
            {
                return;
            }
            std::vector<std::pair<std::string, bool>> released; // filter, responder
            for (auto it = m_Listeners.begin(); it != m_Listeners.end();)
            {
                if (it->second.m_Owner == owner)
                {
                    released.emplace_back(it->second.m_Filter.m_Key, false);
                    it = m_Listeners.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            for (auto it = m_Servers.begin(); it != m_Servers.end();)
            {
                if (it->second.m_Owner == owner)
                {
                    released.emplace_back(it->second.m_Filter.m_Key, true);
                    it = m_Servers.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            for (auto it = m_Pending.begin(); it != m_Pending.end();)
            {
                if (it->second.m_Owner == owner)
                {
                    m_Draining[it->first] = Drain{it->second.m_Remaining, Clock::now() + DRAIN_LIMIT};
                    it = m_Pending.erase(it);
                }
                else
                {
                    ++it;
                }
            }
            for (const auto &[key, responder] : released)
            {
                this->Release(key, responder);
            }
            this->ReleaseIdle();
        }

        void Close()
        {
            if (m_Closed)
            {
                return;
            }
            m_Closed = true;
            m_Link->Close();
            if (m_Timer)
            {
                m_Client.CancelTimer(m_Timer);
                m_Timer = 0;
            }
            for (auto id : m_Events)
            {
                m_Client.Off(id);
            }
            m_Events.clear();
            m_Listeners.clear();
            m_Servers.clear();
            for (const auto &[filter, subscription] : m_Subscriptions)
            {
                try
                {
                    (void)m_Client.Unsubscribe(filter);
                }
                catch (const std::exception &)
                {
                }
            }
            m_Subscriptions.clear();
            this->FailAll({MMQP_REASON_UNSPECIFIED, "Bus closed"});
        }
    };

    Bus::Bus(eMQ::Service &client, BusOptions options) : m_Impl(std::make_unique<Impl>(client, std::move(options)))
    {
        m_Impl->Start();
    }

    Bus::~Bus()
    {
        try
        {
            m_Impl->Close();
        }
        catch (const std::exception &error)
        {
            EMPY_WARN("Bus: close failed: {}", error.what());
        }
    }

    eMQ::Reason Bus::Expose(const std::string &topic, bool requests, uint8_t retain)
    {
        return m_Impl->Expose(topic, requests, retain);
    }

    std::future<eMQ::Reason> Bus::Publish(const std::string &topic, const Envelope &message, uint8_t qos,
                                          uint8_t retain)
    {
        if (!ValidTopic(topic, false) || IsFilter(topic))
        {
            return Ready({MMQP_REASON_INVALID_TOPIC, "Publications need a concrete topic: " + topic});
        }
        if (retain && !qos)
        {
            return Ready({MMQP_REASON_POLICY_VIOLATION, "Retained publications need QoS 1"});
        }
        auto reason = m_Impl->Expose(topic, false, retain);
        if (Failed(reason))
        {
            return Ready(std::move(reason));
        }
        eCore::SharedBytes payload;
        try
        {
            payload = m_Impl->Stamp(message).Encode();
        }
        catch (const std::exception &error)
        {
            return Ready({MMQP_REASON_MALFORMED_PACKET, std::string("Envelope is not encodable: ") + error.what()});
        }
        return m_Impl->m_Client.Publish(topic, std::move(payload), qos);
    }

    std::future<eMQ::Reason> Bus::PublishRaw(const std::string &topic, eCore::SharedBytes payload, uint8_t qos)
    {
        if (!ValidTopic(topic, false) || IsFilter(topic))
        {
            return Ready({MMQP_REASON_INVALID_TOPIC, "Publications need a concrete topic: " + topic});
        }
        auto reason = m_Impl->Expose(topic, false, 0);
        if (Failed(reason))
        {
            return Ready(std::move(reason));
        }
        return m_Impl->m_Client.Publish(topic, std::move(payload), qos);
    }

    eMQ::Reason Bus::Listen(const std::string &filter, Handler handler, uint8_t qos, const void *owner)
    {
        if (!handler)
        {
            throw std::invalid_argument("Bus::Listen needs a handler");
        }
        if (m_Impl->m_Closed)
        {
            return {MMQP_REASON_UNSPECIFIED, "Bus is closed"};
        }
        auto key = ParseFilter(filter);
        if (!key || qos > 1)
        {
            return {MMQP_REASON_INVALID_TOPIC, "Invalid subscription filter or QoS: " + filter};
        }
        auto reason = m_Impl->Acquire(*key, qos, false);
        if (!Failed(reason))
        {
            m_Impl->m_Listeners.emplace(
                m_Impl->m_NextHandler++,
                Impl::Listener{*key, std::make_shared<Handler>(std::move(handler)), nullptr, owner});
        }
        return reason;
    }

    eMQ::Reason Bus::ListenRaw(const std::string &filter, RawHandler handler, uint8_t qos, const void *owner)
    {
        if (!handler)
        {
            throw std::invalid_argument("Bus::ListenRaw needs a handler");
        }
        if (m_Impl->m_Closed)
        {
            return {MMQP_REASON_UNSPECIFIED, "Bus is closed"};
        }
        auto key = ParseFilter(filter);
        if (!key || qos > 1)
        {
            return {MMQP_REASON_INVALID_TOPIC, "Invalid subscription filter or QoS: " + filter};
        }
        auto reason = m_Impl->Acquire(*key, qos, false);
        if (!Failed(reason))
        {
            m_Impl->m_Listeners.emplace(
                m_Impl->m_NextHandler++,
                Impl::Listener{*key, nullptr, std::make_shared<RawHandler>(std::move(handler)), owner});
        }
        return reason;
    }

    eMQ::Reason Bus::Serve(const std::string &filter, RequestHandler handler, const void *owner)
    {
        if (!handler)
        {
            throw std::invalid_argument("Bus::Serve needs a handler");
        }
        if (m_Impl->m_Closed)
        {
            return {MMQP_REASON_UNSPECIFIED, "Bus is closed"};
        }
        auto key = ParseFilter(filter);
        if (!key)
        {
            return {MMQP_REASON_INVALID_TOPIC, "Invalid responder filter: " + filter};
        }
        auto reason = m_Impl->Acquire(*key, 1, true);
        if (!Failed(reason))
        {
            auto specificity = Specificity(key->m_Match);
            m_Impl->m_Servers.emplace(
                m_Impl->m_NextHandler++,
                Impl::Server{*key, std::make_shared<RequestHandler>(std::move(handler)), owner, specificity});
        }
        return reason;
    }

    void Bus::Gather(const std::string &topic, const Envelope &request, uint8_t responses, Replies replies,
                     uint32_t timeoutMs, const void *owner)
    {
        if (!replies)
        {
            throw std::invalid_argument("Bus::Gather needs a completion");
        }
        m_Impl->Start(topic, request, responses, false, std::move(replies), nullptr, timeoutMs, owner);
    }

    void Bus::Call(const std::string &topic, const Envelope &request, Reply reply, uint32_t timeoutMs,
                   const void *owner)
    {
        if (!reply)
        {
            throw std::invalid_argument("Bus::Call needs a completion");
        }
        m_Impl->Start(topic, request, 1, true, nullptr, std::move(reply), timeoutMs, owner);
    }

    std::pair<eMQ::Reason, std::optional<Envelope>> Bus::CallSync(const std::string &topic, const Envelope &request,
                                                                  const std::function<void()> &poll, uint32_t timeoutMs)
    {
        if (!poll)
        {
            throw std::invalid_argument("Bus::CallSync needs a poll function");
        }
        struct Outcome
        {
            bool m_Done = false;
            eMQ::Reason m_Reason;
            std::optional<Envelope> m_Envelope;
        };
        auto outcome = std::make_shared<Outcome>();
        const void *tag = outcome.get();
        this->Call(
            topic, request,
            [outcome](eMQ::Reason reason, std::optional<Envelope> envelope)
            {
                outcome->m_Done = true;
                outcome->m_Reason = std::move(reason);
                outcome->m_Envelope = std::move(envelope);
            },
            timeoutMs, tag);
        try
        {
            while (!outcome->m_Done)
            {
                poll();
                m_Impl->Sweep();
            }
        }
        catch (...)
        {
            m_Impl->Remove(tag);
            throw;
        }
        return {std::move(outcome->m_Reason), std::move(outcome->m_Envelope)};
    }

    void Bus::Sweep()
    {
        m_Impl->Sweep();
    }

    void Bus::Remove(const void *owner)
    {
        m_Impl->Remove(owner);
    }

    void Bus::Close()
    {
        m_Impl->Close();
    }

    size_t Bus::Pending() const
    {
        return m_Impl->m_Pending.size();
    }

    const BusOptions &Bus::Options() const noexcept
    {
        return m_Impl->m_Options;
    }

    // -- BusModule

    BusModule::BusModule(BusOptions options) : m_Options(std::move(options))
    {
    }

    BusModule::~BusModule() = default;

    Bus &BusModule::Get()
    {
        if (!m_Bus)
        {
            throw std::logic_error("BusModule is not attached");
        }
        return *m_Bus;
    }

    void BusModule::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        m_Bus = std::make_unique<Bus>(*service, m_Options);
    }

    void BusModule::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
        if (m_Bus)
        {
            m_Bus->Sweep();
        }
    }

    void BusModule::OnDetach(eMQ::Service *, eMQ::Context *)
    {
        if (m_Bus)
        {
            m_Bus->Close();
            m_Bus.reset();
        }
    }

    Bus &BusOf(eMQ::Service &service)
    {
        auto *module = service.GetModule<BusModule>();
        if (!module)
        {
            throw std::logic_error("No BusModule is attached to this MMQP service");
        }
        return module->Get();
    }
} // namespace eOps::Fabric
