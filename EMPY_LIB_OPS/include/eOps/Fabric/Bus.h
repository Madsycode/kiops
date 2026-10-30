#pragma once
#include <memory>
#include <string>
#include <vector>
#include <future>
#include <cstdint>
#include <utility>
#include <optional>
#include <functional>
#include <eMQ/Client.h>
#include "Envelope.h"

// owner-thread messaging over one eMQ::Service: envelope publish/listen, request serving and
// correlated calls with timeouts. One global Receive listener dispatches (a) responses
// (MMQP_PUBREQ_TYPE_RES) by correlation ID to pending exchanges, (b) requests (REQ) to Serve
// handlers whose filter matches and (c) messages to Listen handlers whose filter matches.
// Requests use explicit nonzero correlation IDs allocated by the Bus. A broker terminal outcome
// (NO_RESPONDER, TIMEOUT) completes its exchange with that reason; a responder's error reason
// completes a Call, while a Gather keeps it as that responder's reply (kind "error") and goes on;
// late responses are ignored. Malformed envelopes are logged and dropped; a malformed request is
// answered with an error reason. Every handler runs on the owner thread (inside Poll).
//
// Ownership: handlers and pending exchanges may carry an `owner` tag (typically a module's
// `this`); Remove(owner) drops them without invoking them, so a module detaching before the
// BusModule leaves no dangling callbacks. Timeouts are driven by a sweep the Bus schedules from
// BusModule (or by CallSync's polling), not by timers owned by the calling module.
//
// Broker behaviour the Bus relies on (verified against eMQ::Broker):
// - publishing needs a REG profile; Expose registers it, and a wildcard REG (e.g. WorkloadFilter with
//   requests=true) authorises MSG and REQUEST publications on every concrete topic it matches
// - a subscription of QoS q only receives publications of QoS <= q: Listen/ListenRaw with qos 0 miss
//   QoS 1 publications
// - the broker keeps the latest QoS 1 publication of every registered topic for the profile expiry
//   (10 s here) and the last `retain` records of retained topics (1 day); late subscribers get them
// - an exchange also ends at the broker's request_timeout_ms (default 5000) and the client's
//   response_timeout_ms (default 10000) with TIMEOUT, whichever is first; raise both for long calls
// - a request nobody serves completes at once with NO_RESPONDER; a session's own responder
//   subscription receives its own requests

namespace eOps::Fabric
{
    struct BusOptions
    {
        std::string m_Source;                // Envelope::m_Source of outgoing messages
        uint32_t m_DefaultTimeoutMs = 10000; // Call/Gather timeout when 0 is passed
        size_t m_MaxPending = 1024;          // concurrent exchanges; beyond → immediate error reason
        JsonLimits m_Limits;
    };

    struct Inbound
    {
        std::string m_Topic;
        uint16_t m_CorrelationId = 0;
        uint8_t m_Qos = 1;
        Envelope m_Envelope;
    };

    class EMPY_OPS_API Bus
    {
    public:
        Bus(eMQ::Service &client, BusOptions options); // construct/use on the client's owner thread
        ~Bus();                                        // Close
        Bus(const Bus &) = delete;
        Bus &operator=(const Bus &) = delete;

        // REG profile for publishing on `topic` (requests = REQUEST publications allowed); cached and
        // idempotent; blocks ≤ the client operation timeout for a not-yet-registered topic
        eMQ::Reason Expose(const std::string &topic, bool requests = false, uint8_t retain = 0);
        std::future<eMQ::Reason> Publish(const std::string &topic, const Envelope &message, uint8_t qos = 1,
                                         uint8_t retain = 0);
        std::future<eMQ::Reason> PublishRaw(const std::string &topic, eCore::SharedBytes payload, uint8_t qos = 0);

        using Handler = std::function<void(const std::string &topic, const Envelope &)>;
        eMQ::Reason Listen(const std::string &filter, Handler handler, uint8_t qos = 1, const void *owner = nullptr);
        using RawHandler = std::function<void(const eMQ::PublishRequest &)>;
        eMQ::Reason ListenRaw(const std::string &filter, RawHandler handler, uint8_t qos = 0,
                              const void *owner = nullptr);

        using Responder = std::function<void(Envelope response, eMQ::Reason reason)>; // call once, owner thread
        using RequestHandler = std::function<void(const Inbound &, Responder)>;
        // subscribes `filter` as a responder; the handler may answer later (after Async work) through the
        // Responder; an unanswered Responder that is destroyed answers with an error reason
        eMQ::Reason Serve(const std::string &filter, RequestHandler handler, const void *owner = nullptr);

        using Replies = std::function<void(eMQ::Reason, std::vector<Envelope>)>; // exactly once, owner thread
        // REQUEST expecting up to `responses` replies (0 = all responders until the timeout); completes on
        // the last expected reply, on a broker terminal outcome, or at the timeout with what arrived. A
        // responder's error answer counts as its reply: an envelope of kind "error" {code, text, reply}. The
        // reason is success when any responder succeeded, else the last responder error
        void Gather(const std::string &topic, const Envelope &request, uint8_t responses, Replies replies,
                    uint32_t timeoutMs = 0, const void *owner = nullptr);
        using Reply = std::function<void(eMQ::Reason, std::optional<Envelope>)>; // exactly once, owner thread
        void Call(const std::string &topic, const Envelope &request, Reply reply, uint32_t timeoutMs = 0,
                  const void *owner = nullptr);

        // CLI helper: may only be called outside owner callbacks; polls the runtime through `poll` while waiting
        std::pair<eMQ::Reason, std::optional<Envelope>> CallSync(const std::string &topic, const Envelope &request,
                                                                 const std::function<void()> &poll,
                                                                 uint32_t timeoutMs = 0);

        // expire timed-out exchanges (called by BusModule::OnUpdate and its sweep timer)
        void Sweep();
        // drop every handler and pending exchange tagged with `owner` without invoking them
        void Remove(const void *owner);
        // completes pending exchanges with an error reason, removes listeners
        void Close();

        size_t Pending() const;
        const BusOptions &Options() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    // owns the runtime's Bus; attach it before any module that uses the Bus (detached last)
    class EMPY_OPS_API BusModule : public eCore::ImplModule<eMQ::Service, eMQ::Context>
    {
    public:
        explicit BusModule(BusOptions options);
        ~BusModule() override;

        Bus &Get(); // throws std::logic_error before OnAttach / after OnDetach

        void OnAttach(eMQ::Service *service, eMQ::Context *context) override;
        void OnUpdate(eMQ::Service *service, eMQ::Context *context) override;
        void OnDetach(eMQ::Service *service, eMQ::Context *context) override;

    private:
        BusOptions m_Options;
        std::unique_ptr<Bus> m_Bus;
    };

    // the Bus of an attached BusModule on `service` (throws std::logic_error when absent)
    EMPY_OPS_API Bus &BusOf(eMQ::Service &service);
} // namespace eOps::Fabric
