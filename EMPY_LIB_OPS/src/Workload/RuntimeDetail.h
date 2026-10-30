#pragma once
#include <deque>
#include <chrono>
#include <future>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <functional>
#include <eMQ/Client.h>
#include <string_view>
#include <eOps/Workload/Runtime.h>

// private helpers shared by the workload runtimes (owner-thread use only)

namespace eOps::Workload::Detail
{
    // self re-arming owner-thread timer. Timers re-armed from a timer task are not owned by the
    // module, so every runtime stops its tickers explicitly in OnDetach.
    class Ticker
    {
    public:
        Ticker() = default;
        Ticker(const Ticker &) = delete;
        Ticker &operator=(const Ticker &) = delete;

        // period 0 leaves the ticker stopped; exceptions of `tick` are logged and the ticker keeps running
        void Start(eMQ::Service &service, uint32_t periodMs, std::function<void()> tick);
        void Stop();
        bool Active() const noexcept;

    private:
        void Arm();

        eMQ::Service *m_Service = nullptr;
        eCore::TimerId m_Timer = 0;
        uint32_t m_PeriodMs = 0;
        std::function<void()> m_Tick;
    };

    // outstanding publication futures, harvested without blocking the owner thread
    class Outbox
    {
    public:
        explicit Outbox(size_t limit = 4096);

        // `publish` produces the client's future; a synchronous exception counts as a failure
        void Send(const std::string &topic, const std::function<std::future<eMQ::Reason>()> &publish);
        void Harvest();
        // waits up to `timeout` for every outstanding publication (acknowledged on the network strand), then
        // harvests: a detaching runtime's final retained heartbeat reaches the broker before the session closes
        void Drain(std::chrono::milliseconds timeout);

        uint64_t Accepted() const noexcept;
        uint64_t Failed() const noexcept;
        size_t Outstanding() const noexcept;
        const std::string &LastError() const noexcept;

    private:
        struct Item
        {
            std::future<eMQ::Reason> m_Future;
            std::string m_Topic;
        };

        void Fail(const std::string &topic, const std::string &text);

        size_t m_Limit = 4096;
        std::deque<Item> m_Items;
        uint64_t m_Accepted = 0;
        uint64_t m_Failed = 0;
        std::string m_LastError;
    };

    // the Bus of the runtime's BusModule; nullptr when it is absent or already detached
    Fabric::Bus *FindBus(eMQ::Service *service) noexcept;
    // throws std::runtime_error naming `what` when `reason` is an error
    void Require(const eMQ::Reason &reason, const std::string &what);
    bool Failed(const eMQ::Reason &reason) noexcept;

    // checks the options shared by the inference and training runtimes and defaults the instance to
    // "<service>-<role>-<node|local>-0"; throws std::invalid_argument
    void Resolve(WorkloadOptions &options, std::string_view role);

    Json Number(std::optional<double> value); // null when absent
} // namespace eOps::Workload::Detail
