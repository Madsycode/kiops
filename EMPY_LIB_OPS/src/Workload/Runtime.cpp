#include "RuntimeDetail.h"
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/Stdout.h>
#include <stdexcept>
#include <cmath>

namespace eOps::Workload::Detail
{
    // -- Ticker

    void Ticker::Start(eMQ::Service &service, uint32_t periodMs, std::function<void()> tick)
    {
        this->Stop();
        if (!periodMs || !tick)
        {
            return;
        }
        m_Service = &service;
        m_PeriodMs = periodMs;
        m_Tick = std::move(tick);
        this->Arm();
    }

    void Ticker::Stop()
    {
        if (m_Service && m_Timer)
        {
            m_Service->CancelTimer(m_Timer);
        }
        m_Service = nullptr;
        m_Timer = 0;
    }

    bool Ticker::Active() const noexcept
    {
        return m_Service != nullptr;
    }

    void Ticker::Arm()
    {
        auto fire = [this]
        {
            m_Timer = 0;
            try
            {
                m_Tick();
            }
            catch (const std::exception &error)
            {
                EMPY_WARN("Workload: periodic task failed: {}", error.what());
            }
            // the tick may have stopped (or restarted) the ticker
            if (m_Service && !m_Timer)
            {
                this->Arm();
            }
        };
        m_Timer = m_Service->Schedule(eCore::Milliseconds(m_PeriodMs), eCore::Task(std::move(fire)));
    }

    // -- Outbox

    Outbox::Outbox(size_t limit) : m_Limit(limit ? limit : 1)
    {
    }

    void Outbox::Send(const std::string &topic, const std::function<std::future<eMQ::Reason>()> &publish)
    {
        std::future<eMQ::Reason> future;
        try
        {
            future = publish();
        }
        catch (const std::exception &error)
        {
            this->Fail(topic, error.what());
            return;
        }
        if (!future.valid())
        {
            this->Fail(topic, "publication has no result");
            return;
        }
        if (m_Items.size() >= m_Limit)
        {
            this->Harvest();
            if (m_Items.size() >= m_Limit)
            {
                // unresolved for too long: stop tracking the oldest without blocking
                m_Items.pop_front();
            }
        }
        m_Items.push_back(Item{std::move(future), topic});
    }

    void Outbox::Harvest()
    {
        for (auto it = m_Items.begin(); it != m_Items.end();)
        {
            if (it->m_Future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            {
                ++it;
                continue;
            }
            try
            {
                auto reason = it->m_Future.get();
                if (Detail::Failed(reason))
                {
                    this->Fail(it->m_Topic, std::to_string(reason.m_Code) + " " + reason.m_Text);
                }
                else
                {
                    ++m_Accepted;
                }
            }
            catch (const std::exception &error)
            {
                this->Fail(it->m_Topic, error.what());
            }
            it = m_Items.erase(it);
        }
    }

    void Outbox::Drain(std::chrono::milliseconds timeout)
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        for (Item &item : m_Items)
        {
            if (item.m_Future.wait_until(deadline) != std::future_status::ready)
            {
                break;
            }
        }
        this->Harvest();
    }

    uint64_t Outbox::Accepted() const noexcept
    {
        return m_Accepted;
    }

    uint64_t Outbox::Failed() const noexcept
    {
        return m_Failed;
    }

    size_t Outbox::Outstanding() const noexcept
    {
        return m_Items.size();
    }

    const std::string &Outbox::LastError() const noexcept
    {
        return m_LastError;
    }

    void Outbox::Fail(const std::string &topic, const std::string &text)
    {
        // the first failure and then every 1000th, so a lost broker does not flood the log
        if (m_Failed++ % 1000 == 0)
        {
            EMPY_WARN("Workload: publication on {} failed: {} ({} failures)", topic, text, m_Failed);
        }
        m_LastError = text;
    }

    // -- helpers

    Fabric::Bus *FindBus(eMQ::Service *service) noexcept
    {
        if (!service)
        {
            return nullptr;
        }
        auto *module = service->GetModule<Fabric::BusModule>();
        if (!module)
        {
            return nullptr;
        }
        try
        {
            return &module->Get();
        }
        catch (const std::exception &)
        {
            return nullptr;
        }
    }

    bool Failed(const eMQ::Reason &reason) noexcept
    {
        return reason.m_Code >= 0x80;
    }

    void Require(const eMQ::Reason &reason, const std::string &what)
    {
        if (Failed(reason))
        {
            throw std::runtime_error(what + " failed: " + std::to_string(reason.m_Code) + " " + reason.m_Text);
        }
    }

    void Resolve(WorkloadOptions &options, std::string_view role)
    {
        const auto &service = options.m_Service;
        const auto &name = service.m_Metadata.m_Name;
        if (service.m_Observables.empty())
        {
            throw std::invalid_argument("Workload '" + name + "' declares no observables");
        }
        for (const auto &observable : service.m_Observables)
        {
            if (!ValidTopic(observable.m_Name, true) || observable.m_Qos > 1)
            {
                throw std::invalid_argument("Workload observable '" + observable.m_Name +
                                            "' is not a valid filter with QoS 0 or 1");
            }
        }
        if (options.m_Instance.empty())
        {
            options.m_Instance =
                name + "-" + std::string(role) + "-" + (options.m_Node.empty() ? "local" : options.m_Node) + "-0";
        }
        // validates the service and instance as topic levels (throws std::invalid_argument)
        (void)options.m_Topics.Heartbeat(name, options.m_Instance);
        (void)options.m_Topics.Workload(name, "model");
    }

    Json Number(std::optional<double> value)
    {
        return value && std::isfinite(*value) ? Json(*value) : Json();
    }
} // namespace eOps::Workload::Detail
