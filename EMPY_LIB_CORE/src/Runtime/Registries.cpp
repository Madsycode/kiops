#include <eCore/Runtime/Registries.h>
#include <eCore/Common/Stdout.h>

namespace eCore
{
    // -------------------------
    // DispatchQueue
    // -------------------------

    DispatchQueue::DispatchQueue(size_t capacity, size_t byteCapacity)
        : m_Capacity(capacity == 0 ? 1 : capacity), m_ByteCapacity(byteCapacity)
    {
        m_Tasks.reserve(std::min<size_t>(m_Capacity, 4096));
    }

    DispatchQueue::~DispatchQueue()
    {
        Close();
    }

    bool DispatchQueue::TryPush(Task &&task, size_t bytes)
    {
        bool notify = false;
        {
            std::scoped_lock lock(m_Mutex);
            if (m_Closed || m_Tasks.size() >= m_Capacity || bytes > m_ByteCapacity - m_Bytes)
            {
                return false;
            }
            notify = m_Tasks.empty();
            m_Tasks.push_back(std::move(task));
            m_Bytes += bytes;
        }
        // the consumer only sleeps on an empty queue, so only the first push must wake it
        if (notify)
        {
            m_Ready.notify_one();
        }
        return true;
    }

    bool DispatchQueue::Push(Task &&task)
    {
        bool notify = false;
        {
            std::unique_lock lock(m_Mutex);
            m_Space.wait(lock, [this] { return m_Closed || m_Tasks.size() < m_Capacity; });
            if (m_Closed)
            {
                return false;
            }
            notify = m_Tasks.empty();
            m_Tasks.push_back(std::move(task));
        }
        if (notify)
        {
            m_Ready.notify_one();
        }
        return true;
    }

    void DispatchQueue::ForcePush(Task &&task)
    {
        bool notify = false;
        {
            std::scoped_lock lock(m_Mutex);
            if (m_Closed)
            {
                return;
            }
            notify = m_Tasks.empty();
            m_Tasks.push_back(std::move(task));
        }
        if (notify)
        {
            m_Ready.notify_one();
        }
    }

    void DispatchQueue::WhenSpace(Task &&resume)
    {
        bool runNow = false;
        {
            std::scoped_lock lock(m_Mutex);
            if (m_Closed)
            {
                return;
            }
            // the queue may have drained between the failed TryPush and this registration
            if (m_Tasks.size() < std::max<size_t>(1, m_Capacity / 2))
            {
                runNow = true;
            }
            else
            {
                m_Waiters.push_back(std::move(resume));
            }
        }
        if (runNow)
        {
            resume();
        }
    }

    size_t DispatchQueue::Drain(std::vector<Task> &out, Clock::time_point deadline)
    {
        out.clear();
        {
            std::unique_lock lock(m_Mutex);
            if (m_Tasks.empty() && !m_Woken && !m_Closed)
            {
                if (deadline == Clock::time_point::max())
                {
                    m_Ready.wait(lock, [this] { return !m_Tasks.empty() || m_Woken || m_Closed; });
                }
                else
                {
                    m_Ready.wait_until(lock, deadline, [this] { return !m_Tasks.empty() || m_Woken || m_Closed; });
                }
            }
            m_Woken = false;
            out.swap(m_Tasks);
            m_Bytes = 0;
        }
        if (!out.empty())
        {
            m_Space.notify_all();
        }
        return out.size();
    }

    void DispatchQueue::ReleaseWaiters()
    {
        std::vector<Task> waiters;
        {
            std::scoped_lock lock(m_Mutex);
            if (m_Waiters.empty() || m_Tasks.size() >= std::max<size_t>(1, m_Capacity / 2))
            {
                return;
            }
            waiters.swap(m_Waiters);
        }
        for (Task &resume : waiters)
        {
            resume();
        }
    }

    void DispatchQueue::Wake()
    {
        {
            std::scoped_lock lock(m_Mutex);
            m_Woken = true;
        }
        m_Ready.notify_one();
    }

    void DispatchQueue::Close()
    {
        {
            std::scoped_lock lock(m_Mutex);
            m_Closed = true;
        }
        m_Ready.notify_all();
        m_Space.notify_all();
    }

    size_t DispatchQueue::Size() const
    {
        std::scoped_lock lock(m_Mutex);
        return m_Tasks.size();
    }

    // -------------------------
    // TimerQueue
    // -------------------------

    TimerId TimerQueue::Schedule(Clock::time_point deadline, Task &&task, uint32_t owner)
    {
        TimerId id = m_NextId++;
        m_Tasks.emplace(id, Entry{std::move(task), deadline, owner});
        m_Order.emplace(deadline, id);
        return id;
    }

    bool TimerQueue::Cancel(TimerId id)
    {
        auto found = m_Tasks.find(id);
        if (found == m_Tasks.end())
        {
            return false;
        }
        auto range = m_Order.equal_range(found->second.m_Deadline);
        for (auto it = range.first; it != range.second; ++it)
        {
            if (it->second == id)
            {
                m_Order.erase(it);
                break;
            }
        }
        m_Tasks.erase(found);
        return true;
    }

    void TimerQueue::CancelOwned(uint32_t owner)
    {
        std::vector<TimerId> owned;
        for (const auto &[id, entry] : m_Tasks)
        {
            if (entry.m_Owner == owner)
            {
                owned.push_back(id);
            }
        }
        for (TimerId id : owned)
        {
            Cancel(id);
        }
    }

    Clock::time_point TimerQueue::NextDeadline() const
    {
        return m_Order.empty() ? Clock::time_point::max() : m_Order.begin()->first;
    }

    size_t TimerQueue::RunExpired(Clock::time_point now)
    {
        size_t fired = 0;
        const size_t budget = m_Tasks.size();
        // callbacks may schedule or cancel timers, so detach each entry before invoking it
        while (fired < budget && !m_Order.empty() && m_Order.begin()->first <= now)
        {
            TimerId id = m_Order.begin()->second;
            m_Order.erase(m_Order.begin());
            auto found = m_Tasks.find(id);
            if (found == m_Tasks.end())
            {
                continue;
            }
            Task callback = std::move(found->second.m_Callback);
            m_Tasks.erase(found);
            callback();
            ++fired;
        }
        return fired;
    }

    // -------------------------
    // WorkerPool
    // -------------------------

    WorkerPool::WorkerPool(uint32_t threads, size_t capacity)
        : m_ThreadCount(threads == 0 ? 1 : threads), m_Capacity(capacity)
    {
        if (!capacity || m_ThreadCount > 256)
        {
            throw std::invalid_argument("Invalid worker pool bounds");
        }
        m_Threads.reserve(m_ThreadCount);
    }

    WorkerPool::~WorkerPool()
    {
        Stop();
    }

    void WorkerPool::Submit(Task &&task)
    {
        {
            std::scoped_lock lock(m_Mutex);
            if (m_Stopping)
            {
                throw std::logic_error("Worker pool is stopped");
            }
            if (m_Tasks.size() >= m_Capacity)
            {
                throw std::length_error("Worker queue is full");
            }
            while (m_Threads.size() < m_ThreadCount)
            {
                m_Threads.emplace_back([this] { Run(); });
            }
            m_Tasks.push_back(std::move(task));
        }
        m_Ready.notify_one();
    }

    void WorkerPool::Stop()
    {
        {
            std::scoped_lock lock(m_Mutex);
            for (const auto &thread : m_Threads)
            {
                if (thread.get_id() == std::this_thread::get_id())
                {
                    throw std::logic_error("A worker cannot stop its own pool");
                }
            }
            if (m_Stopping && m_Threads.empty())
            {
                return;
            }
            m_Stopping = true;
        }
        m_Ready.notify_all();
        for (std::thread &thread : m_Threads)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
        m_Threads.clear();
    }

    void WorkerPool::Run()
    {
        for (;;)
        {
            Task task;
            {
                std::unique_lock lock(m_Mutex);
                m_Ready.wait(lock, [this] { return m_Stopping || !m_Tasks.empty(); });
                if (m_Tasks.empty())
                {
                    return;
                }
                // take one task at a time: long-running work must not hold back siblings
                task = std::move(m_Tasks.front());
                m_Tasks.pop_front();
            }
            try
            {
                task();
            }
            catch (const std::exception &error)
            {
                EMPY_ERROR("worker task failed: {}", error.what());
            }
            catch (...)
            {
                EMPY_ERROR("worker task failed with a non-standard exception");
            }
        }
    }
} // namespace eCore
