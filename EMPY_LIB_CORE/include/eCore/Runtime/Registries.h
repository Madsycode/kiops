#pragma once
#include <map>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <thread>
#include <memory>
#include <vector>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <condition_variable>
#include "../Common/Globals.h"
#include "../Vendors/EnTT.h"

// building blocks of the service registries: tasks, the owner-thread dispatch queue,
// owner-thread timers, the worker pool, and the entity/event registries

namespace eCore
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::milliseconds;

    // -------------------------
    // Task
    // -------------------------

    // Move-only type-erased `void()` callable with inline storage. Dispatched work items are
    // created per network read and per event, so small captures must not allocate.
    class Task
    {
    public:
        static constexpr size_t EMPY_TASK_INLINE = 48;

        EMPY_INLINE Task() noexcept = default;
        EMPY_INLINE Task(std::nullptr_t) noexcept
        {
        }

        template <typename _Callable, typename = std::enable_if_t<!std::is_same_v<std::decay_t<_Callable>, Task>>>
        EMPY_INLINE Task(_Callable &&callable)
        {
            using Stored = std::decay_t<_Callable>;
            if constexpr (sizeof(Stored) <= EMPY_TASK_INLINE && alignof(Stored) <= alignof(std::max_align_t) &&
                          std::is_nothrow_move_constructible_v<Stored>)
            {
                new (m_Storage) Stored(std::forward<_Callable>(callable));
                m_Ops = &InlineOps<Stored>;
            }
            else
            {
                *reinterpret_cast<Stored **>(m_Storage) = new Stored(std::forward<_Callable>(callable));
                m_Ops = &HeapOps<Stored>;
            }
        }

        EMPY_INLINE Task(Task &&other) noexcept : m_Ops(other.m_Ops)
        {
            if (m_Ops)
            {
                m_Ops->m_Move(other.m_Storage, m_Storage);
                other.m_Ops = nullptr;
            }
        }

        EMPY_INLINE Task &operator=(Task &&other) noexcept
        {
            if (this != &other)
            {
                Reset();
                m_Ops = other.m_Ops;
                if (m_Ops)
                {
                    m_Ops->m_Move(other.m_Storage, m_Storage);
                    other.m_Ops = nullptr;
                }
            }
            return *this;
        }

        Task(const Task &) = delete;
        Task &operator=(const Task &) = delete;

        EMPY_INLINE ~Task()
        {
            Reset();
        }

        EMPY_INLINE void operator()()
        {
            m_Ops->m_Invoke(m_Storage);
        }
        EMPY_INLINE explicit operator bool() const noexcept
        {
            return m_Ops != nullptr;
        }

        EMPY_INLINE void Reset() noexcept
        {
            if (m_Ops)
            {
                m_Ops->m_Destroy(m_Storage);
                m_Ops = nullptr;
            }
        }

    private:
        struct Ops
        {
            void (*m_Invoke)(void *);
            void (*m_Move)(void *from, void *to) noexcept;
            void (*m_Destroy)(void *) noexcept;
        };

        template <typename _Stored>
        static constexpr Ops InlineOps = {[](void *s) { (*static_cast<_Stored *>(s))(); },
                                          [](void *from, void *to) noexcept
                                          {
                                              new (to) _Stored(std::move(*static_cast<_Stored *>(from)));
                                              static_cast<_Stored *>(from)->~_Stored();
                                          },
                                          [](void *s) noexcept { static_cast<_Stored *>(s)->~_Stored(); }};

        template <typename _Stored>
        static constexpr Ops HeapOps = {[](void *s) { (**static_cast<_Stored **>(s))(); },
                                        [](void *from, void *to) noexcept
                                        { *static_cast<_Stored **>(to) = *static_cast<_Stored **>(from); },
                                        [](void *s) noexcept { delete *static_cast<_Stored **>(s); }};

        alignas(std::max_align_t) unsigned char m_Storage[EMPY_TASK_INLINE];
        const Ops *m_Ops = nullptr;
    };

    // -------------------------
    // DispatchQueue
    // -------------------------

    // Bounded multi-producer / single-consumer queue feeding the application owner thread.
    // Producers that must not block (network strands) use TryPush and, when it fails,
    // register a resume callback via WhenSpace; the consumer drains in batches by swapping
    // vectors so one lock acquisition moves an arbitrary number of tasks.
    class EMPY_CORE_API DispatchQueue
    {
    public:
        explicit DispatchQueue(size_t capacity, size_t byteCapacity = 16 * 1024 * 1024);
        ~DispatchQueue();

        DispatchQueue(const DispatchQueue &) = delete;
        DispatchQueue &operator=(const DispatchQueue &) = delete;

        // non-blocking enqueue; false when the queue is full or closed
        bool TryPush(Task &&task, size_t bytes = 0);

        // blocking enqueue (waits for space); false when closed
        bool Push(Task &&task);

        // enqueue ignoring the bound (owner-thread re-posts must never wait on themselves)
        void ForcePush(Task &&task);

        // register a callback invoked once the queue has drained below its low watermark
        void WhenSpace(Task &&resume);

        // wait until tasks are available or `deadline` passes, then move all queued tasks
        // into `out` (cleared first); returns the number of tasks moved
        size_t Drain(std::vector<Task> &out, Clock::time_point deadline);

        // run resume callbacks registered by blocked producers (consumer side)
        void ReleaseWaiters();

        void Wake();
        void Close();

        size_t Size() const;
        size_t Capacity() const noexcept
        {
            return m_Capacity;
        }

    private:
        mutable std::mutex m_Mutex;
        std::condition_variable m_Ready;
        std::condition_variable m_Space;
        std::vector<Task> m_Tasks;
        std::vector<Task> m_Waiters;
        size_t m_Capacity;
        size_t m_ByteCapacity;
        size_t m_Bytes = 0;
        bool m_Woken = false;
        bool m_Closed = false;
    };

    // -------------------------
    // TimerQueue
    // -------------------------

    using TimerId = uint64_t;

    // Owner-thread timers ordered by deadline. Not thread-safe by design: timers are created,
    // cancelled and fired on the thread that runs the service loop.
    class EMPY_CORE_API TimerQueue
    {
    public:
        TimerId Schedule(Clock::time_point deadline, Task &&task, uint32_t owner = 0);
        bool Cancel(TimerId id);
        void CancelOwned(uint32_t owner);

        // earliest pending deadline, or time_point::max() when idle
        Clock::time_point NextDeadline() const;

        // fire every timer whose deadline is <= now; returns the number fired
        size_t RunExpired(Clock::time_point now);

        size_t Size() const noexcept
        {
            return m_Tasks.size();
        }

    private:
        struct Entry
        {
            Task m_Callback;
            Clock::time_point m_Deadline;
            uint32_t m_Owner = 0;
        };

        std::multimap<Clock::time_point, TimerId> m_Order;
        std::unordered_map<TimerId, Entry> m_Tasks;
        TimerId m_NextId = 1;
    };

    // -------------------------
    // WorkerPool
    // -------------------------

    // Worker registry: lazily started, bounded pool for Async work. Stop belongs
    // to the owner thread; a worker must finish before the pool can be destroyed.
    class EMPY_CORE_API WorkerPool
    {
    public:
        explicit WorkerPool(uint32_t threads, size_t capacity = 1024);
        ~WorkerPool();

        WorkerPool(const WorkerPool &) = delete;
        WorkerPool &operator=(const WorkerPool &) = delete;

        void Submit(Task &&task);
        void Stop();
        uint32_t Size() const noexcept
        {
            return m_ThreadCount;
        }

    private:
        void Run();

        std::mutex m_Mutex;
        std::condition_variable m_Ready;
        std::deque<Task> m_Tasks;
        std::vector<std::thread> m_Threads;
        uint32_t m_ThreadCount;
        size_t m_Capacity;
        bool m_Stopping = false;
    };

    // -------------------------
    // EntityRegistry
    // -------------------------

    using Entity = entt::entity;
    inline constexpr Entity EMPY_NULL_ENTITY = entt::null;

    // Entity registry: data-oriented storage for sessions, topics and other managed objects.
    // Owner-thread only. Handles stay stable while components come and go, so auxiliary
    // indices (e.g. the subscription trie) can reference entities without owning them.
    class EntityRegistry
    {
    public:
        EMPY_INLINE Entity Create()
        {
            return m_Registry.create();
        }

        EMPY_INLINE void Destroy(Entity entity)
        {
            if (m_Registry.valid(entity))
            {
                m_Registry.destroy(entity);
            }
        }

        EMPY_INLINE bool Valid(Entity entity) const
        {
            return m_Registry.valid(entity);
        }

        template <typename _Component, typename... _Args> EMPY_INLINE _Component &Attach(Entity entity, _Args &&...args)
        {
            return m_Registry.emplace_or_replace<_Component>(entity, std::forward<_Args>(args)...);
        }

        template <typename _Component> EMPY_INLINE void Detach(Entity entity)
        {
            m_Registry.remove<_Component>(entity);
        }

        template <typename... _Components> EMPY_INLINE bool Has(Entity entity) const
        {
            return m_Registry.all_of<_Components...>(entity);
        }

        template <typename _Component> EMPY_INLINE _Component &Get(Entity entity)
        {
            return m_Registry.get<_Component>(entity);
        }

        template <typename _Component> EMPY_INLINE const _Component &Get(Entity entity) const
        {
            return m_Registry.get<_Component>(entity);
        }

        template <typename _Component> EMPY_INLINE _Component *TryGet(Entity entity)
        {
            return m_Registry.try_get<_Component>(entity);
        }

        template <typename _Component> EMPY_INLINE const _Component *TryGet(Entity entity) const
        {
            return m_Registry.try_get<_Component>(entity);
        }

        // invoke `fn` for every entity owning all listed components; `fn` may take
        // (Entity, Components&...) or (Components&...)
        template <typename... _Components, typename _Callback> EMPY_INLINE void View(_Callback &&fn)
        {
            m_Registry.view<_Components...>().each(std::forward<_Callback>(fn));
        }

        template <typename... _Components> EMPY_INLINE size_t Count()
        {
            auto view = m_Registry.view<_Components...>();
            if constexpr (sizeof...(_Components) == 1)
            {
                return view.size();
            }
            else
            {
                return static_cast<size_t>(std::distance(view.begin(), view.end()));
            }
        }

        EMPY_INLINE void Clear()
        {
            m_Registry.clear();
        }

        // escape hatch for advanced EnTT usage (groups, sorting, signals)
        EMPY_INLINE entt::registry &Native() noexcept
        {
            return m_Registry;
        }

    private:
        entt::registry m_Registry;
    };

    // -------------------------
    // EventRegistry
    // -------------------------

    using ListenerId = uint64_t;

    // Event registry: typed listeners keyed by a module-independent type key. Owner-thread only.
    // Listeners may be added or removed while an event is being delivered: removal leaves a
    // tombstone compacted after the outermost delivery, additions see the next event.
    class EventRegistry
    {
    public:
        template <typename _Event, typename _Callback> EMPY_INLINE ListenerId Add(_Callback &&fn, uint32_t owner)
        {
            ListenerId id = m_NextId++;
            auto &list = m_Listeners[TypeKey<_Event>()];
            list.push_back(
                Listener{id, owner, true,
                         std::function<void(const void *)>([fn = std::forward<_Callback>(fn)](const void *event) mutable
                                                           { fn(*static_cast<const _Event *>(event)); })});
            return id;
        }

        EMPY_INLINE bool Remove(ListenerId id)
        {
            for (auto &[key, list] : m_Listeners)
            {
                for (Listener &listener : list)
                {
                    if (listener.m_Id == id && listener.m_Active)
                    {
                        Retire(listener);
                        if (m_Depth == 0)
                        {
                            Compact();
                        }
                        return true;
                    }
                }
            }
            return false;
        }

        EMPY_INLINE void RemoveOwned(uint32_t owner)
        {
            for (auto &[key, list] : m_Listeners)
            {
                for (Listener &listener : list)
                {
                    if (listener.m_Owner == owner && listener.m_Active)
                    {
                        Retire(listener);
                    }
                }
            }
            if (m_Depth == 0 && m_Retired != 0)
            {
                Compact();
            }
        }

        template <typename _Event> EMPY_INLINE void Deliver(const _Event &event)
        {
            auto found = m_Listeners.find(TypeKey<_Event>());
            if (found == m_Listeners.end())
            {
                return;
            }
            ++m_Depth;
            // deque elements never move on push_back, and retired listeners keep their callable
            // until compaction, so a callback may (un)register listeners while it runs
            auto &list = found->second;
            size_t count = list.size();
            try
            {
                for (size_t i = 0; i < count; ++i)
                {
                    if (list[i].m_Active)
                    {
                        list[i].m_Callback(&event);
                    }
                }
            }
            catch (...)
            {
                if (--m_Depth == 0 && m_Retired != 0)
                {
                    Compact();
                }
                throw;
            }
            if (--m_Depth == 0 && m_Retired != 0)
            {
                Compact();
            }
        }

        template <typename _Event> EMPY_INLINE size_t CountOf() const
        {
            auto found = m_Listeners.find(TypeKey<_Event>());
            if (found == m_Listeners.end())
            {
                return 0;
            }
            size_t count = 0;
            for (const Listener &listener : found->second)
            {
                count += listener.m_Active ? 1 : 0;
            }
            return count;
        }

    private:
        struct Listener
        {
            ListenerId m_Id = 0;
            uint32_t m_Owner = 0;
            bool m_Active = true;
            std::function<void(const void *)> m_Callback;
        };

        EMPY_INLINE void Retire(Listener &listener)
        {
            listener.m_Active = false;
            ++m_Retired;
        }

        EMPY_INLINE void Compact()
        {
            for (auto &[key, list] : m_Listeners)
            {
                std::erase_if(list, [](const Listener &l) { return !l.m_Active; });
            }
            m_Retired = 0;
        }

        std::unordered_map<uint64_t, std::deque<Listener>> m_Listeners;
        ListenerId m_NextId = 1;
        uint32_t m_Depth = 0;
        uint32_t m_Retired = 0;
    };
} // namespace eCore
