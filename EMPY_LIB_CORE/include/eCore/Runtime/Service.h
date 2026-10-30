#pragma once
#include <tuple>
#include <future>
#include "Registries.h"

namespace eCore
{
    enum ServiceClass : uint8_t
    {
        NONE = 0,
        CLIENT,
        SERVER
    };

    // result type of an asynchronous task
    template <typename _Callback, typename... _Args>
    using _Result = std::invoke_result_t<std::decay_t<_Callback>, std::decay_t<_Args>...>;

    // sizing of the generic registries (protocol services add their own configuration)
    struct ServiceOptions
    {
        uint32_t m_Workers = 2;              // worker registry threads (Async)
        size_t m_DispatchCapacity = 1 << 16; // bounded owner-thread dispatch queue
        size_t m_DispatchBytes = 16 * 1024 * 1024;
        size_t m_WorkerCapacity = 1024;
    };

    // Coordinates data and control flows through four registries:
    //  - Worker Registry: worker threads for Async tasks
    //  - Entity Registry: ECS storage for sessions, topics and their state (owner thread)
    //  - Event Registry: typed listeners (On), synchronous (Invoke) and queued (Post) events
    //  - Module Registry: attached modules; listeners and timers they register are owned by
    //    their module identifier and removed automatically when the module detaches
    // Entity, event and module operations belong to the owner thread, i.e. the thread that
    // runs the service loop. Other threads hand work over through Post/Dispatch.
    template <ServiceClass _Class> struct ImplService
    {
        static constexpr ServiceClass Class = _Class;

        EMPY_INLINE ImplService &operator=(const ImplService &) = delete;
        EMPY_INLINE ImplService(const ImplService &) = delete;

        EMPY_INLINE virtual ~ImplService()
        {
            m_Dispatch.Close();
            m_Workers.Stop();
        }

        EMPY_INLINE explicit ImplService(const ServiceOptions &options = {})
            : m_Workers(options.m_Workers, options.m_WorkerCapacity),
              m_Dispatch(options.m_DispatchCapacity, options.m_DispatchBytes), m_OwnerThread(std::this_thread::get_id())
        {
        }

        // -- Worker Registry

        // run `fn(args...)` on a worker thread; the future carries the result or exception
        template <typename _Callback, typename... _Args>
        EMPY_INLINE auto Async(_Callback &&fn, _Args &&...args) -> std::future<_Result<_Callback, _Args...>>
        {
            using Result = _Result<_Callback, _Args...>;
            std::packaged_task<Result()> task(
                [fn = std::forward<_Callback>(fn), tuple = std::make_tuple(std::forward<_Args>(args)...)]() mutable
                { return std::apply(std::move(fn), std::move(tuple)); });
            std::future<Result> future = task.get_future();
            m_Workers.Submit(Task([task = std::move(task)]() mutable { task(); }));
            return future;
        }

        // -- Event Registry

        // deliver an event to its listeners immediately (owner thread)
        template <typename _Event, typename... _Args> EMPY_INLINE void Invoke(_Args &&...args)
        {
            EMPY_ASSERT(IsOwnerThread());
            const _Event event = MakeEvent<_Event>(std::forward<_Args>(args)...);
            m_Events.Deliver(event);
        }

        // queue an event for delivery on the owner thread (any thread); blocks while the
        // dispatch queue is full unless called from the owner thread itself
        template <typename _Event, typename... _Args> EMPY_INLINE void Post(_Args &&...args)
        {
            Dispatch(
                Task([this, event = MakeEvent<_Event>(std::forward<_Args>(args)...)]() { m_Events.Deliver(event); }));
        }

        // Network actors never block waiting for application callbacks. A rejected
        // publication remains unacknowledged and can be retried within its session.
        template <typename _Event, typename... _Args> EMPY_INLINE bool TryPost(size_t bytes, _Args &&...args)
        {
            if (bytes > SIZE_MAX - sizeof(_Event))
            {
                return false;
            }
            return m_Dispatch.TryPush(
                Task([this, event = MakeEvent<_Event>(std::forward<_Args>(args)...)] { m_Events.Deliver(event); }),
                bytes + sizeof(_Event));
        }

        // register a listener for `_Event` (owner thread); returns an id for Off()
        template <typename _Event, typename _Callback> EMPY_INLINE ListenerId On(_Callback &&fn)
        {
            EMPY_ASSERT(IsOwnerThread());
            return m_Events.template Add<_Event>(std::forward<_Callback>(fn), m_CurrentModule);
        }

        EMPY_INLINE bool Off(ListenerId id)
        {
            EMPY_ASSERT(IsOwnerThread());
            return m_Events.Remove(id);
        }

        template <typename _Event> EMPY_INLINE size_t ListenerCount() const
        {
            return m_Events.template CountOf<_Event>();
        }

        // -- owner-thread task dispatch

        // hand a task to the owner thread; blocks while the queue is full (off-owner only)
        EMPY_INLINE void Dispatch(Task &&task)
        {
            if (IsOwnerThread())
            {
                if (!m_Dispatch.TryPush(std::move(task)))
                {
                    throw std::length_error("Owner dispatch queue is full or closed");
                }
            }
            else
            {
                m_Dispatch.Push(std::move(task));
            }
        }

        // non-blocking hand-over used by network strands; on false, call WhenDispatchSpace
        EMPY_INLINE bool TryDispatch(Task &&task)
        {
            return m_Dispatch.TryPush(std::move(task));
        }

        // hand-over that ignores the bound and never blocks; reserved for rare control
        // notifications (e.g. link closure) that must not be lost or stall an I/O thread
        EMPY_INLINE void DispatchUrgent(Task &&task)
        {
            m_Dispatch.ForcePush(std::move(task));
        }

        // `resume` runs once the dispatch queue has room again (back-pressure release)
        EMPY_INLINE void WhenDispatchSpace(Task &&resume)
        {
            m_Dispatch.WhenSpace(std::move(resume));
        }

        EMPY_INLINE size_t DispatchBacklog() const
        {
            return m_Dispatch.Size();
        }

        // -- Entity Registry

        EMPY_INLINE EntityRegistry *GetRegistry() noexcept
        {
            return &m_Entities;
        }

        // -- owner-thread timers

        EMPY_INLINE TimerId Schedule(Milliseconds delay, Task &&task)
        {
            EMPY_ASSERT(IsOwnerThread());
            return m_Timers.Schedule(Clock::now() + delay, std::move(task), m_CurrentModule);
        }

        EMPY_INLINE bool CancelTimer(TimerId id)
        {
            EMPY_ASSERT(IsOwnerThread());
            return m_Timers.Cancel(id);
        }

        // -- Module Registry

        // look up an attached module by type (owner thread); nullptr when absent
        template <typename _Module> EMPY_INLINE _Module *GetModule() const
        {
            for (const ModuleRecord &record : m_Modules)
            {
                if (record.m_Type == TypeKey<_Module>())
                {
                    return static_cast<_Module *>(record.m_Instance);
                }
            }
            return nullptr;
        }

        EMPY_INLINE size_t ModuleCount() const noexcept
        {
            return m_Modules.size();
        }

        // -- lifecycle

        // request the service loop to stop (any thread)
        EMPY_INLINE void Terminate()
        {
            m_Terminated.store(true, std::memory_order_release);
            m_Dispatch.Wake();
        }

        EMPY_INLINE bool IsTerminated() const noexcept
        {
            return m_Terminated.load(std::memory_order_acquire);
        }

        EMPY_INLINE bool IsOwnerThread() const noexcept
        {
            return std::this_thread::get_id() == m_OwnerThread.load(std::memory_order_relaxed);
        }

    protected:
        struct ModuleRecord
        {
            uint32_t m_Id = 0;
            uint64_t m_Type = 0;
            void *m_Instance = nullptr;
        };

        // group-commit point: invoked by the runtime after each drained batch of tasks
        EMPY_INLINE virtual void OnDispatched()
        {
        }

        template <typename _Event, typename... _Args> static EMPY_INLINE _Event MakeEvent(_Args &&...args)
        {
            if constexpr (sizeof...(_Args) == 1 && (std::is_same_v<std::decay_t<_Args>, _Event> && ...))
            {
                return _Event(std::forward<_Args>(args)...);
            }
            else
            {
                return _Event{std::forward<_Args>(args)...};
            }
        }

        EMPY_INLINE void BindOwnerThread() noexcept
        {
            m_OwnerThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
        }

        EMPY_INLINE uint32_t RegisterModule(uint64_t type, void *instance)
        {
            uint32_t id = m_NextModuleId++;
            m_Modules.push_back(ModuleRecord{id, type, instance});
            return id;
        }

        // drop a module's record and every listener/timer it owns
        EMPY_INLINE void ReleaseModule(uint32_t id)
        {
            m_Events.RemoveOwned(id);
            m_Timers.CancelOwned(id);
            std::erase_if(m_Modules, [id](const ModuleRecord &r) { return r.m_Id == id; });
        }

        // run `fn` with `id` as the owner of anything it registers
        template <typename _Callback> EMPY_INLINE void AsModule(uint32_t id, _Callback &&fn)
        {
            uint32_t previous = std::exchange(m_CurrentModule, id);
            try
            {
                fn();
            }
            catch (...)
            {
                m_CurrentModule = previous;
                throw;
            }
            m_CurrentModule = previous;
        }

        WorkerPool m_Workers;
        DispatchQueue m_Dispatch;
        TimerQueue m_Timers;
        EntityRegistry m_Entities;
        EventRegistry m_Events;
        std::vector<ModuleRecord> m_Modules;
        std::atomic<std::thread::id> m_OwnerThread;
        std::atomic<bool> m_Terminated{false};
        uint32_t m_CurrentModule = 0;
        uint32_t m_NextModuleId = 1;
    };
} // namespace eCore
