#pragma once
#include "Service.h"
#include "Context.h"
#include "Module.h"
#include "../Common/Stdout.h"

namespace eCore
{
    // Integrates a service, its context and modules: drives the owner-thread loop, dispatches
    // queued work and events, fires owner-thread timers and invokes lifecycle callbacks.
    // The runtime is the service (it derives from `_Service`), so application code calls
    // runtime->On<...>(), runtime->Connect(...) etc. directly on it.
    template <typename _Service, typename _Context> struct ImplRuntime : _Service
    {
        using Module = ImplModule<_Service, _Context>;

        EMPY_INLINE ImplRuntime &operator=(const ImplRuntime &) = delete;
        EMPY_INLINE ImplRuntime(const ImplRuntime &) = delete;

        // constructor arguments are forwarded to the service (e.g. a configuration path)
        template <typename... _Args>
        EMPY_INLINE explicit ImplRuntime(_Args &&...args)
            : _Service(std::forward<_Args>(args)...), m_Context(std::make_unique<_Context>())
        {
        }

        EMPY_INLINE virtual ~ImplRuntime()
        {
            Shutdown();
        }

        // attach a module; its OnAttach runs immediately when the loop is running, otherwise
        // right after the context's OnStart so modules observe a started service
        template <typename _Module, typename... _Args> EMPY_INLINE _Module *AttachModule(_Args &&...args)
        {
            EMPY_STATIC(std::is_base_of_v<Module, _Module>, "modules must derive from ImplModule<Service, Context>");
            EMPY_ASSERT(this->IsOwnerThread());
            if (m_Released || this->template GetModule<_Module>())
            {
                throw std::logic_error("Runtime is released or this module type is already attached");
            }
            auto module = std::make_unique<_Module>(std::forward<_Args>(args)...);
            _Module *instance = module.get();
            module->m_ModuleId = this->RegisterModule(TypeKey<_Module>(), instance);
            m_Attached.push_back(std::move(module));
            if (m_Running)
            {
                try
                {
                    StartModule(*instance);
                }
                catch (...)
                {
                    DetachById(instance->GetModuleId());
                    throw;
                }
            }
            return instance;
        }

        // detach a module by type; deferred to the next loop iteration while running so a
        // module may safely detach itself from inside one of its own callbacks
        template <typename _Module> EMPY_INLINE bool DetachModule()
        {
            EMPY_ASSERT(this->IsOwnerThread());
            _Module *instance = this->template GetModule<_Module>();
            if (!instance)
            {
                return false;
            }
            uint32_t id = instance->GetModuleId();
            if (m_Running)
            {
                this->Dispatch(Task([this, id] { DetachById(id); }));
            }
            else
            {
                DetachById(id);
            }
            return true;
        }

        EMPY_INLINE _Context *GetContext() noexcept
        {
            return m_Context.get();
        }

        // run the service loop on the calling thread until Terminate(); `ms` is the update
        // interval for OnUpdate callbacks. Returns the process exit code.
        EMPY_INLINE virtual int32_t StartServiceLoop(uint32_t ms)
        {
            this->BindOwnerThread();
            m_Interval = Milliseconds(ms == 0 ? 1 : ms);
            if (!m_Running)
            {
                Start();
            }
            m_NextTick = Clock::now() + m_Interval;
            while (!this->IsTerminated())
            {
                PollOnce(Clock::time_point::max());
            }
            // deliver whatever was queued before termination was requested
            PollOnce(Clock::now());
            Shutdown();
            return m_ExitCode;
        }

        // one loop iteration for applications that drive their own loop: waits at most
        // `waitMs` for work, then dispatches tasks, timers and due updates
        EMPY_INLINE void Poll(uint32_t waitMs = 0)
        {
            if (m_Released)
            {
                return;
            }
            if (!m_Running)
            {
                this->BindOwnerThread();
                if (m_Interval.count() == 0)
                {
                    m_Interval = Milliseconds(100);
                }
                Start();
                m_NextTick = Clock::now() + m_Interval;
            }
            PollOnce(Clock::now() + Milliseconds(waitMs));
        }

        EMPY_INLINE void SetExitCode(int32_t code) noexcept
        {
            m_ExitCode = code;
        }
        EMPY_INLINE int32_t StartLoop(uint32_t ms)
        {
            return StartServiceLoop(ms);
        }
        EMPY_INLINE bool IsRunning() const noexcept
        {
            return m_Running;
        }

        // Idempotent owner-thread release, including runtimes driven only through Poll.
        EMPY_INLINE void Shutdown()
        {
            if (m_Released)
            {
                return;
            }
            m_Released = true;
            this->Terminate();
            this->m_Dispatch.Close();
            this->m_Workers.Stop();
            DetachAll();
            if (m_Started)
            {
                RunGuarded([this] { m_Context->OnRelease(this); });
            }
            m_Running = false;
        }

    private:
        EMPY_INLINE void Start()
        {
            if (m_Released)
            {
                throw std::logic_error("Runtime has been released");
            }
            m_Started = true;
            try
            {
                m_Context->OnStart(this);
                m_Running = true;
                for (size_t i = 0; i < m_Attached.size(); ++i)
                {
                    if (!m_Attached[i]->m_Started)
                    {
                        StartModule(*m_Attached[i]);
                    }
                }
            }
            catch (...)
            {
                Shutdown();
                throw;
            }
        }

        EMPY_INLINE void PollOnce(Clock::time_point limit)
        {
            Clock::time_point deadline = std::min({limit, m_NextTick, this->m_Timers.NextDeadline()});
            this->m_Dispatch.Drain(m_Batch, deadline);
            for (Task &task : m_Batch)
            {
                RunGuarded(task);
            }
            if (!m_Batch.empty())
            {
                m_Batch.clear();
                RunGuarded([this] { this->OnDispatched(); });
                this->m_Dispatch.ReleaseWaiters();
            }
            Clock::time_point now = Clock::now();
            RunGuarded([this, now] { this->m_Timers.RunExpired(now); });
            if (now >= m_NextTick)
            {
                RunGuarded([this] { m_Context->OnUpdate(this); });
                for (size_t i = 0; i < m_Attached.size(); ++i)
                {
                    Module *module = m_Attached[i].get();
                    if (!module->m_Started)
                    {
                        continue;
                    }
                    RunGuarded(
                        [&]
                        { this->AsModule(module->GetModuleId(), [&] { module->OnUpdate(this, m_Context.get()); }); });
                }
                // skip missed ticks instead of bursting to catch up after a stall
                m_NextTick += m_Interval;
                if (m_NextTick <= now)
                {
                    m_NextTick = now + m_Interval;
                }
            }
        }

        template <typename _Callback> EMPY_INLINE void RunGuarded(_Callback &&fn)
        {
            try
            {
                fn();
            }
            catch (const std::exception &error)
            {
                EMPY_ERROR("service loop task failed: {}", error.what());
            }
            catch (...)
            {
                EMPY_ERROR("service loop task failed with a non-standard exception");
            }
        }

        EMPY_INLINE void StartModule(Module &module)
        {
            module.m_Started = true;
            this->AsModule(module.GetModuleId(), [&] { module.OnAttach(this, m_Context.get()); });
        }

        EMPY_INLINE void DetachById(uint32_t id)
        {
            for (size_t i = 0; i < m_Attached.size(); ++i)
            {
                if (m_Attached[i]->GetModuleId() != id)
                {
                    continue;
                }
                std::unique_ptr<Module> module = std::move(m_Attached[i]);
                m_Attached.erase(m_Attached.begin() + static_cast<std::ptrdiff_t>(i));
                if (module->m_Started)
                {
                    RunGuarded([&] { this->AsModule(id, [&] { module->OnDetach(this, m_Context.get()); }); });
                }
                this->ReleaseModule(id);
                return;
            }
        }

        // detach in reverse attachment order so later modules may depend on earlier ones
        EMPY_INLINE void DetachAll()
        {
            while (!m_Attached.empty())
            {
                DetachById(m_Attached.back()->GetModuleId());
            }
        }

        std::unique_ptr<_Context> m_Context;
        std::vector<std::unique_ptr<Module>> m_Attached;
        std::vector<Task> m_Batch;
        Milliseconds m_Interval{0};
        Clock::time_point m_NextTick{};
        int32_t m_ExitCode = 0;
        bool m_Running = false;
        bool m_Started = false;
        bool m_Released = false;
    };
} // namespace eCore
