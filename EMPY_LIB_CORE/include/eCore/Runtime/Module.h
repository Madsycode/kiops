#pragma once
#include "../Common/Globals.h"

namespace eCore
{
    template <typename _Service, typename _Context> struct ImplRuntime;

    // Extends a runtime through lifecycle callbacks, event subscriptions and tasks.
    // Every module receives a unique identifier when attached; listeners and owner-thread
    // timers it registers from its callbacks are tied to that identifier and removed
    // automatically when the module is detached or destroyed.
    template <typename _Service, typename _Context> struct ImplModule
    {
        EMPY_INLINE ImplModule &operator=(const ImplModule &) = delete;
        EMPY_INLINE ImplModule(const ImplModule &) = delete;
        EMPY_INLINE virtual ~ImplModule() = default;
        EMPY_INLINE ImplModule() = default;

        EMPY_INLINE virtual void OnAttach(_Service *, _Context *) = 0;
        EMPY_INLINE virtual void OnUpdate(_Service *, _Context *) = 0;
        EMPY_INLINE virtual void OnDetach(_Service *, _Context *) = 0;

        EMPY_INLINE uint32_t GetModuleId() const noexcept
        {
            return m_ModuleId;
        }

    private:
        template <typename, typename> friend struct ImplRuntime;
        uint32_t m_ModuleId = 0;
        bool m_Started = false;
    };
} // namespace eCore
