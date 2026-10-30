#pragma once
#include "../Common/Globals.h"

namespace eCore
{
    // Bridges application-specific state and logic with the service and its modules.
    // The runtime invokes the hooks on the owner thread: OnStart before the first loop
    // iteration, OnUpdate once per loop interval, OnRelease after the loop ends.
    template <typename _Service> struct ImplContext
    {
        EMPY_INLINE ImplContext &operator=(const ImplContext &) = delete;
        EMPY_INLINE ImplContext(const ImplContext &) = delete;
        EMPY_INLINE virtual ~ImplContext() = default;
        EMPY_INLINE ImplContext() = default;

        EMPY_INLINE virtual void OnRelease(_Service *) = 0;
        EMPY_INLINE virtual void OnUpdate(_Service *) = 0;
        EMPY_INLINE virtual void OnStart(_Service *) = 0;
    };
} // namespace eCore
