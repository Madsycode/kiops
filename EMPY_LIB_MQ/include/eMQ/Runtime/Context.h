#pragma once
#include "Service.h"

namespace eMQ
{
    struct Context : eCore::ImplContext<Service>
    {
        // Network operations run independently; application callbacks are driven by Poll.
        EMPY_INLINE void OnStart(Service *) override
        {
        }
        EMPY_INLINE void OnUpdate(Service *) override
        {
        }
        EMPY_INLINE void OnRelease(Service *service) override
        {
            service->Stop();
        }
    };
} // namespace eMQ
