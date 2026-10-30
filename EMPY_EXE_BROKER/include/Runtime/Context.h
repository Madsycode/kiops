#pragma once
#include "Service.h"

namespace eMQ::Server
{
    struct Context : eCore::ImplContext<Service>
    {
        EMPY_INLINE void OnStart(Service *service) override
        {
            if (!service->Port())
            {
                service->Listen();
            }
        }
        EMPY_INLINE void OnUpdate(Service *) override
        {
        }
        EMPY_INLINE void OnRelease(Service *service) override
        {
            service->Stop();
        }
    };
} // namespace eMQ::Server
