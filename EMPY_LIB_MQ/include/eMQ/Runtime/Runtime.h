#pragma once
#include "Context.h"

namespace eMQ
{
    struct Client : eCore::ImplRuntime<Service, Context>
    {
        // The application-facing runtime owns its context and attached modules.
        using eCore::ImplRuntime<Service, Context>::ImplRuntime;
    };
} // namespace eMQ
