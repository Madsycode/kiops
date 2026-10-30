#pragma once
#include "Context.h"

namespace eMQ::Server
{
    struct Broker : eCore::ImplRuntime<Service, Context>
    {
        using eCore::ImplRuntime<Service, Context>::ImplRuntime;
    };
} // namespace eMQ::Server

namespace eMQ
{
    using Server::Broker;
}
