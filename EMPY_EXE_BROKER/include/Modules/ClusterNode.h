#pragma once
#include "../Runtime/Runtime.h"

namespace eMQ::Server
{
    struct ClusterNode : eCore::ImplModule<Service, Context>
    {
        EMPY_INLINE void OnAttach(Service *service, Context *) override
        {
            service->StartCluster();
        }

        EMPY_INLINE void OnUpdate(Service *, Context *) override
        {
        }

        EMPY_INLINE void OnDetach(Service *service, Context *) override
        {
            service->StopCluster();
        }
    };
} // namespace eMQ::Server

namespace eMQ
{
    using Server::ClusterNode;
}
