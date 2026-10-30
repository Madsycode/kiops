#include <eOps/Substrate/Substrate.h>
#include "Detail.h"
#include <map>
#include <mutex>

namespace eOps::Substrate
{
    struct SimulatedSubstrate::Impl
    {
        mutable std::mutex m_Mutex;
        std::map<std::string, DeploymentSpec> m_Instances; // ordered: deterministic listings
        Failure m_Failure;
    };

    SimulatedSubstrate::SimulatedSubstrate() : m_Impl(std::make_unique<Impl>())
    {
    }

    SimulatedSubstrate::~SimulatedSubstrate() = default;

    std::string SimulatedSubstrate::Name() const
    {
        return "simulated";
    }

    EnactResult SimulatedSubstrate::Deploy(const DeploymentSpec &spec)
    {
        DeploymentSpec resolved;
        if (std::optional<std::string> error = Detail::Resolve(spec, resolved))
        {
            return Detail::Failure(resolved.m_Instance, *error);
        }
        Failure failure;
        {
            std::lock_guard lock(m_Impl->m_Mutex);
            failure = m_Impl->m_Failure;
        }
        // the hook runs unlocked so it may inspect the substrate itself
        if (failure)
        {
            if (std::optional<std::string> detail = failure(resolved))
            {
                return Detail::Failure(resolved.m_Instance, *detail);
            }
        }
        std::lock_guard lock(m_Impl->m_Mutex);
        if (m_Impl->m_Instances.count(resolved.m_Instance))
        {
            return Detail::Failure(resolved.m_Instance, "instance '" + resolved.m_Instance + "' is already deployed");
        }
        EnactResult result;
        result.m_Ok = true;
        result.m_Instance = resolved.m_Instance;
        result.m_Detail = "simulated " + resolved.m_Role + " instance on " + resolved.m_Node;
        m_Impl->m_Instances.emplace(resolved.m_Instance, std::move(resolved));
        return result;
    }

    EnactResult SimulatedSubstrate::Retire(const std::string &instance)
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        if (!m_Impl->m_Instances.erase(instance))
        {
            return Detail::Failure(instance, "unknown instance '" + instance + "'");
        }
        EnactResult result;
        result.m_Ok = true;
        result.m_Instance = instance;
        result.m_Detail = "retired";
        return result;
    }

    std::vector<InstanceStatus> SimulatedSubstrate::List()
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        std::vector<InstanceStatus> statuses;
        for (const auto &[instance, spec] : m_Impl->m_Instances)
        {
            statuses.push_back(InstanceStatus{instance, spec.m_Service, spec.m_Role, spec.m_Node, "running"});
        }
        return statuses;
    }

    void SimulatedSubstrate::SetFailure(Failure failure)
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        m_Impl->m_Failure = std::move(failure);
    }

    std::vector<DeploymentSpec> SimulatedSubstrate::Deployed() const
    {
        std::lock_guard lock(m_Impl->m_Mutex);
        std::vector<DeploymentSpec> specs;
        for (const auto &[instance, spec] : m_Impl->m_Instances)
        {
            specs.push_back(spec);
        }
        return specs;
    }
} // namespace eOps::Substrate
