#pragma once
#include "../Context/Comas.h"
#include "../Semantics/Ontology.h"
#include "../Intelligence/Provider.h"

// agent swarm (thesis §2.1.3): pure reasoning cores without MMQP access. Agents are stateless
// between requests (the harness supplies state in each request), not thread-safe, and their
// outputs are proposals that the harness validates (Harness/Guard.h) before any enactment.
// Every member of the environment is optional; null members disable the matching behaviour.

namespace eOps::Agents
{
    struct AgentEnvironment
    {
        Context::Comas *m_Comas = nullptr;                      // retrieval + Remember
        const Intelligence::ProviderSet *m_Providers = nullptr; // advisory LLMs
        Semantics::Ontology *m_Ontology = nullptr;              // grounding; Executive may Extend it
    };
} // namespace eOps::Agents
