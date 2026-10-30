#pragma once
#include "../Checks.h"
#include <eCore/Network/State.h>

template <typename _Function> void Throws(_Function &&function)
{
    bool thrown = false;
    try
    {
        function();
    }
    catch (const std::exception &)
    {
        thrown = true;
    }
    EMPY_CHECK(thrown);
}
