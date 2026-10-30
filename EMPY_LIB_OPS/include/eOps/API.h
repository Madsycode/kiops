#pragma once
#include <eCore/Common/Globals.h>

// shared library macros: EmpyOps owns its exported declarations (see eCore/Common/Globals.h)
#ifdef EMPY_OPS_SHARED
#define EMPY_OPS_API EMPY_EXPORT
#else
#define EMPY_OPS_API EMPY_IMPORT
#endif

namespace eOps
{
    EMPY_OPS_API const char *EmpyOpsTag();
}
