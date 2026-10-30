#include <eCore/Common/Stdout.h>

namespace eCore
{
    std::atomic<LogLevel> &LogThreshold() noexcept
    {
#ifdef EMPY_DEBUG_BUILD
        static std::atomic<LogLevel> s_Threshold{LogLevel::Debug};
#else
        static std::atomic<LogLevel> s_Threshold{LogLevel::Info};
#endif
        return s_Threshold;
    }

    std::mutex &LogMutex() noexcept
    {
        static std::mutex s_Mutex;
        return s_Mutex;
    }
} // namespace eCore
