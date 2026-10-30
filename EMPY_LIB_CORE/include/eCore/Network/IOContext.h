#pragma once
#include "Connection.h"
#include <future>

namespace eCore::MMQP
{
    // Every connection and task of one context runs on its serial executor.
    // Public transport methods are called there; Submit is the cross-thread boundary.
    struct EMPY_CORE_API IoContext
    {
        using Accept = std::function<void(Connection::Pointer)>;

        explicit IoContext(size_t maxTasks = 4096, size_t maxBytes = 16 * 1024 * 1024);
        ~IoContext();
        IoContext(const IoContext &) = delete;
        IoContext &operator=(const IoContext &) = delete;

        uint16_t Listen(std::string_view host, uint16_t port, TransportOptions options, Accept accept,
                        size_t maxConnections = 1024);
        Connection::Pointer Connect(std::string host, uint16_t port, TransportOptions options, Accept connected,
                                    Connection::Closed closed);
        void Init(uint32_t workers = 1);
        void Stop(std::function<void()> beforeClose = {});
        // Nonblocking admission. Payload-bearing tasks supply their retained byte cost.
        void Post(std::function<void()> task, size_t bytes = 0);
        bool IsCurrent() const noexcept;

        template <typename _Task, typename... _Args>
        auto Submit(_Task &&task, _Args &&...args)
            -> std::future<std::invoke_result_t<std::decay_t<_Task>, std::decay_t<_Args>...>>
        {
            using Result = std::invoke_result_t<std::decay_t<_Task>, std::decay_t<_Args>...>;
            auto work = std::make_shared<std::packaged_task<Result()>>(
                std::bind(std::forward<_Task>(task), std::forward<_Args>(args)...));
            auto result = work->get_future();
            if (IsCurrent())
            {
                (*work)();
            }
            else
            {
                Post([work] { (*work)(); });
            }
            return result;
        }

    private:
        friend class IoTimer;
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    // Timer callbacks share the context's serial executor. The context outlives its timers.
    class EMPY_CORE_API IoTimer
    {
    public:
        explicit IoTimer(IoContext &context);
        ~IoTimer();
        IoTimer(const IoTimer &) = delete;
        IoTimer &operator=(const IoTimer &) = delete;
        void Schedule(std::chrono::milliseconds delay, std::function<void()> callback);
        void Cancel();

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    EMPY_CORE_API bool QuicAvailable();
    EMPY_CORE_API bool IsLoopback(std::string_view address);
} // namespace eCore::MMQP
