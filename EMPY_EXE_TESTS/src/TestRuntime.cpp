#include "Checks.h"
#include <eCore/API.h>

using namespace eCore;

namespace
{
    struct Ping
    {
        int32_t m_Value = 0;
    };
    struct Pong
    {
        std::string m_Text;
    };

    struct TestService : ImplService<SERVER>
    {
        explicit TestService(int32_t tag = 0) : ImplService(ServiceOptions{2, 8}), m_Tag(tag)
        {
        }
        int32_t m_Tag;
        int32_t m_Dispatched = 0;

    protected:
        void OnDispatched() override
        {
            ++m_Dispatched;
        }
    };

    struct TestContext : ImplContext<TestService>
    {
        std::vector<std::string> m_Calls;
        void OnStart(TestService *) override
        {
            m_Calls.push_back("start");
        }
        void OnUpdate(TestService *) override
        {
            m_Calls.push_back("update");
        }
        void OnRelease(TestService *) override
        {
            m_Calls.push_back("release");
        }
    };

    struct TestRuntime : ImplRuntime<TestService, TestContext>
    {
        using ImplRuntime::ImplRuntime;
    };

    struct CountingModule : ImplModule<TestService, TestContext>
    {
        int32_t *m_Attached;
        int32_t *m_Detached;
        int32_t m_Updates = 0;
        int32_t m_Pings = 0;
        CountingModule(int32_t *attached, int32_t *detached) : m_Attached(attached), m_Detached(detached)
        {
        }
        void OnAttach(TestService *service, TestContext *) override
        {
            ++*m_Attached;
            service->On<Ping>([this](const Ping &ping) { m_Pings += ping.m_Value; });
        }
        void OnUpdate(TestService *, TestContext *) override
        {
            ++m_Updates;
        }
        void OnDetach(TestService *, TestContext *) override
        {
            ++*m_Detached;
        }
    };

    struct SelfDetachingModule : ImplModule<TestService, TestContext>
    {
        void OnAttach(TestService *service, TestContext *) override
        {
            service->On<Ping>([service](const Ping &)
                              { static_cast<TestRuntime *>(service)->DetachModule<SelfDetachingModule>(); });
        }
        void OnUpdate(TestService *, TestContext *) override
        {
        }
        void OnDetach(TestService *, TestContext *) override
        {
        }
    };
} // namespace

EMPY_TEST(Runtime, TaskInlineAndHeap)
{
    int32_t calls = 0;
    Task small([&calls] { ++calls; });
    small();
    std::array<uint8_t, 256> big{};
    big[0] = 7;
    Task large([&calls, big] { calls += big[0]; });
    Task moved = std::move(large);
    EMPY_CHECK(!large);
    moved();
    EMPY_CHECK_EQ(calls, 8);
    auto owned = std::make_unique<int32_t>(5);
    Task moveOnly([owned = std::move(owned), &calls] { calls += *owned; });
    moveOnly();
    EMPY_CHECK_EQ(calls, 13);
}

EMPY_TEST(Runtime, DispatchQueueBackPressure)
{
    DispatchQueue queue(4);
    int32_t ran = 0;
    for (int32_t i = 0; i < 4; ++i)
    {
        EMPY_CHECK(queue.TryPush(Task([&ran] { ++ran; })));
    }
    EMPY_CHECK(!queue.TryPush(Task([&ran] { ++ran; })));
    bool resumed = false;
    queue.WhenSpace(Task([&resumed] { resumed = true; }));
    EMPY_CHECK(!resumed);

    std::vector<Task> batch;
    EMPY_CHECK_EQ(queue.Drain(batch, Clock::now()), 4u);
    for (Task &task : batch)
    {
        task();
    }
    EMPY_CHECK_EQ(ran, 4);
    queue.ReleaseWaiters();
    EMPY_CHECK(resumed);

    // ForcePush ignores the bound; Drain with a past deadline does not block
    for (int32_t i = 0; i < 10; ++i)
    {
        queue.ForcePush(Task([] {}));
    }
    EMPY_CHECK_EQ(queue.Size(), 10u);
    EMPY_CHECK_EQ(queue.Drain(batch, Clock::now()), 10u);
    auto start = Clock::now();
    EMPY_CHECK_EQ(queue.Drain(batch, Clock::now() + Milliseconds(20)), 0u);
    EMPY_CHECK(Clock::now() - start >= Milliseconds(15));

    // a blocked producer proceeds once the consumer drains
    for (int32_t i = 0; i < 4; ++i)
    {
        queue.TryPush(Task([] {}));
    }
    std::atomic<bool> pushed{false};
    std::thread producer(
        [&]
        {
            queue.Push(Task([] {}));
            pushed = true;
        });
    std::this_thread::sleep_for(Milliseconds(20));
    EMPY_CHECK(!pushed.load());
    queue.Drain(batch, Clock::now());
    producer.join();
    EMPY_CHECK(pushed.load());
    queue.Close();
    EMPY_CHECK(!queue.TryPush(Task([] {})));
}

EMPY_TEST(Runtime, TimerQueueOrderingAndCancel)
{
    TimerQueue timers;
    std::vector<int32_t> order;
    auto now = Clock::now();
    timers.Schedule(now + Milliseconds(30), Task([&] { order.push_back(3); }));
    timers.Schedule(now + Milliseconds(10), Task([&] { order.push_back(1); }));
    TimerId cancelled = timers.Schedule(now + Milliseconds(20), Task([&] { order.push_back(2); }));
    timers.Schedule(now + Milliseconds(5), Task([&] { order.push_back(0); }), 9);
    EMPY_CHECK(timers.Cancel(cancelled));
    EMPY_CHECK(!timers.Cancel(cancelled));
    EMPY_CHECK(timers.NextDeadline() == now + Milliseconds(5));
    timers.CancelOwned(9);
    EMPY_CHECK_EQ(timers.RunExpired(now + Milliseconds(100)), 2u);
    EMPY_CHECK_EQ(order.size(), 2u);
    EMPY_CHECK(order == std::vector<int32_t>({1, 3}));
    EMPY_CHECK(timers.NextDeadline() == Clock::time_point::max());

    // a timer may re-arm itself from its own callback
    int32_t fired = 0;
    std::function<void()> rearm = [&]
    {
        if (++fired < 3)
        {
            timers.Schedule(now, Task([&] { rearm(); }));
        }
    };
    timers.Schedule(now, Task([&] { rearm(); }));
    for (int32_t i = 0; i < 5; ++i)
    {
        timers.RunExpired(now);
    }
    EMPY_CHECK_EQ(fired, 3);
}

EMPY_TEST(Runtime, AsyncWorkers)
{
    TestService service;
    auto sum = service.Async([](int32_t a, int32_t b) { return a + b; }, 20, 22);
    EMPY_CHECK_EQ(sum.get(), 42);
    auto failing = service.Async([]() -> int32_t { throw std::runtime_error("boom"); });
    EMPY_CHECK_THROWS(failing.get());
    std::atomic<int32_t> counter{0};
    std::vector<std::future<void>> futures;
    for (int32_t i = 0; i < 100; ++i)
    {
        futures.push_back(service.Async([&counter] { counter++; }));
    }
    for (auto &future : futures)
    {
        future.get();
    }
    EMPY_CHECK_EQ(counter.load(), 100);
}

EMPY_TEST(Runtime, EventsInvokePostAndRemoval)
{
    TestService service;
    int32_t total = 0;
    ListenerId first = service.On<Ping>([&](const Ping &ping) { total += ping.m_Value; });
    ListenerId second = service.On<Ping>([&](const Ping &ping) { total += 10 * ping.m_Value; });
    std::string text;
    service.On<Pong>([&](const Pong &pong) { text = pong.m_Text; });
    service.Invoke<Ping>(1);
    EMPY_CHECK_EQ(total, 11);
    service.Invoke<Pong>(Pong{"hi"});
    EMPY_CHECK_EQ(text, std::string("hi"));
    EMPY_CHECK_EQ(service.ListenerCount<Ping>(), 2u);
    EMPY_CHECK(service.Off(second));
    EMPY_CHECK(!service.Off(second));
    service.Invoke<Ping>(1);
    EMPY_CHECK_EQ(total, 12);

    // a listener removing itself and registering another during delivery
    ListenerId self = 0;
    int32_t added = 0;
    self = service.On<Ping>(
        [&](const Ping &)
        {
            service.Off(self);
            service.On<Ping>([&](const Ping &) { ++added; });
        });
    service.Invoke<Ping>(0);
    EMPY_CHECK_EQ(added, 0);
    service.Invoke<Ping>(0);
    EMPY_CHECK_EQ(added, 1);
    service.Off(first);
}

EMPY_TEST(Runtime, LoopLifecycleModulesAndCrossThreadPost)
{
    TestRuntime runtime(5);
    EMPY_CHECK_EQ(runtime.m_Tag, 5);
    int32_t attached = 0, detached = 0;
    CountingModule *module = runtime.AttachModule<CountingModule>(&attached, &detached);
    EMPY_CHECK_EQ(attached, 0);
    EMPY_CHECK(runtime.GetModule<CountingModule>() == module);

    std::thread poster(
        [&]
        {
            std::this_thread::sleep_for(Milliseconds(30));
            for (int32_t i = 0; i < 50; ++i)
            {
                runtime.Post<Ping>(1);
            }
            std::this_thread::sleep_for(Milliseconds(60));
            runtime.Terminate();
        });
    int32_t code = runtime.StartServiceLoop(10);
    poster.join();
    EMPY_CHECK_EQ(code, 0);
    EMPY_CHECK_EQ(attached, 1);
    EMPY_CHECK_EQ(detached, 1);
    EMPY_CHECK(runtime.GetModule<CountingModule>() == nullptr);
    EMPY_CHECK_EQ(runtime.ListenerCount<Ping>(), 0u);
    EMPY_CHECK(runtime.m_Dispatched > 0);
    auto &calls = runtime.GetContext()->m_Calls;
    EMPY_REQUIRE(calls.size() >= 3);
    EMPY_CHECK_EQ(calls.front(), std::string("start"));
    EMPY_CHECK_EQ(calls.back(), std::string("release"));
    EMPY_CHECK(std::count(calls.begin(), calls.end(), std::string("update")) >= 3);
}

EMPY_TEST(Runtime, ModuleOwnershipAndSelfDetach)
{
    TestRuntime runtime;
    int32_t attached = 0, detached = 0;
    runtime.AttachModule<SelfDetachingModule>();
    std::thread poster(
        [&]
        {
            std::this_thread::sleep_for(Milliseconds(20));
            runtime.Post<Ping>(1);
            std::this_thread::sleep_for(Milliseconds(40));
            runtime.Terminate();
        });
    runtime.StartServiceLoop(5);
    poster.join();
    EMPY_CHECK(runtime.GetModule<SelfDetachingModule>() == nullptr);
    EMPY_CHECK_EQ(runtime.ModuleCount(), 0u);

    // timers scheduled by a module are cancelled with it
    TestRuntime second;
    struct TimerModule : ImplModule<TestService, TestContext>
    {
        int32_t *m_Fired;
        explicit TimerModule(int32_t *fired) : m_Fired(fired)
        {
        }
        void OnAttach(TestService *service, TestContext *) override
        {
            service->Schedule(Milliseconds(50), Task([this] { ++*m_Fired; }));
        }
        void OnUpdate(TestService *, TestContext *) override
        {
        }
        void OnDetach(TestService *, TestContext *) override
        {
        }
    };
    int32_t fired = 0;
    second.AttachModule<TimerModule>(&fired);
    second.Poll(0);
    EMPY_CHECK(second.DetachModule<TimerModule>());
    second.Poll(0);
    second.Poll(80);
    EMPY_CHECK_EQ(fired, 0);
    (void)attached;
    (void)detached;
}

EMPY_TEST(Runtime, EntityRegistry)
{
    struct Connection
    {
        int32_t m_Id;
    };
    struct Traffic
    {
        uint64_t m_Bytes = 0;
    };
    TestService service;
    EntityRegistry *registry = service.GetRegistry();
    Entity a = registry->Create();
    Entity b = registry->Create();
    registry->Attach<Connection>(a, 1);
    registry->Attach<Connection>(b, 2);
    registry->Attach<Traffic>(b);
    EMPY_CHECK(registry->Has<Connection>(a));
    EMPY_CHECK(!registry->Has<Traffic>(a));
    EMPY_CHECK_EQ(registry->Count<Connection>(), 2u);
    EMPY_CHECK_EQ((registry->Count<Connection, Traffic>()), 1u);
    int32_t sum = 0;
    registry->View<Connection>([&](Connection &c) { sum += c.m_Id; });
    EMPY_CHECK_EQ(sum, 3);
    registry->View<Connection, Traffic>(
        [&](Entity e, Connection &, Traffic &t)
        {
            t.m_Bytes = 9;
            EMPY_CHECK(e == b);
        });
    EMPY_CHECK_EQ(registry->Get<Traffic>(b).m_Bytes, 9u);
    registry->Detach<Connection>(a);
    EMPY_CHECK(registry->TryGet<Connection>(a) == nullptr);
    registry->Destroy(b);
    EMPY_CHECK(!registry->Valid(b));
    registry->Destroy(b);
}

EMPY_TEST(Runtime, CapacityOneDispatchResumes)
{
    DispatchQueue queue(1);
    EMPY_REQUIRE(queue.TryPush(Task([] {})));
    int32_t resumed = 0;
    queue.WhenSpace(Task([&] { ++resumed; }));
    std::vector<Task> batch;
    queue.Drain(batch, Clock::now());
    queue.ReleaseWaiters();
    EMPY_CHECK_EQ(resumed, 1);
    queue.WhenSpace(Task([&] { ++resumed; }));
    EMPY_CHECK_EQ(resumed, 2);
}

EMPY_TEST(Runtime, ListenerCleanupSurvivesExceptions)
{
    EventRegistry events;
    int32_t calls = 0;
    for (int32_t i = 0; i < 12; ++i)
    {
        events.Add<Ping>([&](const Ping &) { ++calls; }, 7);
    }
    events.Add<Pong>([](const Pong &) {}, 7);
    events.RemoveOwned(7);
    EMPY_CHECK_EQ(events.CountOf<Ping>(), 0u);
    EMPY_CHECK_EQ(events.CountOf<Pong>(), 0u);
    ListenerId failing = events.Add<Ping>([](const Ping &) { throw std::runtime_error("listener"); }, 9);
    EMPY_CHECK_THROWS(events.Deliver(Ping{}));
    EMPY_REQUIRE(events.Remove(failing));
    events.Add<Ping>([&](const Ping &) { ++calls; }, 0);
    events.Deliver(Ping{});
    EMPY_CHECK_EQ(calls, 1);
}

EMPY_TEST(Runtime, WorkerBoundsAndSelfStop)
{
    WorkerPool pool(1, 1);
    std::promise<void> entered, release;
    auto gate = release.get_future().share();
    pool.Submit(Task(
        [&]
        {
            entered.set_value();
            gate.wait();
        }));
    entered.get_future().wait();
    pool.Submit(Task([] {}));
    EMPY_CHECK_THROWS(pool.Submit(Task([] {})));
    release.set_value();
    pool.Stop();
    EMPY_CHECK_THROWS(pool.Submit(Task([] {})));

    WorkerPool second(1);
    std::promise<bool> refused;
    second.Submit(Task(
        [&]
        {
            try
            {
                second.Stop();
                refused.set_value(false);
            }
            catch (const std::logic_error &)
            {
                refused.set_value(true);
            }
        }));
    EMPY_CHECK(refused.get_future().get());
    second.Stop();
}

EMPY_TEST(Runtime, PolledRuntimeReleasesOnce)
{
    int32_t detached = 0, released = 0;
    struct Context : ImplContext<TestService>
    {
        int32_t *m_Count = nullptr;
        void OnStart(TestService *) override
        {
        }
        void OnUpdate(TestService *) override
        {
        }
        void OnRelease(TestService *) override
        {
            ++*m_Count;
        }
    };
    struct Module : ImplModule<TestService, Context>
    {
        int32_t *m_Count;
        explicit Module(int32_t *count) : m_Count(count)
        {
        }
        void OnAttach(TestService *, Context *) override
        {
        }
        void OnUpdate(TestService *, Context *) override
        {
        }
        void OnDetach(TestService *, Context *context) override
        {
            EMPY_CHECK_EQ(*context->m_Count, 0);
            ++*m_Count;
        }
    };
    {
        ImplRuntime<TestService, Context> runtime;
        runtime.GetContext()->m_Count = &released;
        runtime.AttachModule<Module>(&detached);
        EMPY_CHECK_THROWS(runtime.AttachModule<Module>(&detached));
        runtime.Poll();
        runtime.Shutdown();
        runtime.Shutdown();
    }
    EMPY_CHECK_EQ(detached, 1);
    EMPY_CHECK_EQ(released, 1);
}

EMPY_TEST(Runtime, FailedModuleAttachReleasesOwnedResources)
{
    struct Failing : ImplModule<TestService, TestContext>
    {
        void OnAttach(TestService *service, TestContext *) override
        {
            service->On<Ping>([](const Ping &) {});
            service->Schedule(Milliseconds(0), Task([] { throw std::runtime_error("orphan timer"); }));
            throw std::runtime_error("attach");
        }
        void OnUpdate(TestService *, TestContext *) override
        {
        }
        void OnDetach(TestService *, TestContext *) override
        {
        }
    };
    TestRuntime runtime;
    runtime.AttachModule<Failing>();
    EMPY_CHECK_THROWS(runtime.Poll());
    EMPY_CHECK_EQ(runtime.ModuleCount(), 0u);
    EMPY_CHECK_EQ(runtime.ListenerCount<Ping>(), 0u);
    EMPY_CHECK_EQ(runtime.GetContext()->m_Calls.back(), std::string("release"));
}
