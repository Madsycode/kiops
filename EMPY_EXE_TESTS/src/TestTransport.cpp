#include "Checks.h"
#include <eCore/Network/IOContext.h>
#include <eCore/Network/Authentication.h>

using namespace eCore;
using namespace eCore::MMQP;

EMPY_TEST(Transport, SubmissionBoundsAndStoppedLifecycle)
{
    IoContext context(2, 32);
    int32_t calls = 0;
    context.Post([&] { ++calls; }, 32);
    EMPY_CHECK_THROWS(context.Post([] {}, 1));
    context.Post([&] { ++calls; });
    EMPY_CHECK_THROWS(context.Post([] {}));
    context.Stop([&] { ++calls; });
    EMPY_CHECK_EQ(calls, 3);
    EMPY_CHECK_THROWS(context.Post([] {}));
    context.Stop();
}

EMPY_TEST(Transport, GatheredFramesSharePayload)
{
    PublishRequest publication;
    publication.m_Alias = 1;
    publication.m_Topic = "shared/value";
    publication.m_Qos = 1;
    publication.m_PacketId = 1;
    publication.m_Payload = SharedBytes::Allocate(128 * 1024);
    std::fill_n(publication.m_Payload.MutableData(), publication.m_Payload.Size(), uint8_t{0x5A});
    auto first = Codec::EncodeFrame(publication);
    publication.m_Alias = 0x8000;
    publication.m_PacketId = 2;
    auto second = Codec::EncodeFrame(publication);
    EMPY_CHECK(first.m_Tail.Data() == publication.m_Payload.Data());
    EMPY_CHECK(second.m_Tail.Data() == publication.m_Payload.Data());
    EMPY_CHECK_EQ(publication.m_Payload.UseCount(), 3u);
    EMPY_CHECK_EQ(second.Flatten(), Codec::Encode(publication));
    EMPY_CHECK_EQ(FrameDigest(second), eCore::Sha256(second.Flatten()));
    EMPY_CHECK(first.m_Head.Size() < 64);
    EMPY_CHECK_EQ(Codec::PeekFrame(first).m_Payload, publication.m_Payload.Size());
}

EMPY_TEST(Transport, CallbackFailureDoesNotLoseShutdown)
{
    for (const bool threaded : {false, true})
    {
        IoContext context;
        std::promise<void> completed;
        auto result = completed.get_future();
        context.Post([] { throw std::runtime_error("Intentional callback failure"); });
        context.Post([&] { completed.set_value(); });
        if (threaded)
        {
            context.Init();
        }
        context.Stop();
        EMPY_CHECK(result.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    }
}
