#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <eMQ/Client.h>
#include <Runtime/Runtime.h>
#include <eCore/Network/IOContext.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <filesystem>
#include <thread>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    template <typename _Predicate>
    void Until(_Predicate &&predicate, std::initializer_list<Client *> clients = {}, uint32_t timeout = 3000)
    {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        do
        {
            for (auto client : clients)
            {
                client->Poll();
            }
            if (predicate())
            {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } while (Clock::now() < deadline);
        throw std::runtime_error("Integration condition timed out");
    }

    void Accepted(std::future<Reason> result)
    {
        EMPY_REQUIRE(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        auto reason = result.get();
        if (reason.m_Code >= 0x80)
        {
            throw std::runtime_error("Operation rejected: " + reason.m_Text);
        }
    }

    struct Certificate
    {
        std::filesystem::path m_Folder;
        std::string m_Certificate;
        std::string m_PrivateKey;

        Certificate()
        {
            m_Folder = std::filesystem::temp_directory_path() / ("EmpyTlsTest-" + ToHex(RandomBytes(8)));
            std::filesystem::create_directory(m_Folder);
            m_Certificate = (m_Folder / "certificate.pem").string();
            m_PrivateKey = (m_Folder / "private-key.pem").string();
            std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
                EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "prime256v1"), EVP_PKEY_free);
            std::unique_ptr<X509, decltype(&X509_free)> cert(X509_new(), X509_free);
            EMPY_REQUIRE(key && cert);
            EMPY_REQUIRE(X509_set_version(cert.get(), 2) == 1);
            EMPY_REQUIRE(ASN1_INTEGER_set(X509_get_serialNumber(cert.get()), 1) == 1);
            EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notBefore(cert.get()), -60));
            EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notAfter(cert.get()), 3600));
            EMPY_REQUIRE(X509_set_pubkey(cert.get(), key.get()) == 1);
            auto subject = X509_get_subject_name(cert.get());
            EMPY_REQUIRE(X509_NAME_add_entry_by_txt(subject, "CN", MBSTRING_ASC,
                                                    reinterpret_cast<const unsigned char *>("localhost"), -1, -1,
                                                    0) == 1);
            EMPY_REQUIRE(X509_set_issuer_name(cert.get(), subject) == 1);
            std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> san(
                X509V3_EXT_conf_nid(nullptr, nullptr, NID_subject_alt_name,
                                    const_cast<char *>("DNS:localhost,IP:127.0.0.1")),
                X509_EXTENSION_free);
            EMPY_REQUIRE(san && X509_add_ext(cert.get(), san.get(), -1) == 1);
            EMPY_REQUIRE(X509_sign(cert.get(), key.get(), EVP_sha256()) > 0);
            auto certFile = std::fopen(m_Certificate.c_str(), "wb");
            EMPY_REQUIRE(certFile);
            EMPY_REQUIRE(PEM_write_X509(certFile, cert.get()) == 1);
            std::fclose(certFile);
            auto keyFile = std::fopen(m_PrivateKey.c_str(), "wb");
            EMPY_REQUIRE(keyFile);
            EMPY_REQUIRE(PEM_write_PrivateKey(keyFile, key.get(), nullptr, nullptr, 0, nullptr, nullptr) == 1);
            std::fclose(keyFile);
        }

        ~Certificate()
        {
            std::error_code ignored;
            std::filesystem::remove(m_Certificate, ignored);
            std::filesystem::remove(m_PrivateKey, ignored);
            std::filesystem::remove(m_Folder, ignored);
        }
    };
} // namespace

EMPY_TEST(Integration, ApplicationEventsUseOwnerThread)
{
    eMQ::Server::Configs settings;
    settings.m_Port = 0;
    Broker broker(settings);
    const auto port = broker.Listen();
    Client client;
    const auto owner = std::this_thread::get_id();
    size_t opened = 0, closed = 0, published = 0;
    client.On<eMQ::Session>(
        [&](const auto &event)
        {
            EMPY_REQUIRE(std::this_thread::get_id() == owner);
            if (event.m_Connected)
            {
                ++opened;
            }
            else
            {
                ++closed;
            }
        });
    client.On<eMQ::Publish>("events/#",
                            [&](const auto &event)
                            {
                                EMPY_REQUIRE(std::this_thread::get_id() == owner);
                                EMPY_REQUIRE(event.m_Message.m_Topic == "events/value" &&
                                             event.m_Message.m_Payload == Buffer{'A'});
                                EMPY_REQUIRE(event.m_Reason.m_Code < 0x80);
                                ++published;
                            });
    client.Connect("127.0.0.1", port);
    Accepted(client.Register("events/value"));
    Accepted(client.Publish("events/value", {'A'}, 0));
    Accepted(client.Publish("events/value", {'A'}, 1));
    EMPY_REQUIRE(opened == 0 && published == 0);
    Until([&] { return opened == 1 && published == 2; }, {&client});
    client.Disconnect();
    Until([&] { return closed == 1; }, {&client});
    client.Shutdown();
    client.Shutdown();
}

namespace
{
    // More than 65535 QoS 1 publications in one session: the publisher's PIDs and
    // the broker's subscriber PIDs both wrap without renegotiating the session.
    void PidWrap()
    {
        eMQ::Server::Configs options;
        options.m_Port = 0;
        Broker broker(options);
        auto port = broker.Listen();
        Client publisher, subscriber;
        auto sid = publisher.Connect("127.0.0.1", port);
        subscriber.Connect("127.0.0.1", port);
        uint32_t next = 0;
        bool ordered = true;
        subscriber.On<Receive>("wrap/#",
                               [&](const Receive &event)
                               {
                                   Reader reader(event.m_Message.m_Payload);
                                   ordered = ordered && reader.U32() == next;
                                   ++next;
                               });
        Accepted(subscriber.Subscribe("wrap/#", 1));
        Accepted(publisher.Register("wrap/value"));
        constexpr uint32_t total = 65535 + 700;
        constexpr uint32_t batch = 240;
        for (uint32_t first = 0; first < total; first += batch)
        {
            std::vector<std::future<Reason>> results;
            for (uint32_t i = first; i < std::min(total, first + batch); ++i)
            {
                Writer payload;
                payload.U32(i);
                results.push_back(publisher.Publish("wrap/value", payload.Take(), 1));
            }
            for (auto &result : results)
            {
                Accepted(std::move(result));
            }
            Until([&] { return next >= std::min(total, first + batch) && broker.Metrics().m_Pending == 0; },
                  {&subscriber}, 10000);
        }
        EMPY_REQUIRE(ordered && next == total);
        auto metrics = broker.Metrics();
        EMPY_REQUIRE(metrics.m_Duplicates == 0 && metrics.m_Publications >= total);
        // The same session continues after the wrap.
        EMPY_REQUIRE(publisher.Connected());
        publisher.Disconnect();
        EMPY_REQUIRE(publisher.Connect("127.0.0.1", port, true) == sid);
        Writer payload;
        payload.U32(total);
        Accepted(publisher.Publish("wrap/value", payload.Take(), 1));
        Until([&] { return next == total + 1; }, {&subscriber});
        EMPY_REQUIRE(ordered);
        publisher.Disconnect();
        subscriber.Disconnect();
    }
} // namespace

EMPY_TEST(Integration, ReferenceCases)
{
    PidWrap();
    eMQ::Server::Configs settings;
    settings.m_Port = 0;
    settings.m_RequestTimeout = 800;
    settings.m_RetryInterval = 100;
    Broker broker(settings);
    auto port = broker.Listen();
    EMPY_REQUIRE(port != 0);

    Client publisher;
    Client subscriber;
    std::vector<PublishRequest> received;
    subscriber.On<Receive>("room/#", [&](const Receive &event) { received.push_back(event.m_Message); });
    publisher.Connect("127.0.0.1", port);
    auto session = subscriber.Connect("127.0.0.1", port);
    Accepted(subscriber.Subscribe("room/#"));
    Accepted(publisher.Register("room/#", 1, 60, 2));
    Accepted(publisher.Publish("room/temp", {'0'}, 0));
    Until([&] { return received.size() == 1; }, {&subscriber});
    EMPY_REQUIRE(received.back().m_Payload == Buffer{'0'});
    for (char value : {'A', 'B', 'C'})
    {
        Accepted(publisher.Publish("room/temp", Buffer{static_cast<uint8_t>(value)}, 1));
    }
    Until([&] { return received.size() == 4; }, {&subscriber});
    EMPY_REQUIRE(received.back().m_Payload == Buffer{'C'});

    Client late;
    std::vector<eCore::SharedBytes> replay;
    late.On<Receive>("room/#", [&](const Receive &event) { replay.push_back(event.m_Message.m_Payload); });
    late.Connect("127.0.0.1", port);
    Accepted(late.Subscribe("room/#"));
    Until([&] { return replay.size() == 2; }, {&late});
    EMPY_REQUIRE(replay[0] == Buffer{'B'} && replay[1] == Buffer{'C'});

    subscriber.Disconnect(false);
    Until([&] { return broker.Metrics().m_Online == 2; });
    Accepted(publisher.Publish("room/temp", {'D'}, 1));
    EMPY_REQUIRE(subscriber.Connect("127.0.0.1", port, true) == session);
    Until([&] { return received.size() == 5; }, {&subscriber});
    EMPY_REQUIRE(received.back().m_Payload == Buffer{'D'});
    Accepted(subscriber.Unsubscribe("room/#"));
    Accepted(publisher.Publish("room/temp", {'E'}, 1));
    Until([&] { return broker.Metrics().m_Pending == 0; }, {&subscriber, &late});
    EMPY_REQUIRE(received.size() == 5);

    Client sharedA, sharedB, qosZero;
    size_t first = 0, second = 0, zero = 0;
    sharedA.On<Receive>("work/#", [&](const auto &) { ++first; });
    sharedB.On<Receive>("work/#", [&](const auto &) { ++second; });
    qosZero.On<Receive>("work/#", [&](const auto &) { ++zero; });
    sharedA.Connect("127.0.0.1", port);
    sharedB.Connect("127.0.0.1", port);
    qosZero.Connect("127.0.0.1", port);
    Accepted(sharedA.Subscribe("$share/workers/work/#"));
    Accepted(sharedB.Subscribe("$share/workers/work/#"));
    Accepted(qosZero.Subscribe("work/#", 0));
    Accepted(publisher.Register("work/#"));
    for (int i = 0; i < 4; ++i)
    {
        Accepted(publisher.Publish("work/task", {'x'}, 0));
    }
    Until([&] { return first + second == 4 && zero == 4; }, {&sharedA, &sharedB, &qosZero});
    EMPY_REQUIRE(first == 2 && second == 2);
    Accepted(publisher.Publish("work/task", {'y'}, 1));
    Until([&] { return first + second == 5; }, {&sharedA, &sharedB, &qosZero});
    EMPY_REQUIRE(zero == 4);

    Client responder;
    responder.Connect("127.0.0.1", port);
    responder.On<Receive>("rpc/temp",
                          [&](const Receive &event)
                          {
                              EMPY_REQUIRE(event.m_Message.m_Type == MMQP_PUBREQ_TYPE_REQ);
                              Accepted(responder.Respond(event.m_Message, {'4', '2'}));
                          });
    std::vector<PublishRequest> responses;
    publisher.On<Receive>("rpc/#", [&](const Receive &event) { responses.push_back(event.m_Message); });
    Accepted(responder.Subscribe("rpc/temp", 1, true));
    Accepted(responder.Subscribe("rpc/slow", 1, true));
    Accepted(publisher.Register("rpc/#", 0, 60, 1, true));
    Accepted(publisher.Request("rpc/temp", {'?'}, 1));
    Until([&] { return responses.size() == 1; }, {&responder, &publisher});
    EMPY_REQUIRE(responses.back().m_Payload == Buffer({'4', '2'}));
    EMPY_REQUIRE(publisher.Request("rpc/missing", {}).get().m_Code == MMQP_REASON_SUCCESS);
    Until([&] { return responses.size() == 2; }, {&publisher});
    EMPY_REQUIRE(responses.back().m_Reason.m_Code == MMQP_REASON_NO_RESPONDER && responses.back().m_Payload.Empty());
    Accepted(publisher.Request("rpc/slow", {}));
    Accepted(publisher.Request("rpc/slow", {}));
    Until([&] { return responses.size() == 4; }, {&publisher, &responder});
    EMPY_REQUIRE(responses[2].m_Reason.m_Code == MMQP_REASON_TIMEOUT &&
                 responses[3].m_Reason.m_Code == MMQP_REASON_TIMEOUT);
    EMPY_REQUIRE(responses[2].m_CorrelationId != responses[3].m_CorrelationId);

    Client monitor;
    monitor.Connect("127.0.0.1", port);
    std::vector<std::string> wills;
    monitor.On<Receive>("$bwm/#", [&](const auto &event) { wills.push_back(event.m_Message.m_Topic); });
    monitor.On<Receive>("$lwm/#", [&](const auto &event) { wills.push_back(event.m_Message.m_Topic); });
    Accepted(monitor.Subscribe("$bwm/#"));
    Accepted(monitor.Subscribe("$lwm/#"));
    eMQ::Configs willOptions;
    willOptions.m_Properties.m_Will = SessionWill{Buffer{'1'}, Buffer{'0'}, "device/1", 0};
    Client status(willOptions);
    status.Connect("127.0.0.1", port);
    Until([&] { return wills.size() == 1; }, {&monitor});
    status.Disconnect();
    Until([&] { return wills.size() == 2; }, {&monitor});
    EMPY_REQUIRE(wills[0] == "$bwm/device/1" && wills[1] == "$lwm/device/1");

    auto secret = FromHex("000102030405060708090a0b0c0d0e0f");
    eMQ::Configs producerOptions, consumerOptions;
    producerOptions.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{secret}, 1, 1);
    consumerOptions.m_Cipher = std::make_shared<PayloadCipher>(KeyGrant{secret}, 1, 1);
    Client encryptedPublisher(producerOptions), encryptedSubscriber(consumerOptions);
    encryptedPublisher.Connect("127.0.0.1", port);
    encryptedSubscriber.Connect("127.0.0.1", port);
    bool decrypted = false;
    encryptedSubscriber.On<Receive>("secret/#",
                                    [&](const Receive &event)
                                    {
                                        EMPY_REQUIRE(event.m_Decrypted &&
                                                     event.m_Message.m_Payload == Buffer({'h', 'i'}));
                                        decrypted = true;
                                    });
    Accepted(encryptedPublisher.Register("secret/#"));
    Accepted(encryptedSubscriber.Subscribe("secret/#"));
    Accepted(encryptedPublisher.Publish("secret/value", {'h', 'i'}, 1));
    Until([&] { return decrypted; }, {&encryptedSubscriber});

    Certificate certificate;
    eMQ::Server::Configs secured;
    secured.m_Port = 0;
    secured.m_Transport.m_Tls = true;
    secured.m_Transport.m_Certificate = certificate.m_Certificate;
    secured.m_Transport.m_PrivateKey = certificate.m_PrivateKey;
    secured.m_RequireAuth = true;
    secured.m_AuthKey = Buffer(32, 0x42);
    Broker secureBroker(secured);
    auto securePort = secureBroker.Listen();
    eMQ::Configs trusted;
    trusted.m_Transport.m_Tls = true;
    trusted.m_Transport.m_Authority = certificate.m_Certificate;
    trusted.m_Transport.m_ServerName = "localhost";
    trusted.m_AuthKey = secured.m_AuthKey;
    Client trustedClient(trusted);
    EMPY_REQUIRE(!trustedClient.Connect("127.0.0.1", securePort).empty());
    eMQ::Configs anonymous = trusted;
    anonymous.m_AuthKey.clear();
    Client anonymousClient(anonymous);
    Throws([&] { anonymousClient.Connect("127.0.0.1", securePort); });
    eMQ::Configs wrongKey = trusted;
    wrongKey.m_AuthKey.assign(32, 0x43);
    Client wrongClient(wrongKey);
    Throws([&] { wrongClient.Connect("127.0.0.1", securePort); });
    eMQ::Configs wrongName = trusted;
    wrongName.m_Transport.m_ServerName = "untrusted.example";
    Client wrongHost(wrongName);
    Throws([&] { wrongHost.Connect("127.0.0.1", securePort); });
    EMPY_REQUIRE(broker.Metrics().m_Publications >= 10);
}

EMPY_TEST(Quic, ReferenceCases)
{
    EMPY_REQUIRE(QuicAvailable());
    Certificate certificate;
    eMQ::Server::Configs settings;
    settings.m_Port = 0;
    settings.m_Transport.m_Quic = true;
    settings.m_Transport.m_Certificate = certificate.m_Certificate;
    settings.m_Transport.m_PrivateKey = certificate.m_PrivateKey;
    settings.m_Transport.m_QuicPolicy = {MMQP_QUIC_STRATEGY_BULK, BulkPolicy{2, 1024}, 1200};
    settings.m_RequireAuth = true;
    settings.m_AuthKey = Buffer(32, 0x42);
    Broker broker(settings);
    auto port = broker.Listen();
    eMQ::Configs trusted;
    trusted.m_Transport.m_Quic = true;
    trusted.m_Transport.m_Authority = certificate.m_Certificate;
    trusted.m_Transport.m_ServerName = "localhost";
    trusted.m_AuthKey = settings.m_AuthKey;
    for (auto strategy : {MMQP_QUIC_STRATEGY_SINGLE, MMQP_QUIC_STRATEGY_SEMANTIC, MMQP_QUIC_STRATEGY_BULK})
    {
        auto options = trusted;
        options.m_Properties.m_Quic = QuicPolicy{strategy, {}, 1200};
        if (strategy == MMQP_QUIC_STRATEGY_BULK)
        {
            options.m_Properties.m_Quic->m_Bulk = BulkPolicy{2, 1024};
        }
        Client subscriber(options), publisher(options);
        auto identifier = subscriber.Connect("127.0.0.1", port);
        publisher.Connect("127.0.0.1", port);
        auto topic = "quic/" + std::to_string(strategy) + "/value";
        Accepted(subscriber.Subscribe(topic));
        Accepted(publisher.Register(topic));
        std::map<size_t, size_t> received;
        subscriber.On<Receive>(topic,
                               [&](const Receive &event)
                               {
                                   EMPY_REQUIRE(!event.m_Message.m_Payload.Empty() &&
                                                event.m_Message.m_Payload.Span().front() ==
                                                    event.m_Message.m_Payload.Span().back());
                                   ++received[event.m_Message.m_Payload.Size()];
                               });
        Accepted(publisher.Publish(topic, Buffer(32, 1), 0));
        Accepted(publisher.Publish(topic, Buffer(2048, 2), 0)); // Too large for a datagram; uses a stream.
        Accepted(publisher.Publish(topic, Buffer(128 * 1024, 3), 1));
        Until([&] { return received[32] == 1 && received[2048] == 1 && received[128 * 1024] == 1; }, {&subscriber},
              5000);
        std::vector<std::future<Reason>> pending;
        for (size_t i = 0; i < 8; ++i)
        {
            pending.push_back(publisher.Publish(topic, Buffer(i % 2 ? 4096 : 128 * 1024, static_cast<uint8_t>(i)), 1));
        }
        for (auto &result : pending)
        {
            Accepted(std::move(result));
        }
        Until([&] { return received[4096] == 4 && received[128 * 1024] == 5; }, {&subscriber}, 5000);
        auto rpc = topic + "/rpc";
        Accepted(subscriber.Subscribe(rpc, 1, true));
        Accepted(publisher.Register(rpc, 0, 60, 1, true));
        std::vector<PublishRequest> requests, responses;
        subscriber.On<Receive>(rpc, [&](const auto &event) { requests.push_back(event.m_Message); });
        publisher.On<Receive>(rpc, [&](const auto &event) { responses.push_back(event.m_Message); });
        pending.clear();
        for (uint16_t id = 1; id <= 3; ++id)
        {
            pending.push_back(publisher.Request(rpc,
                                                Buffer(id == 1   ? 32
                                                       : id == 2 ? 2048
                                                                 : 128 * 1024,
                                                       static_cast<uint8_t>(id)),
                                                1, id == 3 ? 1 : 0, id));
        }
        for (auto &result : pending)
        {
            Accepted(std::move(result));
        }
        Until([&] { return requests.size() == 3; }, {&subscriber}, 5000);
        for (auto it = requests.rbegin(); it != requests.rend(); ++it)
        {
            Accepted(subscriber.Respond(*it, it->m_Payload));
        }
        Until([&] { return responses.size() == 3; }, {&publisher}, 5000);
        std::set<uint16_t> correlations;
        for (const auto &response : responses)
        {
            EMPY_REQUIRE(response.m_CorrelationId == response.m_Payload.Span().front());
            correlations.insert(response.m_CorrelationId);
        }
        EMPY_REQUIRE(correlations == std::set<uint16_t>({1, 2, 3}));
        subscriber.Disconnect();
        EMPY_REQUIRE(subscriber.Connect("127.0.0.1", port, true) == identifier);
        Accepted(publisher.Publish(topic, Buffer(64, 4), 1));
        Until([&] { return received[64] == 1; }, {&subscriber});
        std::vector<std::future<Reason>> canceled;
        for (size_t i = 0; i < 8; ++i)
        {
            canceled.push_back(publisher.Publish(topic, Buffer(256 * 1024, 5), 1));
        }
        publisher.Disconnect(false);
        for (auto &operation : canceled)
        {
            EMPY_REQUIRE(operation.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
            try
            {
                operation.get();
            }
            catch (const std::exception &)
            {
            } // Delivery may finish or be canceled.
        }
        subscriber.Disconnect();
    }
    auto wrongName = trusted;
    wrongName.m_Transport.m_ServerName = "untrusted.example";
    Client wrongHost(wrongName);
    Throws([&] { wrongHost.Connect("127.0.0.1", port); });
    auto wrongKey = trusted;
    wrongKey.m_AuthKey.assign(32, 0x43);
    Client unauthorized(wrongKey);
    Throws([&] { unauthorized.Connect("127.0.0.1", port); });
    auto mutual = settings;
    mutual.m_Transport.m_Mutual = true;
    mutual.m_Transport.m_Authority = certificate.m_Certificate;
    Broker mutualBroker(mutual);
    auto mutualPort = mutualBroker.Listen();
    Client missingCertificate(trusted);
    Throws([&] { missingCertificate.Connect("127.0.0.1", mutualPort); });
    auto identified = trusted;
    identified.m_Transport.m_Certificate = certificate.m_Certificate;
    identified.m_Transport.m_PrivateKey = certificate.m_PrivateKey;
    Client mutualClient(identified);
    EMPY_REQUIRE(!mutualClient.Connect("127.0.0.1", mutualPort).empty());
    mutualClient.Disconnect();
}
