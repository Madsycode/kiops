#include "Checks.h"
#include "Support/Protocol.h"
#include <asio.hpp>
#include <eMQ/Client.h>
#include <Runtime/Runtime.h>
#include <eCore/Network/IOContext.h>
#include <Cluster/Clock.h>
#include <Cluster/Storage.h>
#include <openssl/pem.h>
#include <openssl/x509v3.h>
#include <fstream>
#include <iostream>
#include <thread>

using namespace eMQ;
using namespace eMQ::Server;

namespace
{
    template <typename _Predicate>
    void Await(_Predicate &&predicate, std::initializer_list<Client *> clients = {}, uint32_t timeout = 8000)
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
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } while (Clock::now() < deadline);
        throw std::runtime_error("Recovery test condition timed out");
    }

    void Okay(std::future<Reason> result)
    {
        EMPY_REQUIRE(result.wait_for(std::chrono::seconds(12)) == std::future_status::ready);
        auto reason = result.get();
        if (reason.m_Code >= 0x80)
        {
            throw std::runtime_error("Recovery operation rejected: " + reason.m_Text);
        }
    }

    struct Fixture
    {
        std::filesystem::path m_Directory;
        std::vector<std::string> m_Certificates;
        std::vector<std::string> m_Keys;
        std::vector<std::string> m_Fingerprints;
        std::string m_Authority;

        Fixture()
        {
            m_Directory = std::filesystem::temp_directory_path() / ("EmpyRecovery-" + ToHex(RandomBytes(8)));
            std::filesystem::create_directories(m_Directory);
            m_Authority = (m_Directory / "authority.pem").string();
            std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> root(
                EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "prime256v1"), EVP_PKEY_free);
            std::unique_ptr<X509, decltype(&X509_free)> ca(X509_new(), X509_free);
            auto setup = [](X509 *cert, EVP_PKEY *key, int serial, const char *name)
            {
                EMPY_REQUIRE(cert && key && X509_set_version(cert, 2) == 1 &&
                             ASN1_INTEGER_set(X509_get_serialNumber(cert), serial) == 1);
                EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notBefore(cert), -60));
                EMPY_REQUIRE(X509_gmtime_adj(X509_getm_notAfter(cert), 3600));
                EMPY_REQUIRE(X509_set_pubkey(cert, key) == 1);
                EMPY_REQUIRE(X509_NAME_add_entry_by_txt(X509_get_subject_name(cert), "CN", MBSTRING_ASC,
                                                        reinterpret_cast<const unsigned char *>(name), -1, -1, 0) == 1);
            };
            auto extension = [](X509 *cert, int nid, const char *value)
            {
                std::unique_ptr<X509_EXTENSION, decltype(&X509_EXTENSION_free)> entry(
                    X509V3_EXT_conf_nid(nullptr, nullptr, nid, const_cast<char *>(value)), X509_EXTENSION_free);
                EMPY_REQUIRE(entry && X509_add_ext(cert, entry.get(), -1) == 1);
            };
            auto writeCertificate = [](const std::string &path, X509 *cert)
            {
                auto file = std::fopen(path.c_str(), "wb");
                EMPY_REQUIRE(file);
                auto result = PEM_write_X509(file, cert);
                std::fclose(file);
                EMPY_REQUIRE(result == 1);
            };
            setup(ca.get(), root.get(), 1, "Empy recovery test CA");
            X509_set_issuer_name(ca.get(), X509_get_subject_name(ca.get()));
            extension(ca.get(), NID_basic_constraints, "critical,CA:TRUE");
            extension(ca.get(), NID_key_usage, "critical,keyCertSign,cRLSign");
            EMPY_REQUIRE(X509_sign(ca.get(), root.get(), EVP_sha256()) > 0);
            writeCertificate(m_Authority, ca.get());
            for (int i = 0; i < 3; ++i)
            {
                std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
                    EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "prime256v1"), EVP_PKEY_free);
                std::unique_ptr<X509, decltype(&X509_free)> certificate(X509_new(), X509_free);
                setup(certificate.get(), key.get(), i + 2, "localhost");
                X509_set_issuer_name(certificate.get(), X509_get_subject_name(ca.get()));
                extension(certificate.get(), NID_basic_constraints, "critical,CA:FALSE");
                extension(certificate.get(), NID_key_usage, "critical,digitalSignature");
                extension(certificate.get(), NID_ext_key_usage, "serverAuth,clientAuth");
                extension(certificate.get(), NID_subject_alt_name, "DNS:localhost,IP:127.0.0.1");
                EMPY_REQUIRE(X509_sign(certificate.get(), root.get(), EVP_sha256()) > 0);
                m_Certificates.push_back((m_Directory / ("broker-" + std::to_string(i) + ".pem")).string());
                m_Keys.push_back((m_Directory / ("broker-" + std::to_string(i) + ".key")).string());
                writeCertificate(m_Certificates.back(), certificate.get());
                auto file = std::fopen(m_Keys.back().c_str(), "wb");
                EMPY_REQUIRE(file);
                auto success = PEM_write_PrivateKey(file, key.get(), nullptr, nullptr, 0, nullptr, nullptr);
                std::fclose(file);
                EMPY_REQUIRE(success == 1);
                Digest fingerprint{};
                unsigned int size = 0;
                EMPY_REQUIRE(X509_digest(certificate.get(), EVP_sha256(), fingerprint.data(), &size) == 1 &&
                             size == 32);
                m_Fingerprints.push_back(ToHex(fingerprint));
            }
        }

        ~Fixture()
        {
            std::error_code error;
            std::filesystem::remove_all(m_Directory, error);
        }

        std::array<ClusterOptions, 3> Options(bool quic)
        {
            asio::io_context context;
            std::vector<std::unique_ptr<asio::ip::tcp::acceptor>> reservations;
            std::vector<std::unique_ptr<asio::ip::udp::socket>> datagrams;
            std::vector<ClusterPeer> peers;
            for (size_t i = 0; i < 3; ++i)
            {
                auto socket = std::make_unique<asio::ip::tcp::acceptor>(
                    context, asio::ip::tcp::endpoint(asio::ip::make_address("127.0.0.1"), 0));
                auto datagram = std::make_unique<asio::ip::udp::socket>(context);
                datagram->open(asio::ip::udp::v4());
                asio::error_code error;
                datagram->bind(
                    asio::ip::udp::endpoint(asio::ip::make_address("127.0.0.1"), socket->local_endpoint().port()),
                    error);
                if (error)
                {
                    --i;
                    continue;
                }
                peers.push_back({"broker" + std::to_string(i), "127.0.0.1", socket->local_endpoint().port(),
                                 "localhost", m_Fingerprints[i]});
                reservations.push_back(std::move(socket));
                datagrams.push_back(std::move(datagram));
            }
            std::array<ClusterOptions, 3> settings;
            for (size_t i = 0; i < settings.size(); ++i)
            {
                auto &option = settings[i];
                option.m_Identifier = peers[i].m_Identifier;
                option.m_Port = peers[i].m_Port;
                option.m_Directory = (m_Directory / ("state-" + std::to_string(i))).string();
                option.m_Peers = peers;
                option.m_Fabric.m_Brokers = {"broker0", "broker1", "broker2"};
                option.m_Fabric.m_Shards = 4;
                option.m_Fabric.m_Replicas = 3;
                option.m_Fabric.m_Fanout = 2;
                option.m_EpochKey = Buffer(32, 0x39);
                option.m_Transport.m_Tls = true;
                option.m_Transport.m_Quic = quic;
                option.m_Transport.m_Mutual = true;
                option.m_Transport.m_Certificate = m_Certificates[i];
                option.m_Transport.m_PrivateKey = m_Keys[i];
                option.m_Transport.m_Authority = m_Authority;
                option.m_Transport.m_MaxPacket = 8 * 1024 * 1024;
                option.m_Transport.m_MaxQueue = 32 * 1024 * 1024;
                option.m_LeaseMilliseconds = 3000;
                option.m_TimeoutMilliseconds = 500;
            }
            return settings;
        }
    };

    void Storage()
    {
        Fixture fixture;
        auto path = fixture.m_Directory / "store";
        {
            FileStore store(path, 4096, 4);
            store.Put("checkpoint", {1, 2, 3});
            Throws([&] { FileStore duplicate(path); });
            Throws([&] { store.Put("oversized", Buffer(4097)); });
            store.Put("checkpoint", {4, 5});
        }
        {
            std::ofstream abandoned(path / "incomplete.rec.tmp", std::ios::binary);
            abandoned << "partial";
        }
        {
            FileStore store(path, 4096, 4);
            EMPY_REQUIRE(store.Get("checkpoint") == std::optional(Buffer{4, 5}));
        }
        for (const auto &file : std::filesystem::directory_iterator(path))
        {
            if (file.path().extension() == ".rec")
            {
                std::fstream corrupt(file.path(), std::ios::in | std::ios::out | std::ios::binary);
                corrupt.seekp(7);
                corrupt.put('\x7F');
            }
        }
        Throws([&] { FileStore corrupt(path, 4096, 4); });
        ForwardRecord record;
        record.m_Session = "durable-origin";
        record.m_Sequence = 1;
        record.m_AcceptedEpoch = record.m_RoutingEpoch = 1;
        record.m_Expires = UnixMilliseconds() + 60000;
        record.m_Publication.m_Topic = "durable/descriptor";
        record.m_Publication.m_Alias = 1;
        record.m_Publication.m_Qos = 1;
        record.m_Publication.m_PacketId = 1;
        record.m_Publication.m_Payload = {1, 2, 3};
        record.m_Identity = MessageIdentity(record.m_Session, record.m_Sequence, record.m_Publication.m_Topic,
                                            record.m_AcceptedEpoch, record.m_Publication.m_Payload);
        RetainedDescriptor descriptor;
        {
            FileDurabilityStore store(fixture.m_Directory / "descriptors");
            descriptor = store.Append(record);
            EMPY_REQUIRE(store.Append(record).m_Identity == descriptor.m_Identity);
        }
        {
            FileDurabilityStore store(fixture.m_Directory / "descriptors");
            EMPY_REQUIRE(store.Read(descriptor) == Codec::Encode(record.m_Publication, {true, UINT32_MAX}));
            auto forged = descriptor;
            forged.m_PayloadHash[0] ^= 1;
            Throws([&] { store.Read(forged); });
            store.Compact(record.m_Expires);
            Throws([&] { store.Read(descriptor); });
        }
    }

    void SnapshotSchemas()
    {
        Fixture fixture;
        eMQ::Server::Configs settings;
        settings.m_Port = 0;
        settings.m_StateDirectory = (fixture.m_Directory / "legacy-broker").string();
        // A complete, empty schema-3 snapshot must fail before loading any state.
        Buffer snapshot{'E', 'B', 'R', '3', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        {
            FileStore store(settings.m_StateDirectory);
            store.Put("broker", snapshot);
        }
        Throws([&] { Broker legacy(settings); });
        {
            FileStore store(settings.m_StateDirectory);
            EMPY_REQUIRE(store.Get("broker") == std::optional(snapshot));
        }

        auto options = fixture.Options(false).front();
        Writer writer;
        writer.U32(3);
        writer.U64(options.m_Fabric.m_Epoch);
        writer.U32(options.m_Fabric.m_Shards);
        writer.U16(options.m_Fabric.m_Replicas);
        writer.U16(options.m_Fabric.m_Fanout);
        for (const auto &peer : options.m_Peers)
        {
            writer.String(peer.m_Identifier);
            writer.String(peer.m_Fingerprint);
        }
        auto digest = Sha256(writer.Data());
        Buffer manifest(digest.begin(), digest.end());
        {
            FabricNode current(options);
        }
        // Cluster nodes use journaled stores; plant the old manifest the same way.
        {
            FileStore store(options.m_Directory, options.m_MaxBytes, options.m_MaxRecords, true);
            EMPY_REQUIRE(store.Get("manifest") != std::optional(manifest));
            store.Put("manifest", manifest);
        }
        Throws([&] { FabricNode legacy(options); });
        {
            FileStore store(options.m_Directory, options.m_MaxBytes, options.m_MaxRecords, true);
            EMPY_REQUIRE(store.Get("manifest") == std::optional(manifest));
        }
    }

    void PayloadState()
    {
        Fixture fixture;
        auto sender = (fixture.m_Directory / "cipher-sender").string();
        auto receiver = (fixture.m_Directory / "cipher-receiver").string();
        Buffer root(16, 0x63);
        PublishRequest first;
        first.m_Topic = "durable/encrypted";
        first.m_Payload = {'S'};
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 2, 2, 8, sender);
            Throws([&] { PayloadCipher duplicate(KeyGrant{root}, 1, 2, 2, 8, sender); });
            cipher.Encrypt(first, 1);
        }
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 2, 2, 8, receiver);
            EMPY_REQUIRE(cipher.Decrypt(first, first.m_Topic) == Buffer{'S'});
        }
        PublishRequest second;
        second.m_Topic = first.m_Topic;
        second.m_Payload = {'T'};
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 2, 2, 8, sender);
            cipher.Encrypt(second, 1);
            EMPY_REQUIRE(first.m_Encryption->m_Nonce != second.m_Encryption->m_Nonce);
            EMPY_REQUIRE(std::equal(first.m_Encryption->m_Nonce.begin(), first.m_Encryption->m_Nonce.begin() + 8,
                                    second.m_Encryption->m_Nonce.begin()));
        }
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 2, 2, 8, receiver);
            Throws([&] { cipher.Decrypt(first, first.m_Topic); });
            auto altered = second;
            altered.m_Encryption->m_Tag[0] ^= 1;
            Throws([&] { cipher.Decrypt(altered, altered.m_Topic); });
            EMPY_REQUIRE(cipher.Decrypt(second, second.m_Topic) == Buffer{'T'});
            cipher.Rotate(2, 3);
        }
        Throws([&] { PayloadCipher stale(KeyGrant{root}, 1, 2, 2, 8, receiver); });
        Throws([&] { PayloadCipher wrong(KeyGrant{Buffer(16, 0x64)}, 2, 3, 2, 8, receiver); });
        PayloadCipher restored(KeyGrant{root}, 2, 3, 2, 8, receiver);
        Throws([&] { restored.Decrypt(second, second.m_Topic); });
    }

    uint64_t NonceCounter(const PublishRequest &publication)
    {
        Reader reader(publication.m_Encryption->m_Nonce);
        reader.U64();
        return reader.U64();
    }

    void PayloadReservation()
    {
        Fixture fixture;
        auto sender = (fixture.m_Directory / "reserve-sender").string();
        auto receiver = (fixture.m_Directory / "reserve-receiver").string();
        Buffer root(16, 0x65);
        std::set<std::array<uint8_t, 16>> used;
        std::vector<PublishRequest> sent;
        auto encrypt = [&](PayloadCipher &cipher, std::string topic)
        {
            PublishRequest publication;
            publication.m_Topic = std::move(topic);
            publication.m_Payload = {'R'};
            cipher.Encrypt(publication, 1);
            EMPY_REQUIRE(used.insert(publication.m_Encryption->m_Nonce).second);
            sent.push_back(publication);
            return NonceCounter(publication);
        };
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 1, 4, 16, sender, 4);
            for (uint64_t i = 0; i < 3; ++i)
            {
                EMPY_REQUIRE(encrypt(cipher, "reserve/a") == i);
            }
        }
        {
            // A restart resumes at the reserved bound, skipping but never reusing counters.
            PayloadCipher cipher(KeyGrant{root}, 1, 1, 4, 16, sender, 4);
            for (uint64_t i = 4; i < 10; ++i)
            {
                EMPY_REQUIRE(encrypt(cipher, i % 2 ? "reserve/a" : "reserve/b") == i);
            }
        }
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 1, 4, 16, sender, 4);
            EMPY_REQUIRE(encrypt(cipher, "reserve/a") == 12);
            Throws([] { PayloadCipher invalid(KeyGrant{Buffer(16, 1)}, 1, 1, 4, 16, {}, 0); });
        }
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 1, 4, 16, receiver);
            for (const auto &publication : sent)
            {
                EMPY_REQUIRE(cipher.Decrypt(publication, publication.m_Topic) == Buffer{'R'});
            }
        }
        {
            FileStore store(receiver, 1 << 20, 64);
            EMPY_REQUIRE(!store.Get("payload") && store.Get("payload/policy") &&
                         store.Keys("payload/replay/").size() == 2);
        }
        {
            // Each replay tuple record survives restart independently.
            PayloadCipher cipher(KeyGrant{root}, 1, 1, 4, 16, receiver);
            for (const auto &publication : sent)
            {
                Throws([&] { cipher.Decrypt(publication, publication.m_Topic); });
            }
        }

        // Migrate the legacy single-record layout and then erase it.
        auto legacy = (fixture.m_Directory / "legacy").string();
        PayloadCipher producer(KeyGrant{root}, 1, 2);
        PublishRequest seen, fresh;
        seen.m_Topic = fresh.m_Topic = "legacy/value";
        seen.m_Payload = {'S'};
        fresh.m_Payload = {'F'};
        producer.Encrypt(seen, 1);
        producer.Encrypt(fresh, 1);
        {
            Writer binding;
            binding.String("MMQP/PAYLOAD/STATE/1");
            binding.String("");
            binding.U8(0);
            Writer writer;
            writer.U32(0x45504331);
            writer.Bytes(Hmac256(root, binding.Data()));
            writer.U32(1);
            writer.U32(2);
            NonceSequence({7, 7, 7, 7, 7, 7, 7, 7}, 41).Save(writer);
            writer.U32(1);
            writer.String("legacy/value");
            writer.U32(1);
            writer.Bytes(std::span(seen.m_Encryption->m_Nonce).first(8));
            writer.U64(NonceCounter(seen));
            writer.U8(1);
            for (size_t i = 1; i < 32; ++i)
            {
                writer.U8(0);
            }
            FileStore store(legacy, 1 << 20, 1);
            store.Put("payload", writer.Take());
        }
        for (uint64_t expected : {41ULL, 41ULL + 1024})
        {
            PayloadCipher cipher(KeyGrant{root}, 1, 2, 4, 16, legacy);
            Throws([&] { cipher.Decrypt(seen, seen.m_Topic); });
            PublishRequest own;
            own.m_Topic = "legacy/own";
            own.m_Payload = {'O'};
            cipher.Encrypt(own, 2);
            EMPY_REQUIRE(own.m_Encryption->m_Nonce[0] == 7 && NonceCounter(own) == expected);
            if (expected == 41)
            {
                EMPY_REQUIRE(cipher.Decrypt(fresh, fresh.m_Topic) == Buffer{'F'});
            }
        }
        FileStore store(legacy, 1 << 20, 64);
        EMPY_REQUIRE(!store.Get("payload") && store.Get("payload/nonce") && store.Keys("payload/replay/").size() == 1);
    }

    void Standalone()
    {
        Fixture fixture;
        eMQ::Server::Configs options;
        options.m_Port = 0;
        options.m_StateDirectory = (fixture.m_Directory / "broker").string();
        options.m_MaxInflight = 2;
        options.m_RetryInterval = 100;
        auto broker = std::make_unique<Broker>(options);
        auto port = broker->Listen();
        Client publisher, subscriber;
        publisher.Connect("127.0.0.1", port);
        auto sid = subscriber.Connect("127.0.0.1", port);
        Okay(publisher.Register("durable/value", 1, 60, 8));
        for (uint8_t i = 0; i < 8; ++i)
        {
            Okay(publisher.Publish("durable/value", {i}, 1));
        }
        std::vector<uint8_t> values;
        subscriber.On<Receive>("durable/#", [&](const auto &event) { values.push_back(event.m_Message.m_Payload[0]); });
        Okay(subscriber.Subscribe("durable/#"));
        Await([&] { return values.size() == 8; }, {&subscriber});
        for (size_t i = 0; i < values.size(); ++i)
        {
            EMPY_REQUIRE(values[i] == i);
        }
        subscriber.Disconnect();
        publisher.Disconnect();
        broker.reset();
        broker = std::make_unique<Broker>(options);
        port = broker->Listen();
        EMPY_REQUIRE(subscriber.Connect("127.0.0.1", port, true) == sid);
        Client late;
        size_t replayed = 0;
        late.On<Receive>("durable/#", [&](const auto &) { ++replayed; });
        late.Connect("127.0.0.1", port);
        Okay(late.Subscribe("durable/#"));
        Await([&] { return replayed == 8; }, {&subscriber, &late});
        late.Disconnect();
        subscriber.Disconnect();
    }

    void Journal()
    {
        Fixture fixture;
        auto directory = fixture.m_Directory / "journal";
        auto log = [&]
        {
            for (const auto &entry : std::filesystem::directory_iterator(directory))
            {
                if (entry.path().extension() == ".log")
                {
                    return entry.path();
                }
            }
            throw std::runtime_error("Journal file is missing");
        };
        {
            FileStore store(directory);
            EMPY_REQUIRE(store.Journal("test").empty());
            for (uint8_t i = 1; i <= 3; ++i)
            {
                store.Append("test", Buffer(i, i));
            }
            EMPY_REQUIRE(store.JournalBytes("test") > 0);
        }
        // A crash during the last flush leaves a torn tail, which is dropped.
        auto intact = std::filesystem::file_size(log());
        {
            std::ofstream file(log(), std::ios::binary | std::ios::app);
            file.write("\x45\x4A\x4E\x31\x00\x00\x00\x09"
                       "abc",
                       11);
        }
        {
            FileStore store(directory);
            auto entries = store.Journal("test");
            EMPY_REQUIRE(entries.size() == 3 && entries[2] == Buffer(3, 3));
            EMPY_REQUIRE(std::filesystem::file_size(log()) == intact);
            store.Append("test", Buffer{4});
            EMPY_REQUIRE(store.Journal("test").size() == 4);
        }
        // Damage before the final entry fails closed.
        {
            std::fstream file(log(), std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(10);
            file.put('\x7F');
        }
        {
            FileStore store(directory);
            Throws([&] { store.Journal("test"); });
            store.Reset("test");
            EMPY_REQUIRE(store.Journal("test").empty());
        }

        // A complete final record with a bad checksum is corruption, not a torn tail.
        {
            FileStore store(directory);
            store.Append("test", Buffer{1, 2, 3});
        }
        const auto completeBytes = std::filesystem::file_size(log());
        {
            std::fstream file(log(), std::ios::binary | std::ios::in | std::ios::out);
            file.seekg(-1, std::ios::end);
            const auto byte = file.get();
            EMPY_REQUIRE(byte != std::char_traits<char>::eof());
            file.seekp(-1, std::ios::end);
            file.put(static_cast<char>(byte ^ 1));
        }
        {
            FileStore store(directory);
            Throws([&] { store.Journal("test"); });
            EMPY_REQUIRE(std::filesystem::file_size(log()) == completeBytes);
        }

        // Journaled stores migrate per-record files, replay flushed appends and compact into a snapshot.
        auto journaled = fixture.m_Directory / "journaled";
        {
            FileStore legacy(journaled);
            legacy.Put("old", Buffer{1});
            legacy.Put("gone", Buffer{2});
        }
        {
            FileStore store(journaled, 64 << 20, 1024, true);
            EMPY_REQUIRE(store.Get("old") == Buffer{1} && store.Get("gone") == Buffer{2});
            store.Erase("gone");
            store.Put("new", Buffer(10, 3));
        }
        {
            FileStore store(journaled, 64 << 20, 1024, true);
            EMPY_REQUIRE(!store.Get("gone") && store.Get("new") == Buffer(10, 3) && store.Get("old") == Buffer{1});
            for (int i = 0; i < 40; ++i)
            {
                store.Put("big" + std::to_string(i % 4), Buffer(256 * 1024, static_cast<uint8_t>(i)));
            }
            EMPY_REQUIRE(std::filesystem::exists(journaled / "store.snapshot"));
        }
        {
            FileStore store(journaled, 64 << 20, 1024, true);
            EMPY_REQUIRE(store.Get("big3") == Buffer(256 * 1024, 39) && !store.Get("gone") &&
                         store.Get("old") == Buffer{1});
            for (const auto &entry : std::filesystem::directory_iterator(journaled))
            {
                EMPY_REQUIRE(entry.path().extension() != ".rec");
            }
        }

        // Large retained queues force compaction; restart restores the compacted base.
        eMQ::Server::Configs options;
        options.m_Port = 0;
        options.m_StateDirectory = (fixture.m_Directory / "compacted").string();
        options.m_MaxRetainedBytes = 8 * 1024 * 1024;
        options.m_MaxStateBytes = 32 * 1024 * 1024;
        auto broker = std::make_unique<Broker>(options);
        auto port = broker->Listen();
        Client publisher;
        publisher.Connect("127.0.0.1", port);
        Okay(publisher.Register("compact/value", 1, 60, 8));
        for (uint8_t i = 0; i < 12; ++i)
        {
            Okay(publisher.Publish("compact/value", Buffer(200 * 1024, i), 1));
        }
        publisher.Disconnect();
        broker.reset();
        {
            FileStore store(options.m_StateDirectory, options.m_MaxStateBytes);
            auto base = store.Get("broker");
            EMPY_REQUIRE(base && base->size() > 4 && (*base)[3] == '6');
            EMPY_REQUIRE(store.Journal("broker").size() < 12);
        }
        broker = std::make_unique<Broker>(options);
        port = broker->Listen();
        Client late;
        std::vector<uint8_t> values;
        late.On<Receive>("compact/#", [&](const auto &event) { values.push_back(event.m_Message.m_Payload[0]); });
        late.Connect("127.0.0.1", port);
        Okay(late.Subscribe("compact/#"));
        Await([&] { return values.size() == 8; }, {&late});
        for (size_t i = 0; i < values.size(); ++i)
        {
            EMPY_REQUIRE(values[i] == i + 4);
        }
        late.Disconnect();
    }

    void Tenants()
    {
        Fixture fixture;
        eMQ::Server::Configs options;
        options.m_Port = 0;
        options.m_Transport.m_Tls = true;
        options.m_Transport.m_Certificate = fixture.m_Certificates[0];
        options.m_Transport.m_PrivateKey = fixture.m_Keys[0];
        options.m_Tenants = {{"alpha", "alpha", Buffer(32, 0xA1), {}}, {"beta", "beta", Buffer(32, 0xB2), {}}};
        Broker broker(options);
        auto port = broker.Listen();
        eMQ::Configs alpha;
        alpha.m_Transport.m_Tls = true;
        alpha.m_Transport.m_Authority = fixture.m_Authority;
        alpha.m_Transport.m_ServerName = "localhost";
        alpha.m_AuthIdentity = "alpha";
        alpha.m_AuthKey = Buffer(32, 0xA1);
        auto beta = alpha;
        beta.m_AuthIdentity = "beta";
        beta.m_AuthKey = Buffer(32, 0xB2);
        Client a(alpha), b(beta);
        a.Connect("127.0.0.1", port);
        b.Connect("127.0.0.1", port);
        Okay(a.Register("alpha/value"));
        Okay(b.Subscribe("beta/#"));
        EMPY_REQUIRE(a.Subscribe("beta/#").get().m_Code == MMQP_REASON_UNAUTHORIZED);
        EMPY_REQUIRE(b.Register("alpha/value").get().m_Code == MMQP_REASON_UNAUTHORIZED);
        auto wrong = alpha;
        wrong.m_AuthIdentity = "beta";
        Client rejected(wrong);
        Throws([&] { rejected.Connect("127.0.0.1", port); });
        a.Disconnect();
        b.Disconnect();
    }

    void Quorum(bool quic)
    {
        Fixture fixture;
        auto options = fixture.Options(quic);
        std::array<std::unique_ptr<FabricNode>, 3> nodes;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            nodes[i] = std::make_unique<FabricNode>(options[i]);
            nodes[i]->Start();
        }
        EMPY_REQUIRE(nodes[0]->Acquire("session-test").empty());
        nodes[0]->Commit("session-test", {'a'});
        Throws([&] { nodes[1]->Acquire("session-test", 200); });
        EMPY_REQUIRE(nodes[0]->Owns("session-test"));
        auto observation = nodes[0]->Inspect({"session-test", "missing"});
        EMPY_REQUIRE(observation.m_Peers.size() == 2 && observation.m_ActiveLeases >= 1);
        EMPY_REQUIRE(observation.m_Ownership.contains("session-test") && !observation.m_Ownership.contains("missing"));
        EMPY_REQUIRE(observation.m_Ownership.at("session-test").m_RemainingMs > 0);
        EMPY_REQUIRE(nodes[2]->Read("session-test") == Buffer{'a'});
        nodes[0].reset();
        EMPY_REQUIRE(nodes[1]->Acquire("session-test", 6000) == Buffer{'a'});
        nodes[1]->Commit("session-test", {'b'});
        nodes[0] = std::make_unique<FabricNode>(options[0]);
        nodes[0]->Start();
        EMPY_REQUIRE(nodes[0]->Read("session-test") == Buffer{'b'});
        nodes[0].reset();
        nodes[2].reset();
        Throws([&] { nodes[1]->Commit("session-test", {'c'}); });
        EMPY_REQUIRE(!nodes[1]->Owns("session-test"));
        EMPY_REQUIRE(!nodes[1]->Inspect({"session-test"}).m_Ownership.contains("session-test"));
    }

    void Inboxes()
    {
        Fixture fixture;
        auto options = fixture.Options(false);
        std::array<std::unique_ptr<FabricNode>, 3> nodes;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            options[i].m_Transport.m_MaxPacket = 1024 * 1024;
            options[i].m_MaxInboxRecords = 14;
            nodes[i] = std::make_unique<FabricNode>(options[i]);
            nodes[i]->Start();
        }
        ProfileEntry profile;
        profile.m_Type = MMQP_PROREQ_TYPE_SUB;
        profile.m_Topic = "large/value";
        profile.m_Qos = 1;
        std::vector<ClusterProfile> targets{{"", "large-inbox", profile, UINT64_MAX, false}};
        auto expiry = UnixMilliseconds() + 60000;
        auto publish = [&](uint64_t sequence)
        {
            PublishRequest publication;
            publication.m_Topic = "large/value";
            publication.m_Qos = 1;
            publication.m_PacketId = static_cast<uint16_t>(sequence);
            publication.m_Payload = Buffer(160 * 1024, static_cast<uint8_t>(sequence));
            nodes[sequence % nodes.size()]->Publish("large-sender", sequence, publication, targets, expiry);
        };
        // The inbox exceeds a single 1 MiB scan reply, so it is read in several pages.
        for (uint64_t sequence = 1; sequence <= 14; ++sequence)
        {
            publish(sequence);
        }
        auto inbox = nodes[1]->Inbox("large-inbox");
        EMPY_REQUIRE(inbox.size() == 14);
        for (size_t i = 0; i < inbox.size(); ++i)
        {
            EMPY_REQUIRE(inbox[i].m_Publication.m_PacketId == i + 1 &&
                         inbox[i].m_Publication.m_Payload == Buffer(160 * 1024, static_cast<uint8_t>(i + 1)));
        }
        // The per-session record budget rejects admission instead of accepting it.
        uint8_t reason = 0;
        try
        {
            publish(15);
        }
        catch (const ProtocolError &error)
        {
            reason = error.m_Reason;
        }
        EMPY_REQUIRE(reason == MMQP_REASON_BUSY);
        EMPY_REQUIRE(nodes[2]->Inbox("large-inbox").size() == 14);
        // A consumed record releases its budget; the retry keeps the original identity.
        nodes[0]->Acquire("large-inbox");
        nodes[0]->Acknowledge(inbox.front());
        publish(15);
        auto drained = nodes[0]->Inbox("large-inbox");
        EMPY_REQUIRE(drained.size() == 14 && drained.back().m_Publication.m_PacketId == 15);
        EMPY_REQUIRE(std::none_of(drained.begin(), drained.end(),
                                  [&](const auto &item) { return item.m_Identity == inbox.front().m_Identity; }));
    }

    void Brokers(bool quic)
    {
        Fixture fixture;
        auto settings = fixture.Options(quic);
        std::array<eMQ::Server::Configs, 3> options;
        std::array<std::unique_ptr<Broker>, 3> brokers;
        std::array<uint16_t, 3> ports{};
        for (size_t i = 0; i < 3; ++i)
        {
            options[i].m_Port = 0;
            options[i].m_Cluster = settings[i];
            options[i].m_RetryInterval = 100;
            options[i].m_RequestTimeout = 15000;
            brokers[i] = std::make_unique<Broker>(options[i]);
            ports[i] = brokers[i]->Listen();
        }
        Client publisher;
        eMQ::Configs client;
        client.m_AutoReconnect = true;
        client.m_FailoverEndpoints = {{"127.0.0.1", ports[1]}, {"127.0.0.1", ports[2]}};
        Client subscriber(client);
        std::vector<eCore::SharedBytes> received;
        subscriber.On<Receive>("cluster/#", [&](const auto &event) { received.push_back(event.m_Message.m_Payload); });
        auto sid = subscriber.Connect("127.0.0.1", ports[0]);
        publisher.Connect("127.0.0.1", ports[2]);
        Okay(subscriber.Subscribe("cluster/#"));
        Okay(publisher.Register("cluster/value", 1, 60, 4));
        Okay(publisher.Publish("cluster/value", {'A'}, 1));
        Await([&] { return received.size() == 1; }, {&subscriber});
        EMPY_REQUIRE(received.back() == Buffer{'A'});
        auto telemetry = brokers[0]->Inspect();
        EMPY_REQUIRE(telemetry.m_Cluster && telemetry.m_Durable && telemetry.m_Cluster->m_Replicas == 3);
        EMPY_REQUIRE(telemetry.m_StoredBytes && *telemetry.m_StoredBytes > 0 &&
                     *telemetry.m_StoredBytes <= telemetry.m_MaxStateBytes);
        EMPY_REQUIRE(telemetry.m_Sessions.size() == 1 && telemetry.m_Sessions.front().m_Owner &&
                     telemetry.m_Sessions.front().m_Lease);
        Client sharedA, sharedB, responder;
        size_t first = 0, second = 0, responses = 0;
        sharedA.On<Receive>("work/#", [&](const auto &) { ++first; });
        sharedB.On<Receive>("work/#", [&](const auto &) { ++second; });
        publisher.On<Receive>("rpc/cluster",
                              [&](const auto &event)
                              {
                                  EMPY_REQUIRE(event.m_Message.m_Payload == Buffer{'R'});
                                  ++responses;
                              });
        sharedA.Connect("127.0.0.1", ports[0]);
        sharedB.Connect("127.0.0.1", ports[1]);
        responder.Connect("127.0.0.1", ports[1]);
        responder.On<Receive>("rpc/cluster",
                              [&](const auto &event) { Okay(responder.Respond(event.m_Message, {'R'})); });
        Okay(sharedA.Subscribe("$share/workers/work/#"));
        Okay(sharedB.Subscribe("$share/workers/work/#"));
        Okay(publisher.Register("work/task"));
        for (int i = 0; i < 4; ++i)
        {
            Okay(publisher.Publish("work/task", {'W'}, 1));
        }
        Await([&] { return first + second == 4; }, {&sharedA, &sharedB});
        EMPY_REQUIRE(first == 2 && second == 2);
        Okay(responder.Subscribe("rpc/cluster", 1, true));
        Okay(publisher.Register("rpc/cluster", 0, 60, 1, true));
        for (uint8_t qos : {uint8_t{1}, uint8_t{0}})
        {
            auto before = responses;
            Okay(publisher.Request("rpc/cluster", {'?'}, 1, qos));
            Await([&] { return responses > before; }, {&publisher, &responder});
        }

        Client requester(client), movingResponder(client);
        auto requesterId = requester.Connect("127.0.0.1", ports[0]);
        auto responderId = movingResponder.Connect("127.0.0.1", ports[0]);
        size_t requesterDisconnects = 0, responderDisconnects = 0;
        requester.On<Disconnect>([&](const auto &) { ++requesterDisconnects; });
        movingResponder.On<Disconnect>([&](const auto &) { ++responderDisconnects; });
        std::vector<PublishRequest> fixedRequests, movingRequests, publisherReplies, requesterReplies;
        responder.On<Receive>("rpc/concurrent", [&](const auto &event) { fixedRequests.push_back(event.m_Message); });
        movingResponder.On<Receive>("rpc/concurrent",
                                    [&](const auto &event) { movingRequests.push_back(event.m_Message); });
        publisher.On<Receive>("rpc/concurrent",
                              [&](const auto &event) { publisherReplies.push_back(event.m_Message); });
        requester.On<Receive>("rpc/concurrent",
                              [&](const auto &event) { requesterReplies.push_back(event.m_Message); });
        Okay(responder.Subscribe("rpc/concurrent", 1, true));
        Okay(movingResponder.Subscribe("rpc/concurrent", 1, true));
        Okay(publisher.Register("rpc/concurrent", 0, 60, 1, true));
        Okay(requester.Register("rpc/concurrent", 0, 60, 1, true));
        for (uint16_t id : {uint16_t{350}, uint16_t{351}})
        {
            Okay(publisher.Request("rpc/concurrent", {'P', static_cast<uint8_t>(id - 350)}, 2, 1, id));
            Okay(requester.Request("rpc/concurrent", {'Q', static_cast<uint8_t>(id - 350)}, 2, 1, id));
        }
        Await([&] { return fixedRequests.size() == 4 && movingRequests.size() == 4; }, {&responder, &movingResponder});
        std::set<uint16_t> routes;
        for (const auto &request : fixedRequests)
        {
            EMPY_REQUIRE(routes.insert(request.m_CorrelationId).second);
            Okay(responder.Respond(request, request.m_Payload));
        }
        Await([&] { return publisherReplies.size() == 2 && requesterReplies.size() == 2; }, {&publisher, &requester});
        sharedA.Disconnect();
        sharedB.Disconnect();
        brokers[0].reset();
        Await(
            [&]
            {
                return requesterDisconnects && responderDisconnects && requester.Connected() &&
                       movingResponder.Connected();
            },
            {&requester, &movingResponder, &subscriber}, 12000);
        EMPY_REQUIRE(requester.Identifier() == requesterId && movingResponder.Identifier() == responderId);
        EMPY_REQUIRE(responder.Respond(fixedRequests[0], {'X'}).get().m_Code == MMQP_REASON_INVALID_CORRELATION);
        for (auto it = movingRequests.rbegin(); it != movingRequests.rend(); ++it)
        {
            Okay(movingResponder.Respond(*it, it->m_Payload));
        }
        Await([&] { return publisherReplies.size() == 4 && requesterReplies.size() == 4; }, {&publisher, &requester});
        for (auto replies : {&publisherReplies, &requesterReplies})
        {
            std::map<uint16_t, size_t> counts;
            for (const auto &reply : *replies)
            {
                EMPY_REQUIRE(reply.m_Payload[0] == (replies == &publisherReplies ? 'P' : 'Q'));
                EMPY_REQUIRE(reply.m_CorrelationId == 350 + reply.m_Payload[1]);
                ++counts[reply.m_CorrelationId];
            }
            EMPY_REQUIRE(counts[350] == 2 && counts[351] == 2);
        }
        requester.Disconnect();
        movingResponder.Disconnect();
        responder.Disconnect();
        Await([&] { return subscriber.Connected(); }, {&subscriber}, 9000);
        EMPY_REQUIRE(subscriber.Identifier() == sid);
        Okay(publisher.Publish("cluster/value", {'B'}, 1));
        Await([&] { return received.size() == 2; }, {&subscriber});
        EMPY_REQUIRE(received.back() == Buffer{'B'});
        Client late;
        std::vector<eCore::SharedBytes> retained;
        late.On<Receive>("cluster/#", [&](const auto &event) { retained.push_back(event.m_Message.m_Payload); });
        late.Connect("127.0.0.1", ports[1]);
        Okay(late.Subscribe("cluster/#"));
        Await([&] { return retained.size() == 2; }, {&subscriber, &late});
        EMPY_REQUIRE(retained[0] == Buffer{'A'} && retained[1] == Buffer{'B'});
        late.Disconnect();
        subscriber.Disconnect();
        publisher.Disconnect();
        for (auto &broker : brokers)
        {
            broker.reset();
        }
        for (size_t i = 0; i < 3; ++i)
        {
            brokers[i] = std::make_unique<Broker>(options[i]);
            ports[i] = brokers[i]->Listen();
        }
        Client restarted;
        size_t replay = 0;
        restarted.On<Receive>("cluster/#", [&](const auto &) { ++replay; });
        restarted.Connect("127.0.0.1", ports[1]);
        Okay(restarted.Subscribe("cluster/#"));
        Await([&] { return replay == 2; }, {&restarted});
        restarted.Disconnect();
    }
} // namespace

EMPY_TEST(Recovery, ReferenceCases)
{
    Storage();
    SnapshotSchemas();
    PayloadState();
    PayloadReservation();
    Standalone();
    Journal();
    Tenants();
    Quorum(false);
    Inboxes();
    Brokers(false);
}

EMPY_TEST(FabricQuic, ReferenceCases)
{
    EMPY_REQUIRE(QuicAvailable());
    Quorum(true);
    Brokers(true);
}
