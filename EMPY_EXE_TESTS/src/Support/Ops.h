#pragma once
#include "../Checks.h"
#include <eOps/Common/Text.h>
#include <eOps/Fabric/Bus.h>
#include <eMQ/Client.h>
#include <Runtime/Runtime.h>
#include <memory>
#include <string>
#include <cstdint>
#include <utility>
#include <filesystem>
#include <string_view>

// fixtures shared by the Agentic-MLOps suites (EMPY_EXE_TESTS/src/Ops): the shipped Configs/MLOps
// files, scratch directories, an in-process broker and BusModule client peers

namespace eTest::Ops
{
    // the repository root (the directory holding Configs/MLOps) above the test data directory; a missing
    // Configs/MLOps fails the case instead of surfacing later as confusing YAML load errors
    inline std::filesystem::path RepositoryRoot()
    {
        std::filesystem::path path = std::filesystem::absolute(EMPY_TEST_DATA_DIR);
        for (int32_t i = 0; i < 6 && !path.empty(); ++i)
        {
            if (std::filesystem::exists(path / "Configs" / "MLOps"))
            {
                return path;
            }
            path = path.parent_path();
        }
        ::eTest::Fail(__FILE__, __LINE__, "Configs/MLOps not found above " + std::string(EMPY_TEST_DATA_DIR));
        throw ::eTest::AbortCase{};
    }

    // a shipped Configs/MLOps file
    inline std::filesystem::path ConfigFile(std::string_view name)
    {
        return RepositoryRoot() / "Configs" / "MLOps" / std::string(name);
    }

    // unique directory under the system temp path, removed on destruction
    struct TempDir
    {
        std::filesystem::path m_Path;

        explicit TempDir(std::string_view prefix = "EmpyOps")
            : m_Path(std::filesystem::temp_directory_path() / (std::string(prefix) + "-" + eOps::NewId({})))
        {
            std::filesystem::create_directories(m_Path);
        }
        ~TempDir()
        {
            std::error_code ignored;
            std::filesystem::remove_all(m_Path, ignored);
        }
        TempDir(const TempDir &) = delete;
        TempDir &operator=(const TempDir &) = delete;
    };

    // in-process broker on an ephemeral loopback port; `requestTimeoutMs` > 0 bounds request/response
    // exchanges (longer than any agent, workflow or HITL round trip of the suite)
    struct LocalBroker
    {
        eMQ::Server::Configs m_Settings;
        eMQ::Broker m_Broker;
        uint16_t m_Port;

        explicit LocalBroker(uint32_t requestTimeoutMs = 0)
            : m_Settings(Settings(requestTimeoutMs)), m_Broker(m_Settings), m_Port(m_Broker.Listen())
        {
        }

        static eMQ::Server::Configs Settings(uint32_t requestTimeoutMs)
        {
            eMQ::Server::Configs settings;
            settings.m_Port = 0;
            if (requestTimeoutMs > 0)
            {
                settings.m_RequestTimeout = requestTimeoutMs;
            }
            return settings;
        }
    };

    // client settings whose response timeout (> 0) covers the suite's longest exchange
    inline eMQ::Configs ClientConfigs(uint32_t responseTimeoutMs = 0)
    {
        eMQ::Configs configs;
        if (responseTimeoutMs > 0)
        {
            configs.m_ResponseTimeout = responseTimeoutMs;
        }
        return configs;
    }

    // a client runtime connected to a local broker with an attached BusModule. A started peer has run its
    // first Poll (OnAttach of every module) and polls once more after each Attach; an unstarted one collects
    // modules until Start
    struct Peer
    {
        std::unique_ptr<eMQ::Client> m_Client;
        bool m_Started = false;

        Peer(uint16_t port, std::string source, eMQ::Configs configs = {}, bool start = true)
            : m_Client(std::make_unique<eMQ::Client>(std::move(configs)))
        {
            m_Client->Connect("127.0.0.1", port);
            eOps::Fabric::BusOptions options;
            options.m_Source = std::move(source);
            m_Client->AttachModule<eOps::Fabric::BusModule>(options);
            if (start)
            {
                Start();
            }
        }
        ~Peer()
        {
            m_Client->Shutdown();
        }
        Peer(const Peer &) = delete;
        Peer &operator=(const Peer &) = delete;

        void Start()
        {
            m_Client->Poll();
            m_Started = true;
        }

        eOps::Fabric::Bus &Get()
        {
            return eOps::Fabric::BusOf(*m_Client);
        }

        template <typename _Module, typename... _Args> _Module *Attach(_Args &&...args)
        {
            _Module *module = m_Client->AttachModule<_Module>(std::forward<_Args>(args)...);
            if (m_Started)
            {
                m_Client->Poll();
            }
            return module;
        }
    };
} // namespace eTest::Ops
