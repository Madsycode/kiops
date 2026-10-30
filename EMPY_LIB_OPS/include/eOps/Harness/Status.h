#pragma once
#include <memory>
#include <string>
#include <cstdint>
#include "../Common/Json.h"

// read-only HTTP status endpoint of the harness (eOps::HttpServer, loopback by default):
//   GET /api/mlops/status  → latest snapshot JSON
//   GET /api/mlops/healthz → {"status": "ok", "time": <Unix ms>, "snapshotAge": <ms>}
// other paths → 404, other methods → 405. The owner thread pushes snapshots with Update; the
// server thread only reads a copy under a mutex.

namespace eOps::Harness
{
    class EMPY_OPS_API StatusServer
    {
    public:
        explicit StatusServer(std::string address = "127.0.0.1", uint16_t port = 0);
        ~StatusServer(); // Stop
        StatusServer(const StatusServer &) = delete;
        StatusServer &operator=(const StatusServer &) = delete;

        uint16_t Start(); // bound port; throws HttpError
        void Stop();
        void Update(Json snapshot); // any thread
        uint16_t Port() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
} // namespace eOps::Harness
