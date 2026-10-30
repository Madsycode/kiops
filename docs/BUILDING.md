# Building and packaging

## Targets and configurations

The root CMake project builds the original two shared libraries and three
executables. `EMPY_BUILD_TESTS=OFF` omits only the test executable.

The optional Agentic-MLOps extension ([MLOPS.md](MLOPS.md)) adds three targets,
built after `EmpyClient` while `EMPY_BUILD_MLOPS` is ON (the default):

| Target | Directory | Kind | Links |
| --- | --- | --- | --- |
| `EmpyOps` | `EMPY_LIB_OPS` | shared library (`EmpyOps.dll`) | PUBLIC `EmpyMQ`; PRIVATE Asio, OpenSSL (and `crypt32` on Windows) for its HTTPS client |
| `EmpyHarness` | `EMPY_EXE_HARNESS` | executable | PRIVATE `EmpyOps` |
| `EmpyWorkload` | `EMPY_EXE_WORKLOAD` | executable | PRIVATE `EmpyOps` |

With the option ON, `EmpyTests` links `EmpyOps`, defines `EMPY_WITH_MLOPS=1`,
builds the `EMPY_EXE_TESTS/src/Ops` suites (`OpsCommon` … `OpsIntegration`), and
depends on `EmpyHarness`/`EmpyWorkload` and their staging targets, because the
process-level tests launch them from beside `EmpyTests`. With `-DEMPY_BUILD_MLOPS=OFF`
the three targets and the `src/Ops` sources are omitted, and the original targets
build and test unchanged. The extension adds no third-party dependency: it uses
the existing Asio, OpenSSL and the vendored YAML parser.

| Configuration | Purpose | GNU/Clang configuration |
| --- | --- | --- |
| Debug | Development and diagnostics | `-O0 -g3 -fno-omit-frame-pointer`, standard-library assertions, debug logging |
| Release | Optimised operation | `-O2`, `NDEBUG`, trace/debug logging compiled out |

The common warnings are `-Wall -Wextra`; MSVC uses `/W4 /permissive- /utf-8`.
MinGW `-Wa,-mbig-obj` and MSVC `/bigobj` accommodate the existing header-implemented
broker in Debug. Release does not remove input validation or resource limits.
Unsupported single-configuration build types are rejected. No sanitizer or
benchmark configuration is added.

Requirements are CMake 3.21+, Conan 2, and C++20 including `std::format`.
Dependencies remain Asio 1.38.0, OpenSSL 3.5.2, the existing vendored EnTT/YAML/Rang,
and the pinned Ascon source. MsQuic is optional and uses its native API v2.

## Windows

Use native MinGW executables on PATH, not the MSYS POSIX compiler. A typical
PowerShell session for this workspace is:

```powershell
$env:PATH = 'C:/Libraries/msys64/ucrt64/bin;' + $env:PATH
./tools/Install-MsQuic.ps1
./EmpyWin.bat Debug
./EmpyWin.bat Release
```

Configure a Conan profile matching the compiler/architecture before the first
install. The wrapper supplies absolute compiler paths to Conan so its MSYS build
tools cannot silently substitute another GCC. It defaults to four build jobs;
`CMAKE_BUILD_PARALLEL_LEVEL` overrides that choice. Dependencies' generated CMake
files coexist under `Targets/Conan`; compiled targets use `Targets/Debug` and
`Targets/Release`.

`Install-MsQuic.ps1` installs a local x64 SDK, verifies every download with SHA-256,
preserves upstream notices, and sets `MSQUIC_ROOT` and `EMPY_MSQUIC_RUNTIME` in the
calling PowerShell session. It uses the
[Microsoft MsQuic OpenSSL 2.6.1 package](https://www.nuget.org/packages/Microsoft.Native.Quic.MsQuic.OpenSSL/2.6.1).
The runtime is 2.6.1; the stable API v2 headers are unmodified upstream 2.4.8 headers
compatible with this MinGW SDK. The 2.6.1 Windows helper header refers to
`RtlIpv6StringToAddressA`, absent from this SDK, and uses a different constness for
`RtlIpv4StringToAddressA`. Older **headers**, rather than an older runtime or
`-fpermissive`, avoid those unused helper incompatibilities.

PEM certificate configurations and automated QUIC tests require the OpenSSL
MsQuic provider. A Schannel runtime instead uses Windows certificate credentials;
the broker supports `certificate_hash` for that deployment, but the test
certificates are PEM files. Merely finding `msquic.dll` does not identify its TLS
provider. `EMPY_MSQUIC_RUNTIME` selects the file to stage explicitly.

For an existing CMake cache, select all SDK paths explicitly once:

```powershell
$sdk = (Resolve-Path Targets/Dependencies/MsQuic).Path
cmake -S . -B Targets/Debug `
    "-DEMPY_MSQUIC_ROOT=$sdk" `
    "-DEMPY_MSQUIC_INCLUDE_DIR=$sdk/include" `
    "-DEMPY_MSQUIC_LIBRARY=$sdk/lib/msquic.lib" `
    "-DEMPY_MSQUIC_RUNTIME=$sdk/bin/msquic.dll"
cmake --build Targets/Debug --parallel 4
ctest --test-dir Targets/Debug -C Debug --output-on-failure --parallel 4
```

Use `Targets/Release` and `-DCMAKE_BUILD_TYPE=Release` for Release. Fresh CMake
configuration also needs `-G "MinGW Makefiles"` and
`-DCMAKE_TOOLCHAIN_FILE=<absolute workspace>/Targets/Conan/conan_toolchain.cmake`.

Runtime staging walks the actual PE import graph, copies transitive compiler
and framework DLLs, includes explicitly selected dynamic providers, and records
SHA-256 hashes in `EmpyBroker.exe.runtime.txt`, `EmpyClient.exe.runtime.txt`, and
`EmpyTests.exe.runtime.txt`, and with the extension also in
`EmpyHarness.exe.runtime.txt` and `EmpyWorkload.exe.runtime.txt` (which include
`EmpyOps.dll`). It fails for missing dependencies. OpenSSL dynamic
compression providers are included only when enabled in the linked provider's
headers; the installed Conan OpenSSL is static and does not require them.
`EMPY_RUNTIME_DIRECTORIES` and `EMPY_RUNTIME_LIBRARIES` are explicit escape hatches
for a different toolchain. A successful rebuild also refreshes staging when the
executable did not relink.

## Linux

With a suitable Conan compiler profile and native C++20 toolchain:

```bash
bash ./EmpyLinux.sh Debug
bash ./EmpyLinux.sh Release
```

Install a supported native MsQuic shared library/development package or set
`EMPY_MSQUIC_ROOT` through CMake. The Windows installer is not used on Linux.
Executable/shared-library search paths use `$ORIGIN` for the adjacent framework
libraries. Broker lease timing uses `CLOCK_BOOTTIME`; on Windows it uses
`GetTickCount64`, so host suspension advances lease expiry. Other OS clock
bindings are deliberately rejected until implemented. The MLOps extension's POSIX
branches (child processes, per-user state directory ownership checks) have never
been compiled; pass `-DEMPY_BUILD_MLOPS=OFF` if they fail on a new platform. See
VALIDATION.md for which platforms were actually exercised.

## TCP/TLS-only

This is a transport option, not a third build configuration:

```powershell
cmake -S . -B Targets/TcpOnly -G 'MinGW Makefiles' `
    -DCMAKE_BUILD_TYPE=Debug -DEMPY_WITH_QUIC=OFF `
    "-DCMAKE_TOOLCHAIN_FILE=$PWD/Targets/Conan/conan_toolchain.cmake"
cmake --build Targets/TcpOnly --parallel 4
ctest --test-dir Targets/TcpOnly --output-on-failure --parallel 4
```

QUIC-only suites are not registered in this build. Selecting QUIC at runtime
fails explicitly; it does not substitute TCP. The MLOps targets build in this
configuration too, and all of their suites run over TCP. To omit them, add
`-DEMPY_BUILD_MLOPS=OFF`:

```powershell
cmake -S . -B Targets/NoMLOps -G 'MinGW Makefiles' `
    -DCMAKE_BUILD_TYPE=Debug -DEMPY_BUILD_MLOPS=OFF `
    "-DCMAKE_TOOLCHAIN_FILE=$PWD/Targets/Conan/conan_toolchain.cmake" `
    "-DEMPY_MSQUIC_ROOT=$PWD/Targets/Dependencies/MsQuic"
```

A fresh tree finds MsQuic only through `EMPY_MSQUIC_ROOT` (or `MSQUIC_ROOT`);
without it the QUIC suites are skipped.

## Other checks

```powershell
python tools/MLOpsDemo.py --bin Targets/Debug/bin
python -m unittest discover -s EMPY_DASHBOARD/tests -p 'test_*.py' -v
node --test EMPY_DASHBOARD/tests/model.test.js
python EMPY_DASHBOARD/tests/process_smoke.py --bin Targets/Debug/bin
```

Optional browser QA uses Playwright and a Chromium-family browser as development
tools. Neither is a dashboard runtime dependency. The exact executed validation
matrix and artifact paths are recorded separately in [VALIDATION.md](VALIDATION.md).
