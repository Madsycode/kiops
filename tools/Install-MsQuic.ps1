param(
    [string] $Destination = (Join-Path $PSScriptRoot '../Targets/Dependencies/MsQuic')
)

# Local x64 SDK only; neither the OS trust store nor the system PATH is modified.
$ErrorActionPreference = 'Stop'
if (-not [Environment]::Is64BitProcess)
{
    throw 'Run this installer from 64-bit PowerShell.'
}
$sdk = [IO.Path]::GetFullPath($Destination)
$cache = Join-Path $sdk 'download'
New-Item -ItemType Directory -Force -Path $cache | Out-Null

function Get-PinnedFile([string] $Url, [string] $Path, [string] $Sha256)
{
    if ((Test-Path -LiteralPath $Path) -and
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Sha256)
    {
        return
    }
    $temporary = $Path + '.download'
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $temporary
    if ((Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash -ne $Sha256)
    {
        throw "Downloaded file checksum mismatch: $Url"
    }
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$package = Join-Path $cache 'msquic.openssl.2.6.1.zip'
Get-PinnedFile 'https://api.nuget.org/v3-flatcontainer/microsoft.native.quic.msquic.openssl/2.6.1/microsoft.native.quic.msquic.openssl.2.6.1.nupkg' `
    $package 'e8bfdd6a4d4e561fd5213eab3c617d231d2f2e5e926e3d3de682ffa540938b08'
$extracted = Join-Path $cache 'package'
Expand-Archive -LiteralPath $package -DestinationPath $extracted -Force
foreach ($directory in @('include', 'lib', 'bin'))
{
    New-Item -ItemType Directory -Force -Path (Join-Path $sdk $directory) | Out-Null
}

# Stable API v2 headers from 2.4.8 match MinGW's Windows SDK declarations.
# The newer runtime is 2.6.1. No older runtime is installed and no -fpermissive
# or modified vendor header is needed. See docs/BUILDING.md.
Get-PinnedFile 'https://raw.githubusercontent.com/microsoft/msquic/v2.4.8/src/inc/msquic.h' `
    (Join-Path $sdk 'include/msquic.h') 'e08b79148cb3ab860b330de4694385b86da8bdc0fe0b570ae3d156d6b14ed89d'
Get-PinnedFile 'https://raw.githubusercontent.com/microsoft/msquic/v2.4.8/src/inc/msquic_winuser.h' `
    (Join-Path $sdk 'include/msquic_winuser.h') '5a42d0bb41afdc5d9871a6aa86758fd69755ecc4be5a7b3bf6348b87edbffa15'
Copy-Item -LiteralPath (Join-Path $extracted 'build/native/bin/x64/msquic.dll') -Destination (Join-Path $sdk 'bin/msquic.dll') -Force
Copy-Item -LiteralPath (Join-Path $extracted 'build/native/lib/x64/msquic.lib') -Destination (Join-Path $sdk 'lib/msquic.lib') -Force
foreach ($notice in @('LICENSE', 'THIRD-PARTY-NOTICES'))
{
    Copy-Item -LiteralPath (Join-Path $extracted $notice) -Destination (Join-Path $sdk $notice) -Force
}
$env:MSQUIC_ROOT = $sdk
$env:EMPY_MSQUIC_RUNTIME = Join-Path $sdk 'bin/msquic.dll'
Write-Output "MsQuic SDK: $sdk"
Write-Output 'MSQUIC_ROOT and EMPY_MSQUIC_RUNTIME are set for this PowerShell session.'
