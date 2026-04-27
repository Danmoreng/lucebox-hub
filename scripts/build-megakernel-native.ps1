[CmdletBinding()]
param (
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    [string]$CudaArch = "sm_120",
    [string]$BuildDir = "build-native",
    [string]$RepoRootOverride = "",
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Import-VSEnv {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }
    $vswhere = Join-Path ${Env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found." }
    $vsroot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if ([string]::IsNullOrWhiteSpace($vsroot)) { throw "Visual Studio C++ toolchain not found." }
    $vcvars = Join-Path $vsroot "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
    Write-Host "Importing Visual Studio environment from: $vcvars" -ForegroundColor Cyan
    $envDump = cmd /s /c "`"$vcvars`" > nul && set"
    foreach ($line in $envDump) {
        if ($line -match "^(.*?)=(.*)$") { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
    }
}

$RepoRoot = if ([string]::IsNullOrWhiteSpace($RepoRootOverride)) {
    Split-Path -Parent $PSScriptRoot
} else {
    $RepoRootOverride
}
$MegaDir = Join-Path $RepoRoot "megakernel"
$NativeDir = Join-Path $MegaDir "native"
if (-not (Test-Path $NativeDir)) {
    $NativeDir = Join-Path (Split-Path -Parent $PSScriptRoot) "megakernel\native"
}
$ResolvedBuildDir = if ([IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $MegaDir $BuildDir }

if (-not (Test-Path $CudaRoot)) { throw "CUDA root not found: $CudaRoot" }
Import-VSEnv
$env:Path = (Join-Path $CudaRoot "bin") + [IO.Path]::PathSeparator + $env:Path

if ($Clean -and (Test-Path $ResolvedBuildDir)) {
    Remove-Item -LiteralPath $ResolvedBuildDir -Recurse -Force
}
New-Item -Path $ResolvedBuildDir -ItemType Directory -Force | Out-Null

$objPrefill = Join-Path $ResolvedBuildDir "prefill.obj"
$objKernel = Join-Path $ResolvedBuildDir "kernel.obj"
$objBench = Join-Path $ResolvedBuildDir "bench_prefill_native.obj"
$objSafe = Join-Path $ResolvedBuildDir "safetensors.obj"
$exe = Join-Path $ResolvedBuildDir "bench_prefill_native.exe"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$prefillSource = Join-Path $MegaDir "prefill.cu"
$prefillText = Get-Content $prefillSource -Raw
$prefillVersionDefine = if ($prefillText.Contains("dn_pre_qkv")) { "-DMEGAKERNEL_PREFILL_V2=1" } else { "-DMEGAKERNEL_PREFILL_V2=0" }

& $nvcc -std=c++17 "-arch=$CudaArch" --use_fast_math -O3 -c $prefillSource -o $objPrefill
if ($LASTEXITCODE -ne 0) { throw "prefill.cu compile failed." }
& $nvcc -std=c++17 "-arch=$CudaArch" --use_fast_math -O3 $prefillVersionDefine -c (Join-Path $NativeDir "bench_prefill_native.cu") -o $objBench
if ($LASTEXITCODE -ne 0) { throw "bench_prefill_native.cu compile failed." }
& cl.exe /nologo /O2 /std:c++17 /EHsc /I$NativeDir /c (Join-Path $NativeDir "safetensors.cpp") /Fo$objSafe
if ($LASTEXITCODE -ne 0) { throw "safetensors.cpp compile failed." }
& $nvcc "-arch=$CudaArch" $objPrefill $objBench $objSafe -lcublas -o $exe
if ($LASTEXITCODE -ne 0) { throw "native benchmark link failed." }

Write-Host "Native benchmark built: $exe" -ForegroundColor Green
