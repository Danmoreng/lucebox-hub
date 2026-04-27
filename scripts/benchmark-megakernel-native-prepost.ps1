[CmdletBinding()]
param (
    [string]$BaseRef = "origin/main",
    [string]$HeadRef = "HEAD",
    [string]$ModelDir = "C:\Development\vibe-inference\models\qwen3.5-0.8b",
    [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
    [string]$CudaArch = "sm_120",
    [int]$PromptTokens = 520,
    [int]$MaxSeqLen = 2048,
    [int]$Warmup = 5,
    [int]$Runs = 20,
    [string]$OutDir = "benchmarks\prefill-mlp-chunking-native",
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentUICulture = [System.Globalization.CultureInfo]::InvariantCulture

function Resolve-RepoPath([string]$root, [string]$path) {
    if ([IO.Path]::IsPathRooted($path)) { return $path }
    return Join-Path $root $path
}

function Invoke-Checked([string]$label, [scriptblock]$body) {
    Write-Host $label -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE." }
}

function Parse-PpLine([string[]]$lines) {
    $match = $null
    foreach ($line in $lines) {
        if ($line -match "^pp(\d+):\s+([0-9.]+)\s+tok/s\s+\(([0-9.]+)ms\)") {
            $match = [pscustomobject]@{
                prompt_tokens = [int]$Matches[1]
                tok_s = [double]$Matches[2]
                ms = [double]$Matches[3]
            }
        }
    }
    return $match
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutDir = Resolve-RepoPath $RepoRoot $OutDir
$WorktreeRoot = Join-Path $ResolvedOutDir "worktrees"
$BaseTree = Join-Path $WorktreeRoot "base"
$HeadTree = Join-Path $WorktreeRoot "head"
New-Item -Path $ResolvedOutDir -ItemType Directory -Force | Out-Null
New-Item -Path $WorktreeRoot -ItemType Directory -Force | Out-Null

$baseSha = (& git -C $RepoRoot rev-parse $BaseRef).Trim()
$headSha = (& git -C $RepoRoot rev-parse $HeadRef).Trim()
if ([string]::IsNullOrWhiteSpace($baseSha) -or [string]::IsNullOrWhiteSpace($headSha)) {
    throw "Could not resolve BaseRef/HeadRef."
}

if ($Clean) {
    foreach ($tree in @($BaseTree, $HeadTree)) {
        if (Test-Path $tree) {
            $resolved = (Resolve-Path $tree).Path
            $allowed = (Resolve-Path $WorktreeRoot).Path
            if (-not $resolved.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove path outside worktree root: $resolved"
            }
            git -C $RepoRoot worktree remove --force $resolved 2>$null
            if (Test-Path $resolved) {
                Remove-Item -LiteralPath $resolved -Recurse -Force
            }
        }
    }
    git -C $RepoRoot worktree prune
}

foreach ($entry in @(@("base", $BaseTree, $baseSha), @("head", $HeadTree, $headSha))) {
    $name = $entry[0]
    $tree = $entry[1]
    $sha = $entry[2]
    if (-not (Test-Path $tree)) {
        Invoke-Checked "Creating $name worktree at $sha" {
            git -C $RepoRoot worktree add --detach $tree $sha
        }
    }

    $buildScript = Join-Path $tree "scripts\build-megakernel-native.ps1"
    if (-not (Test-Path $buildScript)) {
        $buildScript = Join-Path $RepoRoot "scripts\build-megakernel-native.ps1"
    }

    $buildArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $buildScript,
        "-CudaRoot", $CudaRoot,
        "-CudaArch", $CudaArch,
        "-RepoRootOverride", $tree
    )
    if ($Clean) { $buildArgs += "-Clean" }
    Invoke-Checked "Building $name native benchmark" {
        pwsh @buildArgs
    }

    $exe = Join-Path $tree "megakernel\build-native\bench_prefill_native.exe"
    if (-not (Test-Path $exe)) { throw "Native benchmark exe not found: $exe" }

    $logPath = Join-Path $ResolvedOutDir "$name.log"
    Write-Host "Running $name native benchmark" -ForegroundColor Cyan
    $output = & $exe --hf-model-dir $ModelDir --prompt-tokens $PromptTokens --max-seq-len $MaxSeqLen --warmup $Warmup --runs $Runs 2>&1
    $output | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) { throw "$name native benchmark failed with exit code $LASTEXITCODE." }
}

$base = Parse-PpLine (Get-Content (Join-Path $ResolvedOutDir "base.log"))
$head = Parse-PpLine (Get-Content (Join-Path $ResolvedOutDir "head.log"))
if (-not $base -or -not $head) { throw "Could not parse native benchmark logs." }

$speedup = $head.tok_s / $base.tok_s
$deltaPct = ($speedup - 1.0) * 100.0
$summary = @(
    [pscustomobject]@{ ref = "base"; sha = $baseSha; prompt_tokens = $base.prompt_tokens; tok_s = $base.tok_s; ms = $base.ms; speedup_vs_base = 1.0; delta_pct = 0.0 },
    [pscustomobject]@{ ref = "head"; sha = $headSha; prompt_tokens = $head.prompt_tokens; tok_s = $head.tok_s; ms = $head.ms; speedup_vs_base = $speedup; delta_pct = $deltaPct }
)
$csvPath = Join-Path $ResolvedOutDir "summary.csv"
$summary | Export-Csv -NoTypeInformation -Path $csvPath
Write-Host ("Native prefill benchmark complete. head/base = {0:N3}x ({1:N2}%)." -f $speedup, $deltaPct) -ForegroundColor Green
Write-Host "Summary: $csvPath" -ForegroundColor Green
