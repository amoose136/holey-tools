[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Version,

    [switch] $DevelopmentControls
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Version) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'git is required to determine the release version'
    }

    $Tags = & git -C $RepoRoot for-each-ref `
        --merged=HEAD `
        --sort=-creatordate `
        '--format=%(refname:strip=2)' `
        refs/tags
    if ($LASTEXITCODE -ne 0) {
        throw 'could not read Git tags'
    }

    $Version = $Tags |
        Where-Object { $_ -match '^\d+\.\d+\.\d+$' } |
        Select-Object -First 1
    if (-not $Version) {
        throw 'no reachable Git tag matching #.#.# was found'
    }
}

$VersionMatch = [regex]::Match($Version, '^(\d+)\.(\d+)\.(\d+)$')
if (-not $VersionMatch.Success) {
    throw "version '$Version' must have the form #.#.#"
}
$VersionTuple = '{0}, {1}, {2}' -f (
    $VersionMatch.Groups[1].Value,
    $VersionMatch.Groups[2].Value,
    $VersionMatch.Groups[3].Value
)

if ($env:BLENDER) {
    $Blender = $env:BLENDER
} else {
    $BlenderCommand = Get-Command blender -ErrorAction SilentlyContinue
    if ($BlenderCommand) {
        $Blender = $BlenderCommand.Source
    } else {
        $InstallRoot = Join-Path $env:ProgramFiles 'Blender Foundation'
        $Blender = Get-ChildItem `
            -Path $InstallRoot `
            -Filter blender.exe `
            -File `
            -Recurse `
            -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -ExpandProperty FullName -First 1
    }
}
if (-not $Blender) {
    throw 'Blender was not found; put it on PATH or set BLENDER to its executable'
}

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'holey-tools-build-{0}' -f [guid]::NewGuid()
)

function Set-VersionLine {
    param(
        [string] $Path,
        [string] $Pattern,
        [string] $Replacement
    )

    $Contents = [System.IO.File]::ReadAllText($Path)
    $LinePattern = [regex]::new($Pattern)
    if (-not $LinePattern.IsMatch($Contents)) {
        throw "could not set the version in $Path"
    }
    $Contents = $LinePattern.Replace($Contents, $Replacement, 1)
    $Utf8WithoutBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Contents, $Utf8WithoutBom)
}

try {
    New-Item -ItemType Directory -Path $TempRoot | Out-Null
    $SourceDir = Join-Path $TempRoot 'src'
    Copy-Item -Recurse -Path (Join-Path $RepoRoot 'src') -Destination $SourceDir

    Set-VersionLine `
        -Path (Join-Path $SourceDir 'blender_manifest.toml') `
        -Pattern '(?m)^version = "[^"]*"$' `
        -Replacement "version = `"$Version`""
    Set-VersionLine `
        -Path (Join-Path $SourceDir 'constants.py') `
        -Pattern '(?m)^ADDON_VERSION = "[^"]*"$' `
        -Replacement "ADDON_VERSION = `"$Version`""
    $DevelopmentControlsValue = if ($DevelopmentControls) { 'True' } else { 'False' }
    Set-VersionLine `
        -Path (Join-Path $SourceDir 'constants.py') `
        -Pattern '(?m)^DEVELOPMENT_CONTROLS = .*$' `
        -Replacement "DEVELOPMENT_CONTROLS = $DevelopmentControlsValue"
    Set-VersionLine `
        -Path (Join-Path $SourceDir '__init__.py') `
        -Pattern '(?m)^    "version": \(.*\),$' `
        -Replacement "    `"version`": ($VersionTuple),"

    Write-Host "Building Holey Tools $Version (development controls: $DevelopmentControlsValue)..."
    # Keep packaging independent of enabled user add-ons and saved preferences.
    # Startup options precede --command, which consumes subsequent arguments.
    & $Blender --factory-startup --command extension build `
        --source-dir $SourceDir `
        --output-dir $RepoRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Blender extension build failed with exit code $LASTEXITCODE"
    }
} finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
