$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $root '.env.local'
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Missing environment file: $configPath"
}

Get-Content -LiteralPath $configPath | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

if ($env:PARTICIPANT_MEMORY_API_KEY.Length -lt 32 -or
    $env:PARTICIPANT_MEMORY_API_KEY.StartsWith('replace-')) {
    throw 'Set a random PARTICIPANT_MEMORY_API_KEY of at least 32 characters'
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_EMBEDDING_API_KEY) -or
    $env:OPENAI_EMBEDDING_API_KEY.StartsWith('replace-')) {
    throw 'Set OPENAI_EMBEDDING_API_KEY before starting the service'
}

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment: $python"
}

Set-Location -LiteralPath $root
& $python -m uvicorn participant_text_memory.app:create_app --factory --host 127.0.0.1 --port 8094
