$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if (Get-Command gradle -ErrorAction SilentlyContinue) { gradle @args }
    else { & ../../androidapp/gradlew.bat -p $PSScriptRoot @args }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally { Pop-Location }
