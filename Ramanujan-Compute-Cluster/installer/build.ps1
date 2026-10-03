$ErrorActionPreference = "Stop"
# Package the current client sources, not a stale jar left in client/target.
mvn -q -f (Join-Path $PSScriptRoot "..\client\pom.xml") package
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python (Join-Path $PSScriptRoot "package.py") @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
