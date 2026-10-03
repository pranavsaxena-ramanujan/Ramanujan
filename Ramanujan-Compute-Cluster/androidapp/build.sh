#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v gradle >/dev/null 2>&1; then
  exec gradle "$@"
fi
# Reuse the checked-in Gradle 8.5 wrapper without changing the legacy app.
exec ../../androidapp/gradlew -p "$PWD" "$@"
