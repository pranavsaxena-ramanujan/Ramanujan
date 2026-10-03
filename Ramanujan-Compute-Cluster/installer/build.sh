#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
# Package the current client sources, not a stale jar left in client/target.
mvn -q -f "$script_dir/../client/pom.xml" package
exec python3 "$script_dir/package.py" "$@"
