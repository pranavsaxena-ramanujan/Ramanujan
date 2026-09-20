#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAMANUJAN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$RAMANUJAN_ROOT/.." && pwd)"
CONVERTER_ROOT="$RAMANUJAN_ROOT/sharded-llm/converter"

MODEL_DIR="$WORKSPACE_ROOT/Phi-3-mini-4k-instruct"
OUTPUT_DIR="$WORKSPACE_ROOT/phi3_rj_ir_shards"
REFERENCE_KERNEL="$RAMANUJAN_ROOT/ramanujan-test-codes/phi3/phi3_transformer_stack_4bit.py"
VERIFY_ONLY="false"

usage() {
    cat <<'EOF'
Usage: convert_phi3_to_rj_shards.sh [options]

Options:
  --model-dir PATH       Phi-3 safetensors directory
  --output-dir PATH      Destination package directory
  --reference-kernel P   Ramanujan Phi-3 kernel used to generate shard programs
  --verify-only          Verify an existing output package without converting
  -h, --help             Show this help

The converter refuses to overwrite OUTPUT_DIR. Remove or move an old package
explicitly before regenerating it. Partial output is deleted automatically.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --reference-kernel) REFERENCE_KERNEL="$2"; shift 2 ;;
        --verify-only) VERIFY_ONLY="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$CONVERTER_ROOT"
if [[ "$VERIFY_ONLY" != "true" ]]; then
    PYTHONPATH=. python3 -m ramanujan_shards.emit_phi3 \
        --model-dir "$MODEL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --reference-kernel "$REFERENCE_KERNEL"
fi

PYTHONPATH=. python3 -m ramanujan_shards.verify_phi3 \
    --package-dir "$OUTPUT_DIR"