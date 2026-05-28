#!/usr/bin/env bash
#
# Generates Secrets.swift from .env.
# Both .env and Secrets.swift are gitignored — secrets never get committed.
#
# Run after changing any value in .env:
#   ./scripts/gen-secrets.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
SECRETS_FILE="$REPO_ROOT/ios/Roboflow Starter Project/Roboflow Starter Project/Secrets.swift"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env not found at $ENV_FILE"
  echo "Copy .env.example to .env and fill in your values first:"
  echo "  cp .env.example .env"
  exit 1
fi

# Extract a value for KEY from .env (first match; trims surrounding whitespace)
env_get() {
  grep -E "^$1=" "$ENV_FILE" | head -1 | cut -d= -f2- | xargs
}

API_KEY="$(env_get ROBOFLOW_API_KEY)"
MODEL="$(env_get ROBOFLOW_MODEL)"
VERSION="$(env_get ROBOFLOW_MODEL_VERSION)"

if [[ -z "$API_KEY" || -z "$MODEL" || -z "$VERSION" ]]; then
  echo "Error: ROBOFLOW_API_KEY, ROBOFLOW_MODEL, and ROBOFLOW_MODEL_VERSION must all be set in .env"
  echo "  ROBOFLOW_API_KEY=${API_KEY:-<empty>}"
  echo "  ROBOFLOW_MODEL=${MODEL:-<empty>}"
  echo "  ROBOFLOW_MODEL_VERSION=${VERSION:-<empty>}"
  exit 1
fi

cat > "$SECRETS_FILE" <<EOF
// AUTO-GENERATED from .env by scripts/gen-secrets.sh
// DO NOT EDIT — your changes will be overwritten.
// DO NOT COMMIT — gitignored via **/Secrets.swift (contains your API key).
//
// Regenerate with: ./scripts/gen-secrets.sh

import Foundation

enum Secrets {
    static let apiKey = "$API_KEY"
    static let model = "$MODEL"
    static let modelVersion = $VERSION
}
EOF

echo "✓ Generated Secrets.swift"
echo "    model:  $MODEL  (v$VERSION)"
echo "    apiKey: ${API_KEY:0:4}…  (redacted)"
echo ""
echo "If this is the FIRST time generating Secrets.swift, add it to the Xcode target:"
echo "  In Xcode: right-click the 'Roboflow Starter Project' source group →"
echo "  'Add Files to ...' → select Secrets.swift → ensure the app target is checked."
