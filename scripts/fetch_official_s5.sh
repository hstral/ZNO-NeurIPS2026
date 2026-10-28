#!/usr/bin/env bash
# Fetch the official S5 implementation at the commit used in the paper.
# The S5 core is used unmodified; see THIRD_PARTY_NOTICES.md (MIT License).
set -euo pipefail

S5_URL="https://github.com/lindermanlab/S5"
S5_COMMIT="3c18fdb6b06414da35e77b94b9cd855f6a95ef17"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ZNO_S5_ROOT:-$REPO_ROOT/third_party/S5}"

if [ ! -d "$TARGET/.git" ]; then
  mkdir -p "$(dirname "$TARGET")"
  git clone "$S5_URL" "$TARGET"
fi
git -C "$TARGET" fetch --quiet origin "$S5_COMMIT" || true
git -C "$TARGET" checkout --quiet "$S5_COMMIT"
ACTUAL="$(git -C "$TARGET" rev-parse HEAD)"
if [ "$ACTUAL" != "$S5_COMMIT" ]; then
  echo "S5 checkout is at $ACTUAL, expected $S5_COMMIT" >&2
  exit 1
fi
echo "Official S5 ready at $TARGET ($S5_COMMIT)"
