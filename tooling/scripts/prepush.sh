#!/usr/bin/env bash
# Run the full local gate before push (CI + harness + integration).
# Usage: pnpm prepush   OR   ./tooling/scripts/prepush.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "==> lint"
pnpm lint

echo "==> typecheck"
pnpm typecheck

echo "==> test"
pnpm test

echo "==> test:integration"
pnpm test:integration

echo "==> build"
pnpm build

echo "==> cargo check"
pnpm cargo:check

echo "==> cargo test"
pnpm cargo:test

echo "==> check:harness"
pnpm check:harness

echo "All prepush checks passed."
