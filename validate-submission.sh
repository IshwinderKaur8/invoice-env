#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper so users can run: bash validate-submission.sh <space-url>
bash ./scripts/validate-submission.sh "$@"
