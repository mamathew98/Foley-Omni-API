#!/usr/bin/env bash
# Alias entrypoint — same as ./deploy.sh (requested .bash extension).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy.sh" "$@"
