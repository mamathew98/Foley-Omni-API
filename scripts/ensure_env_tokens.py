#!/usr/bin/env python3
"""Ensure .env exists and ACCESS_TOKEN / ADMIN_TOKEN are set (not placeholders).

Prints three lines to stdout:
  1) GENERATED | OK
  2) ACCESS_TOKEN value
  3) ADMIN_TOKEN value
"""

from __future__ import annotations

import re
import secrets
import sys
from pathlib import Path


PLACEHOLDERS = {
    "ACCESS_TOKEN": "change-me-guest-preview-token",
    "ADMIN_TOKEN": "change-me-admin-token-different",
}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    example = root / ".env.example"

    if not env_path.is_file():
        if not example.is_file():
            print(".env.example is missing", file=sys.stderr)
            return 1
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    text = env_path.read_text(encoding="utf-8")
    changed = False
    values: dict[str, str] = {}

    for key, placeholder in PLACEHOLDERS.items():
        match = re.search(rf"^{key}=(.*)$", text, flags=re.M)
        current = match.group(1).strip() if match else ""
        if (not current) or current == placeholder:
            token = secrets.token_urlsafe(24)
            if match:
                text = re.sub(rf"^{key}=.*$", f"{key}={token}", text, count=1, flags=re.M)
            else:
                text = text.rstrip() + f"\n{key}={token}\n"
            values[key] = token
            changed = True
        else:
            values[key] = current

    if values["ACCESS_TOKEN"] == values["ADMIN_TOKEN"]:
        print("ACCESS_TOKEN and ADMIN_TOKEN must be different", file=sys.stderr)
        return 1

    if changed:
        env_path.write_text(text, encoding="utf-8")
        print("GENERATED")
    else:
        print("OK")
    print(values["ACCESS_TOKEN"])
    print(values["ADMIN_TOKEN"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
