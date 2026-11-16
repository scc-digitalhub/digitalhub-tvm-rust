#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Refuse to overwrite a registry tag; fail closed on registry lookup errors."""

import subprocess
import sys


def require_unused(reference):
    result = subprocess.run(
        ["docker", "manifest", "inspect", reference], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        raise SystemExit(f"Image tag already exists: {reference}; choose a new release tag")
    error = result.stderr.lower()
    if not any(message in error for message in ("manifest unknown", "no such manifest")):
        raise SystemExit(f"Cannot establish that {reference} is unused: {result.stderr.strip()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: check-image-tag.py IMAGE:TAG [IMAGE:TAG ...]")
    for reference in sys.argv[1:]:
        require_unused(reference)
