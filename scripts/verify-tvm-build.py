#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REQUIRED_LIBRARIES = {
    "libtvm_runtime.so",
    "libtvm_runtime_extra.so",
    "libtvm_ffi.so",
    "libtvm_compiler.so",
}


def git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    build = args.build.resolve()
    release_ref = git(source, "describe", "--tags", "--exact-match")
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)", release_ref)
    require(match is not None, f"TVM checkout is not an exact stable release: {release_ref}")
    assert match is not None
    require(
        not git(source, "status", "--porcelain", "--untracked-files=no"),
        f"TVM checkout contains tracked changes: {source}",
    )
    submodules = git(source, "submodule", "status", "--recursive").splitlines()
    invalid_submodules = [line for line in submodules if line[:1] in {"-", "+", "U"}]
    require(
        not invalid_submodules,
        "TVM submodules are not aligned:\n" + "\n".join(invalid_submodules),
    )

    version = match.group(1)
    tvm_commit = git(source, "rev-parse", "HEAD")
    ffi_source = source / "3rdparty/tvm-ffi"
    ffi_commit = git(ffi_source, "rev-parse", "HEAD")
    provenance_path = build / "digitalhub-build-provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"invalid or missing {provenance_path}; attest the build with "
            "digitalhub-tvm-toolkit/build-tvm.sh"
        ) from error

    expected = {
        "schema_version": 1,
        "tvm_version": version,
        "tvm_git_commit": tvm_commit,
        "tvm_ffi_git_commit": ffi_commit,
        "source_dir": str(source),
    }
    mismatches = [
        f"{key}: {provenance.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if provenance.get(key) != value
    ]
    require(not mismatches, "TVM provenance mismatch: " + "; ".join(mismatches))

    cache_source = next(
        (
            Path(line.partition("=")[2]).resolve()
            for line in (build / "CMakeCache.txt").read_text().splitlines()
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")
        ),
        None,
    )
    require(
        cache_source == source,
        f"TVM build {build} was configured from {cache_source}, not {source}",
    )

    libraries = provenance.get("libraries", {})
    require(
        set(libraries) == REQUIRED_LIBRARIES,
        "TVM provenance does not contain the complete library set",
    )
    changed_libraries = [
        str(build / "lib" / name)
        for name, expected_hash in libraries.items()
        if not (build / "lib" / name).is_file()
        or digest(build / "lib" / name) != expected_hash
    ]
    require(
        not changed_libraries,
        "TVM libraries do not match provenance: " + ", ".join(changed_libraries),
    )

    tag = ".".join(version.split(".")[:2])
    print(
        "\t".join(
            (
                version,
                tag,
                tvm_commit,
                ffi_commit,
                provenance["tvm_ffi_version"],
            )
        )
    )


if __name__ == "__main__":
    main()