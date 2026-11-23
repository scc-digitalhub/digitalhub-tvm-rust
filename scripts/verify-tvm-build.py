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
CUDA_REVISION = "c7b458e946bc4266915da582457476bdcd9705ae"


def required_libraries(provenance: dict, build: Path, backend: str) -> set[str]:
    backends = provenance.get("compiled_backends", ["cpu"])
    require(backends in (["cpu"], ["cpu", "cuda"]), "invalid TVM compiled_backends")
    cuda = "cuda" in backends
    require(backend != "cuda" or cuda, "CUDA image requires an attested CUDA backend")
    if cuda:
        require(provenance.get("tvm_git_commit") == CUDA_REVISION, "CUDA image requires the pinned TVM 0.26.0 revision")
        proof = provenance.get("cuda_build", {})
        require(isinstance(proof, dict), "invalid CUDA build provenance")
        for key in ("tvm_git_commit", "tvm_ffi_git_commit"):
            require(proof.get(key) == provenance.get(key), f"CUDA provenance {key} mismatch")
        require(proof.get("build_target") == "tvm_runtime_cuda", "CUDA provenance has no runtime build target")
        require(proof.get("cuda_version") == "13.2.1", "CUDA image requires CUDA 13.2.1")
        require(proof.get("library_sha256") == provenance.get("libraries", {}).get("libtvm_runtime_cuda.so"),
                "CUDA sidecar hash differs from TVM library provenance")
        # The toolkit adds a separately built CUDA plugin to its existing CPU
        # libraries. Its original CPU cache therefore stays CPU-only.
        cache = build / "cuda-CMakeCache.txt"
        if not cache.is_file():
            cache = build / "CMakeCache.txt"
        require(cache.is_file() and digest(cache) == proof.get("cmake_cache_sha256"),
                "CUDA CMake cache does not match provenance")
        settings = re.findall(r"(?m)^USE_CUDA:[^=]+=(.*)$", cache.read_text())
        require(bool(settings), "CUDA CMake cache does not record USE_CUDA")
        cuda_setting = settings[-1]
    else:
        # TVM's legacy CPU config uses ordinary variables outside CMakeCache.
        config = build / "config.cmake"
        settings = re.findall(r'(?im)^\s*set\(\s*USE_CUDA\s+([^\s)]+)\s*\)',
                              config.read_text() if config.is_file() else "")
        if not settings:
            settings = re.findall(r"(?m)^USE_CUDA:[^=]+=(.*)$", (build / "CMakeCache.txt").read_text())
        require(bool(settings), "TVM build does not record USE_CUDA")
        cuda_setting = settings[-1].strip('"')
    require("$" not in cuda_setting, "TVM USE_CUDA must be an explicit value")
    configured_cuda = cuda_setting.upper() not in {"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"}
    require(configured_cuda == cuda,
            f"TVM compiled_backends {backends!r} does not match USE_CUDA={cuda_setting}")
    return REQUIRED_LIBRARIES | ({"libtvm_runtime_cuda.so"} if cuda else set())


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


def verify_libraries(provenance: dict, build: Path, backend: str) -> None:
    libraries = provenance.get("libraries", {})
    require(set(libraries) == required_libraries(provenance, build, backend),
            "TVM provenance does not contain the complete library set")
    changed = [str(build / "lib" / name) for name, expected_hash in libraries.items()
               if not (build / "lib" / name).is_file()
               or digest(build / "lib" / name) != expected_hash]
    require(not changed, "TVM libraries do not match provenance: " + ", ".join(changed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--exported-artifacts", action="store_true",
                        help="verify a CUDA toolkit library export against this pinned source checkout")
    args = parser.parse_args()
    require(not args.exported_artifacts or args.backend == "cuda",
            "exported-artifact verification is only available for the CUDA variant")

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
    if args.exported_artifacts and not provenance_path.is_file():
        provenance_path = build / "provenance.json"
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
    }
    if not args.exported_artifacts:
        expected["source_dir"] = str(source)
    mismatches = [
        f"{key}: {provenance.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if provenance.get(key) != value
    ]
    require(not mismatches, "TVM provenance mismatch: " + "; ".join(mismatches))

    cache_source = next(
        (
            Path(line.partition("=")[2]).resolve()
            for line in ((build / "CMakeCache.txt").read_text().splitlines()
                         if not args.exported_artifacts else [])
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")
        ),
        None,
    )
    require(
        args.exported_artifacts or cache_source == source,
        f"TVM build {build} was configured from {cache_source}, not {source}",
    )

    verify_libraries(provenance, build, args.backend)

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
