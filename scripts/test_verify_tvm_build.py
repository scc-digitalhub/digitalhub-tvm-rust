#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Small provenance contract tests; no native library or GPU is required."""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("verify", Path(__file__).with_name("verify-tvm-build.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.build = Path(self.directory.name)
        (self.build / "lib").mkdir()
        (self.build / "config.cmake").write_text("set(USE_CUDA OFF)\n")
        (self.build / "CMakeCache.txt").write_text("USE_CUDA:STRING=OFF\n")
        self.provenance = {
            "tvm_git_commit": verify.CUDA_REVISION,
            "tvm_ffi_git_commit": "12dbf053b3d9ba4ebd9da3123b1aeca79cf74229",
            "libraries": {},
        }
        for library in verify.REQUIRED_LIBRARIES:
            path = self.build / "lib" / library
            path.write_text(library)
            self.provenance["libraries"][library] = verify.digest(path)

    def enable_cuda(self):
        library = self.build / "lib/libtvm_runtime_cuda.so"
        library.write_text("CUDA plugin")
        cache = self.build / "cuda-CMakeCache.txt"
        cache.write_text("USE_CUDA:UNINITIALIZED=/usr/local/cuda\n")
        self.provenance["libraries"][library.name] = verify.digest(library)
        self.provenance["compiled_backends"] = ["cpu", "cuda"]
        self.provenance["cuda_build"] = {
            "tvm_git_commit": self.provenance["tvm_git_commit"],
            "tvm_ffi_git_commit": self.provenance["tvm_ffi_git_commit"],
            "cuda_version": "13.2.1",
            "build_target": "tvm_runtime_cuda",
            "library_sha256": verify.digest(library),
            "cmake_cache_sha256": verify.digest(cache),
        }

    def test_legacy_cpu_and_cuda_export_keep_exact_hashes(self):
        verify.verify_libraries(self.provenance, self.build, "cpu")
        with self.assertRaisesRegex(SystemExit, "attested CUDA backend"):
            verify.verify_libraries(self.provenance, self.build, "cuda")
        self.enable_cuda()
        verify.verify_libraries(self.provenance, self.build, "cuda")
        verify.verify_libraries(self.provenance, self.build, "cpu")
        (self.build / "lib/libtvm_runtime_cuda.so").write_text("tampered")
        with self.assertRaisesRegex(SystemExit, "libraries do not match"):
            verify.verify_libraries(self.provenance, self.build, "cuda")

    def test_cuda_attestation_rejects_missing_or_mismatched_evidence(self):
        self.enable_cuda()
        for key in ("tvm_git_commit", "tvm_ffi_git_commit", "cuda_version", "build_target",
                    "library_sha256", "cmake_cache_sha256"):
            provenance = copy.deepcopy(self.provenance)
            provenance["cuda_build"][key] = "different"
            with self.subTest(key=key), self.assertRaises(SystemExit):
                verify.verify_libraries(provenance, self.build, "cuda")
        (self.build / "cuda-CMakeCache.txt").write_text("USE_CUDA:STRING=OFF\n")
        self.provenance["cuda_build"]["cmake_cache_sha256"] = verify.digest(self.build / "cuda-CMakeCache.txt")
        with self.assertRaisesRegex(SystemExit, "does not match USE_CUDA"):
            verify.verify_libraries(self.provenance, self.build, "cuda")

    def test_library_sets_and_compiled_backends_are_strict(self):
        for backends in (["cuda"], ["cpu", "cuda", "cuda"], ["cpu", "vulkan"]):
            self.provenance["compiled_backends"] = backends
            with self.assertRaisesRegex(SystemExit, "invalid TVM compiled_backends"):
                verify.verify_libraries(self.provenance, self.build, "cpu")
        self.provenance["compiled_backends"] = ["cpu"]
        del self.provenance["libraries"]["libtvm_ffi.so"]
        with self.assertRaisesRegex(SystemExit, "complete library set"):
            verify.verify_libraries(self.provenance, self.build, "cpu")


if __name__ == "__main__":
    unittest.main()
