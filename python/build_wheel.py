#!/usr/bin/env python3
"""Build and audit Loom's Python-ABI-independent Linux native wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any
import zipfile

import torch


TORCH_TARGET_VERSION = "2.10"
TORCH_MAX_VERSION = "2.12"
BRIDGE_ABI_VERSION = 1
LIBRARIES = (
    "libloom_cuda_bridge.so",
    "libloom_kernels_torch.so",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(
            os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
        ),
    )
    parser.add_argument(
        "--archs",
        default=os.environ.get("LOOM_CUDA_ARCHS"),
        help="comma-separated CUDA SM numbers; required for a matrix artifact",
    )
    parser.add_argument("--wheel-dir", type=Path)
    return parser.parse_args()


def run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(python_root: Path) -> str:
    pyproject = (python_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    if match is None:
        raise RuntimeError("python/pyproject.toml has no static project version")
    return match.group(1)


def cuda_release(cuda_home: Path) -> str:
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(f"nvcc was not found below {cuda_home}")
    result = subprocess.run(
        [str(nvcc), "--version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
    if match is None:
        raise RuntimeError("could not parse the CUDA release from nvcc --version")
    return match.group(1)


def git_revision(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if status:
        raise RuntimeError(
            "native wheels must be built from a clean Git revision; "
            f"found:\n{status}"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout.strip()


def validate_archs(raw_archs: str | None) -> list[int]:
    if raw_archs is None:
        raise ValueError(
            "--archs or LOOM_CUDA_ARCHS is required; matrix wheels must declare "
            "their compiled SM targets"
        )
    values = [value.strip() for value in raw_archs.split(",") if value.strip()]
    if not values or any(not value.isdigit() for value in values):
        raise ValueError(f"invalid CUDA architecture list: {raw_archs!r}")
    archs = [int(value) for value in values]
    if len(set(archs)) != len(archs):
        raise ValueError(f"duplicate CUDA architecture in {raw_archs!r}")
    return archs


def platform_tag() -> str:
    if sys.platform != "linux":
        raise RuntimeError("Loom native wheels currently support Linux only")
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            f"Loom's first native-wheel matrix supports x86_64, not {machine}"
        )
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def wheel_build_tag(cuda: str, archs: list[int]) -> str:
    cuda_digits = cuda.replace(".", "")
    arch_suffix = "".join(f"sm{arch}" for arch in archs)
    return f"1cu{cuda_digits}torch210{arch_suffix}"


def needed_libraries(readelf_output: str) -> list[str]:
    return re.findall(r"Shared library: \[([^\]]+)\]", readelf_output)


def audit_native_libraries(repository: Path, native_dir: Path) -> dict[str, Any]:
    for tool in ("readelf", "nm", "c++filt"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"{tool} is required to audit the native wheel")

    bridge = native_dir / LIBRARIES[0]
    dispatcher = native_dir / LIBRARIES[1]
    for library in (bridge, dispatcher):
        if not library.is_file():
            raise RuntimeError(f"{library} is not a Linux ELF shared library")
        with library.open("rb") as source:
            elf_magic = source.read(4)
        if elf_magic != b"\x7fELF":
            raise RuntimeError(f"{library} is not a Linux ELF shared library")

    dispatcher_dynamic = run(
        ["readelf", "-d", str(dispatcher)],
        cwd=repository,
    )
    if "libloom_cuda_bridge.so" not in needed_libraries(dispatcher_dynamic):
        raise RuntimeError("dispatcher does not depend on libloom_cuda_bridge.so")
    if "$ORIGIN" not in dispatcher_dynamic:
        raise RuntimeError("dispatcher has no relative $ORIGIN runtime search path")

    undefined_symbols = run(
        ["nm", "-D", "--undefined-only", str(dispatcher)],
        cwd=repository,
    )
    if "loom_cuda_launch_" in undefined_symbols:
        raise RuntimeError("dispatcher bypasses Rust through a raw CUDA launch symbol")
    demangled = subprocess.run(
        ["c++filt"],
        input=undefined_symbols,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if re.search(r"(^|\s)(at|c10)::", demangled):
        raise RuntimeError("dispatcher consumes an unstable ATen/c10 C++ symbol")

    bridge_dynamic = run(
        ["readelf", "-d", str(bridge)],
        cwd=repository,
    )
    return {
        "needed": {
            bridge.name: needed_libraries(bridge_dynamic),
            dispatcher.name: needed_libraries(dispatcher_dynamic),
        },
        "sha256": {
            library.name: sha256(library) for library in (bridge, dispatcher)
        },
        "dispatcher_raw_cuda_launch_symbols": 0,
        "dispatcher_aten_c10_cpp_symbols": 0,
    }


def make_manifest(
    *,
    repository: Path,
    python_root: Path,
    cuda: str,
    archs: list[int],
    build_tag: str,
    native_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "distribution": {
            "name": "loom-kernels",
            "version": package_version(python_root),
            "wheel_build_tag": build_tag,
            "python_tag": "py3",
            "abi_tag": "none",
            "platform_tag": platform_tag(),
        },
        "build": {
            "revision": git_revision(repository),
            "cuda_toolkit": cuda,
            "cuda_architectures": [f"sm_{arch}" for arch in archs],
            "torch": torch.__version__,
            "torch_stable_abi_target": TORCH_TARGET_VERSION,
            "bridge_abi_version": BRIDGE_ABI_VERSION,
        },
        "runtime": {
            "python": ">=3.10",
            "torch": {
                "minimum": TORCH_TARGET_VERSION,
                "maximum_exclusive": TORCH_MAX_VERSION,
            },
            "operating_system": "Linux",
            "architecture": "x86_64",
            "cuda_runtime": "external",
        },
        "libraries": native_audit,
    }


def stage_python_project(
    python_root: Path,
    staging_root: Path,
    native_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    staging_python = staging_root / "python"
    shutil.copytree(
        python_root,
        staging_python,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    package_lib = staging_python / "src" / "loom_kernels" / "lib"
    if package_lib.exists():
        shutil.rmtree(package_lib)
    package_lib.mkdir(parents=True)
    for library in LIBRARIES:
        shutil.copy2(native_dir / library, package_lib / library)
    (package_lib / "native.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return staging_python


def inspect_wheel(
    wheel: Path,
    *,
    expected_platform: str,
    expected_build_tag: str,
    expected_manifest: dict[str, Any],
) -> None:
    if (
        f"-{expected_build_tag}-py3-none-{expected_platform}.whl"
        not in wheel.name
    ):
        raise RuntimeError(f"unexpected native wheel tag: {wheel.name}")

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        shared_libraries = sorted(name for name in names if name.endswith(".so"))
        expected_libraries = sorted(
            f"loom_kernels/lib/{library}" for library in LIBRARIES
        )
        if shared_libraries != expected_libraries:
            raise RuntimeError(
                f"wheel must contain exactly the two Loom .so files: "
                f"{shared_libraries}"
            )

        wheel_metadata_paths = [
            name for name in names if name.endswith(".dist-info/WHEEL")
        ]
        if len(wheel_metadata_paths) != 1:
            raise RuntimeError("wheel has an invalid .dist-info/WHEEL layout")
        wheel_metadata = archive.read(wheel_metadata_paths[0]).decode()
        if "Root-Is-Purelib: false" not in wheel_metadata:
            raise RuntimeError("native payload was incorrectly marked as pure Python")
        expected_tag = f"Tag: py3-none-{expected_platform}"
        if expected_tag not in wheel_metadata:
            raise RuntimeError(f"wheel metadata is missing {expected_tag!r}")

        manifest_path = "loom_kernels/lib/native.json"
        if names.count(manifest_path) != 1:
            raise RuntimeError("wheel has no native build manifest at package root")
        actual_manifest = json.loads(archive.read(manifest_path))
        if actual_manifest != expected_manifest:
            raise RuntimeError("wheel native manifest does not match the build")


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    python_root = repository / "python"
    cuda_home = args.cuda_home.resolve()
    archs = validate_archs(args.archs)
    cuda = cuda_release(cuda_home)
    build_tag = wheel_build_tag(cuda, archs)
    wheel_dir = (args.wheel_dir or repository / "dist").resolve()
    wheel_dir.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec("build") is None:
        raise RuntimeError(
            "the Python 'build' package is required; install build, setuptools, "
            "and wheel in the build environment"
        )

    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(cuda_home)
    environment["LOOM_CUDA_ARCHS"] = ",".join(str(arch) for arch in archs)
    environment["LOOM_KERNELS_WHEEL_BUILD"] = build_tag
    build_parent = repository / "build"
    build_parent.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="loom-native-wheel-",
        dir=build_parent,
    ) as temporary:
        temporary_root = Path(temporary)
        native_dir = temporary_root / "native"
        native_dir.mkdir()
        run(
            [
                sys.executable,
                str(python_root / "build_native.py"),
                "--cuda-home",
                str(cuda_home),
                "--archs",
                ",".join(str(arch) for arch in archs),
                "--output-dir",
                str(native_dir),
            ],
            cwd=repository,
            environment=environment,
        )
        run(
            [
                sys.executable,
                str(python_root / "build_torch_extension.py"),
                "--cuda-home",
                str(cuda_home),
                "--build-dir",
                str(native_dir),
            ],
            cwd=repository,
            environment=environment,
        )
        native_audit = audit_native_libraries(repository, native_dir)
        manifest = make_manifest(
            repository=repository,
            python_root=python_root,
            cuda=cuda,
            archs=archs,
            build_tag=build_tag,
            native_audit=native_audit,
        )
        staging_python = stage_python_project(
            python_root,
            temporary_root / "staging",
            native_dir,
            manifest,
        )

        version = package_version(python_root)
        expected_pattern = f"loom_kernels-{version}-{build_tag}-*.whl"
        for stale_wheel in wheel_dir.glob(expected_pattern):
            stale_wheel.unlink()
        run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(wheel_dir),
                str(staging_python),
            ],
            cwd=repository,
            environment=environment,
        )
        wheels = list(wheel_dir.glob(expected_pattern))
        if len(wheels) != 1:
            raise RuntimeError(
                f"expected one native wheel matching {expected_pattern}, got {wheels}"
            )
        wheel = wheels[0]
        inspect_wheel(
            wheel,
            expected_platform=platform_tag(),
            expected_build_tag=build_tag,
            expected_manifest=manifest,
        )
        print(wheel)


if __name__ == "__main__":
    main()
