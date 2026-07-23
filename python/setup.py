from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution


PACKAGE_ROOT = Path(__file__).parent / "src" / "loom_kernels"
NATIVE_PAYLOAD = (
    PACKAGE_ROOT / "lib" / "libloom_cuda_bridge.so",
    PACKAGE_ROOT / "lib" / "libloom_kernels_torch.so",
    PACKAGE_ROOT / "lib" / "native.json",
)


class LoomBinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class LoomNativeWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        build_number = os.environ.get("LOOM_KERNELS_WHEEL_BUILD")
        if build_number:
            self.build_number = build_number

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag

    def run(self) -> None:
        missing = [str(path) for path in NATIVE_PAYLOAD if not path.is_file()]
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise RuntimeError(
                "Loom native wheel requires both compiled libraries and its "
                f"manifest; use python/build_wheel.py. Missing:\n{formatted}"
            )
        super().run()


setup(
    cmdclass={"bdist_wheel": LoomNativeWheel},
    distclass=LoomBinaryDistribution,
)
