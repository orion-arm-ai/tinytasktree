from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class build_py(_build_py):
    def run(self) -> None:
        self._reset_build_lib()
        super().run()
        self._copy_ui_dist()

    def _reset_build_lib(self) -> None:
        build_root = Path(self.build_lib)
        if build_root.exists():
            shutil.rmtree(build_root)

    def _copy_ui_dist(self) -> None:
        root = Path(__file__).resolve().parent
        src = root / "ui" / "dist"
        if not src.is_dir():
            return

        dst = Path(self.build_lib) / "tinytasktree" / "ui_dist"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _build_frontend() -> None:
    root = Path(__file__).resolve().parent
    ui_dir = root / "ui"
    ui_src = ui_dir / "src"
    ui_dist = ui_dir / "dist"
    if not ui_src.is_dir():
        return

    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to bundle the UI into Python packages, but it was not found")

    if not (ui_dir / "node_modules").is_dir():
        install_cmd = [npm, "ci"] if (ui_dir / "package-lock.json").is_file() else [npm, "install"]
        subprocess.run(install_cmd, cwd=ui_dir, check=True)

    subprocess.run([npm, "run", "build"], cwd=ui_dir, check=True)
    if not ui_dist.joinpath("index.html").is_file():
        raise RuntimeError("UI build completed, but ui/dist/index.html was not produced")


class sdist(_sdist):
    def run(self) -> None:
        _build_frontend()
        super().run()


class bdist_wheel(_bdist_wheel):
    def run(self) -> None:
        _build_frontend()
        super().run()


setup(cmdclass={"build_py": build_py, "sdist": sdist, "bdist_wheel": bdist_wheel})
