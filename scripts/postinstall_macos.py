#!/usr/bin/env python
"""Make LightGBM importable on Apple Silicon without Homebrew.

LightGBM's macOS wheel links against ``@rpath/libomp.dylib`` but bakes in only
Homebrew and MacPorts rpaths, so on a machine without either it fails at
import. scikit-learn's wheel already vendors a compatible ``libomp``, so the
fix is to copy that one next to LightGBM's dylib, add ``@loader_path`` as an
rpath, and re-sign - arm64 rejects a modified dylib whose signature no longer
matches.

Idempotent. Safe to re-run.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin":
        print("not macOS - nothing to do")
        return 0

    site = Path(sysconfig.get_paths()["purelib"])
    src = site / "sklearn" / ".dylibs" / "libomp.dylib"
    lgb_dir = site / "lightgbm" / "lib"
    dylib = lgb_dir / "lib_lightgbm.dylib"

    if not dylib.exists():
        print(f"lightgbm dylib not found at {dylib}")
        return 1
    if not src.exists():
        print(f"no vendored libomp at {src} - install scikit-learn first")
        return 1

    shutil.copy2(src, lgb_dir / "libomp.dylib")
    print(f"copied libomp -> {lgb_dir / 'libomp.dylib'}")

    rpaths = subprocess.run(["otool", "-l", str(dylib)], capture_output=True, text=True).stdout
    if "@loader_path" not in rpaths:
        subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", str(dylib)], check=True)
        print("added @loader_path rpath")
    else:
        print("@loader_path rpath already present")

    subprocess.run(["codesign", "--force", "--sign", "-", str(dylib)],
                   check=True, capture_output=True)
    print("re-signed dylib")

    subprocess.run([sys.executable, "-c", "import lightgbm; print('lightgbm', lightgbm.__version__, 'imports OK')"],
                   check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
