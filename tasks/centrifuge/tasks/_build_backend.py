# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom PEP 517 build backend: setuptools + inject the auto-register .pth shim.

This wraps ``setuptools.build_meta`` so that both regular wheels
(``pip install .``) and PEP 660 editable wheels (``pip install -e .``) end up
with ``rpl_centrifuge_autoregister.pth`` at the wheel root. pip drops
files at the wheel root directly into the target venv's ``site-packages/``,
which is where Python needs the ``.pth`` to be for auto-import at startup.

The old ``setup.py``-based custom install commands were only invoked for the
legacy install path — PEP 660's ``build_editable`` skips them entirely.
Injecting the file into the wheel is the one place that works everywhere.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

from setuptools import build_meta as _st

# Re-export the PEP 517 hooks that don't need to change.
get_requires_for_build_wheel = _st.get_requires_for_build_wheel
get_requires_for_build_sdist = _st.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _st.prepare_metadata_for_build_wheel
build_sdist = _st.build_sdist

# Optional PEP 660 hooks — re-export if present.
if hasattr(_st, "get_requires_for_build_editable"):
    get_requires_for_build_editable = _st.get_requires_for_build_editable
if hasattr(_st, "prepare_metadata_for_build_editable"):
    prepare_metadata_for_build_editable = _st.prepare_metadata_for_build_editable


_PTH_NAME = "rpl_centrifuge_autoregister.pth"
_PTH_SRC = Path(__file__).parent / "src" / "rpl_centrifuge" / _PTH_NAME


def _urlsafe_b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _inject_pth(wheel_path: Path) -> None:
    """Add the autoregister .pth to the wheel root and update RECORD."""
    if not _PTH_SRC.exists():
        raise RuntimeError(f"missing source .pth: {_PTH_SRC}")
    pth_bytes = _PTH_SRC.read_bytes()
    pth_hash = "sha256=" + _urlsafe_b64(hashlib.sha256(pth_bytes).digest())
    pth_len = str(len(pth_bytes))

    with tempfile.TemporaryDirectory() as td:
        new_path = Path(td) / wheel_path.name
        with zipfile.ZipFile(wheel_path, "r") as src:
            record_name = next(
                (n for n in src.namelist() if n.endswith(".dist-info/RECORD")), None
            )
            if record_name is None:
                raise RuntimeError(f"wheel missing RECORD: {wheel_path}")

            with zipfile.ZipFile(new_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for item in src.infolist():
                    # Drop any pre-existing copies; we rewrite RECORD ourselves.
                    if item.filename == _PTH_NAME or item.filename == record_name:
                        continue
                    dst.writestr(item, src.read(item.filename))

                # Add our .pth at the wheel root.
                dst.writestr(_PTH_NAME, pth_bytes)

                # Rewrite RECORD: strip any stale entry for our .pth, append fresh.
                record_text = src.read(record_name).decode("utf-8")
                lines = [
                    ln
                    for ln in record_text.splitlines()
                    if ln and not ln.startswith(_PTH_NAME + ",")
                ]
                lines.append(f"{_PTH_NAME},{pth_hash},{pth_len}")
                dst.writestr(record_name, "\n".join(lines) + "\n")

        shutil.move(str(new_path), str(wheel_path))


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = _st.build_wheel(wheel_directory, config_settings, metadata_directory)
    _inject_pth(Path(wheel_directory) / name)
    return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    name = _st.build_editable(wheel_directory, config_settings, metadata_directory)
    _inject_pth(Path(wheel_directory) / name)
    return name
