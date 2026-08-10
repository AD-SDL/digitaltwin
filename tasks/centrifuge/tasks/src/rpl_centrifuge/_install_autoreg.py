# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installs the auto-import ``.pth`` shim into the active venv's site-packages.

Setuptools' PEP 660 editable installer doesn't honor ``data_files`` for ``.pth``
files, so we install the shim with a one-shot console script instead. Run via:

    rpl-centrifuge-install-autoreg

After this runs, every Python process inside the Isaac Lab venv will execute
``import rpl_centrifuge`` at startup, which triggers ``gym.register``
without any user script edits.
"""

from __future__ import annotations

import shutil
import sysconfig
from importlib.resources import files as _pkg_files
from pathlib import Path

_PTH_BASENAME = "rpl_centrifuge_autoregister.pth"


def main() -> None:
    src = _pkg_files("rpl_centrifuge").joinpath(_PTH_BASENAME)
    purelib = Path(sysconfig.get_paths()["purelib"])
    if not purelib.exists():
        raise RuntimeError(f"purelib does not exist: {purelib}")
    dst = purelib / _PTH_BASENAME
    shutil.copyfile(str(src), str(dst))
    print(f"[rpl_centrifuge] auto-register hook installed: {dst}")


if __name__ == "__main__":
    main()
