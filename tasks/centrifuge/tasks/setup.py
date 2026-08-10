# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metadata lives in pyproject.toml. The auto-register .pth is injected into
the wheel by the custom build backend in ``_build_backend.py`` — this file is
only here as a fallback for tools that still shell out to ``setup.py``.
"""

from setuptools import setup

setup()
