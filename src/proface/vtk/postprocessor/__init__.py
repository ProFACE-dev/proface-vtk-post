# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

from importlib.metadata import PackageNotFoundError, version

try:
    # FIXME: check this is the same as pyproject.toml -> project.version
    __version__ = version("proface-vtk-post")
except PackageNotFoundError:
    __version__ = "0.0.0"
