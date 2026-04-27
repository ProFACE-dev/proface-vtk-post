# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

__all__ = ["__version__"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"
