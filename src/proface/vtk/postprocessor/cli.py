# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import dataclasses
import pathlib
import sys

import h5py  # type: ignore[import-untyped]
import meshio  # type: ignore[import-untyped]
import tyro

from proface.vtk.postprocessor.mesh import Mesh, topotable


@dataclasses.dataclass
class Config:
    """ProFACE to VTU translator."""

    out: pathlib.Path
    """output VTU file (.vtu)"""

    fea: pathlib.Path
    """input FEA file (.h5)"""

    pfa: pathlib.Path | None = None
    """input PfA file (.h5) [optional]"""


def main() -> int:
    config = tyro.cli(Config)

    try:
        with h5py.File(config.fea) as h5:
            inmesh = Mesh(h5)
    except (OSError, ValueError) as err:
        print(f"Unable to parse FEA file '{config.fea}'", file=sys.stderr)
        print(err, file=sys.stderr)
        return 1

    if config.pfa is not None:
        try:
            with h5py.File(config.pfa) as h5:
                inmesh.load_results(h5)
        except (OSError, ValueError) as err:
            print(f"Unable to parse PfA file '{config.pfa}'", file=sys.stderr)
            print(err, file=sys.stderr)
            return 1

    mesh = meshio.Mesh(
        points=inmesh.points,
        cells=[
            meshio.CellBlock(
                cell_type=topotable[abq_topo],
                data=conn,
            )
            for abq_topo, conn in inmesh.cells_zerobased()
        ],
        cell_data=inmesh.cell_data,
    )
    mesh.write(config.out, file_format="vtu", binary=True, compression="zlib")
    print(config.out)

    return 0
