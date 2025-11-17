# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import dataclasses
import os

import h5py  # type: ignore[import-untyped]
import meshio  # type: ignore[import-untyped]
import tyro

from proface.vtk.postprocessor.mesh import Mesh, topotable


@dataclasses.dataclass
class Config:
    out: os.PathLike  # output VTU file (.vtu)
    fea: os.PathLike  # input FEA file (.h5)
    pfa: os.PathLike | None = None  # input PFA file (.h5)


def main() -> int:
    config = tyro.cli(Config)

    with h5py.File(config.fea) as h5:
        inmesh = Mesh(h5)

    if config.pfa is not None:
        with h5py.File(config.pfa) as h5:
            inmesh.load_results(h5)

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
