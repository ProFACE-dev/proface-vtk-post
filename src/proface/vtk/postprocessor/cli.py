# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import dataclasses
import os

import h5py
import meshio
import numpy as np
import tyro

from proface.vtk.postprocessor import topotable


@dataclasses.dataclass
class Config:
    fea: os.PathLike  # input FEA file (.h5)
    out: os.PathLike  # output VTU file (.vtu)


def main() -> None:
    config = tyro.cli(Config)

    cells = []
    cell_data = {}

    with h5py.File(config.fea) as h5:
        points = h5["nodes"]["coordinates"].astype(np.float32)[()]
        numbering = h5["nodes"]["numbers"][()]
        for abq_topo, dataset in h5["elements"].items():
            cb = meshio.CellBlock(
                cell_type=topotable[abq_topo],
                data=dataset["incidences"].astype(np.int32)[()],
            )
            cells.append(cb)

    znum = np.arange(len(numbering), dtype=np.int32)
    delta = numbering[0]
    if np.all(numbering - znum == delta):
        for cb in cells:
            cb.data -= delta
    else:
        msg = "Non consecutive node numbering"
        raise NotImplementedError(msg)

    mesh = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    mesh.write(config.out, binary=True, compression="zlib")
    print(config.out)
