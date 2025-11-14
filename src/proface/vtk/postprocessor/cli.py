# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import dataclasses
import itertools
import os

import h5py  # type: ignore[import-untyped]
import meshio  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import tyro

from proface.vtk.postprocessor import topotable


@dataclasses.dataclass
class Config:
    fea: os.PathLike  # input FEA file (.h5)
    out: os.PathLike  # output VTU file (.vtu)


class Mesh:
    """Container for ProFACE mesh object, as saved in neutral h5 format"""

    def __init__(self, h5: h5py.File) -> None:
        self.points: npt.NDArray[np.float32] = np.asarray(
            h5["nodes"]["coordinates"], dtype=np.float32
        )
        self.point_ids: npt.NDArray[np.int32] = np.asarray(
            h5["nodes"]["numbers"], dtype=np.int32
        )
        assert len(self.points) == len(self.point_ids)
        assert np.all(self.point_ids[:-1] < self.point_ids[1:]), (
            "point ids are not strictly sorted"
        )

        self.cells: list[tuple[str, npt.NDArray[np.int32]]] = []
        for abq_topo, dataset in h5["elements"].items():
            self.cells.append(
                (
                    topotable[abq_topo],
                    dataset["incidences"].astype(np.int32)[()],
                )
            )

        self._to_zerobased()

    @property
    def n_points(self) -> int:
        return len(self.points)

    def _to_zerobased(self) -> None:
        z_num = np.arange(self.n_points, dtype=np.int32)
        delta = self.point_ids - z_num
        if np.all(delta == delta[0]):
            for _, c in self.cells:
                c -= delta[0]
        else:
            assert np.all(delta[:-1] <= delta[1:])
            uv, uc = np.unique_counts(delta)
            assert np.all(uv[:-1] < uv[1:])
            # here we assume that uc is smallish
            for d, (a, b) in zip(
                uv,
                itertools.pairwise(itertools.accumulate(uc, initial=0)),
                strict=True,
            ):
                for _, c in self.cells:
                    c[
                        (c >= self.point_ids[a]) & (c <= self.point_ids[b - 1])
                    ] -= d


def main() -> int:
    config = tyro.cli(Config)

    with h5py.File(config.fea) as h5:
        inmesh = Mesh(h5)

    mesh = meshio.Mesh(points=inmesh.points, cells=inmesh.cells, cell_data={})
    mesh.write(config.out, file_format="vtu", binary=True, compression="zlib")
    print(config.out)

    return 0
