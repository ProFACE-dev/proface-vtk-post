# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import itertools
from collections.abc import Iterator

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

# see <https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells>
topotable = {
    "C3D4": "tetra",
    "C3D5": "pyramid",
    "C3D6": "wedge",
    "C3D8": "hexahedron",
    "C3D10": "tetra10",
    "C3D15": "wedge15",
    "C3D20": "hexahedron20",
}

dtype_bool = np.uint8
dtype_id = np.int32
dtype_fl = np.float32
NDArrVals = npt.NDArray[dtype_fl | dtype_bool]
NDArrIds = npt.NDArray[dtype_id]


class Mesh:
    """Container for ProFACE mesh object, as saved in neutral h5 format"""

    def __init__(self, h5: h5py.File, *, load_elsets: bool = False) -> None:
        self.points: NDArrVals
        self.point_ids: npt.NDArrIds
        self.cells: list[tuple[str, NDArrIds]] = []
        self.cell_ids: list[NDArrIds] = []
        self.cell_data: dict[str, list[NDArrVals]] = {}

        # populate points and point_ids
        try:
            self.points = np.asarray(h5["nodes"]["coordinates"], dtype=dtype_fl)
            self.point_ids = np.asarray(h5["nodes"]["numbers"], dtype=dtype_id)
        except KeyError as err:
            msg = f"Invalid mesh file structure: {err}"
            raise ValueError(msg) from err

        if len(self.points) != len(self.point_ids):
            msg = (
                "nodes/coordinates and nodes/number "
                "do not have same cardinality"
            )
            raise ValueError(msg)
        if not np.all(self.point_ids[:-1] < self.point_ids[1:]):
            msg = "point ids are not strictly sorted"
            raise ValueError(msg)

        # populate cells
        for abq_topo, dataset in h5["elements"].items():
            self.cells.append(
                (abq_topo, np.asarray(dataset["incidences"], dtype=dtype_id))
            )
            self.cell_ids.append(np.asarray(dataset["numbers"], dtype=dtype_id))

        # fake elsets with cell data
        self._elset_to_cell_data(h5)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def cells_zerobased(self) -> Iterator[tuple[str, NDArrIds]]:
        z_num = np.arange(self.n_points, dtype=dtype_id)
        delta = self.point_ids - z_num
        if np.all(delta == delta[0]):
            for k, c in self.cells:
                yield k, c - delta[0]
        else:
            assert np.all(delta[:-1] <= delta[1:])
            uv, uc = np.unique_counts(delta)
            assert np.all(uv[:-1] < uv[1:])
            # here we assume that uc is smallish
            for k, c in self.cells:
                c = np.copy(c)
                for d, (a, b) in zip(
                    uv,
                    itertools.pairwise(itertools.accumulate(uc, initial=0)),
                    strict=True,
                ):
                    c[
                        (c >= self.point_ids[a]) & (c <= self.point_ids[b - 1])
                    ] -= d
                yield k, c

    def load_results(self, h5: h5py.File) -> None:
        """load Local results from h5 file"""

        try:
            loc = h5["ProFACE"]["Local"]
        except KeyError as err:
            msg = f"Invalid results file structure: {err}"
            raise ValueError(msg) from err

        for k in loc:
            for v in loc[k]:
                name = f"{k}::{v}"
                self.cell_data[name] = []
                for e, m in self.cells:
                    try:
                        ds = loc[k][v]["integration_point"][e]
                    except KeyError as err:
                        msg = f"Incomplete ProFACE results: {err}"
                        raise ValueError(msg) from err
                    if len(ds) != len(m) or np.ndim(ds) != 2:
                        msg = (
                            "Invalid ProFACE results "
                            f"'{k}/{v}/integration_point/{e}'"
                        )
                        raise ValueError(msg)
                    self.cell_data[name].append(
                        np.mean(np.asarray(ds, dtype=dtype_fl), axis=1)
                    )

    def _elset_to_cell_data(self, h5: h5py.File) -> None:
        """load element sets as binary 1/0 cell data"""

        try:
            elsets = h5["sets"]["element"]
        except KeyError as err:
            msg = f"Invalid mesh file structure: {err}"
            raise ValueError(msg) from err

        for k, v in elsets.items():
            name = f"ElSet::{k}"
            self.cell_data[name] = []
            for (e, m), i in zip(self.cells, self.cell_ids, strict=True):
                ds = np.zeros((len(m),), dtype=dtype_bool)
                assert ds.shape == i.shape
                ds[np.isin(i, v, assume_unique=True)] = 1
                self.cell_data[name].append(ds)
