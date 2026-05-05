# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT


import logging
from collections.abc import Iterator

import h5py  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# see <https://github.com/nschloe/meshio/wiki/Node-ordering-in-cells>
topotable = {
    "C3D4": "tetra",
    "C3D5": "pyramid",
    "C3D6": "wedge",
    "C3D8": "hexahedron",
    "C3D10": "tetra10",
    "C3D13": "pyramid13",
    "C3D15": "wedge15",
    "C3D20": "hexahedron20",
}

dtype_bool = np.uint8
dtype_id = np.int32
dtype_fl = np.float32
NDArrVals = npt.NDArray[dtype_fl | dtype_bool | dtype_id]
NDArrIds = npt.NDArray[dtype_id]


class Mesh:
    """Container for ProFACE mesh object, as saved in neutral h5 format"""

    def __init__(
        self,
        h5: h5py.File,
        *,
        load_elsets: bool = False,
        load_nodesets: bool = False,
    ) -> None:
        self.points: NDArrVals
        self.point_ids: npt.NDArrIds
        self.point_data: dict[str, NDArrVals] = {}
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
            n_ids = np.asarray(dataset["incidences"], dtype=dtype_id)
            if np.any(np.isin(n_ids, self.point_ids, invert=True)):
                msg = (
                    f"element incidences for {abq_topo} "
                    "reference non existing node ids."
                )
                raise ValueError(msg)
            self.cells.append((abq_topo, n_ids))
            self.cell_ids.append(np.asarray(dataset["numbers"], dtype=dtype_id))

        # fake elsets with cell data
        if load_elsets:
            self._elset_to_cell_data(h5)

        # fake nodesets with point data
        if load_nodesets:
            self._nodeset_to_point_data(h5)

    @property
    def n_points(self) -> int:
        return len(self.points)

    def cells_zerobased(self) -> Iterator[tuple[str, NDArrIds]]:
        if not self.cells:
            return

        z_num = np.arange(self.n_points, dtype=dtype_id)
        delta: NDArrIds = self.point_ids - z_num
        if np.all(delta == delta[0]):
            d: dtype_id = delta[0]

            def remap(i: NDArrIds) -> NDArrIds:
                return i - d
        else:

            def remap(i: NDArrIds) -> NDArrIds:
                idx = np.searchsorted(self.point_ids, i)
                return idx.astype(dtype_id)

        for k, c in self.cells:
            yield k, remap(c)

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

    def load_fea_results(self, h5: h5py.File) -> None:
        """load neutral FEA results from h5 file"""

        results = h5.get("results")
        if results is None or len(results) == 0:
            # no results available in file
            return

        if not self.cells:
            # only results at topology sets are supported
            msg = "Cannot load FEA results without cells"
            raise ValueError(msg)

        (
            nodal_average_point_indices,
            nodal_average_topology_count,
        ) = self._build_fea_nodal_average_topology_mapping(h5)
        self.point_data["DEBUG::nodal_topology_count"] = (
            nodal_average_topology_count
        )
        for load_case in results:
            for quantity, paths in results[load_case].items():
                name = f"FEA::{load_case}::{quantity}"
                if "integration_point" in paths:
                    self._fea_integration_points_to_cell_data(
                        name,
                        paths["integration_point"],
                    )
                if "nodal_averaged" in paths:
                    self._fea_nodal_average_to_point_data(
                        name,
                        paths["nodal_averaged"],
                        nodal_average_point_indices,
                        nodal_average_topology_count,
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

    def _nodeset_to_point_data(self, h5: h5py.File) -> None:
        """load node sets as binary 1/0 point data"""

        try:
            nodesets = h5["sets"]["node"]
        except KeyError as err:
            msg = f"Invalid mesh file structure: {err}"
            raise ValueError(msg) from err

        for k, v in nodesets.items():
            name = f"NSet::{k}"
            ds = np.zeros((len(self.points),), dtype=dtype_bool)
            ds[np.isin(self.point_ids, v, assume_unique=True)] = 1
            self.point_data[name] = ds

    def _fea_integration_points_to_cell_data(
        self,
        name: str,
        ip_group: h5py.Group,
    ) -> None:
        """average FEA integration-point data onto cells."""

        self.cell_data[name] = []
        for e, m in self.cells:
            try:
                ds = ip_group[e]
            except KeyError as err:
                msg = f"Incomplete FEA results: {err}"
                raise ValueError(msg) from err
            if len(ds) != len(m) or ds.ndim < 2:
                msg = f"Invalid FEA results '{ds.name}'"
                raise ValueError(msg)
            # data structure:
            # axis0 -> element number
            # axis1 -> integration point number
            # axis2 (if present) -> vector/tensor component number
            values = np.asarray(ds, dtype=dtype_fl)
            # cell data is obtained by averaging over integration points
            self.cell_data[name].append(np.mean(values, axis=1))

    def _fea_nodal_average_to_point_data(
        self,
        name: str,
        nd_group: h5py.Group,
        point_indices_by_topology: tuple[NDArrIds, ...],
        topology_count: NDArrIds,
    ) -> None:
        """merge topology-specific FEA nodal averages as point data."""

        first_topology, _ = self.cells[0]
        try:
            value_shape = nd_group[first_topology].shape[1:]
        except KeyError as err:
            msg = f"Incomplete FEA results: {err}"
            raise ValueError(msg) from err

        accumulated = np.zeros(
            (self.n_points, *value_shape),
            dtype=dtype_fl,
        )
        for (topology, _), point_indices in zip(
            self.cells,
            point_indices_by_topology,
            strict=True,
        ):
            try:
                ds = nd_group[topology]
            except KeyError as err:
                msg = f"Incomplete FEA results: {err}"
                raise ValueError(msg) from err

            if len(ds) != len(point_indices) or ds.ndim < 1:
                msg = f"Invalid FEA results '{ds.name}'"
                raise ValueError(msg)

            if ds.shape[1:] != value_shape:
                msg = f"Inconsistent FEA results '{ds.name}'"
                raise ValueError(msg)

            # data structure:
            # axis0 -> node number
            # axis1 (if present) -> vector/tensor component number
            values = np.asarray(ds, dtype=dtype_fl)
            # nodal data is obtained by averaging topology contributions
            np.add.at(accumulated, point_indices, values)

        assert np.ndim(accumulated) == 1 + len(value_shape)
        assert np.ndim(topology_count) == 1
        # add singleton dimensions to 'topology_count' so that it
        # can broadcast to 'accumulated'
        topology_count_reshaped = topology_count[
            (...,) + (np.newaxis,) * len(value_shape)
        ]
        # compute actual average
        np.divide(
            accumulated,
            topology_count_reshaped,
            out=accumulated,
            where=topology_count_reshaped != 0,
        )
        # and set to nan nodes with no results
        accumulated[topology_count == 0] = np.nan

        self.point_data[name] = accumulated

    def _build_fea_nodal_average_topology_mapping(
        self,
        h5: h5py.File,
    ) -> tuple[tuple[NDArrIds, ...], NDArrIds]:
        """Build topology-specific nodal result mapping."""

        point_indices_by_topology: list[NDArrIds] = []
        topology_count = np.zeros((self.n_points,), dtype=dtype_id)
        for e, _ in self.cells:
            try:
                nodes = h5["elements"][e]["nodes"]
            except KeyError as err:
                msg = f"Incomplete FEA results: {err}"
                raise ValueError(msg) from err

            point_indices = self._point_indices(
                np.asarray(nodes, dtype=dtype_id)
            )
            point_indices_by_topology.append(point_indices)
            np.add.at(topology_count, point_indices, 1)

        return tuple(point_indices_by_topology), topology_count

    def _point_indices(self, point_ids: NDArrIds) -> NDArrIds:
        indices = np.searchsorted(self.point_ids, point_ids).astype(dtype_id)
        if (
            np.any(
                indices == self.n_points
            )  # insertion point past end of self.point_ids
            or np.any(
                self.point_ids[indices] != point_ids
            )  # insertion point not index of point_ids
        ):
            msg = "FEA result references unknown node ids"
            raise ValueError(msg)
        return indices
