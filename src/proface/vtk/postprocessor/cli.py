# SPDX-FileCopyrightText: 2025 ProFACE developers
#
# SPDX-License-Identifier: MIT

import argparse
import dataclasses
import enum
import logging
import pathlib

import h5py  # type: ignore[import-untyped]
import meshio  # type: ignore[import-untyped]
import tyro
from rich.console import Console
from rich.logging import RichHandler

from proface.vtk.postprocessor import __version__
from proface.vtk.postprocessor.mesh import Mesh, topotable


class Loglevel(enum.Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    """ProFACE to VTU translator."""

    out: pathlib.Path
    """output VTU file (.vtu)"""

    fea: pathlib.Path
    """input FEA file (.h5)"""

    pfa: pathlib.Path | None = None
    """input PfA file (.h5) [optional]"""

    save_elsets: bool = True
    """save element sets as cell data"""

    save_nodesets: bool = True
    """save nodesets as point data"""

    save_fea_results: bool = False
    """save FEA results as cell and point data"""

    log_level: Loglevel = Loglevel.WARNING
    """logging verbosity"""


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    _, remaining = parser.parse_known_args()

    config = tyro.cli(Config, args=remaining)
    logging.basicConfig(
        level=config.log_level.value,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[RichHandler(console=Console(stderr=True), show_path=False)],
    )

    logger.info("\N{BLACK RIGHT-POINTING SMALL TRIANGLE} Translation started")
    try:
        with h5py.File(config.fea) as h5:
            logger.info("Parsing %s", config.fea)
            inmesh = Mesh(
                h5,
                load_elsets=config.save_elsets,
                load_nodesets=config.save_nodesets,
            )
            if config.save_fea_results:
                inmesh.load_fea_results(h5)
    except (OSError, ValueError) as err:
        logger.error("Unable to parse FEA file '%s'", config.fea)  # noqa: TRY400
        logger.error("%s", err)  # noqa: TRY400
        return 1

    if config.pfa is not None:
        try:
            with h5py.File(config.pfa) as h5:
                logger.info("Parsing %s", config.pfa)
                inmesh.load_results(h5)
        except (OSError, ValueError) as err:
            logger.error("Unable to parse PfA file '%s'", config.pfa)  # noqa: TRY400
            logger.error("%s", err)  # noqa: TRY400
            return 1

    logger.info("Creating meshio data structure")
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
        point_data=inmesh.point_data,
    )
    logger.info("%s", str(mesh))
    logger.info("Writing %s", config.out)
    mesh.write(config.out, file_format="vtu", binary=True, compression="zlib")
    logger.info("\N{CHECK MARK} Translation completed")
    print(config.out)

    return 0
