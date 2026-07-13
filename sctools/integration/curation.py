"""
sctools.integration.curation
==============================

Iterative low-quality-cluster curation log, used between successive
integration + clustering passes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def UpdateCellsToRemove(
    prefix_dir: str,
    original_file: str,
    res: str,
    clusters_to_remove: list,
    cells_to_remove_now: list,
) -> list:
    """
    Append a low-quality-cluster removal decision to a running log file and
    return the cumulative list of cells to remove across all logged decisions.

    Used for iterative curation: after each integration + clustering pass,
    low-quality clusters (doublets, stressed/dying cells, etc.) are identified
    and logged here before re-running integration on the filtered object. The
    log lives at `{prefix_dir}.cells_to_remove.txt` (tab-separated: original
    file, resolution, clusters removed, cell IDs).
    """
    file = Path(f"{prefix_dir}.cells_to_remove.txt")

    new_original_file = str(original_file)
    new_resolution = f"leiden_res{res}"
    new_clusters = ",".join(map(str, clusters_to_remove))
    new_cells = ",".join(map(str, cells_to_remove_now))

    if not file.exists():
        file.write_text("Original file\tResolution\tClusters to remove\tCells\n", encoding="utf-8")

    df_prev = pd.read_csv(file, sep="\t")

    if not df_prev.empty:
        same_row = (
            (df_prev["Original file"].astype(str) == new_original_file)
            & (df_prev["Resolution"].astype(str) == new_resolution)
            & (df_prev["Clusters to remove"].astype(str) == new_clusters)
            & (df_prev["Cells"].astype(str) == new_cells)
        )

        if same_row.any():
            logger.info("Identical row already logged; not adding it again.")
        else:
            with file.open("a", encoding="utf-8") as f:
                f.write(f"{new_original_file}\t{new_resolution}\t{new_clusters}\t{new_cells}\n")
            logger.info("cells_to_remove_now: %d", len(cells_to_remove_now))
    else:
        with file.open("a", encoding="utf-8") as f:
            f.write(f"{new_original_file}\t{new_resolution}\t{new_clusters}\t{new_cells}\n")
        logger.info("cells_to_remove_now: %d", len(cells_to_remove_now))

    df_prev = pd.read_csv(file, sep="\t")
    cells_to_remove: list = []

    if "Cells" in df_prev.columns and not df_prev.empty:
        for _, prev_row in df_prev.iterrows():
            prev_cells = prev_row["Cells"]
            if pd.notna(prev_cells):
                cells = [c.strip() for c in str(prev_cells).split(",") if c.strip()]
                cells_to_remove.extend(cells)
                logger.info(
                    "cells_to_remove_individual: clusters %s - %d",
                    prev_row["Clusters to remove"], len(cells),
                )

    cells_to_remove = list(dict.fromkeys(cells_to_remove))
    logger.info("cells_to_remove_all: %d", len(cells_to_remove))

    return cells_to_remove
