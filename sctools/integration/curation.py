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
import scanpy as sc
from anndata import AnnData

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
    # One log file per prefix_dir accumulates every curation decision made
    # against that lineage of adata files, across however many notebook cells
    # call this function.
    file = Path(f"{prefix_dir}.cells_to_remove.txt")

    # Stringify up front: these values get written as tab-separated text below,
    # and are compared as strings against df_prev (which round-trips through
    # CSV, so its dtypes wouldn't otherwise match cleanly).
    new_original_file = str(original_file)
    new_resolution = f"leiden_res{res}"
    new_clusters = ",".join(map(str, clusters_to_remove))
    new_cells = ",".join(map(str, cells_to_remove_now))

    # First call for this prefix_dir: create the log with a header row.
    if not file.exists():
        file.write_text("Original file\tResolution\tClusters to remove\tCells\n", encoding="utf-8")

    df_prev = pd.read_csv(file, sep="\t")

    if not df_prev.empty:
        # Idempotency guard: rerunning a notebook cell (debug, restart-and-rerun)
        # calls this with the same identifying params and, since clustering is
        # deterministic, the same cells -- without this check every rerun would
        # duplicate the row in the audit log. astype(str) on both sides avoids
        # int/float-vs-str mismatches from CSV round-tripping through read_csv.
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
        # Nothing to compare against yet (file has only the header) -- append
        # unconditionally.
        with file.open("a", encoding="utf-8") as f:
            f.write(f"{new_original_file}\t{new_resolution}\t{new_clusters}\t{new_cells}\n")
        logger.info("cells_to_remove_now: %d", len(cells_to_remove_now))

    # Re-read from disk rather than reusing df_prev/appending in memory, so
    # the row just written (whether new or an already-logged duplicate) is
    # included in the aggregation below through the same code path.
    df_prev = pd.read_csv(file, sep="\t")
    cells_to_remove: list = []

    # "Cells" in df_prev.columns is defensive (the header write above
    # guarantees it); not df_prev.empty skips a file that's just the header.
    if "Cells" in df_prev.columns and not df_prev.empty:
        # Walk every decision ever logged for this prefix_dir, not just this
        # call's -- the caller (notebook) re-applies the full cumulative
        # exclusion list against the pristine original adata on each curation
        # pass, not just the most recent removal.
        for _, prev_row in df_prev.iterrows():
            prev_cells = prev_row["Cells"]
            if pd.notna(prev_cells):
                # Reverse the ",".join(...) from when this row was written;
                # the empty-string filter guards against a row logged with
                # cells_to_remove_now == [] (would otherwise yield [""]).
                cells = [c.strip() for c in str(prev_cells).split(",") if c.strip()]
                cells_to_remove.extend(cells)
                logger.info(
                    "cells_to_remove_individual: clusters %s - %d",
                    prev_row["Clusters to remove"], len(cells),
                )

    # A cell could be listed in more than one logged row (e.g. a re-run that
    # got logged before this idempotency guard existed, or overlapping manual
    # decisions) -- dict.fromkeys dedupes while preserving log order, unlike set().
    cells_to_remove = list(dict.fromkeys(cells_to_remove))
    logger.info("cells_to_remove_all: %d", len(cells_to_remove))

    return cells_to_remove


def RemoveClustersFromOriginal(
    prefix_dir: str,
    integrated_path: str | Path,
    original_path: str | Path,
    res: str,
    clusters_to_remove: list,
) -> AnnData:
    """
    Flag cells in `clusters_to_remove` from an integrated+clustered AnnData,
    log that decision via `UpdateCellsToRemove`, then reload the pristine
    `original_path` AnnData and drop every cell logged so far under `prefix_dir`.

    Used between successive integration + clustering passes: after each round
    of integration you inspect the clusters and decide which are low-quality
    or off-target, then call this to get back a freshly filtered copy of the
    *original* (pre-any-exclusion) AnnData, ready to re-integrate from scratch.
    The returned object is filtered against the full cumulative log, not just
    this pass's clusters -- matching `UpdateCellsToRemove`'s accumulation
    across every prior decision logged under `prefix_dir`.

    Parameters
    ----------
    prefix_dir
        Passed through to `UpdateCellsToRemove`; the curation log lives at
        `{prefix_dir}.cells_to_remove.txt`.
    integrated_path
        Path to the integrated+clustered AnnData (.h5ad) from the previous pass.
    original_path
        Path to the pristine AnnData to reload and filter.
    res
        Leiden resolution suffix, matching an existing `obs["leiden_res{res}"]`
        column in the integrated AnnData.
    clusters_to_remove
        Cluster labels (as they appear in `leiden_res{res}`) to exclude.

    Returns
    -------
    AnnData
        `original_path`, reloaded and filtered to exclude every cell logged
        under `prefix_dir` so far (this pass and all previous ones).
    """
    integrated = sc.read_h5ad(integrated_path)
    logger.info("Integrated adata: %s", integrated.shape)

    cluster_col = f"leiden_res{res}"
    if cluster_col not in integrated.obs:
        raise KeyError(
            f"'{cluster_col}' not found in adata.obs. Available leiden columns: "
            f"{[c for c in integrated.obs.columns if c.startswith('leiden_')]}"
        )

    cells_to_remove_now = integrated.obs.index[
        integrated.obs[cluster_col].isin(clusters_to_remove)
    ].tolist()
    logger.info("Cells to remove now: %d", len(cells_to_remove_now))
    del integrated

    # original_file logs *which integrated snapshot* this decision was made
    # from, not the pristine file being filtered below -- matches
    # UpdateCellsToRemove's existing semantics (see its docstring/log header).
    cells_to_remove = UpdateCellsToRemove(
        prefix_dir=prefix_dir,
        original_file=str(integrated_path),
        res=res,
        clusters_to_remove=clusters_to_remove,
        cells_to_remove_now=cells_to_remove_now,
    )

    adata_original = sc.read_h5ad(original_path)
    logger.info("Original adata: %s", adata_original.shape)

    adata = adata_original[~adata_original.obs_names.isin(cells_to_remove)].copy()
    logger.info("Filtered adata: %s", adata.shape)
    del adata_original

    return adata
