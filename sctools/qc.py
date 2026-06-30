
"""
sctools.qc
==========
Quality-control functions for single-cell RNA-seq data.

Typical usage order
-------------------
1. :func:`AmbientRNA`           – ambient RNA correction via SoupX (requires R)
2. :func:`ScoreDoublets`        – score cells as potential doublets via Scrublet
3. :func:`FilterDoublets`       – remove doublets by score threshold
4. :func:`CalculateQcMetrics`   – annotate adata with per-cell QC metrics
5. :func:`EvaluateQcThresholds` – explore threshold combinations without filtering
6. :func:`FilterQcCells`        – apply chosen thresholds to remove low-quality cells

Quick start
-----------
>>> from sctools.qc import *
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation

import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

# Gene-symbol regex patterns — case-insensitive to cover human (MT-) and mouse (mt-)
_MT_PATTERN   = r"(?i)^mt-"
_RIBO_PATTERN = r"(?i)^rp[sl]"
_HB_PATTERN   = r"(?i)^hb(?:a|b|d|e|g|m|q|z)"

########################################################################################
# ── 1. AMBIENT RNA CORRECTION ─────────────────────────────────────────────────────────


def _SoupXClustering(adata):

    """
    Runs Leiden clustering on the input AnnData object to generate clusters for SoupX.
    Code exactly as in the Single Cell Best Practices book (https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html).
    """
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)

    sc.pp.pca(adata_pp)
    sc.pp.neighbors(adata_pp)
    sc.tl.leiden(adata_pp, key_added="soupx_groups", flavor="igraph", n_iterations=2, directed=False)

    # Preprocess variables for SoupX
    soupx_groups = adata_pp.obs["soupx_groups"]

    return soupx_groups

def _RunSoupx(data, data_tod, genes, cells, soupx_groups):
    from rpy2.robjects import r, globalenv
    import rpy2.robjects.packages as rpackages
    import rpy2.rinterface_lib.callbacks as rcb
    import rpy2.robjects as ro
    import anndata2ri
    anndata2ri.activate()

    # Pass variables to the R environment
    globalenv['data'] = data
    globalenv['data_tod'] = data_tod
    globalenv['genes'] = genes
    globalenv['cells'] = cells
    globalenv['soupx_groups'] = soupx_groups

    r_code = """
    library(SoupX)
    
    output_stdout <- capture.output({
      
      ### Run SoupX as in the Single Cell Best Practices book ###
      rownames(data) <- genes
      colnames(data) <- cells
      
      # ensure correct sparse format for table of counts and table of droplets
      data <- as(data, "sparseMatrix")
      data_tod <- as(data_tod, "sparseMatrix")
      
      # Generate SoupChannel Object for SoupX 
      sc <- SoupChannel(data_tod, data, calcSoupProfile = FALSE)
      
      # Add extra meta data to the SoupChannel object
      soupProf <- data.frame(row.names = rownames(data), 
                             est = rowSums(data)/sum(data), 
                             counts = rowSums(data))
      sc <- setSoupProfile(sc, soupProf)

      # Set cluster information in SoupChannel
      sc <- setClusters(sc, soupx_groups)

      # Estimate contamination fractions
      sc <- autoEstCont(sc, doPlot = FALSE, forceAccept = TRUE)

      # Infer corrected table of counts and rount to integers
      out <- adjustCounts(sc, roundToInt = TRUE)
    }, type = "output")
    
    ### Capture stdout and stderr messages ###
    output_messages <- capture.output({
      rownames(data) <- genes
      colnames(data) <- cells
      
      data <- as(data, "sparseMatrix")
      data_tod <- as(data_tod, "sparseMatrix")
      
      sc <- SoupChannel(data_tod, data, calcSoupProfile = FALSE)
      soupProf <- data.frame(row.names = rownames(data), 
                             est = rowSums(data)/sum(data), 
                             counts = rowSums(data))
      sc <- setSoupProfile(sc, soupProf)
      sc <- setClusters(sc, soupx_groups)
      sc <- autoEstCont(sc, doPlot = FALSE, forceAccept = TRUE)
      out <- adjustCounts(sc, roundToInt = TRUE)
    }, type = "message")
    
    all_output <- c(output_stdout, output_messages)
    result <- list(out = out, messages = all_output)
    """
    r(r_code)
    result = globalenv['result']
    # Join R output lines into a single string
    messages = "\n".join(list(result["messages"]))
    out = result["out"]

    return messages, out

def AmbientRNA(adata, adata_raw):

    # Cluster adata to generate SoupX groups
    soupx_groups = _SoupXClustering(adata)

    # Subset to genes present in both matrices (BD unfiltered pool may differ)
    common_genes = adata.var_names[adata.var_names.isin(adata_raw.var_names)]
    adata_raw = adata_raw[:, common_genes].copy()
    data_tod = adata_raw.X.T

    # Extract required variables
    cells = adata.obs_names
    genes = common_genes
    data = adata[:, common_genes].X.T

    del adata_raw

    # Run SoupX
   
    msg, out = _RunSoupx(data, data_tod, genes, cells, soupx_groups)

    rho = float(msg.split('\n')[2].split('of ')[-1])

    # SoupX successfully inferred corrected counts, which we can now store as an additional layer. 
    # In all following analysis steps, we would like to use the SoupX corrected count matrix, so we overwrite 
    # .X with the soupX corrected matrix.
    import scipy.sparse as _sp
    adata.layers["counts"] = adata.X.copy()
    # SoupX only returns corrected counts for common_genes; copy original for the rest
    soupx_counts = adata.X.copy()
    if _sp.issparse(soupx_counts):
        soupx_counts = soupx_counts.toarray()
    soupx_counts = soupx_counts.astype("float32")
    gene_idx = adata.var_names.get_indexer(common_genes)
    soupx_out = out.T if not _sp.issparse(out) else out.T.toarray()
    soupx_counts[:, gene_idx] = soupx_out.astype("float32")
    import scipy.sparse as _sp2
    adata.layers["soupX_counts"] = _sp2.csr_matrix(soupx_counts)
    adata.X = adata.layers["soupX_counts"]

    adata.uns.setdefault("qc_params", {})["soupx"] = {"rho": rho}

    return adata


########################################################################################
# ── 2. Doublet Detection ─────────────────────────────────────────────────────────

def ScoreDoublets(
    adata: AnnData,
    *,
    layer: str | None = "counts",
    expected_doublet_rate: float = 0.06,
    batch_key: str | None = None,
    random_state: int = 42,
    copy: bool = False,
) -> "AnnData | dict":
    
    """Score cells as potential doublets using Scrublet.

    Runs on raw counts (``layer`` or ``.X``). Does not threshold,
    plot, or filter — call :func:`CallDoublets` next.

    Parameters
    ----------
    layer :
        Layer with count data (raw or ambient-corrected) — never normalized.
        ``None`` uses ``.X`` directly, which is correct after :func:`AmbientRNA`
        since ``.X`` is already the corrected counts.
    copy :
        ``True`` → return modified copy of ``adata``.
        ``False`` → modify in-place, return summary dict.
    """
    if copy:
        adata = adata.copy()

    # Validate layer and temporarily swap .X so scrublet sees raw counts.
    # .X is restored in the finally block even if scrublet raises an exception.
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}. "
                "Use layer=None to use .X directly."
            )
        original_X, adata.X = adata.X, adata.layers[layer]

    try:
        sc.pp.scrublet(
            adata,
            expected_doublet_rate=expected_doublet_rate,
            batch_key=batch_key,
            random_state=random_state,
            threshold=None,
            verbose=False,
        )
    finally:
        if layer is not None:
            adata.X = original_X

    adata.uns.setdefault("qc_params", {})
    adata.uns["qc_params"]["scrublet"] = {
        "layer": layer,
        "expected_doublet_rate": expected_doublet_rate,
        "batch_key": batch_key,
        "random_state": random_state,
    }

    summary = {
        "n_cells": adata.n_obs,
        "layer": layer,
        "expected_doublet_rate": expected_doublet_rate,
        "batch_key": batch_key,
    }
    logger.info(
        "ScoreDoublets: %d cells scored [layer=%s, expected_doublet_rate=%.2f].",
        adata.n_obs, layer, expected_doublet_rate,
    )
    return adata if copy else summary


def PlotScrubletScores(
    adata: AnnData,
    *,
    threshold: float | None = None,
    show: bool = True,
    save_path: str | None = None,
):
    """Plot observed and simulated Scrublet doublet score distributions.

    Mirrors the two-panel view of ``sc.pl.scrublet_score_distribution``,
    with an optional threshold line on both panels.
    Does not modify ``adata``.

    Parameters
    ----------
    threshold :
        If provided, draws a vertical dashed line at this value on both panels.
    show :
        Whether to display the figure interactively.
    save_path :
        Full file path to save the figure. ``None`` skips saving.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if "doublet_score" not in adata.obs:
        raise ValueError("'doublet_score' not found. Run ScoreDoublets() first.")
    if "scrublet" not in adata.uns or "doublet_scores_sim" not in adata.uns["scrublet"]:
        raise ValueError("Simulated scores not found. Run ScoreDoublets() first.")

    obs_scores = adata.obs["doublet_score"].values
    sim_scores = adata.uns["scrublet"]["doublet_scores_sim"]

    fig, (ax_obs, ax_sim) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, scores, title in [
        (ax_obs, obs_scores, "Observed transcriptomes"),
        (ax_sim, sim_scores, "Simulated doublets"),
    ]:
        ax.hist(scores, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
        if threshold is not None:
            ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"threshold = {threshold}")
            ax.legend(frameon=False)
        ax.set_xlabel("Doublet score")
        ax.set_ylabel("Number of cells")
        ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return fig


def CallDoublets(
    adata: AnnData,
    *,
    threshold: float,
    score_col: str = "doublet_score",
    doublet_col: str = "is_doublet",
) -> dict:
    """Label cells as doublets based on a chosen threshold.

    Creates/updates ``adata.obs[doublet_col]`` (boolean) and records the
    threshold in ``adata.uns["qc_params"]["scrublet"]``.
    Does not filter — call :func:`FilterDoublets` next.

    Parameters
    ----------
    threshold :
        Score cutoff. Cells with ``score >= threshold`` are labelled doublets.
    score_col :
        Column in ``adata.obs`` with Scrublet scores.
    doublet_col :
        Column name to write the boolean doublet label into.

    Returns
    -------
    dict
        Summary with ``threshold``, ``n_doublets``, and ``doublet_rate``.
    """
    if score_col not in adata.obs:
        raise ValueError(
            f"'{score_col}' not found in adata.obs. Run ScoreDoublets() first."
        )

    adata.obs[doublet_col] = adata.obs[score_col] >= threshold

    n_doublets = int(adata.obs[doublet_col].sum())
    doublet_rate = round(n_doublets / adata.n_obs, 4)

    # Update scrublet metadata with the chosen threshold and results
    adata.uns.setdefault("qc_params", {}).setdefault("scrublet", {})
    adata.uns["qc_params"]["scrublet"].update({
        "threshold": threshold,
        "n_doublets": n_doublets,
        "doublet_rate": doublet_rate,
    })

    summary = {
        "threshold": threshold,
        "n_doublets": n_doublets,
        "doublet_rate": doublet_rate,
    }
    logger.info(
        "CallDoublets: threshold=%.3f → %d doublets (%.1f%% of %d cells).",
        threshold, n_doublets, doublet_rate * 100, adata.n_obs,
    )
    return summary


def FilterDoublets(
    adata: AnnData,
    *,
    threshold: float | None = None,
    score_col: str = "doublet_score",
    doublet_col: str = "is_doublet",
    run_if_missing: bool = True,
    layer: str | None = None,
    expected_doublet_rate: float = 0.06,
    batch_key: str | None = None,
    random_state: int = 0,
    drop_doublet_col: bool = False,
) -> tuple[AnnData, dict]:
    """Filter doublets from an AnnData object.

    Resolution paths, in order of priority:

    1. ``threshold`` provided + ``score_col`` exists → re-labels with new
       threshold (overwrites ``doublet_col`` if it existed) then filters.
    2. ``threshold=None`` + ``doublet_col`` exists → use existing label.
    3. ``threshold=None`` + ``doublet_col`` missing → raises ``ValueError``.
    4. Neither column exists + ``run_if_missing=True`` → runs
       :func:`ScoreDoublets`, :func:`CallDoublets`, then filters.
       ``threshold`` is required in this case.

    Parameters
    ----------
    threshold :
        When provided, always re-labels cells via :func:`CallDoublets`
        (even if ``doublet_col`` already exists), so you can safely try
        different thresholds after inspecting the score histogram.
        When ``None``, the existing ``doublet_col`` is used directly.
    drop_doublet_col :
        Remove ``doublet_col`` from the filtered object before returning.

    Returns
    -------
    tuple[AnnData, dict]
        ``(adata_singlets, summary)`` where summary contains
        ``n_removed``, ``n_remaining``, and ``threshold`` (if used).
    """
    has_label = doublet_col in adata.obs
    has_score = score_col in adata.obs

    if threshold is not None and has_score:
        # Explicit threshold always wins — re-label even if doublet_col exists
        CallDoublets(adata, threshold=threshold, score_col=score_col,
                      doublet_col=doublet_col)

    elif threshold is not None and not has_score:
        if run_if_missing:
            ScoreDoublets(adata, layer=layer,
                           expected_doublet_rate=expected_doublet_rate,
                           batch_key=batch_key, random_state=random_state)
            CallDoublets(adata, threshold=threshold, score_col=score_col,
                          doublet_col=doublet_col)
        else:
            raise ValueError(
                f"'{score_col}' not found. Run ScoreDoublets() first, or set "
                "run_if_missing=True to score and label automatically."
            )

    elif threshold is None and not has_label:
        raise ValueError(
            f"'{doublet_col}' not found in adata.obs and no threshold was provided. "
            "Either run CallDoublets() first, or pass threshold= to this function."
        )

    n_doublets = int(adata.obs[doublet_col].sum())
    adata_singlets = adata[~adata.obs[doublet_col]].copy()

    if drop_doublet_col and doublet_col in adata_singlets.obs:
        adata_singlets.obs.drop(columns=[doublet_col], inplace=True)

    summary = {
        "n_removed": n_doublets,
        "n_remaining": adata_singlets.n_obs,
        "threshold": threshold,
    }
    # Store in uns so QcSummaryTable can retrieve it later
    adata_singlets.uns.setdefault("scrublet", {})["n_doublets_removed"] = n_doublets
    logger.info(
        "FilterDoublets: removed %d doublets, %d cells remaining.",
        n_doublets, adata_singlets.n_obs,
    )
    return adata_singlets, summary


##################################################################################
# ── 3. QC Metrics ────────────────────────────────────────────────────────────

def CalculateQcMetrics(adata) -> None:
    adata.var["mt"]   = adata.var_names.str.contains(_MT_PATTERN,   regex=True)
    adata.var["ribo"] = adata.var_names.str.contains(_RIBO_PATTERN, regex=True)
    adata.var["hb"]   = adata.var_names.str.contains(_HB_PATTERN,   regex=True)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )


def QCmetric(adata):
    CalculateQcMetrics(adata)


## Evaluate threshold combinations without filtering

_LOG1P_COUNT_COLS = frozenset({"n_genes_by_counts", "total_counts"})
_SUMMARY_STAT_COLS = {
    "median_n_genes":      "n_genes_by_counts",
    "median_total_counts": "total_counts",
    "median_pct_mt":       "pct_counts_mt",
}


def _scenarios_absolute(adata, threshold_grid):
    """Yield (meta_dict, keep_mask) for each combination of absolute thresholds.

    threshold_grid example::

        {"n_genes_by_counts": {"min": [200, 500]},
         "pct_counts_mt":     {"max": [10, 15, 20]}}
    """
    # Flatten each metric's spec into a list of (direction, value) pairs.
    # e.g. {"min": [200, 500]} → [("min", 200), ("min", 500)]
    per_metric = {}                                    # will hold {metric: [(direction, value), ...]}
    for metric, spec in threshold_grid.items():        # loop over each metric in the grid
        entries = []
        for direction, values in spec.items():         # direction is "min" or "max"
            if direction not in ("min", "max"):
                raise ValueError(f"Direction '{direction}' must be 'min' or 'max'.")
            for v in values:                           # one entry per threshold value
                entries.append((direction, v))
        per_metric[metric] = entries                   # store all candidates for this metric

    # Cartesian product across metrics — one combo per scenario
    # e.g. combo = (("min", 200), ("max", 10)) for one scenario
    metrics = list(per_metric)                                # ordered list of metric names
    for combo in itertools.product(*per_metric.values()):    # one combo = one threshold per metric
        keep = np.ones(adata.n_obs, dtype=bool)              # start with all cells passing; AND each condition below
        meta = {}                                            # will store the threshold values for this scenario
        for metric, (direction, value) in zip(metrics, combo):  # pair each metric name with its chosen threshold
            meta[f"{metric}_{direction}"] = value               # record threshold in summary row
            col = adata.obs[metric].values                      # get the per-cell values for this metric
            keep &= col >= value if direction == "min" else col <= value  # AND the condition into the mask
        yield meta, keep                                        # deliver this scenario's metadata and cell mask


def _mad_bounds(values, nmads, tail):
    med = np.median(values)                            # median of the distribution
    mad = median_abs_deviation(values)                 # median absolute deviation
    lo = med - nmads * mad if tail in ("lower", "both") else -np.inf   # lower bound (or -inf if not needed)
    hi = med + nmads * mad if tail in ("upper", "both") else  np.inf   # upper bound (or +inf if not needed)
    return lo, hi                                      # return the two cut-off values


def _scenarios_mads(adata, threshold_grid):
    """Yield (meta_dict, keep_mask) for each combination of MAD-based thresholds."""
    obs = adata.obs
    per_metric = {}                                    # will hold {metric: [(lo, hi, meta), ...]}
    for metric, spec in threshold_grid.items():        # loop over each metric in the grid
        tail = spec.get("tail", "both")                # which side(s) to cut: "lower", "upper", or "both"
        # Count metrics are right-skewed: compute MADs on log1p scale if available
        use_log1p = metric in _LOG1P_COUNT_COLS and f"log1p_{metric}" in obs.columns
        source_col = f"log1p_{metric}" if use_log1p else metric   # column to compute MAD on
        entries = []
        for nmads in spec["mads"]:                     # one entry per nmads value
            lo_t, hi_t = _mad_bounds(obs[source_col].values, nmads, tail)  # bounds in transformed space
            meta = {f"{metric}_nmads": nmads}          # record nmads in summary row
            if use_log1p:
                # Report bounds in log1p space and convert back to original scale
                meta[f"{metric}_lower_log1p"] = round(lo_t, 4) if np.isfinite(lo_t) else None  # lower bound in log1p
                meta[f"{metric}_upper_log1p"] = round(hi_t, 4) if np.isfinite(hi_t) else None  # upper bound in log1p
                lo_orig = np.expm1(lo_t) if np.isfinite(lo_t) else -np.inf   # convert lower bound back to counts
                hi_orig = np.expm1(hi_t) if np.isfinite(hi_t) else  np.inf   # convert upper bound back to counts
            else:
                lo_orig, hi_orig = lo_t, hi_t          # pct metrics: bounds already in original scale
            meta[f"{metric}_lower_bound_original"] = round(lo_orig, 2) if np.isfinite(lo_orig) else None
            meta[f"{metric}_upper_bound_original"] = round(hi_orig, 2) if np.isfinite(hi_orig) else None
            entries.append((lo_orig, hi_orig, meta))   # store bounds + metadata for this nmads
        per_metric[metric] = entries                   # store all nmads entries for this metric

    # Iterate over every cross-metric combination and apply bounds on original scale
    metrics = list(per_metric)                         # ordered list of metric names
    for combo in itertools.product(*per_metric.values()):   # one combo = one nmads per metric
        keep = np.ones(adata.n_obs, dtype=bool)        # start with all cells passing
        meta = {}
        for metric, (lo, hi, m) in zip(metrics, combo):    # pair metric name with its chosen bounds
            meta.update(m)                             # merge this metric's metadata into the row
            col = obs[metric].values                   # per-cell values in original scale
            keep &= (col >= lo) & (col <= hi)          # AND both bounds into the mask
        yield meta, keep                               # deliver this scenario


def EvaluateQcThresholds(adata, threshold_grid, *, method="absolute", return_cell_flags=False):
    """Test combinations of QC thresholds without filtering any cells.

    Parameters
    ----------
    threshold_grid :
        Absolute mode::

            {"n_genes_by_counts": {"min": [200, 500]},
             "pct_counts_mt":     {"max": [10, 15, 20]}}

        MADs mode::

            {"n_genes_by_counts": {"mads": [3, 4, 5], "tail": "both"},
             "pct_counts_mt":     {"mads": [3, 4, 5], "tail": "upper"}}

    method : ``"absolute"`` or ``"mads"``
    return_cell_flags :
        If ``True``, also return a per-cell boolean DataFrame (True = kept).

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
    """
    if method not in ("absolute", "mads"):             # validate method before doing any work
        raise ValueError(f"method must be 'absolute' or 'mads', got '{method}'.")

    for metric in threshold_grid:                      # check all requested metrics exist in adata.obs
        if metric not in adata.obs.columns:
            raise ValueError(
                f"'{metric}' not found in adata.obs. Run CalculateQcMetrics() first."
            )

    gen = _scenarios_absolute if method == "absolute" else _scenarios_mads   # pick the right generator
    obs = adata.obs
    rows, flags = [], {}                               # rows → summary table; flags → per-cell masks

    for i, (row_meta, keep) in enumerate(gen(adata, threshold_grid)):   # iterate over each scenario
        n_kept    = int(keep.sum())                    # number of cells that pass all filters
        n_removed = adata.n_obs - n_kept              # number of cells that would be removed
        removed   = ~keep                              # boolean mask of removed cells

        row = {
            "cutoff_id": i, "method": method, **row_meta,   # scenario id + threshold values
            "n_cells_total": adata.n_obs,
            "n_removed":     n_removed,
            "pct_removed":   round(100 * n_removed / adata.n_obs, 2),
            "n_kept":        n_kept,
        }
        for name, col in _SUMMARY_STAT_COLS.items():  # add median stats for removed and kept cells
            if col in obs:
                s = obs[col]
                row[f"{name}_removed"] = round(float(s[removed].median()), 2) if n_removed else np.nan
                row[f"{name}_kept"]    = round(float(s[keep].median()),    2) if n_kept    else np.nan

        rows.append(row)                               # add this scenario to the result list
        if return_cell_flags:
            flags[i] = keep                            # store per-cell mask for this scenario

    summary_df = pd.DataFrame(rows)                    # convert list of dicts to DataFrame

    if return_cell_flags:
        flags_df = pd.DataFrame(flags, index=adata.obs_names).rename(
            columns=lambda c: f"cutoff_{c}"           # rename columns to cutoff_0, cutoff_1, …
        )
        return summary_df, flags_df

    return summary_df



def FilterQcCells(adata, cutoffs, *, method="absolute"):
    """Filter low-quality cells by absolute thresholds or MAD-based bounds.

    Parameters
    ----------
    cutoffs : dict
        Absolute mode::

            {"n_genes_by_counts": {"min": 200},
             "pct_counts_mt":     {"max": 15}}

        MADs mode::

            {"n_genes_by_counts": {"mads": 3, "tail": "lower"},
             "pct_counts_mt":     {"mads": 3, "tail": "upper"}}

    method : "absolute" or "mads"
    """
    if method not in ("absolute", "mads"):
        raise ValueError(f"method must be 'absolute' or 'mads', got '{method}'.")

    mask = np.ones(adata.n_obs, dtype=bool)

    for metric, spec in cutoffs.items():
        if metric not in adata.obs.columns:
            raise ValueError(f"'{metric}' not in adata.obs. Run CalculateQcMetrics() first.")

        if method == "absolute":
            col = adata.obs[metric].values
            if "min" in spec:
                mask &= col >= spec["min"]
            if "max" in spec:
                mask &= col <= spec["max"]

        else:  # mads
            tail = spec.get("tail", "both")
            use_log1p = metric in _LOG1P_COUNT_COLS and f"log1p_{metric}" in adata.obs.columns
            source_col = f"log1p_{metric}" if use_log1p else metric
            lo_t, hi_t = _mad_bounds(adata.obs[source_col].values, spec["mads"], tail)
            lo = np.expm1(lo_t) if use_log1p and np.isfinite(lo_t) else lo_t
            hi = np.expm1(hi_t) if use_log1p and np.isfinite(hi_t) else hi_t
            col = adata.obs[metric].values
            mask &= (col >= lo) & (col <= hi)

    n_removed = int((~mask).sum())
    adata_filt = adata[mask].copy()
    print(f"{adata.n_obs} → {adata_filt.n_obs} células  ({n_removed} removidas)")
    return adata_filt


################################################################################
# ── 4. Plots ──────────────────────────────────────────────────────────────────

## Violin grid containing all samples in the same plot, for comparision
def PlotQCViolinsGrid(
    adata,
    l_metricas=['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],  # default metrics
    title='Quality Control Metrics',
    sample_key="sample",
    samples=None,          # can be a list of sample names or a single string
    save_dir=None,
    show_plot=True,
    suffix='',
    jitter=0.3,            # jitter spread for strip plot
    point_size=0.2,        # dot size in strip plot
    rotation=0,
    dpi=200,
    figsize=(15, 4),       # wider figure for 3 side-by-side panels
    ymin=0,                # lower Y-axis limit (None = auto)
    ymax=None,             # upper Y-axis limit (None = auto)
):
    import matplotlib.pyplot as plt

    # Validate that sample_key and all requested metrics exist
    if sample_key not in adata.obs:
        raise KeyError(f"'{sample_key}' not found in adata.obs.")
    missing = [m for m in l_metricas if m not in adata.obs.columns]
    if missing:
        raise KeyError(f"Columns missing from adata.obs: {missing}")

    # Allow samples to be passed as a single string
    if samples is not None and isinstance(samples, str):
        samples = [samples]

    # Subset to requested samples if provided
    ad = adata if samples is None else adata[adata.obs[sample_key].isin(samples)].copy()
    ad.uns.pop(f"{sample_key}_colors", None)  # avoid color mismatch after concat
    order = sorted(ad.obs[sample_key].unique(), key=lambda x: str(x))

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # One axis per metric, all in a single row
    fig, axes = plt.subplots(1, len(l_metricas), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, metrica in zip(axes, l_metricas):
        sc.pl.violin(
            ad,
            keys=metrica,
            groupby=sample_key,
            order=order,
            stripplot=True,
            jitter=jitter,
            size=point_size,
            rotation=0,        # rotation applied below via tick labels
            show=False,
            ax=ax,
        )
        ax.set_ylabel(metrica)
        ax.set_xlabel("Sample")
        # Y-axis limits: accept a single value or a per-metric dict
        _ymin = ymin.get(metrica, 0) if isinstance(ymin, dict) else ymin
        _ymax = ymax.get(metrica, None) if isinstance(ymax, dict) else ymax
        cur_min, cur_max = ax.get_ylim()
        ax.set_ylim(
            _ymin if _ymin is not None else cur_min,
            _ymax if _ymax is not None else cur_max,
        )
        # Rotate x-axis tick labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
            tick.set_ha('right')
        # Soft horizontal grid
        ax.grid(True, which='major', axis='y', alpha=0.25, linestyle='--', linewidth=0.6)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_dir is not None:
        Path(save_dir).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()

    return fig, axes

################################################################################


def PlotQcHeatmap(
    summary_df,
    *,
    x: str,
    y: str,
    title: str = "QC threshold impact",
    figsize: tuple = (7, 5),
    show: bool = True,
    save_path: str | None = None,
):
    """Heatmap of QC threshold combinations.

    Colour encodes ``pct_removed``; cell annotations show ``n_removed``.
    ``x`` and ``y`` are columns from ``summary_df`` returned by
    :func:`EvaluateQcThresholds` — choose them in the notebook.

    Parameters
    ----------
    x, y : threshold columns (e.g. ``"n_genes_by_counts_min"``,
           ``"pct_counts_mt_nmads"``).
    title : plot title.
    """
    from sctools.plots import PlotHeatmap

    total_unique = summary_df["n_cells_total"].dropna().unique()
    full_title = (
        f"{title}\nTotal cells before filtering: {int(total_unique[0]):,}\nN cells removed:"
        if len(total_unique) == 1 else title
    )

    fig, ax, _ = PlotHeatmap(
        summary_df,
        x=x, y=y,
        color_col="pct_removed",
        annotation_col="n_removed",
        title=full_title,
        figsize=figsize,
        cmap="RdYlGn_r",
        colorbar_label="% cells removed",
        show=show,
        save_path=save_path,
    )
    return fig, ax


def PlotQcThresholdsGrid(
    summary_dfs: dict,
    *,
    x: str,
    y: str,
    ncols: int = 2,
    title: str = "QC threshold impact",
    figsize_per_panel: tuple = (5, 4),
    show: bool = True,
    save_path: str | None = None,
):
    """Plot QC threshold heatmaps for multiple samples side by side in a grid.

    Parameters
    ----------
    summary_dfs : dict of {sample_name: summary_df} from EvaluateQcThresholds.
    x, y : threshold columns to use as axes.
    ncols : number of columns in the grid (default 2).
    figsize_per_panel : size of each individual panel in inches.
    """
    from sctools.plots import PlotHeatmap

    if not summary_dfs:
        raise ValueError(
            "summary_dfs is empty. Build it with: "
            "summary_dfs[sample] = EvaluateQcThresholds(...) inside your loop."
        )

    samples = list(summary_dfs.keys())
    n = len(samples)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, sample in enumerate(samples):
        ax = axes[idx // ncols][idx % ncols]
        sdf = summary_dfs[sample]
        total_unique = sdf["n_cells_total"].dropna().unique()
        panel_title = (
            f"{sample}\n{title}\nTotal: {int(total_unique[0]):,} cells — N removed:"
            if len(total_unique) == 1 else f"{sample}\n{title}"
        )
        _, _, im = PlotHeatmap(
            sdf, x=x, y=y,
            color_col="pct_removed",
            annotation_col="n_removed",
            title=panel_title,
            cmap="RdYlGn_r",
            ax=ax,
            show=False,
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("% removed")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return fig

################################################################################

def PlotQCScatterFullZoom(
    adata,
    x="total_counts",
    y="n_genes_by_counts",
    color="pct_counts_mt",
    xlim_zoom=(0, 8_000),
    ylim_zoom=(0, 1_000),
    figsize=(12, 5),
    title_full="Full view",
    title_zoom="Zoomed view",
    cmap="viridis",
    point_size=6,
    alpha=0.8,
    save_path=None,
    show=True,
):
    """
    Plot two QC scatter plots side by side with one shared colorbar.

    The first panel shows the full distribution.
    The second panel shows the same data with adjusted x/y limits.
    A single colorbar is placed on the right side of the figure.
    """

    # Extract plotting values from adata.obs
    x_values = adata.obs[x]
    y_values = adata.obs[y]
    color_values = adata.obs[color]

    # Create side-by-side plots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        constrained_layout=True,
    )

    # Use the same color scale for both plots
    vmin = color_values.min()
    vmax = color_values.max()

    # Full view
    scatter_full = axes[0].scatter(
        x_values,
        y_values,
        c=color_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )
    axes[0].set_xlabel(x)
    axes[0].set_ylabel(y)
    axes[0].set_title(title_full)

    # Zoomed view
    scatter_zoom = axes[1].scatter(
        x_values,
        y_values,
        c=color_values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )
    axes[1].set_xlabel(x)
    axes[1].set_ylabel(y)
    axes[1].set_xlim(*xlim_zoom)
    axes[1].set_ylim(*ylim_zoom)
    axes[1].set_title(title_zoom)

    # Add one shared colorbar on the right
    cbar = fig.colorbar(
        scatter_zoom,
        ax=axes,
        location="right",
        shrink=0.85,
        pad=0.02,
    )
    cbar.set_label(color)

    # Save figure if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    return fig, axes

################################################################################
# ── 5. QC Summary Table ───────────────────────────────────────────────────────

def QcSummaryTable(
    adatas_before: dict,
    adatas_after: dict,
    *,
    cutoffs: dict | None = None,
) -> pd.DataFrame:
    """Build a per-sample QC summary table comparing before and after filtering.

    Parameters
    ----------
    adatas_before : dict of {sample_name: AnnData} before QC cell filtering
                    (typically adatas_dic["scroublet"]).
    adatas_after  : dict of {sample_name: AnnData} after QC cell filtering
                    (typically adatas_dic["filtered"]).
    cutoffs       : threshold dict passed to :func:`FilterQcCells`, used to
                    populate the "Cutoff …" rows.  ``None`` skips those rows.

    Returns
    -------
    pd.DataFrame  Rows = metrics, columns = sample names.
    """
    if set(adatas_before) != set(adatas_after):
        raise ValueError("adatas_before and adatas_after must have the same sample keys.")

    samples = list(adatas_before.keys())
    rows: dict[str, dict] = {}   # {metric_label: {sample: value}}

    def _med(adata, col):
        """Return median of adata.obs[col], or NaN if column is missing."""
        return round(float(adata.obs[col].median()), 2) if col in adata.obs else float("nan")

    def _var_med(adata, col):
        """Return median of adata.var[col], or NaN if column is missing."""
        return round(float(adata.var[col].median()), 2) if col in adata.var else float("nan")

    for sample in samples:
        ab = adatas_before[sample]
        af = adatas_after[sample]

        # Number of doublets: prefer the value stored by FilterDoublets in uns;
        # fall back to reading the is_doublet column if it still exists in adatas_before.
        n_doublets = ab.uns.get("scrublet", {}).get("n_doublets_removed", float("nan"))
        if np.isnan(n_doublets) and "is_doublet" in ab.obs:
            n_doublets = int(ab.obs["is_doublet"].sum())

        # Ambient RNA contamination fraction estimated by SoupX
        rho = ab.uns.get("qc_params", {}).get("soupx", {}).get("rho", float("nan"))

        col_vals = {
            "Initial Cell Counts":                        ab.n_obs,
            "Final Cell Counts":                          af.n_obs,
            "Initial Gene Counts":                        ab.n_vars,
            "Final Gene Counts":                          af.n_vars,
            "Initial N Cell by Genes (Median)":           _var_med(ab, "n_cells_by_counts"),
            "Final N Cell by Genes (Median)":             _var_med(af, "n_cells_by_counts"),
            "Initial N Genes by Cells (Median)":          _med(ab, "n_genes_by_counts"),
            "Final N Genes by Cells (Median)":            _med(af, "n_genes_by_counts"),
            "Initial Pct counts in top 20 genes (Median)": _med(ab, "pct_counts_in_top_20_genes"),
            "Final Pct counts in top 20 genes (Median)":   _med(af, "pct_counts_in_top_20_genes"),
            "Initial Pct counts MT (Median)":             _med(ab, "pct_counts_mt"),
            "Final Pct counts MT (Median)":               _med(af, "pct_counts_mt"),
            "Initial Pct counts Hb (Median)":             _med(ab, "pct_counts_hb"),
            "Final Pct counts Hb (Median)":               _med(af, "pct_counts_hb"),
            "Initial Pct counts Ribo (Median)":           _med(ab, "pct_counts_ribo"),
            "Final Pct counts Ribo (Median)":             _med(af, "pct_counts_ribo"),
            "N Doublets":                                 n_doublets,
            "Pct Ambient RNA contamination":              rho,
        }

        # Add one row per cutoff if provided
        if cutoffs:
            for metric, spec in cutoffs.items():
                # Absolute cutoffs have "min"/"max" keys; MADs have "mads"
                if "min" in spec:
                    col_vals[f"Cutoff {metric} (min)"] = spec["min"]
                if "max" in spec:
                    col_vals[f"Cutoff {metric} (max)"] = spec["max"]
                if "mads" in spec:
                    col_vals[f"Cutoff {metric} (mads)"] = spec["mads"]

        for metric, val in col_vals.items():
            rows.setdefault(metric, {})[sample] = val

    return pd.DataFrame(rows, index=samples).T


################################################################################

def PlotQcBoxplots(
    adatas_before: dict,
    adatas_after: dict,
    *,
    metric: str = "n_genes_by_counts",
    labels: tuple = ("before filtering", "after filtering"),
    show: bool = True,
    save_path: str | None = None,
):
    """Paired boxplot comparing a QC metric before and after filtering.

    Parameters
    ----------
    adatas_before : dict of {sample_name: AnnData} before QC filtering.
    adatas_after  : dict of {sample_name: AnnData} after QC filtering.
    metric : ``adata.obs`` column to compare across both groups.
    labels : legend labels for the two groups.
    show : display the figure.
    save_path : file path to save the figure.
    """
    from sctools.plots import plot_metric_pairs

    # Both dicts must have the same samples in the same order
    if set(adatas_before) != set(adatas_after):
        raise ValueError("adatas_before and adatas_after must have the same sample keys.")

    sample_names = list(adatas_before.keys())

    return plot_metric_pairs(
        adatas_before=[adatas_before[s] for s in sample_names],
        adatas_after=[adatas_after[s] for s in sample_names],
        sample_names=sample_names,
        metric=metric,
        labels=labels,
        show_plot=show,
        save_path=save_path,
    )
