"""
sctools.annotation
===================

Marker-based cell-type annotation helpers: module scores and AUCell scores,
plus UMAP plotting for each.

First version: ports, as-is, the module-score and AUCell scoring/plotting
functions used in the Ffar2 annotation notebooks
(`06sx.scNewFfar2_Annotation.ipynb` / `06sx.scNewFfar2_Annotation_ALRA.ipynb`
define these four functions identically). Only these four are ported for
now; the manual/second-level annotation helpers from the same notebooks
(`ImportMarkers`, `RankGenesByCluster`, `Anot2_with_Markers`,
`ScoreTableFromDics`) are not ported yet.

Module scores (Scanpy `sc.tl.score_genes`):
    CalculateModuleScores()
    PlotModuleScoresUMAPs()

AUCell scores (requires `decoupler`):
    CalculateAUCellWithDecoupler()
    PlotAUCellUMAPs()
"""

from __future__ import annotations

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


# ── Module scores (sc.tl.score_genes) ───────────────────────────────────────

def CalculateModuleScores(
    adata: AnnData,
    marker_genes_dic: dict[str, list[str]],
    layer: str = "QC_filtered_log1p",
    min_genes: int = 5,
) -> AnnData:
    """
    Calculate a module score per marker gene set, normalized by sqrt(n_genes).

    For each entry in `marker_genes_dic`, runs `sc.tl.score_genes` on the
    genes that are present in `adata.var_names` (case-insensitive match),
    then divides the raw score by `sqrt(n_genes_found)` so sets of
    different sizes are more comparable.

    Parameters
    ----------
    adata
        AnnData object. Scores are written to `adata.obs[f"score_{name}"]`.
    marker_genes_dic
        Mapping of {signature/cell-type name: list of gene symbols}.
    layer
        Layer with normalized/log-transformed expression, passed to
        `sc.tl.score_genes(..., layer=layer)`.
    min_genes
        Kept from the source notebook function; not currently used to
        filter or warn (see Notes).

    Returns
    -------
    AnnData
        Same object, with one `score_<name>` column added to `adata.obs`
        per key in `marker_genes_dic`.

    Notes
    -----
    Ported as-is from the notebook function of the same name. Gene sets
    with zero valid genes get a score of 0.0 instead of being skipped, so
    every key in `marker_genes_dic` always produces a column. The
    low-gene-count warning is commented out in the source notebook; kept
    commented out here too (not silently re-enabled), since this first
    version is meant to match notebook behavior exactly.
    """
    var_map = {g.upper(): g for g in adata.var_names}

    for name, genes in marker_genes_dic.items():
        genes_ok = [var_map[g.upper()] for g in genes if g.upper() in var_map]

        # if len(genes_ok) < min_genes:
        #     print(f"Warning: '{name}' has only {len(genes_ok)} valid genes -- result may be unstable.")

        if not genes_ok:
            adata.obs[f"score_{name}"] = 0.0
            continue

        sc.tl.score_genes(
            adata,
            gene_list=genes_ok,
            score_name=f"score_{name}_raw",
            use_raw=False,
            layer=layer,
        )

        n_genes = len(genes_ok)
        adata.obs[f"score_{name}"] = adata.obs[f"score_{name}_raw"] / np.sqrt(n_genes)
        adata.obs.drop(columns=[f"score_{name}_raw"], inplace=True)

    return adata


def PlotModuleScoresUMAPs(
    adata: AnnData,
    score_names: list[str] | None = None,
    ncols: int = 4,
    cmap: str = "bwr",
    share_scale: bool = False,
    percentile: float = 99,
    figsize: tuple[float, float] = (4, 4),
    show_colorbar: bool = True,
):
    """
    Plot one UMAP per module score, on a diverging color scale centered at 0.

    Parameters
    ----------
    adata
        AnnData with module scores in `adata.obs` (see `CalculateModuleScores`).
    score_names
        Columns to plot. Defaults to every `adata.obs` column starting with
        `"score_"`.
    ncols
        Number of panels per row.
    cmap
        Diverging colormap: blue (negative) / white (0) / red (positive).
    share_scale
        If True, use the same symmetric-around-0 color scale for every
        panel. If False, each panel gets its own scale.
    percentile
        Percentile of `|score|` used as the color-scale limit (robust to
        outliers).
    figsize
        Size of each individual subplot.
    show_colorbar
        Whether to draw a colorbar next to each panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if score_names is None:
        score_names = [c for c in adata.obs.columns if c.startswith("score_")]

    if share_scale:
        vals = np.concatenate([adata.obs[s].values for s in score_names])
        vmax = np.percentile(np.abs(vals), percentile)
        vmins = [-vmax] * len(score_names)
        vmaxs = [vmax] * len(score_names)
    else:
        vmaxs = [np.percentile(np.abs(adata.obs[s].values), percentile) for s in score_names]
        vmins = [-v for v in vmaxs]

    n = len(score_names)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, s in enumerate(score_names):
        sc.pl.umap(
            adata,
            color=s,
            color_map=cmap,
            vmin=vmins[i],
            vmax=vmaxs[i],
            ax=axes[i],
            show=False,
            frameon=False,
            colorbar_loc=("right" if show_colorbar else "none"),
        )
        axes[i].set_title(s.replace("score_", ""), fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


# ── AUCell scores (decoupler) ────────────────────────────────────────────────

def _BuildDecouplerNet(adata: AnnData, marker_genes_dic: dict[str, list[str]]) -> pd.DataFrame:
    """Build a decoupler `net` table (source, target) restricted to genes present in `adata`."""
    var_map = {g.upper(): g for g in adata.var_names}
    rows = []
    for src, genes in marker_genes_dic.items():
        for g in genes:
            up = g.upper()
            if up in var_map:
                rows.append((src, var_map[up]))
    return pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates()


def CalculateAUCellWithDecoupler(
    adata: AnnData,
    marker_genes_dic: dict[str, list[str]],
    layer: str = "QC_filtered_log1p",
    n_up: int | None = None,
    tmin: int = 1,
    min_genes: int = 5,
    prefix: str = "aucell_",
) -> AnnData:
    """
    Calculate AUCell scores per marker gene set via `decoupler`, normalized
    by sqrt(n_genes).

    Builds a `decoupler` "net" table (source=signature name, target=gene)
    from `marker_genes_dic` restricted to genes present in
    `adata.var_names`, runs `decoupler.mt.aucell`, then divides each raw
    AUCell score by `sqrt(n_genes_found)`.

    Parameters
    ----------
    adata
        AnnData object. Scores are written to `adata.obs[f"{prefix}{name}"]`.
    marker_genes_dic
        Mapping of {signature/cell-type name: list of gene symbols}.
    layer
        Layer with normalized/log-transformed expression, passed to
        `decoupler.mt.aucell(..., layer=layer)`.
    n_up
        Number of top-ranked genes per cell considered by AUCell. `None`
        uses decoupler's default.
    tmin
        Minimum number of targets (genes) required per source (signature)
        for decoupler to score it.
    min_genes
        Kept from the source notebook function; not currently used to
        filter or warn (see Notes).
    prefix
        Prefix for the `adata.obs` output columns.

    Returns
    -------
    AnnData
        Same object, with one `<prefix><name>` column added to `adata.obs`
        per key in `marker_genes_dic` (0.0 for sets that decoupler did not
        score, e.g. below `tmin`).

    Notes
    -----
    Ported as-is from the notebook function of the same name. Requires the
    `decoupler` package, imported lazily here so importing
    `sctools.annotation` does not require it (same pattern as the R/rpy2
    import in `sctools.alra`). The low-gene-count warning is commented out
    in the source notebook; kept commented out here too, for the same
    reason as in `CalculateModuleScores`.
    """
    import decoupler as dc

    net = _BuildDecouplerNet(adata, marker_genes_dic)
    dc.mt.aucell(data=adata, net=net, tmin=tmin, layer=layer, raw=False, n_up=n_up, verbose=False)

    sc_adata = dc.pp.get_obsm(adata, key="score_aucell")
    df = pd.DataFrame(sc_adata.X, index=adata.obs_names, columns=sc_adata.var_names)

    for name, genes in marker_genes_dic.items():
        valid_genes = [g for g in genes if g.upper() in (x.upper() for x in adata.var_names)]
        n_genes = len(valid_genes)

        # if n_genes < min_genes:
        #     print(f"Warning: '{name}' has only {n_genes} valid genes -- AUCell may be unstable.")

        if name in df.columns:
            adata.obs[f"{prefix}{name}"] = df[name] / np.sqrt(max(n_genes, 1))
        else:
            adata.obs[f"{prefix}{name}"] = 0.0

    return adata


def PlotAUCellUMAPs(
    adata: AnnData,
    cols: list[str] | None = None,
    ncols: int = 4,
    cmap: str = "viridis",
    percentile: float = 99,
    share_scale: bool = False,
    figsize: tuple[float, float] = (4, 4),
):
    """
    Plot one UMAP per AUCell score, on a sequential color scale starting at 0.

    Parameters
    ----------
    adata
        AnnData with AUCell scores in `adata.obs` (see
        `CalculateAUCellWithDecoupler`).
    cols
        Columns to plot. Defaults to every `adata.obs` column starting with
        `"aucell_"`.
    ncols
        Number of panels per row.
    cmap
        Sequential colormap for the AUCell scores.
    percentile
        Percentile used as the upper color-scale limit (robust to
        outliers). Lower limit is always 0.
    share_scale
        If True, use the same 0-to-vmax color scale for every panel. If
        False, each panel gets its own scale.
    figsize
        Size of each individual subplot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if cols is None:
        cols = [c for c in adata.obs.columns if c.startswith("aucell_")]

    if share_scale:
        vals = np.concatenate([adata.obs[c].values for c in cols]).astype(float)
        vals = vals[np.isfinite(vals)]
        vmax = np.percentile(vals, percentile)
        vmins = [0.0] * len(cols)
        vmaxs = [vmax] * len(cols)
    else:
        vmins = [0.0] * len(cols)
        vmaxs = []
        for c in cols:
            v = adata.obs[c].astype(float).values
            v = v[np.isfinite(v)]
            vmaxs.append(np.percentile(v, percentile))

    n = len(cols)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, c in enumerate(cols):
        sc.pl.umap(
            adata,
            color=c,
            vmin=vmins[i],
            vmax=vmaxs[i],
            color_map=cmap,
            ax=axes[i],
            show=False,
            frameon=False,
            colorbar_loc="right",
        )
        axes[i].set_title(c.replace("aucell_", ""), fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
