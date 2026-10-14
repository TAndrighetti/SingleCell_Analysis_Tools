"""
sctools.annotation
===================

Marker-based cell-type annotation helpers: marker matching, top DEGs per
cluster, module scores and AUCell scores, plus plots for each.

Ported from the annotation notebooks (`06sx.scNewFfar2_Annotation.ipynb`,
`06sx.scNewFfar2_Annotation_ALRA.ipynb`, `06.scNeuSheng_Annotation.ipynb`).
The manual/second-level helpers from the same notebooks (`ImportMarkers`,
`Anota1`, `Anot2_with_Markers`, `InterDics`, `ScoreTableFromDics`) are not
ported yet.

Typical usage order
-------------------
Markers and top DEGs per cluster:
    MarkerInData()
    RankGenesByCluster()
    MeanExpressionByCluster()   (any gene list or marker dict)
    PlotTopDegsHeatmap()

Module scores (Scanpy `sc.tl.score_genes`):
    CalculateModuleScores()
    PlotModuleScoresUMAPs()

AUCell scores (requires `decoupler`):
    CalculateAUCellWithDecoupler()
    PlotAUCellUMAPs()

Cell cycle (requires `scib` for the gene lists):
    ScoreCellCycle()

Label suggestion per cluster, with confidence:
    AssignLabelsFromScores()

Final annotation (map clusters to labels, UMAPs + marker dotplot):
    PlotFinalAnnotation()

Quick start
-----------
>>> from sctools.annotation import (
...     MarkerInData, RankGenesByCluster, PlotTopDegsHeatmap,
...     CalculateModuleScores, PlotModuleScoresUMAPs,
...     CalculateAUCellWithDecoupler, PlotAUCellUMAPs,
...     ScoreCellCycle, AssignLabelsFromScores,
... )
>>> cluster_genes, scaled_df = RankGenesByCluster(
...     adata, groupby="leiden_res1", filter_mt_ribo_hb=True, specific_markers=True,
... )
>>> PlotTopDegsHeatmap(scaled_df, title="Top 10 DEGs per cluster")
>>> markers = MarkerInData({"Goblet": ["Muc2", "Tff3"]}, adata)
>>> adata = CalculateModuleScores(adata, markers)
>>> PlotModuleScoresUMAPs(adata)
>>> adata = CalculateAUCellWithDecoupler(adata, markers)
>>> PlotAUCellUMAPs(adata)
>>> adata = ScoreCellCycle(adata, organism="mouse")
>>> cluster_scores, summary = AssignLabelsFromScores(adata, groupby="leiden_res1", prefix="score_")
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData

# Public API: names exported by `from sctools.annotation import *`
__all__ = [
    "MarkerInData",
    "RankGenesByCluster",
    "MeanExpressionByCluster",
    "PlotTopDegsHeatmap",
    "PlotTopDEGsHeatmap",
    "CalculateModuleScores",
    "PlotModuleScoresUMAPs",
    "CalculateAUCellWithDecoupler",
    "PlotAUCellUMAPs",
    "ScoreCellCycle",
    "AssignLabelsFromScores",
    "PlotFinalAnnotation",
]

logger = logging.getLogger(__name__)

# Prefix patterns for mitochondrial, ribosomal and hemoglobin genes (mouse and
# human symbols, matched case-insensitively). Prefix match, not substring, so
# genes like Mtor, Serpinb1a, Trp53 are kept. The lookaheads keep the S6
# kinases (Rps6ka*/Rps6kb*) and Hbegf, which are not ribosomal/hemoglobin genes.
# Same rules as the patterns in sctools.qc.
_MT_PATTERN = r"^mt-"
_RIBO_PATTERN = r"^rp[sl](?!6k)"
_HB_PATTERN = r"^hb(?!egf)[abdegmqz]"
_MT_RIBO_HB_PATTERN = "|".join((_MT_PATTERN, _RIBO_PATTERN, _HB_PATTERN))


# ── Shared helpers ────────────────────────────────────────────────────────────

def _CheckLayer(adata: AnnData, layer: str) -> None:
    if layer not in adata.layers:
        raise KeyError(
            f"Layer '{layer}' not found in adata.layers. Available: {list(adata.layers.keys())}"
        )


def _VarMap(var_names) -> dict[str, str]:
    """Map upper-case gene symbol -> symbol as spelled in `var_names`."""
    return {g.upper(): g for g in var_names}


def _MatchGenes(genes, var_map: dict[str, str]) -> list[str]:
    """Genes found in `var_map` (case-insensitive), spelled as in the data, unique, in input order."""
    matched = (var_map.get(str(g).upper()) for g in genes)
    return list(dict.fromkeys(g for g in matched if g is not None))


############################################
# ── Markers ──────────────────────────────────────────────────────────────────

def MarkerInData(marker_genes: dict[str, list[str]], adata: AnnData) -> dict[str, list[str]]:
    """
    Keep only the marker genes present in `adata.var_names`.

    Matching is case-insensitive and returns each gene spelled as in the
    data, so `"H2-ab1"` (e.g. after `str.capitalize()`) resolves to
    `"H2-Ab1"`. Duplicates are dropped and input order is kept.

    Parameters
    ----------
    marker_genes
        Mapping of {cell type: list of gene symbols}.
    adata
        AnnData whose `var_names` are used as the gene universe.

    Returns
    -------
    dict
        Same keys, with only the genes found in the data. Sets with no gene
        found are kept as empty lists.

    Notes
    -----
    Domain note: case-insensitive matching is not ortholog mapping. A human
    symbol like "CD8A" matches mouse "Cd8a" only because the names happen
    to agree; many human/mouse orthologs have different symbols.
    """
    var_map = _VarMap(adata.var_names)
    return {ct: _MatchGenes(markers, var_map) for ct, markers in marker_genes.items()}


# ── Top DEGs per cluster ─────────────────────────────────────────────────────

def RankGenesByCluster(
    adata: AnnData,
    groupby: str = "leiden_res1",
    layer: str = "QC_filtered_log1p",
    n_genes: int = 10,
    filter_mt_ribo_hb: bool = False,
    key_added: str = "rank_genes_groups",
    specific_markers: bool = False,
    min_in_group_fraction: float = 0.25,
    max_out_group_fraction: float = 0.5,
    min_log2_fold_change: float = 1.0,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """
    Top marker genes per cluster (Wilcoxon) and their z-scored mean expression.

    Runs `sc.tl.rank_genes_groups(method="wilcoxon")`, takes the top
    `n_genes` per group, then computes the mean expression of the union of
    those genes per group and z-scores each gene across groups (via
    `MeanExpressionByCluster`).

    Parameters
    ----------
    adata
        AnnData object. Results are written to `adata.uns[key_added]`.
    groupby
        `adata.obs` column with the cluster labels.
    layer
        Layer with log-normalized expression (log1p), used both for the
        ranking and for the mean expression.
    n_genes
        Number of top genes kept per group.
    filter_mt_ribo_hb
        If True, drop mitochondrial (`mt-`), ribosomal (`Rps`/`Rpl`) and
        hemoglobin (`Hba`/`Hbb`/...) genes before taking the top
        `n_genes`. Prefix match, so Mtor, Serpinb1a, Rps6ka1, Hbegf are kept.
    key_added
        `adata.uns` key for the `rank_genes_groups` results.
    specific_markers
        If True, keep only genes that pass `sc.tl.filter_rank_genes_groups`
        with the three thresholds below (filtered results in
        `adata.uns[f"{key_added}_filtered"]`, filtered-out genes stored as
        ""). Removes genes that are high in several clusters.
    min_in_group_fraction
        Minimum fraction of cells in the group expressing the gene.
    max_out_group_fraction
        Maximum fraction of cells outside the group expressing the gene.
    min_log2_fold_change
        Minimum log2 fold change (1 = 2x), from `rank_genes_groups`.

    Returns
    -------
    cluster_genes : dict
        {group: list of top genes}.
    scaled_df : pandas.DataFrame
        Genes x groups, mean expression z-scored per gene across groups.

    Notes
    -----
    Domain note: Wilcoxon on single cells treats each cell as an independent
    replicate. That is fine to rank marker genes, but its p-values are not
    valid for differential expression between conditions (use pseudobulk,
    see `sctools.degs`).

    Domain note: `specific_markers` follows the marker filtering of the
    sc-best-practices book. Fractions and log2FC come from the same
    `layer` used for the ranking (`pts=True`). Neutrophils and other cells
    with low RNA content express many genes in few cells, so
    `min_in_group_fraction=0.25` can drop lowly expressed transcription
    factors. Lower it if a group returns too few genes.

    A group can return fewer than `n_genes` genes when the filters remove
    too many. A warning is logged when that happens.
    """
    _CheckLayer(adata, layer)
    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")

    sc.tl.rank_genes_groups(
        adata, groupby=groupby, layer=layer, method="wilcoxon", key_added=key_added,
        pts=specific_markers,
    )
    names_key = key_added
    if specific_markers:
        # Uses pts/pts_rest and logfoldchanges stored by rank_genes_groups (same layer)
        names_key = f"{key_added}_filtered"
        sc.tl.filter_rank_genes_groups(
            adata, key=key_added, groupby=groupby, key_added=names_key,
            min_in_group_fraction=min_in_group_fraction,
            max_out_group_fraction=max_out_group_fraction,
            min_fold_change=min_log2_fold_change,
        )

    # Top genes per group, optionally without mt/ribo/hb genes
    names = pd.DataFrame(adata.uns[names_key]["names"])
    if specific_markers:
        # scanpy marks filtered-out genes as NaN, which write_h5ad cannot store
        adata.uns[names_key]["names"] = names.fillna("").to_records(index=False)
    cluster_genes = {}
    for group in names.columns:
        genes = pd.Index(names[group]).dropna()  # filtered-out genes are NaN
        if filter_mt_ribo_hb:
            genes = genes[~genes.str.contains(_MT_RIBO_HB_PATTERN, case=False, regex=True)]
        if len(genes) < n_genes:
            logger.warning("Group '%s': only %d genes left (n_genes=%d).", group, len(genes), n_genes)
        cluster_genes[group] = list(genes[:n_genes])

    # Union of top genes (no duplicates, order kept), z-scored mean per group
    top_genes = list(dict.fromkeys(g for genes in cluster_genes.values() for g in genes))
    scaled_df = MeanExpressionByCluster(adata, top_genes, groupby=groupby, layer=layer, zscore=True)

    return cluster_genes, scaled_df


def MeanExpressionByCluster(
    adata: AnnData,
    genes: list[str] | dict[str, list[str]],
    groupby: str = "leiden_res1",
    layer: str = "QC_filtered_log1p",
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Mean expression of a gene list per cluster, optionally z-scored per gene.

    Uses the official Scanpy functions: `sc.get.aggregate(func="mean")` for
    the mean per group (works on sparse data) and `sc.pp.scale` for the
    z-score of each gene across groups.

    Parameters
    ----------
    adata
        AnnData object (not modified).
    genes
        List of genes, or a marker dict {cell type: genes} (e.g. the
        `cluster_genes` output of `RankGenesByCluster`, or known markers).
        A dict is flattened in order, duplicates are dropped. Genes are
        matched case-insensitively; genes not in the data are skipped with
        a warning.
    groupby
        `adata.obs` column with the cluster labels.
    layer
        Layer with log-normalized expression (log1p).
    zscore
        If True, z-score each gene across groups (mean 0, sd 1 over
        groups). A gene with the same mean in every group gets 0.

    Returns
    -------
    pandas.DataFrame
        Genes x groups, groups in the category order of `groupby`. Ready
        for `PlotTopDegsHeatmap`.
    """
    # Stop early with a clear error if the layer does not exist
    _CheckLayer(adata, layer)
    
    # Stop early if the cluster column does not exist in adata.obs
    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")

    # A marker dict becomes one flat gene list, keeping the dict order
    if isinstance(genes, dict):
        genes = [g for gene_list in genes.values() for g in gene_list]

    # Keep only genes present in the data (case-insensitive), spelled as in
    # var_names, without duplicates, in input order
    matched = _MatchGenes(genes, _VarMap(adata.var_names))

    # Count unique input genes (ignoring case) that were not found
    n_missing = len(dict.fromkeys(str(g).upper() for g in genes)) - len(matched)
    # Warn, but go on, when some genes are missing
    if n_missing:
        logger.warning("%d genes not found in adata.var_names were skipped.", n_missing)
    # Stop if no gene is left, there is nothing to average
    if not matched:
        raise ValueError("None of the genes were found in adata.var_names.")

    # Mean of each gene per cluster on the chosen layer. Returns a small
    # AnnData: one row per cluster, one column per gene, means in layers["mean"]
    agg = sc.get.aggregate(adata[:, matched], by=groupby, func="mean", layer=layer)
    # Z-score each gene across clusters (mean 0, sd 1 over clusters), in place
    if zscore:
        sc.pp.scale(agg, layer="mean")
    # Clusters x genes table, transposed to genes x clusters for the heatmap
    return agg.to_df(layer="mean").T


def PlotTopDegsHeatmap(
    scaled_df: pd.DataFrame,
    title: str = "",
    cmap: str = "magma",
    figsize: tuple[float, float] = (10, 8),
    ax=None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple:
    """
    Heatmap of z-scored mean expression of the top genes per cluster.

    Parameters
    ----------
    scaled_df
        Genes x clusters table, e.g. the second output of `RankGenesByCluster`.
    title
        Plot title.
    cmap
        Colormap. Kept as "magma" to match the original notebook figures.
        The values are z-scores centered at 0 (diverging data), so a
        diverging colormap like "RdBu_r" makes 0 look neutral and is more
        coherent with the data.
    figsize
        Figure size. Ignored when `ax` is provided.
    ax
        Existing Axes to draw on. If None, a new figure is created.
    show
        Display the figure. Ignored when `ax` is provided.
    save_path
        Path to save the figure. Ignored when `ax` is provided.

    Returns
    -------
    fig, ax
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        scaled_df,
        cmap=cmap,
        center=0,
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        square=False,
        linewidths=0,
        rasterized=True,  # continuous rendering, no pixel lines between cells
        ax=ax,
    )

    # Remove borders and residual grid
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Genes")

    if own_fig:
        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return fig, ax


# Backward-compatible alias (old notebook name)
PlotTopDEGsHeatmap = PlotTopDegsHeatmap


############################################
# ── Module scores (sc.tl.score_genes) ───────────────────────────────────────

def CalculateModuleScores(
    adata: AnnData,
    marker_genes_dic: dict[str, list[str]],
    layer: str = "QC_filtered_log1p",
    min_genes: int = 5,
    normalize_by_sqrt_n: bool = False,
    prefix: str = "score_",
) -> AnnData:
    """
    Calculate a module score (`sc.tl.score_genes`) per marker gene set.

    For each entry in `marker_genes_dic`, runs `sc.tl.score_genes` on the
    genes present in `adata.var_names` (case-insensitive match, see
    `MarkerInData`).

    Parameters
    ----------
    adata
        AnnData object. Scores are written to `adata.obs[f"{prefix}{name}"]`.
    marker_genes_dic
        Mapping of {signature/cell-type name: list of gene symbols}.
    layer
        Layer with normalized/log-transformed expression.
    min_genes
        A warning is logged for sets with fewer valid genes than this.
    normalize_by_sqrt_n
        If True, divide each score by sqrt(n_genes_found), as in the
        original notebook function. Off by default, see Notes.
    prefix
        Prefix for the `adata.obs` output columns.

    Returns
    -------
    AnnData
        Same object (modified in place), with one `<prefix><name>` column in
        `adata.obs` per key in `marker_genes_dic`. Sets with no valid gene
        get NaN. Parameters are recorded in
        `adata.uns["annotation_params"]["module_scores"]`.

    Notes
    -----
    Domain note: `score_genes` is the mean expression of the set minus the
    mean of a matched control set, so it is already on a per-gene scale and
    does not grow with set size. Its noise goes down roughly as
    1/sqrt(n_genes). Dividing again by sqrt(n_genes) shrinks the signal of
    large sets, so when comparing signatures (e.g. which cell type scores
    highest) small sets are favored. Use `normalize_by_sqrt_n=True` only to
    reproduce older results.
    """
    _CheckLayer(adata, layer)
    var_map = _VarMap(adata.var_names)
    n_genes_used = []

    for name, genes in marker_genes_dic.items():
        col = f"{prefix}{name}"
        genes_ok = _MatchGenes(genes, var_map)
        n_genes = len(genes_ok)
        n_genes_used.append(n_genes)

        if n_genes == 0:
            logger.warning("'%s': no valid genes in the data, score set to NaN.", name)
            adata.obs[col] = np.nan
            continue
        if n_genes < min_genes:
            logger.warning("'%s' has only %d valid genes, result may be unstable.", name, n_genes)

        sc.tl.score_genes(adata, gene_list=genes_ok, score_name=col, use_raw=False, layer=layer)

        if normalize_by_sqrt_n:
            adata.obs[col] = adata.obs[col] / np.sqrt(n_genes)

    adata.uns.setdefault("annotation_params", {})["module_scores"] = {
        "layer": layer,
        "prefix": prefix,
        "min_genes": min_genes,
        "normalize_by_sqrt_n": normalize_by_sqrt_n,
        "signatures": list(marker_genes_dic.keys()),
        "n_genes_used": n_genes_used,
    }
    return adata


# ── AUCell scores (decoupler) ────────────────────────────────────────────────

def _BuildDecouplerNet(adata: AnnData, marker_genes_dic: dict[str, list[str]]) -> pd.DataFrame:
    """Build a decoupler `net` table (source, target) restricted to genes present in `adata`."""
    var_map = _VarMap(adata.var_names)
    rows = [
        (src, g)
        for src, genes in marker_genes_dic.items()
        for g in _MatchGenes(genes, var_map)
    ]
    return pd.DataFrame(rows, columns=["source", "target"])


def CalculateAUCellWithDecoupler(
    adata: AnnData,
    marker_genes_dic: dict[str, list[str]],
    layer: str = "QC_filtered_log1p",
    n_up: int | None = None,
    tmin: int = 1,
    min_genes: int = 5,
    prefix: str = "aucell_",
    normalize_by_sqrt_n: bool = False,
) -> AnnData:
    """
    Calculate AUCell scores per marker gene set via `decoupler`.

    Builds a `decoupler` "net" table (source=signature name, target=gene)
    from `marker_genes_dic` restricted to genes present in `adata.var_names`
    (case-insensitive match), then runs `decoupler.mt.aucell`.

    Parameters
    ----------
    adata
        AnnData object. Scores are written to `adata.obs[f"{prefix}{name}"]`.
    marker_genes_dic
        Mapping of {signature/cell-type name: list of gene symbols}.
    layer
        Expression layer passed to `decoupler.mt.aucell(..., layer=layer)`.
    n_up
        Number of top-ranked genes per cell considered by AUCell. `None`
        uses decoupler's default (top 5% of genes).
    tmin
        Minimum number of genes per set for decoupler to score it.
    min_genes
        A warning is logged for sets with fewer valid genes than this.
    prefix
        Prefix for the `adata.obs` output columns.
    normalize_by_sqrt_n
        If True, divide each score by sqrt(n_genes_found), as in the
        original notebook function. Off by default, see Notes.

    Returns
    -------
    AnnData
        Same object (modified in place), with one `<prefix><name>` column in
        `adata.obs` per key in `marker_genes_dic`. Sets not scored (no valid
        gene, or fewer than `tmin`) get NaN. decoupler also leaves the raw
        scores in `adata.obsm["score_aucell"]`. Parameters are recorded in
        `adata.uns["annotation_params"]["aucell"]`.

    Notes
    -----
    Domain note: decoupler already divides each AUC by the maximum AUC
    possible for that set, so AUCell is in [0, 1] and comparable between
    sets of different sizes. Dividing again by sqrt(n_genes) breaks the
    [0, 1] scale and penalizes large sets. Use `normalize_by_sqrt_n=True`
    only to reproduce older results.

    Domain note: AUCell ranks genes inside each cell, so library-size
    normalization and log1p do not change the result (they keep the
    ranking). Imputed or scaled layers do change it.

    Requires the `decoupler` package, imported lazily here so importing
    `sctools.annotation` does not require it.
    """
    import decoupler as dc

    _CheckLayer(adata, layer)
    net = _BuildDecouplerNet(adata, marker_genes_dic)
    n_genes_by_set = net.groupby("source").size()

    scores = pd.DataFrame(index=adata.obs_names)
    if (n_genes_by_set >= tmin).any():
        dc.mt.aucell(data=adata, net=net, tmin=tmin, layer=layer, raw=False, n_up=n_up, verbose=False)
        sc_adata = dc.pp.get_obsm(adata, key="score_aucell")
        scores = pd.DataFrame(sc_adata.X, index=adata.obs_names, columns=sc_adata.var_names)

    n_genes_used = []
    for name in marker_genes_dic:
        col = f"{prefix}{name}"
        n_genes = int(n_genes_by_set.get(name, 0))
        n_genes_used.append(n_genes)

        if name not in scores.columns:
            logger.warning(
                "'%s': %d valid genes (tmin=%d), not scored by decoupler, set to NaN.", name, n_genes, tmin
            )
            adata.obs[col] = np.nan
            continue
        if n_genes < min_genes:
            logger.warning("'%s' has only %d valid genes, AUCell may be unstable.", name, n_genes)

        adata.obs[col] = scores[name] / np.sqrt(n_genes) if normalize_by_sqrt_n else scores[name]

    adata.uns.setdefault("annotation_params", {})["aucell"] = {
        "layer": layer,
        "prefix": prefix,
        "n_up": -1 if n_up is None else n_up,  # -1 = decoupler default (h5ad cannot store None)
        "tmin": tmin,
        "min_genes": min_genes,
        "normalize_by_sqrt_n": normalize_by_sqrt_n,
        "signatures": list(marker_genes_dic.keys()),
        "n_genes_used": n_genes_used,
    }
    return adata


# ── Cell cycle (scanpy score_genes_cell_cycle) ──────────────────────────────

def ScoreCellCycle(
    adata: AnnData,
    organism: str = "mouse",
    layer: str = "QC_filtered_log1p",
) -> AnnData:
    """
    Score S and G2/M phases per cell and assign a cell-cycle phase.

    Uses the Tirosh et al. (2016) S and G2/M gene lists as packaged by
    `scib` (with real mouse orthologs for `organism="mouse"`), matched to
    `adata.var_names` case-insensitively, then runs
    `sc.tl.score_genes_cell_cycle` on `layer`.

    Parameters
    ----------
    adata
        AnnData object. Writes `S_score`, `G2M_score` and `phase` to
        `adata.obs`.
    organism
        Organism of the gene symbols, passed to
        `scib.preprocessing.get_cell_cycle_genes` (e.g. "mouse", "human").
    layer
        Layer with log-normalized expression.

    Returns
    -------
    AnnData
        Same object (modified in place). Parameters are recorded in
        `adata.uns["annotation_params"]["cell_cycle"]`.

    Notes
    -----
    Domain note: phase "G1" also covers G0 and post-mitotic cells, so
    mature neutrophils are expected to be almost all "G1". In Xie et al.
    (2020), proliferation is concentrated in G0 to G2 (GMP-like, proNeu,
    preNeu), so the score is useful evidence to split precursors from
    post-mitotic neutrophils. Here it is only evidence for annotation, do
    not regress it out at this step.

    Requires the `scib` package, imported lazily.
    """
    from scib.preprocessing import get_cell_cycle_genes

    _CheckLayer(adata, layer)
    gene_map = get_cell_cycle_genes(organism)
    var_map = _VarMap(adata.var_names)
    s_genes = _MatchGenes(gene_map.loc[gene_map["phase"] == "S", "gene_name"], var_map)
    g2m_genes = _MatchGenes(gene_map.loc[gene_map["phase"] == "G2/M", "gene_name"], var_map)
    if not s_genes or not g2m_genes:
        raise ValueError(
            f"Cell-cycle genes not found in adata.var_names for organism='{organism}' "
            f"(S: {len(s_genes)}, G2/M: {len(g2m_genes)} genes found)."
        )

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes, layer=layer, use_raw=False)

    adata.uns.setdefault("annotation_params", {})["cell_cycle"] = {
        "layer": layer,
        "organism": organism,
        "n_s_genes": len(s_genes),
        "n_g2m_genes": len(g2m_genes),
    }
    return adata


# ── Cluster-level label suggestion from scores ───────────────────────────────

def AssignLabelsFromScores(
    adata: AnnData,
    groupby: str,
    score_cols: list[str] | None = None,
    prefix: str = "score_",
    agg: str = "median",
    standardize: bool = False,
    min_purity: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Suggest a label per cluster from per-cell signature scores, with confidence.

    Aggregates each score per cluster and takes the best signature (top1).
    Confidence comes from the gap to the second best (margin) and from the
    fraction of cells in the cluster whose own best signature is top1
    (purity).

    Parameters
    ----------
    adata
        AnnData with signature scores in `adata.obs` (from
        `CalculateModuleScores` or `CalculateAUCellWithDecoupler`).
    groupby
        `adata.obs` column with the cluster labels.
    score_cols
        Score columns to compare. Defaults to every `adata.obs` column
        starting with `prefix`. Columns with only NaN (empty marker sets)
        are skipped with a warning.
    prefix
        Column prefix ("score_" for module scores, "aucell_" for AUCell),
        also removed from the signature names in the output.
    agg
        "median" (default, robust to outlier cells) or "mean".
    standardize
        If True, z-score each signature across cells before comparing. See
        Notes.
    min_purity
        Clusters with purity below this get `suggested_label = "ambiguous"`.

    Returns
    -------
    cluster_scores : pandas.DataFrame
        Clusters x signatures, aggregated score.
    summary : pandas.DataFrame
        One row per cluster: `n_cells`, `top1`, `top2`, `margin` (top1 -
        top2, in score units), `purity`, `suggested_label`, and
        `frac_cycling` (fraction of S/G2M cells) when `adata.obs["phase"]`
        exists (see `ScoreCellCycle`).

    Notes
    -----
    Domain note: this is a suggestion to review, not a replacement for
    manual annotation. Purity is the same idea as CellTypist majority
    voting (Dominguez Conde et al. 2022). An "ambiguous" cluster in a
    continuum (e.g. neutrophil maturation, G2/3) is often a real
    intermediate state, not an error.

    Domain note on `standardize`: without it, signatures with highly
    expressed genes or a wider score range can win the comparison. With
    it, each signature is on the same scale, but the result depends on
    composition: a very abundant cell type pulls the mean of its own
    signature up, which lowers its z-scores.

    The function does not modify `adata`.
    """
    # Stop early if the cluster column does not exist in adata.obs
    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")
    # Only median or mean are allowed to summarize scores per cluster
    if agg not in ("median", "mean"):
        raise ValueError(f"agg must be 'median' or 'mean', got '{agg}'.")
    # No columns given: use every obs column that starts with the prefix
    if score_cols is None:
        score_cols = [c for c in adata.obs.columns if c.startswith(prefix)]
    # Stop if a requested score column does not exist
    missing = [c for c in score_cols if c not in adata.obs]
    if missing:
        raise KeyError(f"Score columns not found in adata.obs: {missing}")

    # Cells x signatures table with the scores, as float
    scores = adata.obs[score_cols].astype(float)
    # Signatures with NaN in every cell (empty marker sets) cannot be compared
    all_nan = scores.columns[scores.isna().all()].tolist()
    # Warn and remove those signatures
    if all_nan:
        logger.warning("Skipping score columns with only NaN: %s", all_nan)
        scores = scores.drop(columns=all_nan)
    # A comparison needs at least two signatures
    if scores.shape[1] < 2:
        raise ValueError("At least 2 non-empty score columns are needed to compare signatures.")

    # Optional: z-score each signature across cells (mean 0, sd 1),
    # constant signatures become NaN instead of dividing by zero
    if standardize:
        scores = (scores - scores.mean()) / scores.std().replace(0, np.nan)
    # Signature names without the prefix, e.g. "score_G3" -> "G3"
    scores.columns = [c.removeprefix(prefix) for c in scores.columns]

    # Cluster label of each cell
    groups = adata.obs[groupby]
    # Clusters x signatures table: median (or mean) score of each cluster
    cluster_scores = scores.groupby(groups, observed=True).agg(agg)

    # Best and second-best signature per cluster
    vals = cluster_scores.to_numpy()
    # Column positions sorted from highest to lowest score, for each cluster
    order = np.argsort(-vals, axis=1)  # NaN goes last
    # Row positions, used to pick one value per row below
    rows = np.arange(len(vals))
    # Name of the highest-scoring signature of each cluster
    top1 = cluster_scores.columns[order[:, 0]]
    # Name of the second highest
    top2 = cluster_scores.columns[order[:, 1]]
    # Gap between the best and the second best score (small = less clear)
    margin = vals[rows, order[:, 0]] - vals[rows, order[:, 1]]

    # Purity: fraction of cells whose own best signature is the cluster top1
    # Cells with at least one non-NaN score
    has_score = scores.notna().any(axis=1)
    # Best signature of each cell (NaN for cells without any score)
    cell_top = pd.Series(np.nan, index=scores.index, dtype=object)
    cell_top[has_score] = scores[has_score].idxmax(axis=1)
    # For each cell, the top1 of its cluster
    expected = pd.Series(top1, index=cluster_scores.index).reindex(groups.to_numpy()).to_numpy()
    # True when the cell agrees with its cluster top1
    agree = pd.Series(cell_top.to_numpy() == expected, index=scores.index)
    # Mean of True/False per cluster = fraction of agreeing cells
    purity = agree.groupby(groups, observed=True).mean().reindex(cluster_scores.index)

    # One row per cluster with the label suggestion and its confidence
    summary = pd.DataFrame(
        {
            "n_cells": groups.value_counts().reindex(cluster_scores.index).to_numpy(),
            "top1": top1,
            "top2": top2,
            "margin": margin,
            "purity": purity.to_numpy(),
        },
        index=cluster_scores.index,
    )
    # Keep top1 as the label when purity is high enough, else "ambiguous"
    summary["suggested_label"] = np.where(summary["purity"] >= min_purity, summary["top1"], "ambiguous")

    # If ScoreCellCycle was run, add the fraction of cycling cells per cluster
    if "phase" in adata.obs:
        # True for cells in S or G2M phase
        cycling = adata.obs["phase"].isin(["S", "G2M"])
        # Mean of True/False per cluster = fraction of cycling cells
        summary["frac_cycling"] = cycling.groupby(groups, observed=True).mean().reindex(cluster_scores.index)

    return cluster_scores, summary


# ── Final annotation: map clusters to labels and plot ───────────────────────

def _RemoveNumberPrefix(label):
    """Remove a leading number like "1. " from a label ("1. G0" -> "G0")."""
    if pd.isna(label):
        return label
    return re.sub(r"^\d+\.\s*", "", str(label))


def _NaturalKey(cluster: str):
    """Sort key so "2" comes before "10"."""
    return (0, int(cluster)) if cluster.isdigit() else (1, cluster)


def PlotFinalAnnotation(
    adata: AnnData,
    annotation: dict[str, str],
    marker_genes: dict[str, list[str]],
    label_order: list[str],
    marker_order: list[str] | None = None,
    res: float | str = 0.6,
    annot_col_name: str = "celltype",
    leiden_layer_prefix: str = "leiden_res",
    figsize: tuple[float, float] = (25, 10),
) -> None:
    """
    Map clusters to final labels, then plot UMAPs and a marker dotplot.

    Writes `adata.obs[annot_col_name]` from the {cluster: label} dict, as an
    ordered categorical following `label_order`. Then plots the clusters and
    the new labels side by side on the UMAP, and a dotplot of
    `marker_genes` grouped by the new labels.

    Parameters
    ----------
    adata
        AnnData object. `adata.obs[annot_col_name]` is created or replaced,
        and `adata.uns[f"{annot_col_name}_colors"]` is reset.
    annotation
        {cluster id: label}. Keys can be str or int. A leading number in the
        label ("1. G0") is removed, so the dict can be kept numbered.
    marker_genes
        {label: genes} for the dotplot, e.g. the output of `MarkerInData`.
        Keys may also have a leading number.
    label_order
        Order of the labels on the y axis of the dotplot and in the legend.
        Labels not in this list become NaN (see Notes).
    marker_order
        Order of the marker groups on top of the dotplot. Defaults to
        `label_order`.
    res
        Leiden resolution. The cluster column is
        `f"{leiden_layer_prefix}{res}"`.
    annot_col_name
        Name of the new `adata.obs` column.
    leiden_layer_prefix
        Prefix of the Leiden cluster columns.
    figsize
        Size of the figure with the two UMAPs.

    Notes
    -----
    Cells become NaN (gray on the UMAP, not in the dotplot) when their
    cluster is not in `annotation`, or when its label is not in
    `label_order`. This is useful to drop clusters on purpose (e.g. label
    them "TIRA"), but it also hides typos like "G5" for "G5a". A warning
    lists every cluster that became NaN, so both cases are visible.
    """
    groupby = f"{leiden_layer_prefix}{res}"
    if groupby not in adata.obs:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")
    if marker_order is None:
        marker_order = label_order

    # Keys as str (Leiden labels are str), labels without leading numbers
    annotation_clean = {str(k): _RemoveNumberPrefix(v) for k, v in annotation.items()}

    # Clusters that will become NaN: not in the dict, or label not in label_order
    clusters = adata.obs[groupby].astype(str).unique()
    not_mapped = sorted((c for c in clusters if c not in annotation_clean), key=_NaturalKey)
    not_in_order = {
        c: annotation_clean[c] for c in sorted(clusters, key=_NaturalKey)
        if c in annotation_clean and annotation_clean[c] not in label_order
    }
    if not_mapped:
        logger.warning("Clusters not in `annotation`, set to NaN: %s", not_mapped)
    if not_in_order:
        logger.warning("Clusters with a label not in `label_order`, set to NaN: %s", not_in_order)

    # New label column, ordered by label_order (only labels really present)
    labels = adata.obs[groupby].astype(str).map(annotation_clean)
    present_labels = [ct for ct in label_order if ct in set(labels.dropna())]
    if not present_labels:
        raise ValueError("No cluster got a label from `label_order`; nothing to plot.")
    adata.obs[annot_col_name] = pd.Categorical(labels, categories=present_labels, ordered=True)

    # Drop old colors so Scanpy builds a palette for the new categories
    adata.uns.pop(f"{annot_col_name}_colors", None)

    # Marker groups for the top of the dotplot, in marker_order
    markers_clean = {_RemoveNumberPrefix(k): genes for k, genes in marker_genes.items()}
    markers_ordered = {ct: markers_clean[ct] for ct in marker_order if ct in markers_clean}

    # UMAPs: clusters (left) and final labels (right)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sc.pl.umap(adata, color=groupby, ax=axes[0], show=False, size=25, legend_loc="on data")
    sc.pl.umap(adata, color=annot_col_name, ax=axes[1], show=False, size=25)
    fig.subplots_adjust(wspace=0.01)
    plt.show()

    # Dotplot: y axis in label order, marker groups on top in marker_order
    sc.pl.dotplot(
        adata,
        groupby=annot_col_name,
        var_names=markers_ordered,
        categories_order=present_labels,
        standard_scale="var",
    )


# ── Score plots on an embedding ──────────────────────────────────────────────

def _PercentileLimits(
    adata: AnnData,
    cols: list[str],
    q_low: float | None,
    q_high: float,
    share_scale: bool,
) -> tuple[list[float | None], list[float | None]]:
    """
    Color limits (vmin, vmax) per column from percentiles, ignoring NaN.

    `q_low=None` gives vmin=None. With `share_scale`, the percentiles are
    computed on all columns together and used for every panel. A column
    with only NaN gets None (matplotlib autoscale).
    """
    def limits(values) -> tuple[float | None, float | None]:
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).any():
            return None, None
        low = None if q_low is None else float(np.nanpercentile(values, q_low))
        return low, float(np.nanpercentile(values, q_high))

    if share_scale:
        shared = limits(np.concatenate([adata.obs[c].to_numpy(dtype=float) for c in cols]))
        pairs = [shared] * len(cols)
    else:
        pairs = [limits(adata.obs[c]) for c in cols]
    return [lo for lo, _ in pairs], [hi for _, hi in pairs]


def PlotModuleScoresUMAPs(
    adata: AnnData,
    score_names: list[str] | None = None,
    ncols: int = 4,
    cmap: str = "bwr",
    share_scale: bool = False,
    percentile: float = 99,
    figsize: tuple[float, float] = (4, 4),
    show_colorbar: bool = True,
    basis: str = "umap",
    prefix: str = "score_",
):
    """
    Plot one embedding (UMAP by default) per module score, on a diverging
    color scale centered at 0, with `sc.pl.embedding`.

    Parameters
    ----------
    adata
        AnnData with module scores in `adata.obs` (see `CalculateModuleScores`).
    score_names
        Columns to plot. Defaults to every `adata.obs` column starting with
        `prefix`.
    ncols
        Maximum number of panels per row.
    cmap
        Diverging colormap: blue (negative) / white (0) / red (positive).
        Coherent with module scores, which are centered on the control set.
    share_scale
        If True, use the same color limits for every panel. If False, each
        panel gets its own limits.
    percentile
        Color limits are the `100 - percentile` and `percentile`
        percentiles of the score (robust to outliers), NaN ignored. White
        is at 0 when the limits cross 0 (`vcenter=0`), so the two sides of
        the scale can have different lengths.
    figsize
        Size of each individual subplot.
    show_colorbar
        Whether to draw a colorbar next to each panel.
    basis
        Embedding in `adata.obsm` (e.g. "umap" for `X_umap`, or another
        key like "X_umap_harmony"), so `X_umap` does not need overwriting.
    prefix
        Column prefix used for the default `score_names` and removed from
        the panel titles.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if there is no column to plot.
    """
    if score_names is None:
        score_names = [c for c in adata.obs.columns if c.startswith(prefix)]
    if not score_names:
        logger.warning("No score columns to plot.")
        return None

    vmin, vmax = _PercentileLimits(adata, score_names, 100 - percentile, percentile, share_scale)
    # White at 0 only where the color range crosses 0 (matplotlib needs vmin < 0 < vmax)
    vcenter = [0 if lo is not None and lo < 0 < hi else None for lo, hi in zip(vmin, vmax)]

    with plt.rc_context({"figure.figsize": figsize}):  # size of each panel
        return sc.pl.embedding(
            adata, basis=basis, color=score_names,
            title=[c.removeprefix(prefix) for c in score_names],
            ncols=min(ncols, len(score_names)), color_map=cmap,
            vmin=vmin, vmax=vmax, vcenter=vcenter, frameon=False,
            colorbar_loc=("right" if show_colorbar else None),
            show=False, return_fig=True,
        )


def PlotAUCellUMAPs(
    adata: AnnData,
    cols: list[str] | None = None,
    ncols: int = 4,
    cmap: str = "viridis",
    percentile: float = 99,
    share_scale: bool = False,
    figsize: tuple[float, float] = (4, 4),
    show_colorbar: bool = True,
    basis: str = "umap",
    prefix: str = "aucell_",
):
    """
    Plot one embedding (UMAP by default) per AUCell score, on a sequential
    color scale starting at 0, with `sc.pl.embedding`.

    Parameters
    ----------
    adata
        AnnData with AUCell scores in `adata.obs` (see
        `CalculateAUCellWithDecoupler`).
    cols
        Columns to plot. Defaults to every `adata.obs` column starting with
        `prefix`.
    ncols
        Maximum number of panels per row.
    cmap
        Sequential colormap. Coherent with AUCell, which is never negative.
    percentile
        Percentile used as the upper color-scale limit (robust to
        outliers). Lower limit is always 0. NaN values are ignored.
    share_scale
        If True, use the same 0-to-vmax color scale for every panel. If
        False, each panel gets its own scale.
    figsize
        Size of each individual subplot.
    show_colorbar
        Whether to draw a colorbar next to each panel.
    basis
        Embedding in `adata.obsm` (e.g. "umap" for `X_umap`).
    prefix
        Column prefix used for the default `cols` and removed from the
        panel titles.

    Returns
    -------
    matplotlib.figure.Figure or None
        None if there is no column to plot.
    """
    if cols is None:
        cols = [c for c in adata.obs.columns if c.startswith(prefix)]
    if not cols:
        logger.warning("No score columns to plot.")
        return None

    _, vmax = _PercentileLimits(adata, cols, None, percentile, share_scale)
    # AUCell is never negative, so the scale always starts at 0
    vmin = [0.0 if hi is not None else None for hi in vmax]

    with plt.rc_context({"figure.figsize": figsize}):  # size of each panel
        return sc.pl.embedding(
            adata, basis=basis, color=cols,
            title=[c.removeprefix(prefix) for c in cols],
            ncols=min(ncols, len(cols)), color_map=cmap,
            vmin=vmin, vmax=vmax, frameon=False,
            colorbar_loc=("right" if show_colorbar else None),
            show=False, return_fig=True,
        )
