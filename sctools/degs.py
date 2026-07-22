"""
sctools.degs
============

Pseudobulk differential expression (PyDESeq2) and hallmark/pathway activity
(decoupler ULM) workflow.

Core principle: pseudobulk DEGs never depend on PCA. `PseudoPCA` is an
optional QC/exploration step -- it never modifies `pdata.X`, which always
keeps raw pseudobulk counts. `PseudoDESeq2` (via PyDESeq2) always fits on
those raw counts, regardless of whether `PseudoPCA` was run.

Not ported:
- `PlotVolcanoGrid` -- superseded by `VolcanoGridByGroup` (better labeling/
  layout); no call site left in either source notebook.
- `BuildHallmarkHeatmapTablesAndPlot`/`PlotScoreHeatmap` -- both notebooks
  originally defined a hallmark-specific heatmap builder, but it had no
  hallmark-specific logic left once genericized (just "restrict to rows
  significant somewhere, then plot"). That's now
  `sctools.plots.PlotSignificanceHeatmap(..., restrict_to_significant=True)`.
- `RunULMHeatmap` -- a composed "loop + melt + heatmap for all celltypes"
  convenience, but it was never actually used anywhere: the barplot grid
  a real notebook needed isn't something `dc.pl.barplot` does per-figure,
  so that per-celltype loop ended up living directly in the notebook
  instead (`RunULM(..., ax=...)` into a shared `plt.subplots` grid). Removed
  to avoid keeping an unused, harder-to-maintain duplicate around.

Typical usage order:
    EvaluateCelltypesForPseudobulk()    -> (notebook-local, not in this module) pick celltypes with enough cells/replicates
    Pseudobulking()                     -> aggregate counts per (celltype, sample)
    PseudoPCA()                         -> optional QC / variance exploration on pseudobulk (never required for DEGs)
    PseudoFeatSelection()               -> per-celltype gene filtering (decoupler)
    PseudoDESeq2()                      -> DESeq2 contrast on raw pseudobulk counts (PyDESeq2)
    VolcanoGridByGroup()                -> volcano grid, one panel per celltype

Feature-set / pathway activity (decoupler ULM, e.g. dc.op.hallmark,
dc.op.progeny, dc.op.collectri, or any other decoupler net -- everything
below is generic over the network and its feature naming, not specific to
MSigDB hallmarks):
    RunULM()                       -> run ULM for ONE input against any net, + optional barplot (pass `ax=` for a grid)
    MeltActsPadjToLong()           -> combine several celltypes' stored acts/padj into one tidy table
    BuildSignificantFeatureTable() -> long-format table restricted to significant hits, + category/direction columns
    SummarizeFeatureCategories()   -> per-celltype/category summary from that table

The heatmap itself is `sctools.plots.PlotSignificanceHeatmap` -- generic
(any long-format table with an x column, a y column, a color value column,
and a p-value/padj column). Pass `restrict_to_significant=True` to only
show y-values (e.g. hallmarks) that are significant in at least one x (e.g.
celltype); its own `category_map` param takes the same `{category:
[features]}` shape as `_BuildFeatureToCategoryMap`/`BuildSignificantFeatureTable`.

`RunULM` runs on a single input, matching `PseudoFeatSelection`/
`PseudoDESeq2`'s "one call per celltype, loop lives in the caller" pattern
instead of looping internally over a `pdata_by_celltype` dict.

`_BuildFeatureToCategoryMap`'s `prefix` param strips a naming-convention
prefix (e.g. `prefix="HALLMARK_"` for MSigDB) -- leave it empty for feature
sets with no such prefix (progeny pathways, collectri TFs, ...).
`BuildSignificantFeatureTable`/`SummarizeFeatureCategories` derive their
"direction" labels (e.g. "increased_in_ANTIB") from the actual
`compare_condition` half of `contrast_name` (`"{compare_condition}.vs.
{normal_condition}"`) rather than a hardcoded "KO"/"WT" -- works for any
two-condition contrast name.
"""

from __future__ import annotations

import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

__all__ = [
    "Pseudobulking",
    "PseudoPCA",
    "PseudoFeatSelection",
    "PseudoDESeq2",
    "VolcanoGridByGroup",
    "RunULM",
    "MeltActsPadjToLong",
    "BuildSignificantFeatureTable",
    "SummarizeFeatureCategories",
]


def _EnsureParentDir(path: str) -> None:
    """Create the parent directory of `path` if it doesn't already exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


########################################################################################
# ── 1. Pseudobulk aggregation & PCA ───────────────────────────────────────────────────

def Pseudobulking(
    adata: AnnData,
    celltype_col: str,
    *,
    min_cells: int = 10,
    min_counts: int = 1000,
    sample_col: str = "sample",
    counts_layer: str = "QC_filtered",
    condition_col: str = "condition",
    id_col: str = "id",
    make_pca_plots: bool = True,
    make_bar_plot: bool = True,
) -> AnnData:
    """
    Perform pseudobulk aggregation for Decoupler.

    Parameters
    ----------
    adata : AnnData
    celltype_col : str -> column with celltype annotation
    sample_col : str -> column defining biological replicates
    counts_layer : str -> layer containing raw counts
    condition_col : str -> condition column
    id_col : str -> biological ID column
    min_cells : int -> minimum cells per (celltype, sample) pseudobulk group to keep
    min_counts : int -> minimum total counts per (celltype, sample) pseudobulk group to keep
    make_pca_plots : bool -> whether to display the QC filter-samples plot
    make_bar_plot : bool -> whether to display the celltype-by-condition bar plot
    """
    import decoupler as dc

    pdata = dc.pp.pseudobulk(
        adata=adata,
        sample_col=sample_col,
        groups_col=celltype_col,
        mode="sum",
        layer=counts_layer,
    )
    # dc.pp.pseudobulk also generates QC metrics used below and by
    # PseudoFeatSelection: pdata.obs["psbulk_cells"] (cells aggregated per
    # pseudobulk sample), pdata.obs["psbulk_counts"] (total counts per
    # sample), and pdata.layers["psbulk_props"] (per gene, the fraction of
    # those cells with a non-zero count -- what filter_by_prop filters on).

    # QC plots (optional)
    if make_pca_plots:
        dc.pl.filter_samples(
            adata=pdata,
            groupby=[condition_col, id_col, celltype_col],
            min_cells=min_cells,
            min_counts=min_counts,
            figsize=(5, 8),
        )

    if make_bar_plot:
        dc.pl.obsbar(
            adata=pdata,
            y=celltype_col,
            hue=condition_col,
            figsize=(6, 3),
        )

    # Filtering (always applied) -- must match the thresholds shown in the QC plot above.
    dc.pp.filter_samples(pdata, min_cells=min_cells, min_counts=min_counts)

    pdata.uns["pseudobulk_params"] = {
        "celltype_col": celltype_col,
        "sample_col": sample_col,
        "counts_layer": counts_layer,
        "condition_col": condition_col,
        "id_col": id_col,
        "min_cells": min_cells,
        "min_counts": min_counts,
    }

    return pdata


def _GetValidObsKeysForRankObsm(
    pdata,
    obs_keys: list[str],
    min_group_size: int = 2,
) -> list[str]:
    """
    Keep only obs columns that can be safely tested by decoupler.rankby_obsm.

    Categorical columns need at least two groups and at least `min_group_size`
    samples per group. Numeric columns need variation.
    """
    valid_keys = []

    for key in obs_keys:
        if key not in pdata.obs:
            logger.warning("Skipping obs key '%s': not found in pdata.obs.", key)
            continue

        values = pdata.obs[key].dropna()

        if values.empty:
            continue

        if pd.api.types.is_numeric_dtype(values):
            if values.nunique() > 1:
                valid_keys.append(key)
            continue

        value_counts = values.astype(str).value_counts()

        if len(value_counts) < 2:
            continue

        if (value_counts >= min_group_size).all():
            valid_keys.append(key)

    return valid_keys


def PseudoPCA(
    pdata: AnnData,
    *,
    id_col: str = "id",
    celltype_col: str = "celltype",
    condition_col: str = "condition",
    counts_layer: str = "counts",
    log_layer: str | None = None,
    target_sum: float = 1e4,
    n_comps: int | None = None,
    scale_before_pca: bool = True,
    max_value: float = 10,
    random_state: int = 42,
    rank_obs_keys: list[str] | None = None,
    min_group_size_for_rank: int = 2,
    make_plots: bool = True,
    copy: bool = True,
) -> AnnData:
    """
    Compute PCA for pseudobulk QC / variance exploration.

    This is a QC/exploration step only -- it never modifies `pdata.X`, which
    keeps the raw pseudobulk counts required by `PseudoDESeq2` untouched.
    Pseudobulk DEGs never depend on this function having been run.

    Preprocessing recipe (matches decoupler's own pseudobulk tutorial):
        raw pseudobulk counts -> normalize_total -> log1p -> [scale] -> PCA
    No highly-variable-gene selection or neighbor graph/UMAP is used here --
    both assume many single-cell samples and are not statistically
    appropriate for the handful of pseudobulk samples typical here.

    Results are written to:
        pdata.layers[log_layer]   (normalized + log1p, QC/visualization only)
        pdata.obsm["X_pca"], pdata.uns["pca"], pdata.varm["PCs"]
        pdata.uns["pseudo_pca_params"]
    """
    import decoupler as dc

    pdata_out = pdata.copy() if copy else pdata

    log_layer = log_layer or f"{counts_layer}_log1p"

    # Keep raw pseudobulk counts safely stored.
    if counts_layer not in pdata_out.layers:
        pdata_out.layers[counts_layer] = pdata_out.X.copy()

    # Normalize + log1p on a temporary object; pdata_out.X is never touched,
    # so it keeps raw pseudobulk counts for downstream DESeq2.
    pdata_norm = pdata_out.copy()
    pdata_norm.X = pdata_out.layers[counts_layer].copy()
    sc.pp.normalize_total(pdata_norm, target_sum=target_sum)
    sc.pp.log1p(pdata_norm)
    pdata_out.layers[log_layer] = pdata_norm.X.copy()

    # PCA runs on its own temporary object, so scaling never touches
    # pdata_out.layers[log_layer] (kept as log1p, unscaled, for reuse/plots).
    pdata_pca = pdata_norm.copy()
    if scale_before_pca:
        sc.pp.scale(pdata_pca, max_value=max_value)

    # Limit the number of PCs to what the data dimensions allow.
    max_n_comps = min(pdata_pca.n_obs - 1, pdata_pca.n_vars - 1)

    if max_n_comps < 1:
        raise ValueError(
            "PCA cannot be computed: pseudobulk object has too few samples or genes."
        )

    n_comps_run = min(50 if n_comps is None else n_comps, max_n_comps)

    sc.tl.pca(
        pdata_pca,
        n_comps=n_comps_run,
        svd_solver="arpack",
        random_state=random_state,
    )

    # Copy PCA results from the temporary object back to the final pdata.
    # pdata_out.X remains raw pseudobulk counts for DESeq2.
    pdata_out.obsm["X_pca"] = pdata_pca.obsm["X_pca"].copy()
    pdata_out.uns["pca"] = pdata_pca.uns["pca"].copy()
    pdata_out.varm["PCs"] = pdata_pca.varm["PCs"].copy()

    pdata_out.uns["pseudo_pca_params"] = {
        "counts_layer": counts_layer,
        "log_layer": log_layer,
        "target_sum": target_sum,
        "n_comps": n_comps_run,
        "scale_before_pca": scale_before_pca,
        "max_value": max_value,
        "random_state": random_state,
    }

    # Choose metadata columns to test for association with PCs.
    if rank_obs_keys is None:
        rank_obs_keys = [condition_col, celltype_col, id_col]

    valid_rank_obs_keys = _GetValidObsKeysForRankObsm(
        pdata_out,
        obs_keys=rank_obs_keys,
        min_group_size=min_group_size_for_rank,
    )

    if valid_rank_obs_keys:
        dc.tl.rankby_obsm(
            pdata_out,
            key="X_pca",
            obs_keys=valid_rank_obs_keys,
        )
    else:
        logger.info("No valid obs columns found for rankby_obsm.")

    if make_plots:
        sc.pl.pca_variance_ratio(pdata_out)

        if "rank_obsm" in pdata_out.uns:
            # dc.pl.obsm dendrogram-clusters the (few) tested obs_keys
            # (celltype/condition/id, typically only 2-3 rows). If two of
            # them end up with numerically identical -log10(padj) across
            # the shown PCs (e.g. both non-significant -> both ~0), that
            # pair merges at exactly distance 0, producing a zero-height
            # dendrogram segment that marsilea's coordinate normalization
            # divides by (ValueError: Axis limits cannot be NaN or Inf).
            # dendrogram=False avoids this class of tie entirely; the
            # try/except stays as a safety net for anything else, since
            # PCA itself already succeeded and is stored above -- this is
            # an optional QC plot, not worth failing the whole function.
            try:
                dc.pl.obsm(
                    adata=pdata_out,
                    key="rank_obsm",
                    nvar=5,
                    dendrogram=False,
                    titles=["PC scores", "Adjusted p-values"],
                    figsize=(5, 5),
                )
            except ValueError as error:
                logger.warning("Skipping rank_obsm plot (%s).", error)

        plot_cols = [
            col for col in [id_col, condition_col, celltype_col]
            if col in pdata_out.obs
        ]

        if plot_cols:
            sc.pl.pca(
                pdata_out,
                color=plot_cols,
                ncols=min(3, len(plot_cols)),
                size=300,
                frameon=True,
            )

    return pdata_out


########################################################################################
# ── 2. Per-celltype pseudobulk DE (PyDESeq2) ──────────────────────────────────────────

def PseudoFeatSelection(
    pdata: AnnData,
    celltype: str,
    *,
    celltype_col: str = "celltype",
    design_col: str = "condition",
    min_count: int = 10,
    min_total_count: int = 15,
    large_n: int = 10,
    min_prop_expr: float = 0.7,
    min_prop: float = 0.1,
    min_smpls: int = 2,
    plot: bool = True,
) -> AnnData:
    """
    Filter a single celltype's pseudobulk to well-expressed genes (decoupler).

    With few pseudobulk samples per celltype, genes with low or sporadic
    counts don't carry enough information for DESeq2's negative-binomial
    dispersion estimate to be reliable -- they just add noise and inflate
    the multiple-testing burden without contributing real signal. This is
    the same rationale behind edgeR's `filterByExpr` for bulk RNA-seq;
    `dc.pp.filter_by_expr`/`dc.pp.filter_by_prop` apply that idea to
    pseudobulk. Filtering is done per celltype (not once for the whole
    dataset) because which genes count as "well expressed" depends on that
    celltype's own expression profile -- a global filter would drop genes
    that matter for some celltypes or keep noise for others.

    Default thresholds match decoupler's official pseudobulk tutorial example
    (https://decoupler.readthedocs.io/en/latest/notebooks/scell/rna_psbk.html).
    The tutorial notes these are dataset-specific and recommends inspecting
    the count-distribution plots (`plot=True`) before finalizing them.

    Parameters
    ----------
    min_count, min_total_count, large_n, min_prop_expr : passed to `dc.pp.filter_by_expr`
        (`min_prop_expr` maps to that function's own `min_prop` argument).
    min_prop, min_smpls : passed to `dc.pp.filter_by_prop`.
    """
    import decoupler as dc

    if celltype not in pdata.obs[celltype_col].unique():
        raise ValueError(
            f"celltype='{celltype}' not found in adata.obs['{celltype_col}']. "
            f"Available: {sorted(pdata.obs[celltype_col].unique())}."
        )

    pdata_cells = pdata[pdata.obs[celltype_col] == celltype].copy()

    filter_expr_kwargs = dict(
        group=design_col,
        min_count=min_count,
        min_total_count=min_total_count,
        large_n=large_n,
        min_prop=min_prop_expr,
    )
    filter_prop_kwargs = dict(min_prop=min_prop, min_smpls=min_smpls)

    if plot:
        dc.pl.filter_by_expr(adata=pdata_cells, **filter_expr_kwargs)
        dc.pl.filter_by_prop(adata=pdata_cells, **filter_prop_kwargs)

    dc.pp.filter_by_expr(adata=pdata_cells, **filter_expr_kwargs)
    dc.pp.filter_by_prop(adata=pdata_cells, **filter_prop_kwargs)

    return pdata_cells


def PseudoDESeq2(
    pdata_cells: AnnData,
    design_col: str,
    normal_condition: str,
    compare_condition: str,
    *,
    min_replicates: int = 1,
    n_cpus: int = 8,
    plot_volcano: bool = True,
    deseq_quiet:bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run PyDESeq2 for one celltype's pseudobulk and return results_df + stat row.

    Expects `pdata_cells.X` to hold raw pseudobulk counts -- true by
    construction, since neither `PseudoPCA` nor `PseudoFeatSelection` modify
    `.X`. Uses `design=f"~{design_col}"` (PyDESeq2's current, non-deprecated
    API; equivalent to the deprecated `design_factors=[design_col]`, which
    is internally converted to the exact same formula string for a single
    factor).

    Requires at least `min_replicates` samples for each of `normal_condition`
    and `compare_condition` in `pdata_cells.obs[design_col]` -- DESeq2's
    negative-binomial dispersion estimate is not reliable with a single
    replicate per group, so this is checked explicitly up front rather than
    left to fail deep inside PyDESeq2/statsmodels.
    """
    import decoupler as dc
    from pydeseq2.dds import DeseqDataSet, DefaultInference
    from pydeseq2.ds import DeseqStats

    counts_per_condition = pdata_cells.obs[design_col].value_counts()

    for condition in (normal_condition, compare_condition):
        n = int(counts_per_condition.get(condition, 0))
        if n < min_replicates:
            raise ValueError(
                f"Condition '{condition}' has {n} replicate(s) in "
                f"pdata_cells.obs['{design_col}'], need at least "
                f"{min_replicates} for a reliable DESeq2 dispersion estimate. "
                f"Available levels: {counts_per_condition.to_dict()}."
            )

    # Build DESeq2 object (raw pseudobulk counts in pdata_cells.X)
    inference = DefaultInference(n_cpus=n_cpus)
    dds = DeseqDataSet(
        adata=pdata_cells,
        design=f"~{design_col}",
        refit_cooks=True,
        inference=inference,
    )

    # Compute LFCs
    dds.deseq2()

    # Extract contrast between conditions
    stat_res = DeseqStats(
        dds,
        contrast=[design_col, compare_condition, normal_condition],
        inference=inference,
        quiet=deseq_quiet,
    )

    # Compute Wald test
    stat_res.summary()

    # Extract results
    results_df = stat_res.results_df

    if plot_volcano:
        dc.pl.volcano(results_df, x="log2FoldChange", y="pvalue")

    data = results_df[["stat"]].T.rename(index={"stat": f"{compare_condition}.vs.{normal_condition}"})

    return results_df, data


########################################################################################
# ── 3. Volcano plots ───────────────────────────────────────────────────────────────────

def VolcanoGridByGroup(
    df: pd.DataFrame,
    group_col: str = "celltype",          # era "tissue" no seu docstring; aqui é o agrupador
    lfc_thr: float = 1.0,
    p_thr: float = 0.05,
    top_labels: int = 10,
    col_gene: str = "gene_id",
    col_pvalue: str = "padj",
    col_fc: str = "log2FoldChange",
    n_cols: int = 3,                      # <- você escolhe quantas colunas
    figsize_per_ax=(4.2, 3.6),            # tamanho por painel
    point_size: float = 12,
    alpha: float = 0.75,
    y_quantile: float = 0.995,
    balance_labels: bool = True,
    sharex: bool = True,                  # True = Todos os subplots terão o mesmo eixo X (mesma escala de log2FC).
    sharey: bool = False,                 # False =  Cada subplot terá seu próprio eixo Y (-log10 p).
    save_path: str | None = None,         # salva UMA figura
    dpi: int = 180,
    return_fig: bool = False
):
    """
    Volcano plots em grid, um painel por grupo (ex: celltype).

    `col_pvalue` defaults to "padj" (multiple-testing-corrected), not raw
    "pvalue" -- using an uncorrected p-value as the default significance
    column for a genome-wide volcano plot would be statistically incorrect.
    Pass `col_pvalue="pvalue"` explicitly if you really want the raw value.
    """

    df = df.copy()
    name_log10 = f"-log10_{col_pvalue}"

    # 1) p==0 e p inválidos -> clip e -log10
    #    Floor = smallest observed positive p-value (falls back to 1e-300
    #    only when there is no positive p-value at all in the data).
    ps = df[col_pvalue].replace([np.inf, -np.inf], np.nan)
    min_pos = ps[ps > 0].min()
    if pd.isna(min_pos):
        min_pos = 1e-300
    min_pos = float(min_pos)

    df[f"{col_pvalue}_clip"] = np.clip(df[col_pvalue].astype(float), min_pos, 1.0)
    df[name_log10] = -np.log10(df[f"{col_pvalue}_clip"])

    # 2) status
    def _status(r):
        if (r[col_fc] >= lfc_thr) and (r[col_pvalue] <= p_thr):
            return "Up"
        if (r[col_fc] <= -lfc_thr) and (r[col_pvalue] <= p_thr):
            return "Down"
        return "NS"

    df["status"] = df.apply(_status, axis=1)

    colors = {"Up": "#d62728", "Down": "#1f77b4", "NS": "#b0b0b0"}
    y_thr_line = -math.log10(max(p_thr, 1e-300))

    # grupos
    groups = list(df[group_col].dropna().unique())
    groups = sorted(groups)

    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError(f"No groups found in column '{group_col}'.")

    n_cols = max(1, int(n_cols))
    n_rows = int(math.ceil(n_groups / n_cols))

    # figura
    fig_w = figsize_per_ax[0] * n_cols
    fig_h = figsize_per_ax[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        sharex=sharex,
        sharey=sharey
    )
    axes = np.array(axes).reshape(-1)

    # para ter limites coerentes (opcional)
    if sharex:
        xmax_global = float(np.nanmax(np.abs(df[col_fc].values)))
        xmax_global = 1.05 * max(xmax_global, lfc_thr)
    else:
        xmax_global = None

    if sharey:
        # define um teto global robusto
        ymax = float(df[name_log10].max())
        qy = float(np.nanquantile(df[name_log10], y_quantile))
        if ymax > qy * 1.6:
            y_upper_global = max(qy * 1.05, y_thr_line * 1.5)
        else:
            y_upper_global = max(ymax * 1.05, y_thr_line * 1.5)
    else:
        y_upper_global = None

    for i, grp in enumerate(groups):
        ax = axes[i]
        dfg = df[df[group_col] == grp].copy()

        # labels: maiores |log2FC| entre significativos
        sig = dfg[dfg[col_pvalue] <= p_thr].copy()
        to_label = pd.DataFrame()

        if top_labels and len(sig):
            sig["abs_lfc"] = sig[col_fc].abs()
            if balance_labels:
                n_each = max(1, top_labels // 2)
                up = sig[sig[col_fc] > 0].sort_values("abs_lfc", ascending=False).head(n_each)
                down = sig[sig[col_fc] < 0].sort_values("abs_lfc", ascending=False).head(n_each)
                to_label = pd.concat([up, down], axis=0)
            else:
                to_label = sig.sort_values("abs_lfc", ascending=False).head(top_labels)

        # y-lim por painel (se não compartilhar y)
        if not sharey:
            ymax = float(dfg[name_log10].max()) if len(dfg) else y_thr_line * 2
            qy = float(np.nanquantile(dfg[name_log10], y_quantile)) if len(dfg) else y_thr_line * 2
            labels_top = float(to_label[name_log10].max()) if len(to_label) else 0.0

            if ymax > qy * 1.6:
                y_upper = max(qy * 1.05, labels_top * 1.10, y_thr_line * 1.5)
            else:
                y_upper = max(ymax * 1.05, labels_top * 1.10, y_thr_line * 1.5)
        else:
            y_upper = y_upper_global

        # x-lim por painel (se não compartilhar x)
        if not sharex:
            xmax = float(np.nanmax(np.abs(dfg[col_fc]))) if len(dfg) else 1.0
            xmax = 1.05 * max(xmax, lfc_thr)
        else:
            xmax = xmax_global

        # contagens
        n_ns = (dfg["status"] == "NS").sum()
        n_down = (dfg["status"] == "Down").sum()
        n_up = (dfg["status"] == "Up").sum()

        # plot com labels incluindo contagem
        for status, n in [("NS", n_ns), ("Down", n_down), ("Up", n_up)]:
            sub = dfg[dfg["status"] == status]
            ax.scatter(
                sub[col_fc],
                sub[name_log10],
                s=point_size,
                c=colors[status],
                alpha=alpha if status != "NS" else 0.5,
                linewidths=0,
                label=f"{status} (n={n})"
            )

        # título (pad=18 evita sobrepor a legenda logo abaixo)
        ax.set_title(str(grp), fontsize=11, pad=18)

        # legenda logo abaixo do título
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03), #1.05),   # ↓ mais próximo do título
            ncol=3,
            frameon=False,
            fontsize=8,
            handletextpad=0.3,
            columnspacing=0.8
        )

        # linhas
        ax.axvline(+lfc_thr, ls="--", lw=1, color="#888888")
        ax.axvline(-lfc_thr, ls="--", lw=1, color="#888888")
        ax.axhline(y_thr_line, ls="--", lw=1, color="#888888")

        # labels
        for _, r in to_label.iterrows():
            ax.annotate(
                str(r[col_gene]),
                (r[col_fc], r[name_log10]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7
            )

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(0, y_upper)
        ax.grid(True, axis="y", linestyle=":", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # só coloca rótulos de eixo nas bordas para não poluir
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("log2FC")
        if i % n_cols == 0:
            ax.set_ylabel(r"$-\log_{10}(p)$")

    # esconde axes vazios
    for j in range(n_groups, len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        _EnsureParentDir(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig, axes[:n_groups]
    else:
        plt.show()
        return None


########################################################################################
# ── 4. Hallmark / pathway activity (decoupler ULM) ────────────────────────────────────


def MeltActsPadjToLong(
    pdata_by_celltype: dict,
    name_df_acts: str,
    name_df_padj: str,
    contrast_name: str,
    feature_col: str = "feature",
) -> pd.DataFrame:
    """
    Melt each celltype's 1-row (contrast) acts/padj tables into one long
    table: one row per (celltype, feature), with columns
    [celltype, feature_col, "contrast", "score", "padj"].

    Use this after your own per-celltype loop over `RunULM` has stored
    `hm_acts`/`hm_padj` into `pdata_by_celltype[celltype][name_df_acts/padj]`,
    to combine them into one tidy table for
    `sctools.plots.PlotSignificanceHeatmap`.
    """
    records = []

    for celltype, d in pdata_by_celltype.items():
        hm_acts = d.get(name_df_acts)
        hm_padj = d.get(name_df_padj)

        if hm_acts is None or hm_padj is None:
            continue
        if contrast_name not in hm_acts.index or contrast_name not in hm_padj.index:
            continue

        scores = hm_acts.loc[contrast_name]
        padjs = hm_padj.loc[contrast_name]

        for feature in scores.index:
            records.append({
                "celltype": celltype,
                feature_col: feature,
                "contrast": contrast_name,
                "score": scores[feature],
                "padj": padjs.get(feature, np.nan),
            })

    return pd.DataFrame(records)


def RunULM(
    data,
    net,
    *,
    contrast_name: str = "KO.vs.WT",
    padj_threshold: float = 0.05,
    plot: bool = True,
    ax=None,
    save_path: str | None = None,
    figsize=(6, 5),
    dpi: int = 300,
    title: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ULM for one input against any decoupler network, and optionally
    plot a barplot of the significant activities.

    `data` is whatever `dc.mt.ulm` accepts -- typically one celltype's
    `PseudoDESeq2` `data` (a 1-row stat table), but also a full pseudobulk
    AnnData if you want activities per sample instead of per contrast.

    This runs on ONE input; to cover several celltypes, loop over your own
    `pdata_by_celltype` and call this once per celltype, storing
    `hm_acts`/`hm_padj` yourself (e.g. `pdata_by_celltype[ct]["hallmark_acts"]
    = hm_acts`). See `MeltActsPadjToLong` to combine what you stored back
    into one tidy table for `sctools.plots.PlotSignificanceHeatmap`. Pass
    `ax` (e.g. one panel of your own `plt.subplots` grid) to draw several
    celltypes' barplots into a shared grid figure instead of one figure per
    celltype -- when `ax` is given, `RunULM` only draws into it and sets its
    title; it doesn't show/save/close the figure, since you own its
    lifecycle in that case (`save_path`/`plot`/`dpi` are ignored).

    Parameters
    ----------
    data : network input passed to `dc.mt.ulm` (e.g. a PseudoDESeq2 `data` row, or an AnnData).
    net : network passed to `dc.mt.ulm` -- any decoupler net (hallmark, progeny, collectri, ...).
    contrast_name : str --> Row in `hm_acts`/`hm_padj` used to select which
        activities to barplot (e.g. "KO.vs.WT"). If not found (e.g. `data`
        has one row per sample rather than per contrast), the barplot is
        skipped but `hm_acts`/`hm_padj` are still returned.
    padj_threshold : float --> Significance cutoff (for the barplot only).
    plot : bool --> Whether to display the barplot. Ignored if `ax` is given.
    ax : matplotlib Axes or None --> Draw into this existing Axes (e.g. one
        panel of a grid) instead of creating a new figure.
    save_path : str or None --> If provided, the barplot is saved here.
        Ignored if `ax` is given -- save the whole shared figure yourself once.
    dpi : int --> Resolution for the saved image. Ignored if `ax` is given.
    title : str or None --> Barplot title. Defaults to `contrast_name`.

    Returns
    -------
    hm_acts, hm_padj : pd.DataFrame
    """
    import decoupler as dc

    hm_acts, hm_padj = dc.mt.ulm(data=data, net=net)

    if contrast_name not in hm_padj.index:
        if ax is not None:
            ax.set_title(title or contrast_name)
            ax.axis("off")
        return hm_acts, hm_padj

    msk = (hm_padj.T < padj_threshold).iloc[:, 0]
    hm_sig = hm_acts.loc[:, msk]

    if hm_sig.shape[1] == 0:
        print(f"No significant activities for '{contrast_name}'.")
        if ax is not None:
            ax.set_title(title or contrast_name)
            ax.text(0.5, 0.5, "no significant\nactivities", ha="center", va="center", fontsize=8)
            ax.axis("off")
        return hm_acts, hm_padj

    if plot or save_path is not None or ax is not None:
        if ax is not None:
            dc.pl.barplot(data=hm_sig, name=contrast_name, ax=ax)
            ax.set_title(title or contrast_name)
        else:
            dc.pl.barplot(data=hm_sig, name=contrast_name, figsize=figsize, return_fig=True)
            plt.title(title or contrast_name)

            if save_path is not None:
                _EnsureParentDir(save_path)
                plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

            if plot:
                plt.show()

            plt.close()

    return hm_acts, hm_padj



def _BuildFeatureToCategoryMap(category_map: dict, prefix: str = "") -> dict:
    """
    Invert a {category: [features]} dict into {feature: category}.

    Parameters
    ----------
    category_map : dict -> {category: list/tuple of features}.
    prefix : str -> optional prefix to strip, e.g. `prefix="HALLMARK_"` for
        MSigDB hallmark names. Both the prefixed and unprefixed forms of
        each feature get mapped, so lookups work regardless of which form
        your data uses. Leave empty (default) for feature sets with no such
        prefix convention (progeny pathways, collectri TFs, ...).
    """
    feature_to_category = {}

    for category, features in category_map.items():
        for feature in features:
            feature_clean = feature.replace(prefix, "") if prefix else feature
            feature_to_category[feature_clean] = category
            if prefix:
                feature_to_category[f"{prefix}{feature_clean}"] = category

    return feature_to_category

##################################################
# The functions below build two levels of enrichment summary on top of a
# tidy per-feature table (e.g. MeltActsPadjToLong's output):
#   BuildSignificantFeatureTable -> one row per significant (celltype,
#     feature) hit, with a "category" (from category_map) and "direction"
#     (increased/decreased in compare_condition) column added.
#   SummarizeFeatureCategories -> one row per (celltype, category), built
#     on top of that -- counts, mean score, dominant direction. Answers
#     "which functional categories change most, and in which direction,
#     per celltype" without reading the heatmap feature by feature.


def BuildSignificantFeatureTable(
    long_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    celltype_col: str = "celltype",
    value_col: str = "score",
    pvalue_col: str = "padj",
    category_map: dict | None = None,
    category_prefix: str = "",
    padj_threshold: float = 0.05,
    contrast_name: str = "KO.vs.WT",
) -> pd.DataFrame:
    """
    Restrict a long-format ULM/enrichment table (e.g. from
    `MeltActsPadjToLong`) to significant hits, and add `category` and
    `direction` columns.

    Parameters
    ----------
    long_df : long-format table, e.g. `MeltActsPadjToLong`'s output.
    feature_col, celltype_col, value_col, pvalue_col : column names in `long_df`.
    category_map, category_prefix : passed to `_BuildFeatureToCategoryMap`
        if `category_map` is given; features not found get "uncategorized".
    padj_threshold : float -> cutoff applied to `pvalue_col`.
    contrast_name : str -> formatted as "{compare_condition}.vs.{normal_condition}"
        (matches `PseudoDESeq2`'s `data` naming). The "compare_condition"
        half labels the "direction" column (e.g. "increased_in_ANTIB") --
        not tied to any specific condition names.
    """
    compare_condition = contrast_name.split(".vs.")[0]
    direction_up = f"increased_in_{compare_condition}"
    direction_down = f"decreased_in_{compare_condition}"

    feature_to_category = (
        _BuildFeatureToCategoryMap(category_map, prefix=category_prefix)
        if category_map is not None
        else {}
    )

    df = long_df.dropna(subset=[value_col, pvalue_col])
    df = df[df[pvalue_col] < padj_threshold].copy()

    if df.empty:
        return df

    df["category"] = df[feature_col].map(feature_to_category).fillna("uncategorized")
    df["direction"] = np.where(df[value_col] > 0, direction_up, direction_down)
    df["contrast"] = contrast_name

    return df.sort_values(
        [celltype_col, "category", "direction", pvalue_col, value_col],
        ascending=[True, True, True, True, False],
    )


def SummarizeFeatureCategories(
    significant_long_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    celltype_col: str = "celltype",
    value_col: str = "score",
    pvalue_col: str = "padj",
    category_map: dict | None = None,
) -> pd.DataFrame:
    """
    Per-celltype/category summary (counts, mean/median score, dominant
    direction) from `BuildSignificantFeatureTable`'s output.

    "up"/"down" are resolved from whichever `significant_long_df["direction"]`
    values are actually present (e.g. "increased_in_ANTIB"/
    "decreased_in_ANTIB") -- not hardcoded to any specific condition name.

    category_map : optional {category: [features]}, used to compute
        fraction-of-category-significant columns.
    """
    df = significant_long_df

    if df.empty:
        return pd.DataFrame()

    directions = [d for d in df["direction"].unique() if d != "mixed"]
    direction_up = next((d for d in directions if d.startswith("increased_in_")), "increased")
    direction_down = next((d for d in directions if d.startswith("decreased_in_")), "decreased")

    def _features_with_direction(idx, direction):
        matches = df.loc[idx, "direction"].eq(direction)
        return "; ".join(sorted(df.loc[idx[matches], feature_col].unique()))

    summary = df.groupby([celltype_col, "category"]).agg(
        n_significant_features=(feature_col, "nunique"),
        n_up=("direction", lambda x: (x == direction_up).sum()),
        n_down=("direction", lambda x: (x == direction_down).sum()),
        mean_score=(value_col, "mean"),
        median_score=(value_col, "median"),
        min_padj=(pvalue_col, "min"),
        features_up=(feature_col, lambda x: _features_with_direction(x.index, direction_up)),
        features_down=(feature_col, lambda x: _features_with_direction(x.index, direction_down)),
    ).reset_index()

    summary["dominant_direction"] = np.select(
        [summary["n_up"] > summary["n_down"], summary["n_down"] > summary["n_up"]],
        [direction_up, direction_down],
        default="mixed",
    )

    if category_map is not None:
        category_size = {category: len(set(features)) for category, features in category_map.items()}

        summary["n_total_features_in_category"] = summary["category"].map(category_size).fillna(np.nan)
        summary["fraction_significant"] = summary["n_significant_features"] / summary["n_total_features_in_category"]
        summary["fraction_up"] = summary["n_up"] / summary["n_total_features_in_category"]
        summary["fraction_down"] = summary["n_down"] / summary["n_total_features_in_category"]

    return summary.sort_values([celltype_col, "dominant_direction", "category"])
