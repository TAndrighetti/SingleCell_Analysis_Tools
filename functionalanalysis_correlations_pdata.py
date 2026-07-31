"""
sctools.functionalanalysis
===========================

Feature-set / pathway activity analysis (decoupler ULM) on top of pseudobulk
differential expression results from `sctools.degs`.

Generic over the network (dc.op.hallmark, dc.op.progeny, dc.op.collectri, or
any other decoupler net) and its feature naming -- nothing here is specific
to MSigDB hallmarks.

hallmark = dc.op.hallmark(organism="mouse")
progeny = dc.op.progeny(organism="mouse")
collectri = dc.op.collectri(organism="mouse")


Typical usage order:
    PrepareULMInput()              -> optional: build one celltype's ULM input from a long-format DE table (e.g. reloaded from CSV in a separate notebook); skip this if you still have `PseudoDESeq2`'s `data` in memory
    RunULM()                       -> run ULM for ONE input against any net, + optional barplot (pass `ax=` for a grid)
    MeltActsPadjToLong()           -> combine several celltypes' stored acts/padj into one tidy table
    BuildSignificantFeatureTable() -> long-format table restricted to significant hits, + category/direction columns
    SummarizeFeatureCategories()   -> per-celltype/category summary from that table

The heatmap itself is `sctools.plots.PlotSignificanceHeatmap` -- generic
(any long-format table with an x column, a y column, a color value column,
and a p-value/padj column), not defined here since it isn't specific to
functional analysis either. Pass `restrict_to_significant=True` to only
show y-values (e.g. hallmarks) that are significant in at least one x (e.g.
celltype); its own `category_map` param takes the same `{category:
[features]}` shape as `_BuildFeatureToCategoryMap`/`BuildSignificantFeatureTable`.

`RunULM` runs on a single input, matching `sctools.degs.PseudoFeatSelection`/
`PseudoDESeq2`'s "one call per celltype, loop lives in the caller" pattern
instead of looping internally over a `pdata_by_celltype` dict. Its `data`
argument is typically `PseudoDESeq2`'s second return value -- that function
is the boundary between DEGs (`sctools.degs`) and functional analysis
(this module), when both run in the same session/notebook.

When DEGs and functional analysis run in *separate* notebooks instead (DE
results reloaded from a long-format CSV rather than kept in memory),
`PseudoDESeq2`'s `data` no longer exists as an object -- use
`PrepareULMInput` to rebuild the same 1-row-per-contrast shape from that
long table, one celltype at a time.

`_BuildFeatureToCategoryMap`'s `prefix` param strips a naming-convention
prefix (e.g. `prefix="HALLMARK_"` for MSigDB) -- leave it empty for feature
sets with no such prefix (progeny pathways, collectri TFs, ...).
`BuildSignificantFeatureTable`/`SummarizeFeatureCategories` derive their
"direction" labels (e.g. "increased_in_ANTIB") from the actual
`compare_condition` half of `contrast_name` (`"{compare_condition}.vs.
{normal_condition}"`) rather than a hardcoded "KO"/"WT" -- works for any
two-condition contrast name.

Gene profile correlation (pseudobulk expression of one target gene vs every
other gene, across an ordered set of stages, e.g. maturation/pseudotime --
not tied to functional gene sets/networks like the ULM section above, but
kept here since it's the other kind of "which genes matter" analysis run on
top of the same pseudobulk `pdata`):

Typical usage order:
    SummarizePseudobulkExpression()    -> log2(CPM+1) + per-sample/gene profile z-score for one or more genes across an ordered stage axis
    PlotPseudobulkExpression()         -> mean +/- SD profile plot (or z-score) per condition, with individual sample points
    CalculateGeneProfileCorrelations() -> correlate one target gene's stage profile against every other gene, per sample, combined per condition (Fisher z), with an analytic and a permutation p-value
    SelectTopCorrelatedGenes()         -> filter/rank that table's strongest hits for one condition

`CalculateGeneProfileCorrelations` computes its own per-condition BH-adjusted
`padj` (via the private `_AdjustPValuesBH`) -- independent from any DESeq2/
ULM significance elsewhere in the package, since this is a different
statistical test (profile correlation, not differential expression/pathway
activity).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import sparse
from scipy.stats import norm

logger = logging.getLogger(__name__)

__all__ = [
    "PrepareULMInput",
    "RunULM",
    "MeltActsPadjToLong",
    "BuildSignificantFeatureTable",
    "SummarizeFeatureCategories",
    "SummarizePseudobulkExpression",
    "PlotPseudobulkExpression",
    "CalculateGeneProfileCorrelations",
    "SelectTopCorrelatedGenes",
]

#####################################################################
#### ULM Enrichment


def PrepareULMInput(
    long_df: pd.DataFrame,
    celltype: str,
    *,
    celltype_col: str = "celltype",
    gene_col: str = "gene_id",
    value_col: str = "stat",
    contrast_name: str = "KO.vs.WT",
) -> pd.DataFrame:
    """
    Build one celltype's 1-row ULM input matrix from a long-format DE table.

    Rebuilds the same shape `PseudoDESeq2` builds in memory (`data`), for
    when DEGs and functional analysis run in separate notebooks and the DE
    results were reloaded from a long-format CSV (one row per
    (celltype, gene)) instead.

    `value_col` defaults to "stat" -- PyDESeq2's Wald statistic, the same
    column `sctools.degs.PseudoDESeq2` itself uses to build its `data`
    return value. Pass a different `value_col` if your long table's effect
    size comes from elsewhere (e.g. a different DE method/column).
    """
    sub = long_df[long_df[celltype_col] == celltype]

    if sub.empty:
        raise ValueError(
            f"celltype='{celltype}' not found in long_df['{celltype_col}']. "
            f"Available: {sorted(long_df[celltype_col].unique())}."
        )

    sub = sub.set_index(gene_col)

    return sub[[value_col]].T.rename(index={value_col: contrast_name})


def MeltActsPadjToLong(
    source,
    name_df_acts: str,
    name_df_padj: str,
    contrast_name: str,
    feature_col: str = "feature",
) -> pd.DataFrame:
    """
    Melt each celltype's 1-row (contrast) acts/padj tables into one long
    table: one row per (celltype, feature), with columns
    [celltype, feature_col, "contrast", "score", "padj"].

    `source` accepts either:
    - a `pdata_by_celltype` dict, i.e. `{celltype: {name_df_acts: hm_acts,
      name_df_padj: hm_padj}}`, built by your own per-celltype loop over
      `RunULM` (`pdata_by_celltype[celltype][name_df_acts/padj] = hm_acts/hm_padj`).
    - a single pseudobulk `pdata` AnnData with `pdata.uns[name_df_acts]`/
      `pdata.uns[name_df_padj]` already holding `{celltype: hm_acts}`/
      `{celltype: hm_padj}` dicts -- lets acts/padj live on the same object
      you already have in scope (e.g. when DEGs and functional analysis run
      in separate notebooks and only `pdata` was reloaded, not a
      `pdata_by_celltype` dict) instead of a separate loose dict.
    """
    if hasattr(source, "uns"):
        acts_by_celltype = source.uns.get(name_df_acts, {})
        padj_by_celltype = source.uns.get(name_df_padj, {})
        pdata_by_celltype = {
            celltype: {name_df_acts: acts_by_celltype.get(celltype), name_df_padj: padj_by_celltype.get(celltype)}
            for celltype in set(acts_by_celltype) | set(padj_by_celltype)
        }
    else:
        pdata_by_celltype = source

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
    celltypes and call this once per celltype, storing `hm_acts`/`hm_padj`
    yourself -- either in a `pdata_by_celltype` dict (e.g.
    `pdata_by_celltype[ct]["hallmark_acts"] = hm_acts`) or directly on a
    single `pdata` AnnData (e.g. `pdata.uns["hallmark_acts"][ct] = hm_acts`).
    See `MeltActsPadjToLong`, which accepts either, to combine what you
    stored back into one tidy table for `sctools.plots.PlotSignificanceHeatmap`. Pass
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
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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

#################################################################


#####################################################################
#### Gene profile correlation (target gene vs all genes, across stages)


def _ToDense(matrix) -> np.ndarray:
    """Convert a sparse or dense matrix into a NumPy array."""

    if sparse.issparse(matrix):
        return matrix.toarray()

    return np.asarray(matrix)


def _ProfileZScore(values: pd.Series) -> pd.Series:
    """Calculate a z-score across maturation stages."""

    values = values.astype(float)
    valid = values.notna()

    result = pd.Series(np.nan, index=values.index)

    if valid.sum() < 2:
        return result

    standard_deviation = values.loc[valid].std(ddof=0)

    if not np.isfinite(standard_deviation) or np.isclose(standard_deviation, 0):
        return result

    result.loc[valid] = (values.loc[valid] - values.loc[valid].mean()) / standard_deviation

    return result


def _AdjustPValuesBH(pvalues: Sequence[float]) -> np.ndarray:
    """Adjust p-values using the Benjamini-Hochberg method."""

    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)

    valid = np.isfinite(pvalues)

    if valid.sum() == 0:
        return adjusted

    valid_pvalues = pvalues[valid]
    order = np.argsort(valid_pvalues)

    ordered_pvalues = valid_pvalues[order]
    n_tests = len(ordered_pvalues)

    ordered_adjusted = ordered_pvalues * n_tests / np.arange(1, n_tests + 1)
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0, 1)

    valid_adjusted = np.empty(n_tests, dtype=float)
    valid_adjusted[order] = ordered_adjusted
    adjusted[valid] = valid_adjusted

    return adjusted


def SummarizePseudobulkExpression(
    pdata: AnnData,
    genes: str | Sequence[str],
    *,
    sample_col: str = "sample",
    stage_col: str = "celltype",
    condition_col: str = "condition",
    counts_layer: str = "counts",
    cell_count_col: str = "psbulk_cells",
    stage_order: Sequence[str] | None = None,
    min_cells: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate log2(CPM + 1) pseudobulk expression.

    A profile z-score is calculated independently for each biological
    sample and gene across maturation stages.
    """

    # Convert one gene name into a list.
    if isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(genes)

    if not genes:
        raise ValueError("At least one gene must be provided.")

    # Check requested genes.
    missing_genes = [gene for gene in genes if gene not in pdata.var_names]

    if missing_genes:
        raise KeyError(f"Genes not found: {missing_genes}")

    # Check required metadata.
    required_columns = [sample_col, stage_col, condition_col, cell_count_col]

    missing_columns = [column for column in required_columns if column not in pdata.obs.columns]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    if counts_layer not in pdata.layers:
        raise KeyError(f"Layer {counts_layer!r} was not found.")

    # Copy pseudobulk metadata.
    metadata = pdata.obs[required_columns].copy()
    metadata["_row"] = np.arange(pdata.n_obs)

    # Keep and order the requested stages.
    if stage_order is not None:
        stage_order = list(stage_order)

        metadata = metadata.loc[metadata[stage_col].isin(stage_order)].copy()
        metadata[stage_col] = pd.Categorical(metadata[stage_col], categories=stage_order, ordered=True)

    # Check for duplicated pseudobulks.
    duplicated = metadata.duplicated([sample_col, condition_col, stage_col], keep=False)

    if duplicated.any():
        raise ValueError(
            "More than one pseudobulk was found for the same sample, condition and maturation stage."
        )

    # Select pseudobulk counts.
    row_positions = metadata["_row"].to_numpy(dtype=int)
    count_matrix = pdata.layers[counts_layer][row_positions]

    # Calculate library sizes.
    library_sizes = np.asarray(count_matrix.sum(axis=1)).ravel().astype(float)
    metadata["library_size"] = library_sizes

    # Mark valid pseudobulks.
    metadata["included"] = metadata[cell_count_col].ge(min_cells) & metadata["library_size"].gt(0)

    # Extract counts for selected genes.
    gene_positions = pdata.var_names.get_indexer(genes)
    gene_counts = _ToDense(count_matrix[:, gene_positions]).astype(float)

    # Calculate CPM.
    cpm = np.divide(
        gene_counts,
        library_sizes[:, None],
        out=np.full(gene_counts.shape, np.nan, dtype=float),
        where=library_sizes[:, None] > 0,
    ) * 1_000_000

    # Transform expression.
    expression = np.log2(cpm + 1)

    # Create a long-format table.
    gene_tables = []

    for position, gene in enumerate(genes):
        gene_table = metadata.copy()
        gene_table["gene"] = gene
        gene_table["gene_count"] = gene_counts[:, position]
        gene_table["cpm"] = cpm[:, position]
        gene_table["expression"] = expression[:, position]
        gene_tables.append(gene_table)

    sample_table = pd.concat(gene_tables, ignore_index=True)

    # Remove expression values from invalid pseudobulks.
    sample_table.loc[~sample_table["included"], ["cpm", "expression"]] = np.nan

    # Calculate one standardized profile per sample and gene.
    sample_table["profile_zscore"] = (
        sample_table
        .groupby([sample_col, condition_col, "gene"], observed=True, sort=False)["expression"]
        .transform(_ProfileZScore)
    )

    # Summarize biological samples.
    summary_table = (
        sample_table.loc[sample_table["included"]]
        .groupby([condition_col, stage_col, "gene"], observed=True, sort=False)
        .agg(
            n_samples=(sample_col, "nunique"),
            mean_expression=("expression", "mean"),
            sd_expression=("expression", "std"),
            mean_profile_zscore=("profile_zscore", "mean"),
            sd_profile_zscore=("profile_zscore", "std"),
            total_cells=(cell_count_col, "sum"),
        )
        .reset_index()
    )

    return sample_table, summary_table


def PlotPseudobulkExpression(
    sample_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    *,
    genes: str | Sequence[str],
    conditions: str | Sequence[str] | None = None,
    sample_col: str = "sample",
    stage_col: str = "celltype",
    condition_col: str = "condition",
    stage_order: Sequence[str] | None = None,
    scale: str = "expression",
    show_samples: bool = True,
    palette: dict | None = None,
    title: str | None = None,
    out_figure: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot mean pseudobulk expression with standard-deviation error bars.

    Points represent individual biological samples.
    """

    # Convert genes into a list.
    if isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(genes)

    # Select conditions.
    if conditions is None:
        conditions = list(pd.unique(summary_table[condition_col]))
    elif isinstance(conditions, str):
        conditions = [conditions]
    else:
        conditions = list(conditions)

    # Multiple genes are plotted one condition at a time.
    if len(genes) > 1 and len(conditions) > 1:
        raise ValueError("Select only one condition when plotting multiple genes.")

    # Define stage order.
    if stage_order is None:
        stage_order = list(pd.unique(summary_table[stage_col]))
    else:
        stage_order = list(stage_order)

    # Select plotted values.
    if scale == "expression":
        sample_value = "expression"
        mean_value = "mean_expression"
        sd_value = "sd_expression"
        ylabel = "log2(CPM + 1)"
    elif scale == "zscore":
        sample_value = "profile_zscore"
        mean_value = "mean_profile_zscore"
        sd_value = "sd_profile_zscore"
        ylabel = "Profile z-score"
    else:
        raise ValueError("`scale` must be 'expression' or 'zscore'.")

    # Create the figure.
    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 5))
    else:
        figure = ax.figure

    x_positions = np.arange(len(stage_order))
    series = [(gene, condition) for gene in genes for condition in conditions]

    # Separate series slightly on the x-axis.
    if len(series) == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.08, 0.08, len(series))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Color by condition for one gene. Color by gene for multiple genes.
    color_groups = conditions if len(genes) == 1 else genes

    colors = {
        group: (
            palette[group] if palette is not None and group in palette
            else default_colors[index % len(default_colors)]
        )
        for index, group in enumerate(color_groups)
    }

    # Plot each gene-condition combination.
    for (gene, condition), offset in zip(series, offsets):
        sample_mask = (
            sample_table["gene"].eq(gene)
            & sample_table[condition_col].eq(condition)
            & sample_table["included"]
        )
        summary_mask = summary_table["gene"].eq(gene) & summary_table[condition_col].eq(condition)

        gene_samples = sample_table.loc[sample_mask]
        gene_summary = summary_table.loc[summary_mask].set_index(stage_col).reindex(stage_order)

        if gene_summary.empty or gene_summary[mean_value].isna().all():
            continue

        color_key = condition if len(genes) == 1 else gene
        color = colors[color_key]
        label = str(condition) if len(genes) == 1 else gene

        means = gene_summary[mean_value].to_numpy(dtype=float)
        standard_deviations = gene_summary[sd_value].fillna(0).to_numpy(dtype=float)

        # Plot mean +/- SD.
        ax.errorbar(
            x_positions + offset,
            means,
            yerr=standard_deviations,
            color=color,
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
            zorder=3,
        )

        # Plot individual biological samples.
        if show_samples:
            for stage_index, stage in enumerate(stage_order):
                values = gene_samples.loc[gene_samples[stage_col].eq(stage), sample_value].dropna()

                if values.empty:
                    continue

                point_offsets = np.linspace(-0.02, 0.02, len(values))

                ax.scatter(
                    x_positions[stage_index] + offset + point_offsets,
                    values,
                    color=color,
                    s=25,
                    alpha=0.7,
                    zorder=4,
                )

    # Format the plot.
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stage_order)
    ax.set_xlabel("Maturation stage")
    ax.set_ylabel(ylabel)

    if scale == "zscore":
        ax.axhline(0, color="gray", linewidth=1, alpha=0.5)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    if len(series) > 1:
        ax.legend(frameon=False)

    figure.tight_layout()

    if out_figure is not None:
        figure.savefig(out_figure, dpi=500, bbox_inches="tight")

    return figure, ax

def CalculateGeneProfileCorrelations(
    pdata: AnnData,
    target_gene: str,
    *,
    sample_col: str = "sample",
    stage_col: str = "celltype",
    condition_col: str = "condition",
    counts_layer: str = "counts",
    cell_count_col: str = "psbulk_cells",
    stage_order: Sequence[str] | None = None,
    min_cells: int = 20,
    min_cpm: float = 1,
    min_detected_stages: int = 3,
    min_stages: int = 4,
    min_samples: int = 2,
    method: str = "pearson",
    n_permutations: int = 2000,
    permutation_batch_size: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Correlate one target gene against all detected genes.

    Correlations are calculated independently in each biological sample
    across maturation stages.

    Sample correlations are combined within each condition using Fisher's z
    transformation.

    The returned p-values are:

    pvalue
        Approximate two-sided p-value based on the combined Fisher z.

    padj
        Benjamini-Hochberg adjusted p-value within each condition.

    permutation_pvalue
        Empirical two-sided p-value obtained by independently permuting the
        target-gene stage order in each biological sample.
    """

    if method not in {"pearson", "spearman"}:
        raise ValueError("`method` must be 'pearson' or 'spearman'.")

    if n_permutations < 0:
        raise ValueError("`n_permutations` cannot be negative.")

    if target_gene not in pdata.var_names:
        raise KeyError(f"{target_gene!r} was not found.")

    if counts_layer not in pdata.layers:
        raise KeyError(f"Layer {counts_layer!r} was not found.")

    required_columns = [sample_col, stage_col, condition_col, cell_count_col]
    missing_columns = [column for column in required_columns if column not in pdata.obs.columns]

    if missing_columns:
        raise KeyError(f"Columns not found: {missing_columns}")

    # Copy metadata.
    metadata = pdata.obs[required_columns].copy()
    metadata["_row"] = np.arange(pdata.n_obs)

    # Keep selected stages.
    if stage_order is not None:
        stage_order = list(stage_order)

        metadata = metadata.loc[metadata[stage_col].isin(stage_order)].copy()
        metadata[stage_col] = pd.Categorical(metadata[stage_col], categories=stage_order, ordered=True)

    # Check duplicated pseudobulks.
    duplicated = metadata.duplicated([sample_col, condition_col, stage_col], keep=False)

    if duplicated.any():
        raise ValueError(
            "More than one pseudobulk was found for the same sample, condition and maturation stage."
        )

    count_matrix = pdata.layers[counts_layer]

    # Calculate library size for every pseudobulk.
    library_sizes = np.asarray(count_matrix.sum(axis=1)).ravel().astype(float)
    row_positions = metadata["_row"].to_numpy(dtype=int)
    metadata["library_size"] = library_sizes[row_positions]

    # Keep valid pseudobulks.
    metadata = metadata.loc[
        metadata[cell_count_col].ge(min_cells) & metadata["library_size"].gt(0)
    ].copy()

    target_position = pdata.var_names.get_loc(target_gene)
    gene_names = np.asarray(pdata.var_names)
    rng = np.random.default_rng(random_state)

    sample_tables = []
    condition_tables = []

    # Analyze each condition independently.
    for condition, condition_data in metadata.groupby(condition_col, observed=True, sort=False):
        samples = []

        # Calculate one correlation profile per biological sample.
        for sample, sample_data in condition_data.groupby(sample_col, observed=True, sort=False):
            if stage_order is not None:
                sample_data = (
                    sample_data.set_index(stage_col).reindex(stage_order)
                    .dropna(subset=["_row"]).reset_index()
                )
            else:
                sample_data = sample_data.sort_values(stage_col)

            rows = sample_data["_row"].to_numpy(dtype=int)

            if len(rows) < min_stages:
                continue

            # Extract pseudobulk counts.
            sample_counts = _ToDense(count_matrix[rows]).astype(float)
            sample_library_sizes = library_sizes[rows]

            # Calculate CPM.
            sample_cpm = np.divide(
                sample_counts,
                sample_library_sizes[:, None],
                out=np.zeros_like(sample_counts, dtype=float),
                where=(sample_library_sizes[:, None] > 0),
            ) * 1_000_000

            # Calculate log2(CPM + 1).
            sample_expression = np.log2(sample_cpm + 1)

            # Check target-gene detection.
            target_detected_stages = (sample_cpm[:, target_position] >= min_cpm).sum()

            if target_detected_stages < min_detected_stages:
                continue

            # Convert values into ranks for Spearman.
            if method == "spearman":
                correlation_expression = (
                    pd.DataFrame(sample_expression).rank(axis=0, method="average").to_numpy()
                )
            else:
                correlation_expression = sample_expression

            # Center all gene profiles.
            centered_genes = correlation_expression - correlation_expression.mean(axis=0)
            gene_norms = np.sqrt(np.sum(centered_genes**2, axis=0))

            # Normalize each gene profile to unit length.
            gene_units = np.divide(
                centered_genes,
                gene_norms[None, :],
                out=np.zeros_like(centered_genes, dtype=float),
                where=gene_norms[None, :] > 0,
            )

            # Obtain the target-gene profile.
            target_values = correlation_expression[:, target_position]
            centered_target = target_values - target_values.mean()
            target_norm = np.sqrt(np.sum(centered_target**2))

            if not np.isfinite(target_norm) or np.isclose(target_norm, 0):
                continue

            target_unit = centered_target / target_norm

            # Correlate target against every gene.
            correlations = target_unit @ gene_units

            # Require gene detection in enough stages.
            detected_genes = (sample_cpm >= min_cpm).sum(axis=0) >= min_detected_stages
            valid_genes = detected_genes & (gene_norms > 0) & np.isfinite(correlations)

            # Exclude the target itself.
            valid_genes[target_position] = False
            valid_positions = np.flatnonzero(valid_genes)

            sample_table = pd.DataFrame({
                "condition": condition,
                "sample": sample,
                "gene": gene_names[valid_positions],
                "rho": correlations[valid_positions],
                "n_stages": len(rows),
            })

            sample_tables.append(sample_table)

            samples.append({
                "sample": sample,
                "rho": correlations,
                "valid": valid_genes,
                "n_stages": len(rows),
                "weight": len(rows) - 3,
                "target_unit": target_unit,
                "gene_units": gene_units,
            })

        if len(samples) < min_samples:
            warnings.warn(
                f"Condition {condition!r} did not have enough valid samples for "
                f"{target_gene!r} correlations."
            )
            continue

        # Collect sample-level correlations.
        rho_matrix = np.vstack([sample["rho"] for sample in samples])
        valid_matrix = np.vstack([sample["valid"] for sample in samples])
        weights = np.asarray([sample["weight"] for sample in samples], dtype=float)

        valid_sample_count = valid_matrix.sum(axis=0)
        keep_genes = valid_sample_count >= min_samples
        keep_positions = np.flatnonzero(keep_genes)

        if len(keep_positions) == 0:
            continue

        # Fisher-z transformation.
        clipped_rho = np.clip(rho_matrix, -0.999999, 0.999999)
        fisher_z = np.arctanh(clipped_rho)

        weight_matrix = valid_matrix * weights[:, None]
        weight_sum = weight_matrix.sum(axis=0)

        combined_z = np.divide(
            np.sum(np.where(valid_matrix, fisher_z * weights[:, None], 0), axis=0),
            weight_sum,
            out=np.full(pdata.n_vars, np.nan),
            where=weight_sum > 0,
        )

        combined_rho = np.tanh(combined_z)

        # Approximate Fisher-z p-value.
        standard_error = np.divide(
            1, np.sqrt(weight_sum),
            out=np.full(pdata.n_vars, np.nan),
            where=weight_sum > 0,
        )

        z_statistic = np.divide(
            combined_z, standard_error,
            out=np.full(pdata.n_vars, np.nan),
            where=standard_error > 0,
        )

        pvalues = 2 * norm.sf(np.abs(z_statistic))

        # Calculate empirical permutation p-values.
        permutation_pvalues = np.full(pdata.n_vars, np.nan)

        if n_permutations > 0:
            observed_absolute = np.abs(combined_rho[keep_positions])
            extreme_counts = np.zeros(len(keep_positions), dtype=int)
            keep_weight_sum = weight_sum[keep_positions]

            completed = 0

            while completed < n_permutations:
                current_batch = min(permutation_batch_size, n_permutations - completed)

                permutation_z_sum = np.zeros((current_batch, len(keep_positions)), dtype=float)

                for sample in samples:
                    n_sample_stages = sample["n_stages"]

                    # Generate independent random stage permutations.
                    permutation_indices = np.argsort(
                        rng.random((current_batch, n_sample_stages)), axis=1
                    )

                    permuted_target = sample["target_unit"][permutation_indices]
                    permuted_rho = permuted_target @ sample["gene_units"][:, keep_positions]

                    sample_valid = sample["valid"][keep_positions]

                    if not sample_valid.any():
                        continue

                    permutation_z_sum[:, sample_valid] += sample["weight"] * np.arctanh(
                        np.clip(permuted_rho[:, sample_valid], -0.999999, 0.999999)
                    )

                permutation_combined_rho = np.tanh(permutation_z_sum / keep_weight_sum[None, :])

                extreme_counts += np.sum(
                    np.abs(permutation_combined_rho) >= observed_absolute[None, :], axis=0
                )

                completed += current_batch

            permutation_pvalues[keep_positions] = (extreme_counts + 1) / (n_permutations + 1)

        # Sample-level descriptive statistics.
        rho_valid = np.where(valid_matrix, rho_matrix, np.nan)

        mean_rho = np.nanmean(rho_valid, axis=0)
        min_rho = np.nanmin(rho_valid, axis=0)
        max_rho = np.nanmax(rho_valid, axis=0)
        min_abs_rho = np.nanmin(np.abs(rho_valid), axis=0)

        all_positive = np.all(np.where(valid_matrix, rho_matrix > 0, True), axis=0)
        all_negative = np.all(np.where(valid_matrix, rho_matrix < 0, True), axis=0)
        direction_consistent = all_positive | all_negative

        condition_table = pd.DataFrame({
            "condition": condition,
            "gene": gene_names[keep_positions],
            "n_samples": valid_sample_count[keep_positions],
            "combined_rho": combined_rho[keep_positions],
            "mean_rho": mean_rho[keep_positions],
            "min_rho": min_rho[keep_positions],
            "max_rho": max_rho[keep_positions],
            "min_abs_rho": min_abs_rho[keep_positions],
            "direction_consistent": direction_consistent[keep_positions],
            "pvalue": pvalues[keep_positions],
            "permutation_pvalue": permutation_pvalues[keep_positions],
        })

        condition_tables.append(condition_table)

    if not sample_tables:
        raise ValueError("No valid sample-level correlations were calculated.")

    if not condition_tables:
        raise ValueError("No condition had enough valid correlations.")

    sample_correlations = pd.concat(sample_tables, ignore_index=True)
    correlation_table = pd.concat(condition_tables, ignore_index=True)

    # Adjust p-values independently in each condition.
    correlation_table["padj"] = (
        correlation_table.groupby("condition", observed=True)["pvalue"].transform(_AdjustPValuesBH)
    )

    correlation_table["abs_rho"] = correlation_table["combined_rho"].abs()
    correlation_table["direction"] = np.where(correlation_table["combined_rho"] >= 0, "positive", "negative")

    # Rank consistent and stronger correlations first.
    correlation_table = correlation_table.sort_values(
        ["condition", "direction_consistent", "abs_rho"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return sample_correlations, correlation_table


def SelectTopCorrelatedGenes(
    correlation_table: pd.DataFrame,
    condition: str,
    *,
    n_genes: int = 5,
    require_consistent: bool = True,
    max_padj: float | None = None,
) -> pd.DataFrame:
    """
    Select genes with the strongest absolute correlations.

    Positive and negative correlations are both included.
    """

    result = correlation_table.loc[correlation_table["condition"].eq(condition)].copy()

    if require_consistent:
        result = result.loc[result["direction_consistent"]]

    if max_padj is not None:
        result = result.loc[result["padj"] <= max_padj]

    return result.sort_values("abs_rho", ascending=False).head(n_genes).reset_index(drop=True)