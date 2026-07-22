"""
sctools.functionalanalysis
===========================

Feature-set / pathway activity analysis (decoupler ULM) on top of pseudobulk
differential expression results from `sctools.degs`.

Generic over the network (dc.op.hallmark, dc.op.progeny, dc.op.collectri, or
any other decoupler net) and its feature naming -- nothing here is specific
to MSigDB hallmarks.

Typical usage order:
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
(this module).

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "RunULM",
    "MeltActsPadjToLong",
    "BuildSignificantFeatureTable",
    "SummarizeFeatureCategories",
]


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
