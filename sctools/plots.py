"""
sctools.plots
=============
Plotting utilities for single-cell RNA-seq analysis.

Functions
---------
PlotHeatmap             – generic heatmap from a long-format DataFrame
PlotSignificanceHeatmap – generic value+significance heatmap (score/log2FC
                          colored, "*" by p-value/padj threshold, optional
                          category sidebar) from a long-format DataFrame
plot_qc_violins       – violin grid per QC metric, grouped by sample
plot_metric_pairs     – paired boxplot comparing two AnnData sets (before/after)
plot_covariates       – 4-panel covariate overview (violin × 3 + scatter)
plot_doublet_umap     – UMAP colored by Scrublet doublet score and prediction
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


# ── Generic heatmap ───────────────────────────────────────────────────────────

def PlotHeatmap(
    df,
    *,
    x: str,
    y: str,
    color_col: str,
    annotation_col: str | None = None,
    annotation_fmt: str = "{:,}",
    title: str = "",
    figsize: tuple[float, float] = (7, 5),
    cmap: str = "RdYlGn_r",
    colorbar_label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    ax=None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> tuple:
    """Generic heatmap from a long-format DataFrame.

    Parameters
    ----------
    df : long-format DataFrame (one row per combination of x/y values).
    x, y : columns used as heatmap axes.
    color_col : column mapped to cell colour.
    annotation_col : column displayed as text inside each cell. ``None`` → no text.
    annotation_fmt : Python format string applied to annotation values, e.g.
        ``"{:,}"`` (integer with thousands sep), ``"{:.1f}%"`` (percentage).
    title : plot title.
    figsize : figure size — ignored when ``ax`` is provided.
    cmap : matplotlib colormap name.
    colorbar_label : label for the colourbar — ignored when ``ax`` is provided.
    vmin, vmax : colour-scale limits. ``None`` uses data min/max.
    ax : existing Axes to draw on. If ``None``, a new figure is created.
    show : display the figure — ignored when ``ax`` is provided.
    save_path : path to save the figure — ignored when ``ax`` is provided.

    Returns
    -------
    fig, ax, im
    """
    pivot_color = df.pivot_table(index=y, columns=x, values=color_col, aggfunc="mean")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    _vmin = vmin if vmin is not None else 0
    _vmax = vmax if vmax is not None else np.nanmax(pivot_color.values)
    im = ax.imshow(pivot_color.values, cmap=cmap, aspect="auto", vmin=_vmin, vmax=_vmax)

    ax.set_xticks(np.arange(len(pivot_color.columns)))
    ax.set_xticklabels(pivot_color.columns)
    ax.set_yticks(np.arange(len(pivot_color.index)))
    ax.set_yticklabels(pivot_color.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)

    ax.set_xticks(np.arange(-0.5, len(pivot_color.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot_color.index),   1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotation_col is not None:
        pivot_annot = df.pivot_table(index=y, columns=x, values=annotation_col, aggfunc="mean")
        for i in range(len(pivot_color.index)):
            for j in range(len(pivot_color.columns)):
                val = pivot_color.iloc[i, j]
                if np.isnan(val):
                    text, text_color = "NA", "black"
                else:
                    ann = pivot_annot.iloc[i, j]
                    text = annotation_fmt.format(ann) if not np.isnan(ann) else "NA"
                    r, g, b, _ = im.cmap(im.norm(val))
                    text_color = "white" if (0.299*r + 0.587*g + 0.114*b) < 0.45 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=12, color=text_color)

    if own_fig:
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        if colorbar_label:
            cbar.set_label(colorbar_label)

        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        plt.close(fig)

    return fig, ax, im


def PlotSignificanceHeatmap(
    df,
    *,
    x: str,
    y: str,
    value_col: str,
    pvalue_col: str,
    padj_threshold: float = 0.05,
    restrict_to_significant: bool = False,
    order_x: list | None = None,
    order_y: list | None = None,
    category_map: dict | None = None,
    sort_by_magnitude: bool = True,
    title: str = "",
    figsize: tuple[float, float] = (10, 6),
    cmap: str = "RdBu_r",
    center: float = 0,
    xlabel: str | None = None,
    ylabel: str = "",
    cbar_label: str | None = None,
    category_legend_title: str = "Category",
    plot: bool = True,
    save: str | Path | None = None,
) -> tuple:
    """Generic significance heatmap from a long-format DataFrame.

    Pivots `df` into a `y`-by-`x` matrix colored by `value_col` (e.g. log2FC,
    a module/pathway activity score), with a "*" wherever `pvalue_col` is
    below `padj_threshold`. `pvalue_col` can point at either a raw p-value
    or an already-adjusted one -- whichever column you pass is what gets
    thresholded, this function doesn't correct it for you.

    Row (`y`) order is resolved in this priority: `order_y` if given: use it
    as-is; else `category_map` if given: group rows by category (in the
    dict's iteration order), sorting each group by `sort_by_magnitude` if
    set; else just `sort_by_magnitude` alone, or insertion order.
    `category_map` additionally draws a colored sidebar + legend grouping
    the rows, independent of whether it also drove the ordering (pass
    `order_y` alongside it to keep sidebar colors but control order
    yourself).

    Parameters
    ----------
    df : long-format DataFrame -- one row per (x, y) combination.
    x, y : columns used as the heatmap's column/row axes.
    value_col : column mapped to cell color.
    pvalue_col : column thresholded against `padj_threshold` for the "*" annotation.
    restrict_to_significant : drop `y` values that aren't below
        `padj_threshold` for at least one `x` -- e.g. only show
        hallmarks/pathways/TFs significant in at least one celltype.
    order_x, order_y : optional explicit axis orders (values not present are dropped).
    category_map : optional {category: [y values]} -- draws a left-side
        color bar grouping rows, and (unless `order_y` is given) also
        drives row order.
    sort_by_magnitude : sort rows (or each category's rows) by summed |value_col|.
    title, figsize, cmap, center, xlabel, ylabel : passed through to the plot.
    cbar_label : colorbar label -- defaults to `value_col`.
    category_legend_title : legend title for the category sidebar.
    plot : whether to render the figure. Set False to just get the pivoted tables back.
    save : path to save the figure to. `None` skips saving.

    Returns
    -------
    value_table, padj_table, annot_table : the pivoted, ordered (y-by-x)
        DataFrames -- `annot_table` is "*"/"" by `padj_threshold`. Empty
        DataFrames if nothing is left to plot (e.g. `restrict_to_significant`
        dropped everything).
    """
    if restrict_to_significant:
        significant_rows = df.loc[df[pvalue_col] < padj_threshold, y].unique()
        df = df[df[y].isin(significant_rows)]

    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    value_table = df.pivot(index=y, columns=x, values=value_col)
    padj_table = df.pivot(index=y, columns=x, values=pvalue_col)

    if order_x is not None:
        cols = [c for c in order_x if c in value_table.columns]
        value_table = value_table[cols]
        padj_table = padj_table[cols]

    row_to_category = {}
    if category_map is not None:
        for category, items in category_map.items():
            for item in items:
                row_to_category[item] = category

    if order_y is not None:
        rows = [r for r in order_y if r in value_table.index]
    elif category_map is not None:
        def _sorted_within(items):
            items = [item for item in items if item in value_table.index]
            if not sort_by_magnitude:
                return items
            return (
                value_table.loc[items]
                .abs().sum(axis=1)
                .sort_values(ascending=False)
                .index.tolist()
            )

        ordered = []
        for items in category_map.values():
            ordered.extend(_sorted_within(items))

        remaining = _sorted_within([r for r in value_table.index if r not in ordered])
        rows = ordered + remaining
    elif sort_by_magnitude:
        rows = value_table.abs().sum(axis=1).sort_values(ascending=False).index.tolist()
    else:
        rows = list(value_table.index)

    value_table = value_table.loc[rows]
    padj_table = padj_table.loc[rows]
    annot_table = (padj_table < padj_threshold).replace({True: "*", False: ""})

    if plot:
        import seaborn as sns
        from matplotlib.patches import Patch, Rectangle

        plt.figure(figsize=figsize)
        ax = plt.gca()
        sns.heatmap(
            value_table,
            ax=ax,
            cmap=cmap,
            center=center,
            linewidths=0.4,
            linecolor="white",
            cbar_kws={"label": cbar_label or value_col},
            annot=annot_table,
            fmt="",
        )

        plt.title(title, fontsize=14, weight="bold", pad=12)
        plt.xlabel(xlabel if xlabel is not None else x)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        if row_to_category:
            categories = [row_to_category.get(r, "uncategorized") for r in value_table.index]
            unique_categories = list(dict.fromkeys(categories))
            palette = sns.color_palette("tab20", n_colors=len(unique_categories))
            category_colors = dict(zip(unique_categories, palette))

            bar_width = 0.25
            for pos, category in enumerate(categories):
                ax.add_patch(
                    Rectangle(
                        (-bar_width, pos), bar_width, 1,
                        color=category_colors[category],
                        clip_on=False, linewidth=0,
                    )
                )
            ax.set_xlim(-bar_width, value_table.shape[1])

            legend_handles = [
                Patch(color=category_colors[cat], label=cat)
                for cat in unique_categories
            ]
            ax.legend(
                handles=legend_handles,
                title=category_legend_title,
                bbox_to_anchor=(1.35, 1),
                loc="upper left",
                frameon=False,
            )

        plt.tight_layout()

        if save is not None:
            Path(save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save, dpi=300, bbox_inches="tight")

        plt.show()

    return value_table, padj_table, annot_table


# ── QC violins ────────────────────────────────────────────────────────────────

def plot_qc_violins(
    adata: AnnData,
    metrics: list[str] | None = None,
    *,
    title: str = "Quality Control Metrics",
    sample_key: str = "sample",
    samples: list[str] | str | None = None,
    save_dir: str | Path | None = None,
    suffix: str = "",
    show_plot: bool = True,
    jitter: float = 0.3,
    point_size: float = 0.2,
    rotation: int = 0,
    dpi: int = 200,
    figsize: tuple[float, float] = (15, 4),
    ymin: float | dict | None = 0,
    ymax: float | dict | None = None,
) -> plt.Figure:
    """Violin grid with one panel per QC metric, grouped by sample.

    Parameters
    ----------
    adata :
        AnnData with QC columns in ``adata.obs``. Run
        :func:`~sctools.qc.calculate_qc_metrics` first.
    metrics :
        List of ``adata.obs`` columns to plot. Defaults to
        ``["n_genes_by_counts", "total_counts", "pct_counts_mt"]``.
    title :
        Figure title.
    sample_key :
        Column in ``adata.obs`` used to group violins.
    samples :
        Subset of samples to plot. ``None`` plots all.
    save_dir :
        Directory to save the figure. ``None`` skips saving.
    suffix :
        Appended to the filename when saving.
    show_plot :
        Whether to display the figure.
    jitter, point_size :
        Strip plot aesthetics.
    rotation :
        X-tick label rotation in degrees.
    dpi :
        Resolution for saved figure.
    figsize :
        Figure size ``(width, height)`` in inches.
    ymin, ymax :
        Y-axis limits. Can be a single value (applied to all panels) or a
        ``{metric: value}`` dict for per-panel control.
    """
    if metrics is None:
        metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]

    if sample_key not in adata.obs:
        raise KeyError(f"'{sample_key}' not found in adata.obs.")
    missing = [m for m in metrics if m not in adata.obs.columns]
    if missing:
        raise KeyError(f"Columns missing from adata.obs: {missing}")

    if isinstance(samples, str):
        samples = [samples]

    ad = adata if samples is None else adata[adata.obs[sample_key].isin(samples)].copy()
    ad.uns.pop(f"{sample_key}_colors", None)
    order = sorted(ad.obs[sample_key].unique(), key=str)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        sc.pl.violin(
            ad,
            keys=metric,
            groupby=sample_key,
            order=order,
            stripplot=True,
            jitter=jitter,
            size=point_size,
            rotation=0,
            show=False,
            ax=ax,
        )
        ax.set_ylabel(metric)
        ax.set_xlabel("Sample")

        _ymin = ymin.get(metric, 0) if isinstance(ymin, dict) else ymin
        _ymax = ymax.get(metric, None) if isinstance(ymax, dict) else ymax
        cur_min, cur_max = ax.get_ylim()
        ax.set_ylim(
            _ymin if _ymin is not None else cur_min,
            _ymax if _ymax is not None else cur_max,
        )
        for tick in ax.get_xticklabels():
            tick.set_rotation(rotation)
            tick.set_ha("right")
        ax.grid(True, which="major", axis="y", alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_dir:
        fig.savefig(Path(save_dir, f"QC_violins_{suffix}.png"), dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

    return fig


# ── Paired boxplot ─────────────────────────────────────────────────────────────

def plot_metric_pairs(
    adatas_before: list[AnnData],
    adatas_after: list[AnnData],
    sample_names: list[str],
    *,
    metric: str = "n_genes_by_counts",
    labels: tuple[str, str] = ("before", "after"),
    palette: tuple[str, str] = ("green", "purple"),
    show_plot: bool = True,
    save_path: str | Path | None = None,
    group_gap: float = 2.6,
    pair_offset: float = 0.35,
    box_width: float = 0.6,
    annot_fs: int = 8,
    median_fs: int = 9,
    y_top_pad: float = 0.12,
) -> tuple[plt.Figure, plt.Axes]:
    """Paired boxplot comparing a QC metric before and after filtering.

    Parameters
    ----------
    adatas_before, adatas_after :
        Lists of AnnData objects (one per sample) for each condition.
    sample_names :
        Sample labels for the X axis.
    metric :
        ``adata.obs`` column to compare.
    labels :
        Legend labels for the two groups.
    palette :
        Box fill colours for each group.
    show_plot :
        Whether to display the figure.
    save_path :
        Full file path to save the figure. ``None`` skips saving.
    group_gap, pair_offset, box_width :
        Layout parameters for box positioning.
    annot_fs, median_fs :
        Font sizes for annotations and median labels.
    y_top_pad :
        Fraction of extra space above the highest whisker.
    """
    from matplotlib.patches import Patch

    assert len(adatas_before) == len(adatas_after) == len(sample_names)

    def _vals(ad: AnnData) -> np.ndarray:
        if metric not in ad.obs:
            raise KeyError(f"'{metric}' not found in adata.obs.")
        arr = ad.obs[metric].to_numpy()
        return arr[np.isfinite(arr)]

    data_a = [_vals(ad) for ad in adatas_before]
    data_b = [_vals(ad) for ad in adatas_after]

    fig, ax = plt.subplots(figsize=(14, 8))
    centers   = np.arange(len(sample_names)) * group_gap
    pos_a     = centers - pair_offset
    pos_b     = centers + pair_offset

    bp_a = ax.boxplot(data_a, positions=pos_a, widths=box_width,
                      patch_artist=True, showfliers=False, whis=1.5)
    bp_b = ax.boxplot(data_b, positions=pos_b, widths=box_width,
                      patch_artist=True, showfliers=False, whis=1.5)

    for p in bp_a["boxes"]:
        p.set_facecolor(palette[0]); p.set_alpha(0.6)
    for p in bp_b["boxes"]:
        p.set_facecolor(palette[1]); p.set_alpha(0.6)
    for bp in (bp_a, bp_b):
        for k in ("medians", "whiskers", "caps"):
            for line in bp[k]:
                line.set_linewidth(1.5)

    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    vpad_hi_small = 0.012 * yr
    vpad_hi_big   = 0.035 * yr
    vpad_q1_down  = 0.016 * yr
    vpad_lo       = 0.025 * yr
    vpad_med      = 0.015 * yr

    def _tukey_stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return dict(n=0, q1=np.nan, med=np.nan, q3=np.nan, lo=np.nan, hi=np.nan)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        low_b, high_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lo = np.min(arr[arr >= low_b]) if np.any(arr >= low_b) else np.min(arr)
        hi = np.max(arr[arr <= high_b]) if np.any(arr <= high_b) else np.max(arr)
        return dict(n=len(arr), q1=q1, med=med, q3=q3, lo=lo, hi=hi)

    def _annotate(xs, datasets):
        for x, arr in zip(xs, datasets):
            st = _tukey_stats(arr)
            if st["n"] == 0:
                continue
            ax.text(x, st["hi"] + vpad_hi_small,              f"hi={int(round(st['hi']))}",  ha="center", va="bottom", fontsize=annot_fs)
            ax.text(x, st["hi"] + vpad_hi_small + vpad_hi_big, f"n={st['n']}",               ha="center", va="bottom", fontsize=annot_fs)
            ax.text(x, st["med"] + vpad_med,                   f"med={int(round(st['med']))}", ha="center", va="bottom", fontsize=median_fs, fontweight="bold")
            ax.text(x, st["q1"] - vpad_q1_down,               f"Q1={int(round(st['q1']))}",  ha="center", va="top",    fontsize=annot_fs)
            ax.text(x, st["q3"],                               f"Q3={int(round(st['q3']))}",  ha="center", va="bottom", fontsize=annot_fs)
            ax.text(x, st["lo"] - vpad_lo,                    f"lo={int(round(st['lo']))}",  ha="center", va="top",    fontsize=annot_fs)

    _annotate(pos_a, data_a)
    _annotate(pos_b, data_b)

    def _whisker_top(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return 0.0
        q1, q3 = np.percentile(arr, [25, 75])
        high_b = q3 + 1.5 * (q3 - q1)
        return float(np.max(arr[arr <= high_b]) if np.any(arr <= high_b) else np.max(arr))

    all_hi = [_whisker_top(a) for a in data_a + data_b if len(a)]
    if all_hi:
        cur_min, _ = ax.get_ylim()
        ax.set_ylim(cur_min, max(all_hi) * (1.0 + y_top_pad))

    ax.set_xlim(centers.min() - 1, centers.max() + 1)
    ax.set_xticks(centers)
    ax.set_xticklabels(sample_names)
    ax.set_ylabel(metric)
    ax.legend(
        handles=[
            Patch(facecolor=palette[0], alpha=0.6, label=labels[0]),
            Patch(facecolor=palette[1], alpha=0.6, label=labels[1]),
        ],
        frameon=False, ncols=2, loc="upper right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

    return fig, ax


# ── Covariate overview ─────────────────────────────────────────────────────────

def plot_covariates(
    adata: AnnData,
    label: str,
    *,
    show_plot: bool = False,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """4-panel covariate overview: total_counts, n_genes, pct_mt violins + scatter.

    Parameters
    ----------
    adata :
        AnnData with QC columns in ``adata.obs``.
    label :
        Title prefix for each panel (e.g. sample name or pipeline step).
    show_plot :
        Whether to display the figure.
    save_path :
        Full file path to save the figure. ``None`` skips saving.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    sc.pl.violin(adata, "total_counts",     ax=axes[0], show=False, jitter=0.4)
    axes[0].set_title(f"{label}\nTotal Counts")

    sc.pl.violin(adata, "n_genes_by_counts", ax=axes[1], show=False, jitter=0.4)
    ymin = adata.obs["n_genes_by_counts"].min()
    axes[1].set_ylim(ymin * 0.95, None)
    axes[1].set_title(f"{label}\nN Genes by Counts")

    sc.pl.violin(adata, "pct_counts_mt",    ax=axes[2], show=False, jitter=0.4)
    axes[2].set_title(f"{label}\nMitochondrial Counts (%)")

    # Scatter: total_counts vs n_genes colored by pct_counts_mt
    obs = adata.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].dropna()
    sc_plot = axes[3].scatter(
        obs["total_counts"],
        obs["n_genes_by_counts"],
        c=obs["pct_counts_mt"],
        cmap="viridis",
        s=1,
        alpha=0.5,
    )
    plt.colorbar(sc_plot, ax=axes[3], label="pct_counts_mt")
    axes[3].set_xlabel("total_counts")
    axes[3].set_ylabel("n_genes_by_counts")
    axes[3].set_title(f"{label}\nTotal Counts vs Genes\nvs Pct MT")

    plt.subplots_adjust(wspace=0.4)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    if show_plot:
        plt.show()
    plt.close(fig)

    return fig


# ── Doublet UMAP ───────────────────────────────────────────────────────────────

def plot_doublet_umap(
    adata: AnnData,
    *,
    threshold: float = 0.25,
    show_plot: bool = True,
    save_dir: str | Path | None = None,
) -> None:
    """UMAP colored by Scrublet doublet score and binary prediction.

    Computes a temporary UMAP on a normalized copy of ``adata``
    (does not modify the original). Requires ``doublet_score`` in
    ``adata.obs`` — run :func:`~sctools.qc.score_doublets` first.

    Parameters
    ----------
    adata :
        AnnData with ``doublet_score`` in ``adata.obs``.
    threshold :
        Score cutoff for the binary prediction panel.
    show_plot :
        Whether to display the figures.
    save_dir :
        Directory to save both figures. ``None`` skips saving.
    """
    if "doublet_score" not in adata.obs:
        raise ValueError("'doublet_score' not found in adata.obs. Run score_doublets first.")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    ad = adata.copy()
    ad.obs["predicted_doublet_plot"] = (ad.obs["doublet_score"] >= threshold).astype(str)

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad)
    ad = ad[:, ad.var["highly_variable"]].copy()
    sc.pp.scale(ad)
    sc.tl.pca(ad)
    sc.pp.neighbors(ad)
    sc.tl.umap(ad)

    fig1 = sc.pl.umap(ad, color="doublet_score", cmap="viridis",
                      title="Doublet score", show=show_plot, return_fig=True)
    if save_dir:
        fig1.savefig(Path(save_dir, f"umap_doublet_score_thr{threshold}.png"),
                     dpi=300, bbox_inches="tight")
    if not show_plot:
        plt.close(fig1)

    fig2 = sc.pl.umap(ad, color="predicted_doublet_plot",
                      palette={"False": "#2ca02c", "True": "#d62728"},
                      title=f"Predicted doublets (threshold={threshold})",
                      show=show_plot, return_fig=True)
    if save_dir:
        fig2.savefig(Path(save_dir, f"umap_doublet_binary_thr{threshold}.png"),
                     dpi=300, bbox_inches="tight")
    if not show_plot:
        plt.close(fig2)
