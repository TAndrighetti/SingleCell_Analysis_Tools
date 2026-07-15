"""
sctools.integration.benchmarking
==================================

Grid-search benchmarking of integration methods, scored with scIB metrics.

Minimal usage
-------------
    combinations = BuildBenchmarkGrid(flavors=["seurat"], n_top_genes_list=[2000], n_pcs_list=[35])
    all_metrics, params, ranking = RunIntegrationBenchmark(
        adata, combinations,
        out_dir="results/04_integration", out_prefix="scNeu",
        batch_key="sample", counts_layer="soupX_counts", log_layer="soupX_counts_log1p",
        organism="mouse", methods=("harmony", "scanorama", "scvi"),
    )

Only embedding-based methods (see `methods.SUPPORTED_METHODS`) are scored
here -- BBKNN is currently retired (see `sctools/non_used_functions.py`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData

from sctools.preprocessing import RunHighlyVariableGenes, RunPcaOnHvgs

from .methods import SCIB_EMBED_BY_METHOD, SUPPORTED_METHODS, _METHOD_FUNCS, FilterSupportedKwargs

logger = logging.getLogger(__name__)

TAB_SCIB_SUBDIR = "tab_scib"

# Canonical scIB metric groups (Luecken et al. 2022): batch-correction vs
# bio-conservation. Used as SummarizeScibBenchmarkResults' defaults so both
# label-free and labeled runs (RunScibMetricsLabelFree vs RunScibMetricsWithLabel)
# get the right metrics automatically -- each run's table only ever has the
# columns it actually computed, and downstream filtering already drops the rest.
BATCH_METRICS = ("PCR_batch", "iLISI", "ASW_label/batch", "graph_conn", "kBET")
BIO_METRICS = (
    "cell_cycle_conservation", "hvg_overlap", "NMI_cluster/label", "ARI_cluster/label",
    "ASW_label", "isolated_label_F1", "isolated_label_silhouette", "cLISI",
)


def _Prefixed(name: str, out_prefix: str | None) -> str:
    """Build a filename, adding "{out_prefix}." only if out_prefix is given."""
    return f"{out_prefix}.{name}" if out_prefix else name


def _LoadScibRunMetrics(
    metrics_dir: str | Path,
    out_prefix: str | None = None,
    run_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate one source's per-run scIB metric CSVs into a single
    table (rows = method_run combinations, columns = metrics).

    Reads `{metrics_dir}/[{out_prefix}.]scib_metrics_run{run_id}.csv`
    (tab-separated despite the `.csv` extension). If `run_ids` is None,
    discovers them by globbing instead of assuming a fixed run count.
    """
    metrics_dir = Path(metrics_dir)
    run_glob = _Prefixed("scib_metrics_run*.csv", out_prefix)

    if run_ids is None:
        # `glob` matches the files but doesn't sort them numerically (run10
        # would sort before run2 as plain strings), so build a regex that
        # pulls the run_id digits out of each filename and use that captured
        # number as the sort key below.
        pattern = re.compile(rf"{re.escape(out_prefix) + '.' if out_prefix else ''}scib_metrics_run(\d+)\.csv$")
        run_files = sorted(metrics_dir.glob(run_glob), key=lambda p: int(pattern.search(p.name).group(1)))
    else:
        run_files = [metrics_dir / _Prefixed(f"scib_metrics_run{run_id}.csv", out_prefix) for run_id in run_ids]

    if not run_files:
        raise FileNotFoundError(f"No metrics files found matching {run_glob} in {metrics_dir}.")

    metric_tables = [pd.read_csv(path, sep="\t", index_col=0) for path in run_files]
    return pd.concat(metric_tables, axis=1).T


def _LoadScibParams(
    metrics_dir: str | Path,
    out_prefix: str | None = None,
    params_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load one source's `scib_params_all_runs.csv`, one row per run_id."""
    if params_file is None:
        params_file = Path(metrics_dir) / _Prefixed("scib_params_all_runs.csv", out_prefix)
    # The saved params CSV has one row per method x run; drop_duplicates
    # keeps one row per run_id since params don't vary by method.
    return (
        pd.read_csv(params_file, sep="\t")
        .drop(columns=["method", "col"], errors="ignore")
        .drop_duplicates(subset="run_id")
    )


def BuildBenchmarkGrid(
    flavors: Iterable[str],
    n_top_genes_list: Iterable[int],
    n_pcs_list: Iterable[int],
) -> dict:
    """
    Build the grid-search combinations (flavor x n_top_genes x n_pcs) used by
    `RunIntegrationBenchmark`: {run_id: {"flavor", "n_top_genes", "n_pcs"}}.
    """
    grid = {}
    run_id = 1
    for flavor in flavors:
        for n_top_genes in n_top_genes_list:
            for n_pcs in n_pcs_list:
                grid[run_id] = {"flavor": flavor, "n_top_genes": n_top_genes, "n_pcs": n_pcs}
                run_id += 1
    return grid


def RunScibMetricsLabelFree(
    adata_ref: AnnData,
    adata_int: AnnData,
    organism: str,
    *,
    batch_key: str = "sample",
    embed: str = "X_pca",
):
    """
    scIB metrics that need no biological label: PCR_batch, iLISI, HVG
    overlap, cell-cycle conservation. Use when no cell-type label exists yet
    (the common case before integration).

    Assumes an embedding-based integration method (type_="embed" is
    hardcoded below) -- this module doesn't currently support graph-based
    methods like BBKNN (see `sctools/non_used_functions.py`).
    """
    import scib as sciblib

    # hvg_overlap (called both here in the fallback and inside
    # scib.metrics.metrics() above) requires batch_key to be a categorical
    # column -- it reads adata.obs[batch_key].cat.categories internally and
    # raises AttributeError on a plain string/object column.
    adata_ref.obs[batch_key] = adata_ref.obs[batch_key].astype("category")
    adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")

    # scib reads `embed` only from adata_int (the unintegrated adata_ref
    # always uses its own existing X_pca internally) -- except ilisi_graph,
    # which internally defaults to reading adata_int.obsm["X_emb"] regardless
    # of `embed`. Alias it so iLISI still scores the right representation.
    adata_int.obsm["X_emb"] = adata_int.obsm[embed]

    try:
        return sciblib.metrics.metrics(
            adata_ref, adata_int, batch_key=batch_key, label_key=None, embed=embed, organism=organism,
            pcr_=True, ilisi_=True, hvg_score_=True, cell_cycle_=True,
            silhouette_=False, isolated_labels_=False, nmi_=False, ari_=False, graph_conn_=False,
            type_="embed", verbose=False,
        )
    except (TypeError, KeyError, ValueError):
        # scib.metrics.metrics() always validates label_key against adata.obs
        # (even when no label-dependent metric is requested), so label_key=None
        # reliably fails here. Fall back to the label-free metric functions.
        logger.info("scib.metrics.metrics() rejected label_key=None; using individual metric functions.")
        return pd.Series(
            {
                "PCR_batch": sciblib.metrics.pcr_comparison(adata_ref, adata_int, covariate=batch_key, embed=embed, verbose=False),
                "iLISI": sciblib.metrics.ilisi_graph(adata_int, batch_key=batch_key, type_="embed", use_rep=embed),
                "hvg_overlap": sciblib.metrics.hvg_overlap(adata_ref, adata_int, batch_key=batch_key),
                "cell_cycle_conservation": sciblib.metrics.cell_cycle(
                    adata_ref, adata_int, batch_key=batch_key, embed=embed, organism=organism, verbose=False,
                ),
            }
        )


def RunScibMetricsWithLabel(
    adata_ref: AnnData,
    adata_int: AnnData,
    organism: str,
    label_key: str,
    *,
    batch_key: str = "sample",
    embed: str = "X_pca",
):
    """
    scIB metrics using a real biological label (`label_key`): adds ASW, NMI,
    ARI, isolated-labels, and graph-connectivity on top of the label-free
    metrics -- all genuine signals here, since `label_key` is an independent
    ground truth, not derived from the integration being scored.

    Assumes an embedding-based integration method (type_="embed" is
    hardcoded below) -- this module doesn't currently support graph-based
    methods like BBKNN (see `sctools/non_used_functions.py`).
    """
    import scib as sciblib

    if label_key not in adata_ref.obs or label_key not in adata_int.obs:
        raise KeyError(f"label_key '{label_key}' must exist in both adata_ref.obs and adata_int.obs.")

    # hvg_overlap (called inside scib.metrics.metrics() below) requires
    # batch_key to be a categorical column -- it reads
    # adata.obs[batch_key].cat.categories internally and raises AttributeError
    # on a plain string/object column.
    adata_ref.obs[batch_key] = adata_ref.obs[batch_key].astype("category")
    adata_int.obs[batch_key] = adata_int.obs[batch_key].astype("category")

    # scib reads `embed` only from adata_int (the unintegrated adata_ref
    # always uses its own existing X_pca internally) -- except ilisi_graph/
    # clisi_graph, which internally default to reading adata_int.obsm["X_emb"]
    # regardless of `embed`. Alias it so those metrics score the right representation.
    adata_int.obsm["X_emb"] = adata_int.obsm[embed]

    # NMI/ARI compare label_key against a resolution-optimized clustering
    # (scib's own internal search) vs. the true label; isolated_labels_/
    # graph_conn_/silhouette_ check whether rare cell types and same-label
    # cells stayed distinguishable/connected/tight after integration.
    # n_cores speeds up that resolution search plus clisi_/kBET_ below.
    metric_kwargs = dict(
        pcr_=True, ilisi_=True, hvg_score_=True, cell_cycle_=True, type_="embed", verbose=False,
        silhouette_=True, isolated_labels_=True, nmi_=True, ari_=True, graph_conn_=True, n_cores=16,
    )

    try:
        # clisi_/kBET_ aren't supported by every scib version, and kBET
        # needs a working R/rpy2 setup -- fall back without them if either fails.
        return sciblib.metrics.metrics(
            adata_ref, adata_int, batch_key=batch_key, label_key=label_key,
            embed=embed, organism=organism, clisi_=True, kBET_=True, **metric_kwargs,
        )
    except Exception:
        logger.info("clisi_/kBET_ unavailable (unsupported scib version or no R/rpy2); skipping them.")

    # scib.metrics.metrics() returns a single-column DataFrame (metric name ->
    # row index), not a Series -- `run_metrics[col] = <this>` still works
    # because pandas aligns by row index and takes that one column's values.
    return sciblib.metrics.metrics(
        adata_ref, adata_int, batch_key=batch_key, label_key=label_key, embed=embed, organism=organism, **metric_kwargs,
    )


def RunIntegrationBenchmark(
    adata: AnnData,
    combinations: dict,
    *,
    out_dir: str | Path,
    out_prefix: str | None = None,
    batch_key: str = "sample",
    counts_layer: str,
    log_layer: str,
    organism: str,
    methods: Iterable[str] | None = None,
    method_kwargs: dict | None = None,
    label_key: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Grid-search benchmark: for each combination in `combinations` (from
    `BuildBenchmarkGrid`), preprocess, run the requested integration
    methods, and score each against its own matching (same flavor/
    n_top_genes/n_pcs) unintegrated HVG baseline -- not against raw `adata`.

    Metric mode:
        label_key=None (default): label-free metrics (PCR_batch, iLISI,
            hvg_overlap, cell_cycle_conservation).
        label_key="...": real biological label; adds ASW/NMI/ARI/graph
            connectivity/cLISI/kBET (where the scib version supports them).

    counts_layer: raw counts in `adata.layers`, used by scVI/Seurat directly,
    and by HVG selection when `flavor="seurat_v3"` (that flavor requires counts).

    log_layer: an already normalized/log1p layer in `adata.layers` -- used as
    the HVG/PCA baseline for every run here, and as Seurat's "data" slot.
    This function does not (re)compute normalization; `log_layer` must
    already exist on `adata`.

    Writes one metrics-per-run CSV per run, plus params/all-runs summaries
    and the final ranking, all under `{out_dir}/tab_scib/`.

    Returns
    -------
    (all_metrics_df, params_df, ranking)
    """
    if label_key is not None and label_key not in adata.obs:
        raise KeyError(f"label_key '{label_key}' not found in adata.obs.")

    # create the output directory if it doesn't exist, including the subdir for scIB tabular outputs
    out_dir = Path(out_dir) / TAB_SCIB_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # if no methods are specified, use all supported methods; otherwise validate the requested methods
    if methods is None:
        methods = SUPPORTED_METHODS
    methods = tuple(methods)

    # catch typos/invalid method names now, with a clear message, instead of
    # a KeyError from _METHOD_FUNCS[method] deep inside the loop below.
    # SUPPORTED_METHODS is embedding-based only -- this module scores
    # PCR_batch/cell_cycle_conservation by comparing adata_ref.obsm[embed] vs
    # adata_int.obsm[embed], which only makes sense for methods that actually
    # write a corrected embedding there. Graph-based methods (e.g. BBKNN) are
    # currently retired (see sctools/non_used_functions.py) rather than
    # scored incorrectly.
    unknown = set(methods) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported integration method(s): {sorted(unknown)}. Supported: {SUPPORTED_METHODS}")

    # method_kwargs lets the caller override kwargs for one specific method
    # (e.g. {"seurat": {"anchor_features": 2000}}); defaults to None. Normalize
    # to {} here so `user_method_kwargs.get(m, {})` below never hits
    # AttributeError on None.
    user_method_kwargs = method_kwargs or {}

    logger.info(
        "Running integration benchmark (embedding-based methods only): "
        "methods=%s, label_key=%s",
        methods, label_key,
    )

    all_metric_dfs = []
    param_rows = []

    # adata is expected to already carry a normalized/log1p layer (log_layer)
    # -- no normalization is (re)computed here. Only HVG selection and PCA
    # vary per run; no neighbors/UMAP either, since benchmarking only ever
    # scores X_pca/X_emb, so computing them would be wasted work.
    np.random.seed(random_state)
    adata_norm = adata.copy()
    adata_norm.X = adata_norm.layers[log_layer].copy()

    input_layer_by_flavor = {"seurat_v3": counts_layer, 
                              "seurat_v3_paper": counts_layer, 
                              "cell_ranger": log_layer, 
                              "seurat": log_layer}


    for run_id, params in combinations.items():
        flavor = params["flavor"]
        n_top_genes = params["n_top_genes"]
        n_pcs = params["n_pcs"]

        # Each flavor uses a different input layer for HVG selection; map them here.
        if flavor not in input_layer_by_flavor:
            raise ValueError(
                f"Unsupported HVG flavor '{flavor}'. "
                f"Expected one of: {sorted(input_layer_by_flavor)}."
            )

        layer = input_layer_by_flavor[flavor]
        #

        logger.info(
            "RUN %s | flavor=%s, n_top_genes=%s, n_pcs=%s",
            run_id, flavor, n_top_genes, n_pcs,
        )

        adata_run = adata_norm.copy()

        RunHighlyVariableGenes(
            adata_run,
            layer=layer,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
            flavor=flavor,
        )

        _ = RunPcaOnHvgs(
            adata_run,
            n_pcs=n_pcs,
            random_state=random_state,
        )

        # adata_hvg_ref: HVG-only subset, used as *input to the integration
        # methods* (Seurat/Harmony/scVI expect a matrix already restricted to
        # the run's HVGs). It is NOT used as the scib scoring reference below
        # -- cell_cycle_conservation and hvg_overlap need genes outside the
        # HVG set (cell-cycle markers routinely fall outside HVG selection,
        # and hvg_overlap recomputes HVGs from scratch to compare against).
        # adata_run itself stays full-gene and already carries a valid
        # HVG-based X_pca (RunPcaOnHvgs computes PCA on a temporary HVG-only
        # copy and writes obsm["X_pca"] back onto adata_run), so it is the
        # correct scoring reference -- mirrors what the pre-port notebook did
        # with `sc.tl.pca(..., use_highly_variable=True)` on the full adata.
        adata_hvg_ref = adata_run[:, adata_run.var["highly_variable"]].copy()

        # n_pcs is this run's value -- methods that use it are listed here;
        # RunScanoramaIntegration doesn't take n_pcs at all (uses the
        # existing X_pca as-is), so it's not included. Caller-supplied
        # method_kwargs win over these defaults.
        method_defaults = {
            "harmony": {"n_pcs": n_pcs},
            "seurat": {"n_pcs": n_pcs, "counts_layer": counts_layer, "log_layer": log_layer},
            "scvi": {"counts_layer": counts_layer},
        }
        # For each method being run, combine its defaults (n_pcs/counts_layer/
        # log_layer above) with whatever the caller passed in method_kwargs
        # for that method -- caller-supplied keys win on conflicts.
        merged_method_kwargs = {
            m: {**method_defaults.get(m, {}), **user_method_kwargs.get(m, {})} for m in methods
        }

        dic_int = {}
        for method in methods:
            kwargs = {"batch_key": batch_key, "random_state": random_state, **merged_method_kwargs.get(method, {})}
            # Not every method function accepts every one of these base
            # kwargs (e.g. RunScanoramaIntegration takes neither n_pcs nor
            # random_state) -- drop whatever this specific method doesn't accept.
            kwargs = FilterSupportedKwargs(_METHOD_FUNCS[method], kwargs)

            # _METHOD_FUNCS[method] is the actual function that runs the integration method (e.g. RunHarmonyIntegration, RunScanoramaIntegration, etc.)
            # adata_int becames the integrated AnnData object returned by the integration method
            dic_int[method] = _METHOD_FUNCS[method](adata_hvg_ref, **kwargs)

        run_metrics = pd.DataFrame()

        for method, adata_int in dic_int.items():
            logger.info("Scoring %s (run_id=%s)", method, run_id)

            # embed tells scib which adata_int.obsm key holds this method's
            # corrected representation (scib never reads it from adata_ref --
            # the unintegrated baseline always uses its own existing X_pca).
            # Every method in SUPPORTED_METHODS is embedding-based, so
            # RunScibMetricsWithLabel/RunScibMetricsLabelFree always score
            # with type_="embed" internally -- no per-method lookup needed.
            embed = SCIB_EMBED_BY_METHOD[method]

            # adata_run (full-gene, not adata_hvg_ref) is the scoring
            # reference -- see comment above adata_hvg_ref's definition.
            if label_key is not None:
                metrics_series = RunScibMetricsWithLabel(
                    adata_run, adata_int, organism, label_key, batch_key=batch_key, embed=embed,
                )
            else:
                metrics_series = RunScibMetricsLabelFree(
                    adata_run, adata_int, organism, batch_key=batch_key, embed=embed,
                )

            col_name = f"{method}_{run_id}"
            run_metrics[col_name] = metrics_series

            row = {"col": col_name, "run_id": run_id, "method": method, **params}
            param_rows.append(row)

        # run_metrics so far: rows = metric names, one column per method
        # scored in this run (e.g. "harmony_3", "seurat_3"). Drop metric rows
        # that came back NaN for every method in this run (nothing to report),
        # then save this run's table on its own -- one CSV per run_id.
        run_metrics.dropna(how="all", inplace=True)
        run_metrics.to_csv(out_dir / _Prefixed(f"scib_metrics_run{run_id}.csv", out_prefix), sep="\t")

        # Keep this run's table to stitch all runs together after the loop.
        all_metric_dfs.append(run_metrics)

    # One row per (method, run) combination: which grid params (flavor/
    # n_top_genes/n_pcs) produced that column in the metrics tables above.
    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(out_dir / _Prefixed("scib_params_all_runs.csv", out_prefix), sep="\t", index=False)

    # Glue every run's run_metrics side by side (axis=1, same metric-name row
    # index) into one table: rows = metrics, columns = every method x run
    # combination in the whole benchmark.
    all_metrics_df = pd.concat(all_metric_dfs, axis=1) if all_metric_dfs else pd.DataFrame()
    all_metrics_df.to_csv(out_dir / _Prefixed("scib_metrics_all_runs.csv", out_prefix), sep="\t")

    logger.info("Saving benchmark summary for out_prefix=%s in %s", out_prefix, out_dir)
    ranking, _, _ = SummarizeScibBenchmarkResults(metrics_dir=out_dir, out_prefix=out_prefix)

    return all_metrics_df, params_df, ranking


def SummarizeScibBenchmarkResults(
    metrics_dir: str | Path | None = None,
    out_prefix: str | None = None,
    *,
    sources: dict[str, dict] | None = None,
    label_col: str = "imputation",
    params_file: str | Path | None = None,
    run_ids: Iterable[int] | None = None,
    selected_metrics: tuple[str, ...] | None = None,
    batch_metrics: tuple[str, ...] = BATCH_METRICS,
    bio_metrics: tuple[str, ...] = BIO_METRICS,
    batch_weight: float = 0.4,
    bio_weight: float = 0.6,
    save: bool = True,
    save_dir: str | Path | None = None,
    save_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine per-run scIB metric files into one scaled, weighted ranking of
    method/parameter combinations, merged with their run parameters.

    Single-source usage (unchanged): pass `metrics_dir` (+ optional
    `out_prefix`). Reads `{metrics_dir}/[{out_prefix}.]scib_metrics_run{run_id}.csv`
    (tab-separated despite the `.csv` extension). If `run_ids` is None,
    discovers them by globbing instead of assuming a fixed run count.

    Multi-source usage: pass `sources` instead of `metrics_dir`, a
    `{label: {"metrics_dir": ..., "out_prefix": ..., "params_file": ...,
    "run_ids": ...}}` mapping (only "metrics_dir" is required per source,
    e.g. `{"ALRA": {"metrics_dir": ...}, "raw": {"metrics_dir": ...}}`). All
    sources are pooled *before* min-max scaling, so runs from different
    sources (e.g. with vs. without ALRA imputation) end up on one shared
    [0, 1] scale and are directly comparable -- scaling each source
    separately and concatenating afterwards would not give comparable scores.
    Each source's label is attached as the `label_col` column (default
    "imputation") and folded into the params merge key, since `run_id` alone
    repeats across sources.

    `save=True` (the default) writes the combined tables like the
    single-source case does, but there's no single natural `metrics_dir` to
    write into when pooling multiple sources -- pass `save_dir` (+ optional
    `save_prefix`) to say where, or pass `save=False`.

    Returns
    -------
    (results_with_meta, metrics_raw, metrics_scaled)
    """
    # `multi` selects which calling convention is in use: True for the
    # multi-source path (a `sources` mapping was passed), False for the
    # plain single-`metrics_dir` call. Everything below branches on it.
    multi = sources is not None
    # Reject the two invalid combinations (both given, or neither given) in
    # one check: `multi == (metrics_dir is not None)` is True exactly when
    # the two booleans agree, i.e. both True (both args passed) or both
    # False (neither passed) -- the only valid case is that they disagree.
    if multi == (metrics_dir is not None):
        raise ValueError("Pass exactly one of `metrics_dir` or `sources`.")

    if not multi:
        # Wrap the single-source call into a one-entry `sources` dict (keyed
        # by None, since there's no real label) so the loading/pooling logic
        # below doesn't need a separate branch for the old single-dir shape.
        sources = {None: dict(metrics_dir=metrics_dir, out_prefix=out_prefix, params_file=params_file, run_ids=run_ids)}

    # Load each source independently, then pool: metrics_raw gets a
    # (label_col, method_run) MultiIndex when multi=True so rows stay unique
    # across sources that reuse the same run_id/method combinations; a plain
    # method_run Index otherwise (identical to the old single-source shape).
    raw_by_label = {}
    params_by_label = {}
    for label, src in sources.items():
        mdir = Path(src["metrics_dir"])
        pfx = src.get("out_prefix")
        raw_by_label[label] = _LoadScibRunMetrics(mdir, pfx, src.get("run_ids"))
        params_by_label[label] = _LoadScibParams(mdir, pfx, src.get("params_file"))

    if multi:
        metrics_raw = pd.concat(raw_by_label, axis=0, names=[label_col, None])
        # params_meta doesn't need the same MultiIndex trick as metrics_raw --
        # it's only ever looked up via merge(on=merge_keys), so the label just
        # needs to be an explicit column (not part of the index) for `run_id`
        # + label_col to work as the join key.
        params_meta = pd.concat(
            {label: df.assign(**{label_col: label}) for label, df in params_by_label.items()},
            ignore_index=True,
        )
        merge_keys = ["run_id", label_col]
    else:
        metrics_raw = next(iter(raw_by_label.values()))
        params_meta = next(iter(params_by_label.values()))
        merge_keys = ["run_id"]

    # None -> derive from batch_metrics + bio_metrics, so label-free and
    # labeled runs each pick up exactly the metrics they actually computed.
    if selected_metrics is None:
        selected_metrics = tuple(batch_metrics) + tuple(bio_metrics)

    # Keep only the metrics that both were requested (selected_metrics) and
    # actually exist in metrics_raw (label-free vs labeled runs produce
    # different columns), then drop any that came back all-NaN across every run.
    metrics = metrics_raw[[c for c in selected_metrics if c in metrics_raw.columns]].dropna(axis=1, how="all")

    # Min-max scale per column (pooled across all sources when multi=True);
    # zero-range columns become NaN instead of a divide-by-zero.
    col_min = metrics.min()
    col_range = (metrics.max() - col_min).replace(0, np.nan)
    metrics_scaled = (metrics - col_min) / col_range

    # batch_metrics/bio_metrics are the full canonical lists; narrow each down
    # to whichever of those columns actually survived into metrics_scaled
    # (i.e. were computed for this run and weren't all-NaN).
    avail_batch = [c for c in batch_metrics if c in metrics_scaled.columns]
    avail_bio = [c for c in bio_metrics if c in metrics_scaled.columns]

    # One row per method_run combination. batch_score/bio_score are each the
    # row-wise mean of their already-[0,1]-scaled metrics (NaN if none are
    # available); overall combines them with batch_weight/bio_weight (default
    # 0.4/0.6, matching the scIB paper's batch-correction vs bio-conservation
    # weighting).
    scores = pd.DataFrame(index=metrics_scaled.index)
    scores["batch_score"] = metrics_scaled[avail_batch].mean(axis=1) if avail_batch else np.nan
    scores["bio_score"] = metrics_scaled[avail_bio].mean(axis=1) if avail_bio else np.nan
    scores["overall"] = batch_weight * scores["batch_score"] + bio_weight * scores["bio_score"]

    results = pd.concat([metrics_scaled, scores], axis=1).sort_values("overall", ascending=False)

    # Extract method/run_id from the index, formatted as "{method}_{run_id}"
    # (get_level_values(-1) is a no-op on a plain Index, so this also covers
    # the single-source case; the label itself is the outer MultiIndex level).
    method_run = pd.Series(results.index.get_level_values(-1), index=results.index)
    results[["method", "run_id"]] = method_run.str.rsplit("_", n=1, expand=True)
    results["run_id"] = pd.to_numeric(results["run_id"], errors="coerce")
    if multi:
        results[label_col] = results.index.get_level_values(0)
        # The index's outer level is also named label_col (from the concat
        # above), which makes it ambiguous with the column of the same name
        # during merge -- drop the index now that both pieces of information
        # it held (method/run_id/label) live in columns.
        results = results.reset_index(drop=True)

    results_with_meta = results.merge(params_meta, on=merge_keys, how="left").reset_index(drop=True)

    meta_cols = ["method", "run_id"] + ([label_col] if multi else [])
    score_cols = [c for c in selected_metrics + ("batch_score", "bio_score", "overall") if c in results_with_meta.columns]
    param_cols = [c for c in ("flavor", "n_top_genes", "n_pcs") if c in results_with_meta.columns]

    results_with_meta = results_with_meta[meta_cols + score_cols + param_cols]

    if save:
        if multi:
            if save_dir is None:
                raise ValueError("save=True with `sources` requires `save_dir` (optionally with `save_prefix`).")
            out_dir = Path(save_dir)
        else:
            # sources[None] is the single-source dict built above -- reusing
            # its metrics_dir/out_prefix here keeps the saved filenames
            # identical to pre-`sources` behavior when save_prefix isn't
            # explicitly overridden.
            out_dir = Path(sources[None]["metrics_dir"])
            if save_prefix is None:
                save_prefix = sources[None].get("out_prefix")
        metrics_raw.to_csv(out_dir / _Prefixed("all_metrics_raw.tsv", save_prefix), sep="\t")
        results.to_csv(out_dir / _Prefixed("scib_metrics_summary.tsv", save_prefix), sep="\t")
        results_with_meta.to_csv(out_dir / _Prefixed("scib_metrics_with_params.tsv", save_prefix), sep="\t", index=False)

    return results_with_meta, metrics_raw, metrics_scaled
