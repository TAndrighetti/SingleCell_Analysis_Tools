"""
sctools.integration.workflow
==============================

Orchestrates preprocessing (sctools.preprocessing) + one or more integration
methods (sctools.integration.methods) + Leiden clustering.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData

from sctools.preprocessing import NormalizeHvgPcaKnn, RunNeighborsAndUmap

from .methods import SCIB_EMBED_BY_METHOD, SUPPORTED_METHODS, _METHOD_FUNCS, FilterSupportedKwargs

logger = logging.getLogger(__name__)


def PlotUmap(
    adata: AnnData,
    title: str,
    *,
    batch_key: str = "sample",
    rep: str = "X_pca",
    recompute_neighbors: bool = True,
    n_neighbors: int = 15,
    random_state: int = 42,
    plot_dir: str | Path | None = None,
    dpi: int = 200,
) -> Path | None:
    """
    Plot a UMAP colored by `batch_key`. Saves to disk only if `plot_dir` is
    given; otherwise renders inline.

    recompute_neighbors=False if the neighbor graph was already built by a
    graph-based method (e.g. a re-enabled BBKNN) and shouldn't be overwritten.
    Every method currently in `SUPPORTED_METHODS` is embedding-based, so
    `RunIntegrationComplete` always uses the default (True).
    """
    if recompute_neighbors:
        RunNeighborsAndUmap(adata, n_neighbors=n_neighbors, use_rep=rep, random_state=random_state)
    else:
        sc.tl.umap(adata, random_state=random_state)

    show = plot_dir is None
    sc.pl.umap(adata, color=[batch_key], wspace=1, title=title, show=show, save=False)

    if plot_dir is None:
        return None

    import matplotlib.pyplot as plt

    out_dir = Path(plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(title))
    out_path = out_dir / f"{safe_title}.png"

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def RunLeidenClustering(
    adata: AnnData,
    resolutions: tuple[float, ...] = (0.6, 0.8, 1.0),
    *,
    random_state: int = 42,
) -> tuple[AnnData, list[str]]:
    """
    Run Leiden clustering at multiple resolutions. Adds one
    adata.obs["leiden_res{resolution}"] column per resolution. Operates on
    and returns a copy, plus the list of added column names.
    """
    adata = adata.copy()

    leiden_keys = []
    for res in resolutions:
        key = f"leiden_res{res:g}"
        logger.info("Running Leiden clustering (resolution=%s)", res)
        sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added=key)
        leiden_keys.append(key)

    return adata, leiden_keys


def RunIntegrationComplete(
    adata: AnnData,
    method: str,
    *,
    flavor: str,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    resolutions: tuple[float, ...] = (0.6, 0.8, 1, 1.5),
    random_state: int = 42,
    batch_key: str = "sample",
    counts_layer: str,
    log_layer: str,
    method_kwargs: dict | None = None,
    plot_dir: str | Path | None = None,
    store_params: bool = False,
) -> tuple[AnnData, AnnData]:
    """
    Full single-method integration pipeline for production runs:
    normalize/HVG/PCA/kNN -> apply one integration method -> Leiden cluster.

    method: one of "seurat", "scanorama", "harmony", "scvi".

    counts_layer: raw counts, used both for HVG/PCA preprocessing
    (NormalizeHvgPcaKnn) and directly by scVI/Seurat.

    log_layer: normalized/log1p expression, used only by Seurat (its "data" slot).

    store_params: if True, store this run's parameters in
    `adata_run.uns["integration_params"]`.

    Returns (adata_run, adata_cl_hvg):
        adata_run: full (all-genes) AnnData after NormalizeHvgPcaKnn.
        adata_cl_hvg: HVG-subset AnnData after integration + clustering.
    """
    method = method.lower().strip()

    logger.info(
        "RunIntegrationComplete: method=%s, flavor=%s, n_top_genes=%d, n_pcs=%d, n_neighbors=%d",
        method, flavor, n_top_genes, n_pcs, n_neighbors,
    )

    adata_run = NormalizeHvgPcaKnn(
        adata,
        batch_key=batch_key,
        input_layer=counts_layer,
        flavor=flavor,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        copy=True,
    )

    adata_hvg = adata_run[:, adata_run.var["highly_variable"]].copy()

    # n_pcs/counts_layer/log_layer from this call always reach the method,
    # unless method_kwargs explicitly overrides them. RunScanoramaIntegration
    # doesn't take n_pcs at all (uses the existing X_pca as-is), so it's not
    # included here.
    method_defaults = {
        "harmony": {"n_pcs": n_pcs},
        "seurat": {"n_pcs": n_pcs, "counts_layer": counts_layer, "log_layer": log_layer},
        "scvi": {"counts_layer": counts_layer},
    }
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported integration method '{method}'. Supported: {SUPPORTED_METHODS}")

    merged_kwargs = {**method_defaults.get(method, {}), **(method_kwargs or {}).get(method, {})}
    kwargs = {"batch_key": batch_key, "random_state": random_state, **merged_kwargs}
    # Not every method function accepts every one of these base kwargs (e.g.
    # RunScanoramaIntegration takes neither n_pcs nor random_state) -- drop
    # whatever this specific method doesn't accept.
    kwargs = FilterSupportedKwargs(_METHOD_FUNCS[method], kwargs)
    adata_int = _METHOD_FUNCS[method](adata_hvg, **kwargs)

    if plot_dir is not None:
        PlotUmap(
            adata_int,
            title=method,
            batch_key=batch_key,
            rep=SCIB_EMBED_BY_METHOD[method],
            n_neighbors=n_neighbors,
            random_state=random_state,
            plot_dir=plot_dir,
        )

    adata_cl_hvg, leiden_keys = RunLeidenClustering(adata_int, resolutions=resolutions, random_state=random_state)

    if store_params:
        adata_run.uns["integration_params"] = {
            "method": method,
            "flavor": flavor,
            "n_top_genes": int(n_top_genes),
            "n_pcs": int(n_pcs),
            "n_neighbors": int(n_neighbors),
            "resolutions": list(resolutions),
            "batch_key": batch_key,
            "random_state": int(random_state),
            "leiden_keys": leiden_keys,
        }

    return adata_run, adata_cl_hvg


def AttachHvgResultsToFullAdata(
    adata_full: AnnData,
    adata_hvg: AnnData,
    method: str,
    store_integrated_layer: bool = True,
) -> AnnData:
    """
    Attach HVG-based integration results back onto the full (all-genes) AnnData.

    Keeps all genes while carrying over the embeddings/clusters computed on
    the HVG subset. The original (pre-integration) PCA/UMAP are preserved
    under "*_raw" keys.

    adata_full: full object with all genes (e.g. adata_run from RunIntegrationComplete).
    adata_hvg: HVG object used for integration/clustering (e.g. adata_cl_hvg).
    store_integrated_layer: if True and adata_hvg.layers[f"integrated_{method}"]
        exists, store it in adata_full, with non-HVG genes filled as NaN.
    """
    adata_full = adata_full.copy()

    adata_full.obsm["X_pca_raw"] = adata_full.obsm["X_pca"].copy()
    adata_full.obsm["X_umap_raw"] = adata_full.obsm["X_umap"].copy()

    adata_full.obsm["X_pca"] = adata_hvg.obsm["X_pca"].copy()
    adata_full.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()

    leiden_keys = [key for key in adata_hvg.obs.columns if key.startswith("leiden_")]
    for key in leiden_keys:
        adata_full.obs[key] = adata_hvg.obs[key].copy()

    if store_integrated_layer and f"integrated_{method}" in adata_hvg.layers:
        full_integrated = np.full(adata_full.shape, np.nan, dtype=np.float32)
        hvg_mask = adata_full.var["highly_variable"].to_numpy()
        full_integrated[:, hvg_mask] = adata_hvg.layers[f"integrated_{method}"]
        adata_full.layers[f"integrated_{method}_HVG"] = full_integrated

    return adata_full
