"""
sctools.integration.workflow
==============================

Applies one integration method (sctools.integration.methods) to an
already-preprocessed AnnData (run sctools.preprocessing.NormalizeHvgPcaKnn
yourself first) and computes neighbors/UMAP on the integrated embedding.
Leiden clustering (RunLeidenClustering) is a separate step, called by hand
on the result.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

from sctools.preprocessing import RunNeighborsAndUmap

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
    `RunIntegration` always uses the default (True).
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


def RunIntegration(
    adata_run: AnnData,
    method: str,
    *,
    n_pcs: int,
    n_neighbors: int,
    random_state: int = 42,
    batch_key: str = "sample",
    counts_layer: str,
    log_layer: str,
    method_kwargs: dict | None = None,
    plot_dir: str | Path | None = None,
    store_params: bool = True,
) -> AnnData:
    """
    Apply one integration method to an already-preprocessed AnnData, then
    compute neighbors/UMAP on the integrated embedding.

    method: one of "seurat", "scanorama", "harmony", "scvi".

    adata_run: full (all-genes) AnnData already processed by
    NormalizeHvgPcaKnn -- must already have adata_run.var["highly_variable"]
    and adata_run.obsm["X_pca"]. This function does not (re)run preprocessing.

    counts_layer: raw counts, used directly by scVI/Seurat.

    log_layer: normalized/log1p expression, used only by Seurat (its "data" slot).

    store_params: if True, store this run's parameters in
    `adata_int.uns["integration_params"]`.

    Does not run clustering -- call RunLeidenClustering on the returned
    AnnData yourself afterward.

    Returns
    -------
    adata_int: HVG-subset AnnData after integration, with neighbors/UMAP computed.
    """
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported integration method '{method}'. Supported: {SUPPORTED_METHODS}")

    logger.info(
        "RunIntegration: method=%s, n_pcs=%d, n_neighbors=%d",
        method, n_pcs, n_neighbors,
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

    merged_kwargs = {**method_defaults.get(method, {}), **(method_kwargs or {}).get(method, {})}
    kwargs = {"batch_key": batch_key, "random_state": random_state, **merged_kwargs}
    
    # Not every method function accepts every one of these base kwargs (e.g.
    # RunScanoramaIntegration takes neither n_pcs nor random_state) -- drop
    # whatever this specific method doesn't accept.
    # INTEGRATION FUNCTION APLICATION!
    kwargs = FilterSupportedKwargs(_METHOD_FUNCS[method], kwargs)
    adata_int = _METHOD_FUNCS[method](adata_hvg, **kwargs)

    # Neighbors/UMAP on the integrated embedding -- needed for downstream
    # clustering (call RunLeidenClustering on adata_int yourself) and for
    # AttachHvgResultsToFullAdata, which expects adata_int.obsm["X_umap"].
    RunNeighborsAndUmap(
        adata_int,
        n_neighbors=n_neighbors,
        use_rep=SCIB_EMBED_BY_METHOD[method],
        random_state=random_state,
    )

    if plot_dir is not None:
        # Neighbors/UMAP were already computed above -- just render/save the plot here.
        PlotUmap(
            adata_int,
            title=method,
            batch_key=batch_key,
            rep=SCIB_EMBED_BY_METHOD[method],
            recompute_neighbors=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
            plot_dir=plot_dir,
        )

    if store_params:
        adata_int.uns["integration_params"] = {
            "method": method,
            "n_pcs": int(n_pcs),
            "n_neighbors": int(n_neighbors),
            "batch_key": batch_key,
            "random_state": int(random_state),
        }

    return adata_int


def AttachHvgResultsToFullAdata(
    adata_full: AnnData,
    adata_hvg: AnnData,
    method: str,
    store_integrated_layer: bool = True,
) -> AnnData:
    """
    Attach HVG-based integration results back onto the full (all-genes) AnnData.

    Keeps all genes while carrying over the embeddings/clusters computed on
    the HVG subset. adata_full.obsm["X_pca"] (the original, pre-integration
    PCA) is never touched -- for harmony/scanorama/scvi, the corrected
    embedding is attached separately under its own name (X_harmony/
    X_scanorama/X_scVI). Seurat's corrected embedding lives in
    adata_hvg.obsm["X_pca"] itself (no distinct name, since Seurat
    integration recomputes PCA in place) -- there's no separate key to
    attach without colliding with the original, so it's left out of
    adata_full; inspect adata_hvg directly for Seurat's result.

    adata_full: full object with all genes (e.g. adata_run from RunIntegration).
    adata_hvg: HVG object used for integration/clustering (e.g. adata_cl_hvg).
    store_integrated_layer: if True and this is a method that writes a
        corrected-expression layer (currently only Seurat, adata_hvg.layers["seurat"]),
        store it in adata_full as f"integrated_{method}_HVG", with non-HVG genes
        filled as NaN. Embedding-only methods (harmony/scanorama/scvi) have no
        such layer, so this has no effect for them.
    """
    adata_full = adata_full.copy()

    # UMAP is always overwritten with the one computed on the integrated
    # embedding (no original preserved here).
    adata_full.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()

    # Attach the corrected embedding under its own name, only when it has one
    # distinct from "X_pca" (harmony/scanorama/scvi). Seurat's embed_key is
    # "X_pca" itself, so there's nothing to attach without overwriting the
    # original -- adata_full.obsm["X_pca"] is left as the untouched original for it.
    embed_key = SCIB_EMBED_BY_METHOD[method]
    if embed_key != "X_pca":
        adata_full.obsm[embed_key] = adata_hvg.obsm[embed_key].copy()

    # Leiden clustering results are always attached, since they are computed on the integrated embedding and are not present in the original adata_full.
    leiden_keys = [key for key in adata_hvg.obs.columns if key.startswith("leiden_")]
    for key in leiden_keys:
        adata_full.obs[key] = adata_hvg.obs[key].copy()

    # Only Seurat writes a corrected-expression layer (named "seurat", no
    # "integrated_" prefix) -- harmony/scanorama/scvi are embedding-only and
    # never write one.
    source_layer = "seurat" if method == "seurat" else None
    if store_integrated_layer and source_layer is not None and source_layer in adata_hvg.layers:
        full_integrated = np.full(adata_full.shape, np.nan, dtype=np.float32)
        hvg_mask = adata_full.var["highly_variable"].to_numpy()
        integrated_layer = adata_hvg.layers[source_layer]
        if issparse(integrated_layer):
            integrated_layer = integrated_layer.toarray()
        full_integrated[:, hvg_mask] = integrated_layer
        adata_full.layers[f"integrated_{method}_HVG"] = full_integrated

    return adata_full
