"""
sctools.integration
====================

Batch-integration methods, post-integration clustering, and scIB-based
benchmarking for single-cell RNA-seq data.

Ported from ``04.1sx.Integration-benchmarking_scNeuAntib.ipynb`` and
``04sx.scNewFfar2_Integration.ipynb`` (both notebooks defined the same
integration functions almost verbatim; this module merges them into one
non-redundant implementation). See module-level comments marked "FIX" for
behavioral differences from the original notebooks and why they were made.

Typical usage order:

    Single production run (one method, one parameter set):
        RunIntegrationComplete()
            -> preprocessing.NormalizeHvgPcaKnn()
            -> ApplyIntegrationMethods()
            -> Clustering()
        AttachHvgResultsToFullAdata()
        UpdateCellsToRemove()  # iterative low-quality cluster curation

    Benchmarking a grid of methods/parameters:
        BuildCombinationsDictAndParamsDf()
        RunIntegrationTests()
            -> preprocessing.NormalizeHvgPcaKnn()
            -> ApplyIntegrationMethods()
            -> RunScibMetricsWithLeiden()

Reusable steps:
    RunSeuratAnchors()
    PlotUmap()
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from anndata import AnnData

from sctools.preprocessing import NormalizeHvgPcaKnn

logger = logging.getLogger(__name__)

# scIB output type for each integration method: whether its result is a new
# embedding ("embed", e.g. Harmony/Scanorama/scVI/Seurat-PCA) or a modified
# kNN graph with no new embedding ("knn", BBKNN). scib.metrics.metrics needs
# this to pick the right batch/bio-conservation metrics.
# FIX: the original notebooks always used type_="embed" for every method,
# including BBKNN, even though BBKNN never writes a new adata.obsm embedding
# -- it only rewrites the neighbor graph. Benchmarking BBKNN with type_="embed"
# silently scores it on the *pre*-integration PCA. See scib docs on `type_`
# ("full"/"embed"/"knn") and the original scIB paper (Luecken et al. 2022,
# Nat Methods), which evaluates graph-output methods with knn-specific metrics.
SCIB_TYPE_BY_METHOD = {
    "bbknn": "knn",
    "seurat": "embed",
    "scanorama": "embed",
    "harmony": "embed",
    "scvi": "embed",
}

# Embedding key written into adata.obsm by each integration method.
# FIX: the original RunIntegrationTests picked the embedding key with
# `next(k for k in adata_int.obsm if method in k)`, a case-sensitive
# substring match. For method="scvi" this never matches "X_scVI" (lowercase
# "scvi" is not a substring of "X_scVI"), silently falling back to the
# pre-integration "X_pca" -- so every scVI run was benchmarked on the
# uncorrected embedding instead of the actual scVI latent space. An explicit
# mapping avoids this class of bug.
SCIB_EMBED_BY_METHOD = {
    "bbknn": "X_pca",
    "seurat": "X_pca",
    "scanorama": "X_scanorama",
    "harmony": "X_harmony",
    "scvi": "X_scVI",
}


def RunSeuratAnchors(
    adata_seurat: AnnData,
    batch_key: str,
    counts_layer: str = "QC_filtered",
    data_layer: str = "logcounts",
):
    """
    Integrate batches with Seurat's CCA anchor-based workflow (via rpy2/R).

    Converts `adata_seurat` to a Seurat object, splits it by `batch_key`,
    runs `FindIntegrationAnchors()` + `IntegrateData()` using all genes in
    `adata_seurat` as anchor features (in practice the HVGs, since callers
    subset to HVGs before this step), and returns the integrated expression
    matrix (cells x genes).

    Parameters
    ----------
    counts_layer
        Layer used as the Seurat "counts" slot.

    data_layer
        Layer used as the Seurat "data" slot (normalized/log1p expression).
    """
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    pandas2ri.activate()
    anndata2ri.activate()

    ro.globalenv["adata_seurat"] = adata_seurat
    ro.globalenv["batch_key"] = batch_key
    ro.globalenv["counts_layer"] = counts_layer
    ro.globalenv["data_layer"] = data_layer

    ro.r("""
        suppressMessages({
            suppressWarnings({
                library(Seurat)

                seurat <- as.Seurat(
                    adata_seurat,
                    counts = counts_layer,
                    data = data_layer
                )

                batch_list <- SplitObject(seurat, split.by = batch_key)

                anchors <- FindIntegrationAnchors(
                    object.list = batch_list,
                    anchor.features = rownames(seurat),
                    verbose = FALSE
                )

                integrated <- IntegrateData(
                    anchorset = anchors,
                    verbose = FALSE
                )

                integrated_expr <- GetAssayData(integrated)
                integrated_expr <- integrated_expr[rownames(seurat), colnames(seurat)]
                integrated_expr <- t(integrated_expr)
            })
        })
    """)

    return ro.r("integrated_expr")


def PlotUmap(
    adata: AnnData,
    title: str,
    out_dir: str,
    *,
    batch_key: str = "sample",
    rep: str = "X_pca",
    recompute_neighbors: bool = True,
    n_neighbors: int = 15,
    random_state: int = 42,
    dpi: int = 200,
) -> str:
    """
    Save a UMAP plot to disk (instead of displaying it).

    Parameters
    ----------
    out_dir
        Directory where the figure will be saved (created if missing).

    rep
        `adata.obsm` key used as the neighbor-graph/UMAP input representation.

    recompute_neighbors
        If False, assumes the neighbor graph is already built (e.g. after
        BBKNN, which builds its own batch-balanced graph -- recomputing
        neighbors here would overwrite it).

    random_state
        Random seed for `sc.pp.neighbors` / `sc.tl.umap`.

        FIX: the benchmarking notebook's version of this function read a
        module-level global `random_state = 42` instead of taking it as a
        parameter, so this function would raise `NameError` (or silently use
        an unrelated variable) outside that notebook. The Ffar2 notebook's
        version already took it as an explicit parameter; that is the
        version kept here.
    """
    os.makedirs(out_dir, exist_ok=True)

    if recompute_neighbors:
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=rep,
            random_state=random_state,
        )

    sc.tl.umap(adata, random_state=random_state)

    safe_title = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(title))
    out_path = os.path.join(out_dir, f"{safe_title}.png")

    sc.pl.umap(
        adata,
        color=[batch_key],
        wspace=1,
        title=title,
        show=False,
        save=False,
    )

    import matplotlib.pyplot as plt

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return out_path


def ApplyIntegrationMethods(
    adata_hvg: AnnData,
    run_id,
    out_dir: str,
    *,
    batch_key: str = "sample",
    counts_layer: str = "QC_filtered",
    log_layer: str = "QC_filtered_log1p",
    bbknn: bool = True,
    seurat: bool = True,
    scanorama: bool = True,
    harmony: bool = True,
    scvi_: bool = True,
    n_pcs: int = 35,
    n_neighbors: int = 15,
    scale_seurat_before_pca: bool = True,
    random_state: int = 42,
) -> dict[str, AnnData]:
    """
    Apply multiple integration methods and return one AnnData per method.

    Parameters
    ----------
    adata_hvg
        AnnData subset to highly variable genes (or at least with
        `.var['highly_variable']` set), typically produced by
        `preprocessing.NormalizeHvgPcaKnn()`.

    out_dir
        Directory for the diagnostic UMAP PNGs written by each method.

        FIX: the original notebooks hardcoded
        `out_dir = "/home/tahila/2509_scRNA_Hcar/plots_tests"` inside this
        function -- a path from a different, older project
        (`2509_scRNA_Hcar`), unrelated to the scNeu datasets these notebooks
        actually process. Every call silently wrote plots there regardless
        of the `out_dir` passed to the notebook-level `PlotUmap` calls
        elsewhere. This is now a required parameter with no hardcoded default.

    counts_layer, log_layer
        Layers with raw counts / log1p data, used for scVI and Seurat
        conversion respectively. scVI's likelihood is a count model
        (negative binomial / ZINB), so it requires the raw-count layer, not
        log1p data -- see the scvi-tools documentation.

    scale_seurat_before_pca
        Whether to `sc.pp.scale()` the Seurat-integrated expression matrix
        before computing PCA on it.

        FIX: the original code ran `sc.tl.pca()` directly on the
        Seurat-integrated matrix without scaling first. Seurat's own
        integration vignettes (`ScaleData()` -> `RunPCA()` on the
        `"integrated"` assay) always scale before PCA on that assay, and
        this package's own `preprocessing.RunPcaOnHvgs()` already scales by
        default for the same reason (unscaled PCA lets the most
        highly-expressed/variable genes dominate the components). Set to
        False to reproduce the original notebook behavior exactly.

    Returns
    -------
    dict mapping method name -> integrated AnnData.
    """
    res: dict[str, AnnData] = {}

    def EnsurePca(adata: AnnData) -> None:
        """Ensure PCA exists in adata.obsm['X_pca']."""
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)

    # 1) BBKNN
    if bbknn:
        logger.info("Applying BBKNN (run_id=%s)", run_id)
        adata_bbknn = adata_hvg.copy()

        EnsurePca(adata_bbknn)
        neighbors_within_batch = 25 if adata_bbknn.n_obs > 100000 else 3

        # BBKNN directly builds a batch-balanced neighbor graph.
        sce.pp.bbknn(
            adata_bbknn,
            batch_key=batch_key,
            neighbors_within_batch=neighbors_within_batch,
            n_pcs=n_pcs,
        )

        # Do not overwrite the BBKNN graph by recomputing neighbors.
        PlotUmap(
            adata_bbknn,
            title=f"BBKNN_{run_id}",
            out_dir=out_dir,
            batch_key=batch_key,
            recompute_neighbors=False,
            random_state=random_state,
        )
        res["bbknn"] = adata_bbknn

    # 2) Seurat Anchors
    if seurat:
        logger.info("Applying Seurat Anchors (run_id=%s)", run_id)
        adata_seurat = adata_hvg.copy()

        # Ensure batch column is string/categorical-friendly for R.
        adata_seurat.obs[batch_key] = adata_seurat.obs[batch_key].astype(str)

        # Clear uns to avoid conversion issues with arbitrary python objects.
        adata_seurat.uns = {}

        integrated_mat = RunSeuratAnchors(
            adata_seurat,
            batch_key=batch_key,
            counts_layer=counts_layer,
            data_layer=log_layer,
        )

        adata_seurat.layers["seurat"] = integrated_mat

        # PCA on the integrated matrix (compatible across scanpy versions by
        # temporarily swapping X).
        X_backup = adata_seurat.X
        adata_seurat.X = adata_seurat.layers["seurat"]

        if scale_seurat_before_pca:
            sc.pp.scale(adata_seurat, max_value=10)

        sc.tl.pca(adata_seurat, n_comps=n_pcs, svd_solver="arpack", random_state=random_state)
        adata_seurat.X = X_backup

        PlotUmap(
            adata_seurat,
            title=f"SeuratAnchors_{run_id}",
            out_dir=out_dir,
            batch_key=batch_key,
            rep="X_pca",
            recompute_neighbors=True,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        res["seurat"] = adata_seurat

    # 3) Scanorama
    if scanorama:
        logger.info("Applying Scanorama (run_id=%s)", run_id)
        adata_scanorama = adata_hvg.copy()

        # FIX: the original code never called EnsurePca() here (unlike the
        # BBKNN and Harmony branches), even though
        # `sc.external.pp.scanorama_integrate()` requires `adata.obsm["X_pca"]`
        # to already exist -- it does not compute PCA internally. This
        # worked in the notebooks only because `adata_hvg` always already
        # carried `X_pca` from the upstream NormalizeHvgPcaKnn() call; adding
        # it here makes this function safe to call standalone too.
        EnsurePca(adata_scanorama)

        sce.pp.scanorama_integrate(adata_scanorama, key=batch_key, verbose=1)

        PlotUmap(
            adata_scanorama,
            title=f"Scanorama_{run_id}",
            out_dir=out_dir,
            batch_key=batch_key,
            rep="X_scanorama",
            recompute_neighbors=True,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        res["scanorama"] = adata_scanorama

    # 4) Harmony
    if harmony:
        logger.info("Applying Harmony (run_id=%s)", run_id)
        adata_harmony = adata_hvg.copy()

        EnsurePca(adata_harmony)

        sce.pp.harmony_integrate(
            adata_harmony,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_harmony",
            verbose=1,
        )

        PlotUmap(
            adata_harmony,
            title=f"Harmony_{run_id}",
            out_dir=out_dir,
            batch_key=batch_key,
            rep="X_harmony",
            recompute_neighbors=True,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        res["harmony"] = adata_harmony

    # 5) scVI
    if scvi_:
        logger.info("Applying scVI (run_id=%s)", run_id)
        import scvi

        scvi.settings.seed = random_state

        adata_scvi = adata_hvg.copy()

        scvi.model.SCVI.setup_anndata(
            adata_scvi,
            layer=counts_layer,
            batch_key=batch_key,
        )

        model = scvi.model.SCVI(adata_scvi)
        model.train(max_epochs=200, early_stopping=True)

        adata_scvi.obsm["X_scVI"] = model.get_latent_representation()

        PlotUmap(
            adata_scvi,
            title=f"scVI_{run_id}",
            out_dir=out_dir,
            batch_key=batch_key,
            rep="X_scVI",
            recompute_neighbors=True,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        res["scvi"] = adata_scvi

    return res


def Clustering(
    adata: AnnData,
    resolutions: tuple[float, ...] = (0.6, 0.8, 1.0),
    random_state: int = 42,
) -> AnnData:
    """
    Run Leiden clustering at multiple resolutions and plot each on UMAP.

    Adds one `adata.obs["leiden_res{resolution}"]` column per resolution.
    Operates on (and returns) a copy of `adata`.
    """
    adata_hvg = adata.copy()

    leiden_keys = []
    for res in resolutions:
        key = f"leiden_res{res:g}"
        sc.tl.leiden(
            adata_hvg,
            resolution=res,
            random_state=random_state,
            key_added=key,
        )
        leiden_keys.append(key)

    mid = len(leiden_keys) // 2

    sc.pl.umap(adata_hvg, color=leiden_keys[:mid], legend_loc="on data")
    sc.pl.umap(adata_hvg, color=leiden_keys[mid:], legend_loc="on data")

    return adata_hvg


def RunIntegrationComplete(
    adata: AnnData,
    method: str,
    out_dir: str,
    flavor: str,
    n_top_genes: int,
    n_pcs: int,
    n_neighbors: int,
    *,
    resolutions: tuple[float, ...] = (0.6, 0.8, 1, 1.5),
    random_state: int = 42,
    batch_key: str = "sample",
    input_layer: str = "QC_filtered",
    counts_layer: str = "QC_filtered",
    log_layer: str = "QC_filtered_log1p",
):
    """
    Run the full single-method integration pipeline used for production runs:
    normalize/HVG/PCA/kNN -> apply one integration method -> Leiden cluster it.

    This is `preprocessing.NormalizeHvgPcaKnn()` + `ApplyIntegrationMethods()`
    (only `method` enabled) + `Clustering()`, chained together. It replaces
    the notebooks' local, duplicated `Normaliza_HVG_PCA_kNN` definition with
    the package's single `NormalizeHvgPcaKnn` implementation.

    Parameters
    ----------
    method
        One of "bbknn", "seurat", "scanorama", "harmony", "scvi".

    Returns
    -------
    (adata_run, adata_cl_hvg)
        `adata_run` : full (all-genes) AnnData after NormalizeHvgPcaKnn.
        `adata_cl_hvg` : HVG-subset AnnData after integration + clustering.
    """
    logger.info(
        "RunIntegrationComplete: method=%s, flavor=%s, n_top_genes=%d, n_pcs=%d, n_neighbors=%d",
        method, flavor, n_top_genes, n_pcs, n_neighbors,
    )

    adata_run = adata.copy()

    adata_run = NormalizeHvgPcaKnn(
        adata_run,
        batch_key=batch_key,
        input_layer=input_layer,
        flavor=flavor,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        plot_before_integration=True,
    )

    adata_hvg = adata_run[:, adata_run.var["highly_variable"]].copy()

    method = method.lower().strip()

    dic_int = ApplyIntegrationMethods(
        adata_hvg,
        run_id=method,
        out_dir=out_dir,
        batch_key=batch_key,
        counts_layer=counts_layer,
        log_layer=log_layer,
        bbknn=(method == "bbknn"),
        seurat=(method == "seurat"),
        scanorama=(method == "scanorama"),
        harmony=(method == "harmony"),
        scvi_=(method == "scvi"),
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    adata_cl_hvg = Clustering(dic_int[method], resolutions=resolutions, random_state=random_state)

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

    Parameters
    ----------
    adata_full
        Full object containing all genes (e.g. `adata_run` from
        `RunIntegrationComplete`).

    adata_hvg
        HVG object used for integration and clustering (e.g. `adata_cl_hvg`
        from `RunIntegrationComplete`).

    store_integrated_layer
        If True and `adata_hvg.layers[f"integrated_{method}"]` exists, store
        the integrated expression matrix in `adata_full`, with non-HVG genes
        filled as NaN (integration was only computed for HVGs).
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


def UpdateCellsToRemove(
    prefix_dir: str,
    original_file: str,
    res: str,
    clusters_to_remove: list,
    cells_to_remove_now: list,
) -> list:
    """
    Append a low-quality-cluster removal decision to a running log file and
    return the cumulative list of cells to remove across all logged decisions.

    Used for iterative curation: after each `RunIntegrationComplete()` +
    clustering pass, low-quality clusters (doublets, stressed/dying cells,
    etc.) are identified and logged here before re-running integration on the
    filtered object. The log lives at `{prefix_dir}.cells_to_remove.txt`
    (tab-separated: original file, resolution, clusters removed, cell IDs).
    """
    file = f"{prefix_dir}.cells_to_remove.txt"

    new_original_file = str(original_file)
    new_resolution = f"leiden_res{res}"
    new_clusters = ",".join(map(str, clusters_to_remove))
    new_cells = ",".join(map(str, cells_to_remove_now))

    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write("Original file\tResolution\tClusters to remove\tCells\n")

    df_prev = pd.read_csv(file, sep="\t")

    if not df_prev.empty:
        same_row = (
            (df_prev["Original file"].astype(str) == new_original_file)
            & (df_prev["Resolution"].astype(str) == new_resolution)
            & (df_prev["Clusters to remove"].astype(str) == new_clusters)
            & (df_prev["Cells"].astype(str) == new_cells)
        )

        if same_row.any():
            logger.info("Identical row already logged; not adding it again.")
        else:
            with open(file, "a", encoding="utf-8") as f:
                f.write(f"{new_original_file}\t{new_resolution}\t{new_clusters}\t{new_cells}\n")
            logger.info("cells_to_remove_now: %d", len(cells_to_remove_now))
    else:
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"{new_original_file}\t{new_resolution}\t{new_clusters}\t{new_cells}\n")
        logger.info("cells_to_remove_now: %d", len(cells_to_remove_now))

    df_prev = pd.read_csv(file, sep="\t")
    cells_to_remove: list = []

    if "Cells" in df_prev.columns and not df_prev.empty:
        for _, prev_row in df_prev.iterrows():
            prev_cells = prev_row["Cells"]
            if pd.notna(prev_cells):
                cells = [c.strip() for c in str(prev_cells).split(",") if c.strip()]
                cells_to_remove.extend(cells)
                logger.info(
                    "cells_to_remove_individual: clusters %s - %d",
                    prev_row["Clusters to remove"], len(cells),
                )

    cells_to_remove = list(dict.fromkeys(cells_to_remove))
    logger.info("cells_to_remove_all: %d", len(cells_to_remove))

    return cells_to_remove


def RunScibMetricsWithLeiden(
    adata_ref: AnnData,
    adata_int: AnnData,
    organism: str,
    *,
    batch_key: str = "sample",
    embed: str = "X_pca",
    type_: str = "embed",
    leiden_key: str = "leiden_tmp",
    leiden_resolution: float = 1.0,
    random_state: int = 42,
):
    """
    Compute scIB integration-quality metrics using Leiden clusters (computed
    on `adata_int`) as proxy biological labels.

    Parameters
    ----------
    adata_ref
        Unintegrated/reference AnnData. Must already contain
        `adata_ref.obsm[embed]` (e.g. the pre-integration PCA from
        `NormalizeHvgPcaKnn`) -- this is the baseline the integrated object
        is compared against.

    adata_int
        Integrated AnnData for the method being scored.

    organism
        "human" or "mouse", passed to scib's `cell_cycle_` metric so it uses
        species-correct cell-cycle gene sets.

        FIX: the original notebooks called `scib.metrics.metrics(...,
        cell_cycle_=True, ...)` without ever passing `organism`, silently
        relying on scib's default (`organism="mouse"`). If the dataset being
        benchmarked is human, the cell-cycle-conservation score would be
        computed from the wrong species' marker genes. This parameter is now
        required with no default, forcing an explicit choice per dataset.

    embed
        `adata.obsm` key holding the embedding to evaluate. Prefer picking
        this from `SCIB_EMBED_BY_METHOD[method]` rather than guessing.

    type_
        scIB output type: "embed" for a new embedding (Harmony/Scanorama/
        scVI/Seurat-PCA) or "knn" for a modified neighbor graph with no new
        embedding (BBKNN). Prefer `SCIB_TYPE_BY_METHOD[method]`.

    Notes
    -----
    Only label-independent metrics (PCR_batch, iLISI, hvg_score,
    cell_cycle_conservation) and the label-dependent `silhouette_` (ASW) are
    enabled here; NMI/ARI/isolated-labels/graph-connectivity are disabled
    because they compare against `leiden_key`, i.e. clusters derived from
    `adata_int` itself -- using them as ground-truth "cell type" labels for
    metrics computed on that same embedding is circular. Note that
    `silhouette_`/ASW-label is *also* computed from the same Leiden proxy
    labels and is subject to the same circularity concern (and to the
    independently-documented shortcomings of silhouette-based single-cell
    integration metrics in general, e.g. Nature Biotechnology 2025,
    "Shortcomings of silhouette in single-cell integration benchmarking") --
    if you drop `ASW_label`/`ASW_label/batch` downstream (as the original
    "Arruma" summary step does), consider setting `silhouette_=False` here
    to skip the wasted computation.
    """
    import scib as sciblib

    if embed not in adata_ref.obsm:
        raise KeyError(
            f"adata_ref.obsm['{embed}'] not found. adata_ref must carry its own "
            "pre-integration embedding (e.g. from NormalizeHvgPcaKnn) before "
            "calling RunScibMetricsWithLeiden."
        )

    sc.tl.leiden(
        adata_int,
        resolution=leiden_resolution,
        key_added=leiden_key,
        random_state=random_state,
    )

    # Copy proxy labels to the reference object (scIB checks both).
    adata_ref.obs[leiden_key] = adata_int.obs[leiden_key].reindex(adata_ref.obs_names).astype(str)

    # Alias for scIB internals that default to reading "X_emb".
    adata_int.obsm["X_emb"] = adata_int.obsm[embed]
    # FIX: the original code set `adata_ref.obsm["X_emb"] = adata_int.obsm[embed]`
    # here -- i.e. it copied the *integrated* embedding onto the reference
    # object too, so "before" and "after" pointed at the literal same array.
    # Any metric comparing pre- vs post-integration embeddings (e.g.
    # PCR_batch) would then trivially see no difference, regardless of how
    # good or bad the integration actually was. The reference must keep its
    # own (pre-integration) embedding.
    adata_ref.obsm["X_emb"] = adata_ref.obsm[embed]

    return sciblib.metrics.metrics(
        adata_ref,
        adata_int,
        batch_key=batch_key,
        label_key=leiden_key,
        embed=embed,
        organism=organism,
        pcr_=True,
        silhouette_=True,
        ilisi_=True,
        hvg_score_=True,
        cell_cycle_=True,
        isolated_labels_=False,
        nmi_=False,
        ari_=False,
        graph_conn_=False,
        n_cores=16,
        type_=type_,
        verbose=False,
    )


def BuildCombinationsDictAndParamsDf(config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Build the grid-search combinations (flavor x n_top_genes x n_pcs) used by
    `RunIntegrationTests`, plus a matching parameters DataFrame.

    Parameters
    ----------
    config
        Dict with keys "flavors", "n_top_genes_list", "n_pcs_list", each an
        iterable of values to grid over.
    """
    combinations_dict = {}
    rows = []
    run_id = 1

    for flavor in config["flavors"]:
        for n_top in config["n_top_genes_list"]:
            for n_pcs in config["n_pcs_list"]:
                combinations_dict[run_id] = {
                    "flavor": flavor,
                    "n_top_genes": n_top,
                    "n_pcs": n_pcs,
                }
                rows.append(
                    {
                        "run_id": run_id,
                        "flavor": flavor,
                        "n_top_genes": n_top,
                        "n_pcs": n_pcs,
                    }
                )
                run_id += 1

    params_df = pd.DataFrame(rows)
    return combinations_dict, params_df


def RunIntegrationTests(
    adata: AnnData,
    combinations_dict: dict,
    out_dir: str,
    versao: str,
    batch_key: str,
    input_layer: str,
    counts_layer: str,
    log_layer: str,
    organism: str,
    random_state: int = 42,
    methods: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid-search benchmark: for each parameter combination in
    `combinations_dict`, preprocess, apply each requested integration method,
    and score it with `RunScibMetricsWithLeiden`.

    Parameters
    ----------
    adata
        Reference AnnData (raw, post-QC). A copy is preprocessed independently
        for each run (each combination may use different flavor/n_top_genes/n_pcs).

    organism
        "human" or "mouse" -- required, see `RunScibMetricsWithLeiden`.

    Notes
    -----
    `adata` itself (not `adata_run`) is used as `adata_ref` for every run's
    scIB scoring, even though each run preprocesses its own copy with
    different HVG/PCA settings. This means the "unintegrated baseline" is the
    same across the whole grid rather than being recomputed per-combination.
    scib's own docs describe `adata`/`adata_ref` as "unintegrated, preprocessed" --
    consider whether your grid search should compare each run against its own
    matching baseline (same flavor/n_top_genes/n_pcs, no batch correction) or
    against one fixed baseline as done here; this is a benchmarking-design
    choice, not something this function decides for you.

    Writes one `{out_dir}/{versao}.scib_metrics_run_{run_id}.csv` per run,
    plus `{out_dir}/{versao}.scib_params_all_runs.csv` and
    `{out_dir}/{versao}.scib_metrics_all_runs.csv` summaries.
    """
    if methods is None:
        methods = ["bbknn", "seurat", "scanorama", "harmony", "scvi"]

    methods = set(methods)

    all_metric_dfs = []
    param_rows = []

    for run_id, params in combinations_dict.items():
        flavor = params.get("flavor")
        n_top_genes = params.get("n_top_genes", params.get("n_top"))
        n_pcs = params.get("n_pcs")
        n_neighbors = params.get("n_neighbors", 15)

        logger.info(
            "RUN %s | flavor=%s, n_top_genes=%s, n_pcs=%s, n_neighbors=%s",
            run_id, flavor, n_top_genes, n_pcs, n_neighbors,
        )

        adata_run = adata.copy()

        adata_run = NormalizeHvgPcaKnn(
            adata_run,
            batch_key=batch_key,
            input_layer=input_layer,
            flavor=flavor,
            n_top_genes=n_top_genes,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            random_state=random_state,
            plot_before_integration=False,
        )

        adata_hvg = adata_run[:, adata_run.var["highly_variable"]].copy()

        dic_int = ApplyIntegrationMethods(
            adata_hvg,
            run_id=run_id,
            out_dir=out_dir,
            batch_key=batch_key,
            counts_layer=counts_layer,
            log_layer=log_layer,
            bbknn="bbknn" in methods,
            seurat="seurat" in methods,
            scanorama="scanorama" in methods,
            harmony="harmony" in methods,
            scvi_="scvi" in methods,
            n_pcs=n_pcs,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        run_metrics = pd.DataFrame()

        for method, adata_int in dic_int.items():
            logger.info("Metrics for %s", method)

            metrics_series = RunScibMetricsWithLeiden(
                adata,
                adata_int,
                organism=organism,
                batch_key=batch_key,
                embed=SCIB_EMBED_BY_METHOD[method],
                type_=SCIB_TYPE_BY_METHOD[method],
            )

            col_name = f"{method}_{run_id}"
            run_metrics[col_name] = metrics_series

            row = {"col": col_name, "run_id": run_id, "method": method}
            row.update(params)
            if "n_neighbors" not in row:
                row["n_neighbors"] = n_neighbors

            param_rows.append(row)

        run_metrics.dropna(how="all", inplace=True)
        run_metrics.to_csv(f"{out_dir}/{versao}.scib_metrics_run_{run_id}.csv", sep="\t")

        all_metric_dfs.append(run_metrics)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(f"{out_dir}/{versao}.scib_params_all_runs.csv", sep="\t", index=False)

    if all_metric_dfs:
        all_metrics_df = pd.concat(all_metric_dfs, axis=1)
    else:
        all_metrics_df = pd.DataFrame()

    all_metrics_df.to_csv(f"{out_dir}/{versao}.scib_metrics_all_runs.csv", sep="\t")

    return all_metrics_df, params_df
