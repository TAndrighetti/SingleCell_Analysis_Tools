"""
sctools.integration.methods
=============================

One function per batch-integration method. Each function takes an HVG-subset
AnnData (with `.obsm["X_pca"]` already computed by
`preprocessing.NormalizeHvgPcaKnn`), works on and returns a copy, and imports
its heavy dependency only inside the function that needs it.

These functions should usually be run on an HVG-subset AnnData.

For official/full analyses, keep the full AnnData as the final object, but
learn the integrated representation on HVGs and then attach embeddings,
UMAPs, and clusters back to the full object.

- Harmony/Scanorama/scVI/Seurat are embedding/matrix-based: each writes a
  corrected representation to adata.obsm[...] (or a layer, for Seurat).
- Most methods expect `adata_hvg.obsm["X_pca"]` from preprocessing.
scVI and Seurat do not strictly require pre-existing PCA, but in the standard
workflow the same HVG object is used for consistency.

Note: BBKNN (graph-based -- rewrites the neighbor graph, no corrected
embedding) is currently retired. Its implementation is archived, unchanged,
in `sctools/non_used_functions.py`; it isn't wired into `SUPPORTED_METHODS`
or `_METHOD_FUNCS` right now.
"""

from __future__ import annotations

import inspect
import logging

import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("harmony", "scanorama", "scvi", "seurat")

# adata.obsm key holding each method's corrected representation.
SCIB_EMBED_BY_METHOD = {
    "harmony": "X_harmony",
    "scanorama": "X_scanorama",
    "scvi": "X_scVI",
    "seurat": "X_pca",
}


def FilterSupportedKwargs(func, kwargs: dict) -> dict:
    """
    Drop keys from `kwargs` that `func` doesn't accept as a parameter.

    Method signatures vary (e.g. `RunScanoramaIntegration` takes neither
    `n_pcs` nor `random_state`), so callers that build one generic kwargs
    dict for whichever method is being dispatched (`RunIntegrationBenchmark`,
    `RunIntegrationComplete`) need this instead of hardcoding, per caller,
    which methods accept which base kwargs.
    """
    accepted = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


def _RequirePca(adata: AnnData) -> None:
    if "X_pca" not in adata.obsm:
        raise ValueError("X_pca not found. Run NormalizeHvgPcaKnn first.")


def RunHarmonyIntegration(
    adata_hvg: AnnData,
    *,
    batch_key: str = "sample",
    basis: str = "X_pca",
    n_pcs: int = 35,
    random_state: int = 42,
) -> AnnData:
    """Harmony integration. 
    - Corrects the PCA embedding in place, writing to adata.obsm["X_harmony"].
    - Embedding-based: writes adata.obsm["X_harmony"].
    - basis: which representation to use for Harmony (usually "X_pca").

    # --------------------------
    # Harmony is embedding-based integration.
    #
    # It does NOT create:
    #   - a corrected expression matrix
    #   - a corrected layer
    #
    # Harmony uses:
    #   - adata.obsm["X_pca"] as input representation
    #   - adata.obs[batch_key] to know which cells belong to each batch/sample
    #
    # Harmony writes the corrected embedding to:
    #   - adata.obsm["X_harmony"]
    #
    # Therefore, the "integrated result" of Harmony is the corrected PCA-like
    # embedding stored in adata.obsm["X_harmony"].
    #
    # To visualize or cluster Harmony results, compute neighbors using
    # use_rep="X_harmony", then run UMAP and Leiden from that corrected graph.
    #
    # Example downstream workflow:
    #
    #   # 1. Run Harmony.
    #   adata_harmony = RunHarmonyIntegration(
    #       adata_hvg,
    #       batch_key="sample",
    #       random_state=42,
    #   )
    #
    #   # 2. Build the neighbor graph from the Harmony-corrected embedding.
    #   sc.pp.neighbors(
    #       adata_harmony,
    #       use_rep="X_harmony",
    #       n_neighbors=15,
    #       random_state=42,
    #   )
    #
    #   # 3. Compute UMAP from the Harmony-based neighbor graph.
    #   sc.tl.umap(
    #       adata_harmony,
    #       random_state=42,
    #   )
    #
    #   # 4. Run Leiden clustering on the Harmony-based neighbor graph.
    #   sc.tl.leiden(
    #       adata_harmony,
    #       resolution=1.0,
    #       key_added="leiden_res1",
    #       random_state=42,
    #   )
    #
    #   # 5. Plot the integrated UMAP.
    #   sc.pl.umap(
    #       adata_harmony,
    #       color=["sample", "leiden_res1"],
    #       wspace=0.4,
    #   )
    #
    # IMPORTANT:
    # Do not interpret adata.X or any layer as Harmony-corrected expression.
    # Harmony corrects the PCA embedding, not the expression matrix.
    """
    import scanpy.external as sce

    adata = adata_hvg.copy()
    _RequirePca(adata)

    logger.info("Running Harmony (batch_key=%s)", batch_key)

    sce.pp.harmony_integrate(
        adata,
        key=batch_key,
        basis=basis,
        adjusted_basis="X_harmony",
        verbose=1,
    )
    return adata


def RunScanoramaIntegration(
    adata_hvg: AnnData,
    *,
    batch_key: str = "sample",
    basis: str = "X_pca",
) -> AnnData:
    """Scanorama integration. Embedding-based: writes adata.obsm["X_scanorama"].
    
    # ----------------------------
    # Scanorama is embedding-based integration.
    #
    # It does NOT create:
    #   - a corrected expression matrix
    #   - a corrected layer
    #
    # In the Scanpy external API, Scanorama uses:
    #   - adata.obsm["X_pca"] as input representation
    #   - adata.obs[batch_key] to know which cells belong to each batch/sample
    #
    # Scanorama writes the corrected embedding to:
    #   - adata.obsm["X_scanorama"]
    #
    # Therefore, the "integrated result" of Scanorama is the corrected embedding
    # stored in adata.obsm["X_scanorama"].
    #
    # To visualize or cluster Scanorama results, compute neighbors using
    # use_rep="X_scanorama", then run UMAP and Leiden from that corrected graph.
    #
    # Example downstream workflow:
    #
    #   # 1. Run Scanorama.
    #   adata_scanorama = RunScanoramaIntegration(
    #       adata_hvg,
    #       batch_key="sample",
    #       random_state=42,
    #   )
    #
    #   # 2. Build the neighbor graph from the Scanorama-corrected embedding.
    #   sc.pp.neighbors(
    #       adata_scanorama,
    #       use_rep="X_scanorama",
    #       n_neighbors=15,
    #       random_state=42,
    #   )
    #
    #   # 3. Compute UMAP from the Scanorama-based neighbor graph.
    #   sc.tl.umap(
    #       adata_scanorama,
    #       random_state=42,
    #   )
    #
    #   # 4. Run Leiden clustering on the Scanorama-based neighbor graph.
    #   sc.tl.leiden(
    #       adata_scanorama,
    #       resolution=1.0,
    #       key_added="leiden_res1",
    #       random_state=42,
    #   )
    #
    #   # 5. Plot the integrated UMAP.
    #   sc.pl.umap(
    #       adata_scanorama,
    #       color=["sample", "leiden_res1"],
    #       wspace=0.4,
    #   )
    #
    # IMPORTANT:
    # Do not interpret adata.X or any layer as Scanorama-corrected expression.
    # In this workflow, Scanorama corrects the embedding used for graph/UMAP/
    # clustering, not the expression matrix used for downstream DE."""

    import scanpy.external as sce

    adata = adata_hvg.copy()
    _RequirePca(adata)

    logger.info("Running Scanorama (batch_key=%s)", batch_key)

    sce.pp.scanorama_integrate(adata, 
                               key=batch_key, 
                               basis=basis, 
                               adjusted_basis="X_scanorama", 
                               verbose=1)
    return adata


def RunScviIntegration(
    adata_hvg: AnnData,
    *,
    batch_key: str = "sample",
    counts_layer: str = "QC_filtered",
    covariate_keys: list[str] | None = None,
    n_latent: int = 30,
    max_epochs: int = 200,
    early_stopping: bool = True,
    random_state: int = 42,
) -> AnnData:
    '''   # -----------------------
    # scVI is model-based integration.
    #
    # It does NOT use:
    #   - adata.obsm["X_pca"]
    #   - adata.X log1p
    #   - a precomputed neighbor graph
    #
    # scVI uses:
    #   - adata.layers[counts_layer] as count-scale input
    #   - adata.obs[batch_key] as the batch variable
    #   - optional adata.obs[covariate_keys] as extra categorical covariates
    #
    # scVI writes the integrated latent representation to:
    #   - adata.obsm["X_scVI"]
    #
    # Therefore, the "integrated result" of scVI is the latent embedding stored in
    # adata.obsm["X_scVI"].
    #
    # Example downstream workflow:
    #
    #   adata_scvi = RunScviIntegration(
    #       adata_hvg,
    #       batch_key="sample",
    #       counts_layer="soupX_counts",
    #       n_latent=30,
    #       random_state=42,
    #   )
    #
    #   sc.pp.neighbors(
    #       adata_scvi,
    #       use_rep="X_scVI",
    #       n_neighbors=15,
    #   )
    #
    #   sc.tl.umap(
    #       adata_scvi,
    #       random_state=42,
    #   )
    #
    #   sc.tl.leiden(
    #       adata_scvi,
    #       resolution=1.0,
    #       key_added="leiden_res1",
    #       random_state=42,
    #   )
    #
    #   sc.pl.umap(
    #       adata_scvi,
    #       color=["sample", "leiden_res1"],
    #       wspace=0.4,
    #   )
    #
    # IMPORTANT:
    # Do not pass log1p-normalized data to scVI.
    # Use a count-scale layer such as "counts", "soupX_counts", or
    # "cellbender_counts".'''
 
    import scvi

    if counts_layer not in adata_hvg.layers:
        raise KeyError(
            f"counts_layer '{counts_layer}' not found. "
            f"Available layers: {list(adata_hvg.layers.keys())}."
        )

    if batch_key not in adata_hvg.obs:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")

    adata = adata_hvg.copy()

    scvi.settings.seed = random_state

    logger.info(
        "Running scVI (batch_key=%s, counts_layer=%s, n_latent=%s)",
        batch_key,
        counts_layer,
        n_latent,
    )

    scvi.model.SCVI.setup_anndata(
        adata,
        layer=counts_layer,
        batch_key=batch_key,
        categorical_covariate_keys=covariate_keys,
    )

    model = scvi.model.SCVI(
        adata,
        n_latent=n_latent,
    )

    model.train(
        max_epochs=max_epochs,
        early_stopping=early_stopping,
    )

    adata.obsm["X_scVI"] = model.get_latent_representation()

    return adata

###########################################
### SEURAT ANCHORS INTEGRATION
# ---------------------------------
# Seurat Anchors is matrix-based integration.
#
# It does NOT use:
#   - adata.obsm["X_pca"] as the main integration input
#   - a precomputed neighbor graph
#
# Seurat Anchors uses:
#   - adata.layers[counts_layer] as the Seurat "counts" slot
#   - adata.layers[log_layer] as the Seurat "data" slot
#   - adata.obs[batch_key] to split cells into batches/samples
#   - integration features, usually HVGs
#
# In this pipeline, the input object is already subsetted to HVGs.
# Therefore, anchor_features="all_hvgs" means:
#   use all HVGs from the current run as integration features.
#
# Alternatively:
#   anchor_features=2000 lets Seurat select 2000 integration features.
#   anchor_features=[...] passes an explicit gene list.
#
# Seurat returns an integrated assay. With LogNormalize, the integrated values
# are corrected log-normalized expression values from the "integrated" assay.
#
# This function stores the integrated matrix in:
#   - adata.layers["seurat"]
#
# Then it computes PCA on that integrated matrix and stores:
#   - adata.obsm["X_pca"]
#
# Therefore, the "integrated result" used for downstream graph/UMAP/clustering
# is the PCA computed from adata.layers["seurat"].
#
# Example downstream workflow:
#
#   adata_seurat = RunSeuratAnchorsIntegration(
#       adata_hvg,
#       batch_key="sample",
#       counts_layer="soupX_counts",
#       log_layer="soupX_counts_log1p",
#       anchor_features="all_hvgs",
#       n_pcs=35,
#       random_state=42,
#   )
#
#   sc.pp.neighbors(
#       adata_seurat,
#       use_rep="X_pca",
#       n_neighbors=15,
#   )
#
#   sc.tl.umap(
#       adata_seurat,
#       random_state=42,
#   )
#
#   sc.tl.leiden(
#       adata_seurat,
#       resolution=1.0,
#       key_added="leiden_res1",
#       random_state=42,
#   )
#
#   sc.pl.umap(
#       adata_seurat,
#       color=["sample", "leiden_res1"],
#       wspace=0.4,
#   )
#
# IMPORTANT:
# Do not use adata.layers["seurat"] for final differential expression.
# Use the original count/log-normalized non-integrated data for DE.

def RunSeuratAnchors(
    adata_seurat: AnnData,
    batch_key: str,
    counts_layer: str = "QC_filtered",
    data_layer: str = "logcounts",
    anchor_features: str | int | list[str] = "all_hvgs",
):
    """
    Low-level R bridge: Seurat CCA-anchor integration (via rpy2). Converts
    `adata_seurat` to a Seurat object, splits it by `batch_key`, runs
    `FindIntegrationAnchors()` + `IntegrateData()`, and returns the
    integrated expression matrix (cells x genes).

    counts_layer / data_layer: Seurat's "counts" / "data" slots.

    anchor_features: "all_hvgs" (default, uses every gene in `adata_seurat`)
    | int (let Seurat pick that many features) | list[str] (explicit genes).
    """
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    # Prefer a locally-scoped converter (doesn't mutate rpy2's global state).
    # Falls back to global .activate() -- local-converter-only attempts
    # previously failed for some anndata2ri/Seurat version combinations.
    try:
        from rpy2.robjects.conversion import localconverter
        converter_ctx = localconverter(anndata2ri.converter + pandas2ri.converter)
        converter_ctx.__enter__()
    except Exception:
        pandas2ri.activate()
        anndata2ri.activate()
        converter_ctx = None

    try:
        ro.globalenv["adata_seurat"] = adata_seurat
        ro.globalenv["batch_key"] = batch_key
        ro.globalenv["counts_layer"] = counts_layer
        ro.globalenv["data_layer"] = data_layer

        if anchor_features == "all_hvgs":
            ro.globalenv["anchor_features_mode"] = "all_hvgs"
            ro.globalenv["n_integration_features"] = ro.NULL
            ro.globalenv["anchor_features_vec"] = ro.NULL
        elif isinstance(anchor_features, int):
            ro.globalenv["anchor_features_mode"] = "n_features"
            ro.globalenv["n_integration_features"] = anchor_features
            ro.globalenv["anchor_features_vec"] = ro.NULL
        else:
            ro.globalenv["anchor_features_mode"] = "explicit"
            ro.globalenv["n_integration_features"] = ro.NULL
            ro.globalenv["anchor_features_vec"] = ro.StrVector(list(anchor_features))

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

                    if (anchor_features_mode == "all_hvgs") {
                        features <- rownames(seurat)
                    } else if (anchor_features_mode == "n_features") {
                        features <- SelectIntegrationFeatures(
                            object.list = batch_list,
                            nfeatures = n_integration_features
                        )
                    } else {
                        features <- anchor_features_vec
                    }

                    anchors <- FindIntegrationAnchors(
                        object.list = batch_list,
                        anchor.features = features,
                        verbose = FALSE
                    )

                    integrated <- IntegrateData(
                        anchorset = anchors,
                        verbose = FALSE
                    )

                    DefaultAssay(integrated) <- "integrated"

                    integrated_expr <- tryCatch(
                        GetAssayData(integrated, assay = "integrated", layer = "data"),
                        error = function(e) {
                            # Fallback for Seurat v4 / older SeuratObject (slot= instead of layer=).
                            GetAssayData(integrated, assay = "integrated", slot = "data")
                        }
                    )
                    integrated_expr <- integrated_expr[rownames(seurat), colnames(seurat)]
                    integrated_expr <- t(integrated_expr)
                })
            })
        """)

        return ro.r("integrated_expr")
    finally:
        if converter_ctx is not None:
            converter_ctx.__exit__(None, None, None)


def RunSeuratAnchorsIntegration(
    adata_hvg: AnnData,
    *,
    batch_key: str = "sample",
    counts_layer: str = "QC_filtered",
    log_layer: str = "QC_filtered_log1p",
    anchor_features: str | int | list[str] = "all_hvgs",
    scale_before_pca: bool = True,
    n_pcs: int = 35,
    random_state: int = 42,
) -> AnnData:
    """
    Seurat CCA-anchor integration. Embedding-based: writes the integrated
    expression matrix to adata.layers["seurat"] and a PCA of it to
    adata.obsm["X_pca"]. Does not require a pre-existing X_pca.

    scale_before_pca: scale the integrated matrix before PCA (matches
    Seurat's own `ScaleData()` -> `RunPCA()` convention). Set False to
    reproduce an unscaled PCA.
    """
    adata = adata_hvg.copy()

    # Batch column must be string/categorical-friendly for R.
    adata.obs[batch_key] = adata.obs[batch_key].astype(str)
    adata.uns = {}  # avoid R conversion issues with arbitrary python objects

    logger.info("Running Seurat Anchors (batch_key=%s, anchor_features=%s)", batch_key, anchor_features)

    integrated_mat = RunSeuratAnchors(
        adata,
        batch_key=batch_key,
        counts_layer=counts_layer,
        data_layer=log_layer,
        anchor_features=anchor_features,
    )

    adata.layers["seurat"] = integrated_mat

    # PCA on the Seurat-integrated matrix.
    # Use a temporary AnnData object so the main `adata.X` remains unchanged.
    adata_pca = adata.copy()
    adata_pca.X = adata.layers["seurat"].copy()

    if scale_before_pca:
        sc.pp.scale(
            adata_pca,
            max_value=10,
        )

    sc.tl.pca(
        adata_pca,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=random_state,
    )

    # Copy PCA results back to the main object.
    # Downstream neighbors/UMAP/clustering will use this PCA.
    adata.obsm["X_pca"] = adata_pca.obsm["X_pca"].copy()
    adata.uns["pca"] = adata_pca.uns["pca"].copy()
    adata.varm["PCs"] = adata_pca.varm["PCs"].copy()

    return adata


_METHOD_FUNCS = {
    "harmony": RunHarmonyIntegration,
    "scanorama": RunScanoramaIntegration,
    "scvi": RunScviIntegration,
    "seurat": RunSeuratAnchorsIntegration,
}


