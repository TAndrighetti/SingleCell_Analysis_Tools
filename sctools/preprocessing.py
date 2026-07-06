"""
sctools.preprocessing
=====================

Preprocessing functions for single-cell RNA-seq data after QC filtering and
ambient RNA correction.

Main workflow:
    NormalizeHvgPcaKnn()

Reusable steps:
    CheckNormalizedLayer()
    NormalizeLog1pFromLayer()
    RunHighlyVariableGenes()
    ResolveNPcs()
    RunPcaOnHvgs()
    RunNeighborsAndUmap()
"""

from __future__ import annotations

import logging
import random

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def CheckNormalizedLayer(X) -> None:
    """
    Validate that a matrix looks like normalized/log-transformed data.

    Several downstream steps (HVG selection, PCA, ALRA imputation, ...)
    expect a non-negative matrix, usually library-normalized and
    log-transformed. This raises if it instead looks like scaled data
    (e.g. after `sc.pp.scale`), which can contain negative values.
    """
    if issparse(X):
        if X.data.size > 0 and np.nanmin(X.data) < 0:
            raise ValueError(
                "Input contains negative values. "
                "Use a normalized/log-transformed layer, not scaled data."
            )
    else:
        if np.nanmin(X) < 0:
            raise ValueError(
                "Input contains negative values. "
                "Use a normalized/log-transformed layer, not scaled data."
            )


def SetRandomSeed(random_state: int) -> None:
    """
    Set Python and NumPy random seeds for reproducibility.

    This changes the global random state. In general-purpose libraries this is
    often avoided, but here it is intentionally kept because the pipeline is
    expected to be reproducible across notebook runs.
    """
    np.random.seed(random_state)
    random.seed(random_state)


def NormalizeLog1pFromLayer(
    adata: AnnData,
    *,
    input_layer: str,
    target_sum: float = 1e4,
    output_layer: str | None = None,
) -> str:
    """
    Copy a count layer into `.X`, normalize total counts, apply log1p, and save
    the result as a new layer.

    This function does not modify the original count layer.

    Parameters
    ----------
    adata
        AnnData object.

    input_layer
        Layer containing count-scale data, for example:
        "counts", "soupX_counts", "cellbender_counts", or "QC_filtered".

    target_sum
        Target total counts per cell after library-size normalization.

    output_layer
        Name of the layer where normalized + log1p data will be stored.
        If None, uses `f"{input_layer}_log1p"`.

    Returns
    -------
    output_layer
        Name of the layer containing normalized + log1p data.
    """
    if input_layer not in adata.layers:
        raise KeyError(
            f"Layer '{input_layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}."
        )

    if output_layer is None:
        output_layer = f"{input_layer}_log1p"

    # Put the selected count matrix in `.X`.
    # After this function, `.X` will become normalized + log1p.
    adata.X = adata.layers[input_layer].copy()

    # Normalize each cell to the same total count depth.
    sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
    )

    # Log-transform normalized counts.
    sc.pp.log1p(adata)

    # Store the normalized + log1p matrix for reuse.
    adata.layers[output_layer] = adata.X.copy()

    return output_layer


def RunHighlyVariableGenes(
    adata: AnnData,
    *,
    input_layer: str,
    batch_key: str | None = None,
    n_top_genes: int = 2000,
    flavor: str = "cell_ranger",
    check_integer_counts: bool = True,
) -> None:
    """
    Select highly variable genes.

    For `flavor="seurat"` and `flavor="cell_ranger"`, Scanpy expects
    normalized + log1p values, so HVGs are computed from `.X`.

    For `flavor="seurat_v3"`, Scanpy expects count-scale data, so HVGs are
    computed from `adata.layers[input_layer]`.

    Parameters
    ----------
    input_layer
        Count layer used only when `flavor="seurat_v3"`.

    batch_key
        Optional batch/sample column in `adata.obs`.
        Use technical/sample-level variables such as "sample", "library",
        "run", or "batch". Avoid using biological variables such as condition
        or genotype unless this is intentional.

    n_top_genes
        Number of HVGs to select.

    flavor
        HVG method passed to Scanpy.

    check_integer_counts
        Passed to Scanpy when `flavor="seurat_v3"`.
        If the layer is not integer-like, Scanpy may warn.
    """
    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")

    hvg_kwargs = {
        "n_top_genes": n_top_genes,
        "flavor": flavor,
    }

    if batch_key is not None:
        hvg_kwargs["batch_key"] = batch_key

    if flavor == "seurat_v3":
        if input_layer not in adata.layers:
            raise KeyError(
                f"Layer '{input_layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}."
            )

        hvg_kwargs["layer"] = input_layer
        hvg_kwargs["check_values"] = check_integer_counts

    try:
        sc.pp.highly_variable_genes(
            adata,
            **hvg_kwargs,
        )

    except TypeError as error:
        # Compatibility fallback for older Scanpy versions that do not accept
        # `check_values`.
        if "check_values" not in str(error):
            raise

        hvg_kwargs.pop("check_values", None)

        sc.pp.highly_variable_genes(
            adata,
            **hvg_kwargs,
        )


def ResolveNPcs(
    adata: AnnData,
    *,
    requested_n_pcs: int,
    use_highly_variable: bool = True,
    auto_reduce: bool = True,
) -> int:
    """
    Validate the requested number of principal components.

    Why this check exists
    ---------------------
    Even if you request 1000 or 2000 HVGs, PCA can still fail if the number of
    cells is small. With `svd_solver="arpack"`, the number of PCs must be
    strictly smaller than:

        min(number of cells, number of genes used for PCA)

    For example:
        - 1000 HVGs and 5000 cells -> n_pcs=35 is fine.
        - 1000 HVGs and 20 cells   -> n_pcs=35 is not valid.

    This function prevents PCA from failing in small test objects, subsets,
    or rare edge cases after filtering.

    Parameters
    ----------
    requested_n_pcs
        Number of PCs requested by the user.

    use_highly_variable
        If True, checks the number of HVGs. If False, checks all genes.

    auto_reduce
        If True, reduce `requested_n_pcs` to the maximum valid value and log a
        warning. If False, raise an error.

    Returns
    -------
    n_pcs_used
        Valid number of PCs to use.
    """
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise KeyError(
                "Missing adata.var['highly_variable']. "
                "Run RunHighlyVariableGenes() before PCA."
            )

        n_genes_for_pca = int(adata.var["highly_variable"].sum())
        gene_set_name = "HVGs"

    else:
        n_genes_for_pca = int(adata.n_vars)
        gene_set_name = "genes"

    if n_genes_for_pca < 2:
        raise ValueError(
            f"Only {n_genes_for_pca} {gene_set_name} available for PCA. "
            "PCA requires at least 2 variables."
        )

    if adata.n_obs < 2:
        raise ValueError(
            f"Only {adata.n_obs} cells available for PCA. "
            "PCA requires at least 2 cells."
        )

    max_n_pcs = min(adata.n_obs - 1, n_genes_for_pca - 1)

    if requested_n_pcs <= max_n_pcs:
        return int(requested_n_pcs)

    message = (
        f"Requested n_pcs={requested_n_pcs}, but the maximum valid value is "
        f"{max_n_pcs} for n_obs={adata.n_obs} and "
        f"n_{gene_set_name}={n_genes_for_pca}."
    )

    if not auto_reduce:
        raise ValueError(message)

    logger.warning("%s Using n_pcs=%d instead.", message, max_n_pcs)

    return int(max_n_pcs)


def RunPcaOnHvgs(
    adata: AnnData,
    *,
    n_pcs: int = 35,
    scale_for_pca: bool = True,
    max_value: float | None = 10,
    random_state: int = 42,
) -> str:
    """
    Compute PCA using highly variable genes.

    If `scale_for_pca=True`, scaling is applied only to a temporary HVG object.
    This keeps the original `.X` as normalized + log1p, while PCA is computed
    from scaled log1p HVGs.

    This is useful because:
        - `.X` remains interpretable as log-normalized expression.
        - PCA can still benefit from scaling.
        - the original count layer remains untouched.

    Returns
    -------
    pca_input_state
        Either "log1p" or "scaled_log1p".
    """
    if "highly_variable" not in adata.var:
        raise KeyError(
            "Missing adata.var['highly_variable']. "
            "Run RunHighlyVariableGenes() before RunPcaOnHvgs()."
        )

    n_pcs_used = ResolveNPcs(
        adata,
        requested_n_pcs=n_pcs,
        use_highly_variable=True,
        auto_reduce=True,
    )

    hvg_mask = adata.var["highly_variable"].to_numpy()

    # Work on a temporary object containing only HVGs.
    # This avoids changing `.X` in the original AnnData when scaling is enabled.
    adata_hvg = adata[:, hvg_mask].copy()

    pca_input_state = "log1p"

    if scale_for_pca:
        sc.pp.scale(
            adata_hvg,
            max_value=max_value,
        )
        pca_input_state = "scaled_log1p"

    sc.tl.pca(
        adata_hvg,
        n_comps=n_pcs_used,
        svd_solver="arpack",
        random_state=random_state,
    )

    # Copy PCA coordinates back to the full object.
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"].copy()

    # Copy PCA metadata.
    adata.uns["pca"] = adata_hvg.uns["pca"].copy()
    adata.uns["pca"].setdefault("params", {})
    adata.uns["pca"]["params"]["use_highly_variable"] = True
    adata.uns["pca"]["params"]["pca_input_state"] = pca_input_state
    adata.uns["pca"]["params"]["n_pcs_used"] = int(n_pcs_used)

    # Store PCA loadings in the full object.
    # HVGs receive their real loadings; non-HVGs receive zero loadings.
    pcs = np.zeros((adata.n_vars, adata_hvg.varm["PCs"].shape[1]))
    pcs[hvg_mask, :] = adata_hvg.varm["PCs"]
    adata.varm["PCs"] = pcs

    return pca_input_state


def RunNeighborsAndUmap(
    adata: AnnData,
    *,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    random_state: int = 42,
) -> None:
    """
    Build the k-nearest-neighbor graph and compute UMAP.

    Parameters
    ----------
    n_neighbors
        Number of neighbors used to build the graph.

    use_rep
        Representation used for graph construction. Usually "X_pca".

    random_state
        Random seed used by Scanpy.
    """
    if use_rep not in adata.obsm:
        raise KeyError(
            f"Representation '{use_rep}' not found in adata.obsm. "
            "Run PCA first or choose another representation."
        )

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=random_state,
    )

    sc.tl.umap(
        adata,
        random_state=random_state,
    )


def StorePreprocessingParams(
    adata: AnnData,
    *,
    input_layer: str,
    log1p_layer: str,
    batch_key: str | None,
    n_top_genes: int,
    flavor: str,
    target_sum: float,
    n_pcs_requested: int,
    n_neighbors: int,
    scale_for_pca: bool,
    max_value: float | None,
    pca_input_state: str,
    random_state: int,
) -> None:
    """
    Store preprocessing parameters in `adata.uns`.
    """
    n_pcs_used = int(adata.obsm["X_pca"].shape[1])

    adata.uns.setdefault("preprocessing_params", {})

    adata.uns["preprocessing_params"]["NormalizeHvgPcaKnn"] = {
        "input_layer": input_layer,
        "log1p_layer": log1p_layer,
        "x_state_after_function": "normalized_log1p",
        "counts_layer_preserved": input_layer,
        "batch_key": batch_key,
        "n_top_genes": int(n_top_genes),
        "n_hvgs": int(adata.var["highly_variable"].sum()),
        "flavor": flavor,
        "target_sum": float(target_sum),
        "n_pcs_requested": int(n_pcs_requested),
        "n_pcs_used": n_pcs_used,
        "n_neighbors": int(n_neighbors),
        "scale_for_pca": bool(scale_for_pca),
        "max_value": max_value,
        "pca_input_state": pca_input_state,
        "random_state": int(random_state),
    }


def NormalizeHvgPcaKnn(
    adata: AnnData,
    *,
    input_layer: str,
    batch_key: str | None = None,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
    target_sum: float = 1e4,
    n_pcs: int = 35,
    n_neighbors: int = 15,
    scale_for_pca: bool = True,
    max_value: float | None = 10,
    check_integer_counts: bool = True,
    random_state: int = 42,
    copy: bool = False,
) -> AnnData:
    """
    Run normalization, HVG selection, PCA, neighbor graph construction, and UMAP.

    This function is intended for post-QC scRNA-seq data.

    Expected input:
        `adata.layers[input_layer]` should contain count-scale data after the
        preprocessing decisions you want to use.

    Examples of valid input layers:
        - "counts"
        - "soupX_counts"
        - "cellbender_counts"
        - "QC_filtered"

    What this function does:
        1. Copies `adata.layers[input_layer]` into `.X`.
        2. Normalizes total counts.
        3. Applies log1p.
        4. Saves normalized + log1p values in `adata.layers[f"{input_layer}_log1p"]`.
        5. Selects HVGs.
        6. Computes PCA on HVGs.
        7. Builds the neighbor graph.
        8. Computes UMAP.

    Important:
        - `.X` is intentionally left as normalized + log1p.
        - `adata.layers[input_layer]` is preserved.
        - If `scale_for_pca=True`, scaling is used only for PCA on a temporary
          HVG object. The original `.X` remains normalized + log1p.
        - For scVI, use the count layer, not the log1p layer.

    Parameters
    ----------
    input_layer
        Required. Count-scale layer to use as input.

    batch_key
        Optional column in `adata.obs` for per-batch HVG selection.
        Prefer sample/library/run/batch. Avoid condition/genotype/treatment
        unless this is intentional.

    n_top_genes
        Number of HVGs to select.

    flavor
        HVG method. If "seurat_v3", HVG selection uses `input_layer` as counts.

    target_sum
        Target sum for total-count normalization.

    n_pcs
        Number of PCs requested. If too high for the object, it is reduced
        safely by `ResolveNPcs`.

    n_neighbors
        Number of neighbors for graph construction.

    scale_for_pca
        Whether to scale log1p HVGs before PCA.

    max_value
        Maximum value for `sc.pp.scale`.

    check_integer_counts
        Passed to Scanpy for `flavor="seurat_v3"`.

    random_state
        Random seed.

    copy
        If True, operate on a copy. If False, modify in place.

    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    if copy:
        adata = adata.copy()

    SetRandomSeed(random_state)

    log1p_layer = NormalizeLog1pFromLayer(
        adata,
        input_layer=input_layer,
        target_sum=target_sum,
    )

    RunHighlyVariableGenes(
        adata,
        input_layer=input_layer,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        flavor=flavor,
        check_integer_counts=check_integer_counts,
    )

    pca_input_state = RunPcaOnHvgs(
        adata,
        n_pcs=n_pcs,
        scale_for_pca=scale_for_pca,
        max_value=max_value,
        random_state=random_state,
    )

    RunNeighborsAndUmap(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        random_state=random_state,
    )

    StorePreprocessingParams(
        adata,
        input_layer=input_layer,
        log1p_layer=log1p_layer,
        batch_key=batch_key,
        n_top_genes=n_top_genes,
        flavor=flavor,
        target_sum=target_sum,
        n_pcs_requested=n_pcs,
        n_neighbors=n_neighbors,
        scale_for_pca=scale_for_pca,
        max_value=max_value,
        pca_input_state=pca_input_state,
        random_state=random_state,
    )

    logger.info(
        "NormalizeHvgPcaKnn finished: %d cells x %d genes, %d HVGs, %d PCs, "
        "n_neighbors=%d, input_layer='%s', .X='normalized_log1p'.",
        adata.n_obs,
        adata.n_vars,
        int(adata.var["highly_variable"].sum()),
        int(adata.obsm["X_pca"].shape[1]),
        n_neighbors,
        input_layer,
    )

    return adata