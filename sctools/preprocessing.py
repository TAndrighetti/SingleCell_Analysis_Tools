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


from typing import Literal


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
    layer: str | None = None,
    batch_key: str | None = None,
    n_top_genes: int = 2000,
    flavor: Literal[
        "seurat",
        "cell_ranger",
        "seurat_v3",
        "seurat_v3_paper",
    ] = "cell_ranger",
    check_integer_counts: bool = True,
) -> None:
    """
    Select highly variable genes and store the result in `adata.var`.

    Data expectations
    -----------------
    - flavor="seurat" or "cell_ranger":
        expects normalized + log1p expression values.

    - flavor="seurat_v3" or "seurat_v3_paper":
        expects count-scale data, preferably raw integer UMI counts.

    Parameters
    ----------
    adata
        AnnData object.

    layer
        Layer to use as input. If None, uses `adata.X`.

        For "seurat" and "cell_ranger", this should contain normalized + log1p data.
        For "seurat_v3" and "seurat_v3_paper", this should contain count-scale data.

    batch_key
        Optional column in `adata.obs` used to select HVGs within batches and then merge.
        Prefer technical/sample-level variables such as sample, library, run, or batch.
        Avoid condition/genotype unless this is intentional.

    n_top_genes
        Number of highly variable genes to select.

    flavor
        HVG method passed to Scanpy.

    check_integer_counts
        Passed to Scanpy for "seurat_v3" and "seurat_v3_paper".
    """
    allowed_flavors = {
        "seurat",
        "cell_ranger",
        "seurat_v3",
        "seurat_v3_paper",
    }

    if flavor not in allowed_flavors:
        raise ValueError(
            f"Invalid flavor '{flavor}'. "
            f"Expected one of: {sorted(allowed_flavors)}."
        )

    if n_top_genes <= 0:
        raise ValueError("n_top_genes must be a positive integer.")

    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"Layer '{layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}."
        )

    if batch_key is not None and batch_key not in adata.obs:
        raise KeyError(f"batch_key '{batch_key}' not found in adata.obs.")

    hvg_kwargs = {
        "layer": layer,
        "n_top_genes": n_top_genes,
        "flavor": flavor,
        "batch_key": batch_key,
        "subset": False,
        "inplace": True,
    }

    if flavor in {"seurat_v3", "seurat_v3_paper"}:
        hvg_kwargs["check_values"] = check_integer_counts

    try:
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)

    except TypeError as error:
        if "check_values" not in str(error):
            raise

        hvg_kwargs.pop("check_values", None)
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)



def RunPcaOnHvgs(
    adata: AnnData,
    n_pcs: int,
    *,
    scale_for_pca: bool = True,
    max_value: float | None = None,
    random_state: int = 42,
) -> str:
    """
    Compute PCA using highly variable genes.

    `adata.X` is expected to contain normalized + log1p expression.

    PCA is computed on a temporary AnnData containing only HVGs, so the
    original `adata.X` is not modified.

    If `scale_for_pca=True`, HVGs are centered and scaled before PCA.
    By default, no clipping is applied after scaling (`max_value=None`).

    Results are written to:
        adata.obsm["X_pca"]
        adata.varm["PCs"]
        adata.uns["pca"]
        adata.var["used_for_pca"]

    Notes on `max_value`
    --------------------
    `max_value` is passed to `sc.pp.scale`.

    After scaling, each gene is centered to mean 0 and scaled to unit variance.
    If `max_value` is not None, scaled values are clipped to the interval:

        [-max_value, max_value]

    For example, `max_value=10` means that values above 10 become 10,
    and values below -10 become -10.

    This can reduce the influence of extreme scaled expression values on PCA,
    but it also modifies the scaled data before PCA. Therefore, the default
    here is `max_value=None`, meaning no clipping is applied.

    Notes on PCA loadings
    ---------------------
    PCA is computed only on HVGs. However, `adata.varm["PCs"]` is stored in
    the full AnnData object with shape:

        n_total_genes x n_pcs

    HVGs receive their real PCA loadings. Non-HVGs receive zero by convention
    because they were not included in the PCA. These zeros should be interpreted
    as "not used for PCA", not as biological evidence of no contribution.

    The genes actually used for PCA are stored in:

        adata.var["used_for_pca"]

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

    if n_pcs <= 0:
        raise ValueError("n_pcs must be a positive integer.")

    if max_value is not None and max_value <= 0:
        raise ValueError("max_value must be positive or None.")

    hvg_mask = (
        adata.var["highly_variable"]
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    n_hvgs = int(hvg_mask.sum())

    if n_hvgs == 0:
        raise ValueError("No highly variable genes found.")


    # Use a temporary AnnData with only HVGs so the original adata.X is not modified.
    adata_hvg = adata[:, hvg_mask].copy()

    pca_input_state = "log1p"

    if scale_for_pca:
        sc.pp.scale(
            adata_hvg,
            zero_center=True,
            max_value=max_value,
        )
        pca_input_state = "scaled_log1p"

    sc.pp.pca(
        adata_hvg,
        n_comps=n_pcs,
        zero_center=True,
        svd_solver="arpack",
        random_state=random_state,
    )

    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"].copy()
    adata.uns["pca"] = adata_hvg.uns["pca"].copy()

    # Store which genes were actually used for PCA.
    # This is important for interpretation: only HVGs contributed to the PCA.
    adata.var["used_for_pca"] = hvg_mask

    # Store PCA loadings in the full AnnData object.
    #
    # `adata_hvg.varm["PCs"]` has shape:
    #     n_hvgs x n_pcs
    #
    # But `adata.varm["PCs"]` must have shape:
    #     n_total_genes x n_pcs
    #
    # Therefore, HVGs receive their real loadings and non-HVGs receive zero.
    # These zeros should be interpreted as "gene was not included in PCA",
    # not as evidence that the gene has no biological contribution.
    pcs = np.zeros(
        shape=(adata.n_vars, adata_hvg.varm["PCs"].shape[1]),
        dtype=adata_hvg.varm["PCs"].dtype,
    )
    pcs[hvg_mask, :] = adata_hvg.varm["PCs"]
    adata.varm["PCs"] = pcs

    adata.uns["pca"].setdefault("params", {})
    adata.uns["pca"]["params"].update(
        {
            "use_highly_variable": True,
            "n_pcs_used": int(n_pcs),
            "n_hvgs_used": int(n_hvgs),
            "pca_input_state": pca_input_state,
            "scale_for_pca": bool(scale_for_pca),
            "scale_max_value": max_value,
            "non_hvg_loadings": "stored_as_zero_not_used_for_pca",
        }
    )

    # return the state of the input data used for PCA, which is either "log1p" or "scaled_log1p"
    # the main result of this function is the PCA results stored in adata.obsm["X_pca"], adata.varm["PCs"], and adata.uns["pca"]
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
        Number of PCs to compute.

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

    np.random.seed(random_state)
    random.seed(random_state)

    log1p_layer = NormalizeLog1pFromLayer(
        adata,
        input_layer=input_layer,
        target_sum=target_sum,
    )

    # "seurat_v3"/"seurat_v3_paper" need raw counts; "seurat"/"cell_ranger" need
    # the normalized + log1p layer just created above.
    hvg_layer = input_layer if flavor in ("seurat_v3", "seurat_v3_paper") else log1p_layer

    RunHighlyVariableGenes(
        adata,
        layer=hvg_layer,
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