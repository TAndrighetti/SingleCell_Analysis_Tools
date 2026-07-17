"""
sctools.alra
============

Python wrapper around the official R ALRA implementation, called through
rpy2 (requires R).

This module does NOT reimplement the ALRA algorithm in Python. It calls the
official functions from the R `ALRA` package directly:

    ALRA::choose_k()
    ALRA::alra()

Main workflow:
    RunAlraOnAnnData()

Expected input:
    AnnData with a layer that is already normalized/log-transformed.

# Run official ALRA through rpy2.
# ALRA expects a normalized/log-transformed matrix with cells as rows and genes as columns.
# The result is stored as a new AnnData layer and should mainly be used for
# marker visualization, dropout-aware exploration, and optional exploratory embeddings.

adata = RunAlraOnAnnData(
    adata,                              # AnnData object
    input_layer="QC_filtered_log1p",    # normalized/log1p layer used as ALRA input
    output_layer="alra",               # output layer that will store the ALRA-completed matrix
    rank=None,                         # use ALRA::choose_k() to estimate k automatically
    use_mkl=False,                     # use standard ALRA; set True only if ALRA MKL support is installed
    store_as_sparse=True,              # save output as sparse matrix to keep the AnnData object lighter
    random_state=42,                   # seeds R's set.seed() -- choose_k() and alra() (rsvd) are otherwise stochastic
)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from sctools.preprocessing import CheckNormalizedLayer

logger = logging.getLogger(__name__)


def _RunAlra(data, genes, cells, rank: Optional[int] = None, use_mkl: bool = False, random_state: int = 42):
    """
    Run official ALRA from R using rpy2.

    Parameters
    ----------
    data
        Normalized/log-transformed expression matrix.
        Expected orientation: cells x genes.
    genes
        Gene names matching the columns of data.
    cells
        Cell names matching the rows of data.
    rank
        Rank k for ALRA. If None, ALRA::choose_k() is used.
    use_mkl
        Whether to request ALRA's MKL implementation, if available.
    random_state
        Seed applied via R's `set.seed()` before `choose_k()`/`alra()` run.

    Returns
    -------
    messages
        Captured R messages and output.
    out
        ALRA-completed matrix returned by R.
    rank_used
        Rank k used by ALRA.
    """
    from rpy2.robjects import r, globalenv
    import rpy2.robjects as ro
    import anndata2ri

    anndata2ri.activate()

    # Pass variables to the R environment.
    # ALRA expects cells as rows and genes as columns.
    globalenv["data"] = data
    globalenv["genes"] = ro.StrVector(list(map(str, genes)))
    globalenv["cells"] = ro.StrVector(list(map(str, cells)))
    globalenv["estimate_rank"] = ro.BoolVector([rank is None])
    globalenv["alra_rank"] = ro.IntVector([0 if rank is None else int(rank)])
    globalenv["use_mkl"] = ro.BoolVector([bool(use_mkl)])
    globalenv["seed_use"] = ro.IntVector([int(random_state)])

    r_code = """
    library(ALRA)
    library(Matrix)

    # FIX: choose_k() picks a rank via a randomized permutation-style test, and
    # alra() itself uses randomized SVD (rsvd) once the matrix is wider than a
    # couple hundred genes (true here -- this runs on the full log1p layer,
    # not the HVG subset). Neither was seeded before, so rank_used and the
    # imputed matrix changed on every run even with the rest of the pipeline
    # (PCA/UMAP/Leiden/Seurat) fixed at random_state=42.
    set.seed(seed_use)

    # Add names for traceability.
    rownames(data) <- cells
    colnames(data) <- genes

    # ALRA works on a normalized/log expression matrix.
    # If the Python object arrived as an R sparse Matrix, convert to a base matrix.
    # This is safer for the official ALRA implementation.
    if (inherits(data, "sparseMatrix")) {
        data <- as.matrix(data)
    }

    alra_messages <- character()

    output_stdout <- capture.output({

        withCallingHandlers({

            # Estimate k with the official ALRA function, unless k was provided.
            if (isTRUE(estimate_rank)) {
                k_choice <- choose_k(data)
                rank_used <- as.integer(k_choice$k)
            } else {
                k_choice <- NULL
                rank_used <- as.integer(alra_rank)
            }

            # Some ALRA versions expose use.mkl; some may not.
            if ("use.mkl" %in% names(formals(alra))) {
                alra_result <- alra(
                    data,
                    k = rank_used,
                    use.mkl = use_mkl
                )
            } else {
                if (isTRUE(use_mkl)) {
                    warning(
                        "This ALRA version does not expose use.mkl. ",
                        "Running without use.mkl."
                    )
                }

                alra_result <- alra(
                    data,
                    k = rank_used
                )
            }

            out <- alra_result[[3]]

        }, message = function(m) {
            alra_messages <<- c(alra_messages, conditionMessage(m))
            invokeRestart("muffleMessage")
        }, warning = function(w) {
            alra_messages <<- c(alra_messages, conditionMessage(w))
            invokeRestart("muffleWarning")
        })

    }, type = "output")

    result <- list(
        out = out,
        rank_used = rank_used,
        messages = c(output_stdout, alra_messages)
    )
    """

    r(r_code)

    result = globalenv["result"]


    def _GetRListElement(result, name):
        """
        Extract an element from an R list returned through rpy2/anndata2ri.

        Depending on the active converters, the R list may arrive as:
            - an rpy2 ListVector, which supports .rx2()
            - a Python OrderedDict/OrdDict, which supports dictionary-style access
        """
        if hasattr(result, "rx2"):
            return result.rx2(name)

        return result[name]


    messages_obj = _GetRListElement(result, "messages")
    out = _GetRListElement(result, "out")
    rank_used_obj = _GetRListElement(result, "rank_used")

    messages = "\n".join([str(x) for x in list(messages_obj)])
    rank_used = int(list(rank_used_obj)[0])

    return messages, out, rank_used


def RunAlraOnAnnData(
    adata: AnnData,
    input_layer: str = "log_norm",
    output_layer: str = "alra",
    rank: Optional[int] = None,
    use_mkl: bool = False,
    store_as_sparse: bool = True,
    random_state: int = 42,
) -> AnnData:
    """
    Run official ALRA on an AnnData layer and store the output in adata.layers.

    Parameters
    ----------
    adata
        AnnData object.
    input_layer
        Layer containing normalized/log-transformed expression values.
        The matrix must be cells x genes.
    output_layer
        Name of the layer that will receive the ALRA-completed matrix.
    rank
        Rank k for ALRA.
        If None, ALRA::choose_k() is used.
    use_mkl
        Whether to request ALRA's MKL implementation, if available.
    store_as_sparse
        If True, store the ALRA output as scipy csr_matrix.
        If False, store it as a dense numpy array.
    random_state
        Seed for R's `set.seed()`, applied before `choose_k()`/`alra()`.
        Both use randomized algorithms (rank estimation and, for wide
        matrices, randomized SVD) that are otherwise unseeded -- see FIX
        note in `_RunAlra`.

    Returns
    -------
    AnnData
        AnnData object with ALRA output stored in adata.layers[output_layer].

    Notes
    -----
    This function calls the official R implementation of ALRA through rpy2.
    It does not reimplement ALRA in Python.

    Recommended use:
        - marker visualization
        - dropout-aware exploratory analysis
        - optional exploratory PCA/UMAP/clustering

    Not recommended as the primary input for:
        - pseudobulk differential expression
        - DESeq2/edgeR-style differential expression
    """
    if input_layer not in adata.layers:
        raise KeyError(
            f"Layer '{input_layer}' was not found. "
            f"Available layers: {list(adata.layers.keys())}"
        )

    # AnnData stores expression as cells x genes, which is what ALRA expects.
    data = adata.layers[input_layer]
    CheckNormalizedLayer(data)

    genes = adata.var_names
    cells = adata.obs_names

    logger.info(
        "Running official ALRA (R via rpy2) on layer '%s' with shape %d cells x %d genes.",
        input_layer, adata.n_obs, adata.n_vars,
    )

    msg, out, rank_used = _RunAlra(
        data=data,
        genes=genes,
        cells=cells,
        rank=rank,
        use_mkl=use_mkl,
        random_state=random_state,
    )

    if msg:
        logger.info("ALRA (R) output:\n%s", msg)

    # Convert R output back to a Python matrix.
    if issparse(out):
        alra_matrix = out.tocsr().astype("float32")
    else:
        alra_matrix = np.asarray(out, dtype="float32")

    if alra_matrix.shape != adata.shape:
        raise ValueError(
            "ALRA output shape does not match adata shape. "
            f"ALRA output: {alra_matrix.shape}; adata: {adata.shape}. "
            "Check matrix orientation. ALRA should receive cells x genes."
        )

    # Optional: store as sparse to save memory.
    # This is not required, but usually convenient for AnnData.
    if store_as_sparse and not issparse(alra_matrix):
        alra_matrix = csr_matrix(alra_matrix)

    adata.layers[output_layer] = alra_matrix

    adata.uns.setdefault("alra", {})
    adata.uns["alra"][output_layer] = {
        "method": "official R ALRA via rpy2",
        "input_layer": input_layer,
        "output_layer": output_layer,
        "rank_used": rank_used,
        "rank_was_estimated": rank is None,
        "use_mkl": bool(use_mkl),
        "store_as_sparse": bool(store_as_sparse),
        "matrix_orientation": "cells x genes",
        "random_state": int(random_state),
        "messages": msg,
    }

    logger.info(
        "ALRA finished. Output stored in adata.layers['%s'] with rank k=%d.",
        output_layer, rank_used,
    )

    return adata
