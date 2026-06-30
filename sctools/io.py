"""
sctools.io
==========
Data loading for single-cell RNA-seq datasets.

Supported formats
-----------------
- ``"10x_h5"``     : Cell Ranger H5 (filtered or raw)
- ``"cellbender"`` : CellBender corrected H5
- ``"mtx"``        : Generic MTX folder (10x, BD Rhapsody, Illumina — any layout with features.tsv)

Main functions
--------------
:func:`load_sample`   -- load a single sample
:func:`load_samples`  -- load multiple samples and concatenate

Quick example
-------------
>>> from sctools.io import load_sample, load_samples

>>> # Single sample — H5
>>> adata = load_sample(
...     "/data/s1/filtered_feature_bc_matrix.h5",
...     format="10x_h5",
...     sample_name="ctrl_1",
... )

>>> # Single sample — MTX folder (Cell Ranger or BD Rhapsody)
>>> adata = load_sample(
...     "/data/s1/outs/filtered_feature_bc_matrix/",
...     format="mtx",
...     sample_name="ctrl_1",
... )

>>> # Multiple samples, same format
>>> adata = load_samples(
...     {"ctrl_1": "/data/ctrl_1/filtered.h5",
...      "ko_1":   "/data/ko_1/filtered.h5"},
...     format="10x_h5",
... )

>>> # Multiple samples, mixed formats
>>> adata = load_samples(
...     {"cb_1":  "/data/cb_1/cellbender_out.h5",
...      "raw_2": "/data/raw_2/filtered_feature_bc_matrix/"},
...     format={"cb_1": "cellbender", "raw_2": "mtx"},
... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import scanpy as sc
from anndata import AnnData
import anndata as ad


from .qc import CalculateQcMetrics

logger = logging.getLogger(__name__)

FormatType = Literal["h5", "mtx"]

# ── Public API ────────────────────────────────────────────────────────────────

def load_sample(
    path: str | Path,
    *,
    format: FormatType = "h5",
    sample_name: str = "",
    sample_col: str = "sample",
    obs_meta: dict[str, str] | None = None,
    include_qc_metrics: bool = False,
) -> AnnData:
    
    """Load a single scRNA-seq sample into AnnData.

    Parameters
    ----------
    path :
        Path to the H5 file or MTX folder, depending on ``format``.
    format :
        Input format:

        * ``"h5"``     -- Cell Ranger H5 (filtered or raw)
        * ``"mtx"``        -- Generic MTX folder with ``features.tsv``.
          Works for Cell Ranger, BD Rhapsody, and Dragen pipelines.
          Gene symbols are used as var_names.

    sample_name :
        Sample identifier stored in ``adata.obs[sample_col]`` and prepended
        to barcodes to ensure uniqueness when concatenating.
    sample_col :
        Column name in ``adata.obs`` for the sample identifier. Default ``"sample"``.
    obs_meta :
        Extra columns to add to ``adata.obs``, e.g.
        ``{"condition": "KO", "tissue": "colon", "id": "M1"}``.
        Each value is broadcast to all cells and stored as Categorical.

    -------
    AnnData
        Loaded AnnData with:

        * ``adata.X``                -- raw count matrix (CSR)
        * ``adata.obs[sample_col]``  -- sample name as Categorical (if provided)
        * ``adata.obs_names``        -- barcodes prefixed as ``"sample_name_BARCODE"``
        * ``adata.var_names``        -- gene symbols
        * ``adata.var["gene_ids"]``  -- Ensembl IDs when available in the source file

        * Annotations added by :func:`scanpy.pp.calculate_qc_metrics`:

    """
    path = Path(path)
    _require_exists(path)

    loaders = {
        "h5":  lambda: sc.read_10x_h5(path),
        "mtx": lambda: sc.read_10x_mtx(path, gex_only=False),
    }
    if format not in loaders:
        raise ValueError(
            f"Unknown format: {format!r}. "
            f"Choose one of: {list(loaders)}"
        )

    adata = loaders[format]()
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)

    if sample_name:
        # Repeat sample_name once per cell (n_obs times) and store as Categorical.
        # Categorical is more memory-efficient than object dtype and speeds up
        # groupby operations downstream (e.g. per-sample QC, plotting).
        adata.obs[sample_col] = pd.Categorical([sample_name] * adata.n_obs)

        # Prefix every barcode with the sample name: "BARCODE" → "sample_name_BARCODE".
        # Barcodes from different samples are identical (they all come from the same
        # sequencing chemistry pool), so without this prefix concatenating two samples
        # would produce duplicate obs_names and raise an error or silently corrupt data.
        adata.obs_names = [f"{sample_name}_{bc}" for bc in adata.obs_names]

    if obs_meta:
        # obs_meta is a dict of {column_name: value}. Each value is broadcast to all cells.
        for col, val in obs_meta.items():
            adata.obs[col] = pd.Categorical([val] * adata.n_obs)

    if include_qc_metrics:
        CalculateQcMetrics(adata)

    logger.info(
        "load_sample '%s': %d cells x %d genes [%s].",
        sample_name or path.name, adata.n_obs, adata.n_vars, format,
    )
    return adata


def load_samples(
    samples: dict[str, str | Path],
    *,
    format: FormatType | dict[str, FormatType] = "h5",
    sample_col: str = "sample",
    genome: str | None = None,
    join: Literal["inner", "outer"] = "inner",
) -> AnnData:
    """Load multiple samples and return a single concatenated AnnData.

    Parameters
    ----------
    samples :
        Dict of ``{sample_name: path}``. Dict insertion order determines
        sample order in the concatenated AnnData.
    format :
        Input format. Either:

        * A single string -- same format applied to all samples.
        * A dict ``{sample_name: format}`` -- per-sample formats, for mixed
          inputs (e.g. some CellBender H5, others MTX).

    sample_col :
        Column name in ``adata.obs`` for the sample identifier.
    genome :
        For H5 files with multiple genomes. ``None`` uses the first.
    join :
        How to handle genes absent in some samples:

        * ``"inner"`` (default) -- keep only genes present in **all** samples.
          Recommended when all samples come from the same experiment/species.
        * ``"outer"`` -- keep all genes, filling absent ones with zeros.

    Returns
    -------
    AnnData
        Concatenated AnnData. ``adata.obs[sample_col]`` is guaranteed to be
        present. All barcodes are unique (prefixed per sample).

    Examples
    --------
    Same format for all samples::

        adata = load_samples(
            {"ctrl": "/data/ctrl/filtered.h5",
             "ko":   "/data/ko/filtered.h5"},
            format="h5",
        )

    Mixed formats::

        adata = load_samples(
            {"cb":  "/data/cb/cellbender_out.h5",
             "raw": "/data/raw/filtered_feature_bc_matrix/"},
            format={"cb": "h5", "raw": "mtx"},
        )
    """
    if not samples:
        raise ValueError("`samples` dict is empty.")

    adatas: list[AnnData] = []
    for name, path in samples.items():
        # format the format argument: if it's a dict, get the format for this sample; otherwise, use the same format for all samples.
        fmt = format[name] if isinstance(format, dict) else format

        ad = load_sample(
            path,
            format=fmt,
            sample_name=name,
            sample_col=sample_col,
            genome=genome,
        )

        adatas.append(ad)

    combined = sc.concat(adatas, join=join, merge="same")
    combined.obs_names_make_unique()

    logger.info(
        "load_samples: %d samples -> %d cells x %d genes [join='%s'].",
        len(adatas), combined.n_obs, combined.n_vars, join,
    )
    return combined


def CatAdata(
    adatas: dict[str, AnnData] | list[AnnData],
    keys: list[str] | None = None,
    *,
    join: Literal["inner", "outer"] = "outer",
) -> AnnData:
    """Concatenate multiple AnnData objects into one.

    Parameters
    ----------
    adatas :
        Either a ``{sample_name: adata}`` dict (keys used automatically) or a
        plain list paired with ``keys``.
    keys :
        Sample names — required when ``adatas`` is a list, ignored when it is
        a dict.
    join :
        ``"outer"`` keeps all genes (fills missing with 0); ``"inner"`` keeps
        only genes shared by every sample.
    """
    if isinstance(adatas, dict):
        keys = list(adatas.keys())
        adatas_list = list(adatas.values())
    else:
        if keys is None:
            raise ValueError("Pass `keys` when `adatas` is a list.")
        adatas_list = adatas

    combined = sc.concat(
        adatas_list,
        join=join,
        label="sample",
        keys=keys,
        merge="same",
        uns_merge="same",
        index_unique="-",
    )
    combined.obs_names_make_unique()
    logger.info(
        "CatAdata: %d samples -> %d cells x %d genes.",
        len(adatas_list), combined.n_obs, combined.n_vars,
    )
    return combined


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _require_exists(path: Path) -> None:
    """Raise FileNotFoundError if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            "Check the path and the format specified."
        )
