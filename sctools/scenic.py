"""
sctools.scenic
===============
Wrapper functions around the pySCENIC CLI (GRNBoost2 + cisTarget + AUCell)
for gene regulatory network / regulon activity analysis.

Typical usage order:
    DownloadScenicResources()      -- one-time, per species
    ValidateScenicResources()      -- one-time, sanity-check the download
    PrepareScenicInput()           -- per analysis, from a raw-count layer
    ExportScenicLoom()
    RunScenicGrn()                 -- GRNBoost2: candidate TF-target edges
    CheckScenicGrn()                -- optional sanity-check of the adjacency table
    RunScenicCtx()                  -- cisTarget: prune to motif-supported regulons
    RunScenicAucell()               -- per-cell regulon activity
    ReadScenicAucell() / ImportScenicAucell()
    SummarizeScenicActivity()       -- per-sample/celltype summary
    PlotScenicActivity()

Repeated-run / consensus-regulon workflow (Garner et al. 2023), for when a
single GRNBoost2+cisTarget run isn't trusted on its own (GRNBoost2 is
stochastic -- different seeds can return different regulons/targets):
    RunRepeatedScenic()             -- re-runs GRN+ctx N times with different seeds
    BuildScenicConsensus()          -- keeps regulons/targets recurring in >80% of runs
    ExportConsensusRegulons()       -- consensus regulons -> GMT, for a final AUCell run

Quick start
-----------
    tf_path, tfs, ranking_db_paths, motif_path = DownloadScenicResources(
        species="mouse", out_dir="~/0.resources/scenic",
    )
    adata_scenic = PrepareScenicInput(adata, counts_layer="counts", tfs=tfs)
    loom_path = ExportScenicLoom(adata_scenic, loom_path="input/scenic_input.loom")
    adjacency_path = RunScenicGrn(loom_path, tf_path, output_dir="output")
    regulons_path = RunScenicCtx(adjacency_path, ranking_db_paths, motif_path, loom_path, "output")
    aucell_path = RunScenicAucell(loom_path, regulons_path, "output")
    adata_scenic, auc_matrix = ImportScenicAucell(adata_scenic, aucell_path)

Environment note
-----------------
pySCENIC 0.12.1 (the last released version) still uses numpy aliases removed
in numpy>=1.24 (`np.object`, `np.float`), so `pyscenic grn`/`ctx`/`aucell`
crash with `AttributeError: module 'numpy' has no attribute 'object'` under
a modern numpy. Run this module's `Run*` functions (they shell out to the
`pyscenic` CLI via `subprocess`) in an environment pinned to numpy<1.24
-- a `scenic_env` conda/Jupyter kernel with numpy==1.23.5 is known to work.
`RunScenicGrn`/`RunScenicCtx`/`RunScenicAucell` all accept a `pyscenic_path`
override, but by default they no longer rely on `$PATH` alone: many Jupyter
kernelspecs launch the interpreter by absolute path (`.../envs/scenic_env/
bin/python -m ipykernel_launcher`) without an intervening `conda activate`,
so the kernel process's `$PATH` doesn't necessarily include that env's
`bin/` -- `shutil.which("pyscenic")` can then fail even though the correct
`pyscenic` is one directory away from `sys.executable`. The default now
checks `$PATH` first, then falls back to the `pyscenic` sitting next to the
running Python interpreter, before giving up.

TestScenicActivity (a KO-vs-WT statistical comparison of
`SummarizeScenicActivity`'s per-sample output, referenced from downstream
notebooks) is not included here -- its source wasn't recovered from the
tutorial notebook it was written in. Port it once its implementation is
found, rather than guessing its statistic/test from the call site alone.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

import loompy
import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

__all__ = [
    "DownloadScenicResources",
    "ValidateScenicResources",
    "PrepareScenicInput",
    "ExportScenicLoom",
    "RunScenicGrn",
    "CheckScenicGrn",
    "RunScenicCtx",
    "RunScenicAucell",
    "ReadScenicAucell",
    "ImportScenicAucell",
    "SummarizeScenicActivity",
    "PlotScenicActivity",
    "RunRepeatedScenic",
    "BuildScenicConsensus",
    "ExportConsensusRegulons",
]


# ── Resources ──────────────────────────────────────────────────────────────

def DownloadScenicResources(
    species: str,
    out_dir: str | Path,
) -> tuple[Path, list[str], list[Path], Path]:
    """
    Download the TF list, cisTarget ranking databases, and motif
    annotations required by pySCENIC (mouse or human, v10 cisTarget
    resources from resources.aertslab.org).

    Files already present in `out_dir` are not re-downloaded.

    Parameters
    ----------
    species : "mouse" or "human".
    out_dir : download destination.

    Returns
    -------
    (tf_path, tfs, ranking_db_paths, motif_path) :
        tf_path            -- path to the TF list file.
        tfs                -- the TF list itself, as gene symbols.
        ranking_db_paths   -- [promoter-proximal, extended] cisTarget
                               ranking database paths (both needed by ctx).
        motif_path         -- motif-to-TF annotation table path.
    """

    species = species.lower()
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://resources.aertslab.org/cistarget"

    # Two ranking databases per species: a promoter-proximal window (500bp
    # upstream/100bp downstream of the TSS) and a broader 10kb window --
    # ctx uses both search spaces together.
    if species == "mouse":
        tf_url = f"{base_url}/tf_lists/allTFs_mm.txt"

        promoter_db_url = (
            f"{base_url}/databases/mus_musculus/mm10/"
            "refseq_r80/mc_v10_clust/gene_based/"
            "mm10_500bp_up_100bp_down_full_tx_v10_clust."
            "genes_vs_motifs.rankings.feather"
        )

        extended_db_url = (
            f"{base_url}/databases/mus_musculus/mm10/"
            "refseq_r80/mc_v10_clust/gene_based/"
            "mm10_10kbp_up_10kbp_down_full_tx_v10_clust."
            "genes_vs_motifs.rankings.feather"
        )

        motif_url = (
            f"{base_url}/motif2tf/"
            "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
        )

    elif species == "human":
        tf_url = f"{base_url}/tf_lists/allTFs_hg38.txt"

        promoter_db_url = (
            f"{base_url}/databases/homo_sapiens/hg38/"
            "refseq_r80/mc_v10_clust/gene_based/"
            "hg38_500bp_up_100bp_down_full_tx_v10_clust."
            "genes_vs_motifs.rankings.feather"
        )

        extended_db_url = (
            f"{base_url}/databases/homo_sapiens/hg38/"
            "refseq_r80/mc_v10_clust/gene_based/"
            "hg38_10kbp_up_10kbp_down_full_tx_v10_clust."
            "genes_vs_motifs.rankings.feather"
        )

        motif_url = (
            f"{base_url}/motif2tf/"
            "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
        )

    else:
        raise ValueError("`species` must be either 'mouse' or 'human'.")

    urls = [tf_url, promoter_db_url, extended_db_url, motif_url]

    downloaded_paths = []

    for url in urls:
        file_path = out_dir / url.split("/")[-1]

        if file_path.exists():
            print(f"File already exists: {file_path.name}")
        else:
            print(f"Downloading: {file_path.name}")
            urlretrieve(url, file_path)

        downloaded_paths.append(file_path)

    tf_path = downloaded_paths[0]
    ranking_db_paths = [downloaded_paths[1], downloaded_paths[2]]
    motif_path = downloaded_paths[3]

    with open(tf_path) as file:
        tfs = [line.strip() for line in file if line.strip()]

    print(f"\nDownloaded resources for: {species}")
    print(f"Number of TFs: {len(tfs)}")
    print(f"Number of ranking databases: {len(ranking_db_paths)}")

    return tf_path, tfs, ranking_db_paths, motif_path


def ValidateScenicResources(
    species: str,
    tf_path: str | Path,
    ranking_db_paths: list[str | Path],
    motif_path: str | Path,
) -> pd.DataFrame:
    """
    Sanity-check the files downloaded by `DownloadScenicResources`.

    Checks file existence, reads the TF list, opens each cisTarget
    ranking database (via ctxcore), and reports the overlap between the
    TF list and the genes actually present in each database -- a low
    overlap usually means a gene-ID/species/annotation-version mismatch
    between the TF list and the ranking databases, and predicts that
    `RunScenicCtx` will find few or no regulons for the affected TFs.

    Returns
    -------
    pd.DataFrame, one row per ranking database, with
    `n_genes`/`n_tfs_in_database`/`tf_overlap_percentage`.
    """
    from ctxcore.rnkdb import FeatherRankingDatabase

    species = species.lower()

    if species not in ["mouse", "human"]:
        raise ValueError("`species` must be either 'mouse' or 'human'.")

    tf_path = Path(tf_path)
    motif_path = Path(motif_path)
    ranking_db_paths = [Path(path) for path in ranking_db_paths]

    if not tf_path.exists():
        raise FileNotFoundError(f"TF file not found: {tf_path}")

    if not motif_path.exists():
        raise FileNotFoundError(f"Motif annotation file not found: {motif_path}")

    with open(tf_path) as file:
        tfs = [line.strip() for line in file if line.strip()]

    if len(tfs) == 0:
        raise ValueError("The transcription factor list is empty.")

    unique_tfs = set(tfs)

    print(f"Number of TFs: {len(tfs)}")
    print(f"Unique TFs: {len(unique_tfs)}")

    if len(tfs) != len(unique_tfs):
        print(f"Warning: {len(tfs) - len(unique_tfs)} duplicated TF names were found.")

    motif_annotations = pd.read_csv(motif_path, sep="\t")

    if motif_annotations.empty:
        raise ValueError("The motif annotation table is empty.")

    print(f"Motif annotations: {motif_annotations.shape[0]} rows")

    validation_results = []

    for db_path in ranking_db_paths:

        if not db_path.exists():
            raise FileNotFoundError(f"Ranking database not found: {db_path}")

        if not db_path.name.endswith(".genes_vs_motifs.rankings.feather"):
            print(f"Warning: unexpected database filename: {db_path.name}")

        database = FeatherRankingDatabase(str(db_path), name=db_path.stem)
        database_genes = set(database.genes)

        tfs_in_database = unique_tfs.intersection(database_genes)
        overlap_percentage = len(tfs_in_database) / len(unique_tfs) * 100

        validation_results.append(
            {
                "database": db_path.name,
                "n_genes": len(database_genes),
                "n_tfs_in_database": len(tfs_in_database),
                "tf_overlap_percentage": overlap_percentage,
            }
        )

    print("\nSCENIC resource validation completed.")

    return pd.DataFrame(validation_results)


# ── Input preparation ────────────────────────────────────────────────────

def PrepareScenicInput(
    adata,
    counts_layer,
    *,
    sample_key=None,
    samples=None,
    tfs=None,
    min_cell_fraction=0.01,
):
    """
    Prepare a count matrix for pySCENIC.

    Selects cells (optionally a subset of samples), copies `counts_layer`
    (must be raw, non-negative, count-scale -- not integrated, scaled,
    imputed, or a PCA/Harmony representation) into `.X`, and drops genes
    with very low expression. Following the SCENIC protocol, a gene is
    kept when it is detected in at least `min_cell_fraction` of the
    selected cells AND its total expression is >= 3 * that minimum cell
    count. Genes are not restricted to highly variable ones, since TFs
    and their targets may not be HVGs. Never mutates `adata`.

    Parameters
    ----------
    adata : AnnData with `counts_layer` in `.layers`.
    counts_layer : str, name of the raw-count layer to use.
    sample_key, samples : restrict to `adata.obs[sample_key].isin(samples)`;
        both or neither.
    tfs : optional TF list, only used to report how many TFs survive the
        gene filter (does not affect filtering).
    min_cell_fraction : float in (0, 1], SCENIC's per-gene detection threshold.

    Returns
    -------
    AnnData, a new object with `.X` set to the filtered counts.
    """

    if counts_layer not in adata.layers:
        raise ValueError(f"Layer '{counts_layer}' was not found in adata.layers.")

    if not 0 < min_cell_fraction <= 1:
        raise ValueError("`min_cell_fraction` must be between 0 and 1.")

    if samples is not None:

        if sample_key is None:
            raise ValueError("`sample_key` must be provided when using `samples`.")

        if sample_key not in adata.obs:
            raise ValueError(f"Column '{sample_key}' was not found in adata.obs.")

        cell_mask = adata.obs[sample_key].isin(samples)
        adata_scenic = adata[cell_mask].copy()

    else:
        adata_scenic = adata.copy()

    if adata_scenic.n_obs == 0:
        raise ValueError("No cells were selected for the SCENIC analysis.")

    if not adata_scenic.var_names.is_unique:
        raise ValueError(
            "Gene names are not unique. Resolve duplicated gene names before running SCENIC."
        )

    adata_scenic.X = adata_scenic.layers[counts_layer].copy()

    if sparse.issparse(adata_scenic.X):
        minimum_value = adata_scenic.X.data.min() if adata_scenic.X.data.size > 0 else 0
    else:
        minimum_value = np.asarray(adata_scenic.X).min()

    if minimum_value < 0:
        raise ValueError(
            "The selected layer contains negative values. Use a non-negative count matrix."
        )

    # SCENIC gene filter: detected in >= min_cell_fraction of cells, AND
    # total expression >= 3x that minimum cell count.
    min_cells = max(1, int(np.ceil(adata_scenic.n_obs * min_cell_fraction)))
    min_total_counts = 3 * min_cells

    if sparse.issparse(adata_scenic.X):
        cells_per_gene = np.asarray((adata_scenic.X > 0).sum(axis=0)).ravel()
        counts_per_gene = np.asarray(adata_scenic.X.sum(axis=0)).ravel()
    else:
        expression_matrix = np.asarray(adata_scenic.X)
        cells_per_gene = (expression_matrix > 0).sum(axis=0)
        counts_per_gene = expression_matrix.sum(axis=0)

    gene_mask = (cells_per_gene >= min_cells) & (counts_per_gene >= min_total_counts)

    n_genes_before = adata_scenic.n_vars
    adata_scenic = adata_scenic[:, gene_mask].copy()

    print(f"Cells selected: {adata_scenic.n_obs}")
    print(f"Minimum cells per gene: {min_cells}")
    print(f"Minimum total counts per gene: {min_total_counts}")
    print(f"Genes retained: {adata_scenic.n_vars} of {n_genes_before}")

    if tfs is not None:
        tfs_found = adata_scenic.var_names.isin(tfs).sum()
        print(f"TFs retained: {tfs_found} of {len(tfs)}")

    return adata_scenic


def ExportScenicLoom(
    adata_scenic,
    loom_path,
    *,
    overwrite=False,
):
    """
    Export a prepared AnnData object to a loom file for the pySCENIC CLI.

    AnnData stores cells x genes; loom stores genes x cells, so the
    matrix is transposed on export. Does not normalize, transform or
    filter the data again -- it only changes file format.

    Parameters
    ----------
    adata_scenic : AnnData, typically `PrepareScenicInput`'s output.
    loom_path : destination path.
    overwrite : if False (default) and `loom_path` already exists, skip
        export and return the existing path.

    Returns
    -------
    Path to the loom file.
    """

    loom_path = Path(loom_path)
    loom_path.parent.mkdir(parents=True, exist_ok=True)

    if loom_path.exists():
        if not overwrite:
            print(f"File already exists: {loom_path}")
            return loom_path
        loom_path.unlink()

    if adata_scenic.n_obs == 0:
        raise ValueError("The AnnData object contains no cells.")

    if adata_scenic.n_vars == 0:
        raise ValueError("The AnnData object contains no genes.")

    if not adata_scenic.var_names.is_unique:
        raise ValueError("Gene names must be unique.")

    if not adata_scenic.obs_names.is_unique:
        raise ValueError("Cell names must be unique.")

    expression_matrix = adata_scenic.X

    if sparse.issparse(expression_matrix):
        n_genes = np.asarray((expression_matrix > 0).sum(axis=1)).ravel()
        n_umi = np.asarray(expression_matrix.sum(axis=1)).ravel()
        loom_matrix = expression_matrix.T.tocsc()
    else:
        expression_matrix = np.asarray(expression_matrix)
        n_genes = (expression_matrix > 0).sum(axis=1)
        n_umi = expression_matrix.sum(axis=1)
        loom_matrix = expression_matrix.T

    row_attributes = {"Gene": np.asarray(adata_scenic.var_names, dtype=str)}

    column_attributes = {
        "CellID": np.asarray(adata_scenic.obs_names, dtype=str),
        "nGene": n_genes,
        "nUMI": n_umi,
    }

    loompy.create(str(loom_path), loom_matrix, row_attributes, column_attributes)

    print(f"Loom file created: {loom_path}")
    print(f"Matrix shape: {adata_scenic.n_vars} genes x {adata_scenic.n_obs} cells")

    return loom_path


def _resolve_pyscenic_path(pyscenic_path):
    """
    Find a working `pyscenic` CLI binary.

    Tries, in order: (1) the caller's explicit override, (2) `$PATH`
    (`shutil.which`), (3) the `pyscenic` sitting next to the running
    Python interpreter (`sys.executable`'s directory) -- pip installs a
    console script there, in the same env, regardless of what `$PATH`
    the kernel process happened to inherit. Falls back to the literal
    string "pyscenic" (matching the old behavior) only if none of the
    above resolve, so the resulting error message is still informative.
    """
    if pyscenic_path is not None:
        return str(pyscenic_path)

    found = shutil.which("pyscenic")
    if found:
        return found

    candidate = Path(sys.executable).parent / "pyscenic"
    if candidate.exists():
        return str(candidate)

    return "pyscenic"


# ── GRNBoost2 (candidate TF-target inference) ───────────────────────────

def RunScenicGrn(
    loom_path,
    tf_path,
    output_dir,
    *,
    pyscenic_path=None,
    num_workers=3,
    seed=42,
    overwrite=False,
):
    """
    Run the GRNBoost2 step of pySCENIC (`pyscenic grn`).

    Infers candidate TF-target associations from expression variation
    across cells -- a gradient-boosting regression per target gene, TFs
    as predictors. Output importance scores are non-negative predictive
    scores, not correlations, and don't imply activation/repression or
    direct/causal regulation. GRNBoost2 is stochastic (see `seed`); see
    `RunRepeatedScenic` for assessing run-to-run stability.

    Parameters
    ----------
    loom_path, tf_path : `ExportScenicLoom`/`DownloadScenicResources` outputs.
    output_dir : `adjacencies.tsv` is written here.
    pyscenic_path : explicit path to the `pyscenic` binary; if None
        (default), resolved via `_resolve_pyscenic_path` (see module
        docstring for why `$PATH` alone isn't trusted, and for the
        numpy<1.24 requirement).
    num_workers : Dask worker count for GRNBoost2 -- CPU-bound, scale
        with physical cores available.
    seed : GRNBoost2 random seed.
    overwrite : if False (default) and `adjacencies.tsv` already exists,
        skip the run and return the existing path as-is -- note this
        does NOT check the file is non-empty/valid, only that it exists;
        a stale/empty file from an interrupted previous run will be
        silently reused unless removed first or `overwrite=True`.

    Returns
    -------
    Path to `adjacencies.tsv`.
    """

    loom_path = Path(loom_path)
    tf_path = Path(tf_path)
    output_dir = Path(output_dir)

    if not loom_path.exists():
        raise FileNotFoundError(f"Loom file not found: {loom_path}")

    if not tf_path.exists():
        raise FileNotFoundError(f"TF list not found: {tf_path}")

    pyscenic_path = _resolve_pyscenic_path(pyscenic_path)

    if shutil.which(pyscenic_path) is None and not Path(pyscenic_path).exists():
        raise RuntimeError(
            f"The '{pyscenic_path}' command was not found in the current environment."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    adjacency_path = output_dir / "adjacencies.tsv"

    if adjacency_path.exists() and not overwrite:
        print(f"File already exists: {adjacency_path}")
        return adjacency_path

    command = [
        str(pyscenic_path),
        "grn",
        str(loom_path),
        str(tf_path),
        "--method",
        "grnboost2",
        "--output",
        str(adjacency_path),
        "--num_workers",
        str(num_workers),
        "--seed",
        str(seed),
    ]

    print("Running GRNBoost2...")
    print(" ".join(command))

    subprocess.run(command, check=True)

    if not adjacency_path.exists():
        raise RuntimeError("GRNBoost2 finished without creating the adjacency file.")

    print(f"\nAdjacency file created: {adjacency_path}")

    return adjacency_path


def CheckScenicGrn(adjacency_path):
    """
    Read and validate a GRNBoost2 adjacency table (`RunScenicGrn`'s output).

    Checks for the expected columns (`TF`, `target`, `importance`),
    missing/non-finite/negative importance values, and duplicated
    TF-target pairs, then prints a summary. Raises on any of the above
    rather than silently dropping bad rows.

    Returns
    -------
    pd.DataFrame, the adjacency table (with `importance` coerced to numeric).
    """

    adjacency_path = Path(adjacency_path)

    if not adjacency_path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adjacency_path}")

    adjacencies = pd.read_csv(adjacency_path, sep="\t")

    required_columns = {"TF", "target", "importance"}

    if not required_columns.issubset(adjacencies.columns):
        raise ValueError("The adjacency table must contain the columns: 'TF', 'target', and 'importance'.")

    adjacencies["importance"] = pd.to_numeric(adjacencies["importance"], errors="coerce")

    n_missing = adjacencies[["TF", "target", "importance"]].isna().any(axis=1).sum()
    n_invalid = (~np.isfinite(adjacencies["importance"])).sum()
    n_negative = (adjacencies["importance"] < 0).sum()
    n_duplicates = adjacencies.duplicated(subset=["TF", "target"]).sum()

    if n_missing > 0:
        raise ValueError(f"{n_missing} rows contain missing values.")

    if n_invalid > 0:
        raise ValueError(f"{n_invalid} importance values are not finite.")

    if n_negative > 0:
        raise ValueError(f"{n_negative} negative importance values were found.")

    print(f"Associations: {len(adjacencies):,}")
    print(f"Transcription factors: {adjacencies['TF'].nunique():,}")
    print(f"Target genes: {adjacencies['target'].nunique():,}")
    print(f"Duplicated TF-target pairs: {n_duplicates:,}")
    print("\nImportance summary:")
    print(adjacencies["importance"].describe().round(4))

    return adjacencies


# ── cisTarget (motif-enrichment pruning) ────────────────────────────────

def RunScenicCtx(
    adjacency_path,
    ranking_db_paths,
    motif_path,
    loom_path,
    output_dir,
    *,
    pyscenic_path=None,
    num_workers=3,
    overwrite=False,
):
    """
    Run the cisTarget step of pySCENIC (`pyscenic ctx`).

    Tests whether GRNBoost2's candidate TF-target modules are supported
    by cis-regulatory motif enrichment, and prunes each module down to
    the genes responsible for that enrichment signal (its "leading
    edge"). This changes the interpretation from a purely expression-
    based association to one with motif support -- it still does not
    prove the TF physically binds in the analyzed cells. Warnings like
    "Less than 80% of the genes in ... could be mapped" mean a module
    didn't overlap enough with a given ranking database and was skipped
    for that database; that's expected some of the time. The output is
    a collection of regulons (one TF + its motif-supported targets per
    row group), not yet scored per cell -- that's `RunScenicAucell`.

    Parameters
    ----------
    adjacency_path : `RunScenicGrn` output.
    ranking_db_paths, motif_path : `DownloadScenicResources` outputs.
    loom_path : same loom used for `RunScenicGrn` (ctx recomputes
        TF-target correlations from expression to build modules before
        motif testing).
    output_dir : `regulons.csv` and `ctx.log` are written here.
    pyscenic_path : see `RunScenicGrn`.
    overwrite : see `RunScenicGrn` -- same stale-file caveat applies.

    Returns
    -------
    Path to `regulons.csv`.
    """

    adjacency_path = Path(adjacency_path)
    ranking_db_paths = [Path(path) for path in ranking_db_paths]
    motif_path = Path(motif_path)
    loom_path = Path(loom_path)
    output_dir = Path(output_dir)

    input_paths = [adjacency_path, motif_path, loom_path, *ranking_db_paths]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    pyscenic_path = _resolve_pyscenic_path(pyscenic_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    regulons_path = output_dir / "regulons.csv"
    log_path = output_dir / "ctx.log"

    if regulons_path.exists() and not overwrite:
        print(f"File already exists: {regulons_path}")
        return regulons_path

    command = [
        str(pyscenic_path),
        "ctx",
        str(adjacency_path),
        *[str(path) for path in ranking_db_paths],
        "--annotations_fname",
        str(motif_path),
        "--expression_mtx_fname",
        str(loom_path),
        "--mode",
        "custom_multiprocessing",
        "--output",
        str(regulons_path),
        "--num_workers",
        str(num_workers),
    ]

    print("Running cisTarget...")
    print(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    # ctx's own stdout/stderr (incl. the per-module coverage warnings
    # mentioned above) is easy to lose in a long notebook run, so it's
    # always saved to disk regardless of success/failure.
    log_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"cisTarget failed. Check the log file:\n{log_path}")

    if not regulons_path.exists():
        raise RuntimeError("cisTarget finished without creating the regulon file.")

    if regulons_path.stat().st_size == 0:
        raise RuntimeError("cisTarget created an empty regulon file.")

    print(f"\nRegulon file created: {regulons_path}")
    print(f"Log file created: {log_path}")

    return regulons_path


# ── AUCell (per-cell regulon activity) ──────────────────────────────────

def RunScenicAucell(
    loom_path,
    regulons_path,
    output_dir,
    *,
    pyscenic_path=None,
    num_workers=3,
    auc_threshold=0.05,
    seed=42,
    overwrite=False,
):
    """
    Run the AUCell step of pySCENIC (`pyscenic aucell`).

    For each cell, ranks all genes from highest to lowest expression and
    scores each regulon by whether its target genes are concentrated
    near the top of that ranking. `auc_threshold` is NOT a cutoff for
    declaring a regulon "active" -- it sets how much of the ranking
    AUCell looks at: with the default 0.05, it evaluates gene recovery
    within the top 5% of each cell's expression ranking. The resulting
    AUC is a relative enrichment score (not a p-value, probability, or
    ROC AUC, and not a direct TF-activity measurement). `seed` improves
    reproducibility of the ranking when genes are tied.

    Parameters
    ----------
    loom_path : same loom used upstream.
    regulons_path : `RunScenicCtx` output.
    output_dir : `aucell_output.loom` and `aucell.log` are written here.
    pyscenic_path : see `RunScenicGrn`.
    overwrite : see `RunScenicGrn` -- same stale-file caveat applies.

    Returns
    -------
    Path to `aucell_output.loom` (cells x regulons activity matrix,
    read back with `ReadScenicAucell`/`ImportScenicAucell`).
    """

    loom_path = Path(loom_path)
    regulons_path = Path(regulons_path)
    output_dir = Path(output_dir)

    if not loom_path.exists():
        raise FileNotFoundError(f"Loom file not found: {loom_path}")

    if not regulons_path.exists():
        raise FileNotFoundError(f"Regulon file not found: {regulons_path}")

    pyscenic_path = _resolve_pyscenic_path(pyscenic_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    aucell_path = output_dir / "aucell_output.loom"
    log_path = output_dir / "aucell.log"

    if aucell_path.exists() and not overwrite:
        print(f"File already exists: {aucell_path}")
        return aucell_path

    command = [
        str(pyscenic_path),
        "aucell",
        str(loom_path),
        str(regulons_path),
        "--output",
        str(aucell_path),
        "--num_workers",
        str(num_workers),
        "--auc_threshold",
        str(auc_threshold),
        "--seed",
        str(seed),
    ]

    print("Running AUCell...")
    print(" ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    log_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + result.stdout + "\n\nSTDERR\n" + result.stderr
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"AUCell failed. Check the log file:\n{log_path}")

    if not aucell_path.exists():
        raise RuntimeError("AUCell finished without creating the output file.")

    print(f"\nAUCell output created: {aucell_path}")
    print(f"Log file created: {log_path}")

    return aucell_path


def ReadScenicAucell(aucell_path):
    """
    Read the cells x regulons AUCell activity matrix from a pySCENIC
    AUCell output loom file, without touching any AnnData object.

    AUCell stores the matrix as the `RegulonsAUC` column attribute, a
    NumPy structured array (one named field per regulon) rather than a
    plain 2D array -- `auc_data.dtype.names` gives the regulon names.

    Returns
    -------
    pd.DataFrame indexed by `CellID`, one column per regulon.
    """

    aucell_path = Path(aucell_path)

    if not aucell_path.exists():
        raise FileNotFoundError(f"AUCell file not found: {aucell_path}")

    with loompy.connect(str(aucell_path), mode="r", validate=False) as loom:

        if "CellID" not in loom.ca.keys():
            raise ValueError("The loom file does not contain 'CellID'.")

        if "RegulonsAUC" not in loom.ca.keys():
            raise ValueError("The loom file does not contain 'RegulonsAUC'.")

        cell_ids = loom.ca["CellID"].astype(str)
        auc_data = loom.ca["RegulonsAUC"]

    if auc_data.dtype.names is None:
        raise ValueError("Regulon names could not be read from 'RegulonsAUC'.")

    auc_matrix = pd.DataFrame(
        {regulon: auc_data[regulon].astype(float) for regulon in auc_data.dtype.names},
        index=cell_ids,
    )
    auc_matrix.index.name = "CellID"

    print(f"Cells: {auc_matrix.shape[0]}")
    print(f"Regulons: {auc_matrix.shape[1]}")

    return auc_matrix


def ImportScenicAucell(
    adata,
    aucell_path,
    *,
    key="X_scenic_auc",
):
    """
    Import an AUCell regulon activity matrix into `adata.obsm[key]`,
    aligned to `adata.obs_names` (order and presence checked -- raises
    if any AnnData cell is missing from the AUCell output). Also stores
    the regulon names in `adata.uns["scenic_regulons"]`, since `.obsm`
    values don't reliably keep column names through all AnnData paths.

    Returns
    -------
    (adata, auc_matrix) : `adata` (mutated in place, also returned for
    chaining) and the cells x regulons `pd.DataFrame` that was stored.
    """

    aucell_path = Path(aucell_path)

    if not aucell_path.exists():
        raise FileNotFoundError(f"AUCell file not found: {aucell_path}")

    with loompy.connect(str(aucell_path), mode="r", validate=False) as loom:

        if "RegulonsAUC" not in loom.ca.keys():
            raise ValueError("The loom file does not contain 'RegulonsAUC'.")

        if "CellID" not in loom.ca.keys():
            raise ValueError("The loom file does not contain 'CellID'.")

        cell_ids = pd.Index(loom.ca["CellID"].astype(str))
        auc_matrix = pd.DataFrame(loom.ca["RegulonsAUC"], index=cell_ids)

    if not auc_matrix.index.is_unique:
        raise ValueError("Duplicated cell identifiers were found in the AUCell output.")

    adata_cells = pd.Index(adata.obs_names.astype(str))

    missing_cells = adata_cells.difference(auc_matrix.index)

    if len(missing_cells) > 0:
        raise ValueError(f"{len(missing_cells)} AnnData cells were not found in the AUCell output.")

    auc_matrix = auc_matrix.loc[adata_cells].copy()

    adata.obsm[key] = auc_matrix
    adata.uns["scenic_regulons"] = np.asarray(auc_matrix.columns, dtype=str)

    print(f"Cells imported: {auc_matrix.shape[0]}")
    print(f"Regulons imported: {auc_matrix.shape[1]}")
    print(f"Stored in adata.obsm['{key}']")

    return adata, auc_matrix


# ── Sample-level summary and plotting ───────────────────────────────────

def SummarizeScenicActivity(
    adata,
    *,
    auc_key="X_scenic_auc",
    sample_key="Sample",
    group_keys=None,
    statistic="mean",
):
    """
    Summarize cell-level AUCell scores by biological sample (and
    optionally condition/celltype/etc via `group_keys`), because cells
    from the same sample are not independent replicates.

    For each (sample, *group_keys) group and each regulon, this is a
    plain unweighted mean or median of that group's per-cell AUCell
    scores -- NOT pseudobulk (no raw counts are aggregated here; the
    per-cell AUCell score is computed first, and only that already-
    computed score is averaged). `n_cells` is reported per group so
    small, unstable groups can be spotted, but it isn't used to weight
    anything.

    Parameters
    ----------
    adata : must have `auc_key` in `.obsm` (see `ImportScenicAucell`).
    sample_key : `adata.obs` column identifying the biological sample.
    group_keys : additional `adata.obs` columns to group by (e.g.
        condition, celltype) -- summarize separately per combination to
        avoid mixing activity differences with cell-composition shifts.
    statistic : "mean" or "median".

    Returns
    -------
    Long-format pd.DataFrame: one row per (sample, *group_keys, regulon),
    with `n_cells` and `{statistic}_auc` columns.
    """

    if auc_key not in adata.obsm:
        raise ValueError(f"'{auc_key}' was not found in adata.obsm.")

    if sample_key not in adata.obs.columns:
        raise ValueError(f"'{sample_key}' was not found in adata.obs.")

    if group_keys is None:
        group_keys = []
    group_keys = list(group_keys)

    for key in group_keys:
        if key not in adata.obs.columns:
            raise ValueError(f"'{key}' was not found in adata.obs.")

    if statistic not in ["mean", "median"]:
        raise ValueError("`statistic` must be either 'mean' or 'median'.")

    auc_data = adata.obsm[auc_key]

    if isinstance(auc_data, pd.DataFrame):
        auc_matrix = auc_data.copy()
        auc_matrix.index = adata.obs_names
    else:
        if "scenic_regulons" not in adata.uns:
            raise ValueError("Regulon names were not found in adata.uns['scenic_regulons'].")

        regulon_names = adata.uns["scenic_regulons"]
        auc_matrix = pd.DataFrame(np.asarray(auc_data), index=adata.obs_names, columns=regulon_names)

    metadata_columns = [sample_key] + group_keys
    metadata = adata.obs[metadata_columns].copy()

    activity_table = metadata.join(auc_matrix)

    cell_counts = (
        activity_table.groupby(metadata_columns, observed=True).size().rename("n_cells")
    )

    regulon_columns = list(auc_matrix.columns)

    grouped_activity = activity_table.groupby(metadata_columns, observed=True)[regulon_columns]

    summary = grouped_activity.mean() if statistic == "mean" else grouped_activity.median()
    summary = summary.join(cell_counts).reset_index()

    summary_long = summary.melt(
        id_vars=metadata_columns + ["n_cells"],
        value_vars=regulon_columns,
        var_name="regulon",
        value_name=f"{statistic}_auc",
    )

    print(f"Biological samples: {summary_long[sample_key].nunique()}")
    print(f"Regulons summarized: {summary_long['regulon'].nunique()}")
    print(f"Groups summarized: {summary[metadata_columns].drop_duplicates().shape[0]}")

    return summary_long


def PlotScenicActivity(
    scenic_summary,
    regulon,
    condition_key,
    *,
    sample_key="Sample",
    value_key="mean_auc",
    group_filters=None,
    order=None,
    figsize=(6, 4),
    jitter=0.08,
    random_state=42,
):
    """
    Plot sample-level AUCell activity for one regulon: one jittered
    point per biological sample, plus mean +/- SD per condition.
    Descriptive only -- no statistical test is performed (see the
    TestScenicActivity note in the module docstring for that).

    Parameters
    ----------
    scenic_summary : `SummarizeScenicActivity` output.
    regulon : value to select from the "regulon" column.
    condition_key : `scenic_summary` column defining the x-axis groups.
    group_filters : {column: value} to restrict to one cell population
        (e.g. {"Celltype": "G4"}) -- required if `scenic_summary` has
        more than one value per (sample, condition) for this regulon,
        since this function expects exactly one point per sample.
    order : x-axis condition order; defaults to first-seen order.

    Returns
    -------
    (fig, ax, plot_data) : the figure/axes and the exact data plotted.
    """
    import matplotlib.pyplot as plt

    required_columns = {sample_key, condition_key, "regulon", value_key}
    missing_columns = required_columns.difference(scenic_summary.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    plot_data = scenic_summary.loc[scenic_summary["regulon"] == regulon].copy()

    if group_filters is not None:
        for column, value in group_filters.items():
            if column not in plot_data.columns:
                raise ValueError(f"Column '{column}' was not found.")
            plot_data = plot_data.loc[plot_data[column] == value].copy()

    if plot_data.empty:
        raise ValueError("No data were found for the selected regulon and filters.")

    duplicated_samples = plot_data.duplicated(subset=[sample_key, condition_key], keep=False)

    if duplicated_samples.any():
        raise ValueError(
            "More than one value was found per sample and condition. "
            "Use `group_filters` to select one cell population."
        )

    if order is None:
        order = list(pd.unique(plot_data[condition_key]))

    unknown_conditions = set(plot_data[condition_key]).difference(order)

    if unknown_conditions:
        raise ValueError(f"Conditions missing from `order`: {sorted(unknown_conditions)}")

    rng = np.random.default_rng(random_state)

    fig, ax = plt.subplots(figsize=figsize)

    for position, condition in enumerate(order):

        condition_data = plot_data.loc[plot_data[condition_key] == condition].copy()
        values = condition_data[value_key].to_numpy(dtype=float)

        x_positions = rng.normal(loc=position, scale=jitter, size=len(values))
        ax.scatter(x_positions, values, s=55, zorder=3)

        mean_value = values.mean()
        sd_value = values.std(ddof=1) if len(values) > 1 else 0

        ax.errorbar(
            position, mean_value, yerr=sd_value,
            fmt="_", markersize=20, capsize=5, linewidth=2, zorder=4,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel(condition_key)
    ax.set_ylabel("Mean AUCell score")
    ax.set_title(f"{regulon} activity")

    plt.tight_layout()

    return fig, ax, plot_data


# ── Repeated runs / consensus regulons (Garner et al., 2023) ────────────

def RunRepeatedScenic(
    loom_path,
    tf_path,
    ranking_db_paths,
    motif_path,
    output_dir,
    *,
    n_runs=10,
    num_workers=3,
    base_seed=42,
):
    """
    Run GRNBoost2 + cisTarget `n_runs` times with different seeds
    (`base_seed`, `base_seed+1`, ...), each in its own `runs/run_NNN/`
    subfolder, to later assess which regulons/targets are stable across
    runs (`BuildScenicConsensus`). AUCell is deliberately not run here;
    it's recalculated once, after the consensus regulons are built.

    Returns
    -------
    pd.DataFrame with one row per run (`run`, `seed`, `adjacency_path`,
    `regulons_path`), also saved to `runs/run_summary.csv`.
    """

    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_summary = []

    for run_number in range(1, n_runs + 1):

        seed = base_seed + run_number - 1
        run_dir = runs_dir / f"run_{run_number:03d}"

        print("\n" + "=" * 60)
        print(f"Run {run_number}/{n_runs} | Seed: {seed}")
        print("=" * 60)

        adjacency_path = RunScenicGrn(
            loom_path=loom_path,
            tf_path=tf_path,
            output_dir=run_dir,
            num_workers=num_workers,
            seed=seed,
        )

        regulons_path = RunScenicCtx(
            adjacency_path=adjacency_path,
            ranking_db_paths=ranking_db_paths,
            motif_path=motif_path,
            loom_path=loom_path,
            output_dir=run_dir,
            num_workers=num_workers,
        )

        run_summary.append(
            {
                "run": run_number,
                "seed": seed,
                "adjacency_path": str(adjacency_path),
                "regulons_path": str(regulons_path),
            }
        )

    run_summary = pd.DataFrame(run_summary)

    summary_path = runs_dir / "run_summary.csv"
    run_summary.to_csv(summary_path, index=False)

    print("\nRepeated SCENIC inference completed.")
    print(f"Runs completed: {len(run_summary)}")
    print(f"Summary: {summary_path}")

    return run_summary


def BuildScenicConsensus(
    run_summary,
    *,
    recurrence_threshold=0.80,
    min_targets=5,
):
    """
    Build consensus regulons from `RunRepeatedScenic`'s repeated runs,
    following Garner et al. (2023): a regulon is "high-confidence" if
    recovered in more than `recurrence_threshold` of runs AND has at
    least `min_targets` high-confidence targets; a target is
    "high-confidence" if it recurs within the same regulon in more than
    `recurrence_threshold` of runs. Target recurrence is computed over
    ALL runs, not just the runs where the regulon itself was recovered.

    Each run's `regulons.csv` is read with pySCENIC's own
    `load_motifs`/`df2regulons` (the same functions `RunScenicCtx`'s
    output is meant to be consumed with) -- `df2regulons` collapses
    possibly-several enriched motifs per (TF, direction) into one
    `Regulon` object per (TF, direction), with `.transcription_factor`
    and `.gene2weight` (target -> importance). The TF itself is dropped
    from its own target list before counting recurrence.

    Parameters
    ----------
    run_summary : `RunRepeatedScenic` output.

    Returns
    -------
    (regulon_frequency, target_frequency, consensus_targets, consensus_regulons):
        regulon_frequency  -- every (regulon, TF): n_runs, frequency.
        target_frequency   -- every (regulon, TF, target): n_runs, frequency.
        consensus_targets  -- target_frequency rows above threshold,
                               restricted to consensus regulons.
        consensus_regulons -- regulon_frequency rows passing both the
                               recurrence and min_targets criteria.
    """
    from pyscenic.prune import df2regulons
    from pyscenic.utils import load_motifs

    all_targets = []

    for _, row in run_summary.iterrows():

        run = row["run"]
        regulons_path = Path(row["regulons_path"])

        motifs = load_motifs(str(regulons_path))
        regulons = df2regulons(motifs)

        for regulon in regulons:

            tf = regulon.transcription_factor

            for target in regulon.gene2weight:

                if target == tf:
                    continue

                all_targets.append(
                    {"run": run, "regulon": regulon.name, "TF": tf, "target": target}
                )

    regulon_targets = pd.DataFrame(all_targets)

    n_runs = run_summary["run"].nunique()

    regulon_frequency = (
        regulon_targets[["run", "regulon", "TF"]]
        .drop_duplicates()
        .groupby(["regulon", "TF"], observed=True)
        .size()
        .reset_index(name="n_runs")
    )
    regulon_frequency["frequency"] = regulon_frequency["n_runs"] / n_runs

    target_frequency = (
        regulon_targets[["run", "regulon", "TF", "target"]]
        .drop_duplicates()
        .groupby(["regulon", "TF", "target"], observed=True)
        .size()
        .reset_index(name="n_runs")
    )
    target_frequency["frequency"] = target_frequency["n_runs"] / n_runs

    consensus_targets = target_frequency.loc[
        target_frequency["frequency"] > recurrence_threshold
    ].copy()

    target_counts = (
        consensus_targets.groupby(["regulon", "TF"], observed=True)
        .size()
        .reset_index(name="n_consensus_targets")
    )

    consensus_regulons = regulon_frequency.merge(
        target_counts, on=["regulon", "TF"], how="left"
    )
    consensus_regulons["n_consensus_targets"] = (
        consensus_regulons["n_consensus_targets"].fillna(0).astype(int)
    )

    consensus_regulons = consensus_regulons.loc[
        (consensus_regulons["frequency"] > recurrence_threshold)
        & (consensus_regulons["n_consensus_targets"] >= min_targets)
    ].copy()

    # Drop targets whose regulon didn't itself make the consensus cut.
    consensus_targets = consensus_targets.merge(
        consensus_regulons[["regulon", "TF"]], on=["regulon", "TF"], how="inner"
    )

    print(f"Runs analyzed: {n_runs}")
    print(f"Consensus regulons: {len(consensus_regulons)}")
    print(f"Consensus TF-target pairs: {len(consensus_targets)}")

    return regulon_frequency, target_frequency, consensus_targets, consensus_regulons


def ExportConsensusRegulons(
    consensus_targets,
    output_dir,
):
    """
    Export consensus regulons to GMT format (`regulon<TAB>description
    <TAB>target1<TAB>target2...`, one line per regulon), the format
    `pyscenic aucell` reads as its regulon input -- write this and run
    `RunScenicAucell(..., regulons_path=this_gmt_path)` once, using only
    the high-confidence targets, instead of using any single run's
    `regulons.csv` (see `BuildScenicConsensus`).

    Parameters
    ----------
    consensus_targets : `BuildScenicConsensus` output.

    Returns
    -------
    Path to `consensus_regulons.gmt`.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gmt_path = output_dir / "consensus_regulons.gmt"

    with open(gmt_path, "w") as file:

        for regulon, data in consensus_targets.groupby("regulon"):

            targets = sorted(data["target"].unique())
            line = "\t".join([regulon, regulon, *targets])
            file.write(line + "\n")

    print(f"Consensus regulons exported: {consensus_targets['regulon'].nunique()}")
    print(f"Saved to: {gmt_path}")

    return gmt_path
