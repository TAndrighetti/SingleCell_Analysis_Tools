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
    "CalculateScenicRSS",
    "PlotScenicRSSHeatmap",
    "CalculateScenicRSSByCondition",
    "PlotScenicRSSByConditionHeatmap",
    "CalculateScenicActivityDifference",
    "SelectTopScenicActivityDifferences",
    "PlotScenicActivityDifferenceHeatmap",
    "PlotScenicActivityByConditionHeatmap",
    "PlotScenicAucellByConditionHeatmap",
    "CalculateScenicStageVariation",
    "SelectTopScenicStageVariation",
    "PlotScenicActivityAcrossStages",
    "BuildReactorInputTables",
    "RunReactorProcessData",
    "RunReactorDifferentialActivityROTS",
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

    import loompy

    loompy.create(str(loom_path), loom_matrix, row_attributes, column_attributes)

    print(f"Loom file created: {loom_path}")
    print(f"Matrix shape: {adata_scenic.n_vars} genes x {adata_scenic.n_obs} cells")

    return loom_path


def _ResolvePyscenicPath(pyscenic_path):
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
        (default), resolved via `_ResolvePyscenicPath` (see module
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

    pyscenic_path = _ResolvePyscenicPath(pyscenic_path)

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

    pyscenic_path = _ResolvePyscenicPath(pyscenic_path)

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

    pyscenic_path = _ResolvePyscenicPath(pyscenic_path)

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

    import loompy

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

    import loompy

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
    sample_key="sample",
    condition_key=None,
    celltype_key=None,
    statistic="mean",
):
    """
    Summarize continuous AUCell regulon activity by biological sample.

    AUCell scores are first calculated independently for each cell by
    pySCENIC. This function summarizes those continuous scores across
    cells belonging to the same biological sample and, optionally,
    experimental condition and cell type.

    Biological samples are kept separately rather than pooling all
    cells from the same condition. This preserves the biological
    replicate structure for downstream visualization and comparison.

    No statistical test, scaling, z-score transformation or AUCell
    binarization is performed here.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the AUCell matrix in `adata.obsm[auc_key]`.

    auc_key : str
        Key containing the cells x regulons AUCell matrix.

    sample_key : str
        Column in `adata.obs` identifying biological samples.

    condition_key : str or None
        Optional column identifying experimental conditions.

    celltype_key : str or None
        Optional column identifying cell types or maturation stages.

    statistic : {"mean", "median"}
        Summary statistic calculated across cells within each group.
        Default is "mean", matching the average AUCell representation
        used in downstream SCENIC analyses such as Garner et al. (2023).

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per
        sample x condition x celltype x regulon combination.

        The table contains:
            sample
            condition (if provided)
            celltype (if provided)
            n_cells
            regulon
            mean_auc or median_auc
    """

    if auc_key not in adata.obsm:
        raise ValueError(
            f"'{auc_key}' was not found in adata.obsm."
        )

    if sample_key not in adata.obs.columns:
        raise ValueError(
            f"'{sample_key}' was not found in adata.obs."
        )

    if statistic not in {"mean", "median"}:
        raise ValueError(
            "`statistic` must be either 'mean' or 'median'."
        )

    group_columns = [sample_key]

    if condition_key is not None:
        if condition_key not in adata.obs.columns:
            raise ValueError(
                f"'{condition_key}' was not found in adata.obs."
            )
        group_columns.append(condition_key)

    if celltype_key is not None:
        if celltype_key not in adata.obs.columns:
            raise ValueError(
                f"'{celltype_key}' was not found in adata.obs."
            )
        group_columns.append(celltype_key)

    auc_data = adata.obsm[auc_key]

    if isinstance(auc_data, pd.DataFrame):
        auc_matrix = auc_data.copy()
        auc_matrix.index = adata.obs_names

    else:
        if "scenic_regulons" not in adata.uns:
            raise ValueError(
                "Regulon names were not found in "
                "adata.uns['scenic_regulons']."
            )

        auc_matrix = pd.DataFrame(
            np.asarray(auc_data),
            index=adata.obs_names,
            columns=adata.uns["scenic_regulons"],
        )

    metadata = adata.obs[group_columns].copy()

    activity_table = metadata.join(auc_matrix)

    grouped = activity_table.groupby(
        group_columns,
        observed=True,
    )

    # Number of cells contributing to each biological sample x group.
    cell_counts = grouped.size().rename("n_cells")

    regulon_columns = list(auc_matrix.columns)

    if statistic == "mean":
        summary = grouped[regulon_columns].mean()
    else:
        summary = grouped[regulon_columns].median()

    summary = (
        summary
        .join(cell_counts)
        .reset_index()
    )

    value_column = f"{statistic}_auc"

    summary_long = summary.melt(
        id_vars=group_columns + ["n_cells"],
        value_vars=regulon_columns,
        var_name="regulon",
        value_name=value_column,
    )

    print(
        f"Biological samples: "
        f"{summary_long[sample_key].nunique()}"
    )

    if celltype_key is not None:
        print(
            f"Cell types: "
            f"{summary_long[celltype_key].nunique()}"
        )

    print(
        f"Regulons: "
        f"{summary_long['regulon'].nunique()}"
    )

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


# ── Regulon Specificity Score (RSS) ──────────────────────────────────────

def CalculateScenicRSS(
    adata,
    auc_table,
    celltype_key,
):
    """
    Calculate Regulon Specificity Scores (RSS) with pySCENIC's own
    `regulon_specificity_scores`.

    RSS compares each regulon's per-cell activity distribution (from an
    AUCell run, typically the consensus one) against the distribution
    representing membership in a given cell population -- higher RSS
    means that regulon's activity is more specific to that population.
    RSS is a specificity score, not a p-value; there is no associated
    statistical test [Suo, 2018].

    Parameters
    ----------
    adata : AnnData with `celltype_key` in `.obs`, cells matching `auc_table.index`.
    auc_table : cells x regulons AUCell matrix (e.g. `ReadScenicAucell`/
        `ImportScenicAucell` output from the consensus AUCell run).
    celltype_key : `adata.obs` column defining cell populations.

    Returns
    -------
    pd.DataFrame, celltype x regulon RSS matrix (index named `celltype_key`).
    """
    from pyscenic.rss import regulon_specificity_scores

    cell_types = adata.obs.loc[auc_table.index, celltype_key]

    rss = regulon_specificity_scores(auc_table, cell_types)
    rss.index.name = celltype_key

    return rss


def PlotScenicRSSHeatmap(
    rss,
    *,
    celltype_key="celltype",
    top_n=5,
    cmap="viridis",
    figsize=(6, 8),
    title=None,
):
    """
    Heatmap of the top-N regulons per celltype by RSS (`CalculateScenicRSS`
    output), built on `sctools.plots.PlotSignificanceHeatmap` -- reusing
    its color+annotation machinery, but with a rank-based flag standing
    in for the p-value it normally thresholds.

    Columns/rows shown are the union of each celltype's own top-N
    regulons, so a regulon pulled in because it's top-N for one celltype
    also gets a full row for every other celltype. The "*" marks only
    the cells where that regulon is ACTUALLY in that specific celltype's
    own top N -- not every cell in a starred regulon's row is starred.

    Parameters
    ----------
    rss : `CalculateScenicRSS` output.
    celltype_key : name for the celltype axis, used only if `rss.index`
        doesn't already have one (`CalculateScenicRSS` sets it).
    top_n : regulons kept per celltype, deduplicated across celltypes.
    cmap, figsize : passed to `PlotSignificanceHeatmap`. `center` is
        fixed at `None` (RSS isn't diverging like log2FC, so the
        default `center=0` diverging behavior doesn't apply here).
    title : defaults to "Regulon Specificity Score (* = top {top_n} in
        that celltype)".

    Returns
    -------
    (value_table, flag_table, annot_table, rss_long) :
        value_table/flag_table/annot_table are `PlotSignificanceHeatmap`'s
        pivoted (regulon x celltype) outputs. rss_long is the long-format
        table the heatmap was built from -- one row per (celltype,
        regulon), with `RSS` and `is_top_flag` (0 = top-N in that
        celltype) columns -- sorted by celltype, then is_top_flag (top-N
        rows first), then RSS descending.
    """
    from sctools.plots import PlotSignificanceHeatmap

    top_per_celltype = {ct: set(rss.loc[ct].nlargest(top_n).index) for ct in rss.index}
    top_regulons = pd.Index(sorted(set().union(*top_per_celltype.values())))

    rss_indexed = rss.copy()
    rss_indexed.index.name = rss_indexed.index.name or celltype_key
    group_col = rss_indexed.index.name

    rss_long = (
        rss_indexed[top_regulons]
        .reset_index()
        .melt(id_vars=group_col, var_name="regulon", value_name="RSS")
    )

    # Not a p-value: 0 marks a (celltype, regulon) pair where that regulon
    # is actually in THIS celltype's own top N (not just pulled into the
    # shown columns by being top N for a different celltype).
    # padj_threshold=0.5 below turns flag==0 into the "*" annotation.
    rss_long["is_top_flag"] = rss_long.apply(
        lambda row: 0 if row["regulon"] in top_per_celltype[row[group_col]] else 1,
        axis=1,
    )

    if title is None:
        title = f"Regulon Specificity Score (* = top {top_n} in that celltype)"

    value_table, flag_table, annot_table = PlotSignificanceHeatmap(
        rss_long,
        x=group_col,
        y="regulon",
        value_col="RSS",
        pvalue_col="is_top_flag",
        padj_threshold=0.5,
        cmap=cmap,
        center=None,
        cbar_label="RSS",
        title=title,
        figsize=figsize,
    )

    rss_long = rss_long.sort_values(
        [group_col, "is_top_flag", "RSS"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return value_table, flag_table, annot_table, rss_long


def CalculateScenicRSSByCondition(
    adata,
    celltype_key,
    condition_key,
    *,
    auc_key="X_scenic_auc",
):
    """
    Run `CalculateScenicRSS` separately for each value of
    `adata.obs[condition_key]` (e.g. WT/KO), subsetting both `adata` and
    its AUCell matrix per condition. Never mutates `adata`.

    Parameters
    ----------
    adata : AnnData with `auc_key` in `.obsm` and `celltype_key`/
        `condition_key` in `.obs`.
    celltype_key : `adata.obs` column defining cell populations.
    condition_key : `adata.obs` column defining the conditions to split by.
    auc_key : `.obsm` key holding the cells x regulons AUCell matrix.

    Returns
    -------
    dict {condition_value: rss_dataframe}, one `CalculateScenicRSS`
    output per condition.
    """
    rss_by_condition = {}

    for condition_value in adata.obs[condition_key].unique():
        adata_subset = adata[adata.obs[condition_key] == condition_value]
        rss_by_condition[condition_value] = CalculateScenicRSS(
            adata_subset,
            adata_subset.obsm[auc_key],
            celltype_key=celltype_key,
        )

    return rss_by_condition


def _PlotByConditionHeatmap(
    long_data,
    *,
    x_col,
    y_col,
    value_col,
    is_top_col,
    column_order,
    row_order,
    group_size,
    scale_rows,
    cmap,
    center,
    cbar_label,
    title,
    annotate_values=False,
    annotate_top_only=False,
    value_fmt="{:.2f}",
):
    """
    Shared renderer behind `PlotScenicRSSByConditionHeatmap` and
    `PlotScenicActivityDifferenceHeatmap`: pivots `long_data` into a
    regulon x column heatmap using `sctools.plots.PlotSignificanceHeatmap`'s
    pivot/annotation machinery (`plot=False`) for the "*" markers, then
    renders with `seaborn.heatmap`, drawing a black separator line after
    every `group_size` columns (each celltype's condition block).

    Both callers already reduce their data to one row per (x_col, y_col)
    pair with a value and an `is_top_col` flag (0 = starred, the
    `PlotSignificanceHeatmap` `pvalue_col` convention) before calling
    this -- what differs between an RSS heatmap and a continuous-AUCell
    difference heatmap is only the value scale (`scale_rows`) and
    colormap, not the pivot/annotation/separator mechanics, so that part
    lives here once instead of twice.

    Parameters
    ----------
    long_data : one row per (x_col, y_col) combination present in
        `column_order`/`row_order`.
    x_col, y_col, value_col, is_top_col : column names in `long_data`.
    column_order, row_order : explicit, already-deduplicated axis orders.
    group_size : columns per black-separator group (e.g. 2 for a
        normal/compare, or reference/test, condition pair per celltype).
    scale_rows : if True, z-score each row across the displayed columns
        for display only. Continuous AUCell activity differs in overall
        magnitude between regulons, so a shared color scale would be
        dominated by the most active ones; RSS is already a bounded,
        per-celltype-relative specificity score and doesn't need this.
    cmap, center : passed to `seaborn.heatmap` (`center=None` for a
        one-sided scale like RSS, `center=0` for a diverging scale like
        row z-scored AUCell).
    cbar_label, title : figure labels.
    annotate_values : if True, annotate cells with the original (unscaled)
        `value_table` numbers instead of the default "*" top-flag marker
        -- the color still follows `display_table` (row-scaled when
        `scale_rows=True`), so this reads the actual magnitude off cells
        whose color only encodes the relative pattern.
    annotate_top_only : with `annotate_values=True`, restrict the numeric
        annotation to cells where `is_top_col` flags a "top" entry
        (blank elsewhere) instead of labeling every cell.
    value_fmt : format string applied to each annotated number.

    Returns
    -------
    (fig, ax, value_table, display_table, flag_table, annot_table) :
    `value_table` is the original (unscaled) pivoted values; `display_table`
    is what was actually plotted (identical to `value_table` when
    `scale_rows=False`); `annot_table` is whatever was actually drawn in
    each cell ("*"/"" or formatted numbers, depending on `annotate_values`).
    """
    from sctools.plots import PlotSignificanceHeatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    value_table, flag_table, annot_table = PlotSignificanceHeatmap(
        long_data,
        x=x_col,
        y=y_col,
        value_col=value_col,
        pvalue_col=is_top_col,
        padj_threshold=0.5,
        order_x=column_order,
        order_y=row_order,
        sort_by_magnitude=False,
        plot=False,
    )

    if scale_rows:
        # z-score each regulon's row across the displayed columns, for
        # display only -- the returned value_table stays on the
        # original AUCell scale
        row_mean = value_table.mean(axis=1)
        row_sd = value_table.std(axis=1)
        display_table = value_table.sub(row_mean, axis=0).div(row_sd, axis=0)
        # a regulon with zero variance across the displayed columns has
        # no defined z-score -- show it as 0 rather than leaving NaN
        display_table = display_table.fillna(0)
    else:
        display_table = value_table

    if annotate_values:
        # label with the original (unscaled) magnitude, not the z-score
        # actually driving the color -- a z-score number on its own isn't
        # biologically interpretable the way the raw value is
        annot_table = value_table.applymap(
            lambda v: value_fmt.format(v) if pd.notna(v) else ""
        )
        if annotate_top_only:
            # flag_table follows PlotSignificanceHeatmap's pvalue_col
            # convention: 0 = "top" -- blank out every non-top cell
            annot_table = annot_table.where(flag_table == 0, "")

    fig, ax = plt.subplots(
        figsize=(max(10, 0.5 * len(column_order)), max(6, 0.35 * len(row_order)))
    )

    sns.heatmap(
        display_table,
        ax=ax,
        cmap=cmap,
        center=center,
        cbar_kws={"label": cbar_label},
        annot=annot_table,
        fmt="",
        linewidths=0.4,
        linecolor="white",
    )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # black separator after every group_size columns. sns.heatmap places
    # column boundaries at integer positions (0, 1, 2, ...), unlike
    # imshow's -0.5 offset.
    for i in range(1, len(column_order) // group_size):
        ax.axvline(i * group_size, color="black", linewidth=1.5)

    plt.tight_layout()
    plt.show()

    return fig, ax, value_table, display_table, flag_table, annot_table


def PlotScenicRSSByConditionHeatmap(
    rss_by_condition,
    normal_condition,
    compare_condition,
    *,
    top_n=10,
    stage_order=None,
    cmap="viridis",
):
    """
    Heatmap comparing regulon specificity between two conditions, one
    celltype at a time: columns are "{celltype} {normal_condition}",
    "{celltype} {compare_condition}" pairs (e.g. "G0 WT", "G0 KO", "G1
    WT", ...), separated by a black vertical line per celltype. Built on
    the shared `_PlotByConditionHeatmap` renderer, with `scale_rows=False`
    since RSS is already a bounded, per-celltype-relative specificity
    score -- see `PlotScenicActivityDifferenceHeatmap` for the
    continuous-AUCell analog that does need row scaling.

    Rows are the union of each celltype's own top-N regulons across both
    conditions, so every celltype x condition column can be compared
    against the same regulon set. The "*" marks only the cells where
    that regulon is ACTUALLY in that specific column's own top N -- not
    every cell in a starred regulon's row is starred (same convention as
    `PlotScenicRSSHeatmap`).

    Parameters
    ----------
    rss_by_condition : `CalculateScenicRSSByCondition` output.
    normal_condition, compare_condition : keys into `rss_by_condition`;
        also set the left/right order within each celltype's column pair.
    top_n : regulons kept per celltype (union across both conditions).
    stage_order : optional explicit celltype order (e.g. maturation
        stages); celltypes not listed are dropped. Defaults to
        `rss_by_condition[normal_condition].index` order.
    cmap : passed to `seaborn.heatmap`.

    Returns
    -------
    (value_table, flag_table, annot_table, rss_long) : see
    `PlotScenicRSSHeatmap` -- same four objects, now with a
    "celltype_condition" column/axis instead of plain celltype.
    """
    rss_normal = rss_by_condition[normal_condition]
    rss_compare = rss_by_condition[compare_condition]

    if stage_order is not None:
        celltypes = [ct for ct in stage_order if ct in rss_normal.index]
    else:
        celltypes = list(rss_normal.index)

    conditions = [(normal_condition, rss_normal), (compare_condition, rss_compare)]

    # top-N regulons per celltype, unioned across both conditions AND
    # across all celltypes -- one shared regulon axis so every column is
    # comparable against the same rows
    top_regulons = set()
    for ct in celltypes:
        for _, rss_table in conditions:
            top_regulons |= set(rss_table.loc[ct].nlargest(top_n).index)
    top_regulons = sorted(top_regulons)

    # separately: is this regulon actually top-N for THIS specific
    # celltype+condition column? (not just pulled in because it's top-N
    # for some other celltype/condition) -- this is what gets starred
    top_per_column = {
        (ct, condition): set(rss_table.loc[ct].nlargest(top_n).index)
        for ct in celltypes
        for condition, rss_table in conditions
    }

    records = []
    for ct in celltypes:
        for condition, rss_table in conditions:
            top_set = top_per_column[(ct, condition)]
            for regulon in top_regulons:
                records.append({
                    "celltype_condition": f"{ct} {condition}",
                    "regulon": regulon,
                    "RSS": rss_table.loc[ct, regulon],
                    # not a p-value -- 0 means "in this column's own top
                    # N"; padj_threshold=0.5 below turns 0 into "*"
                    "is_top_flag": 0 if regulon in top_set else 1,
                })

    rss_long = pd.DataFrame(records)

    column_order = [
        f"{ct} {cond}" for ct in celltypes for cond in (normal_condition, compare_condition)
    ]

    _fig, _ax, value_table, _display_table, flag_table, annot_table = _PlotByConditionHeatmap(
        rss_long,
        x_col="celltype_condition",
        y_col="regulon",
        value_col="RSS",
        is_top_col="is_top_flag",
        column_order=column_order,
        row_order=top_regulons,
        group_size=2,
        scale_rows=False,
        cmap=cmap,
        center=None,
        cbar_label="RSS",
        title=(
            f"Regulon specificity by celltype x condition "
            f"(top {top_n} per celltype, * = top {top_n} in that column)"
        ),
    )

    return value_table, flag_table, annot_table, rss_long


def CalculateScenicActivityDifference(
    scenic_activity,
    *,
    condition_key,
    celltype_key,
    reference_condition,
    test_condition,
    sample_key="sample",
    value_key="mean_auc",
):
    """
    Calculate the difference in mean continuous AUCell activity between
    two conditions for each celltype x regulon combination.

    The input must contain sample-level AUCell summaries, typically from
    `SummarizeScenicActivity()`. Because AUCell was already summarized
    within each biological sample, condition means are calculated across
    biological replicates rather than by pooling all cells -- so each
    biological sample contributes equally to the comparison, independently
    of how many cells it has.

    The comparison is always:
        auc_difference = mean(test_condition) - mean(reference_condition)

    Therefore, for KO vs WT:
        positive values --> higher mean AUCell activity in KO
        negative values --> higher mean AUCell activity in WT

    This is a descriptive comparison only. No statistical test or
    multiple-testing correction is performed -- see the REACTOR/ROTS
    functions above for a statistically tested alternative.

    Parameters
    ----------
    scenic_activity : sample-level AUCell summary table (e.g.
        `SummarizeScenicActivity` output).
    condition_key : column identifying the experimental condition.
    celltype_key : column identifying the cell type or maturation stage.
    reference_condition, test_condition : e.g. "WT", "KO".
    sample_key : column identifying biological samples.
    value_key : column containing sample-level AUCell values.

    Returns
    -------
    pd.DataFrame : one row per celltype x regulon, with `mean_{reference}`,
    `mean_{test}` and `auc_difference` columns.
    """
    # fail before doing any aggregation if a required column is missing
    required_columns = {sample_key, condition_key, celltype_key, "regulon", value_key}
    missing_columns = required_columns.difference(scenic_activity.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if reference_condition == test_condition:
        raise ValueError("`reference_condition` and `test_condition` must be different.")

    # keep only the two conditions involved in this contrast
    selected_conditions = {reference_condition, test_condition}
    activity = scenic_activity.loc[
        scenic_activity[condition_key].isin(selected_conditions)
    ].copy()

    found_conditions = set(activity[condition_key].unique())
    if found_conditions != selected_conditions:
        raise ValueError(
            f"Expected conditions {sorted(selected_conditions)}, "
            f"but found {sorted(found_conditions)}."
        )

    # scenic_activity already has one AUCell summary per sample x celltype
    # x regulon -- average those sample-level values within each condition,
    # giving equal weight to each biological replicate regardless of how
    # many cells it contributed
    condition_activity = (
        activity
        .groupby([celltype_key, "regulon", condition_key], observed=True)[value_key]
        .mean()
        .unstack(condition_key)
    )

    reference_column = f"mean_{reference_condition}"
    test_column = f"mean_{test_condition}"

    condition_activity = condition_activity.rename(
        columns={reference_condition: reference_column, test_condition: test_column}
    ).reset_index()

    # difference is always test - reference, on the original AUCell
    # scale (not a fold change)
    condition_activity["auc_difference"] = (
        condition_activity[test_column] - condition_activity[reference_column]
    )

    return condition_activity


def SelectTopScenicActivityDifferences(
    scenic_difference,
    *,
    celltype_key,
    top_n,
    difference_key="auc_difference",
):
    """
    Select regulons with the largest absolute AUCell activity differences
    within each cell type or maturation stage.

    Regulons are ranked by |auc_difference|, so both positive and negative
    changes can be selected. Selection is performed independently within
    each cell type to ensure that all populations are represented.

    This is a descriptive ranking only and does not represent statistical
    significance.

    Parameters
    ----------
    scenic_difference : `CalculateScenicActivityDifference` output.
    celltype_key : column identifying cell types or maturation stages.
    top_n : number of regulons selected per cell type.
    difference_key : column containing the AUCell difference.

    Returns
    -------
    pd.DataFrame : selected celltype x regulon comparisons, ranked by
    absolute AUCell difference within each cell type.
    """
    required_columns = {celltype_key, "regulon", difference_key}
    missing_columns = required_columns.difference(scenic_difference.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if top_n < 1:
        raise ValueError("`top_n` must be at least 1.")

    ranked = scenic_difference.copy()

    # absolute difference is used only for ranking: positive and negative
    # AUCell changes are considered equally when selecting the largest effects
    ranked["_abs_difference"] = ranked[difference_key].abs()

    top_differences = (
        ranked
        .sort_values([celltype_key, "_abs_difference"], ascending=[True, False])
        .groupby(celltype_key, observed=True)
        .head(top_n)
        .drop(columns="_abs_difference")
        .reset_index(drop=True)
    )

    return top_differences


def PlotScenicActivityDifferenceHeatmap(
    scenic_difference,
    selected_differences,
    *,
    celltype_key,
    reference_condition,
    test_condition,
    stage_order=None,
    cmap="vlag",
    annotate_values=False,
    annotate_top_only=False,
    value_fmt="{:.2f}",
):
    """
    Heatmap of row z-scored mean AUCell activity for the regulons
    selected by `SelectTopScenicActivityDifferences`, across cell
    types/maturation stages and two experimental conditions -- the
    continuous-AUCell analog of `PlotScenicRSSByConditionHeatmap`, built
    on the same shared `_PlotByConditionHeatmap` renderer.

    Columns are "{celltype} {reference_condition}", "{celltype}
    {test_condition}" pairs (e.g. "G0 WT", "G0 KO", "G1 WT", ...),
    separated by a black vertical line per celltype, matching the RSS
    heatmap's layout. By default (`annotate_values=False`), "*" marks a
    regulon selected as a top AUCell difference for that celltype -- in
    both its reference and test columns, since
    `SelectTopScenicActivityDifferences` ranks by |difference| once per
    celltype, not separately per condition (unlike RSS's own per-column
    top-N). Set `annotate_values=True` to label cells with the actual
    mean AUCell number instead (see below).

    Unlike RSS (already a bounded, per-celltype-relative specificity
    score), raw mean AUCell activity differs in overall magnitude between
    regulons, so each regulon's row is centered and scaled (z-score)
    across the displayed columns for display only -- this changes only
    the color scale, not the returned `mean_auc_table` (always the
    original AUCell means) or any earlier quantitative comparison
    (`CalculateScenicActivityDifference`'s `auc_difference` is unscaled).

    Parameters
    ----------
    scenic_difference : `CalculateScenicActivityDifference` output.
    selected_differences : `SelectTopScenicActivityDifferences` output;
        its regulons get starred, one celltype at a time.
    celltype_key : column identifying cell types/maturation stages in
        both input tables.
    reference_condition, test_condition : e.g. "WT", "KO" -- also fix
        `mean_{reference_condition}`/`mean_{test_condition}` as the
        value columns read from `scenic_difference`.
    stage_order : optional explicit celltype order; celltypes not listed
        are dropped. Defaults to `scenic_difference[celltype_key]`'s own
        appearance order.
    cmap : passed to `seaborn.heatmap`; diverging by default since
        row-scaled values are centered at 0.
    annotate_values : if True, label each cell with its original
        (unscaled) mean AUCell value instead of the "*" top-difference
        marker -- the color still follows the row z-score, so this reads
        the actual AUCell magnitude off a cell whose color only encodes
        the relative pattern.
    annotate_top_only : with `annotate_values=True`, only label the
        cells selected as a top AUCell difference for that celltype
        (blank elsewhere) instead of labeling every cell.
    value_fmt : format string applied to each annotated number.

    Returns
    -------
    (mean_auc_table, scaled_table) : `mean_auc_table` is the original
    (unscaled) celltype x condition mean AUCell pivot; `scaled_table` is
    the row z-scored version actually plotted.
    """
    reference_col = f"mean_{reference_condition}"
    test_col = f"mean_{test_condition}"

    required = {celltype_key, "regulon", reference_col, test_col}
    missing = required.difference(scenic_difference.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if "regulon" not in selected_differences.columns:
        raise ValueError("`selected_differences` must contain a 'regulon' column.")

    if stage_order is not None:
        celltypes = [ct for ct in stage_order if ct in set(scenic_difference[celltype_key])]
    else:
        celltypes = list(pd.unique(scenic_difference[celltype_key]))

    # a regulon is starred in a given celltype's columns if it was
    # selected as a top |difference| regulon for THAT celltype --
    # selection is per-celltype, not per condition, so both the
    # reference and test column get the same flag
    selected_per_celltype = {
        ct: set(selected_differences.loc[selected_differences[celltype_key] == ct, "regulon"])
        for ct in celltypes
    }

    # union of every celltype's selected regulons -- one shared regulon
    # axis so every column is comparable against the same rows, same
    # convention as PlotScenicRSSByConditionHeatmap
    row_order = sorted(set(selected_differences["regulon"]))

    conditions = [(reference_condition, reference_col), (test_condition, test_col)]

    records = []
    for ct in celltypes:
        ct_data = scenic_difference.loc[scenic_difference[celltype_key] == ct]
        selected_set = selected_per_celltype[ct]
        for regulon in row_order:
            regulon_row = ct_data.loc[ct_data["regulon"] == regulon]
            if regulon_row.empty:
                # this regulon was selected as a top difference for a
                # different celltype and simply wasn't tested in this one
                continue
            for condition, value_col in conditions:
                records.append({
                    "celltype_condition": f"{ct} {condition}",
                    "regulon": regulon,
                    "mean_auc": regulon_row[value_col].iloc[0],
                    "is_top_flag": 0 if regulon in selected_set else 1,
                })

    activity_long = pd.DataFrame(records)

    column_order = [f"{ct} {cond}" for ct in celltypes for cond, _ in conditions]

    if annotate_values:
        title = "Regulon activity across maturation stages (labels = mean AUCell"
        title += ", top difference only)" if annotate_top_only else ")"
    else:
        title = "Regulon activity across maturation stages (* = top AUCell difference for that celltype)"

    _fig, _ax, mean_auc_table, scaled_table, _flag_table, _annot_table = _PlotByConditionHeatmap(
        activity_long,
        x_col="celltype_condition",
        y_col="regulon",
        value_col="mean_auc",
        is_top_col="is_top_flag",
        column_order=column_order,
        row_order=row_order,
        group_size=2,
        scale_rows=True,
        cmap=cmap,
        center=0,
        cbar_label="Row-scaled mean AUCell",
        title=title,
        annotate_values=annotate_values,
        annotate_top_only=annotate_top_only,
        value_fmt=value_fmt,
    )

    return mean_auc_table, scaled_table


def PlotScenicActivityByConditionHeatmap(
    scenic_difference,
    *,
    reference_condition,
    test_condition,
    celltype_key="celltype",
    top_n=10,
    stage_order=None,
    cmap="viridis",
    annotate_values=True,
    value_fmt="{:.2f}",
):
    """
    Heatmap of raw mean AUCell activity, one celltype at a time: the
    top-N regulons for each "{celltype} {condition}" column, ranked by
    that column's OWN AUCell value -- the continuous-AUCell analog of
    `PlotScenicRSSByConditionHeatmap`'s per-column top-N convention.

    Unlike `PlotScenicActivityDifferenceHeatmap` (which selects regulons
    once per celltype by |mean_test - mean_reference|, and needs a
    `SelectTopScenicActivityDifferences` selection passed in), this
    selects independently per column and needs nothing beyond
    `CalculateScenicActivityDifference`'s own output -- e.g. a celltype's
    WT column can star a different top-10 than its KO column. No row
    scaling: this shows the plain top-AUCell picture, not a
    cross-condition comparison.

    Parameters
    ----------
    scenic_difference : `CalculateScenicActivityDifference` output.
    reference_condition, test_condition : e.g. "WT", "KO" -- also fix
        `mean_{reference_condition}`/`mean_{test_condition}` as the
        value columns read from `scenic_difference`.
    celltype_key : column identifying cell types/maturation stages.
    top_n : regulons kept per column (independently -- not unioned by
        celltype the way `PlotScenicActivityDifferenceHeatmap` is).
    stage_order : optional explicit celltype order; celltypes not listed
        are dropped. Defaults to `scenic_difference[celltype_key]`'s own
        appearance order.
    cmap : passed to `seaborn.heatmap`; sequential by default (raw AUCell
        is non-negative, like RSS -- no need for a diverging scale).
    annotate_values : label the top-N cells with their raw mean AUCell
        value (the default -- set False for no annotation at all).
    value_fmt : format string applied to each annotated number.

    Returns
    -------
    (mean_auc_table, full_mean_auc_table) : `mean_auc_table` is the
    celltype x condition mean AUCell pivot restricted to the plotted
    top-N regulons (no separate scaled table -- rows aren't scaled
    here); `full_mean_auc_table` is the same pivot for every regulon in
    `scenic_difference`, not just the ones that made a column's top-N.
    """
    reference_col = f"mean_{reference_condition}"
    test_col = f"mean_{test_condition}"

    required = {celltype_key, "regulon", reference_col, test_col}
    missing = required.difference(scenic_difference.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if stage_order is not None:
        celltypes = [ct for ct in stage_order if ct in set(scenic_difference[celltype_key])]
    else:
        celltypes = list(pd.unique(scenic_difference[celltype_key]))

    conditions = [(reference_condition, reference_col), (test_condition, test_col)]

    # top-N regulons for THIS specific celltype+condition column, ranked
    # by that column's own raw AUCell value -- same idea as RSS's
    # top_per_column, just on mean AUCell instead of RSS
    top_per_column = {
        (ct, condition): set(
            scenic_difference.loc[scenic_difference[celltype_key] == ct]
            .nlargest(top_n, value_col)["regulon"]
        )
        for ct in celltypes
        for condition, value_col in conditions
    }

    # union of every column's own top-N -- one shared regulon axis so
    # every column is comparable against the same rows
    top_regulons = sorted(set().union(*top_per_column.values()))

    records = []
    for ct in celltypes:
        ct_data = scenic_difference.loc[scenic_difference[celltype_key] == ct]
        for condition, value_col in conditions:
            top_set = top_per_column[(ct, condition)]
            for regulon in top_regulons:
                regulon_row = ct_data.loc[ct_data["regulon"] == regulon]
                if regulon_row.empty:
                    continue
                records.append({
                    "celltype_condition": f"{ct} {condition}",
                    "regulon": regulon,
                    "mean_auc": regulon_row[value_col].iloc[0],
                    "is_top_flag": 0 if regulon in top_set else 1,
                })

    activity_long = pd.DataFrame(records)
    column_order = [f"{ct} {cond}" for ct in celltypes for cond, _ in conditions]

    _fig, _ax, mean_auc_table, _display_table, _flag_table, _annot_table = _PlotByConditionHeatmap(
        activity_long,
        x_col="celltype_condition",
        y_col="regulon",
        value_col="mean_auc",
        is_top_col="is_top_flag",
        column_order=column_order,
        row_order=top_regulons,
        group_size=2,
        scale_rows=False,
        cmap=cmap,
        center=None,
        cbar_label="Mean AUCell",
        title=f"Top {top_n} mean AUCell regulons per celltype x condition",
        annotate_values=annotate_values,
        annotate_top_only=True,
        value_fmt=value_fmt,
    )

    # every regulon's mean AUCell per column, not just the plotted
    # top-N -- one Series per column, concatenated side by side so a
    # regulon missing from a given celltype (never tested there) is NaN
    # in that column rather than dropping the whole row
    full_parts = [
        scenic_difference.loc[scenic_difference[celltype_key] == ct]
        .set_index("regulon")[value_col]
        .rename(f"{ct} {condition}")
        for ct in celltypes
        for condition, value_col in conditions
    ]
    full_mean_auc_table = pd.concat(full_parts, axis=1)[column_order]

    return mean_auc_table, full_mean_auc_table


def PlotScenicAucellByConditionHeatmap(
    adata,
    *,
    celltype_key,
    condition_key,
    normal_condition,
    compare_condition,
    auc_key="X_scenic_auc",
    top_n=10,
    stage_order=None,
    cmap="viridis",
    annotate_values=True,
    value_fmt="{:.2f}",
):
    """
    Heatmap of raw mean AUCell activity computed directly from
    `adata.obsm[auc_key]`, pooling all cells within each celltype x
    condition group -- the no-pipeline version of
    `PlotScenicActivityByConditionHeatmap`: no `SummarizeScenicActivity`
    or `CalculateScenicActivityDifference` step needed first, just
    `adata` + its AUCell matrix.

    Because this pools every cell in a group directly (not the
    per-sample-then-averaged design `SummarizeScenicActivity` uses), a
    biological sample with more cells contributes proportionally more to
    the mean here -- fine for a quick look, but use
    `PlotScenicActivityByConditionHeatmap` (via `SummarizeScenicActivity`
    + `CalculateScenicActivityDifference`) instead when biological
    replicates need equal weight.

    top-N regulons are selected independently per "{celltype}
    {condition}" column, ranked by that column's own mean AUCell -- same
    convention as `PlotScenicActivityByConditionHeatmap` and
    `PlotScenicRSSByConditionHeatmap`.

    Parameters
    ----------
    adata : AnnData with `auc_key` in `.obsm` (cells x regulons AUCell,
        as a DataFrame) and `celltype_key`/`condition_key` in `.obs`.
    celltype_key, condition_key : `adata.obs` columns.
    normal_condition, compare_condition : values of `condition_key`;
        also set column order within each celltype's pair.
    auc_key : `.obsm` key holding the AUCell matrix.
    top_n : regulons kept per column.
    stage_order : optional explicit celltype order; celltypes not listed
        are dropped. Defaults to `adata.obs[celltype_key]`'s own
        appearance order.
    cmap : passed to `seaborn.heatmap`; sequential by default (raw AUCell
        is non-negative, like RSS).
    annotate_values : label the top-N cells with their raw mean AUCell
        value (the default -- set False for no annotation at all).
    value_fmt : format string applied to each annotated number.

    Returns
    -------
    (mean_auc_table, full_mean_auc_table) : `mean_auc_table` is the
    celltype x condition mean AUCell pivot restricted to the plotted
    top-N regulons; `full_mean_auc_table` is the same pivot for every
    regulon in `adata.obsm[auc_key]`, not just the ones that made a
    column's top-N.
    """
    if auc_key not in adata.obsm:
        raise ValueError(f"'{auc_key}' was not found in adata.obsm.")

    auc_matrix = adata.obsm[auc_key]
    if not isinstance(auc_matrix, pd.DataFrame):
        raise ValueError(f"adata.obsm['{auc_key}'] must be a DataFrame of cells x regulons.")

    for key in (celltype_key, condition_key):
        if key not in adata.obs.columns:
            raise ValueError(f"'{key}' was not found in adata.obs.")

    metadata = adata.obs[[celltype_key, condition_key]].copy()
    activity_table = metadata.join(auc_matrix)
    regulon_columns = list(auc_matrix.columns)

    # pool every cell within each celltype x condition group directly --
    # no per-sample averaging/equal-replicate-weighting here, unlike
    # SummarizeScenicActivity + CalculateScenicActivityDifference
    mean_by_group = activity_table.groupby(
        [celltype_key, condition_key], observed=True,
    )[regulon_columns].mean()

    if stage_order is not None:
        celltypes = [
            ct for ct in stage_order
            if ct in set(mean_by_group.index.get_level_values(celltype_key))
        ]
    else:
        celltypes = list(pd.unique(activity_table[celltype_key]))

    conditions = [normal_condition, compare_condition]

    # top-N regulons for THIS specific celltype+condition column, ranked
    # by that column's own raw mean AUCell
    top_per_column = {
        (ct, cond): set(mean_by_group.loc[(ct, cond)].nlargest(top_n).index)
        for ct in celltypes
        for cond in conditions
        if (ct, cond) in mean_by_group.index
    }

    # union of every column's own top-N -- one shared regulon axis so
    # every column is comparable against the same rows
    top_regulons = sorted(set().union(*top_per_column.values()))

    records = []
    for ct in celltypes:
        for cond in conditions:
            if (ct, cond) not in mean_by_group.index:
                continue
            top_set = top_per_column[(ct, cond)]
            row = mean_by_group.loc[(ct, cond)]
            for regulon in top_regulons:
                records.append({
                    "celltype_condition": f"{ct} {cond}",
                    "regulon": regulon,
                    "mean_auc": row[regulon],
                    "is_top_flag": 0 if regulon in top_set else 1,
                })

    activity_long = pd.DataFrame(records)
    column_order = [f"{ct} {cond}" for ct in celltypes for cond in conditions]

    _fig, _ax, mean_auc_table, _display_table, _flag_table, _annot_table = _PlotByConditionHeatmap(
        activity_long,
        x_col="celltype_condition",
        y_col="regulon",
        value_col="mean_auc",
        is_top_col="is_top_flag",
        column_order=column_order,
        row_order=top_regulons,
        group_size=2,
        scale_rows=False,
        cmap=cmap,
        center=None,
        cbar_label="Mean AUCell",
        title=f"Top {top_n} mean AUCell regulons per celltype x condition",
        annotate_values=annotate_values,
        annotate_top_only=True,
        value_fmt=value_fmt,
    )

    # every regulon's mean AUCell per column, not just the plotted
    # top-N -- mean_by_group already has every regulon as a column, so
    # this is just a transpose-and-select rather than a fresh groupby
    full_parts = [
        mean_by_group.loc[(ct, cond)].rename(f"{ct} {cond}")
        for ct in celltypes
        for cond in conditions
        if (ct, cond) in mean_by_group.index
    ]
    full_mean_auc_table = pd.concat(full_parts, axis=1)[column_order]

    return mean_auc_table, full_mean_auc_table


def CalculateScenicStageVariation(
    scenic_activity,
    *,
    condition_key,
    celltype_key,
    sample_key="sample",
    value_key="mean_auc",
):
    """
    Quantify how much each regulon's continuous AUCell activity varies
    across cell types or maturation stages, separately by condition.

    The input must contain sample-level AUCell summaries from
    `SummarizeScenicActivity()`. Sample values are first averaged within
    each condition x stage x regulon. Variation across stages is then
    summarized using the standard deviation and range.

    This is a descriptive analysis and no statistical test is performed --
    complementary to `CalculateScenicActivityDifference` (which compares
    two conditions at a fixed stage): this instead asks how much a
    regulon's own activity moves across stages, within one condition.

    Parameters
    ----------
    scenic_activity : `SummarizeScenicActivity` output.
    condition_key : column identifying the experimental condition.
    celltype_key : column identifying cell types or maturation stages.
    sample_key : column identifying biological samples.
    value_key : column containing sample-level AUCell values.

    Returns
    -------
    pd.DataFrame : one row per condition x regulon with `stage_sd`,
    `stage_range` and `n_stages`, ranked by decreasing `stage_sd`.
    """
    # columns required to reconstruct sample-level stage profiles
    required = {sample_key, condition_key, celltype_key, "regulon", value_key}
    missing = required.difference(scenic_activity.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # first average the biological samples within each condition and
    # stage -- scenic_activity already has sample-level values, so
    # samples contribute equally regardless of their original cell count
    stage_activity = (
        scenic_activity
        .groupby([condition_key, celltype_key, "regulon"], observed=True)[value_key]
        .mean()
        .reset_index(name="mean_stage_auc")
    )

    # for each condition x regulon, quantify variation across all stages;
    # stage_sd considers the complete activity profile, stage_range
    # reports only the distance between the highest and lowest stage
    stage_variation = (
        stage_activity
        .groupby([condition_key, "regulon"], observed=True)["mean_stage_auc"]
        .agg(stage_sd="std", stage_range=lambda x: x.max() - x.min(), n_stages="count")
        .reset_index()
    )

    # regulons with the largest variation across stages appear first
    # within each experimental condition
    stage_variation = stage_variation.sort_values(
        [condition_key, "stage_sd"], ascending=[True, False],
    ).reset_index(drop=True)

    return stage_variation


def SelectTopScenicStageVariation(
    stage_variation,
    *,
    condition_key,
    top_n=5,
    variation_key="stage_sd",
):
    """
    Select the regulons with the highest activity variation across stages,
    independently within each experimental condition.

    The union of the top regulons from all conditions is returned,
    avoiding duplicated regulons when the same regulon is highly variable
    in more than one condition.

    Parameters
    ----------
    stage_variation : `CalculateScenicStageVariation` output.
    condition_key : column identifying the experimental condition.
    top_n : number of top variable regulons selected per condition.
    variation_key : metric used to rank stage variation.

    Returns
    -------
    (top_variation, top_regulons) : ranked rows per condition, and the
    unique regulon names selected across conditions.
    """
    required = {condition_key, "regulon", variation_key}
    missing = required.difference(stage_variation.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if top_n < 1:
        raise ValueError("`top_n` must be at least 1.")

    # select the most variable regulons independently in each condition
    top_variation = (
        stage_variation
        .sort_values([condition_key, variation_key], ascending=[True, False])
        .groupby(condition_key, observed=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    # union across conditions -- a regulon present in both top lists
    # appears once
    top_regulons = top_variation["regulon"].drop_duplicates().tolist()

    return top_variation, top_regulons


def PlotScenicActivityAcrossStages(
    scenic_activity,
    regulon,
    *,
    condition_key,
    celltype_key,
    sample_key="sample",
    value_key="mean_auc",
    stage_order=None,
    condition_order=None,
    figsize=(8, 5),
):
    """
    Plot sample-level continuous AUCell activity across ordered cell
    states, for one regulon.

    Individual biological samples are shown separately, with the mean
    profile of each condition overlaid. No smoothing or statistical test
    is performed -- celltypes/maturation stages are treated as ordered
    categories, not continuous pseudotime.

    Parameters
    ----------
    scenic_activity : `SummarizeScenicActivity` output.
    regulon : regulon to visualize.
    condition_key : column identifying experimental conditions.
    celltype_key : column identifying maturation stages/cell states.
    sample_key : column identifying biological samples.
    value_key : column containing sample-level AUCell values.
    stage_order : order of maturation stages on the x-axis. Defaults to
        `scenic_activity[celltype_key]`'s own appearance order.
    condition_order : order of experimental conditions (legend/color
        assignment order). Defaults to `scenic_activity[condition_key]`'s
        own appearance order.
    figsize : passed to `plt.subplots`.

    Returns
    -------
    (fig, ax, plot_data) : figure, axes and the sample-level data plotted.
    """
    import matplotlib.pyplot as plt

    required = {sample_key, condition_key, celltype_key, "regulon", value_key}
    missing = required.difference(scenic_activity.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # select one regulon while keeping all biological samples and stages
    plot_data = scenic_activity.loc[scenic_activity["regulon"] == regulon].copy()
    if plot_data.empty:
        raise ValueError(f"Regulon '{regulon}' was not found.")

    # each biological sample should belong to only one condition -- a
    # sample split across conditions would silently double-count it in
    # both condition-mean lines below
    sample_conditions = plot_data[[sample_key, condition_key]].drop_duplicates()
    if sample_conditions[sample_key].duplicated().any():
        raise ValueError("A biological sample is associated with more than one condition.")

    if stage_order is None:
        stage_order = list(pd.unique(plot_data[celltype_key]))

    if condition_order is None:
        condition_order = list(pd.unique(plot_data[condition_key]))

    plot_data[celltype_key] = pd.Categorical(
        plot_data[celltype_key], categories=stage_order, ordered=True,
    )
    plot_data = plot_data.sort_values(celltype_key)

    fig, ax = plt.subplots(figsize=figsize)

    # plot each biological sample separately; samples from the same
    # condition share the same automatically assigned color
    for condition in condition_order:
        condition_data = plot_data.loc[plot_data[condition_key] == condition]
        color = ax._get_lines.get_next_color()

        for sample in condition_data[sample_key].unique():
            sample_data = condition_data.loc[condition_data[sample_key] == sample]
            ax.plot(
                sample_data[celltype_key], sample_data[value_key],
                marker="o", linewidth=1, alpha=0.45, color=color, label=sample,
            )

        # mean across biological samples at each stage
        condition_mean = (
            condition_data
            .groupby(celltype_key, observed=True)[value_key]
            .mean()
            .reindex(stage_order)
        )
        ax.plot(
            stage_order, condition_mean,
            marker="o", linewidth=3, color=color, label=f"{condition} mean",
        )

    ax.set_xlabel("Maturation stage")
    ax.set_ylabel("Mean AUCell score")
    ax.set_title(f"{regulon} activity across maturation stages")
    ax.legend(frameon=False)

    plt.tight_layout()

    return fig, ax, plot_data


# ── REACTOR (differential regulon activity across biological samples) ──
#
# REACTOR (Lindén et al., 2026 -- github.com/elolab/REACTOR) tests
# whether the *fraction of cells with a regulon active* differs between
# conditions, using biological samples (not cells) as replicates, scored
# with ROTS (Reproducibility-Optimized Test Statistic). It's an R
# package; these functions bridge to it via rpy2 and need a working `R`
# with REACTOR and its dependency ROTS (Bioconductor) installed in the
# same environment -- see the module docstring's environment note for
# why that's usually the `scenic_env`-style environment already used for
# the `pyscenic` CLI calls above, not necessarily the one running this
# notebook's own Python kernel.

def BuildReactorInputTables(
    adata,
    auc_table,
    *,
    sample_key,
    condition_key,
    celltype_key,
    seed=42,
    num_workers=3,
):
    """
    Binarize a cells x regulons AUCell matrix and build the three input
    tables `REACTOR::processData()` requires from it + `adata.obs`.
    Never mutates `adata` or `auc_table`.

    Binarization is pySCENIC's own `binarize()`: a threshold estimated
    separately per regulon from its own AUCell-score distribution
    (Hartigan's dip test picks unimodal vs bimodal; unimodal -> mean +
    2*SD, bimodal -> the density minimum between the two modes) -- not
    one fixed cutoff shared by every regulon.

    Parameters
    ----------
    adata : AnnData with `sample_key`/`condition_key`/`celltype_key` in `.obs`.
    auc_table : cells x regulons AUCell matrix (continuous scores),
        indexed the same as (all or a subset of) `adata.obs_names`.
    seed, num_workers : passed to `binarize()`.

    Returns
    -------
    (rbm, clustering, study_design, binary_auc, auc_thresholds) :
        rbm            -- cellID + one 0/1 column per regulon (REACTOR's
                           "RBM" argument).
        clustering     -- cellID, celltype (REACTOR's "Clustering").
        study_design   -- cellID, sample, condition (REACTOR's "StudyDesign").
        binary_auc     -- the raw cells x regulons 0/1 matrix `rbm` was
                           built from, e.g. to store back into
                           `adata.obsm` yourself if wanted.
        auc_thresholds -- per-regulon activity threshold `binarize()` used.
    """
    from pyscenic.binarization import binarize

    binary_auc, auc_thresholds = binarize(auc_table, seed=seed, num_workers=num_workers)

    # REACTOR's "RBM" (regulon binary matrix): one 0/1 column per
    # regulon, cell identifier as its own explicit column rather than
    # the DataFrame index (R data.frames sent over rpy2 don't carry a
    # pandas index the way REACTOR expects to read "cellID" from)
    rbm = binary_auc.copy()
    rbm.insert(0, "cellID", rbm.index.astype(str))
    rbm = rbm.reset_index(drop=True)

    # celltype label per cell, same cell set/order as rbm (REACTOR joins
    # these three tables internally on cellID, but keeping them aligned
    # here too makes the intermediate tables easier to sanity-check)
    clustering = adata.obs.loc[binary_auc.index, [celltype_key]].copy()
    clustering.insert(0, "cellID", clustering.index.astype(str))
    clustering = clustering.reset_index(drop=True)
    clustering.columns = ["cellID", "celltype"]

    # biological sample + experimental condition per cell -- this is
    # what lets REACTOR aggregate cells into (sample, celltype) groups
    # downstream, so replicates become biological samples, not cells
    study_design = adata.obs.loc[binary_auc.index, [sample_key, condition_key]].copy()
    study_design.insert(0, "cellID", study_design.index.astype(str))
    study_design = study_design.reset_index(drop=True)
    study_design.columns = ["cellID", "sample", "condition"]

    return rbm, clustering, study_design, binary_auc, auc_thresholds


def RunReactorProcessData(
    rbm,
    clustering,
    study_design,
    *,
    min_cells=10,
):
    """
    Run `REACTOR::processData()` via rpy2: for each (biological sample,
    celltype) pair with at least `min_cells` cells, calculates the
    percentage of cells in which each regulon is active.

    Parameters
    ----------
    rbm, clustering, study_design : `BuildReactorInputTables` output
        (column names "cellID"/regulons, "cellID"/"celltype",
        "cellID"/"sample"/"condition" respectively -- fixed, since
        that's what `BuildReactorInputTables` always produces).
    min_cells : REACTOR's own default; a (sample, celltype) combination
        below this is dropped before the differential analysis.

    Returns
    -------
    pd.DataFrame: "{celltype} | {regulon}" rows x biological-sample
    columns, percent of cells with the regulon active in that sample
    (NOT AUCell mean) -- pass this straight to
    `RunReactorDifferentialActivityROTS`.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # hand the three tables to R as global variables -- rpy2 has no
    # "call this R function with these Python objects as arguments"
    # shortcut for a bare `ro.r()` string, so the R snippet below reads
    # them back by the same names
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["rbm"] = ro.conversion.py2rpy(rbm)
        ro.globalenv["clustering"] = ro.conversion.py2rpy(clustering)
        ro.globalenv["studyDesign"] = ro.conversion.py2rpy(study_design)

    # condition_cName/donor_cName/cluster_cName are fixed to match the
    # column names BuildReactorInputTables always produces
    ro.r(f"""
    regulonActivity <- REACTOR::processData(
        minCells = {min_cells},
        RBM = rbm,
        StudyDesign = studyDesign,
        Clustering = clustering,
        condition_cName = "condition",
        donor_cName = "sample",
        cluster_cName = "celltype"
    )
    """)

    # read the R result back out of the R global environment into a
    # pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        regulon_activity = ro.conversion.rpy2py(ro.globalenv["regulonActivity"])

    return regulon_activity


def RunReactorDifferentialActivityROTS(
    regulon_activity,
    sample_conditions,
    *,
    reference_condition,
    test_condition,
    seed=42,
    K=100,
):
    """
    Run `REACTOR::differentialActivityAnalysis()` (ROTS-based) on a
    `RunReactorProcessData` table.

    ROTS' `groups` argument is positional -- aligned to `data`'s column
    order, not by name -- so building it by hand from a column list you
    read off separately is an easy way to silently mismatch samples to
    conditions. This derives `groups` directly from
    `regulon_activity.columns`, and raises if any column has no
    condition given.

    The comparison is defined as:
        test_condition      -> group 1
        reference_condition -> group 2

    The function follows the REACTOR vignette by passing the
    sample-level regulon activity matrix directly to
    `differentialActivityAnalysis()` without overriding the ROTS
    `log` argument.

    Parameters
    ----------
    regulon_activity : `RunReactorProcessData` output.

    sample_conditions : {sample_name: condition}, must cover every
        column of `regulon_activity`.

    reference_condition, test_condition : define the two groups used
        by ROTS (test -> group 1, reference -> group 2).

    seed : random seed passed to ROTS.

    K : largest top-list size considered during ROTS reproducibility
        optimization. The REACTOR vignette uses K=100.

    Returns
    -------
    pd.DataFrame
        REACTOR's differential activity table, containing the original
        biological-sample columns plus:
        `d`: Reproducibility-optimized ROTS test statistic.
        `p`: ROTS p-value.
        `fdr`: ROTS false discovery rate.
        `fc`: Effect estimate returned by ROTS.
        Rows are indexed by "{celltype} | {regulon}".

    Notes
    -----
    The complete ROTS object is retained in the R global environment
    as `ROTS_obj`, allowing the diagnostic plots implemented by ROTS
    and shown in the REACTOR vignette to be generated afterwards.
    """

    # fail before touching R at all if a column can't be mapped to a
    # condition -- a silent misalignment here would still "work" (ROTS
    # doesn't know a sample was assigned to the wrong group) and just
    # quietly produce a wrong result
    missing = set(regulon_activity.columns) - set(sample_conditions)

    if missing:
        raise ValueError(
            f"No condition given for sample(s): {sorted(missing)}"
        )

    # make sure only the two requested conditions are represented in
    # the samples that actually occur in regulon_activity
    valid_conditions = {
        test_condition,
        reference_condition,
    }

    unexpected_conditions = {
        sample_conditions[s]
        for s in regulon_activity.columns
        if sample_conditions[s] not in valid_conditions
    }

    if unexpected_conditions:
        raise ValueError(
            f"Unexpected condition(s): {sorted(unexpected_conditions)}. "
            f"Expected only '{test_condition}' and "
            f"'{reference_condition}'."
        )

    # build the positional groups vector in regulon_activity's own
    # column order, rather than trusting it to have been typed by hand
    # to match -- test_condition -> group 1,
    # reference_condition -> group 2
    condition_to_group = {
        test_condition: 1,
        reference_condition: 2,
    }

    groups = [
        condition_to_group[sample_conditions[s]]
        for s in regulon_activity.columns
    ]

    if len(set(groups)) != 2:
        raise ValueError(
            f"Expected exactly the 2 conditions "
            f"'{test_condition}'/'{reference_condition}' "
            f"among regulon_activity's columns; "
            f"found group codes {sorted(set(groups))}."
        )

    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # transfer the sample-level regulon activity matrix to R
    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):
        ro.globalenv["regulonActivity"] = (
            ro.conversion.py2rpy(regulon_activity)
        )

    # a plain integer vector doesn't need the pandas converter
    ro.globalenv["groups"] = ro.IntVector(groups)

    # differentialActivityAnalysis() returns:
    #   DAA_out[[1]] -> complete ROTS object
    #   DAA_out[[2]] -> REACTOR results table
    #
    # Keep the complete ROTS object in R so it remains available for
    # the diagnostic plots shown in the REACTOR vignette.
    ro.r(
        f"""
        DAA_out <- REACTOR::differentialActivityAnalysis(
            data = regulonActivity,
            groups = groups,
            seed = {seed},
            K = {K}
        )

        ROTS_obj <- DAA_out[[1]]
        reactor_results <- DAA_out[[2]]
        """
    )

    # only the result table needs to be transferred back to Python;
    # ROTS_obj remains available in the R global environment
    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):
        reactor_results = ro.conversion.rpy2py(
            ro.globalenv["reactor_results"]
        )

    return reactor_results