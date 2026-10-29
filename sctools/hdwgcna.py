"""
sctools.hdwgcna
===============
Co-expression network analysis (hdWGCNA) for single-cell data, with the
Python port `py_hdWGCNA`. Each function wraps one step of the hdWGCNA
workflow. The defaults follow hdWGCNA, except where the "Design notes"
below explain why another choice is safer with py_hdWGCNA.

The input is an AnnData with cell groups (cell types, clusters or stages),
biological samples and, if you have them, experimental conditions. The
tutorial notebook `tutoriais/07.hdWCGNA_tutorial.ipynb` shows the full
workflow on an example dataset (bone marrow neutrophils of germ-free and
SPF mice, Sheng et al. 2025, maturation stages; one sample per condition). The names used in the examples below are only placeholders.

Typical usage order:
    GetWgcnaPackageVersions()      -- versions of py_hdWGCNA and gseapy (for the record)
    SummarizeWgcnaInput()          -- cells per group x sample, sample -> condition
    PrepareWgcnaInput()            -- check inputs, put the chosen counts in layers['counts']
    GetCommonWgcnaGenes()          -- genes detected in >= 5% of cells in every condition (optional)
    SelectWgcnaGenes()             -- genes detected in >= 5% of cells
    BuildWgcnaMetacells()          -- metacells per group x sample
    SummarizeWgcnaMetacells()      -- cells and metacells per group
    SetWgcnaExpression()           -- metacell expression matrix used by the network
    CheckBicorMad()                -- is bicor safe for this matrix?
    TestWgcnaSoftPowers()          -- scale-free fit for each soft power
    SelectWgcnaSoftPower()         -- lowest power with R2 >= cutoff and slope < 0
    PlotWgcnaSoftPowers()          -- scale-free fit and connectivity plots
    BuildWgcnaNetwork()            -- TOM, module detection, module merging
    FlagSmallWgcnaModules()        -- modules that exist only because of min_module_size
    PlotWgcnaDendrogram()          -- gene tree with the module colors
    ComputeWgcnaEigengenes()       -- module eigengenes per cell (MEs)
    PlotWgcnaModuleFeatures()      -- eigengenes on the UMAP
    ComputeWgcnaKme()              -- kME (gene-module connectivity)
    PlotWgcnaKmes()                -- top genes by kME per module
    GetWgcnaHubGenes()             -- table of hub genes per module
    RunWgcnaEnrichr()              -- enrichment per module (Enrichr: GO, Hallmark, ...)
    SelectSignificantEnrichment()  -- keep terms with adjusted p < cutoff
    PlotWgcnaEnrichmentDotplot()   -- dot plot, top terms per module

Quick start
-----------
Placeholders: "soupx_counts" (ambient-corrected integer counts),
"soupx_log1p" (log-normalized expression of the same counts), "celltype",
"sample" and "condition" (obs columns), "TypeA"/"TypeB" (groups in the
network), "RareType" (a group removed before the analysis).

    from sctools.hdwgcna import *

    adata = PrepareWgcnaInput(
        adata, counts_layer="soupx_counts",
        required_cols=["sample", "condition", "celltype"],
        reduction="X_pca", group_col="celltype", exclude_groups=["RareType"],
    )
    adata = SelectWgcnaGenes(adata, fraction=0.05)
    adata = BuildWgcnaMetacells(
        adata, group_by=["celltype", "sample"], ident_group="celltype", min_cells=100,
    )
    adata = SetWgcnaExpression(adata, groups=["TypeA", "TypeB"], group_by="celltype")
    print(CheckBicorMad(adata))     # many genes with MAD = 0 -> use pearson
    adata, power_table = TestWgcnaSoftPowers(adata, cor_method="pearson")
    power = SelectWgcnaSoftPower(adata, r2_cutoff=0.80)
    fig = PlotWgcnaSoftPowers(adata, selected_power=power)
    adata, module_sizes = BuildWgcnaNetwork(adata, power=power, network_type="signed")
    print(FlagSmallWgcnaModules(module_sizes, min_size=50))
    fig = PlotWgcnaDendrogram(adata)
    adata, mes = ComputeWgcnaEigengenes(adata, layer="soupx_log1p", harmonize=False)
    adata, modules_df = ComputeWgcnaKme(
        adata, group_by="celltype", groups=["TypeA", "TypeB"], layer="soupx_log1p",
    )
    hub_genes = GetWgcnaHubGenes(adata, n_hubs=10)
    enrichment_df = RunWgcnaEnrichr(
        adata, databases=["GO_Biological_Process_2023", "GO_Molecular_Function_2023"],
        organism="mouse",
    )
    enrichment_sig = SelectSignificantEnrichment(enrichment_df, padj_cutoff=0.05)
    fig, ax = PlotWgcnaEnrichmentDotplot(enrichment_df, database="GO_Biological_Process_2023")

Where the results are stored
----------------------------
`py_hdWGCNA` keeps everything in `adata.uns['hdWGCNA'][wgcna_name]` (a dict).
Useful keys: 'genes_use', 'metacell_obj' (AnnData of metacells), 'dat_expr'
(genes x metacells array), 'dat_expr_genes', 'power_table', 'selected_power',
'modules_df' (gene -> module, color, kME_* columns), 'MEs', 'hMEs'.
The functions in this module also save their own arguments under the key
'sctools_params', one entry per step, so you can check later how a result
was made.

Requirements
------------
`py_hdWGCNA` (tested with 0.1.0), `gseapy` (tested with 1.2.1, only for
`RunWgcnaEnrichr`), `seaborn` (only for the dot plot) and `harmonypy` (only
if `harmonize=True`). They are imported inside the functions, so
`import sctools` still works in environments without them, and a notebook
never needs to import py_hdWGCNA itself: the plots and the hub gene table
also have wrappers here. Tested in the conda env `scRNA_NEW_signaling_py311`.
`construct_network` runs the dynamic tree cut in R, so the environment also
needs R with the WGCNA dependencies.

Design notes (read before changing a default)
---------------------------------------------
1. Counts layer. `py_hdWGCNA` always reads `adata.layers['counts']` for the
   gene filter and for the metacells. If you have counts corrected for
   ambient RNA (e.g. by SoupX), use them (`PrepareWgcnaInput` copies them to
   'counts'). Raw counts keep ambient RNA from the most expressed genes of
   the sample, and this can create false correlations between them. Use the
   log-normalized expression of the SAME counts for eigengenes and kME.

2. Correlation method. hdWGCNA in R uses bicor in TestSoftPowers but pearson
   in ConstructNetwork. Here the network reuses the method of the power test
   (py_hdWGCNA stores it), so both steps use the same one. Do not use bicor
   when many genes have MAD = 0 (check with `CheckBicorMad`): R falls back to
   Pearson for these genes, but py_hdWGCNA replaces MAD by 1, which is neither
   bicor nor Pearson. A small MAD is also a problem when the network has many
   groups: bicor gives weight 0 to the high values of group-specific genes.

3. Network type. 'signed' (default) keeps only positive correlations as
   strong links, so a module has genes going in the same direction.
   'unsigned' also links genes with opposite patterns. Choose the type in
   TestWgcnaSoftPowers; BuildWgcnaNetwork reuses it, because the power is
   valid only for that type.

4. min_module_size. The hdWGCNA default is 50. Smaller values (e.g. 20) keep
   small modules. Modules below 50 genes exist only because of that choice;
   list them with `FlagSmallWgcnaModules`.

5. Harmony on eigengenes. Use it only for technical factors that are NOT
   nested in the condition (e.g. a sequencing batch with samples of all
   conditions inside). If each sample belongs to one condition only,
   Harmony by sample also removes part of the difference between
   conditions. Default here: harmonize=False.

6. Enrichment background. The module genes come from the network genes, so
   the right universe is the network genes (background="network"), not all
   genes of the Enrichr library.

7. Statistics. Cells and metacells are not biological replicates. To compare
   conditions, summarize eigengenes per sample (inside each group) before
   testing.
   
8. Reproducibility. py_hdWGCNA 0.1.0 builds metacells with an unseeded random
   generator, so results change a little between runs (see BuildWgcnaMetacells).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

__all__ = [
    "GetWgcnaPackageVersions",
    "SummarizeWgcnaInput",
    "PrepareWgcnaInput",
    "GetCommonWgcnaGenes",
    "SelectWgcnaGenes",
    "BuildWgcnaMetacells",
    "SummarizeWgcnaMetacells",
    "SetWgcnaExpression",
    "CheckBicorMad",
    "TestWgcnaSoftPowers",
    "SelectWgcnaSoftPower",
    "PlotWgcnaSoftPowers",
    "BuildWgcnaNetwork",
    "FlagSmallWgcnaModules",
    "PlotWgcnaDendrogram",
    "ComputeWgcnaEigengenes",
    "PlotWgcnaModuleFeatures",
    "ComputeWgcnaKme",
    "PlotWgcnaKmes",
    "GetWgcnaHubGenes",
    "RunWgcnaEnrichr",
    "SelectSignificantEnrichment",
    "PlotWgcnaEnrichmentDotplot",
    "ExportWgcnaToCytoscape",
    "EvaluateWgcnaCytoscapeFilters",
    "PrepareWgcnaPreservationInputs",
    "RunWgcnaModulePreservation"
]



# ── Helpers ───────────────────────────────────────────────────────────────────

def _GetWgcnaData(adata, wgcna_name):
    """Return the py_hdWGCNA results dict for `wgcna_name`, or raise a clear error."""
    try:
        return adata.uns['hdWGCNA'][wgcna_name]
    except KeyError:
        raise KeyError(
            f"No hdWGCNA results named {wgcna_name!r} in adata.uns['hdWGCNA']. "
            "Run SelectWgcnaGenes() first."
        ) from None


def _RecordParams(adata, wgcna_name, step, params):
    """Save the arguments of one step under adata.uns['hdWGCNA'][wgcna_name]['sctools_params']."""
    wd = _GetWgcnaData(adata, wgcna_name)
    wd.setdefault('sctools_params', {})[step] = params


def _ModuleSortKey(name):
    """Sort module names by number (M1, M2, ..., M10), other names at the end."""
    name = str(name)
    if name[:1] == 'M' and name[1:].isdigit():
        return (0, int(name[1:]), name)
    return (1, 0, name)


def GetWgcnaPackageVersions():
    """Versions of the packages used by this module, to record with the results.

    py_hdWGCNA changes fast and some defaults differ from the R package, so
    it is good to print this at the top of a notebook.

    Returns
    -------
    pd.Series (package -> version). 'not installed' if a package is missing.
    """
    from importlib import metadata

    versions = {}
    for package in ('py-hdwgcna', 'gseapy', 'scanpy', 'anndata'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return pd.Series(versions, name='version')


# ── Input ─────────────────────────────────────────────────────────────────────

def SummarizeWgcnaInput(adata, celltype_col, sample_col, condition_col, group_order=None):
    """Count cells per group x sample and list the condition of each sample.

    Use it before building metacells: group x sample combinations with fewer
    cells than `min_cells` (see `BuildWgcnaMetacells`) get no metacells.

    Parameters
    ----------
    adata : AnnData with cell metadata in `adata.obs`.
    celltype_col : column with the cell groups: cell type, cluster or stage
        (e.g. 'celltype').
    sample_col : column with the biological sample (e.g. 'sample').
    condition_col : column with the experimental condition (e.g. 'condition').
    group_order : optional list with the row order of the table (the order
        of the groups, e.g. a differentiation order). Groups not in the list
        are dropped from the table.

    Returns
    -------
    cells_per_group : DataFrame, groups (rows) x samples (columns), number of cells.
    sample_conditions : DataFrame with one row per sample and its condition.
    """
    cells_per_group = pd.crosstab(adata.obs[celltype_col], adata.obs[sample_col])
    if group_order is not None:
        cells_per_group = cells_per_group.reindex(group_order)

    sample_conditions = (
        adata.obs[[sample_col, condition_col]]
        .drop_duplicates()
        .sort_values(sample_col)
        .reset_index(drop=True)
    )
    return cells_per_group, sample_conditions


def PrepareWgcnaInput(
    adata,
    counts_layer,
    required_cols,
    reduction='X_pca',
    group_col=None,
    exclude_groups=None,
):
    """Check the input AnnData and put the chosen counts in layers['counts'].

    `py_hdWGCNA` always reads `adata.layers['counts']` (gene filter and
    metacells). This function copies `counts_layer` there, so you can choose
    which counts the network uses. The copy changes only the object in
    memory, not the .h5ad file. If you have counts corrected for ambient RNA
    (e.g. a SoupX layer), use them instead of raw counts (see Design note 1
    in the module docstring).

    Parameters
    ----------
    adata : AnnData.
    counts_layer : layer with untransformed integer counts (e.g. a
        SoupX-corrected layer, or 'counts').
    required_cols : obs columns that must exist and have no missing values
        (e.g. [sample_col, condition_col, celltype_col]).
    reduction : obsm key used later to find neighbor cells for metacells.
    group_col, exclude_groups : optional. Remove cells whose `group_col`
        value is in `exclude_groups` (e.g. group_col='celltype',
        exclude_groups=['RareType'], for a group with too few cells to get
        metacells). Do it here, before `SelectWgcnaGenes`: hdWGCNA does not
        support subsetting the object after the setup.

    Returns
    -------
    AnnData. A new object if cells were removed, otherwise the same object.
    """
    # Stop early if metadata, counts layer or embedding is missing.
    missing_cols = [col for col in required_cols if col not in adata.obs]
    if missing_cols:
        raise KeyError(f'Missing metadata columns: {missing_cols}')
    if counts_layer not in adata.layers:
        raise KeyError(f'Counts layer {counts_layer!r} not found. Available: {list(adata.layers)}')
    if reduction not in adata.obsm:
        raise KeyError(f'Reduction {reduction!r} not found. Available: {list(adata.obsm)}')
    # Duplicated names would mix genes or cells during aggregation.
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError('Gene and cell names must be unique.')
    # Cells with missing labels cannot be assigned to a metacell group.
    if adata.obs[required_cols].isna().any().any():
        raise ValueError(f'Columns {required_cols} cannot contain missing values.')

    # Counts cannot be NaN, inf or negative.
    # Only the stored nonzero values are checked for sparse matrices.
    counts = adata.layers[counts_layer]
    values = counts.data if sparse.issparse(counts) else np.asarray(counts).ravel()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite and nonnegative.')

    # Remove groups that should not enter the analysis (e.g. a rare cell type).
    if exclude_groups:
        if group_col is None or group_col not in adata.obs:
            raise KeyError('exclude_groups needs a valid group_col.')
        keep = ~adata.obs[group_col].isin(exclude_groups)
        adata = adata[keep].copy()

    # py_hdWGCNA always reads layers['counts'], so put the chosen counts there.
    if counts_layer != 'counts':
        if 'counts' in adata.layers:
            logger.warning(
                "layers['counts'] is replaced by layers[%r] (in memory only).", counts_layer
            )
        adata.layers['counts'] = adata.layers[counts_layer].copy()

    return adata


# ── Genes and metacells ───────────────────────────────────────────────────────

def GetCommonWgcnaGenes(adatas, fraction=0.05):
    """Genes detected in at least `fraction` of the cells in EVERY object.

    Same rule as `SelectWgcnaGenes` (counts > 0 in `layers['counts']`), but
    applied to each object separately (e.g. one object per condition). Pass
    the result as `genes_use` to `SelectWgcnaGenes`, so independent networks
    start from the same genes in the same order.

    Parameters
    ----------
    adatas : dict {name: AnnData}, each prepared with `PrepareWgcnaInput`.
    fraction : minimum fraction of cells with counts > 0.

    Returns
    -------
    genes : list of common genes, in the gene order of the first object.
    n_genes : pd.Series, genes passing the filter in each object and 'common'.
    """
    detected = {}
    for name, adata in adatas.items():
        if 'counts' not in adata.layers:
            raise KeyError(f"{name!r} has no layers['counts']. Run PrepareWgcnaInput() first.")
        # Same rule as py_hdWGCNA setup_for_wgcna(gene_select='fraction').
        frac = np.asarray((adata.layers['counts'] > 0).sum(axis=0)).ravel() / adata.n_obs
        detected[name] = set(adata.var_names[frac >= fraction])

    common = set.intersection(*detected.values())
    genes = [g for g in next(iter(adatas.values())).var_names if g in common]
    n_genes = pd.Series(
        {**{name: len(g) for name, g in detected.items()}, 'common': len(genes)},
        name='n_genes',
    )
    return genes, n_genes


def SelectWgcnaGenes(adata, fraction=0.05, wgcna_name='hdWGCNA', genes_use=None):
    """Keep genes detected in at least `fraction` of the cells (hdWGCNA tutorial: 0.05).

    The fraction is computed by py_hdWGCNA on `layers['counts']` (see
    `PrepareWgcnaInput`), over all cells together. Do not pre-filter genes
    by differential expression: WGCNA expects the full set of expressed genes.

    genes_use : optional list of genes used instead of the fraction filter,
        e.g. from `GetCommonWgcnaGenes` to compare independent networks. The
        genes must be in this object, in its gene order, because the network
        follows `adata.var_names`. `fraction` is then not applied again, only
        recorded: pass the value used to build the list.

    The final network genes are set later by `SetWgcnaExpression`, which
    removes genes with zero variance in the selected metacells.

    Returns
    -------
    AnnData with the results dict created in adata.uns['hdWGCNA'][wgcna_name].
    """
    from py_hdWGCNA.core import setup_for_wgcna

    if genes_use is None:
        adata = setup_for_wgcna(
            adata,
            gene_select='fraction',
            fraction=fraction,
            wgcna_name=wgcna_name,
        )
    else:
        genes_use = list(genes_use)
        # py_hdWGCNA silently drops missing genes and uses ALL genes if the list is empty.
        requested = set(genes_use)
        if not genes_use or [g for g in adata.var_names if g in requested] != genes_use:
            raise ValueError(
                'genes_use must be a non-empty list of unique genes of this object, '
                'in the order of adata.var_names.'
            )
        adata = setup_for_wgcna(
            adata,
            gene_select='custom',
            genes_use=genes_use,
            wgcna_name=wgcna_name,
        )

    # 'genes_use' is the list read by the next steps.
    genes_kept = list(_GetWgcnaData(adata, wgcna_name)['genes_use'])
    _RecordParams(adata, wgcna_name, 'SelectWgcnaGenes', {
        'fraction': fraction,
        'gene_select': 'fraction' if genes_use is None else 'custom',
        'n_genes': len(genes_kept),
        'genes_use': genes_kept,
    })
    return adata


def BuildWgcnaMetacells(
    adata,
    group_by,
    ident_group,
    k=25,
    max_shared=10,
    min_cells=100,
    target_metacells=1000,
    max_iter=5000,
    reduction='X_pca',
    wgcna_name='hdWGCNA',
):
    """Build metacells inside each group (e.g. cell type x sample).

    A metacell is the average of `k` similar cells (neighbors in
    `reduction`). It reduces sparsity, so gene correlations are less noisy.
    Group by sample so a metacell never mixes cells of different samples.
    py_hdWGCNA normalizes the metacells (counts per 10,000, then log1p) and
    stores them in adata.uns['hdWGCNA'][wgcna_name]['metacell_obj'].

    Parameters
    ----------
    group_by : list of obs columns, e.g. [celltype_col, sample_col].
    ident_group : obs column used as identity of the metacells (e.g. celltype_col).
    k : cells per metacell (hdWGCNA tutorial: 25).
    max_shared : max cells two metacells can share. Must be smaller than k,
        otherwise it does not limit the overlap (tutorial: 10).
    min_cells : groups with fewer cells get no metacells (hdWGCNA default: 100).
        Check `SummarizeWgcnaInput` first: a combination just below the
        cutoff (e.g. 95 cells) removes that sample from that group.
    target_metacells : max number of metacells per group.
    max_iter : max iterations when sampling metacells.
    reduction : obsm key with the embedding used to find neighbors.

    Notes
    -----
    Metacells are not new biological replicates. With many metacells per
    group, each cell is used in several metacells.

    Not reproducible: py_hdWGCNA 0.1.0 samples the metacell centers with
    `np.random.default_rng()` without a seed, and has no parameter to set it.
    Each run gives slightly different metacells (e.g. 3,873 vs 3,860 in two
    runs of the tutorial dataset), so the power table and the modules can
    change a little between runs.
    """
    from py_hdWGCNA import metacells_by_groups

    if not 0 <= max_shared < k:
        raise ValueError('Require 0 <= max_shared < k.')

    adata = metacells_by_groups(
        adata,
        group_by=group_by,
        k=k,
        max_shared=max_shared,
        ident_group=ident_group,
        min_cells=min_cells,
        target_metacells=target_metacells,
        max_iter=max_iter,
        reduction=reduction,
        wgcna_name=wgcna_name,
    )
    _RecordParams(adata, wgcna_name, 'BuildWgcnaMetacells', {
        'group_by': list(group_by), 'ident_group': ident_group, 'k': k,
        'max_shared': max_shared, 'min_cells': min_cells,
        'target_metacells': target_metacells, 'reduction': reduction,
    })
    return adata


def SummarizeWgcnaMetacells(adata, group_by, wgcna_name='hdWGCNA'):
    """Table with the number of cells and metacells per group.

    Groups with 0 metacells were skipped by `min_cells`.

    Parameters
    ----------
    group_by : the same list of obs columns used in `BuildWgcnaMetacells`.

    Returns
    -------
    DataFrame indexed by the group columns, with 'cells' and 'metacells'.
    """
    mc = _GetWgcnaData(adata, wgcna_name)['metacell_obj']
    cell_counts = adata.obs.groupby(group_by, observed=True).size().rename('cells')
    mc_counts = mc.obs.groupby(group_by, observed=True).size().rename('metacells')
    summary = pd.concat([cell_counts, mc_counts], axis=1).fillna(0).astype(int)
    return summary


def SetWgcnaExpression(adata, groups, group_by, wgcna_name='hdWGCNA'):
    """Select the metacells used to build the network.

    The network in this module is always built on metacells (see
    `BuildWgcnaMetacells`), so there is no option for single cells. It uses
    the normalized metacell expression (layer 'X' of the metacell object),
    not a layer of the original AnnData.

    Parameters
    ----------
    groups : values of `group_by` to include (e.g. a list of cell types).
        One group gives a network inside that group. Many groups give a joint
        network, where modules can also follow differences between groups
        (e.g. between cell types, or along a differentiation trajectory),
        not only co-expression inside a group.
    group_by : obs column of the metacells (e.g. celltype_col).

    Returns
    -------
    AnnData with 'dat_expr' (genes x metacells) in the results dict.
    """
    from py_hdWGCNA.core import set_dat_expr

    mc = _GetWgcnaData(adata, wgcna_name)['metacell_obj']
    available = set(mc.obs[group_by].astype(str))
    groups_use = [g for g in groups if g in available]
    missing = [g for g in groups if g not in available]
    if missing:
        logger.warning('Groups without metacells, not used: %s', missing)
    if not groups_use:
        raise ValueError('No requested group has metacells.')

    # set_dat_expr uses the metacells by default (use_metacells=True).
    adata = set_dat_expr(
        adata,
        group_name=groups_use,
        group_by=group_by,
        layer='X',  # normalized metacell expression
        wgcna_name=wgcna_name,
    )
    _RecordParams(adata, wgcna_name, 'SetWgcnaExpression', {
        'groups': groups_use, 'group_by': group_by,
    })
    return adata


# ── Correlation method and soft power ─────────────────────────────────────────

def CheckBicorMad(adata, wgcna_name='hdWGCNA'):
    """Check if bicor is safe for the network matrix.

    bicor divides by the MAD (median absolute deviation) of each gene.
    MAD = 0 happens when more than half of the metacells have the same
    value (usually 0). R falls back to Pearson for these genes, but
    py_hdWGCNA replaces MAD by 1, so the result is neither bicor nor
    Pearson. A small median MAD is also a warning sign: bicor gives weight 0
    to values far from the median, which removes the signal of genes
    expressed in only some groups.

    There is no fixed cutoff: bicor makes sense when almost no gene has
    MAD = 0. For reference, in the tutorial dataset 19% of the genes had
    MAD = 0 and the median MAD was 0.090, so it used cor_method='pearson'.

    Returns
    -------
    pd.Series with n_genes, n_mad_zero, frac_mad_zero and median_mad.
    """
    expr = np.asarray(_GetWgcnaData(adata, wgcna_name)['dat_expr'])  # genes x metacells
    gene_medians = np.median(expr, axis=1, keepdims=True)
    gene_mad = np.median(np.abs(expr - gene_medians), axis=1)
    n_genes = expr.shape[0]
    n_mad_zero = int((gene_mad <= 1e-15).sum())
    return pd.Series({
        'n_genes': n_genes,
        'n_mad_zero': n_mad_zero,
        'frac_mad_zero': n_mad_zero / n_genes,
        'median_mad': float(np.median(gene_mad)),
    })


def TestWgcnaSoftPowers(
    adata,
    cor_method='pearson',
    network_type='signed',
    power_range=None,
    wgcna_name='hdWGCNA',
):
    """Compute the scale-free fit and connectivity for each soft power.

    The method chosen here is saved by py_hdWGCNA and reused by
    `BuildWgcnaNetwork`, so the power is chosen for the same correlation
    used in the network (see Design note 2 in the module docstring).

    Parameters
    ----------
    cor_method : 'pearson' or 'bicor'. Check `CheckBicorMad` before using bicor.
    network_type : 'signed' (hdWGCNA default), 'unsigned' or 'signed hybrid'.
        See `BuildWgcnaNetwork` for the differences. The network reuses this
        type, so choose it here. Signed networks usually need a higher power
        than unsigned ones.
    power_range : list of powers. None uses 1-10, then 12-30 in steps of 2.

    Returns
    -------
    adata, power_table (one row per power: SFT.R.sq, slope, mean.k., ...).
    In py_hdWGCNA, SFT.R.sq has no sign; check the slope too.
    """
    from py_hdWGCNA.network import test_soft_powers

    # The package changes this cache in place, so an old copy is not a plain
    # correlation anymore. Drop it before and after the test.
    wd = _GetWgcnaData(adata, wgcna_name)
    wd.pop('cor_matrix', None)

    adata = test_soft_powers(
        adata,
        power_range=power_range,
        network_type=network_type,
        cor_method=cor_method,
        wgcna_name=wgcna_name,
    )

    wd = _GetWgcnaData(adata, wgcna_name)
    wd.pop('cor_matrix', None)
    power_table = wd['power_table'].sort_values('Power').reset_index(drop=True)
    _RecordParams(adata, wgcna_name, 'TestWgcnaSoftPowers', {
        'cor_method': cor_method, 'network_type': network_type,
    })
    return adata, power_table


def SelectWgcnaSoftPower(adata, r2_cutoff=0.80, wgcna_name='hdWGCNA'):
    """Return the lowest tested power with SFT.R.sq >= r2_cutoff and slope < 0.

    0.80 is the hdWGCNA tutorial cutoff. The negative slope is checked
    because py_hdWGCNA stores R2 without sign: a good fit with a positive
    slope (more highly connected genes than weakly connected ones) is not
    scale-free. The package also prints its own choice; this one replaces it.

    Raises
    ------
    ValueError if no power passes. Then look at the power plots before
    choosing a power by hand.
    """
    wd = _GetWgcnaData(adata, wgcna_name)
    power_table = wd['power_table'].sort_values('Power')
    valid = power_table.loc[
        (power_table['SFT.R.sq'] >= r2_cutoff) & (power_table['slope'] < 0)
    ]
    if valid.empty:
        raise ValueError(
            'No tested power meets both criteria. '
            'Inspect the soft-power plots before constructing the network.'
        )
    selected_power = int(valid.iloc[0]['Power'])
    wd['selected_power'] = selected_power
    _RecordParams(adata, wgcna_name, 'SelectWgcnaSoftPower', {'r2_cutoff': r2_cutoff})
    return selected_power


def PlotWgcnaSoftPowers(adata, selected_power=None, wgcna_name='hdWGCNA'):
    """Plot the scale-free fit and the connectivity for each tested power.

    Wrapper of `py_hdWGCNA.plot_soft_powers`. Four panels: scale-free fit
    (R2, grey area below 0.80), mean, median and maximum connectivity. The
    slope is not shown; check it in the power table.

    Parameters
    ----------
    selected_power : power to highlight, usually from `SelectWgcnaSoftPower`.

    Returns
    -------
    matplotlib Figure. Call plt.show() to display it.
    """
    from py_hdWGCNA import plot_soft_powers

    return plot_soft_powers(adata, selected_power=selected_power, wgcna_name=wgcna_name)


# ── Network and modules ───────────────────────────────────────────────────────

def BuildWgcnaNetwork(
    adata,
    power,
    network_type=None,
    tom_type='signed',
    min_module_size=50,
    deep_split=4,
    pam_stage=False,
    merge_cut_height=0.2,
    tom_name='hdWGCNA_TOM',
    wgcna_name='hdWGCNA',
):
    """Build the network, the TOM and the modules.

    Steps done by py_hdWGCNA: correlation (method from
    `TestWgcnaSoftPowers`), soft power, topological overlap (TOM),
    hierarchical clustering on 1 - TOM, dynamic tree cut, then merging of
    modules with similar eigengenes. Genes not assigned to a module are 'grey'.

    Parameters
    ----------
    power : soft power, usually from `SelectWgcnaSoftPower`.
    network_type : how a correlation becomes a link weight (adjacency):
        'signed'        : ((1 + cor) / 2) ** power. Only positive correlation
                          gives a strong link; negative correlation gives a
                          weight close to 0. A module has genes that go up and
                          down together. hdWGCNA default.
        'unsigned'      : abs(cor) ** power. Positive and negative correlation
                          give the same weight, so a module can mix genes with
                          opposite patterns.
        'signed hybrid' : cor ** power if cor > 0, else 0.
        None (default) uses the type of `TestWgcnaSoftPowers`, because the
        power was chosen for that type. Passing a different type raises an
        error: test the powers again with the new type first.
    tom_type : 'signed' (hdWGCNA default) or 'unsigned'. How the topological
        overlap treats the sign of the links. Keep 'signed' with signed networks.
    min_module_size : smallest module in the tree cut. hdWGCNA default: 50.
        Smaller values keep small modules; check them with `FlagSmallWgcnaModules`.
    deep_split : 0-4, higher splits branches more. It is not the number of modules.
    pam_stage : False leaves unclear genes in 'grey' (hdWGCNA default).
    merge_cut_height : merge modules whose eigengenes have
        1 - correlation below this value (0.2 means correlation >= 0.8).

    Returns
    -------
    adata, module_sizes (number of genes per module, 'grey' included).
    """
    from py_hdWGCNA.network import construct_network

    wd = _GetWgcnaData(adata, wgcna_name)

    # The power is valid only for the network type used in TestWgcnaSoftPowers.
    tested_type = wd.get('network_type')
    if network_type is None:
        if tested_type is None:
            raise ValueError('Run TestWgcnaSoftPowers first, or pass network_type.')
        network_type = tested_type
    elif tested_type is not None and network_type != tested_type:
        raise ValueError(
            f'network_type={network_type!r}, but the powers were tested with '
            f'{tested_type!r}. Run TestWgcnaSoftPowers with the same type first.'
        )

    # A cached tree from a previous network must not be reused.
    for key in ('linkage_matrix', 'linkage'):
        wd.pop(key, None)

    adata = construct_network(
        adata,
        power=power,
        network_type=network_type,
        tom_type=tom_type,
        minModuleSize=min_module_size,
        deepSplit=deep_split,
        pamStage=pam_stage,
        mergeCutHeight=merge_cut_height,
        tom_name=tom_name,
        wgcna_name=wgcna_name,
    )

    wd = _GetWgcnaData(adata, wgcna_name)
    module_sizes = wd['modules_df']['module'].value_counts().rename('n_genes')
    _RecordParams(adata, wgcna_name, 'BuildWgcnaNetwork', {
        'power': power, 'network_type': network_type, 'tom_type': tom_type,
        'min_module_size': min_module_size, 'deep_split': deep_split,
        'pam_stage': pam_stage, 'merge_cut_height': merge_cut_height,
    })
    return adata, module_sizes


def FlagSmallWgcnaModules(module_sizes, min_size=50):
    """Return the modules (not 'grey') with fewer than `min_size` genes.

    With min_module_size below the hdWGCNA default (50), these modules
    exist only because of that choice. Interpret them with care and, if
    possible, check them in a run with min_module_size=50.

    Returns
    -------
    pd.Series (module -> number of genes). Empty if all modules are large enough.
    """
    sizes = module_sizes.drop(index='grey', errors='ignore')
    return sizes[sizes < min_size]


def PlotWgcnaDendrogram(adata, wgcna_name='hdWGCNA'):
    """Plot the gene tree (1 - TOM) with a color bar of the modules.

    Wrapper of `py_hdWGCNA.plot_dendrogram`. Each leaf is a gene; the height
    is the TOM dissimilarity. 'grey' genes are not in any module.

    Returns
    -------
    matplotlib Figure. Call plt.show() to display it.
    """
    from py_hdWGCNA import plot_dendrogram

    return plot_dendrogram(adata, wgcna_name=wgcna_name)


# ── Eigengenes and kME ────────────────────────────────────────────────────────

def ComputeWgcnaEigengenes(
    adata,
    layer,
    harmonize=False,
    group_by_vars=None,
    wgcna_name='hdWGCNA',
):
    """Compute module eigengenes (MEs) in the single cells.

    The eigengene is the first principal component of the scaled
    expression of the module genes: one value per cell and module. Its sign
    is aligned with the mean expression of the module.

    When to use Harmony (harmonize=True):
        Only for a technical factor that is NOT nested in the condition,
        e.g. a sequencing batch that has cells of all conditions. Pass the
        factor in `group_by_vars`. If each sample belongs to one condition
        only, Harmony by sample also removes part of the difference between
        conditions: keep harmonize=False. Default here is False (also the R
        default, group.by.vars = NULL).

    Parameters
    ----------
    layer : log-normalized layer of the same counts used in the network
        (e.g. the log1p of the SoupX counts, if the network used them).
    harmonize : apply Harmony to the eigengenes.
    group_by_vars : obs column(s) for Harmony. Required if harmonize=True.

    Returns
    -------
    adata, eigengenes (DataFrame, cells x modules): 'hMEs' if harmonize=True,
    otherwise 'MEs'. With harmonize=False, py_hdWGCNA stores a copy of MEs in
    'hMEs', so later steps that read 'hMEs' use the uncorrected values.
    """
    from py_hdWGCNA.modules import module_eigengenes

    if harmonize and group_by_vars is None:
        raise ValueError('harmonize=True needs group_by_vars (a technical factor).')

    adata = module_eigengenes(
        adata,
        layer=layer,
        use_metacells=False,
        harmonize=harmonize,
        group_by_vars=group_by_vars,
        wgcna_name=wgcna_name,
    )
    wd = _GetWgcnaData(adata, wgcna_name)
    eigengenes = wd['hMEs'] if harmonize else wd['MEs']
    _RecordParams(adata, wgcna_name, 'ComputeWgcnaEigengenes', {
        'layer': layer, 'harmonize': harmonize, 'group_by_vars': group_by_vars,
    })
    return adata, eigengenes


def PlotWgcnaModuleFeatures(
    adata,
    features='MEs',
    reduction='X_umap',
    ncols=3,
    wgcna_name='hdWGCNA',
):
    """Plot the eigengene of each module on an embedding (one panel per module).

    Wrapper of `py_hdWGCNA.module_feature_plot`.

    Parameters
    ----------
    features : 'MEs' (default, no Harmony) or 'hMEs' (harmonized). Use the
        same kind computed by `ComputeWgcnaEigengenes`.
    reduction : obsm key of the embedding (e.g. 'X_umap'). Same style as
        `reduction` in `PrepareWgcnaInput` and `BuildWgcnaMetacells`.
    ncols : panels per row.

    Returns
    -------
    matplotlib Figure. Call plt.show() to display it.
    """
    from py_hdWGCNA import module_feature_plot

    if reduction not in adata.obsm:
        raise KeyError(f'Reduction {reduction!r} not found. Available: {list(adata.obsm)}')
    if not reduction.startswith('X_'):
        raise ValueError("py_hdWGCNA needs an obsm key that starts with 'X_' (e.g. 'X_umap').")

    # py_hdWGCNA expects the name without 'X_' and adds it back.
    return module_feature_plot(
        adata,
        reduction=reduction[2:],
        features=features,
        ncols=ncols,
        wgcna_name=wgcna_name,
    )


def ComputeWgcnaKme(
    adata,
    group_by,
    groups,
    layer,
    cor_method='pearson',
    sparse=True,
    wgcna_name='hdWGCNA',
):
    """Compute kME: correlation between each gene and each module eigengene.

    kME close to +1: the gene follows the module. Close to 0: no linear
    relation. Close to -1: opposite pattern. Genes with the highest kME in
    their own module are hub candidates. They describe the module well, but
    they are not proven regulators.

    Parameters
    ----------
    group_by, groups : cells used, e.g. the same groups used in the network.
    layer : same log-normalized layer used in `ComputeWgcnaEigengenes`.
    cor_method, sparse : how kME is computed.
        sparse=True  : works on the sparse matrix (fast, low memory). Always
                       Pearson, like the R default (corSparse). cor_method
                       must be 'pearson'.
        sparse=False : dense matrix (cells x genes in memory). Allows
                       cor_method='bicor' or 'spearman'.
        Use the same method as the network. With many zeros per gene in
        single cells, bicor has the same MAD = 0 problem as in the network
        (see `CheckBicorMad`).

    Returns
    -------
    adata, modules_df (gene, module, color and one kME_<module> column per module).
    Hub genes: GetWgcnaHubGenes(adata, n_hubs=10).
    """
    from py_hdWGCNA import module_connectivity

    # py_hdWGCNA ignores cor_method when sparse=True (always Pearson).
    if sparse and cor_method != 'pearson':
        raise ValueError(
            "sparse=True always uses Pearson. Use cor_method='pearson', "
            "or sparse=False to use another method."
        )

    # The package reads 'hMEs'. With harmonize=False, 'hMEs' is a copy of 'MEs'.
    adata = module_connectivity(
        adata,
        group_by=group_by,
        group_name=list(groups),
        layer=layer,
        use_metacells=False,
        cor_method=cor_method,
        sparse=sparse,
        wgcna_name=wgcna_name,
    )
    modules_df = _GetWgcnaData(adata, wgcna_name)['modules_df']
    _RecordParams(adata, wgcna_name, 'ComputeWgcnaKme', {
        'group_by': group_by, 'groups': list(groups), 'layer': layer,
        'cor_method': cor_method, 'sparse': sparse,
    })
    return adata, modules_df


def PlotWgcnaKmes(adata, n_hubs=10, ncols=3, text_size=8, wgcna_name='hdWGCNA'):
    """Plot the kME of the genes of each module, with the top `n_hubs` names.

    Wrapper of `py_hdWGCNA.plot_kmes`. Run `ComputeWgcnaKme` first.

    Returns
    -------
    matplotlib Figure. Call plt.show() to display it.
    """
    from py_hdWGCNA import plot_kmes

    return plot_kmes(adata, n_hubs=n_hubs, ncols=ncols, text_size=text_size, wgcna_name=wgcna_name)


def GetWgcnaHubGenes(adata, n_hubs=10, wgcna_name='hdWGCNA'):
    """Table with the top `n_hubs` genes of each module, ranked by kME.

    Wrapper of `py_hdWGCNA.modules.get_hub_genes`. Only genes assigned to the
    module are ranked. Hub genes describe the module well; they are not
    proven regulators.

    Returns
    -------
    DataFrame with columns gene_name, module, kME.
    """
    from py_hdWGCNA.modules import get_hub_genes

    return get_hub_genes(adata, n_hubs=n_hubs, wgcna_name=wgcna_name)


# ── Enrichment ────────────────────────────────────────────────────────────────

def RunWgcnaEnrichr(
    adata,
    databases,
    organism,
    n_genes=100,
    background='network',
    wgcna_name='hdWGCNA',
):
    """Over-representation analysis (Enrichr via gseapy) for each module.

    Uses the top `n_genes` genes of each module, ranked by kME (hdWGCNA
    tutorial: 100). Run `ComputeWgcnaKme` first: `BuildWgcnaNetwork` also
    stores a kME, computed in the metacells, and without this check it would
    be used silently. Needs internet access.

    Parameters
    ----------
    databases : Enrichr library names, e.g. ['GO_Biological_Process_2023',
        'MSigDB_Hallmark_2020']. List them with gseapy.get_library_name().
    organism : no default, choose it for your data: e.g. 'human', 'mouse',
        'fly', 'yeast', 'worm', 'fish'. In gseapy, 'human' and 'mouse' use
        the same Enrichr site and the same libraries (human gene symbols);
        mouse symbols are matched without case, and mouse-only genes are lost.
    n_genes : genes per module, ranked by kME.

    Background (universe) options:
        'network' : genes of the network (the genes that could enter a
            module). Use this when the modules come from a filtered gene set,
            as in this workflow. Otherwise general terms of the cell type
            look enriched only because all tested genes are expressed in it.
        None : Enrichr default (all genes of the library). Use only when the
            gene list comes from the whole genome.
        list of genes : any custom universe.

    Note: with a background, gseapy uses another Enrichr service
    (speedrichr) and the result has no 'Overlap' column. Count genes with
    the 'Genes' column (genes separated by ';').

    Returns
    -------
    DataFrame with Enrichr results and a 'module' column.
    Adjusted p-values are corrected inside each module and database.
    """
    import gseapy as gp
    from py_hdWGCNA.modules import get_hub_genes

    wd = _GetWgcnaData(adata, wgcna_name)

    # The module genes are ranked by kME. BuildWgcnaNetwork already stores a
    # kME computed in the metacells, so kME columns exist even if
    # ComputeWgcnaKme was not run. Check that the single-cell kME was computed.
    if 'ComputeWgcnaKme' not in wd.get('sctools_params', {}):
        raise ValueError(
            'Run ComputeWgcnaKme before RunWgcnaEnrichr. Otherwise the genes are '
            'ranked by the kME of the metacells, stored by BuildWgcnaNetwork.'
        )

    if isinstance(background, str) and background == 'network':
        background_genes = list(wd.get('dat_expr_genes', wd['genes_use']))
    elif background is None:
        background_genes = None
    else:
        background_genes = list(background)

    # Up to n_genes genes per module, ranked by kME of the module.
    module_genes = get_hub_genes(adata, n_hubs=n_genes, wgcna_name=wgcna_name)

    results = []
    for module, table in module_genes.groupby('module'):
        genes = table['gene_name'].dropna().drop_duplicates().tolist()
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=list(databases),
            organism=organism,
            background=background_genes,
            outdir=None,
        )
        result = enr.results.copy()
        result['module'] = module
        result['n_module_genes'] = len(genes)
        results.append(result)

    _RecordParams(adata, wgcna_name, 'RunWgcnaEnrichr', {
        'databases': list(databases), 'n_genes': n_genes, 'organism': organism,
        'background': 'network' if isinstance(background, str) else (
            None if background is None else f'custom ({len(background_genes)} genes)'),
    })
    return pd.concat(results, ignore_index=True)


def SelectSignificantEnrichment(enrichment_df, padj_cutoff=0.05):
    """Keep terms with 'Adjusted P-value' < padj_cutoff, sorted by module, database, p-value.

    The cutoff is exploratory: p-values are adjusted inside each module and
    database, not across all of them.
    """
    return (
        enrichment_df
        .loc[enrichment_df['Adjusted P-value'] < padj_cutoff]
        .sort_values(['module', 'Gene_set', 'Adjusted P-value'])
    )


def PlotWgcnaEnrichmentDotplot(
    enrichment_df,
    database,
    top_n=5,
    padj_cutoff=0.05,
    title=None,
    term_width=55,
):
    """Dot plot of the top significant terms of one database, per module.

    Each column is a module. The terms are the `top_n` best terms of each
    module, shown in every module where they are significant. Color is
    -log10(adjusted p-value); size is the number of module genes in the term.

    Parameters
    ----------
    enrichment_df : output of `RunWgcnaEnrichr`.
    database : one value of the 'Gene_set' column, e.g. 'GO_Biological_Process_2023'.
    top_n : terms per module.
    padj_cutoff : significance cutoff for 'Adjusted P-value'.
    term_width : wrap long term names at this width.

    Returns
    -------
    fig, ax. (None, None) if no term passes the cutoff.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from textwrap import fill

    # Keep significant results from one database.
    df = enrichment_df.loc[
        (enrichment_df['Gene_set'] == database)
        & (enrichment_df['Adjusted P-value'] < padj_cutoff)
    ].copy()
    if df.empty:
        logger.warning('No significant terms in %s.', database)
        return None, None

    # Select the top terms per module.
    top = df.sort_values('Adjusted P-value').groupby('module', sort=False).head(top_n)

    # Show these terms wherever they are significant across modules.
    terms = top.groupby('Term')['Adjusted P-value'].min().sort_values().index.tolist()
    df = df.loc[df['Term'].isin(terms)].copy()

    # Number of module genes in the term. Count from 'Genes', because the
    # result has no 'Overlap' column when a background is used.
    df['n_genes'] = df['Genes'].str.split(';').str.len()
    df['significance'] = -np.log10(df['Adjusted P-value'].clip(lower=np.finfo(float).tiny))

    modules = sorted(enrichment_df['module'].unique(), key=_ModuleSortKey)
    df['module'] = pd.Categorical(df['module'], categories=modules, ordered=True)
    df['Term'] = pd.Categorical(df['Term'], categories=terms, ordered=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(terms))))
    sns.scatterplot(
        data=df,
        x='module',
        y='Term',
        hue='significance',
        size='n_genes',
        palette='viridis',
        sizes=(40, 300),
        edgecolor='black',
        linewidth=0.4,
        legend='brief',
        ax=ax,
    )
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([fill(term, width=term_width) for term in terms])
    ax.set(xlabel='Module', ylabel='', title=title or f'{database} enrichment')
    ax.grid(axis='both', alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig, ax



# -------
def ExportWgcnaToCytoscape(
    adata,
    output_dir,
    *,
    condition_col='condition',
    prefix='hdWGCNA',
    n_hubs=None,
    min_tom=0.0,
    top_connections=None,
    edge_scope='all',
    include_module_nodes=True,
    wgcna_name='hdWGCNA',
):
    """Export stored hdWGCNA results to Cytoscape TSV tables.

    adata : AnnData for one condition; results must include modules and kME.
    output_dir : destination folder; existing files with the same names are replaced.
    condition_col : obs column containing exactly one nonmissing condition.
    prefix : filename prefix; the condition is appended automatically.
    n_hubs : None or 0 exports every assigned gene; positive integers select
        the top N per module using stored own-module kME. No kME is recomputed.
    min_tom : export gene pairs with TOM > min_tom (0 keeps all positive weights).
        This filters edges only, retains all selected nodes, and is not a p-value.
    top_connections : None keeps every eligible pair. A positive integer keeps
        each selected gene's strongest K eligible TOM connections. An undirected
        edge survives if either endpoint selects it, so degree may exceed K.
        Ties are broken by partner gene name. Ranking uses only selected genes
        within edge_scope and above min_tom, not all original network genes.
    edge_scope : 'all' includes between-module pairs; 'intramodular' restricts
        candidate edges to genes in the same module. Modules will then have no
        between-module edges by construction, not as evidence of independence.
    include_module_nodes : include module nodes and a separate membership edge file.
    wgcna_name : name of the stored hdWGCNA analysis.

    Returns a dict containing the exported DataFrames under 'nodes',
    'gene_edges' and 'membership_edges', plus exported counts. Membership is
    None when include_module_nodes=False. Records arguments and counts in
    sctools_params. Gene IDs remain their gene names; module IDs are their
    module names. Import independent conditions into separate Cytoscape
    collections. Gene edges are undirected, once per pair, without self-loops.
    Sparsified edge counts and degree are visualization properties. Interpret
    connectivity using full-network metrics, not the imposed top-K graph.
    The returned edge table resides in memory. The function does not copy the
    full TOM or recalculate the network, eigengenes or kME, and does not reread
    the TSV files to build the returned DataFrames.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd

    wd = adata.uns['hdWGCNA'][wgcna_name]
    labels = adata.obs[condition_col]
    if labels.isna().any() or labels.nunique() != 1:
        raise ValueError('adata must contain exactly one nonmissing condition.')
    condition = str(labels.iloc[0])

    if n_hubs is not None and (
        isinstance(n_hubs, (bool, np.bool_))
        or not isinstance(n_hubs, (int, np.integer))
        or n_hubs < 0
    ):
        raise ValueError('n_hubs must be None or a nonnegative integer.')
    if not np.isfinite(min_tom) or not 0 <= min_tom <= 1:
        raise ValueError('min_tom must be between 0 and 1.')
    if top_connections is not None and (
        isinstance(top_connections, (bool, np.bool_))
        or not isinstance(top_connections, (int, np.integer))
        or top_connections < 1
    ):
        raise ValueError('top_connections must be None or a positive integer.')
    if edge_scope not in ('all', 'intramodular'):
        raise ValueError("edge_scope must be 'all' or 'intramodular'.")

    # All inputs come from the object's stored results, not notebook variables.
    genes = wd['modules_df'].loc[
        wd['modules_df']['module'].ne('grey')
    ].copy().reset_index(drop=True)
    if genes.empty:
        raise ValueError('No genes assigned to non-grey modules.')

    # Read each gene's association with its own module.
    genes['kME'] = np.nan
    for module in genes['module'].unique():
        column = f'kME_{module}'
        if column not in genes:
            raise ValueError(f'{column} is missing. Run ComputeWgcnaKme first.')
        mask = genes['module'].eq(module)
        genes.loc[mask, 'kME'] = genes.loc[mask, column]

    if n_hubs:
        genes = (
            genes.sort_values(
                ['module', 'kME', 'gene_name'], ascending=[True, False, True]
            )
            .groupby('module', observed=True, sort=False)
            .head(n_hubs)
        )
    genes = genes.reset_index(drop=True)
    names = genes['gene_name'].astype(str).to_numpy()
    module_names = genes['module'].astype(str).to_numpy()

    # Matrix axes follow dat_expr_genes, not necessarily modules_df row order.
    gene_order = pd.Index(wd['dat_expr_genes'])
    positions = gene_order.get_indexer(genes['gene_name'])
    if (positions < 0).any():
        raise ValueError('An exported gene is absent from dat_expr_genes.')

    matrix_key = 'TOM' if 'TOM' in wd else 'tom_dissim'
    matrix = np.asarray(wd[matrix_key])
    if matrix.shape != (len(gene_order), len(gene_order)):
        raise ValueError('TOM dimensions do not match dat_expr_genes.')
    if not pd.Index(names).is_unique:
        raise ValueError('Exported gene IDs must be unique.')
    if include_module_nodes and set(names).intersection(module_names):
        raise ValueError('Gene/module IDs collide. Use include_module_nodes=False.')

    nodes = pd.DataFrame({
        'id': names,
        'label': names,
        'node_type': 'gene',
        'condition': condition,
        'module': module_names,
        'color': genes['color'].to_numpy(),
        'kME': genes['kME'].to_numpy(),
    })
    n_modules = int(genes['module'].nunique())
    if include_module_nodes:
        module_nodes = (
            genes[['module', 'color']].drop_duplicates('module')
            .assign(
                id=lambda x: x['module'].astype(str),
                label=lambda x: x['module'].astype(str),
                node_type='module', condition=condition,
            )
        )
        nodes = pd.concat([nodes, module_nodes], ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'{prefix}_{condition}'
    files = {
        'nodes': output_dir / f'{stem}_nodes.tsv',
        'gene_edges': output_dir / f'{stem}_gene_edges.tsv',
    }
    nodes.to_csv(files['nodes'], sep='\t', index=False)

    if top_connections is not None:
        # Store only the union of top-K pairs, not all eligible gene pairs.
        n = len(genes)
        pairs = set()
        all_indices = np.arange(n)
        for i, position in enumerate(positions):
            candidates = (
                np.flatnonzero(module_names == module_names[i])
                if edge_scope == 'intramodular' else all_indices
            )
            weights = matrix[position, positions[candidates]]
            if matrix_key == 'tom_dissim':
                weights = 1.0 - weights
            keep = np.isfinite(weights) & (weights > min_tom) & (candidates != i)
            partners = candidates[keep]
            ranked = np.lexsort((names[partners], -weights[keep]))
            for j in partners[ranked[:top_connections]]:
                pairs.add(min(i, int(j)) * n + max(i, int(j)))
        codes = np.array(sorted(pairs), dtype=np.int64)
        left, right = codes // n, codes % n
        weights = matrix[positions[left], positions[right]]
        if matrix_key == 'tom_dissim':
            weights = 1.0 - weights
        gene_edges = pd.DataFrame({
            'source': names[left],
            'target': names[right],
            'interaction': 'topological_overlap',
            'TOM': weights,
            'condition': condition,
        })
    else:
        # Extract the upper triangle without copying the full TOM submatrix.
        gene_edge_parts = []
        for i, position in enumerate(positions[:-1]):
            weights = matrix[position, positions[i + 1:]]
            if matrix_key == 'tom_dissim':
                weights = 1.0 - weights
            keep = np.isfinite(weights) & (weights > min_tom)
            if edge_scope == 'intramodular':
                keep &= module_names[i + 1:] == module_names[i]
            if not keep.any():
                continue
            gene_edge_parts.append(pd.DataFrame({
                'source': names[i],
                'target': names[i + 1:][keep],
                'interaction': 'topological_overlap',
                'TOM': weights[keep],
                'condition': condition,
            }))
        gene_edges = (
            pd.concat(gene_edge_parts, ignore_index=True)
            if gene_edge_parts else pd.DataFrame({
                'source': pd.Series(dtype=str),
                'target': pd.Series(dtype=str),
                'interaction': pd.Series(dtype=str),
                'TOM': pd.Series(dtype=float),
                'condition': pd.Series(dtype=str),
            })
        )
        del gene_edge_parts
    gene_edges.to_csv(files['gene_edges'], sep='\t', index=False)
    n_gene_edges = len(gene_edges)
    n_connected = len(set(gene_edges['source']).union(gene_edges['target']))

    membership = None
    n_membership_edges = 0
    if include_module_nodes:
        files['membership_edges'] = output_dir / f'{stem}_membership_edges.tsv'
        membership = pd.DataFrame({
            'source': names,
            'target': module_names,
            'interaction': 'module_membership',
            'kME': genes['kME'].to_numpy(),
            'condition': condition,
        })
        membership.to_csv(files['membership_edges'], sep='\t', index=False)
        n_membership_edges = len(membership)

    summary = {
        'condition': condition,
        'n_genes': len(genes),
        'n_modules': n_modules,
        'n_gene_edges': n_gene_edges,
        'n_connected_genes': n_connected,
        'n_isolated_genes': len(genes) - n_connected,
        'n_membership_edges': n_membership_edges,
    }
    wd.setdefault('sctools_params', {})['ExportWgcnaToCytoscape'] = {
        'condition_col': condition_col,
        'prefix': str(prefix),
        'n_hubs': None if n_hubs is None else int(n_hubs),
        'min_tom': float(min_tom),
        'top_connections': None if top_connections is None else int(top_connections),
        'edge_scope': edge_scope,
        'include_module_nodes': bool(include_module_nodes),
        'output_dir': str(output_dir),
        'matrix_key': matrix_key,
        **summary,
    }
    return {
        **summary,
        'nodes': nodes,
        'gene_edges': gene_edges,
        'membership_edges': membership,
    }



def EvaluateWgcnaCytoscapeFilters(
    adata,
    *,
    condition_col='condition',
    wgcna_name='hdWGCNA',
    tom_cutoffs=(0.10, 0.20, 0.30, 0.35, 0.40),
    kme_cutoffs=(0.30, 0.50, 0.70),
    module_quantiles=(0.90, 0.95, 0.99),
    hub_counts=(25, 50, 100),
    top_connections=(5, 10),
):
    """Summarize full-module TOM and simulate visualization filters.

    Uses stored TOM (or 1 - tom_dissim), modules_df and dat_expr_genes.
    Does not modify adata, infer networks or export reduced networks.
    All simulations concern intramodular, undirected, positive-TOM edges;
    grey genes, self-loops and nonfinite weights are excluded. Between-module
    edges require a separate evaluation before the final export decision.

    Returns a dict of DataFrames:
      tom_by_module: exact pairwise TOM percentiles and full-module summaries;
      gene_metrics: own-module kME and mean/sum TOM to all other module genes;
      filters_by_module: selection, connected/isolated genes and edge counts;
      filters_by_condition: counts pooled across modules for each scenario.

    Thresholds use TOM > cutoff and kME >= cutoff. Module percentile cutoffs
    are estimated from all finite pairs in the original assigned module,
    including zero weights. Ties can change the nominal retained fraction.
    Top-N genes have finite kME, ranked within their assigned module. Top-K
    edges are ranked only among the selected genes in that same module;
    retain an edge if EITHER endpoint selects it. Final degree may exceed K.
    Ties are broken by gene name. Rankings retain the ORIGINAL TOM weights.

    Exact quantiles temporarily hold one module's upper-triangle values in
    memory (8 bytes per pair). No full-network edge DataFrame is constructed.
    Mean/sum TOM exclude self and nonfinite pairs and are not WGCNA kWithin
    (which is defined from adjacency). Filter diagnostics are descriptive,
    not significance or module-preservation tests.
    """
    import numpy as np
    import pandas as pd

    wd = adata.uns['hdWGCNA'][wgcna_name]
    labels = adata.obs[condition_col]
    if labels.isna().any() or labels.nunique() != 1:
        raise ValueError('adata must contain one nonmissing condition.')
    condition = str(labels.iloc[0])
    for q in module_quantiles:
        if not 0 <= q <= 1:
            raise ValueError('module_quantiles must be between 0 and 1.')
    for x in (*hub_counts, *top_connections):
        if not isinstance(x, (int, np.integer)) or x < 1:
            raise ValueError('hub_counts and top_connections must be positive integers.')
    for x in tom_cutoffs:
        if not np.isfinite(x) or not 0 <= x <= 1:
            raise ValueError('tom_cutoffs must be between 0 and 1.')
    for x in kme_cutoffs:
        if not np.isfinite(x) or not -1 <= x <= 1:
            raise ValueError('kme_cutoffs must be between -1 and 1.')

    matrix_key = 'TOM' if 'TOM' in wd else 'tom_dissim'
    matrix = np.asarray(wd[matrix_key])
    gene_order = pd.Index(wd['dat_expr_genes'])
    if matrix.shape != (len(gene_order), len(gene_order)):
        raise ValueError('TOM dimensions do not match dat_expr_genes.')
    table = wd['modules_df'].loc[wd['modules_df']['module'].ne('grey')].copy()
    if table.empty:
        raise ValueError('No assigned genes found.')
    table = table.sort_values('gene_name', kind='stable')

    def Weights(position, other_positions):
        weights = matrix[position, other_positions]
        return 1.0 - weights if matrix_key == 'tom_dissim' else weights

    def Measure(positions, cutoff=0.0, top_k=None):
        n = len(positions)
        degree = np.zeros(n, dtype=np.int64)
        n_edges, total_weight = 0, 0.0
        lowest, highest = np.inf, -np.inf
        if top_k is None:
            for i, position in enumerate(positions[:-1]):
                weights = Weights(position, positions[i + 1:])
                keep = np.isfinite(weights) & (weights > cutoff)
                degree[i] += int(keep.sum())
                degree[i + 1:] += keep
                retained = weights[keep]
                if retained.size:
                    n_edges += retained.size
                    total_weight += float(retained.sum())
                    lowest = min(lowest, float(retained.min()))
                    highest = max(highest, float(retained.max()))
        else:
            pairs = set()
            for i, position in enumerate(positions):
                weights = Weights(position, positions)
                valid = np.isfinite(weights) & (weights > cutoff)
                valid[i] = False
                candidates = np.flatnonzero(valid)
                ranked = candidates[np.argsort(-weights[candidates], kind='stable')]
                for j in ranked[:top_k]:
                    pairs.add(min(i, int(j)) * n + max(i, int(j)))
            if pairs:
                codes = np.fromiter(pairs, dtype=np.int64)
                left, right = codes // n, codes % n
                degree = np.bincount(np.concatenate([left, right]), minlength=n)
                weights = matrix[positions[left], positions[right]]
                if matrix_key == 'tom_dissim':
                    weights = 1.0 - weights
                n_edges = len(codes)
                total_weight = float(weights.sum())
                lowest, highest = float(weights.min()), float(weights.max())
        return {
            'n_selected': n,
            'n_connected': int((degree > 0).sum()),
            'n_isolated_selected': int((degree == 0).sum()),
            'n_edges': int(n_edges),
            'min_retained_TOM': lowest if n_edges else np.nan,
            'mean_retained_TOM': total_weight / n_edges if n_edges else np.nan,
            'max_retained_TOM': highest if n_edges else np.nan,
        }

    quantile_rows, gene_tables, filter_rows = [], [], []
    fixed_probs = (0.50, 0.90, 0.95, 0.99)
    probabilities = sorted(set(fixed_probs).union(module_quantiles))
    for module, genes in table.groupby('module', observed=True, sort=False):
        names = genes['gene_name'].to_numpy()
        positions = gene_order.get_indexer(names)
        if (positions < 0).any():
            raise ValueError(f'{module}: a gene is absent from dat_expr_genes.')
        kme = genes[f'kME_{module}'].to_numpy(dtype=float)
        n = len(genes)
        n_pairs = n * (n - 1) // 2
        values = np.empty(n_pairs, dtype=np.float64)
        weight_sum = np.zeros(n, dtype=float)
        finite_count = np.zeros(n, dtype=np.int64)
        offset = 0
        for i, position in enumerate(positions[:-1]):
            weights = Weights(position, positions[i + 1:])
            values[offset:offset + len(weights)] = weights
            offset += len(weights)
            valid = np.isfinite(weights)
            safe_weights = np.where(valid, weights, 0)
            weight_sum[i] += safe_weights.sum()
            weight_sum[i + 1:] += safe_weights
            finite_count[i] += valid.sum()
            finite_count[i + 1:] += valid
        if not np.isfinite(values).all():
            values = values[np.isfinite(values)]
        n_finite_pairs = len(values)
        n_positive_pairs = int((values > 0).sum())
        quantiles = dict(zip(
            probabilities,
            np.quantile(values, probabilities, overwrite_input=True)
            if n_finite_pairs else [np.nan] * len(probabilities),
        ))
        del values
        mean_tom = np.divide(
            weight_sum, finite_count, out=np.full(n, np.nan), where=finite_count > 0
        )
        valid = np.isfinite(kme) & np.isfinite(mean_tom)
        rho = np.nan
        if valid.sum() > 1 and np.unique(kme[valid]).size > 1 and np.unique(mean_tom[valid]).size > 1:
            rho = pd.Series(kme[valid]).corr(pd.Series(mean_tom[valid]), method='spearman')
        quantile_rows.append({
            'condition': condition, 'module': module, 'n_genes': n,
            'n_pairs': n_pairs, 'n_finite_pairs': n_finite_pairs,
            'n_positive_pairs': n_positive_pairs,
            **{f'TOM_P{round(100 * q)}': quantiles[q] for q in fixed_probs},
            'rho_kME_mean_TOM': rho,
            'n_negative_kME': int((kme < 0).sum()),
            'matrix_key': matrix_key,
        })
        gene_tables.append(pd.DataFrame({
            'condition': condition, 'module': module, 'gene_name': names,
            'kME': kme, 'full_module_mean_TOM': mean_tom,
            'full_module_sum_TOM': weight_sum, 'n_finite_module_partners': finite_count,
        }))

        all_indices = np.arange(n)
        scenarios = [('all_assigned', all_indices, 0.0, None)]
        scenarios += [(f'TOM>{t:g}', all_indices, t, None) for t in tom_cutoffs]
        scenarios += [(f'kME>={t:g}', np.flatnonzero(kme >= t), 0.0, None) for t in kme_cutoffs]
        scenarios += [(f'module_P{100*q:g}', all_indices, quantiles[q], None) for q in module_quantiles]
        eligible = np.flatnonzero(np.isfinite(kme))
        ranked = eligible[np.argsort(-kme[eligible], kind='stable')]
        for top_n in hub_counts:
            chosen = np.sort(ranked[:top_n])
            scenarios.append((f'top{top_n}_kME', chosen, 0.0, None))
            for top_k in top_connections:
                scenarios.append((f'top{top_n}_kME_top{top_k}_TOM', chosen, 0.0, top_k))
        for scenario, chosen, cutoff, top_k in scenarios:
            metrics = Measure(positions[chosen], cutoff, top_k)
            selected_pairs = len(chosen) * (len(chosen) - 1) // 2
            filter_rows.append({
                'condition': condition, 'module': module, 'scenario': scenario,
                'edge_scope': 'intramodular', 'n_genes_full': n,
                'n_pairs_full': n_pairs, 'TOM_cutoff': cutoff,
                'top_connections': top_k, **metrics,
                'pct_genes_selected': 100 * len(chosen) / n,
                'pct_genes_connected_full': 100 * metrics['n_connected'] / n,
                'pct_edges_of_full_pairs': 100 * metrics['n_edges'] / n_pairs if n_pairs else np.nan,
                'selected_pair_density': metrics['n_edges'] / selected_pairs if selected_pairs else np.nan,
            })

    per_module = pd.DataFrame(filter_rows)
    totals = (
        per_module.groupby(['condition', 'scenario', 'edge_scope'], sort=False)[
            ['n_genes_full', 'n_pairs_full', 'n_selected', 'n_connected',
             'n_isolated_selected', 'n_edges']
        ].sum().reset_index()
    )
    totals['pct_genes_selected'] = 100 * totals['n_selected'] / totals['n_genes_full']
    totals['pct_genes_connected_full'] = 100 * totals['n_connected'] / totals['n_genes_full']
    totals['pct_edges_of_full_pairs'] = 100 * totals['n_edges'] / totals['n_pairs_full']
    return {
        'tom_by_module': pd.DataFrame(quantile_rows),
        'gene_metrics': pd.concat(gene_tables, ignore_index=True),
        'filters_by_module': per_module,
        'filters_by_condition': totals,
    }


# -------------- MODULE PRESERVATION

### Original R/WGCNA module preservation from existing py_hdWGCNA results.


def PrepareWgcnaPreservationInputs(
    adata_ref,
    adata_contrast,
    ref_name='reference',
    contrast_name='contrast',
    wgcna_name='hdWGCNA',
):
    """Align saved reference/test metacell expression without modifying AnnData.

    Reads dat_expr (genes x metacells), dat_expr_genes, dat_expr_cells and
    modules_df from uns['hdWGCNA'][wgcna_name] in both objects.
    ref_name and contrast_name must be distinct dataset labels.

    Returns expression (dataset -> metacells x shared genes DataFrame),
    labels (shared genes x datasets), coverage (original/shared module sizes)
    and common_genes (list in reference order). Includes grey in the universe.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    # Keep explicit dataset names so the output records which object was used.
    if any(not isinstance(c, str) or not c for c in (ref_name, contrast_name)):
        raise ValueError('Dataset names must be nonempty strings.')
    if ref_name == contrast_name:
        raise ValueError('ref_name and contrast_name must differ.')
    networks = {ref_name: adata_ref, contrast_name: adata_contrast}

    # Read the existing network matrices; do not repeat cell or gene selection.
    data = {c: obj.uns['hdWGCNA'][wgcna_name] for c, obj in networks.items()}
    gene_lists = {c: pd.Index(wd['dat_expr_genes']) for c, wd in data.items()}
    for condition, genes in gene_lists.items():
        if not genes.is_unique or genes.hasnans:
            raise ValueError(f'{condition}: network gene names must be unique and nonmissing.')

    # Match final network genes, including grey, in the reference gene order.
    # This matching affects this comparison only, not the original networks.
    common_genes = gene_lists[ref_name].intersection(
        gene_lists[contrast_name], sort=False,
    ).tolist()
    if len(common_genes) < 4:
        raise ValueError('At least four shared network genes are required.')

    expression, labels, coverage = {}, {}, []
    for condition, wd in data.items():
        # Locate the shared genes in this object's original matrix order.
        genes = gene_lists[condition]
        cells = pd.Index(wd['dat_expr_cells'], name='metacell')
        values = wd['dat_expr']
        if values.shape != (len(genes), len(cells)):
            raise ValueError(f'{condition}: dat_expr dimensions do not match its labels.')
        positions = genes.get_indexer(common_genes)
        selected = values[positions, :] if sparse.issparse(values) else np.asarray(values)[positions, :]
        selected = selected.toarray() if sparse.issparse(selected) else selected
        # Reject nonfinite values; WGCNA also checks expression quality below.
        # Do not silently remove problematic genes or metacells.
        if len(cells) < 4 or not np.isfinite(selected).all():
            raise ValueError(f'{condition}: need >=4 metacells and finite expression values.')

        # R/WGCNA expects metacells in rows and genes in columns: transpose once.
        expression[condition] = pd.DataFrame(
            selected.T, index=cells, columns=pd.Index(common_genes, name='gene_name'),
        )

        # Align module labels to expression columns, not modules_df row order.
        assignments = wd['modules_df'].set_index('gene_name')['module']
        if not assignments.index.is_unique:
            raise ValueError(f'{condition}: modules_df must contain one row per gene.')
        assignments = assignments.loc[genes]
        if assignments.isna().any():
            raise ValueError(f'{condition}: missing module labels.')
        assignments = assignments.astype(str)
        if assignments.eq('gold').any():
            raise ValueError("'gold' is reserved for the WGCNA random control.")
        labels[condition] = assignments.loc[common_genes]

        # Record coverage, including modules with no genes left after matching.
        before, after = assignments.value_counts(), labels[condition].value_counts()
        for module, n_original in before.items():
            n_shared = int(after.get(module, 0))
            coverage.append({
                'condition': condition, 'module': module,
                'n_genes_original': int(n_original), 'n_genes_shared': n_shared,
                'pct_genes_shared': 100 * n_shared / n_original,
                'n_metacells': len(cells),
            })
    # Return data frames for reuse; neither AnnData object is modified.
    return {
        'expression': expression,
        'labels': pd.DataFrame(labels).rename_axis('gene_name'),
        'coverage': pd.DataFrame(coverage),
        'common_genes': common_genes,
    }


def RunWgcnaModulePreservation(
    adata_ref,
    adata_contrast,
    ref_name='reference',
    contrast_name='contrast',
    wgcna_name='hdWGCNA',
    bidirectional=True,
    n_permutations=250,
    network_type='signed',
    cor_method='pearson',
    seed=42,
    max_module_size=None,
    max_gold_module_size=1000,
    output_dir=None,
    verbose=True,
):
    """Call original R/WGCNA preservation through rpy2 + pandas2ri.

    adata_ref, adata_contrast : AnnData with saved hdWGCNA results.
    ref_name, contrast_name : distinct dataset names used in output tables.
    wgcna_name : analysis name shared by both objects.
    bidirectional : True tests both directions; False only reference -> contrast.
    n_permutations : permutation count (250 by default; >=100 recommended).
    network_type, cor_method : settings matching the original networks.
        Methods: pearson, spearman or bicor (R's individual Pearson fallback).
    seed : nonnegative R random seed.
    max_module_size : None retains all shared module genes; an integer permits
        WGCNA to subsample larger modules. max_gold_module_size caps its benchmark.
    output_dir : optional empty folder for tables, shared genes, parameters,
        versions and complete native RDS. None writes no files.
    verbose : show native WGCNA progress in the notebook.

    Returns summary, observed, Z, log_p, log_p_bonferroni, quality_observed,
    quality_Z and coverage DataFrames, plus common_genes, parameters and versions.
    Summary retains zero-coverage modules with missing statistics, excluding grey.
    Uses saved metacell expression; no original TOM, kME or hME input is needed.
    Does not modify AnnData or global conversion settings. Requires existing
    R/WGCNA and rpy2; no installation or transfer files.
    """
    import json
    import platform
    import warnings
    from importlib import metadata
    from numbers import Integral
    from pathlib import Path

    import pandas as pd

    # Reject settings that would silently change the requested analysis.
    if not isinstance(bidirectional, bool):
        raise ValueError('bidirectional must be True or False.')
    if not isinstance(n_permutations, Integral) or n_permutations < 2:
        raise ValueError('n_permutations must be an integer >=2.')
    if n_permutations < 100:
        warnings.warn('Fewer than 100 permutations: use only for a smoke test.', stacklevel=2)
    if network_type not in {'signed', 'unsigned', 'signed hybrid'}:
        raise ValueError('Unsupported network_type.')
    if cor_method not in {'pearson', 'spearman', 'bicor'}:
        raise ValueError('Unsupported cor_method.')
    if not isinstance(seed, Integral) or not 0 <= seed <= 2147483647:
        raise ValueError('seed must be an integer between 0 and 2147483647.')
    for name, value in [('max_module_size', max_module_size),
                        ('max_gold_module_size', max_gold_module_size)]:
        if value is not None and (not isinstance(value, Integral) or value < 4):
            raise ValueError(f'{name} must be an integer >=4.')
    if max_gold_module_size is None:
        raise ValueError('max_gold_module_size cannot be None.')

    # Read existing matrices and align genes; do not reconstruct either network.
    prepared = PrepareWgcnaPreservationInputs(
        adata_ref, adata_contrast, ref_name, contrast_name, wgcna_name,
    )
    names = [ref_name, contrast_name]
    labels, coverage = prepared['labels'], prepared['coverage']
    for name, obj in zip(names, (adata_ref, adata_contrast)):
        wd = obj.uns['hdWGCNA'][wgcna_name]
        for key, value in [('network_type', network_type), ('cor_method', cor_method)]:
            if key in wd and wd[key] != value:
                raise ValueError(f'{name}: {key} differs from the saved network setting.')

    # A cap equal to the shared universe prevents any module subsampling.
    cap = len(prepared['common_genes']) if max_module_size is None else int(max_module_size)
    if (coverage.loc[coverage['module'].ne('grey'), 'n_genes_shared'] > cap).any():
        warnings.warn('Large modules will be randomly subsampled to max_module_size.', stacklevel=2)
    directions = [(0, 1), (1, 0)] if bidirectional else [(0, 1)]
    cor_options = {
        'pearson': "use = 'p', method = 'pearson'",
        'spearman': "use = 'p', method = 'spearman'",
        'bicor': "use = 'p', pearsonFallback = 'individual'",
    }

    # Protect previous exports before starting the potentially expensive R test.
    folder = Path(output_dir).expanduser() if output_dir is not None else None
    if folder is not None:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError('output_dir must be empty; use a new folder for this run.')
        folder.mkdir(parents=True, exist_ok=True)

    # Only DataFrames cross the bridge; use rpy2's built-in pandas converter.
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    converter = ro.default_converter + pandas2ri.converter
    with localconverter(ro.default_converter):
        wgcna = importr('WGCNA')
        expression_r = {}
        for name, frame in prepared['expression'].items():
            # Transfer only the required matrix, preserving metacell/gene names.
            with localconverter(converter):
                frame_r = ro.conversion.get_conversion().py2rpy(frame)
            expression_r[name] = ro.ListVector({'data': frame_r})
        labels_r = ro.ListVector({name: ro.StrVector(labels[name].tolist()) for name in names})

        # Call the installed function directly. WGCNA checks expression quality;
        # no custom R function, file transfer or statistical reimplementation.
        native = wgcna.modulePreservation(
            multiData=ro.ListVector(expression_r), multiColor=labels_r,
            dataIsExpr=True, networkType=network_type,
            corFnc='bicor' if cor_method == 'bicor' else 'cor',
            corOptions=cor_options[cor_method],
            referenceNetworks=ro.IntVector([ref + 1 for ref, test in directions]),
            nPermutations=int(n_permutations), randomSeed=int(seed),
            maxModuleSize=cap, maxGoldModuleSize=int(max_gold_module_size),
            greyName='grey', goldName='gold', checkData=True,
            parallelCalculation=False, savePermutedStatistics=False,
            loadPermutedStatistics=False, verbose=3 if verbose else 0,
        )

        # Native tables are nested by reference position, then test dataset index.
        # Convert only those tables; the full permutation result stays in R.
        tables = {
            'observed': ('preservation', 'observed'),
            'Z': ('preservation', 'Z'),
            'log_p': ('preservation', 'log.p'),
            'log_p_bonferroni': ('preservation', 'log.pBonf'),
            'quality_observed': ('quality', 'observed'),
            'quality_Z': ('quality', 'Z'),
        }
        results = {}
        for key, (component, statistic) in tables.items():
            frames = []
            for i, (ref, test) in enumerate(directions):
                table_r = native.rx2(component).rx2(statistic)[i][test]
                if table_r is ro.NULL:
                    continue
                with localconverter(converter):
                    frame = ro.conversion.get_conversion().rpy2py(table_r)
                frame.insert(0, 'module', frame.index.astype(str))
                frame.insert(0, 'test', names[test])
                frame.insert(0, 'reference', names[ref])
                frames.append(frame.reset_index(drop=True))
            if not frames:
                raise ValueError(f'WGCNA returned no {component}/{statistic} tables.')
            results[key] = pd.concat(frames, ignore_index=True)

        # Keep R software versions and, optionally, the complete native result.
        versions = {
            'R': str(ro.r('R.version.string')[0]),
            'WGCNA': str(ro.r('as.character(utils::packageVersion("WGCNA"))')[0]),
            'Python': platform.python_version(),
        }
        if folder is not None:
            ro.r['saveRDS'](native, file=str(folder / 'WGCNA_modulePreservation.rds'))

    # Add ranks and coverage to the main Z statistics, retaining untestable modules.
    keys = ['reference', 'test', 'module']
    z_columns = ['moduleSize', 'Zsummary.pres', 'Zdensity.pres', 'Zconnectivity.pres']
    rank_columns = ['medianRank.pres', 'medianRankDensity.pres', 'medianRankConnectivity.pres']
    stats = results['Z'].reindex(columns=keys + z_columns).rename(
        columns={'moduleSize': 'n_genes_tested'},
    ).merge(results['observed'].reindex(columns=keys + rank_columns), on=keys, how='outer')
    pairs = pd.DataFrame([
        {'reference': names[ref], 'test': names[test]} for ref, test in directions
    ])
    results['summary'] = pairs.merge(
        coverage.loc[coverage['module'].ne('grey')].rename(columns={'condition': 'reference'}),
        on='reference', how='left',
    ).merge(stats, on=keys, how='left')

    # Record settings once; raw log-p tables are not p-values for Zsummary.
    parameters = {
        'wgcna_name': wgcna_name, 'reference': ref_name, 'contrast': contrast_name,
        'bidirectional': bidirectional, 'n_permutations': int(n_permutations),
        'network_type': network_type, 'cor_method': cor_method, 'seed': int(seed),
        'max_module_size_requested': None if max_module_size is None else int(max_module_size),
        'max_module_size_used': cap, 'max_gold_module_size': int(max_gold_module_size),
        'n_common_genes': len(prepared['common_genes']),
        'expression_source': 'saved dat_expr (metacells)',
        'integration': 'rpy2 + pandas2ri', 'parallel_calculation': False,
    }
    for package in ['rpy2', 'numpy', 'pandas']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    results.update({
        'coverage': coverage, 'common_genes': prepared['common_genes'],
        'parameters': parameters, 'versions': versions,
    })

    # Export each table once. Everything remains accessible as Python objects.
    if folder is not None:
        for key, table in results.items():
            if isinstance(table, pd.DataFrame):
                table.to_csv(folder / f'{key}.csv', index=False)
        pd.Series(results['common_genes'], name='gene_name').to_csv(
            folder / 'common_genes.csv', index=False,
        )
        for key in ['parameters', 'versions']:
            (folder / f'{key}.json').write_text(json.dumps(results[key], indent=2), encoding='utf-8')
    return results


#----------------------------
def PrepareWgcnaHubExpression(
    adata_ref,
    adata_contrast=None,
    modules_ref=None,
    modules_contrast=None,
    ref_name='WT',
    contrast_name='KO',
    n_hubs=10,
    layer='QC_filtered_log1p',
    group_by='celltype',
    sample_col='sample',
    group_order=None,
    wgcna_name='hdWGCNA',
):
    """Select own-module hubs and prepare jointly scaled gene-by-cell expression.

    With a contrast, use the union of both hub lists without duplicated genes.
    Return expression, scaled_expression, gene_metadata, cell_metadata,
    group_order and conditions. Inputs are not modified.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse

    if not isinstance(n_hubs, (int, np.integer)) or n_hubs < 1:
        raise ValueError('n_hubs must be a positive integer.')
    if adata_contrast is not None and ref_name == contrast_name:
        raise ValueError('Reference and contrast names must differ.')

    # Each condition can contribute hubs from all modules or a requested subset.
    datasets = [(ref_name, adata_ref, modules_ref)]
    if adata_contrast is not None:
        datasets.append((contrast_name, adata_contrast, modules_contrast))
    conditions = [name for name, _, _ in datasets]
    tables, selections, genes = {}, {}, []

    for name, obj, requested in datasets:
        table = obj.uns['hdWGCNA'][wgcna_name]['modules_df'].copy()
        table = table.set_index('gene_name', verify_integrity=True)
        available = sorted(table.loc[table['module'].ne('grey'), 'module'].unique())
        selected_modules = (
            available if requested is None
            else [requested] if isinstance(requested, str) else list(requested)
        )
        if not selected_modules or not set(selected_modules).issubset(available):
            raise ValueError(f'{name}: select existing, non-grey modules: {available}.')

        # Rank each gene by kME for its own module, using the saved results.
        selected = []
        for module in selected_modules:
            ranked = table.loc[table['module'].eq(module), f'kME_{module}']
            ranked = ranked.dropna().sort_values(ascending=False, kind='stable')
            if ranked.empty:
                raise ValueError(f'{name} {module}: no available kME values.')
            selected.extend(ranked.head(n_hubs).index)
        tables[name], selections[name] = table, set(selected)

        # Reference hubs come first; contrast contributes only additional genes.
        # A shared hub appears once in the heatmap.
        for gene in selected:
            if gene not in genes:
                genes.append(gene)

    # Use one stage order and the same gene order in all conditions.
    observed_groups = list(dict.fromkeys(
        str(group) for _, obj, _ in datasets
        for group in obj.obs[group_by].dropna().unique()
    ))
    stage_order = list(dict.fromkeys(str(group) for group in (
        observed_groups if group_order is None else group_order
    )))
    # Unlisted stages are appended rather than silently removing cells.
    stage_order += [group for group in observed_groups if group not in stage_order]
    gene_metadata = pd.DataFrame(index=pd.Index(genes, name='gene_name'))
    expressions, metadata = {}, {}

    for name, obj, _ in datasets:
        if not obj.var_names.is_unique or not obj.obs_names.is_unique:
            raise ValueError(f'{name}: gene and cell identifiers must be unique.')
        missing = pd.Index(genes).difference(obj.var_names)
        if len(missing):
            raise ValueError(f'{name}: selected genes absent from var_names: {list(missing)}.')
        if obj.obs[group_by].isna().any():
            raise ValueError(f'{name}: {group_by} contains missing labels.')

        # Sort cells by annotated stage and then sample; do not cluster cells.
        obs_columns = [group_by] + ([sample_col] if sample_col is not None
                                   and sample_col != group_by else [])
        obs = obj.obs[obs_columns].copy()
        obs[group_by] = pd.Categorical(
            obs[group_by].astype(str), categories=stage_order, ordered=True,
        )
        obs = obs.sort_values(obs_columns, kind='stable')

        # Densify only the selected genes, not the complete expression matrix.
        values = obj[obs.index, genes].layers[layer]
        values = values.toarray() if sparse.issparse(values) else np.asarray(values)
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f'{name}: selected expression contains nonfinite values.')
        expressions[name] = pd.DataFrame(values.T, index=genes, columns=obs.index)
        obs['condition'] = name
        metadata[name] = obs

        # Preserve original membership, even when a selected gene is grey elsewhere.
        table = tables[name].reindex(genes)
        gene_metadata[f'module_{name}'] = table['module'].astype(object).fillna('not_in_network')
        gene_metadata[f'hub_{name}'] = gene_metadata.index.isin(selections[name])
        gene_metadata[f'color_{name}'] = (
            table['color'].astype(object).fillna('#eeeeee')
            if 'color' in table else '#808080'
        )

    # Condition/cell MultiIndexes distinguish identical barcodes across datasets.
    expression = pd.concat(expressions, axis=1, names=['condition', 'cell'])
    cell_metadata = pd.concat(metadata, names=['condition_key', 'cell'])

    # In a two-condition plot, calculate each gene's mean and SD jointly.
    # Separate condition-wise scaling would remove between-condition mean shifts.
    values = expression.to_numpy()
    centered = values - values.mean(axis=1, keepdims=True)
    deviation = values.std(axis=1, keepdims=True, ddof=0)
    scaled = np.divide(centered, deviation, out=np.zeros_like(centered), where=deviation > 0)
    gene_metadata['constant_expression'] = deviation[:, 0] == 0
    scaled = pd.DataFrame(scaled, index=expression.index, columns=expression.columns)
    return {
        'expression': expression,
        'scaled_expression': scaled,
        'gene_metadata': gene_metadata,
        'cell_metadata': cell_metadata,
        'group_by': group_by,
        'group_order': stage_order,
        'conditions': conditions,
    }


def PlotWgcnaHubExpression(prepared, vlim=2, figsize=None, show_gene_names=True):
    """Plot prepared hub expression with stage and module annotation strips.

    Return a Matplotlib Figure. Cells and genes keep their prepared order.
    Colors saturate at +/-vlim; returned expression values remain unchanged.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    if vlim <= 0:
        raise ValueError('vlim must be positive.')
    data = prepared['scaled_expression']
    genes = prepared['gene_metadata'].loc[data.index]
    obs = prepared['cell_metadata'].reindex(data.columns)
    conditions, group_by = prepared['conditions'], prepared['group_by']
    if data.empty:
        raise ValueError('No genes or cells to plot.')
    n_genes, n_cells = data.shape

    # imshow creates a raster heatmap, keeping large cell-level PDFs manageable.
    fig = plt.figure(figsize=figsize or (11, max(4.5, 0.20 * n_genes + 2)))
    grid = fig.add_gridspec(
        4, 1 + len(conditions),
        height_ratios=[0.28, 0.40, max(2.5, 0.18 * n_genes), 0.16],
        width_ratios=[30] + [4] * len(conditions),
        hspace=0.10, wspace=0.05,
    )

    # Identify contiguous runs for readable group labels and divider lines.
    def _Runs(labels):
        labels = np.asarray(labels, dtype=object)
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        stops = np.r_[starts[1:], len(labels)]
        return [(start, stop, labels[start]) for start, stop in zip(starts, stops)]

    # Top strips show condition and stage; stages use the same colors everywhere.
    stages = prepared['group_order']
    stage_palette = dict(zip(stages, plt.get_cmap('Set3')(np.linspace(0, 1, len(stages)))))
    condition_palette = dict(zip(conditions, plt.get_cmap('Set2')(
        np.linspace(0, 1, len(conditions)))))
    for row, labels, palette in [
        (0, obs['condition'].astype(str).to_numpy(), condition_palette),
        (1, obs[group_by].astype(str).to_numpy(), stage_palette),
    ]:
        ax = fig.add_subplot(grid[row, 0])
        rgba = np.array([to_rgba(palette[label]) for label in labels])
        ax.imshow(rgba[None, :, :], aspect='auto', interpolation='nearest')
        for start, stop, label in _Runs(labels):
            ax.text((start + stop - 1) / 2, 0, str(label), ha='center', va='center', fontsize=8)
        ax.set_axis_off()

    # Magenta -> black -> yellow reproduces the supplied reference's color scheme.
    cmap = LinearSegmentedColormap.from_list('hub_expression', ['#cc00cc', '#000000', '#ffff00'])
    ax = fig.add_subplot(grid[2, 0])
    image = ax.imshow(data.to_numpy(), cmap=cmap, vmin=-vlim, vmax=vlim,
                      aspect='auto', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_genes) if show_gene_names else [])
    if show_gene_names:
        ax.set_yticklabels(data.index, fontsize=7, fontstyle='italic')
    ax.tick_params(axis='y', length=0)

    # Draw condition boundaries using the identical cell coordinates as the heatmap.
    for start, _, _ in _Runs(obs['condition'].astype(str)):
        if start:
            ax.axvline(start - 0.5, color='white', linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Right strips retain module assignments in each original network.
    # The same gene is shown once, with a separate membership strip per condition.
    for col, name in enumerate(conditions, start=1):
        strip = fig.add_subplot(grid[2, col])
        colors = np.array([to_rgba(color) for color in genes[f'color_{name}']])
        strip.imshow(colors[:, None, :], aspect='auto', interpolation='nearest')
        # Reserve room for labels inside each strip's own subplot.
        strip.set_xlim(-0.5, 4)
        for start, stop, module in _Runs(genes[f'module_{name}'].astype(str)):
            strip.text(1.15, (start + stop - 1) / 2, str(module),
                       ha='left', va='center', fontsize=7, clip_on=False)
        strip.set_title(name, fontsize=9)
        strip.set_axis_off()

    # Color limits only control display saturation, not gene selection or testing.
    colorbar_ax = fig.add_subplot(grid[3, 0])
    fig.colorbar(image, cax=colorbar_ax, orientation='horizontal',
                 label='Expression z-score per gene', ticks=[-vlim, 0, vlim])
    fig.suptitle('Module hub gene expression', y=0.99, fontsize=13)
    fig.subplots_adjust(left=0.16, right=0.89, top=0.91, bottom=0.13)
    return fig
