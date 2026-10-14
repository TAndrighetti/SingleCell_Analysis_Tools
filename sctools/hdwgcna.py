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

def SelectWgcnaGenes(adata, fraction=0.05, wgcna_name='hdWGCNA'):
    """Keep genes detected in at least `fraction` of the cells (hdWGCNA tutorial: 0.05).

    The fraction is computed by py_hdWGCNA on `layers['counts']` (see
    `PrepareWgcnaInput`), over all cells together. Do not pre-filter genes
    by differential expression: WGCNA expects the full set of expressed genes.

    Returns
    -------
    AnnData with the results dict created in adata.uns['hdWGCNA'][wgcna_name].
    """
    from py_hdWGCNA.core import setup_for_wgcna

    adata = setup_for_wgcna(
        adata,
        gene_select='fraction',
        fraction=fraction,
        wgcna_name=wgcna_name,
    )
    _RecordParams(adata, wgcna_name, 'SelectWgcnaGenes', {'fraction': fraction})
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
