## Reference: hdWGCNA terminology

WGCNA identifies modules of genes with related expression and network connection patterns. In this workflow, networks are constructed from metacells; module eigengenes and kME are subsequently calculated in individual cells [Langfelder and Horvath, 2008; Morabito et al., 2023].

| Term | Meaning in this workflow |
|---|---|
| Network gene | A gene retained in `dat_expr_genes`, whether assigned to a module or labelled `grey`. |
| Metacell | An aggregate of similar cells from the same cell group and biological sample. Metacells reduce sparsity and may share cells; they are not independent biological replicates. |
| Gene–gene correlation | Similarity between two genes' expression profiles across the selected metacells. Pearson is used here. |
| Adjacency | A connection weight derived from the gene–gene correlation. For a signed network, off-diagonal weights are `((1 + correlation) / 2) ** power`. Negative correlations produce small weights. |
| Soft power (`power`, β) | Exponent used to transform correlations into adjacency weights. Increasing it suppresses weak connections relative to strong ones. |
| Scale-free fit (`SFT.R.sq`) and slope | Diagnostics used to choose the power. This notebook selects the lowest tested power with R² ≥ `r2_cutoff` and negative slope. These criteria do not establish biological validity. |
| Connectivity (`k`) | Sum of a gene's adjacency weights to other genes, excluding itself. Intramodular connectivity restricts that sum to genes in the same module. This is distinct from kME. |
| TOM | Topological overlap similarity between **two genes**, considering their connection and shared neighbours. Each gene has a TOM value with every other gene; TOM is not a single score per gene. |
| TOM dissimilarity | `1 - TOM`. Smaller values indicate greater topological similarity. This is the distance used for hierarchical clustering. |
| Module | A group of genes identified by clustering the network. Genes within a module need not all have strong pairwise connections; connections can also exist between modules. |
| `min_module_size` | Minimum cluster size requested during module detection. A value of 50 concerns module membership, not the number of hubs. |
| `grey` | Genes retained in the network but not assigned to a module. They are not automatically unexpressed, noise or biologically irrelevant. |
| Module eigengene (`ME`) | First principal component of the scaled expression of a module's genes. It summarizes the module's expression pattern, with one value per cell after `ComputeWgcnaEigengenes`. It is not a simple expression mean. |
| Harmonized eigengene (`hME`) | Eigengene adjusted using Harmony when requested. With `harmonize=False`, `hMEs` is a copy of `MEs` in this Python implementation. |
| kME | Correlation between a gene's expression and a module eigengene. It is a **gene–module** measure, not a gene–gene edge weight. Here it is Pearson correlation across the selected individual cells. |
| Hub candidate | A gene ranked highly by a specified centrality measure. `GetWgcnaHubGenes` ranks genes by decreasing kME **within their assigned module**. A high kME does not demonstrate regulatory function. |
| `n_hubs` | Number of top-ranked genes returned per module. It changes the selected subset, not the previously inferred modules or network. There is no universal hub cutoff. |
| Enrichment background | Genes eligible for the enrichment test. With `background='network'`, this wrapper uses the final network genes, including `grey`. |

For a gene *g* and module *M*, `kME(g, M) = correlation(expression of g, eigengene of M)`. Values near +1 indicate a similar pattern, values near 0 little linear association, and values near −1 an opposite pattern. kME can be computed against every module, regardless of the gene's assigned module.

For Cytoscape, TOM can represent gene–gene edge weights, module assignment can determine node colours, and own-module kME can determine node size. If modules are represented as explicit nodes, gene–module edges weighted by kME describe membership association. They must be distinguished from TOM edges.

Co-expression does not establish physical interaction or causal regulation. In independent WT and KO networks, module labels are local: `WT M1` and `KO M1` are not automatically corresponding modules. Differences in visual layout or edge weights alone are not a statistical test of network change.

## Reference: storage in `uns['hdWGCNA']`

The following describes the supplied `sctools.hdwgcna` wrapper and the published `py-hdwgcna` 0.1.0 implementation. Keys appear as their corresponding steps are executed; implementations or local modifications can change this inventory.

```python
obj = networks['KO']  # Use adata instead for a single-object tutorial.
container = obj.uns['hdWGCNA']
wd = container[wgcna_name]
```

### Container and analysis name

| Location | Contents | Created or maintained by |
|---|---|---|
| `obj.uns['hdWGCNA']` | Container holding named analyses and an active-analysis pointer. | Initialized by `SelectWgcnaGenes` through `setup_for_wgcna`. |
| `container[wgcna_name]` | Dictionary containing the results of one named analysis. With `wgcna_name='hdWGCNA'`, this is `obj.uns['hdWGCNA']['hdWGCNA']`. | Initialized by `SelectWgcnaGenes`; updated by subsequent steps. |
| `container['active_wgcna']` | String naming the default analysis when an underlying function receives no explicit `wgcna_name`. It contains no network matrix. | Initialized during setup and updated when results are written by the package. |

The repeated name `hdWGCNA` denotes two different levels: the package container and the chosen analysis name. Keep `active_wgcna`; it is package metadata and does not duplicate the network.

### Results inside `wd`

Let **G** be the number of final network genes, **C** the number of selected metacells, **N** the number of individual cells and **M** the number of non-grey modules.

| Key | Contents / dimensions | Function generating or updating it |
|---|---|---|
| `genes_use` | Initial selected gene list; not necessarily the final network gene list. | `SelectWgcnaGenes` |
| `gene_select_method`, `n_genes`, `setup_complete` | Selection mode, initial selected gene count and setup status. | `SelectWgcnaGenes` |
| `active_wgcna` | Analysis-name string also stored inside this dictionary; not another network. | `SelectWgcnaGenes`; refreshed by `BuildWgcnaMetacells`. |
| `metacell_obj` | Metacell AnnData: observations are metacells; variables are genes. `.X` contains log-normalized expression. | `BuildWgcnaMetacells` |
| `metacell_groupby`, `metacell_k`, `metacell_max_shared`, `metacell_reduction` | Grouping columns, aggregation size, allowed overlap and embedding used. | `BuildWgcnaMetacells` |
| `metacell_to_cell_map` | Mapping from generated metacell identifiers to integer positions of contributing cells in the original object. | `BuildWgcnaMetacells` |
| `dat_expr` | **G × C** normalized expression matrix used to construct the network. | Initially populated during `SelectWgcnaGenes` using individual cells; replaced with selected metacell expression by `SetWgcnaExpression`. |
| `dat_expr_genes` | Ordered gene labels for the rows of `dat_expr` and both axes of TOM. | `SetWgcnaExpression`, after the final expression-quality/variance filter; initially populated at setup. |
| `dat_expr_cells` | Ordered observation identifiers for the columns of `dat_expr`; these are metacell identifiers after `SetWgcnaExpression`. | `SetWgcnaExpression`; initially populated at setup. |
| `group_name`, `group_by` | Selected groups and the metadata column used to select them. | `SetWgcnaExpression`; initially populated at setup. |
| `power_table` | Diagnostics for each tested power: fit, slope and connectivity. | `TestWgcnaSoftPowers` |
| `cor_method`, `network_type` | Correlation method and network type used in power testing. | `TestWgcnaSoftPowers`; reused for network construction. |
| `selected_power` | Selected soft power. | Initially assigned by power testing; explicitly updated by `SelectWgcnaSoftPower`. |
| `TOM` | **G × G** topological overlap similarity matrix. Includes assigned and `grey` genes. | `BuildWgcnaNetwork` |
| `tom_dissim` | **G × G** matrix equal to `1 - TOM`. | `BuildWgcnaNetwork` |
| `modules_df` | One row per network gene: `gene_name`, assigned `module`, `color` and `kME_<module>` columns. | `BuildWgcnaNetwork`; kME columns are subsequently updated by `ComputeWgcnaKme`. |
| `module_labels`, `valid_modules` | Numeric gene assignments and module identifiers used internally; assignment 0 denotes `grey`. | `BuildWgcnaNetwork` |
| `network_params` | Construction settings, including power, network type, TOM type and module detection/merging parameters. | `BuildWgcnaNetwork` |
| `tom_name` | TOM name/prefix supplied during construction. Its presence alone does not mean a TOM file was written. | `BuildWgcnaNetwork` |
| `MEs`, `hMEs` | After the eigengene step: **N × M** labelled DataFrames. With no Harmony, their values are identical. | Initially metacell-based arrays from `BuildWgcnaNetwork`; replaced by `ComputeWgcnaEigengenes`. |
| `ME_cells`, `module_names` | Cell identifiers and module names corresponding to the eigengene tables. | `ComputeWgcnaEigengenes` |
| `var_explained_MEs` | Explained variance returned by the single-cell eigengene calculation. | `ComputeWgcnaEigengenes` |
| `var_explained` | Earlier variance information from the initial metacell eigengene calculation, before module merging. Not a substitute for the final single-cell result above. | `BuildWgcnaNetwork` |
| `kME` | **G × M** correlation array in this workflow; use the labelled `modules_df` columns for gene-level interpretation. | Initially calculated on metacells by `BuildWgcnaNetwork`; replaced by the selected single-cell calculation in `ComputeWgcnaKme`. |
| `kME_computed` | Flag indicating that the module-connectivity step has run. | `ComputeWgcnaKme` |
| `linkage_matrix` | Cached hierarchical tree for the dendrogram, if that plot has been generated. | May be created by `PlotWgcnaDendrogram`; cleared when `BuildWgcnaNetwork` reruns. |
| `sctools_params` | Dictionary of arguments recorded by the wrapper, indexed by step name. It is a record of the most recent call for each recorded step, not a complete execution history. | `_RecordParams`, called by the corresponding wrapper functions. |

`MEs`, `hMEs` and kME are first computed during network construction and then replaced by later steps. Their interpretation therefore depends on which steps have run. After rebuilding a network, rerun eigengenes and kME before downstream interpretation.

The correlation cache `cor_matrix` is removed by `TestWgcnaSoftPowers` in this wrapper. Correlation and adjacency matrices used during `BuildWgcnaNetwork` are temporary and are not retained as final result matrices.

Inside `metacell_obj`, `.layers['counts']` stores the pre-normalization aggregate profiles. In this implementation they are averages of contributing cell counts, so they need not be integers. `.X` stores their log1p(CP10k) normalization.

### Results returned outside `uns`

| Function or notebook object | Output | Storage behaviour |
|---|---|---|
| `GetWgcnaHubGenes` | Top-ranked genes per module: gene name, module and own-module kME. | Returns a table; it does not create a separate hub table in `wd`. |
| `RunWgcnaEnrichr` | Enrichment results per module and database. | Returns a table; only its recorded arguments are added to `sctools_params`. |
| `GetCommonWgcnaGenes` | Shared selected gene list and selection counts. | Returns the results without initializing or updating `uns`. Not used to filter the independently constructed WT and KO networks here. |
| Input/metacell summaries and `CheckBicorMad` | Summary tables or diagnostics. | Returned to the caller; not automatically stored in `wd`. |
| `module_sizes`, `gene_catalog`, `module_overlap`, `module_jaccard` | Module counts and notebook-level gene/network comparisons. | Kept as notebook variables or explicitly exported files, not automatically stored in `wd`. |

To inspect the keys actually available at the current stage:

```python
list(wd.keys())
```

### References

- Langfelder P, Horvath S. WGCNA: an R package for weighted correlation network analysis. *BMC Bioinformatics* (2008). https://doi.org/10.1186/1471-2105-9-559
- Morabito S et al. hdWGCNA identifies co-expression networks in high-dimensional transcriptomics data. *Cell Reports Methods* (2023). https://doi.org/10.1016/j.crmeth.2023.100498
- hdWGCNA documentation: [network construction](https://smorabit.github.io/hdWGCNA/reference/ConstructNetwork.html), [module eigengenes](https://smorabit.github.io/hdWGCNA/reference/ModuleEigengenes.html) and [module connectivity](https://smorabit.github.io/hdWGCNA/reference/ModuleConnectivity.html).
- Implementation reference: the supplied `sctools/hdwgcna.py` and the published [py-hdwgcna 0.1.0 source distribution](https://pypi.org/project/py-hdwgcna/0.1.0/). Python storage details were checked in the source code rather than inferred from the R package.
