# sctools

Single-cell RNA-seq analysis utilities built on top of [Scanpy](https://scanpy.readthedocs.io/) and [AnnData](https://anndata.readthedocs.io/).

> IMPORTANT: STILL IN DEVELOPMENT!

## Installation

```bash
pip install -e .
```

## Modules

### `sctools.io` — Data loading

| Function | Description |
|---|---|
| `load_sample` | Load a single sample (MTX or Cell Ranger H5) into AnnData |
| `load_samples` | Load multiple samples and concatenate |
| `CatAdata` | Concatenate a dict or list of AnnData objects |

### `sctools.qc` — Quality control

**Ambient RNA correction**

| Function | Description |
|---|---|
| `AmbientRNA` | Correct ambient RNA contamination with SoupX |

**Doublet detection**

| Function | Description |
|---|---|
| `ScoreDoublets` | Score cells as potential doublets with Scrublet |
| `PlotScrubletScores` | Plot doublet score distribution |
| `CallDoublets` | Label cells as doublets based on a threshold |
| `FilterDoublets` | Remove doublets from AnnData |

**QC metrics and filtering**

| Function | Description |
|---|---|
| `CalculateQcMetrics` | Calculate QC metrics (MT, ribo, Hb genes) |
| `QCmetric` | Alias for `CalculateQcMetrics` |
| `EvaluateQcThresholds` | Evaluate combinations of QC thresholds (absolute or MADs) |
| `FilterQcCells` | Filter cells based on QC thresholds |
| `QcSummaryTable` | Build a per-sample summary table comparing before/after filtering |

**QC plots**

| Function | Description |
|---|---|
| `PlotQCViolinsGrid` | Violin plots of QC metrics per sample |
| `PlotQcHeatmap` | Heatmap of cells removed per threshold combination (single sample) |
| `PlotQcThresholdsGrid` | Grid of heatmaps across multiple samples |
| `PlotQcBoxplots` | Boxplots comparing a metric before and after filtering |

### `sctools.plots` — Generic plots

| Function | Description |
|---|---|
| `PlotHeatmap` | Generic heatmap with optional cell annotations |
| `PlotSignificanceHeatmap` | Generic value+significance heatmap from a long-format table (color by a value column, "*" by a p-value/padj column, optional `restrict_to_significant` row filter, optional category sidebar) |

### `sctools.preprocessing` — Normalization, HVG, PCA, kNN

| Function | Description |
|---|---|
| `NormalizeHvgPcaKnn` | Normalize, log1p, select per-batch HVGs, PCA, kNN graph, UMAP (pre-integration baseline) |
| `CheckNormalizedLayer` | Validate that a matrix is non-negative (normalized/log-transformed, not scaled) |

### `sctools.alra` — Zero-preserving imputation (requires R)

| Function | Description |
|---|---|
| `RunAlraOnAnnData` | Run the official R `ALRA` package (`choose_k()` + `alra()`) via rpy2 |

### `sctools.annotation` — Marker-based cell-type scoring (AUCell requires `decoupler`)

| Function | Description |
|---|---|
| `CalculateModuleScores` | Per-signature module score (`sc.tl.score_genes`), normalized by sqrt(n_genes) |
| `PlotModuleScoresUMAPs` | UMAP grid of module scores, diverging color scale centered at 0 |
| `CalculateAUCellWithDecoupler` | Per-signature AUCell score (`decoupler.mt.aucell`), normalized by sqrt(n_genes) |
| `PlotAUCellUMAPs` | UMAP grid of AUCell scores, sequential color scale from 0 |

### `sctools.integration` — Batch integration, clustering, benchmarking

| Function | Description |
|---|---|
| `RunSeuratAnchors` | Seurat CCA anchor integration (via rpy2/R) |
| `PlotUmap` | Save a UMAP plot to disk for a given representation |
| `ApplyIntegrationMethods` | Apply BBKNN / Seurat / Scanorama / Harmony / scVI, return one AnnData per method |
| `Clustering` | Leiden clustering at multiple resolutions + UMAP plots |
| `RunIntegration` | Full single-method pipeline: `NormalizeHvgPcaKnn` -> integration -> clustering |
| `AttachHvgResultsToFullAdata` | Attach HVG-based embeddings/clusters back onto the full (all-genes) AnnData |
| `RemoveClustersFromOriginal` | Flag clusters in an integrated AnnData, log via `UpdateCellsToRemove`, reload + filter the pristine original |
| `UpdateCellsToRemove` | Log + accumulate low-quality clusters to remove across iterative curation passes |
| `RunScibMetricsWithLeiden` | scIB metrics (PCR_batch, iLISI, hvg_score, cell_cycle, silhouette) using Leiden as proxy labels |
| `BuildCombinationsDictAndParamsDf` | Build a flavor x n_top_genes x n_pcs grid for benchmarking |
| `RunIntegrationTests` | Run the full grid: preprocess, integrate, score each combination with scIB |

### `sctools.degs` — Pseudobulk DE (PyDESeq2) & hallmark/pathway activity (decoupler ULM)

| Function | Description |
|---|---|
| `Pseudobulking` | Aggregate counts per (celltype, sample) with `decoupler` |
| `PseudoPCA` | Optional QC/exploration PCA on pseudobulk profiles -- never required for DEGs, never modifies raw counts |
| `PseudoFeatSelection` | Per-celltype gene filtering before DE (`decoupler` filter_by_expr/filter_by_prop) |
| `PseudoDESeq2` | Run a DESeq2 contrast on raw pseudobulk counts for one celltype (PyDESeq2), with a minimum-replicates check |
| `VolcanoGridByGroup` | Volcano plot grid, one panel per celltype/group |
| `RunULM` | Run ULM (`decoupler.mt.ulm`) for ONE input (one celltype's `PseudoDESeq2` data, or a whole AnnData) against any network (hallmark, progeny, collectri, ...), + optional barplot (pass `ax=` to draw into a shared grid) -- loop over celltypes yourself |
| `MeltActsPadjToLong` | Combine several celltypes' stored acts/padj tables into one tidy long-format table -- feed straight into `sctools.plots.PlotSignificanceHeatmap` |
| `BuildSignificantFeatureTable` | Long-format table restricted to significant hits, + category/direction columns (optional category dict with prefix stripping, e.g. `"HALLMARK_"`) |
| `SummarizeFeatureCategories` | Per-celltype/category summary (counts, mean score, dominant direction) |

## Typical QC workflow

```python
from sctools.io import load_sample, CatAdata
from sctools.qc import (
    AmbientRNA, ScoreDoublets, CallDoublets, FilterDoublets,
    CalculateQcMetrics, EvaluateQcThresholds, FilterQcCells,
    PlotQCViolinsGrid, PlotQcThresholdsGrid, QcSummaryTable,
)

# 1. Load samples
adatas = {name: load_sample(path, format="mtx", obs_meta={"sample": name})
          for name, path in sample_paths.items()}

# 2. Ambient RNA correction (SoupX)
adatas_soupx = {name: AmbientRNA(adata, raw_unfiltered) for name, adata in adatas.items()}

# 3. Doublet detection
for adata in adatas_soupx.values():
    ScoreDoublets(adata)
    CallDoublets(adata, threshold=0.25)

adatas_singlets = {name: FilterDoublets(adata)[0] for name, adata in adatas_soupx.items()}

# 4. QC metrics
for adata in adatas_singlets.values():
    CalculateQcMetrics(adata)

# 5. Evaluate thresholds
thresholds = {"n_genes_by_counts": {"min": [200, 500]}, "pct_counts_mt": {"max": [10, 15, 20]}}
summary_dfs = {name: EvaluateQcThresholds(adata, thresholds) for name, adata in adatas_singlets.items()}
PlotQcThresholdsGrid(summary_dfs, x="n_genes_by_counts_min", y="pct_counts_mt_max")

# 6. Filter
cutoffs = {"n_genes_by_counts": {"min": 200}, "pct_counts_mt": {"max": 15}}
adatas_filtered = {name: FilterQcCells(adata, cutoffs)[0] for name, adata in adatas_singlets.items()}

# 7. Summary table
QcSummaryTable(adatas_before=adatas_singlets, adatas_after=adatas_filtered, cutoffs=cutoffs)
```

## Typical pre-integration workflow

```python
from sctools.io import CatAdata
from sctools.preprocessing import NormalizeHvgPcaKnn

# 8. Concatenate filtered samples and build the pre-integration baseline embedding
adata = CatAdata(adatas_filtered)
adata = NormalizeHvgPcaKnn(adata, input_layer="QC_filtered", batch_key="sample",
                            plot_before_integration=True)
```

## Typical imputation workflow

```python
from sctools.alra import RunAlraOnAnnData

# 9. Impute dropouts for exploratory marker visualization (not for DE)
adata = RunAlraOnAnnData(adata, input_layer="QC_filtered_log1p", output_layer="alra")
```

## Typical integration workflow (single method, production run)

```python
from sctools.integration import RunIntegration, AttachHvgResultsToFullAdata

# 10. Preprocess + integrate with one method + Leiden-cluster the result
adata_run, adata_cl_hvg = RunIntegration(
    adata, method="harmony", out_dir="plots/umaps",
    flavor="cell_ranger", n_top_genes=1000, n_pcs=20, n_neighbors=15,
)

# 11. Carry the HVG-based embeddings/clusters back onto the full-gene object
adata_full = AttachHvgResultsToFullAdata(adata_run, adata_cl_hvg, method="harmony")
```

## Typical integration benchmarking workflow (grid search + scIB metrics)

```python
from sctools.integration import (
    BuildCombinationsDictAndParamsDf, RunIntegrationTests,
)

# 12. Build a flavor x n_top_genes x n_pcs grid
config = {
    "flavors": ("seurat", "seurat_v3", "cell_ranger"),
    "n_top_genes_list": (1000, 2000, 3000),
    "n_pcs_list": (20, 35, 50),
}
combinations_dict, params_df = BuildCombinationsDictAndParamsDf(config)

# 13. Run every combination through every integration method and score with scIB
# organism is required explicitly ("human" or "mouse") -- it is NOT inferred.
all_metrics_df, params_df = RunIntegrationTests(
    adata=adata,
    combinations_dict=combinations_dict,
    out_dir="results/benchmark",
    versao="04.1sx.scNeuAntib",
    batch_key="sample",
    input_layer="QC_filtered",
    counts_layer="counts",
    log_layer="QC_filtered_log1p",
    organism="mouse",
    methods=["seurat", "harmony", "scanorama", "bbknn", "scvi"],
)
```

## Typical annotation workflow (marker-based scoring)

```python
from sctools.annotation import (
    CalculateModuleScores, PlotModuleScoresUMAPs,
    CalculateAUCellWithDecoupler, PlotAUCellUMAPs,
)

marker_genes_dic = {"Goblet": ["Muc2", "Tff3"], "Colonocyte": ["Slc26a3", "Car1"]}

# 14. Module scores (sc.tl.score_genes) + UMAPs
adata = CalculateModuleScores(adata, marker_genes_dic, layer="QC_filtered_log1p")
PlotModuleScoresUMAPs(adata)

# 15. AUCell scores (decoupler) + UMAPs
adata = CalculateAUCellWithDecoupler(adata, marker_genes_dic, layer="QC_filtered_log1p")
PlotAUCellUMAPs(adata)
```

## Typical DEGs / hallmark activity workflow (pseudobulk PyDESeq2 + decoupler ULM)

```python
from sctools.degs import (
    Pseudobulking, PseudoPCA, PseudoFeatSelection, PseudoDESeq2,
    VolcanoGridByGroup, RunULM, MeltActsPadjToLong,
)
from sctools.plots import PlotSignificanceHeatmap

# 16. Aggregate counts per (celltype, sample); PCA here is optional QC only,
#     never required for DEGs -- suitable_celltypes comes from your own
#     cell/replicate-count check (kept notebook-local, not in this package).
pdata = Pseudobulking(adata, celltype_col="celltype", sample_col="sample")
pdata = PseudoPCA(pdata, celltype_col="celltype", condition_col="condition")

# 17. Per-celltype DE: filter genes, run PyDESeq2 on raw pseudobulk counts, collect results
pdata_by_celltype = {ct: {} for ct in suitable_celltypes}
for ct in suitable_celltypes:
    pdata_by_celltype[ct]["pdata"] = PseudoFeatSelection(pdata, celltype=ct, celltype_col="celltype", plot=False)
    pdata_by_celltype[ct]["results_df"], pdata_by_celltype[ct]["data"] = PseudoDESeq2(
        pdata_by_celltype[ct]["pdata"], design_col="condition",
        normal_condition="CTRL", compare_condition="TREATED", plot_volcano=False,
    )

# 18. Volcano grid across celltypes
import pandas as pd
degs = pd.concat(
    [df["results_df"].assign(celltype=ct) for ct, df in pdata_by_celltype.items()]
).reset_index().rename(columns={"index": "gene_id"})
VolcanoGridByGroup(degs, group_col="celltype", col_gene="gene_id")

# 19. Hallmark activity (decoupler ULM) across celltypes -- RunULM handles ONE
#     celltype at a time (like PseudoFeatSelection/PseudoDESeq2), so you loop
#     yourself and choose the storage keys (namespaced by network here, so a
#     later run with a different net, e.g. progeny, won't overwrite this one).
import decoupler as dc
hallmark = dc.op.hallmark(organism="mouse")
for ct in suitable_celltypes:
    hm_acts, hm_padj = RunULM(pdata_by_celltype[ct]["data"], hallmark, contrast_name="TREATED.vs.CTRL", plot=False)
    pdata_by_celltype[ct]["hallmark_acts"] = hm_acts
    pdata_by_celltype[ct]["hallmark_padj"] = hm_padj

ulm_long = MeltActsPadjToLong(
    pdata_by_celltype, "hallmark_acts", "hallmark_padj", "TREATED.vs.CTRL", feature_col="hallmark",
)
heatmap_df, padj_df, annot_df = PlotSignificanceHeatmap(
    ulm_long, x="celltype", y="hallmark", value_col="score", pvalue_col="padj",
    restrict_to_significant=True, title="Hallmark activities (TREATED vs CTRL)",
)
```

## Requirements

- Python ≥ 3.10
- anndata
- scanpy
- numpy
- pandas
- scipy
- matplotlib

## Author

Tahila Andrighetti — tahilaandrighetti@gmail.com

---

> Code improvement and organization supported by [Claude Code](https://claude.ai/code) (Anthropic).
