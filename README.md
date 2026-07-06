# sctools

Single-cell RNA-seq analysis utilities built on top of [Scanpy](https://scanpy.readthedocs.io/) and [AnnData](https://anndata.readthedocs.io/).

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

### `sctools.preprocessing` — Normalization, HVG, PCA, kNN

| Function | Description |
|---|---|
| `NormalizeHvgPcaKnn` | Normalize, log1p, select per-batch HVGs, PCA, kNN graph, UMAP (pre-integration baseline) |
| `CheckNormalizedLayer` | Validate that a matrix is non-negative (normalized/log-transformed, not scaled) |

### `sctools.alra` — Zero-preserving imputation (requires R)

| Function | Description |
|---|---|
| `RunAlraOnAnnData` | Run the official R `ALRA` package (`choose_k()` + `alra()`) via rpy2 |

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
