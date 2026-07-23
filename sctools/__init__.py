"""sctools – single-cell RNA-seq analysis utilities."""

import logging
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sctools")
except PackageNotFoundError:
    __version__ = "unknown"

# Every sctools module logs via logging.getLogger(__name__), e.g.
# "sctools.preprocessing" -- a child of this "sctools" logger. Without a
# handler here, logger.info(...) calls are silently dropped (Python's default
# logging level is WARNING). Attaching the handler only to "sctools" (not the
# root logger via logging.basicConfig()) shows sctools' own INFO messages
# without changing the verbosity of scanpy/other libraries. Guarded so
# re-importing sctools (e.g. Jupyter %autoreload) doesn't duplicate handlers.
_sctools_logger = logging.getLogger("sctools")
if not _sctools_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _sctools_logger.addHandler(_handler)
    _sctools_logger.setLevel(logging.INFO)

from sctools.plots import PlotHeatmap, PlotSignificanceHeatmap
from sctools.io import CatAdata
from sctools.preprocessing import (
    NormalizeHvgPcaKnn,
    CheckNormalizedLayer,
    NormalizeLog1pFromLayer,
    RunHighlyVariableGenes,
    RunPcaOnHvgs,
    RunNeighborsAndUmap,
)
from sctools.alra import RunAlraOnAnnData

from sctools.annotation import (
    CalculateModuleScores,
    PlotModuleScoresUMAPs,
    CalculateAUCellWithDecoupler,
    PlotAUCellUMAPs,
)

from sctools.integration import (
    RunSeuratAnchors,
    PlotUmap,
    RunIntegration,
    AttachHvgResultsToFullAdata,
    RemoveClustersFromOriginal,
    UpdateCellsToRemove,
    RunScibMetricsWithLabel,
    RunScibMetricsLabelFree,
    BuildBenchmarkGrid,
    RunIntegrationBenchmark,
    SummarizeScibBenchmarkResults,
    RunHarmonyIntegration,
    RunScanoramaIntegration,
    RunScviIntegration,
    RunSeuratAnchorsIntegration,
    RunLeidenClustering,
)

from sctools.degs import (
    Pseudobulking,
    PseudoPCA,
    PseudoFeatSelection,
    PseudoDESeq2,
    VolcanoGridByGroup,
)

from sctools.functionalanalysis import (
    PrepareULMInput,
    RunULM,
    MeltActsPadjToLong,
    BuildSignificantFeatureTable,
    SummarizeFeatureCategories,
)

from sctools.qc import (
    # Ambient RNA
    AmbientRNA,
    # Doublets
    ScoreDoublets,
    PlotScrubletScores,
    CallDoublets,
    FilterDoublets,
    # QC metrics & filtering
    CalculateQcMetrics,
    EvaluateQcThresholds,
    QCmetric,
    FilterQcCells,
    QcSummaryTable,
    # Plots
    PlotQCViolinsGrid,
    PlotQcHeatmap,
    PlotQcThresholdsGrid,
    PlotQcBoxplots,
)

__all__ = [
    # I/O
    "CatAdata",
    # Generic plots
    "PlotHeatmap",
    "PlotSignificanceHeatmap",
    # Preprocessing
    "NormalizeHvgPcaKnn",
    "CheckNormalizedLayer",
    "NormalizeLog1pFromLayer",
    "RunHighlyVariableGenes",
    "RunPcaOnHvgs",
    "RunNeighborsAndUmap",
    # Imputation
    "RunAlraOnAnnData",
    # Annotation
    "CalculateModuleScores",
    "PlotModuleScoresUMAPs",
    "CalculateAUCellWithDecoupler",
    "PlotAUCellUMAPs",
    # Integration
    "RunSeuratAnchors",
    "PlotUmap",
    "RunIntegration",
    "AttachHvgResultsToFullAdata",
    "RemoveClustersFromOriginal",
    "UpdateCellsToRemove",
    "RunScibMetricsWithLabel",
    "RunScibMetricsLabelFree",
    "BuildBenchmarkGrid",
    "RunIntegrationBenchmark",
    "SummarizeScibBenchmarkResults",
    "RunHarmonyIntegration",
    "RunScanoramaIntegration",
    "RunScviIntegration",
    "RunSeuratAnchorsIntegration",
    "RunLeidenClustering",
    # DEGs (pseudobulk PyDESeq2)
    "Pseudobulking",
    "PseudoPCA",
    "PseudoFeatSelection",
    "PseudoDESeq2",
    "VolcanoGridByGroup",
    # Functional analysis (hallmark/pathway activity, decoupler ULM)
    "PrepareULMInput",
    "RunULM",
    "MeltActsPadjToLong",
    "BuildSignificantFeatureTable",
    "SummarizeFeatureCategories",
    # Ambient RNA
    "AmbientRNA",
    # Doublets
    "ScoreDoublets",
    "PlotScrubletScores",
    "CallDoublets",
    "FilterDoublets",
    # QC metrics & filtering
    "CalculateQcMetrics",
    "EvaluateQcThresholds",
    "QCmetric",
    "FilterQcCells",
    "QcSummaryTable",
    # Plots
    "PlotQCViolinsGrid",
    "PlotQcHeatmap",
    "PlotQcThresholdsGrid",
    "PlotQcBoxplots",
]
