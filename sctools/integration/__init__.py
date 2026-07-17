"""
sctools.integration
====================

Batch-integration methods, scIB-based benchmarking, and cell curation for
single-cell RNA-seq data.

Single production run (one method, one parameter set):
    preprocessing.NormalizeHvgPcaKnn()  # run this yourself first
    RunIntegration()
        -> methods.Run<Method>Integration()
        -> preprocessing.RunNeighborsAndUmap()
    RunLeidenClustering()  # separate step, called on RunIntegration's output
    AttachHvgResultsToFullAdata()
    RemoveClustersFromOriginal()  # iterative low-quality cluster curation
        -> UpdateCellsToRemove()  # logs the decision, returns cumulative cells

Benchmarking a grid of methods/parameters:
    BuildBenchmarkGrid()
    RunIntegrationBenchmark()
        -> preprocessing.RunHighlyVariableGenes() / RunPcaOnHvgs()
        -> methods.Run<Method>Integration()
        -> RunScibMetricsLabelFree() / RunScibMetricsWithLabel()
        -> SummarizeScibBenchmarkResults()
"""

from __future__ import annotations

from .benchmarking import (
    BuildBenchmarkGrid,
    RunIntegrationBenchmark,
    RunScibMetricsLabelFree,
    RunScibMetricsWithLabel,
    SummarizeScibBenchmarkResults,
)
from .curation import RemoveClustersFromOriginal, UpdateCellsToRemove
from .methods import (
    SCIB_CORRECTED_LAYER_BY_METHOD,
    SCIB_EMBED_BY_METHOD,
    SUPPORTED_METHODS,
    RunHarmonyIntegration,
    RunScanoramaIntegration,
    RunScviIntegration,
    RunSeuratAnchors,
    RunSeuratAnchorsIntegration,
)
from .workflow import (
    AttachHvgResultsToFullAdata,
    PlotUmap,
    RunIntegration,
    RunLeidenClustering,
)

__all__ = [
    # Method maps
    "SCIB_CORRECTED_LAYER_BY_METHOD",
    "SCIB_EMBED_BY_METHOD",
    "SUPPORTED_METHODS",
    # Methods
    "RunHarmonyIntegration",
    "RunScanoramaIntegration",
    "RunScviIntegration",
    "RunSeuratAnchorsIntegration",
    "RunSeuratAnchors",
    # Workflow
    "RunIntegration",
    "AttachHvgResultsToFullAdata",
    "RunLeidenClustering",
    "PlotUmap",
    # Benchmarking
    "BuildBenchmarkGrid",
    "RunIntegrationBenchmark",
    "RunScibMetricsWithLabel",
    "RunScibMetricsLabelFree",
    "SummarizeScibBenchmarkResults",
    # Curation
    "RemoveClustersFromOriginal",
    "UpdateCellsToRemove",
]
