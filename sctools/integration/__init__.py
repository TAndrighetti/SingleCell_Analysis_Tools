"""
sctools.integration
====================

Batch-integration methods, scIB-based benchmarking, and cell curation for
single-cell RNA-seq data.

Single production run (one method, one parameter set):
    RunIntegrationComplete()
        -> preprocessing.NormalizeHvgPcaKnn()
        -> methods.Run<Method>Integration()
        -> workflow.RunLeidenClustering()
    AttachHvgResultsToFullAdata()
    UpdateCellsToRemove()  # iterative low-quality cluster curation

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
from .curation import UpdateCellsToRemove
from .methods import (
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
    RunIntegrationComplete,
    RunLeidenClustering,
)

__all__ = [
    # Method maps
    "SCIB_EMBED_BY_METHOD",
    "SUPPORTED_METHODS",
    # Methods
    "RunHarmonyIntegration",
    "RunScanoramaIntegration",
    "RunScviIntegration",
    "RunSeuratAnchorsIntegration",
    "RunSeuratAnchors",
    # Workflow
    "RunIntegrationComplete",
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
    "UpdateCellsToRemove",
]
