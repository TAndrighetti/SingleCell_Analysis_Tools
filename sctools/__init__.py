"""sctools – single-cell RNA-seq analysis utilities."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sctools")
except PackageNotFoundError:
    __version__ = "unknown"

from sctools.plots import PlotHeatmap
from sctools.io import CatAdata
from sctools.preprocessing import NormalizeHvgPcaKnn
from sctools.alra import RunAlraOnAnnData

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
    # Preprocessing
    "NormalizeHvgPcaKnn",
    # Imputation
    "RunAlraOnAnnData",
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
