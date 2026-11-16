"""
sctools.trajectory
==================

STATUS: UNDER CONSTRUCTION. NOT REVIEWED YET.
    The functions below were copied from a working notebook
    (tutoriais/ongoing/Trajectory_application_2.ipynb, Ffar2 neutrophil
    maturation). They ran on that dataset, but the code, defaults and
    statistical choices still need a careful review before use in other
    analyses. Treat the results as exploratory.

Trajectory inference and gene dynamics along pseudotime, for a known
maturation sequence (for example neutrophils G0 -> G1 -> ... -> G5c),
comparing two conditions (for example WT vs KO).

Requirements:
    Python: scanpy, rpy2.
    R: slingshot, tradeSeq, Matrix, BiocParallel, SummarizedExperiment,
    ggplot2, scales.

Workflow A: diffusion pseudotime (scanpy)
    ComputeDiffusionPaga()            -> diffusion map + PAGA topology
    PlotPagaTrajectory()              -> PAGA graph plot
    SelectRepresentativeCell()        -> root cell for sc.tl.dpt
    RunTradeSeqConditionDynamics()    -> tradeSeq fitGAM + conditionTest on DPT
    RunTradeSeqConditionTest()        -> new conditionTest (l2fc) on a fitted model
    PlotTradeSeqConditionGene()       -> tradeSeq::plotSmoothers for one gene
    SummarizeTradeSeqCurveSeparation() -> custom WT-KO curve distance (descriptive)

Workflow B: Slingshot + tradeSeq (used for the saved Ffar2 results)
    RunSlingshotTrajectory()          -> lineages, pseudotime, cell weights
    EvaluateTradeSeqK()               -> tradeSeq::evaluateK (AIC per nknots)
    RunTradeSeqConditionDynamicsSlingshotChunked()
                                      -> fitGAM in gene chunks, conditionTest
                                         with l2fc=0 and l2fc=log2(fc_threshold)
    SaveTradeSeqResultJSON() / LoadTradeSeqResultJSON()
                                      -> backup of the result dict
    SelectTradeSeqCandidates()        -> genes for visual inspection
    FitTradeSeqSelectedGenes()        -> refit selected genes, one curve per sample
    PlotTradeSeqSampleGenes()         -> PNG per gene + one PDF with all genes

Points to review:
- tradeSeq conditionTest treats each cell as a replicate. With few animals
  (for example 2 vs 2) p-values are too small (pseudoreplication). Use the
  results to rank candidates, not as genotype-level inference. The
  per-sample curves help, but they are not a test.
- SelectTradeSeqCandidates keeps genes with nominal p < effect_pvalue in the
  l2fc test (no multiple-testing correction). This is an exploratory screen.
- A gene with almost no counts in one condition (for example the knocked-out
  gene) can get NaN in conditionTest.
- EvaluateTradeSeqK and FitTradeSeqSelectedGenes build dense count matrices
  (genes x cells). This needs a lot of memory on large datasets.
- R objects are passed through ro.globalenv, so names can collide between
  calls in the same R session.
- Some defaults are dataset-specific (counts_layer="QC_filtered",
  condition_order=("W1", "W2", "K1", "K2")).
- PlotTradeSeqSampleGenes writes plot_status.csv with comma separator.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

__all__ = [
    # Workflow A: diffusion map, PAGA and DPT
    "ComputeDiffusionPaga",
    "PlotPagaTrajectory",
    "SelectRepresentativeCell",
    # Workflow A: tradeSeq on DPT pseudotime
    "RunTradeSeqConditionDynamics",
    "PlotTradeSeqConditionGene",
    "RunTradeSeqConditionTest",
    "SummarizeTradeSeqCurveSeparation",
    # Workflow B: Slingshot
    "RunSlingshotTrajectory",
    # Workflow B: tradeSeq on Slingshot pseudotime
    "EvaluateTradeSeqK",
    "RunTradeSeqConditionDynamicsSlingshotChunked",
    "SaveTradeSeqResultJSON",
    "LoadTradeSeqResultJSON",
    "SelectTradeSeqCandidates",
    "FitTradeSeqSelectedGenes",
    "PlotTradeSeqSampleGenes",
]


# ----------------------------------------------------------------------
# Workflow A: diffusion map, PAGA and DPT
# ----------------------------------------------------------------------


def ComputeDiffusionPaga(
    adata,
    group_col,
    use_rep="X_pca",
    n_neighbors=15,
    n_diffmap=15,
    initial_neighbors_key="trajectory_initial_neighbors",
    trajectory_neighbors_key="trajectory_neighbors",
    random_state=42,
):
    """
    adata : AnnData --> Annotated single-cell dataset.
    group_col : str --> Column defining groups represented as PAGA nodes.
    use_rep : str --> Initial low-dimensional representation.
    n_neighbors : int --> Number of neighbors used for graph construction.
    n_diffmap : int --> Number of diffusion components.
    initial_neighbors_key : str --> Key used for the initial neighbor graph.
    trajectory_neighbors_key : str --> Key used for the diffusion-space neighbor graph.
    random_state : int --> Random seed.
    """
    # check that the group column (PAGA nodes) exists in adata.obs
    if group_col not in adata.obs:
        raise KeyError(f"{group_col} not found in adata.obs.")

    # check that the initial low-dimensional representation (e.g. X_pca) exists in adata.obsm
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep} not found in adata.obsm.")

    # Initial cell-cell graph
    # build the initial neighbor graph from use_rep (e.g. the integrated PCA)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        key_added=initial_neighbors_key,
        random_state=random_state,
    )

    # Diffusion representation
    # compute the diffusion map from the initial neighbor graph
    sc.tl.diffmap(
        adata,
        n_comps=n_diffmap,
        neighbors_key=initial_neighbors_key,
        random_state=random_state,
    )

    # Neighborhood graph in diffusion space
    # rebuild the neighbor graph, now using the diffusion space (X_diffmap)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X_diffmap",
        key_added=trajectory_neighbors_key,
        random_state=random_state,
    )

    # PAGA topology
    # compute the PAGA topology (connectivity between groups) using the diffusion-space graph
    sc.tl.paga(
        adata,
        groups=group_col,
        neighbors_key=trajectory_neighbors_key,
    )

    # store the parameters used in this step, for traceability and reproducibility
    adata.uns["trajectory_params"] = {
        "group_col": group_col,
        "use_rep": use_rep,
        "n_neighbors": n_neighbors,
        "n_diffmap": n_diffmap,
        "initial_neighbors_key": initial_neighbors_key,
        "trajectory_neighbors_key": trajectory_neighbors_key,
        "random_state": random_state,
    }


def PlotPagaTrajectory(
    adata,
    group_col,
    threshold=0.03,
    figsize=(6, 6),
):
    """
    adata : AnnData --> Dataset containing a computed PAGA graph.
    group_col : str --> Group annotation represented by PAGA nodes.
    threshold : float --> Minimum connectivity displayed.
    figsize : tuple --> Figure size.
    """
    # create figure and axis with the given size
    fig, ax = plt.subplots(figsize=figsize)

    # plot the PAGA graph, showing only connections above the connectivity threshold
    sc.pl.paga(
        adata,
        color=group_col,
        threshold=threshold,
        frameon=False,
        ax=ax,
        show=False,
    )

    # display the figure
    plt.show()

    # return the axis, so it can be customized further outside the function
    return ax


def SelectRepresentativeCell(
    adata,
    group_col,
    group,
    rep_key="X_diffmap",
    n_components=10,
):
    """
    adata : AnnData --> Annotated single-cell dataset.
    group_col : str --> Column containing population annotations.
    group : str --> Population used for representative-cell selection.
    rep_key : str --> Low-dimensional representation used for selection.
    n_components : int --> Number of dimensions used to define the population center.
    """
    mask = adata.obs[group_col].astype(str).eq(group).to_numpy()

    cells = adata.obs_names[mask]
    X = adata.obsm[rep_key][mask]

    if len(cells) == 0:
        raise ValueError(f"No cells found for group: {group}")

    # Skip the first diffusion component
    start = 1 if rep_key == "X_diffmap" else 0
    X = X[:, start:start + n_components]

    center = np.median(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)

    return cells[np.argmin(distances)]


# ----------------------------------------------------------------------
# Workflow A: tradeSeq on DPT pseudotime
# ----------------------------------------------------------------------


def RunTradeSeqConditionDynamics(
    adata,
    counts_layer,
    pseudotime_col="dpt_pseudotime",
    condition_col="condition",
    condition_order=("WT", "KO"),
    min_cells=20,
    nknots=6,
    l2fc=0.0,
    n_cpus=4,
    seed=42,
):
    """
    Fit tradeSeq condition-specific NB-GAMs and run conditionTest.

    adata : AnnData --> Single-cell dataset containing raw counts and pseudotime.
    counts_layer : str --> Layer containing raw counts.
    pseudotime_col : str --> Column containing continuous pseudotime.
    condition_col : str --> Column containing experimental conditions.
    condition_order : tuple[str] --> Ordered condition levels.
    min_cells : int --> Minimum cells detecting a gene before GAM fitting.
    nknots : int --> Number of knots used by tradeSeq GAMs.
    n_cpus : int --> Number of CPUs used by tradeSeq.
    seed : int --> Random seed.
    l2fc : float --> Minimum log2 fold-change threshold used by conditionTest.

    Returns
    -------
    dict --> conditionTest results, fitted tradeSeq model, counts matrix and fitted genes.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # Keep cells with complete trajectory metadata
    mask = adata.obs[[pseudotime_col, condition_col]].notna().all(axis=1).to_numpy()

    X = adata.layers[counts_layer][mask]
    cells = adata.obs_names[mask].astype(str)

    # CUSTOM preprocessing: remove extremely sparse genes
    if sparse.issparse(X):
        detected = np.asarray((X > 0).sum(axis=0)).ravel()
    else:
        detected = (np.asarray(X) > 0).sum(axis=0)

    keep = detected >= min_cells

    genes = adata.var_names[keep].astype(str)
    X = X[:, keep]

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    # tradeSeq expects genes x cells
    X = X.T.tocoo()

    # Transfer the sparse count matrix to R without densifying it
    ro.globalenv["i_idx"] = ro.IntVector((X.row + 1).astype(int))
    ro.globalenv["j_idx"] = ro.IntVector((X.col + 1).astype(int))
    ro.globalenv["x_val"] = ro.FloatVector(X.data.astype(float))
    ro.globalenv["n_genes"] = int(X.shape[0])
    ro.globalenv["n_cells"] = int(X.shape[1])
    ro.globalenv["gene_names"] = ro.StrVector(genes.tolist())
    ro.globalenv["cell_names"] = ro.StrVector(cells.tolist())

    # Transfer pseudotime and experimental conditions
    pseudotime = adata.obs.loc[cells, pseudotime_col].astype(float).to_numpy()
    conditions = adata.obs.loc[cells, condition_col].astype(str).to_numpy()

    ro.globalenv["pseudotime_vec"] = ro.FloatVector(pseudotime)
    ro.globalenv["condition_vec"] = ro.StrVector(conditions)
    ro.globalenv["condition_levels"] = ro.StrVector(list(condition_order))
    ro.globalenv["nknots_run"] = int(nknots)
    ro.globalenv["n_cpus_run"] = int(n_cpus)
    ro.globalenv["seed_run"] = int(seed)
    ro.globalenv["l2fc_run"] = float(l2fc)

    ro.r(
        """
        suppressPackageStartupMessages({
            library(Matrix)
            library(tradeSeq)
            library(BiocParallel)
        })

        # Reconstruct the sparse raw-count matrix
        counts_mat <- sparseMatrix(
            i=i_idx,
            j=j_idx,
            x=x_val,
            dims=c(n_genes, n_cells)
        )

        rownames(counts_mat) <- gene_names
        colnames(counts_mat) <- cell_names

        # One pseudotime column represents one lineage
        pseudotime_mat <- matrix(pseudotime_vec, ncol=1)

        # All cells belong to the same maturation lineage
        cell_weights <- matrix(1, nrow=n_cells, ncol=1)

        # Define experimental conditions
        conditions <- factor(condition_vec, levels=condition_levels)

        set.seed(seed_run)

        bp <- MulticoreParam(workers=n_cpus_run)

        # Fit condition-specific NB-GAM smoothers
        trade_model <- fitGAM(
            counts=counts_mat,
            pseudotime=pseudotime_mat,
            cellWeights=cell_weights,
            conditions=conditions,
            nknots=nknots_run,
            verbose=FALSE,
            parallel=n_cpus_run > 1,
            BPPARAM=bp
        )

        # Test whether WT and KO fitted trajectories differ
        condition_res <- as.data.frame(
            conditionTest(
                trade_model,
                global=TRUE,
                pairwise=FALSE,
                l2fc=l2fc_run
            )
        )

        condition_res$gene <- rownames(condition_res)

        # Benjamini-Hochberg multiple-testing correction
        condition_res$padj <- p.adjust(condition_res$pvalue, method="BH")
        """
    )

    # Convert tradeSeq results back to pandas
    with localconverter(ro.default_converter + pandas2ri.converter):
        results = ro.conversion.rpy2py(ro.globalenv["condition_res"])

    results = pd.DataFrame(results)
    results = results.sort_values(["padj", "pvalue", "waldStat"], ascending=[True, True, False], na_position="last").reset_index(drop=True)

    return {
        "results": results,
        "model": ro.globalenv["trade_model"],
        "counts": ro.globalenv["counts_mat"],
        "genes": genes.tolist(),
    }


def PlotTradeSeqConditionGene(
    trade_result,
    gene,
    point_fraction=0.2,
    width=900,
    height=600,
    dpi=150,
    title=None,
):
    """
    Plot condition-specific tradeSeq smoothers for one gene.

    trade_result : dict --> Output from RunTradeSeqConditionDynamics.
    gene : str --> Gene to visualize.
    point_fraction : float --> Fraction of cells displayed as points.
    width : int --> Plot width in pixels.
    height : int --> Plot height in pixels.
    dpi : int --> Plot resolution.
    title : str | None --> Plot title; gene name if None.
    """
    from rpy2 import robjects as ro
    from rpy2.robjects.lib import grdevices
    from IPython.display import Image, display

    ro.globalenv["trade_model_plot"] = trade_result["model"]
    ro.globalenv["trade_counts_plot"] = trade_result["counts"]
    ro.globalenv["gene_plot"] = str(gene)
    ro.globalenv["point_fraction"] = float(point_fraction)
    ro.globalenv["plot_title"] = str(title if title is not None else gene)

    # Render the official tradeSeq smoother plot inside the notebook
    with grdevices.render_to_bytesio(grdevices.png, width=width, height=height, res=dpi) as img:
        ro.r(
            """
            p <- tradeSeq::plotSmoothers(
                trade_model_plot,
                counts=trade_counts_plot,
                gene=gene_plot,
                sample=point_fraction,
                border=FALSE
            ) + ggplot2::ggtitle(plot_title)

            print(p)
            """
        )

    display(Image(data=img.getvalue(), format="png"))


def RunTradeSeqConditionTest(trade_result, l2fc=0.0):
    """
    Run conditionTest on an already fitted tradeSeq model.

    trade_result : dict --> Output from RunTradeSeqConditionDynamics.
    l2fc : float --> Minimum log2 fold-change threshold used by conditionTest.

    Returns
    -------
    DataFrame --> conditionTest results with BH-adjusted p-values.
    """
    import pandas as pd
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    ro.globalenv["trade_model_test"] = trade_result["model"]
    ro.globalenv["l2fc_test"] = float(l2fc)

    # Run only the official conditionTest on the already fitted model
    ro.r(
        """
        condition_res_test <- as.data.frame(
            tradeSeq::conditionTest(
                trade_model_test,
                global=TRUE,
                pairwise=FALSE,
                l2fc=l2fc_test
            )
        )

        condition_res_test$gene <- rownames(condition_res_test)
        condition_res_test$padj <- p.adjust(condition_res_test$pvalue, method="BH")
        """
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        results = ro.conversion.rpy2py(ro.globalenv["condition_res_test"])

    results = pd.DataFrame(results)
    results = results.sort_values(["padj", "pvalue", "waldStat"], ascending=[True, True, False], na_position="last").reset_index(drop=True)

    return results


def SummarizeTradeSeqCurveSeparation(
    trade_result,
    genes,
    reference="WT",
    comparison="KO",
    n_points=100,
    batch_size=100,
):
    """
    CUSTOM descriptive summary of WT-KO fitted-curve separation.

    trade_result : dict --> Output from RunTradeSeqConditionDynamics.
    genes : list[str] --> Genes whose fitted curves will be compared.
    reference : str --> Reference condition.
    comparison : str --> Comparison condition.
    n_points : int --> Number of pseudotime points used for curve comparison.
    batch_size : int --> Number of genes processed per predictSmooth call.

    Returns
    -------
    DataFrame --> Custom descriptive metrics of fitted-curve separation.
    """
    import numpy as np
    import pandas as pd
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    genes = list(genes)
    ro.globalenv["trade_model_curve"] = trade_result["model"]
    ro.globalenv["n_points_curve"] = int(n_points)

    all_smooth = []

    # CUSTOM computational safeguard: process genes in small batches
    for start in range(0, len(genes), batch_size):
        batch = genes[start:start + batch_size]
        ro.globalenv["curve_genes"] = ro.StrVector(batch)

        # Official tradeSeq smoother prediction
        ro.r(
            """
            smooth_batch <- tradeSeq::predictSmooth(
                models=trade_model_curve,
                gene=curve_genes,
                nPoints=n_points_curve,
                tidy=TRUE
            )
            """
        )

        with localconverter(ro.default_converter + pandas2ri.converter):
            smooth_batch = ro.conversion.rpy2py(ro.globalenv["smooth_batch"])

        all_smooth.append(pd.DataFrame(smooth_batch))

    smooth_df = pd.concat(all_smooth, ignore_index=True)
    summaries = []

    # CUSTOM: compare WT and KO curves gene by gene
    for (gene, lineage), group in smooth_df.groupby(["gene", "lineage"], observed=True):
        ref = group[group["condition"] == reference].sort_values("time")
        cmp = group[group["condition"] == comparison].sort_values("time")

        if ref.empty or cmp.empty:
            continue

        # CUSTOM: restrict comparison to shared pseudotime
        overlap_start = max(ref["time"].min(), cmp["time"].min())
        overlap_end = min(ref["time"].max(), cmp["time"].max())

        if overlap_end <= overlap_start:
            continue

        # CUSTOM: interpolate both curves onto exactly the same grid
        grid = np.linspace(overlap_start, overlap_end, n_points)
        ref_values = np.interp(grid, ref["time"], ref["yhat"])
        cmp_values = np.interp(grid, cmp["time"], cmp["yhat"])

        # CUSTOM: use the same log(expression + 1) scale as plotSmoothers
        ref_values = np.log1p(ref_values)
        cmp_values = np.log1p(cmp_values)

        difference = cmp_values - ref_values

        # CUSTOM descriptive effect-size metrics
        summaries.append({
            "gene": gene,
            "lineage": lineage,
            "mean_difference": np.mean(difference),
            "mean_abs_difference": np.mean(np.abs(difference)),
            "max_abs_difference": np.max(np.abs(difference)),
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
        })

    result = pd.DataFrame(summaries)

    if not result.empty:
        result["reference"] = reference
        result["comparison"] = comparison
        result["scale"] = "log(expression + 1)"

    return result


# ----------------------------------------------------------------------
# Workflow B: Slingshot
# ----------------------------------------------------------------------


def RunSlingshotTrajectory(
    adata,
    stage_col,
    start_stage=None,
    end_stages=None,
    rep_key="X_pca",
    n_components=None,
):
    """
    Infer Slingshot lineages and lineage-specific pseudotimes.

    adata : AnnData --> Cells used for trajectory inference.
    stage_col : str --> Column containing cluster or state labels.
    start_stage : str | None --> Known starting cluster.
    end_stages : tuple[str] | list[str] | None --> Optional terminal clusters.
    rep_key : str --> Reduced-dimensional representation used by Slingshot.
    n_components : int | None --> Number of dimensions from rep_key to use.

    Returns
    -------
    dict --> Slingshot model, lineage paths, pseudotimes and lineage weights.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    if stage_col not in adata.obs:
        raise KeyError(f"'{stage_col}' not found in adata.obs.")

    if rep_key not in adata.obsm:
        raise KeyError(f"'{rep_key}' not found in adata.obsm.")

    cells = adata.obs_names.astype(str)
    stages = adata.obs[stage_col].astype(str)

    reduced = adata.obsm[rep_key]

    if sparse.issparse(reduced):
        reduced = reduced.toarray()

    reduced = np.asarray(reduced, dtype=float)

    if n_components is not None:
        reduced = reduced[:, :n_components]

    present_stages = set(stages.unique())

    if start_stage is not None and start_stage not in present_stages:
        raise ValueError(f"Start stage '{start_stage}' is not present.")

    end_stages = tuple(end_stages or ())

    missing_ends = set(end_stages) - present_stages

    if missing_ends:
        raise ValueError(f"Terminal stages not present: {sorted(missing_ends)}")

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["reduced_dim_py"] = ro.conversion.py2rpy(reduced)

    ro.globalenv["cell_names_py"] = ro.StrVector(cells.tolist())
    ro.globalenv["stage_vec_py"] = ro.StrVector(stages.tolist())
    ro.globalenv["use_start_py"] = bool(start_stage is not None)
    ro.globalenv["start_stage_py"] = ro.StrVector([start_stage] if start_stage is not None else [])
    ro.globalenv["end_stages_py"] = ro.StrVector(list(end_stages))

    ro.r("""
        suppressPackageStartupMessages(library(slingshot))

        reduced_dim <- as.matrix(reduced_dim_py)
        rownames(reduced_dim) <- cell_names_py

        stage_labels <- factor(stage_vec_py)

        start_arg <- if (use_start_py) start_stage_py[1] else NULL
        end_arg <- if (length(end_stages_py) > 0) end_stages_py else NULL

        sling_model <- slingshot(
            reduced_dim,
            clusterLabels=stage_labels,
            start.clus=start_arg,
            end.clus=end_arg
        )

        # NA outside a lineage: preferable for descriptive visualization
        sling_pt_plot <- slingPseudotime(
            sling_model,
            na=TRUE
        )

        # Full arclength matrix: preferable downstream together with weights
        sling_pt_trade <- slingPseudotime(
            sling_model,
            na=FALSE
        )

        sling_weights <- slingCurveWeights(sling_model)
    """)

    with localconverter(ro.default_converter + numpy2ri.converter):
        pt_plot_array = np.asarray(
            ro.conversion.rpy2py(ro.globalenv["sling_pt_plot"])
        )
        pt_trade_array = np.asarray(
            ro.conversion.rpy2py(ro.globalenv["sling_pt_trade"])
        )
        weight_array = np.asarray(
            ro.conversion.rpy2py(ro.globalenv["sling_weights"])
        )

    sling_cells = pd.Index(
        [str(x) for x in ro.r("rownames(sling_pt_trade)")]
    )

    lineages_r = ro.r("slingLineages(sling_model)")

    lineage_paths = [
        [str(x) for x in lineage]
        for lineage in lineages_r
    ]

    lineage_strings = [
        " -> ".join(path)
        for path in lineage_paths
    ]

    lineage_names = [
        f"Lineage_{i + 1}_{path[-1]}"
        for i, path in enumerate(lineage_paths)
    ]

    pseudotime_plot = pd.DataFrame(
        pt_plot_array,
        index=sling_cells,
        columns=lineage_names,
    ).reindex(cells)

    pseudotime_trade = pd.DataFrame(
        pt_trade_array,
        index=sling_cells,
        columns=lineage_names,
    ).reindex(cells)

    weights = pd.DataFrame(
        weight_array,
        index=sling_cells,
        columns=lineage_names,
    ).reindex(cells)

    if pseudotime_trade.isna().any().any():
        raise ValueError("TradeSeq pseudotime contains NA values.")

    if weights.isna().any().any():
        raise ValueError("Slingshot weights contain NA values.")

    if not np.isfinite(pseudotime_trade.to_numpy()).all():
        raise ValueError("Slingshot pseudotime contains non-finite values.")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Slingshot weights contain non-finite values.")

    return {
        "model": ro.globalenv["sling_model"],
        "lineages": lineage_strings,
        "lineage_names": lineage_names,
        "pseudotime_plot": pseudotime_plot,
        "pseudotime": pseudotime_trade,
        "weights": weights,
        "representation": rep_key,
        "n_components": n_components,
    }


# ----------------------------------------------------------------------
# Workflow B: tradeSeq on Slingshot pseudotime
# ----------------------------------------------------------------------


def EvaluateTradeSeqK(
    adata,
    slingshot_result,
    counts_layer="QC_filtered",
    k_values=range(3, 10),
    n_genes=500,
    seed=42,
):
    """
    Evaluate candidate numbers of knots for tradeSeq.

    adata : AnnData --> Cells used in the Slingshot trajectory.
    slingshot_result : dict --> Output from RunSlingshotTrajectory.
    counts_layer : str --> Layer containing raw counts.
    k_values : iterable --> Candidate numbers of knots.
    n_genes : int --> Number of genes sampled by evaluateK.
    seed : int --> Random seed.

    Returns
    -------
    DataFrame --> Gene-wise AIC values for each candidate number of knots.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    counts = adata.layers[counts_layer]

    if sparse.issparse(counts):
        counts = counts.toarray()

    counts = np.asarray(counts).T  # genes × cells

    pseudotime = slingshot_result["pseudotime"].loc[
        adata.obs_names
    ].to_numpy()

    weights = slingshot_result["weights"].loc[
        adata.obs_names
    ].to_numpy()

    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["counts_py"] = ro.conversion.py2rpy(counts)
        ro.globalenv["pseudotime_py"] = ro.conversion.py2rpy(pseudotime)
        ro.globalenv["weights_py"] = ro.conversion.py2rpy(weights)

    ro.globalenv["gene_names_py"] = ro.StrVector(
        adata.var_names.astype(str).tolist()
    )

    ro.globalenv["k_values_py"] = ro.IntVector(list(k_values))
    ro.globalenv["n_genes_py"] = int(n_genes)
    ro.globalenv["seed_py"] = int(seed)

    ro.r("""
        suppressPackageStartupMessages(library(tradeSeq))

        counts_r <- as.matrix(counts_py)
        rownames(counts_r) <- gene_names_py

        set.seed(seed_py)

        aic_matrix <- evaluateK(
            counts=counts_r,
            pseudotime=as.matrix(pseudotime_py),
            cellWeights=as.matrix(weights_py),
            k=k_values_py,
            nGenes=n_genes_py,
            plot=FALSE,
            verbose=TRUE
        )
    """)

    with localconverter(ro.default_converter + numpy2ri.converter):
        aic = np.asarray(
            ro.conversion.rpy2py(ro.globalenv["aic_matrix"])
        )

    genes = [str(x) for x in ro.r("rownames(aic_matrix)")]

    return pd.DataFrame(
        aic,
        index=genes,
        columns=list(k_values),
    )


def RunTradeSeqConditionDynamicsSlingshotChunked(
    adata,
    slingshot_result,
    counts_layer,
    condition_col="condition",
    condition_order=("WT", "KO"),
    min_cells=20,
    min_pct_cells=None,
    force_genes=(),
    nknots=6,
    chunk_size=250,
    fc_threshold=1.5,
    seed=42,
):
    """
    Fit condition-specific tradeSeq NB-GAMs in memory-efficient gene chunks.

    adata : AnnData --> Cells used for the Slingshot trajectory.
    slingshot_result : dict --> Output from RunSlingshotTrajectory.
    counts_layer : str --> Layer containing raw counts.
    condition_col : str --> Column containing experimental conditions.
    condition_order : tuple[str] --> Ordered condition levels.
    min_cells : int --> Minimum number of cells detecting a gene.
    min_pct_cells : float | None --> Optional minimum detected-cell fraction.
    force_genes : tuple[str] --> Genes retained regardless of expression filter.
    nknots : int --> Number of knots selected using EvaluateTradeSeqK.
    chunk_size : int --> Number of gene-specific GAMs fitted per chunk.
    fc_threshold : float --> Fold-change threshold for the second conditionTest.
    seed : int --> Random seed.

    Returns
    -------
    dict --> Genome-wide conditionTest results and trajectory inputs.
    """
    import gc
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter

    if counts_layer not in adata.layers:
        raise KeyError(f"Layer '{counts_layer}' not found in adata.layers.")

    if condition_col not in adata.obs:
        raise KeyError(f"Column '{condition_col}' not found in adata.obs.")

    cells = adata.obs_names.astype(str)

    # ------------------------------------------------------------
    # 1. Retrieve Slingshot pseudotime and lineage weights
    # ------------------------------------------------------------
    pseudotime = slingshot_result["pseudotime"].reindex(cells)
    weights = slingshot_result["weights"].reindex(cells)

    if pseudotime.isna().any().any():
        raise ValueError("Slingshot pseudotime contains NA values.")

    if weights.isna().any().any():
        raise ValueError("Slingshot lineage weights contain NA values.")

    if not np.isfinite(pseudotime.to_numpy()).all():
        raise ValueError("Slingshot pseudotime contains non-finite values.")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Slingshot lineage weights contain non-finite values.")

    # ------------------------------------------------------------
    # 2. Filter extremely sparse genes
    # ------------------------------------------------------------
    X_all = adata.layers[counts_layer]

    if sparse.issparse(X_all):
        detected = np.asarray((X_all > 0).sum(axis=0)).ravel()
    else:
        detected = (np.asarray(X_all) > 0).sum(axis=0)

    min_detected = min_cells

    if min_pct_cells is not None:
        min_detected = max(
            min_cells,
            int(np.ceil(min_pct_cells * adata.n_obs)),
        )

    keep = detected >= min_detected

    for gene in force_genes:
        if gene in adata.var_names:
            keep[adata.var_names.get_loc(gene)] = True

    gene_idx = np.flatnonzero(keep)
    genes = adata.var_names[gene_idx].astype(str)

    if len(genes) == 0:
        raise ValueError("No genes passed the expression filter.")

    # ------------------------------------------------------------
    # 3. Validate conditions
    # ------------------------------------------------------------
    conditions = adata.obs.loc[cells, condition_col].astype(str)

    unexpected = set(conditions.unique()) - set(condition_order)

    if unexpected:
        raise ValueError(
            f"Conditions not included in condition_order: {sorted(unexpected)}"
        )

    missing = set(condition_order) - set(conditions.unique())

    if missing:
        raise ValueError(
            f"Conditions not present in the dataset: {sorted(missing)}"
        )

    n_chunks = int(np.ceil(len(genes) / chunk_size))

    print(f"Cells: {adata.n_obs}")
    print(f"Genes retained: {len(genes)}")
    print(f"Minimum detected cells: {min_detected}")
    print(f"Knots: {nknots}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunks: {n_chunks}")

    # ------------------------------------------------------------
    # 4. Transfer complete filtered count matrix once
    # ------------------------------------------------------------
    X = X_all[:, gene_idx]

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    X = X.T.tocoo()  # genes x cells

    ro.globalenv["i_full"] = ro.IntVector(
        (X.row + 1).astype(int)
    )
    ro.globalenv["j_full"] = ro.IntVector(
        (X.col + 1).astype(int)
    )
    ro.globalenv["x_full"] = ro.FloatVector(
        X.data.astype(float)
    )

    ro.globalenv["n_genes"] = int(X.shape[0])
    ro.globalenv["n_cells"] = int(X.shape[1])

    ro.globalenv["gene_names"] = ro.StrVector(
        genes.tolist()
    )
    ro.globalenv["cell_names"] = ro.StrVector(
        cells.tolist()
    )

    with localconverter(
        ro.default_converter + numpy2ri.converter
    ):
        ro.globalenv["pseudotime_mat"] = ro.conversion.py2rpy(
            pseudotime.to_numpy(dtype=float)
        )

        ro.globalenv["cell_weights"] = ro.conversion.py2rpy(
            weights.to_numpy(dtype=float)
        )

    ro.globalenv["condition_vec"] = ro.StrVector(
        conditions.tolist()
    )
    ro.globalenv["condition_levels"] = ro.StrVector(
        list(condition_order)
    )

    ro.globalenv["nknots_run"] = int(nknots)
    ro.globalenv["seed_run"] = int(seed)
    ro.globalenv["l2fc_run"] = float(
        np.log2(fc_threshold)
    )

    ro.r("""
        suppressPackageStartupMessages({
            library(Matrix)
            library(tradeSeq)
            library(SummarizedExperiment)
        })

        counts_full <- sparseMatrix(
            i=i_full,
            j=j_full,
            x=x_full,
            dims=c(n_genes, n_cells)
        )

        rownames(counts_full) <- gene_names
        colnames(counts_full) <- cell_names

        rownames(pseudotime_mat) <- cell_names
        rownames(cell_weights) <- cell_names

        conditions <- factor(
            condition_vec,
            levels=condition_levels
        )

        rm(i_full, j_full, x_full)
        gc()
    """)

    del X
    gc.collect()

    results_standard = []
    results_fc = []

    # ------------------------------------------------------------
    # 5. Fit gene-specific GAMs in sequential blocks
    # ------------------------------------------------------------
    for chunk_number, start in enumerate(
        range(0, len(genes), chunk_size),
        1,
    ):
        stop = min(
            start + chunk_size,
            len(genes),
        )

        # R uses one-based row indices
        genes_run = np.arange(
            start + 1,
            stop + 1,
            dtype=int,
        )

        ro.globalenv["genes_run"] = ro.IntVector(
            genes_run.tolist()
        )

        print(
            f"Chunk {chunk_number}/{n_chunks}: "
            f"genes {start + 1}-{stop}"
        )

        ro.r("""
            set.seed(seed_run)

            trade_model_chunk <- fitGAM(
                counts=counts_full,
                genes=genes_run,
                pseudotime=pseudotime_mat,
                cellWeights=cell_weights,
                conditions=conditions,
                nknots=nknots_run,
                verbose=FALSE,
                parallel=FALSE
            )

            condition_chunk <- as.data.frame(
                conditionTest(
                    trade_model_chunk,
                    global=TRUE,
                    pairwise=FALSE,
                    l2fc=0
                )
            )

            condition_fc_chunk <- as.data.frame(
                conditionTest(
                    trade_model_chunk,
                    global=TRUE,
                    pairwise=FALSE,
                    l2fc=l2fc_run
                )
            )

            condition_chunk$gene <- rownames(
                condition_chunk
            )

            condition_fc_chunk$gene <- rownames(
                condition_fc_chunk
            )

            converged_chunk <- rowData(
                trade_model_chunk
            )$tradeSeq$converged

            names(converged_chunk) <- rownames(
                trade_model_chunk
            )

            condition_chunk$converged <- converged_chunk[
                match(
                    condition_chunk$gene,
                    names(converged_chunk)
                )
            ]

            condition_fc_chunk$converged <- converged_chunk[
                match(
                    condition_fc_chunk$gene,
                    names(converged_chunk)
                )
            ]
        """)

        with localconverter(
            ro.default_converter + pandas2ri.converter
        ):
            res = pd.DataFrame(
                ro.conversion.rpy2py(
                    ro.globalenv["condition_chunk"]
                )
            )

            res_fc = pd.DataFrame(
                ro.conversion.rpy2py(
                    ro.globalenv["condition_fc_chunk"]
                )
            )

        results_standard.append(res)
        results_fc.append(res_fc)

        ro.r("""
            rm(
                trade_model_chunk,
                condition_chunk,
                condition_fc_chunk,
                converged_chunk,
                genes_run
            )

            gc()
        """)

        del res, res_fc
        gc.collect()

    # ------------------------------------------------------------
    # 6. Combine all chunks
    # ------------------------------------------------------------
    results = pd.concat(
        results_standard,
        ignore_index=True,
    )

    results_fc = pd.concat(
        results_fc,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # 7. Apply BH globally across all tested genes
    # ------------------------------------------------------------
    def AddBH(df):
        pvalues = [
            ro.NA_Real
            if not np.isfinite(x)
            else float(x)
            for x in df["pvalue"].to_numpy(dtype=float)
        ]

        ro.globalenv["pvalues_tmp"] = ro.FloatVector(
            pvalues
        )

        padj = np.asarray(
            ro.r(
                'p.adjust(pvalues_tmp, method="BH")'
            ),
            dtype=float,
        )

        out = df.copy()
        out["padj"] = padj

        return out

    results = AddBH(results)
    results_fc = AddBH(results_fc)

    results = results.sort_values(
        ["padj", "pvalue", "waldStat"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    results_fc = results_fc.sort_values(
        ["padj", "pvalue", "waldStat"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 8. Clean large R object
    # ------------------------------------------------------------
    ro.r("""
        rm(counts_full)
        gc()
    """)

    gc.collect()

    return {
        "results": results,
        "results_fc": results_fc,
        "genes": genes.tolist(),
        "pseudotime": pseudotime,
        "weights": weights,
        "lineages": slingshot_result["lineages"],
        "lineage_names": slingshot_result["lineage_names"],
        "condition_order": list(condition_order),
        "fc_threshold": fc_threshold,
        "nknots": nknots,
        "chunk_size": chunk_size,
    }


def SaveTradeSeqResultJSON(trade_result, filename):
    """Save tradeSeq results to a portable JSON file."""

    def SerializeValue(value):
        if isinstance(value, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "data": value.to_dict(orient="split"),
            }
        if isinstance(value, pd.Series):
            return {
                "__type__": "Series",
                "data": value.to_dict(),
                "name": value.name,
            }
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": value.tolist(),
            }
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "data": list(value),
            }
        return value

    output = {
        key: SerializeValue(value)
        for key, value in trade_result.items()
    }

    with open(filename, "w") as f:
        json.dump(output, f)

    print(f"Saved: {filename}")


def LoadTradeSeqResultJSON(filename):
    """Load tradeSeq results previously saved as JSON."""

    with open(filename) as f:
        data = json.load(f)

    result = {}

    for key, value in data.items():

        if isinstance(value, dict) and value.get("__type__") == "DataFrame":
            d = value["data"]
            result[key] = pd.DataFrame(
                d["data"],
                index=d["index"],
                columns=d["columns"],
            )

        elif isinstance(value, dict) and value.get("__type__") == "Series":
            result[key] = pd.Series(
                value["data"],
                name=value["name"],
            )

        elif isinstance(value, dict) and value.get("__type__") == "ndarray":
            result[key] = np.asarray(value["data"])

        elif isinstance(value, dict) and value.get("__type__") == "tuple":
            result[key] = tuple(value["data"])

        else:
            result[key] = value

    return result


def SelectTradeSeqCandidates(
    trade_result,
    fdr=0.05,
    effect_pvalue=0.05,
    include_genes=None,
):
    """Select genes for visual inspection from tradeSeq condition tests."""
    import numpy as np
    import pandas as pd

    standard = trade_result["results"].copy()
    effect = trade_result["results_fc"].copy()

    standard = standard.rename(
        columns={
            "waldStat": "wald_standard",
            "df": "df_standard",
            "pvalue": "pvalue_standard",
            "padj": "padj_standard",
            "converged": "converged_standard",
        }
    )

    effect = effect.rename(
        columns={
            "waldStat": "wald_effect",
            "df": "df_effect",
            "pvalue": "pvalue_effect",
            "padj": "padj_effect",
            "converged": "converged_effect",
        }
    )

    cols_standard = [
        c for c in [
            "gene", "wald_standard", "df_standard",
            "pvalue_standard", "padj_standard",
            "converged_standard",
        ]
        if c in standard.columns
    ]

    cols_effect = [
        c for c in [
            "gene", "wald_effect", "df_effect",
            "pvalue_effect", "padj_effect",
            "converged_effect",
        ]
        if c in effect.columns
    ]

    table = standard[cols_standard].merge(
        effect[cols_effect],
        on="gene",
        how="outer",
    )

    # Remove failed models
    valid = (
        np.isfinite(table["pvalue_standard"])
        & np.isfinite(table["pvalue_effect"])
    )

    if "converged_standard" in table:
        valid &= table["converged_standard"].fillna(False)

    if "converged_effect" in table:
        valid &= table["converged_effect"].fillna(False)

    table["FDR_significant"] = table["padj_standard"] < fdr
    table["effect_candidate"] = table["pvalue_effect"] < effect_pvalue

    table["selected"] = (
        valid
        & (
            table["FDR_significant"]
            | table["effect_candidate"]
        )
    )

    # Optional genes of biological interest
    include_genes = include_genes or []

    table["included_manually"] = table["gene"].isin(include_genes)
    table["selected"] |= table["included_manually"]

    candidates = (
        table[table["selected"]]
        .sort_values(
            ["pvalue_effect", "wald_effect"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    return candidates


def FitTradeSeqSelectedGenes(
    adata,
    trade_result,
    genes,
    counts_layer="QC_filtered",
    condition_col="sample",
    condition_order=("W1", "W2", "K1", "K2"),
    seed=42,
):
    """
    Fit tradeSeq smoothers for selected genes for visualization.

    adata : AnnData --> AnnData used for the trajectory analysis.
    trade_result : dict --> Output from the genome-wide tradeSeq analysis.
    genes : list[str] --> Genes to fit and visualize.
    counts_layer : str --> Layer containing raw counts.
    condition_col : str --> Column defining curves to fit.
    condition_order : tuple[str] --> Order of conditions/samples in the model.
    seed : int --> Random seed.

    Returns
    -------
    dict --> tradeSeq model and information required for plotting.
    """
    import numpy as np
    from scipy import sparse
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter

    if counts_layer not in adata.layers:
        raise KeyError(f"Layer '{counts_layer}' not found.")

    if condition_col not in adata.obs:
        raise KeyError(f"Column '{condition_col}' not found in adata.obs.")

    # Keep the same genes used as normalization context in the genome-wide fit
    normalization_genes = [
        g for g in trade_result["genes"]
        if g in adata.var_names
    ]

    genes = list(dict.fromkeys(genes))

    missing = [
        g for g in genes
        if g not in normalization_genes
    ]

    if missing:
        raise ValueError(
            f"{len(missing)} selected genes were not present in the "
            f"genome-wide tradeSeq analysis: {missing[:10]}"
        )

    cells = adata.obs_names.astype(str)

    # Full count matrix: genes × cells
    counts = adata[:, normalization_genes].layers[counts_layer]

    if sparse.issparse(counts):
        counts = counts.toarray()

    counts = np.asarray(counts).T

    # Reuse exactly the same Slingshot pseudotime and weights
    pseudotime = (
        trade_result["pseudotime"]
        .loc[cells]
        .to_numpy()
    )

    weights = (
        trade_result["weights"]
        .loc[cells]
        .to_numpy()
    )

    conditions = (
        adata.obs.loc[cells, condition_col]
        .astype(str)
        .to_numpy()
    )

    unexpected = set(conditions) - set(condition_order)

    if unexpected:
        raise ValueError(
            f"Unexpected values in '{condition_col}': {sorted(unexpected)}"
        )

    nknots = int(trade_result["nknots"])

    # Transfer data to R
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv["counts_full_py"] = ro.conversion.py2rpy(counts)
        ro.globalenv["pseudotime_py"] = ro.conversion.py2rpy(pseudotime)
        ro.globalenv["weights_py"] = ro.conversion.py2rpy(weights)

    ro.globalenv["gene_names_py"] = ro.StrVector(normalization_genes)
    ro.globalenv["cell_names_py"] = ro.StrVector(cells.tolist())
    ro.globalenv["selected_genes_py"] = ro.StrVector(genes)
    ro.globalenv["condition_py"] = ro.StrVector(conditions.tolist())
    ro.globalenv["condition_order_py"] = ro.StrVector(list(condition_order))
    ro.globalenv["nknots_py"] = nknots
    ro.globalenv["seed_py"] = int(seed)

    ro.r("""
        suppressPackageStartupMessages(library(tradeSeq))

        set.seed(seed_py)

        counts_full <- as.matrix(counts_full_py)
        rownames(counts_full) <- gene_names_py
        colnames(counts_full) <- cell_names_py

        pseudotime_r <- as.matrix(pseudotime_py)
        weights_r <- as.matrix(weights_py)

        conditions_r <- factor(
            condition_py,
            levels=condition_order_py
        )

        selected_idx <- match(
            selected_genes_py,
            rownames(counts_full)
        )

        if (any(is.na(selected_idx))) {
            stop("Some selected genes were not found in counts_full.")
        }

        # Normalization uses the full matrix; GAMs are fitted only
        # for the selected genes.
        plotfit_model <- fitGAM(
            counts=counts_full,
            genes=selected_idx,
            pseudotime=pseudotime_r,
            cellWeights=weights_r,
            conditions=conditions_r,
            nknots=nknots_py,
            parallel=FALSE,
            verbose=TRUE,
            sce=TRUE
        )

        plot_counts <- counts_full[
            selected_idx,
            ,
            drop=FALSE
        ]
    """)

    return {
        "model": ro.globalenv["plotfit_model"],
        "counts": ro.globalenv["plot_counts"],
        "genes": genes,
        "pseudotime": trade_result["pseudotime"].loc[cells].copy(),
        "weights": trade_result["weights"].loc[cells].copy(),
        "samples": conditions,
        "condition_order": tuple(condition_order),
        "nknots": nknots,
    }


def PlotTradeSeqSampleGenes(
    trade_plots,
    genes=None,
    output_dir="tradeSeq_sample_curves",
    n_points=200,
    point_fraction=0.15,
    dpi=300,
):
    """
    Plot and save sample-specific tradeSeq smoothers.

    trade_plots : dict --> Output from FitTradeSeqSelectedGenes.
    genes : list[str] | None --> Genes to plot; defaults to all fitted genes.
    output_dir : str --> Directory where plots are saved.
    n_points : int --> Number of points used to draw each smoother.
    point_fraction : float --> Fraction of cells displayed as points.
    dpi : int --> PNG resolution.

    Returns
    -------
    DataFrame --> Plotting status for each gene.
    """
    import os
    import pandas as pd
    from rpy2 import robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    genes = genes or trade_plots["genes"]
    genes = list(dict.fromkeys(genes))

    os.makedirs(output_dir, exist_ok=True)

    ro.globalenv["plot_models"] = trade_plots["model"]
    ro.globalenv["plot_counts"] = trade_plots["counts"]
    ro.globalenv["genes_to_plot"] = ro.StrVector(genes)
    ro.globalenv["output_dir_py"] = output_dir
    ro.globalenv["n_points_py"] = int(n_points)
    ro.globalenv["point_fraction_py"] = float(point_fraction)
    ro.globalenv["dpi_py"] = int(dpi)

    ro.r("""
        suppressPackageStartupMessages({
            library(tradeSeq)
            library(ggplot2)
            library(scales)
        })

        sample_labels <- colData(plot_models)$tradeSeq$conditions
        sample_levels <- levels(sample_labels)

        sample_cols <- setNames(
            scales::hue_pal()(length(sample_levels)),
            sample_levels
        )

        curve_cols <- unname(sample_cols[sample_levels])

        plot_status <- data.frame(
            gene=character(),
            status=character(),
            message=character(),
            stringsAsFactors=FALSE
        )

        pdf_file <- file.path(
            output_dir_py,
            "all_sample_curves.pdf"
        )

        grDevices::pdf(
            pdf_file,
            width=8,
            height=5.5
        )

        for (gene in genes_to_plot) {

            result <- tryCatch({

                p <- plotSmoothers(
                    models=plot_models,
                    counts=plot_counts,
                    gene=gene,
                    nPoints=n_points_py,
                    lwd=2.2,
                    size=0.4,
                    alpha=0.12,
                    sample=point_fraction_py,
                    pointCol=sample_labels,
                    curvesCols=curve_cols,
                    border=FALSE
                )

                # Dummy lines used only to generate a clear line-based legend
                legend_data <- data.frame(
                    x=rep(c(0, 1), length(sample_levels)),
                    y=NA_real_,
                    Sample=rep(sample_levels, each=2)
                )

                p <- suppressMessages(
                    p +
                    geom_line(
                        data=legend_data,
                        aes(
                            x=x,
                            y=y,
                            color=Sample,
                            group=Sample
                        ),
                        linewidth=2.5,
                        inherit.aes=FALSE,
                        show.legend=TRUE
                    ) +
                    scale_color_manual(
                        values=sample_cols,
                        breaks=sample_levels,
                        name="Sample"
                    ) +
                    guides(
                        color=guide_legend(
                            override.aes=list(
                                shape=NA,
                                alpha=1,
                                linewidth=3
                            )
                        )
                    ) +
                    labs(
                        title=gene,
                        x="Slingshot pseudotime",
                        y="Log(expression + 1)"
                    ) +
                    theme_classic(base_size=13) +
                    theme(
                        plot.title=element_text(
                            face="bold",
                            size=18,
                            hjust=0.5
                        ),
                        axis.title=element_text(size=14),
                        axis.text=element_text(size=11),
                        legend.title=element_text(
                            size=14,
                            face="bold"
                        ),
                        legend.text=element_text(size=13),
                        legend.key.width=unit(1.5, "cm"),
                        legend.key.height=unit(0.6, "cm"),
                        legend.position="right"
                    )
                )

                print(p)

                safe_gene <- gsub(
                    "[^A-Za-z0-9_.-]",
                    "_",
                    gene
                )

                ggsave(
                    filename=file.path(
                        output_dir_py,
                        paste0(safe_gene, ".png")
                    ),
                    plot=p,
                    width=8,
                    height=5.5,
                    dpi=dpi_py,
                    bg="white"
                )

                c("OK", "")

            }, error=function(e) {
                c("FAILED", conditionMessage(e))
            })

            plot_status <- rbind(
                plot_status,
                data.frame(
                    gene=gene,
                    status=result[1],
                    message=result[2],
                    stringsAsFactors=FALSE
                )
            )
        }

        grDevices::dev.off()

        write.csv(
            plot_status,
            file.path(output_dir_py, "plot_status.csv"),
            row.names=FALSE
        )

        plot_status_r <- plot_status
    """)

    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):
        status = ro.conversion.rpy2py(
            ro.globalenv["plot_status_r"]
        )

    print(f"Plots saved to: {output_dir}")
    print(f"Successful: {(status['status'] == 'OK').sum()}")
    print(f"Failed: {(status['status'] == 'FAILED').sum()}")

    return status
