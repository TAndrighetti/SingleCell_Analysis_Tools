"""
sctools.mast
============

Single-cell differential expression with MAST (hurdle model, Finak et al.
2015), called through rpy2 (requires R + the `MAST` package; `lme4` too
for `model="glmer"`).

This module does NOT reimplement MAST. It calls `MAST::zlm()` and the
likelihood-ratio test from `summary(..., doLRT=)`, the same way as the MAST
vignettes and the sc-best-practices DGE chapter.

Main workflow:
    RunMAST()             -> one contrast, one celltype (or all cells)
    RunMASTByCelltype()   -> loop over celltypes, returns one long table

Model choice (there is no default on purpose, the user must decide):
    model="glm"    ~ ngeneson + covariates + condition
        Every cell is treated as an independent replicate. With several
        samples per condition, p-values are too small (pseudoreplication,
        Squair et al. 2021, Zimmerman et al. 2021). Use it for exploration,
        or when there is only one sample per condition. Fitted with the
        MAST defaults (bayesglm + ebayes), as in the official vignette.
    model="glmer"  ~ ngeneson + covariates + condition + (1 | sample)
        Random intercept per biological sample. This is the correct unit
        of replication. With few samples per condition (for example 2 vs 2)
        the random-effect variance is poorly estimated, so results are
        still fragile. Pseudobulk (`sctools.degs.PseudoDESeq2`) is a good
        cross-check. Fitted as in the sc-best-practices MAST-RE example
        (ebayes=FALSE, nAGQ=0, strictConvergence=FALSE).

Reproducing the source notebook (07sx, random_effect=False):
    model="glm", glm_method="glm", ebayes=False, target_sum=1e6

Output columns (one row per tested gene, nothing is filtered by FDR):
    gene_id, log2FoldChange, ci_lo, ci_hi, lfc_source, pvalue, padj, stat,
    pct_ref, pct_contrast, n_cells_ref, n_cells_contrast, celltype, model

The column names follow `PseudoDESeq2`, so the result plugs directly into
existing sctools functions:
    sctools.degs.VolcanoGridByGroup(df, col_gene="gene_id")
    sctools.plots.PlotHeatmap(df, x="celltype", y="gene_id", color_col="log2FoldChange")
    sctools.functionalanalysis.PrepareULMInput(df.dropna(subset=["stat"]), celltype, value_col="stat")

Not ported from the source notebook (07sx..._DEGs_MAST.ipynb):
- `Import4DEGs`: built a fake `replicate` column from the last digit of
  the sample name. That grouped K1 with W1 and K2 with W2 as if they were
  the same animal. Use a real sample column with `sample_col`.
- `ExportAllTogether`: use `df.to_csv(...)` on the returned table.
- `plot_volcano_mast_grid` and `plot_heatmap_top_degs`: replaced by
  `VolcanoGridByGroup` and `PlotHeatmap` (see above).
- `sc_toolbox.tools.de_res_to_anndata`: results are returned as a table,
  not written into `adata.uns`.

Example
-------
df, skipped = RunMASTByCelltype(
    adata,
    celltype_col="celltype",
    model="glmer",                 # or "glm" (no random effect)
    condition_col="condition",
    ref_level="WT",
    contrast_level="KO",
    sample_col="sample",           # biological replicate (mouse / donor)
    counts_layer="QC_filtered",    # raw counts
)
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import norm

from sctools.preprocessing import CheckNormalizedLayer, NormalizeLog1pFromLayer

logger = logging.getLogger(__name__)

__all__ = ["RunMAST", "RunMASTByCelltype"]


class MASTDesignError(ValueError):
    """Raised when a contrast does not have enough cells/samples to be tested."""


# R side: defined once per R session, then called with arguments
# (no parameters are passed through R's globalenv).
_MAST_R_CODE = r"""
function(sce, condition_col, ref_level, contrast_level,
         sample_col, covariates, model, glm_method, ebayes, n_cores) {
    suppressPackageStartupMessages(library(MAST))
    options(mc.cores = n_cores)

    if (model == "glmer" && !requireNamespace("lme4", quietly = TRUE)) {
        stop("model='glmer' needs the R package 'lme4', which is not installed.")
    }

    sca <- SceToSingleCellAssay(sce, class = "SingleCellAssay")

    # Cellular detection rate, scaled (MAST vignette). Computed after the
    # gene frequency filter, as in the vignette.
    colData(sca)$ngeneson <- as.numeric(scale(colSums(assay(sca) > 0)))

    colData(sca)$de_condition <- relevel(
        factor(as.character(colData(sca)[[condition_col]])), ref = ref_level
    )

    terms <- c("ngeneson", covariates, "de_condition")
    for (cv in covariates) {
        if (is.character(colData(sca)[[cv]])) {
            colData(sca)[[cv]] <- factor(colData(sca)[[cv]])
        }
    }

    if (model == "glmer") {
        colData(sca)$de_sample <- factor(as.character(colData(sca)[[sample_col]]))
        fml <- as.formula(paste("~", paste(c(terms, "(1 | de_sample)"), collapse = " + ")))
        fit <- zlm(fml, sca = sca, method = "glmer", ebayes = FALSE,
                   strictConvergence = FALSE, fitArgsD = list(nAGQ = 0))
    } else {
        fml <- as.formula(paste("~", paste(terms, collapse = " + ")))
        fit <- zlm(fml, sca = sca, method = glm_method, ebayes = ebayes)
    }

    coef_name <- paste0("de_condition", contrast_level)
    dt <- summary(fit, doLRT = coef_name)$datatable
    dt <- as.data.frame(dt[dt$contrast == coef_name, ])

    h <- dt[dt$component == "H", c("primerid", "Pr(>Chisq)")]
    lfc <- dt[dt$component == "logFC", c("primerid", "coef", "ci.lo", "ci.hi")]
    d <- dt[dt$component == "D", c("primerid", "coef")]
    cc <- dt[dt$component == "C", c("primerid", "coef", "ci.lo", "ci.hi")]
    names(h) <- c("gene_id", "pvalue")
    names(lfc) <- c("gene_id", "logfc_ln", "ci_lo_ln", "ci_hi_ln")
    names(d) <- c("gene_id", "coef_D")
    names(cc) <- c("gene_id", "coef_C", "ci_lo_C", "ci_hi_C")

    res <- merge(h, lfc, by = "gene_id", all.x = TRUE)
    res <- merge(res, d, by = "gene_id", all.x = TRUE)
    res <- merge(res, cc, by = "gene_id", all.x = TRUE)
    res$padj <- p.adjust(res$pvalue, method = "BH")
    res
}
"""

_MAST_R_FUNC = None


def _GetMASTRFunction():
    """Compile the R function once and cache it."""
    global _MAST_R_FUNC
    if _MAST_R_FUNC is None:
        import rpy2.robjects as ro

        _MAST_R_FUNC = ro.r(_MAST_R_CODE)
    return _MAST_R_FUNC


def _CheckMASTDesign(
    obs: pd.DataFrame,
    *,
    model: str,
    condition_col: str,
    ref_level: str,
    contrast_level: str,
    sample_col: str | None,
    min_cells_per_group: int,
    min_cells_per_sample: int,
    min_samples_per_group: int,
    label: str,
) -> None:
    """
    Raise `MASTDesignError` if the contrast cannot be tested, and log
    warnings about the replication design.
    """
    n_cells = obs[condition_col].value_counts()
    for level in (ref_level, contrast_level):
        n = int(n_cells.get(level, 0))
        if n < min_cells_per_group:
            raise MASTDesignError(
                f"{label}: '{level}' has {n} cells, need at least {min_cells_per_group}."
            )

    if sample_col is None:
        logger.warning(
            "%s: no sample_col given, so replication cannot be checked. Every "
            "cell is treated as an independent replicate (pseudoreplication).",
            label,
        )
        return

    cells_per_sample = obs.groupby([condition_col, sample_col], observed=True).size()
    cells_per_sample = cells_per_sample[cells_per_sample > 0]

    n_samples = {}
    for level in (ref_level, contrast_level):
        per_sample = cells_per_sample.loc[level]
        n_samples[level] = int((per_sample >= min_cells_per_sample).sum())
        if n_samples[level] < min_samples_per_group:
            raise MASTDesignError(
                f"{label}: '{level}' has {n_samples[level]} sample(s) with at least "
                f"{min_cells_per_sample} cells, need {min_samples_per_group}. "
                f"Cells per sample: {per_sample.to_dict()}."
            )

    if model == "glm" and min(n_samples.values()) >= 2:
        logger.warning(
            "%s: model='glm' with %s samples per condition. Cells are treated as "
            "independent replicates, so p-values are anti-conservative. "
            "Consider model='glmer' or pseudobulk.",
            label, n_samples,
        )
    if model == "glmer" and min(n_samples.values()) < 3:
        logger.warning(
            "%s: model='glmer' with only %s samples per condition. The "
            "random-effect variance is poorly estimated, interpret with care.",
            label, n_samples,
        )


def RunMAST(
    adata: AnnData,
    *,
    model: Literal["glm", "glmer"],
    condition_col: str,
    ref_level: str,
    contrast_level: str,
    counts_layer: str = "counts",
    sample_col: str | None = None,
    covariates: Sequence[str] | None = None,
    celltype: str | None = None,
    celltype_col: str | None = None,
    target_sum: float = 1e4,
    freq_filter: float = 0.1,
    min_cells_per_group: int = 20,
    min_cells_per_sample: int = 10,
    min_samples_per_group: int = 1,
    glm_method: Literal["bayesglm", "glm"] = "bayesglm",
    ebayes: bool = True,
    n_cores: int = 1,
) -> pd.DataFrame:
    """
    Run MAST for one contrast (contrast_level vs ref_level) in one celltype.

    The input `adata` is never modified. Cells are subset to the celltype
    (if given) and to the two compared levels, then raw counts from
    `counts_layer` are normalized and log1p transformed on this copy only.

    Parameters
    ----------
    adata : AnnData with raw counts in `counts_layer`.
    model : "glm" (no random effect) or "glmer" (random intercept per
        `sample_col`). No default, see the module docstring.
    condition_col : obs column with the condition labels.
    ref_level, contrast_level : levels compared. A positive log2FoldChange
        means higher in `contrast_level`.
    counts_layer : layer with raw (or ambient-corrected) counts. Not a
        log-normalized layer, since it is normalized here.
    sample_col : obs column with the biological replicate (mouse, donor).
        Required for model="glmer". For model="glm" it is only used to
        check the design and to warn about pseudoreplication.
    covariates : extra obs columns added as fixed effects (for example a
        batch or sex column). They must not be confounded with the condition.
    celltype, celltype_col : restrict to `adata.obs[celltype_col] == celltype`.
        Leave both as None to use all cells.
    target_sum : total counts per cell after normalization. The default 1e4
        matches `NormalizeLog1pFromLayer`. The source notebook used 1e6, which
        gives larger |log2FoldChange| values. Use 1e6 to reproduce old results.
    freq_filter : keep genes detected (count > 0) in more than this fraction
        of the tested cells (same rule as `freq(sca) > freq_filter` in MAST).
    min_cells_per_group : minimum cells per condition level.
    min_cells_per_sample : a sample counts toward `min_samples_per_group`
        only if it has at least this many cells.
    min_samples_per_group : minimum samples per condition level (only
        checked when `sample_col` is given).
    glm_method, ebayes : fitting options for model="glm" only (ignored for
        "glmer", which MAST fits with lme4 and ebayes=FALSE). The defaults
        ("bayesglm", True) are the MAST defaults used in the official
        vignette: a Cauchy prior on the discrete part keeps it estimable
        when a gene is detected in almost all cells, and ebayes shrinks the
        continuous variance. `glm_method="glm", ebayes=False` reproduces
        the source notebook (07sx).
    n_cores : number of R cores used by `zlm` (through `mc.cores`).

    Returns
    -------
    DataFrame with one row per tested gene (see module docstring).
    `log2FoldChange` is the MAST hurdle logFC (combines the discrete and
    continuous parts) converted to log2 units. It is the change in the
    expected value of log1p(normalized expression), not the log2 ratio of
    mean counts. `lfc_source` says where it comes from:
        "hurdle"     : MAST logFC (normal case).
        "continuous" : the discrete part was not estimable because the gene
                       is detected in almost all cells (for example Gapdh,
                       Actg1). Then P(detected) ~ 1 in both groups and the
                       hurdle logFC reduces to the continuous coefficient,
                       which is used instead. The source notebook dropped
                       these genes with `na.omit`.
        "none"       : no estimable effect size (log2FoldChange is NaN).
    All genes are kept with their hurdle test p-value.
    `stat` is a signed z-equivalent of the hurdle LRT p-value, with the
    sign of `log2FoldChange` (or of the discrete coefficient when the
    logFC is NaN). Use it for ranking (ULM/GSEA). `pvalue`, `padj` and
    `stat` are NaN when the LRT fails for a gene (rare, logged). Drop these
    rows before ULM/GSEA.
    """
    import anndata2ri
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    if model not in ("glm", "glmer"):
        raise ValueError(f"model must be 'glm' or 'glmer', got {model!r}.")
    if glm_method not in ("bayesglm", "glm"):
        raise ValueError(f"glm_method must be 'bayesglm' or 'glm', got {glm_method!r}.")
    if model == "glmer" and sample_col is None:
        raise ValueError("model='glmer' needs sample_col (the random-effect grouping).")
    if (celltype is None) != (celltype_col is None):
        raise ValueError("Give both celltype and celltype_col, or neither.")
    if counts_layer not in adata.layers:
        raise KeyError(
            f"Layer '{counts_layer}' not found. Available layers: {list(adata.layers.keys())}."
        )

    covariates = list(covariates or [])
    needed_cols = [condition_col] + covariates
    if sample_col is not None:
        needed_cols.append(sample_col)
    if celltype_col is not None:
        needed_cols.append(celltype_col)
    missing = [c for c in needed_cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Columns not found in adata.obs: {missing}.")

    label = celltype if celltype is not None else "all cells"

    # Subset cells: celltype (optional) and only the two compared levels.
    condition = adata.obs[condition_col].astype(str)
    mask = condition.isin([str(ref_level), str(contrast_level)]).to_numpy()
    if celltype_col is not None:
        mask &= (adata.obs[celltype_col].astype(str) == str(celltype)).to_numpy()

    obs = adata.obs.loc[mask, needed_cols].copy()
    obs[condition_col] = obs[condition_col].astype(str)
    if sample_col is not None:
        obs[sample_col] = obs[sample_col].astype(str)

    _CheckMASTDesign(
        obs,
        model=model,
        condition_col=condition_col,
        ref_level=str(ref_level),
        contrast_level=str(contrast_level),
        sample_col=sample_col,
        min_cells_per_group=min_cells_per_group,
        min_cells_per_sample=min_cells_per_sample,
        min_samples_per_group=min_samples_per_group,
        label=label,
    )

    # Minimal AnnData: only the count layer and the needed obs columns go to R.
    counts = adata.layers[counts_layer][mask]
    sub = AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=adata.var_names.copy()))
    sub.layers["counts"] = sub.X

    NormalizeLog1pFromLayer(sub, input_layer="counts", target_sum=target_sum, output_layer="log1p")
    CheckNormalizedLayer(sub.X)

    # Gene filter: detected in more than `freq_filter` of the tested cells.
    detected = sub.layers["counts"] > 0
    frac_detected = np.asarray(detected.mean(axis=0)).ravel()
    keep = frac_detected > freq_filter
    sub = sub[:, keep].copy()
    logger.info(
        "%s: %d cells (%s), %d of %d genes pass freq_filter=%s.",
        label, sub.n_obs, obs[condition_col].value_counts().to_dict(),
        int(keep.sum()), keep.size, freq_filter,
    )
    if keep.sum() == 0:
        raise MASTDesignError(f"{label}: no gene passes freq_filter={freq_filter}.")

    # Fraction of cells with count > 0 per condition, for the output table.
    is_ref = (sub.obs[condition_col] == str(ref_level)).to_numpy()
    detected = sub.layers["counts"] > 0
    pct_ref = np.asarray(detected[is_ref].mean(axis=0)).ravel()
    pct_contrast = np.asarray(detected[~is_ref].mean(axis=0)).ravel()

    # Send only the log-normalized matrix to R (MAST uses the first assay).
    X = sub.X.toarray() if issparse(sub.X) else np.asarray(sub.X)
    sce_input = AnnData(X=X, obs=sub.obs, var=pd.DataFrame(index=sub.var_names))

    r_func = _GetMASTRFunction()
    r_sample = ro.NULL if sample_col is None else ro.StrVector([sample_col])
    with localconverter(ro.default_converter + pandas2ri.converter + anndata2ri.converter):
        res = r_func(
            sce_input,
            ro.StrVector([condition_col]),
            ro.StrVector([str(ref_level)]),
            ro.StrVector([str(contrast_level)]),
            r_sample,
            ro.StrVector(covariates),
            ro.StrVector([model]),
            ro.StrVector([glm_method]),
            ro.BoolVector([bool(ebayes)]),
            ro.IntVector([int(n_cores)]),
        )
    res = pd.DataFrame(res).reset_index(drop=True)

    # The hurdle logFC is NaN when the discrete part (D) is not estimable.
    # This happens for genes detected in almost every cell (too few zeros).
    # There P(detected) ~ 1 in both groups, so the hurdle logFC reduces to
    # the continuous coefficient (C). Use it, and record the source.
    use_c = res["logfc_ln"].isna() & res["coef_D"].isna() & res["coef_C"].notna()
    res["lfc_source"] = np.where(res["logfc_ln"].notna(), "hurdle", "none")
    res.loc[use_c, "lfc_source"] = "continuous"
    for hurdle_col, c_col in (("logfc_ln", "coef_C"), ("ci_lo_ln", "ci_lo_C"), ("ci_hi_ln", "ci_hi_C")):
        res.loc[use_c, hurdle_col] = res.loc[use_c, c_col]

    # Natural-log units from MAST -> log2 units.
    res["log2FoldChange"] = res["logfc_ln"] / np.log(2)
    res["ci_lo"] = res["ci_lo_ln"] / np.log(2)
    res["ci_hi"] = res["ci_hi_ln"] / np.log(2)

    # Signed z-equivalent of the hurdle LRT p-value, for ranking.
    sign = np.sign(res["log2FoldChange"]).fillna(np.sign(res["coef_D"]))
    res["stat"] = sign * norm.isf(res["pvalue"].clip(lower=1e-300) / 2)

    pct = pd.DataFrame(
        {"pct_ref": pct_ref, "pct_contrast": pct_contrast}, index=sub.var_names
    )
    res = res.join(pct, on="gene_id")
    res["n_cells_ref"] = int(is_ref.sum())
    res["n_cells_contrast"] = int((~is_ref).sum())
    res["celltype"] = label
    res["model"] = model

    n_nan_p = int(res["pvalue"].isna().sum())
    if n_nan_p:
        logger.warning(
            "%s: the LRT failed for %d gene(s) (pvalue/stat are NaN). Drop them "
            "before ranking-based analyses, e.g. df.dropna(subset=['stat']).",
            label, n_nan_p,
        )

    n_c = int(use_c.sum())
    n_nan_lfc = int(res["log2FoldChange"].isna().sum())
    if n_c or n_nan_lfc:
        logger.info(
            "%s: log2FoldChange taken from the continuous part for %d gene(s) "
            "(discrete part not estimable, gene detected in almost all cells); "
            "still NaN for %d gene(s). All are kept with their hurdle p-value.",
            label, n_c, n_nan_lfc,
        )

    cols = [
        "gene_id", "log2FoldChange", "ci_lo", "ci_hi", "lfc_source", "pvalue", "padj", "stat",
        "pct_ref", "pct_contrast", "n_cells_ref", "n_cells_contrast", "celltype", "model",
    ]
    return res[cols].sort_values("pvalue", ignore_index=True)


def RunMASTByCelltype(
    adata: AnnData,
    *,
    celltype_col: str,
    celltypes: Sequence[str] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Run `RunMAST` for each celltype and concatenate the results.

    Parameters
    ----------
    adata : AnnData passed to `RunMAST` (not modified).
    celltype_col : obs column with the celltype labels.
    celltypes : celltypes to test. If None, all values of `celltype_col`
        (category order if categorical).
    **kwargs : passed to `RunMAST` (model, condition_col, ref_level, ...).

    Returns
    -------
    results : long DataFrame, one row per (celltype, gene).
    skipped : {celltype: reason} for celltypes that failed the design
        checks (too few cells or samples). R errors are not caught.
    """
    if celltypes is None:
        col = adata.obs[celltype_col]
        celltypes = list(col.cat.categories) if hasattr(col, "cat") else sorted(col.unique())

    results = []
    skipped = {}
    for ct in celltypes:
        logger.info("MAST: processing %s ...", ct)
        try:
            results.append(RunMAST(adata, celltype=ct, celltype_col=celltype_col, **kwargs))
        except MASTDesignError as e:
            logger.warning("Skipping %s: %s", ct, e)
            skipped[ct] = str(e)

    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return df, skipped
