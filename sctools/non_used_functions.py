def ResolveNPcs(
    adata: AnnData,
    *,
    requested_n_pcs: int,
    use_highly_variable: bool = True,
    auto_reduce: bool = True,
) -> int:
    """
    Validate the requested number of principal components.

    Why this check exists
    ---------------------
    Even if you request 1000 or 2000 HVGs, PCA can still fail if the number of
    cells is small. With `svd_solver="arpack"`, the number of PCs must be
    strictly smaller than:

        min(number of cells, number of genes used for PCA)

    For example:
        - 1000 HVGs and 5000 cells -> n_pcs=35 is fine.
        - 1000 HVGs and 20 cells   -> n_pcs=35 is not valid.

    This function prevents PCA from failing in small test objects, subsets,
    or rare edge cases after filtering.

    Parameters
    ----------
    requested_n_pcs
        Number of PCs requested by the user.

    use_highly_variable
        If True, checks the number of HVGs. If False, checks all genes.

    auto_reduce
        If True, reduce `requested_n_pcs` to the maximum valid value and log a
        warning. If False, raise an error.

    Returns
    -------
    n_pcs_used
        Valid number of PCs to use.
    """
    if use_highly_variable:
        if "highly_variable" not in adata.var:
            raise KeyError(
                "Missing adata.var['highly_variable']. "
                "Run RunHighlyVariableGenes() before PCA."
            )

        n_genes_for_pca = int(adata.var["highly_variable"].sum())
        gene_set_name = "HVGs"

    else:
        n_genes_for_pca = int(adata.n_vars)
        gene_set_name = "genes"

    if n_genes_for_pca < 2:
        raise ValueError(
            f"Only {n_genes_for_pca} {gene_set_name} available for PCA. "
            "PCA requires at least 2 variables."
        )

    if adata.n_obs < 2:
        raise ValueError(
            f"Only {adata.n_obs} cells available for PCA. "
            "PCA requires at least 2 cells."
        )

    max_n_pcs = min(adata.n_obs - 1, n_genes_for_pca - 1)

    if requested_n_pcs <= max_n_pcs:
        return int(requested_n_pcs)

    message = (
        f"Requested n_pcs={requested_n_pcs}, but the maximum valid value is "
        f"{max_n_pcs} for n_obs={adata.n_obs} and "
        f"n_{gene_set_name}={n_genes_for_pca}."
    )

    if not auto_reduce:
        raise ValueError(message)

    logger.warning("%s Using n_pcs=%d instead.", message, max_n_pcs)

    return int(max_n_pcs)


def RunIntegrationMethods(
    adata_hvg: AnnData,
    methods: tuple[str, ...] = SUPPORTED_METHODS,
    *,
    method_kwargs: dict[str, dict] | None = None,
    batch_key: str = "sample",
    plot_dir: str | Path | None = None,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> dict[str, AnnData]:
    """
    Run one or more integration methods on the same HVG-subset AnnData.

    method_kwargs: optional {method_name: {kwarg: value}} overrides, passed
    only to that method's function. batch_key/random_state are always
    forwarded automatically.

    plot_dir: if given, save a diagnostic UMAP per method under this directory.

    Returns dict mapping method name -> integrated AnnData.
    """
    unknown = set(methods) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unsupported integration method(s): {sorted(unknown)}. Supported: {SUPPORTED_METHODS}")

    method_kwargs = method_kwargs or {}
    res: dict[str, AnnData] = {}

    for method in methods:
        logger.info("Applying integration method: %s", method)

        kwargs = {"batch_key": batch_key, "random_state": random_state, **method_kwargs.get(method, {})}
        
        adata_int = _METHOD_FUNCS[method](adata_hvg, **kwargs)

        if plot_dir is not None:
            PlotUmap(
                adata_int,
                title=method,
                batch_key=batch_key,
                rep=SCIB_EMBED_BY_METHOD[method],
                recompute_neighbors=(method != "bbknn"),
                n_neighbors=n_neighbors,
                random_state=random_state,
                plot_dir=plot_dir,
            )

        res[method] = adata_int

    return res


def RunBbknnIntegration(
    adata_hvg: AnnData,
    *,
    batch_key: str = "sample",
    use_rep: str = "X_pca",
    n_pcs: int = 35,
    neighbors_within_batch: int | str = "auto",
    random_state: int = 42,
) -> AnnData:
    """
    Batch-balanced kNN (BBKNN). Graph-based: rewrites the neighbor graph in
    place -- do not recompute `sc.pp.neighbors` afterwards.

    neighbors_within_batch="auto" uses 25 for large atlases (>100k cells),
    3 otherwise (Scanpy's own default is 3).

    use_rep: which representation to use for BBKNN (usually "X_pca").

    # ------------------------
    # BBKNN is graph-based integration.
    #
    # It does NOT create:
    #   - a corrected expression matrix
    #   - a corrected layer
    #   - a corrected embedding such as "X_harmony" or "X_scVI"
    #
    # BBKNN uses:
    #   - adata.obsm["X_pca"] as input representation
    #   - adata.obs[batch_key] to know which cells belong to each batch/sample
    #
    #
    # Therefore, the "integrated result" of BBKNN is the batch-balanced neighbor
    # graph, not a new embedding.
    #
    # To inspect the BBKNN integration result, do NOT look for a new layer or a new
    # corrected embedding such as "X_bbknn". BBKNN does not create one.
    #
    # Instead, BBKNN updates the neighbor graph stored in:
    #   - adata.uns["neighbors"]
    #   - adata.obsp["distances"]
    #   - adata.obsp["connectivities"]
    #
    # These graph objects are then used by UMAP and Leiden.
    #
    # Example downstream workflow:
    #
    #   # 1. Run BBKNN.
    #   adata_bbknn = RunBbknnIntegration(
    #       adata_hvg,
    #       batch_key="sample",
    #       n_pcs=35,
    #       neighbors_within_batch="auto",
    #       random_state=42,
    #   )
    #
    #   # 2. Compute UMAP directly from the BBKNN graph.
    #   #    This uses adata_bbknn.obsp["connectivities"] created by BBKNN.
    #   sc.tl.umap(
    #       adata_bbknn,
    #       random_state=42,
    #   )
    #
    #   # 3. Run Leiden clustering directly on the BBKNN graph.
    #   #    This also uses the BBKNN connectivities graph.
    #   sc.tl.leiden(
    #       adata_bbknn,
    #       resolution=1.0,
    #       key_added="leiden_res1",
    #       random_state=42,
    #   )
    #
    #   # 4. Plot the integrated UMAP.
    #   sc.pl.umap(
    #       adata_bbknn,
    #       color=["sample", "leiden_res1"],
    #       wspace=0.4,
    #   )
    #
    # IMPORTANT:
    # Do NOT run sc.pp.neighbors(adata_bbknn) after BBKNN.
    # That would overwrite the BBKNN graph and discard the integration result.
    #
    """

    import scanpy.external as sce

    adata = adata_hvg.copy()
    _RequirePca(adata)

    # BBKNN's n_pcs must not exceed the number of genes used for PCA (n_genes_for_pca).
    # this part is contained in the Single Cell Best Practices reference
    if neighbors_within_batch == "auto":
        neighbors_within_batch = 25 if adata.n_obs > 100_000 else 3

    logger.info("Running BBKNN (batch_key=%s, neighbors_within_batch=%s)", batch_key, neighbors_within_batch)

    sce.pp.bbknn(
        adata,
        batch_key=batch_key,
        use_rep=use_rep,
        neighbors_within_batch=neighbors_within_batch,
        n_pcs=n_pcs,
    )
    return adata


def RunPrerankByCelltype(
    de_long: pd.DataFrame,
    net: pd.DataFrame,
    *,
    contrast_name: str = "KO.vs.WT",
    celltype_col: str = "celltype",
    gene_col: str = "gene_id",
    value_col: str = "stat",
    celltypes: Sequence[str] | None = None,
    permutation_num: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    seed: int = 42,
    threads: int = 4,
) -> pd.DataFrame:
    """
    Run gseapy prerank GSEA per celltype on a long-format DE table.

    Same input and output as `RunGSEAByCelltype`, but with the original
    Broad GSEA statistics.

    Difference: `RunGSEAByCelltype` (decoupler) gives each pathway its own
    permutation p-value and adjusts them by BH. gseapy computes the FDR from
    the NES null distribution pooled across all pathways (Broad method).
    `padj` here is this Broad FDR q-value, not a BH-adjusted p-value. It is
    computed across the pathways of one celltype (not across celltypes).
    Its lowest value is 1/permutation_num.

    Genes are ranked by `value_col` inside each celltype (default "stat",
    the DESeq2 Wald statistic). Genes with NaN in `value_col` are dropped.

    Parameters
    ----------
    de_long : long-format DE table, one row per (celltype, gene).
    net : decoupler-style network with columns "source" (pathway) and
        "target" (gene), the same format `RunGSEAByCelltype` takes.
    contrast_name : "{compare_condition}.vs.{normal_condition}". Used for
        the `direction` column.
    celltype_col, gene_col, value_col : columns of `de_long`.
    celltypes : celltypes to run. If None, all in `de_long` (order of appearance).
    permutation_num : number of gene set permutations.
    min_size, max_size : pathway size limits, counted on the genes present
        in the ranking (gseapy defaults, 15 and 500).
    seed : random seed for the permutations.
    threads : number of threads for gseapy.

    Returns
    -------
    DataFrame with columns: celltype_col, pathway, NES, pval, padj,
    lead_genes, direction. `pval` is the nominal p-value, `padj` the FDR
    q-value, `lead_genes` the leading edge genes (";"-separated).
    `direction` is compare_condition when NES > 0, normal_condition when
    NES < 0, NaN otherwise.
    """
    import gseapy as gp

    compare_condition, normal_condition = contrast_name.split(".vs.")

    if celltypes is None:
        celltypes = list(pd.unique(de_long[celltype_col]))

    gene_sets = net.groupby("source")["target"].apply(list).to_dict()
    ranked = de_long.dropna(subset=[value_col])

    results = []
    for celltype in celltypes:
        logger.info("Prerank GSEA: %s", celltype)
        rnk = ranked.loc[ranked[celltype_col] == celltype].set_index(gene_col)[value_col]

        res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutation_num,
            seed=seed,
            threads=threads,
            outdir=None,
            no_plot=True,
            verbose=False,
        ).res2d

        # gseapy returns ES/NES as object dtype
        gsea_df = pd.DataFrame({
            celltype_col: celltype,
            "pathway": res["Term"].to_numpy(),
            "NES": pd.to_numeric(res["NES"]).to_numpy(),
            "pval": pd.to_numeric(res["NOM p-val"]).to_numpy(),
            "padj": pd.to_numeric(res["FDR q-val"]).to_numpy(),
            "lead_genes": res["Lead_genes"].to_numpy(),
        })
        results.append(gsea_df.sort_values("NES", ascending=False))

    gsea_long = pd.concat(results, ignore_index=True)

    gsea_long["direction"] = np.select(
        [gsea_long["NES"] > 0, gsea_long["NES"] < 0],
        [compare_condition, normal_condition],
        default=None,
    )
    return gsea_long
