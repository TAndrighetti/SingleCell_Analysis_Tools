"""
sctools.networks
=================
Filtering of prior-knowledge interaction networks (PPI, TRI) downloaded
from OmniPath.

Functions
---------
FilterKnowledgeNetworks – filter PPI/TRI edges by source/target gene lists
BuildCytoscapeEdges – build a Cytoscape-ready TF->DEG edge table with sign/weight
BuildPathwaySif – build a TieDIE-ready pathway.sif table (source, interaction>, target)

hallmark = dc.op.hallmark(organism="mouse")
progeny = dc.op.progeny(organism="mouse")
collectri = dc.op.collectri(organism="mouse")

"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd


def _NormalizeNetworkColumns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a prior-knowledge network table to this module's internal
    column convention (source_genesymbol, target_genesymbol,
    is_stimulation, is_inhibition).

    The native `omnipath` package already uses these names. decoupler's
    `dc.op.collectri()` (and similar `dc.op.*` wrappers) instead return
    `source`/`target`/`weight`, with `weight` encoding the regulation
    sign (positive = activation, negative = repression -- CollecTRI's
    own convention) rather than separate boolean columns. This detects
    the latter shape and renames/derives into the former, so both
    sources work with the same filtering logic below.
    """
    df = df.copy()
    if "source_genesymbol" not in df.columns and "source" in df.columns:
        df = df.rename(columns={"source": "source_genesymbol"})
    if "target_genesymbol" not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": "target_genesymbol"})
    if "is_stimulation" not in df.columns and "weight" in df.columns:
        df["is_stimulation"] = df["weight"] > 0
        df["is_inhibition"] = df["weight"] < 0
    return df


def FilterKnowledgeNetworks(
    source_genes: Iterable[str] | None = None,
    target_genes: Iterable[str] | None = None,
    ppi: pd.DataFrame | None = None,
    tri: pd.DataFrame | None = None,
    networks: Iterable[Literal["ppi", "tri"]] = ("ppi", "tri"),
) -> pd.DataFrame:
    """
    Filter PPI and/or TRI prior-knowledge networks (OmniPath) to edges whose
    source is in `source_genes` and target is in `target_genes`.

    Pass the same gene list as both `source_genes` and `target_genes` to
    filter a single gene set against itself (e.g. a correlated-gene
    network), or two different lists for directional filtering (e.g.
    transcription factors -> differentially expressed genes).

    Parameters
    ----------
    source_genes, target_genes : gene symbols to keep in
        source_genesymbol / target_genesymbol. `None` (the default) keeps
        every gene that appears on that side in the requested `networks`
        -- i.e. no filtering on that side.
    ppi, tri : raw OmniPath tables. Either the native `omnipath` package's
        column convention (source_genesymbol, target_genesymbol,
        is_stimulation, is_inhibition) or decoupler's `dc.op.*` wrapper
        convention (source, target, weight) is accepted -- see
        `_NormalizeNetworkColumns`. Only the table(s) requested in
        `networks` need to be given.
    networks : which network(s) to filter and concatenate.

    Returns
    -------
    pd.DataFrame with columns: source_genesymbol, target_genesymbol,
    effect ("stimulation" / "inhibition" / "stim/inhib" / "undefined"),
    interaction_type ("ppi" / "tri").
    """
    tables = {
        "ppi": _NormalizeNetworkColumns(ppi) if ppi is not None else None,
        "tri": _NormalizeNetworkColumns(tri) if tri is not None else None,
    }

    # a bare string is iterable character-by-character in Python, so
    # `networks="tri"` would otherwise silently loop over 't', 'r', 'i'
    # instead of the single network name
    if isinstance(networks, str):
        networks = [networks]

    # None -> no filtering on that side: keep every gene that actually
    # appears there in the requested networks, instead of dropping
    # everything (an empty `source_genes`/`target_genes` set would match
    # nothing via .isin())
    if source_genes is None:
        source_genes = set().union(*(set(tables[name]["source_genesymbol"]) for name in networks))
    else:
        source_genes = set(source_genes)

    if target_genes is None:
        target_genes = set().union(*(set(tables[name]["target_genesymbol"]) for name in networks))
    else:
        target_genes = set(target_genes)

    filtered = []
    for name in networks:
        df = tables[name]
        subset = df[
            df["source_genesymbol"].isin(source_genes) & df["target_genesymbol"].isin(target_genes)
        ].copy()
        subset["interaction_type"] = name
        filtered.append(subset)

    combined = pd.concat(filtered, ignore_index=True)

    combined["effect"] = np.select(
        [
            combined["is_stimulation"] & combined["is_inhibition"],
            combined["is_stimulation"],
            combined["is_inhibition"],
        ],
        ["stim/inhib", "stimulation", "inhibition"],
        default="undefined",
    )

    return combined[["source_genesymbol", "target_genesymbol", "effect", "interaction_type"]]


def BuildCytoscapeEdges(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    celltype_col: str | None = None,
    interactions: pd.DataFrame | None = None,
    interaction_source_col: str = "source",
    interaction_target_col: str = "target",
    weight_col: str = "weight",
    is_stimulation_col: str | None = None,
    is_inhibition_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a Cytoscape-ready edge table (source -> target) with interaction
    sign, weight, and (optionally) celltype.

    The interaction sign can come from either of two places -- give
    exactly one:
    - `interactions` (+ `interaction_source_col`/`interaction_target_col`/
      `weight_col`): a separate prior-knowledge table to look up a signed
      numeric `weight` from by (source, target), for edge tables that
      don't carry sign themselves -- e.g. a TF-DEG pair table looked up
      against CollecTRI (positive weight = stimulation, negative =
      inhibition).
    - `is_stimulation_col`/`is_inhibition_col`: boolean columns already in
      `edges` -- e.g. OmniPath's PPI convention. `weight` is not
      available in this case and comes back as NaN.

    Parameters
    ----------
    edges : table with one row per edge, e.g. `int_degs_both` (TF-DEG
        pairs, sign looked up via `interactions`) or a raw OmniPath PPI
        table such as `human_ppi` (sign already in `is_stimulation`/
        `is_inhibition`).
    source_col, target_col : column names in `edges` for the edge source
        and target gene.
    celltype_col : column name in `edges` for celltype, if edges are
        celltype-specific. `None` (default) omits the `celltype` column
        -- e.g. for a PPI network with no celltype dimension.
    interactions, interaction_source_col, interaction_target_col,
    weight_col : the weight-lookup table and its column names (see above).
    is_stimulation_col, is_inhibition_col : the boolean sign columns in
        `edges` (see above).

    Returns
    -------
    pd.DataFrame with columns: shared name, source, target, interacao,
    weight, celltype (celltype only if `celltype_col` given) -- one row
    per unique edge.
    """
    if (interactions is not None) == (is_stimulation_col is not None):
        raise ValueError(
            "give exactly one of `interactions` or `is_stimulation_col`/`is_inhibition_col`"
        )

    subset_cols = [source_col, target_col]
    rename_map = {source_col: "source", target_col: "target"}

    if celltype_col is not None:
        subset_cols.append(celltype_col)
        rename_map[celltype_col] = "celltype"

    if is_stimulation_col is not None:
        subset_cols += [is_stimulation_col, is_inhibition_col]

    out = edges[subset_cols].drop_duplicates().rename(columns=rename_map)

    if interactions is not None:
        out = out.merge(
            interactions[[interaction_source_col, interaction_target_col, weight_col]].rename(
                columns={
                    interaction_source_col: "source",
                    interaction_target_col: "target",
                    weight_col: "weight",
                }
            ),
            on=["source", "target"],
            how="left",
        )
        out["interacao"] = out["weight"].map(
            lambda w: "stimulation" if w > 0 else "inhibition" if w < 0 else "interaction"
        )
    else:
        out["interacao"] = np.select(
            [
                out[is_stimulation_col] & out[is_inhibition_col],
                out[is_stimulation_col],
                out[is_inhibition_col],
            ],
            ["stim/inhib", "stimulation", "inhibition"],
            default="undefined",
        )
        out["weight"] = np.nan
        out = out.drop(columns=[is_stimulation_col, is_inhibition_col])

    # combined "source (interacts with) target" column, matching Cytoscape's "shared name"
    out["shared name"] = out["source"] + " (interacts with) " + out["target"]

    result_cols = ["shared name", "source", "target", "interacao", "weight"]
    if celltype_col is not None:
        result_cols.append("celltype")

    return out[result_cols]


def BuildPathwaySif(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    is_stimulation_col: str,
    is_inhibition_col: str,
) -> pd.DataFrame:
    """
    Build a TieDIE-ready pathway table (source, interaction, target), the
    3-column .sif format used as the prior-knowledge network (PKN) input
    to TieDIE -- e.g. `A2m	activates>	Lrp1`.

    `interaction` is one of "activates>", "inhibits>", "interaction>"
    (undirected/unknown effect). When both `is_stimulation_col` and
    `is_inhibition_col` are true for an edge, "inhibits>" wins -- this
    matches the convention used to build the original human/mouse
    pathway.sif files.

    Parameters
    ----------
    edges : raw PPI/TRI prior-knowledge table, e.g. `human_ppi`.
    source_col, target_col : column names in `edges` for the edge source
        and target gene.
    is_stimulation_col, is_inhibition_col : boolean columns in `edges`
        marking a stimulatory / inhibitory interaction.

    Returns
    -------
    pd.DataFrame with columns: source, interaction, target -- write with
    `to_csv(..., sep="\\t", index=False, header=False)` for TieDIE.
    """
    net = (
        edges[[source_col, target_col, is_stimulation_col, is_inhibition_col]]
        .drop_duplicates()
        .rename(columns={source_col: "source", target_col: "target"})
    )

    net["interaction"] = np.select(
        [net[is_inhibition_col].astype(bool), net[is_stimulation_col].astype(bool)],
        ["inhibits>", "activates>"],
        default="interaction>",
    )

    return net[["source", "interaction", "target"]].drop_duplicates().reset_index(drop=True)

# # pathway.sif do PPI humano (source, interaction>, target), pronto pra ser o PKN do TieDIE
# pathway_human = BuildPathwaySif(
#     edges=human_ppi,
#     source_col="source_genesymbol",
#     target_col="target_genesymbol",
#     is_stimulation_col="is_stimulation",
#     is_inhibition_col="is_inhibition",
# )

# pathway_human.to_csv(
#     f"{main_dir}/input_files/pathway_human.sif",
#     sep="\t",
#     index=False,
#     header=False,
# )

# pathway_human