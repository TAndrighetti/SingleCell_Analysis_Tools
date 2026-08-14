"""
sctools.networks
=================
Filtering of prior-knowledge interaction networks (PPI, TRI) downloaded
from OmniPath.

Functions
---------
FilterKnowledgeNetworks – filter PPI/TRI edges by source/target gene lists

hallmark = dc.op.hallmark(organism="mouse")
progeny = dc.op.progeny(organism="mouse")
collectri = dc.op.collectri(organism="mouse")

"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd


def _normalize_network_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        `_normalize_network_columns`. Only the table(s) requested in
        `networks` need to be given.
    networks : which network(s) to filter and concatenate.

    Returns
    -------
    pd.DataFrame with columns: source_genesymbol, target_genesymbol,
    effect ("stimulation" / "inhibition" / "stim/inhib" / "undefined"),
    interaction_type ("ppi" / "tri").
    """
    tables = {
        "ppi": _normalize_network_columns(ppi) if ppi is not None else None,
        "tri": _normalize_network_columns(tri) if tri is not None else None,
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
