"""
sctools.networks
=================
Filtering of prior-knowledge interaction networks (PPI, TRI) downloaded
from OmniPath.

Functions
---------
FilterKnowledgeNetworks – filter PPI/TRI edges by source/target gene lists
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd


def FilterKnowledgeNetworks(
    source_genes: Iterable[str],
    target_genes: Iterable[str],
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
        source_genesymbol / target_genesymbol.
    ppi, tri : raw OmniPath tables, each with source_genesymbol,
        target_genesymbol, is_stimulation, is_inhibition. Only the
        table(s) requested in `networks` need to be given.
    networks : which network(s) to filter and concatenate.

    Returns
    -------
    pd.DataFrame with columns: source_genesymbol, target_genesymbol,
    effect ("stimulation" / "inhibition" / "stim/inhib" / "undefined"),
    interaction_type ("ppi" / "tri").
    """
    source_genes = set(source_genes)
    target_genes = set(target_genes)
    tables = {"ppi": ppi, "tri": tri}

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
