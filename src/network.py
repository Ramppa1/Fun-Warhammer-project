import networkx as nx
import numpy as np

def compute_meta_score(df):
    """
    Compute composite faction scores.
    This function calculates a composite score for each faction based on the provided features.
    """

    df = df.copy()

    #Norm event wins

    event_norm = df["eventwins"] / df["eventwins"].max()

    score = (
        0.5 * (df["winrate"] / 100) +  # Normalize winrate to [0, 1]
        0.3 * df["overrep"] +  # Use overrep directly
        0.2 * event_norm  # Use normalized event wins
    )

    df["meta_score"] = score
    return df

def build_dominance_graph(df):
    """
    Build a dominance graph based on the composite scores.
    This function creates a directed graph where edges point from stronger factions to weaker ones based on their meta scores.
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_node(row["faction"], meta_score=row["meta_score"])

    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if row_i["meta_score"] > row_j["meta_score"]:
                weight = row_i["meta_score"] - row_j["meta_score"]
                G.add_edge(row_i["faction"], row_j["faction"], weight=weight)
    return G