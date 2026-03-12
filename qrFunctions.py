import numpy as np
import pandas as pd

def shift_orderbook_before_event(df, K):

    df_shift = df.copy()

    for i in range(1, K+1):

        df_shift[f"Q_{i}"] = df_shift[f"Q_{i}"].shift(1)
        df_shift[f"Q_-{i}"] = df_shift[f"Q_-{i}"].shift(1)

    return df_shift


def build_qr_sequence(df, level):

    q_col = f"Q_{level}"

    d = df[df["lvl"] == level]

    seq = pd.DataFrame({
        "eta": d["type"],
        "q": d[q_col],
        "dt": d["dtk_l"]
    })

    return seq.dropna()


def qr_mle_from_sequence(seq):

    T = seq.groupby("q")["dt"].sum()

    N = seq.groupby(["q","eta"]).size().unstack(fill_value=0)

    for col in ["L","C","M"]:
        if col not in N.columns:
            N[col] = 0

    N = N[["L","C","M"]]

    lam = N.div(T, axis=0)

    lam["Lambda"] = lam.sum(axis=1)

    return lam.sort_index(), N.sort_index(), T.sort_index()


def build_all_sequences(df, K):

    seqs = {}

    for level in range(-K, K+1):

        if level == 0:
            continue

        seqs[level] = build_qr_sequence(df, level)

    return seqs

def estimate_all_qr(seqs):

    lambdas = {}
    Ns = {}
    Ts = {}

    for level, seq in seqs.items():

        lam, N, T = qr_mle_from_sequence(seq)

        lambdas[level] = lam
        Ns[level] = N
        Ts[level] = T

    return lambdas, Ns, Ts


EVENT_NAME = {"L": "limit", "C": "cancel", "M": "trade"}
RAW_ORDER = ["C", "L", "M"]
FINAL_ORDER = ["cancel", "limit", "trade"]

def qr_transition_matrix(Ns, K=5):
    # agrège les comptages N(q, eta) sur tous les niveaux et tous les q
    total_counts = pd.Series({"L": 0.0, "C": 0.0, "M": 0.0})

    for level in range(-K, K + 1):
        if level == 0:
            continue

        N = Ns[level].copy()

        for col in ["L", "C", "M"]:
            if col not in N.columns:
                N[col] = 0

        total_counts += N[["L", "C", "M"]].sum(axis=0)

    p = total_counts / total_counts.sum()

    row = pd.Series({
        "cancel": p["C"],
        "limit": p["L"],
        "trade": p["M"]
    })

    mat = pd.DataFrame([row, row, row], index=FINAL_ORDER)
    mat = mat[FINAL_ORDER]

    return mat