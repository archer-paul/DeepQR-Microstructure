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

    seq = pd.DataFrame({
        "eta": df["type"],
        "q": df[q_col],
        "dt": df["dtk_l"]
    })

    seq = seq.dropna()

    return seq


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

def compute_hourly_intensity_qr(lambdas, df_train):
    qr_trade_series = pd.concat({lvl: df["M"] for lvl, df in lambdas.items()})
    df_qr = df_train.copy()
    
    df_qr["lambda_qr_trade"] = df_qr.set_index(["lvl", "q_before_event"]).index.map(qr_trade_series)
    df_qr["lambda_qr_trade"] = df_qr["lambda_qr_trade"].fillna(0.0)
    df_qr["expected_qr_trades"] = df_qr["lambda_qr_trade"] * df_qr["dtk_l"]
    grouped_qr = df_qr.groupby("hour_last_event")
    hourly_qr = grouped_qr["expected_qr_trades"].sum() / grouped_qr["dtk_l"].sum()
    
    global_expected_trades = (df_qr["lambda_qr_trade"] * df_qr["dtk_l"]).sum()
    global_time = df_qr["dtk_l"].sum()
    global_qr_intensity = global_expected_trades / global_time
    
    return hourly_qr, global_qr_intensity