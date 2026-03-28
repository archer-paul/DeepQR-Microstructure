import numpy as np
import pandas as pd


def extract_price_changes(df, price_col="price", time_col="time"):
    d = df[df["type"] == "M"].copy()
    changed = d[price_col].ne(d[price_col].shift())
    d = d.loc[changed].reset_index(drop=True)
    
    d = d[[time_col, price_col]].copy()
    d["dp"] = d[price_col].diff()
    d = d.dropna().reset_index(drop=True)
    
    return d


def compute_alternation_continuation(pc, tick_size=None):
    pc = pc.copy()
    
    if tick_size is None:
        tick_size = np.min(np.abs(pc["dp"]))
    
    pc["sign"] = np.sign(pc["dp"]).astype(int)
    pc["L"]    = (pc["dp"].abs() / tick_size).astype(int)

    pc["sign_prev"] = pc["sign"].shift()
    pc = pc.dropna().reset_index(drop=True)
    pc["sign_prev"] = pc["sign_prev"].astype(int)

    pc["is_cont"] = pc["sign"] == pc["sign_prev"]
    pc["is_alt"]  = pc["sign"] == -pc["sign_prev"]

    return pc


def estimate_eta_hat(pc):
    results = []

    for k in sorted(pc["L"].unique()):
        sub = pc[pc["L"]==k]
        
        Nc = sub["is_cont"].sum()
        Na = sub["is_alt"].sum()
        
        if Na > 0:
            u_k = 0.5 * ( k*(Nc/Na - 1) + 1 )
        else :
            u_k = np.nan
        
        results.append({
                "k": k,
                "Nc": Nc,
                "Na": Na,
                "u_k": u_k,
                "N_total": Nc + Na
            })

    res = pd.DataFrame(results)
    res["lambda_k"] = res["N_total"] / res["N_total"].sum()
    
    eta_hat = np.nansum(res["lambda_k"] * res["u_k"])

    return res, eta_hat


def compute_efficient_price(pc, eta_hat, tick_size):
    pc = pc.copy()
    pc["X_hat"] = pc["price"] - tick_size * (0.5 - eta_hat) * pc["sign"]
    return pc


def compute_realized_variance(pc):
    # Calculate the log of the efficient price
    pc = pc.copy()
    pc['log_X_hat'] = np.log(pc['X_hat'])
    
    # Calculate the squared differences
    pc['log_return_sq'] = pc['log_X_hat'].diff() ** 2
    
    # Sum them to get the Realized Variance (RV)
    rv = pc['log_return_sq'].sum()
    
    return rv


def rolling_microstructure_volatility(
    pc,
    window_seconds=600,
    tick_size=None,
    min_price_changes=30,
    annualization_seconds=252 * 6.5 * 60 * 60
):
    """
    Returns a row-level dataframe in the same style as your screenshot:
    one row per price-changing transaction, with both static and rolling columns.
    """

    df = pc.copy().reset_index(drop=True)

    # Infer tick size if needed
    if tick_size is None:
        tick_size = np.min(np.abs(df["dp"][df["dp"] != 0]))

    # Ensure the base microstructure columns exist
    if "sign" not in df.columns or "L" not in df.columns or \
       "sign_prev" not in df.columns or "is_cont" not in df.columns or "is_alt" not in df.columns:
        df = compute_alternation_continuation(df, tick_size=tick_size)

    times = df["time"].values

    eta_hat_rolling = []
    rolling_rv = []
    annualized_vol = []
    X_hat_rolling = []

    left = 0
    for right in range(len(df)):
        t_end = times[right]
        t_start = t_end - window_seconds

        while left < len(times) and times[left] < t_start:
            left += 1

        window_df = df.iloc[left:right + 1].copy()

        # Default NaNs if not enough data
        eta_val = np.nan
        rv_val = np.nan
        vol_val = np.nan
        xhat_val = np.nan

        if len(window_df) >= min_price_changes:
            # Recompute alternation/continuation INSIDE the window
            # to stay methodologically consistent with Code 2
            window_ac = compute_alternation_continuation(
                window_df[["time", "price", "dp"]].copy(),
                tick_size=tick_size
            )

            if len(window_ac) >= max(10, min_price_changes - 1):
                _, eta_window = estimate_eta_hat(window_ac)

                if np.isfinite(eta_window):
                    # Rolling efficient price on that window
                    window_eff = compute_efficient_price(window_ac, eta_window, tick_size)

                    # Rolling RV on that window
                    rv_window = compute_realized_variance(window_eff)

                    ann_var = rv_window * (annualization_seconds / window_seconds)
                    vol_window = np.sqrt(ann_var)

                    # Need the rolling efficient price aligned to the current row
                    # window_ac loses the first row because of shift(), so the last row
                    # in window_eff corresponds to the current right endpoint.
                    eta_val = eta_window
                    rv_val = rv_window
                    vol_val = vol_window
                    xhat_val = window_eff["X_hat"].iloc[-1]

        eta_hat_rolling.append(eta_val)
        rolling_rv.append(rv_val)
        annualized_vol.append(vol_val)
        X_hat_rolling.append(xhat_val)

    # Attach rolling columns
    df["eta_hat_rolling"] = eta_hat_rolling
    df["X_hat_rolling"] = X_hat_rolling
    df["log_X_hat_rolling"] = np.log(df["X_hat_rolling"])
    df["log_return_sq"] = df["log_X_hat_rolling"].diff() ** 2
    df["rolling_RV"] = rolling_rv
    df["annualized_vol"] = annualized_vol

    df = df.dropna().reset_index(drop=True)
    return df