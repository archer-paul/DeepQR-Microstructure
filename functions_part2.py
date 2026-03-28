import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# (1) Simulate sigma_t and logS_t on a fine grid for Example 5 (BM-based SV)
# ============================================================
def simulate_example5_sigma_and_S(
    T: float,
    L: int,
    m: int,
    s0: float = 1.0,
    sigma0: float = 1.0,
    seed: int | None = None,
):
    """
    Example 5: dS_t = sigma_t S_t dB_t, sigma_t = sigma0 * |W_t|
    Simulate on a fine grid with n_steps = L*m increments.

    Returns:
      t_fine (n_steps+1),
      sigma_fine (n_steps+1),
      logS_fine (n_steps+1),
      W_fine (n_steps+1),
      dt_fine
    """
    rng = np.random.default_rng(seed)
    n_steps = L * m
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # independent Brownian increments
    dW = np.sqrt(dt) * rng.standard_normal(n_steps)
    dB = np.sqrt(dt) * rng.standard_normal(n_steps)

    W = np.empty(n_steps + 1)
    W[0] = 0.0
    W[1:] = np.cumsum(dW)

    sigma = sigma0 * np.abs(W)

    logS = np.empty(n_steps + 1)
    logS[0] = np.log(s0)

    # Euler for logS using sigma at left endpoint
    for i in range(n_steps):
        logS[i + 1] = logS[i] - 0.5 * sigma[i] ** 2 * dt + sigma[i] * dB[i]

    return t, sigma, logS, W, dt

# ============================================================
# (2) Coarse (non-overlapping) realized volatility from fine data
# ============================================================
def realized_volatility_coarse_from_fine(
    logS_fine: np.ndarray,
    sigma_fine: np.ndarray,
    dt_fine: float,
    m: int,
):
    """
    Build coarse-grid objects from fine simulation.

    Fine grid:
      - logS_fine length = n_steps+1
      - sigma_fine length = n_steps+1

    Coarse grid:
      - L = n_steps // m
      - RV_j = sqrt(sum_{i=(j-1)m}^{jm-1} (Δ logS_i)^2), j=1..L  (non-overlapping)
      - sigma_proxy_j = RV_j / sqrt(Delta), Delta = m*dt_fine
      - sigma_coarse_k = sigma_fine[k*m], k=0..L

    Returns:
      t_coarse (L+1),
      sigma_coarse (L+1),
      RV (L),
      sigma_proxy (L),
      Delta
    """
    r = np.diff(logS_fine)  # length n_steps
    n_steps = len(r)

    if n_steps % m != 0:
        raise ValueError("Need n_steps divisible by m. Use n_steps = L*m in simulation.")

    L = n_steps // m
    Delta = m * dt_fine

    r_blocks = r.reshape(L, m)                 # (L, m)
    RV2 = np.sum(r_blocks * r_blocks, axis=1)  # (L,)
    RV = np.sqrt(RV2)

    sigma_proxy =  RV / np.sqrt(Delta)        # (L,)

    sigma_coarse = sigma_fine[::m]             # (L+1,)
    t_coarse = np.linspace(0.0, L * Delta, L + 1)  # should match [0, T]

    RV = np.concatenate(([RV[0]], RV))                 # length L+1
    sigma_proxy = np.concatenate(([sigma_proxy[0]], sigma_proxy))  # length L+1

    return t_coarse, sigma_coarse, RV, sigma_proxy, Delta

# ============================================================
# (3) Plot "Figure 6 style" on the COARSE grid 
# ============================================================
def plot_figure6_style_coarse(
    t_fine: np.ndarray,
    sigma_fine: np.ndarray,
    logS_fine: np.ndarray,
    dt_fine: float,
    m: int,
    eps: float = 1e-12,
):
    t_c, sigma_c, RV, sigma_proxy, Delta = realized_volatility_coarse_from_fine(
        logS_fine=logS_fine,
        sigma_fine=sigma_fine,
        dt_fine=dt_fine,
        m=m
    )

    # Align at end of coarse intervals: j=1..L
    t_rv = t_c[1:]                 # length L
    sigma_aligned = sigma_c[1:]    # length L
    sigma_hat = sigma_proxy[1:]    # length L
    RV_aligned = RV[1:]            # length L (optional)

    err = sigma_aligned - sigma_hat
    log_err = np.log(sigma_aligned + eps) - np.log(sigma_hat + eps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(t_rv, sigma_hat, color="black",
                 label=r"$RV: \frac{RV}{\sqrt{\Delta}}$")
    axes[0].plot(t_rv, sigma_aligned, color="red", linewidth=0.7,
                 label=r"$IV: \sigma_t$")
    axes[0].set_title("IV (spot) vs RV (realised)")
    axes[0].set_xlabel("time t")
    axes[0].set_ylabel("volatility")
    axes[0].legend()

    axes[1].plot(t_rv, err, color="black", linewidth=0.7)
    axes[1].set_title("Estimation error")
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel(r"$IV - RV$")

    axes[2].plot(t_rv, log_err, color="black", linewidth=0.7)
    axes[2].set_title("Log-error")
    axes[2].set_xlabel("time t")
    axes[2].set_ylabel(r"$\log IV - \log RV$")

    plt.tight_layout()
    plt.show()

    return t_c, sigma_c, sigma_proxy, RV, Delta


# ============================================================
# (4) HR estimation functions using log regression
# ============================================================
def compute_m_q_delta(V, deltas, qs, eps=1e-12):
    """
    Compute m(q, Δ) = mean_t |log V_{t+Δ} - log V_t|^q  (Eq 14)
    for multiple Δ (lags) and q.

    V: 1D positive series (sigma_t or RV_t)
    deltas: array of integer lags (in sample points)
    qs: array of q values (positive)
    """
    V = np.asarray(V, dtype=float)

    # avoid log(0) issues (sigma_t hits 0 at t=0 and can get very small)
    logV = np.log(V + eps)

    deltas = np.asarray(deltas, dtype=int)
    qs = np.asarray(qs, dtype=float)

    m = np.empty((len(qs), len(deltas)), dtype=float)

    for j, d in enumerate(deltas):
        if d <= 0 or d >= len(logV):
            raise ValueError(f"Bad delta {d}. Must be in [1, len(V)-1].")

        inc = np.abs(logV[d:] - logV[:-d])   # length len(V)-d
        # compute mean |inc|^q for each q
        for i, q in enumerate(qs):
            m[i, j] = np.mean(inc ** q)

    return m  # shape (n_q, n_delta)

def log_regression_xi(m, deltas):
    """
    For each q: regress log m(q,Δ) vs log Δ -> slope ξ_q.
    Returns ξ_q and intercepts.
    """
    deltas = np.asarray(deltas, dtype=float)
    x = np.log(deltas)

    xi = np.empty(m.shape[0], dtype=float)
    intercept = np.empty(m.shape[0], dtype=float)

    for i in range(m.shape[0]):
        y = np.log(m[i, :])
        slope, b = np.polyfit(x, y, 1)
        xi[i] = slope
        intercept[i] = b

    return xi, intercept

def estimate_HR_from_xi(qs, xi, fit_intercept=True):
    """
    Fit ξ_q ≈ q * H_R (paper uses linear behaviour).
    If fit_intercept=False, force through origin: H_R = (q·ξ)/(q·q).
    """
    qs = np.asarray(qs, dtype=float)
    xi = np.asarray(xi, dtype=float)

    if fit_intercept:
        H_R, a = np.polyfit(qs, xi, 1)  # xi ~ H_R*q + a
        return H_R, a
    else:
        H_R = np.dot(qs, xi) / np.dot(qs, qs)
        return H_R, 0.0

def plot_scaling_and_xi(V, deltas, qs, title_left, title_right, eps=1e-12, fit_intercept=True):
    m = compute_m_q_delta(V, deltas, qs, eps=eps)
    xi, _ = log_regression_xi(m, deltas)
    H_R, a = estimate_HR_from_xi(qs, xi, fit_intercept=fit_intercept)

    fig, ax = plt.subplots(2, 1, figsize=(5, 8))

    # ---- Left: scaling curves log m vs log Δ for multiple q
    x = np.log(deltas)
    for i, q in enumerate(qs):
        y = np.log(m[i, :])
        
        # Trace les points originaux et récupère l'objet line pour extraire sa couleur
        line = ax[0].plot(x, y, marker="o", markersize=8, markerfacecolor="none", markeredgewidth=1.5, linewidth=0, label=f"q={q:g}")
        color = line[0].get_color() 
        
        # Calcule la régression linéaire (polynôme de degré 1)
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x + intercept
        
        # Ajoute la droite de régression sur le graphique (en pointillés)
        ax[0].plot(x, y_fit, linestyle="-", color=color, linewidth=1)
    ax[0].set_xlabel(r"$\log \Delta$")
    ax[0].set_ylabel(r"$\log m(q,\Delta)$")
    ax[0].set_title(title_left)
    ax[0].legend(ncol=2, fontsize=8)

    # ---- Right: xi_q vs q + fitted line
    ax[1].plot(qs, xi, marker="o", markersize=8, markerfacecolor="none", markeredgewidth=1.5, linestyle="none", color="black", linewidth=0)

    q_line = np.linspace(qs.min(), qs.max(), 200)
    ax[1].plot(q_line, H_R*q_line + a, color="black", linewidth=1)

    ax[1].set_xlabel("q")
    ax[1].set_ylabel(r"$\xi_q$")
    ax[1].set_title(title_right + rf" $\hat H_R$ = {H_R:.3f}", fontweight="bold")

    plt.tight_layout()
    plt.show()

    return H_R, xi, m


# ============================================================
# (1) Simulate sigma_t and logS_t on a fine grid for Example 6 (OU-SV)
# ============================================================
def simulate_example6_OUSV(
    T: float,
    L: int,
    m: int,
    s0: float = 1.0,
    sigma0: float = 1.0,
    Y0: float = 0.0,
    gamma: float = 1.0,
    theta: float = 1.0,
    seed: int | None = None,
):
    """
    Example 6 (OU-SV):
      dY_t = -gamma Y_t dt + theta dB'_t
      sigma_t = sigma0 * exp(Y_t)
      dlogS_t = -0.5 sigma_t^2 dt + sigma_t dB_t
    with B and B' independent.

    Simulate on fine grid with n_steps = L*m.

    Returns:
      t_fine, sigma_fine, logS_fine, Y_fine, dt_fine
    """
    rng = np.random.default_rng(seed)
    n_steps = L * m
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Independent Brownian increments
    dB  = np.sqrt(dt) * rng.standard_normal(n_steps)
    dBp = np.sqrt(dt) * rng.standard_normal(n_steps)

    # --- OU exact discretization for Y ---
    Y = np.empty(n_steps + 1)
    Y[0] = Y0

    if gamma == 0.0:
        # reduces to Brownian motion with drift 0
        for i in range(n_steps):
            Y[i+1] = Y[i] + theta * dBp[i]
    else:
        a = np.exp(-gamma * dt)
        ou_std = theta * np.sqrt((1.0 - np.exp(-2.0 * gamma * dt)) / (2.0 * gamma))
        Z = rng.standard_normal(n_steps)
        for i in range(n_steps):
            Y[i+1] = a * Y[i] + ou_std * Z[i]

    sigma = sigma0 * np.exp(Y)  # sigma on fine grid

    # --- log-price Euler with left endpoint sigma ---
    logS = np.empty(n_steps + 1)
    logS[0] = np.log(s0)
    for i in range(n_steps):
        logS[i+1] = logS[i] - 0.5 * sigma[i]**2 * dt + sigma[i] * dB[i]

    return t, sigma, logS, Y, dt

def summary_table_H(H_hat_RV_list, H_hat_IV_list):
    stats = {
        "Min.":      [np.min(H_hat_RV_list), np.min(H_hat_IV_list)],
        "5% quantile":  [np.quantile(H_hat_RV_list, 0.05), np.quantile(H_hat_IV_list, 0.05)],
        "25% quantile": [np.quantile(H_hat_RV_list, 0.25), np.quantile(H_hat_IV_list, 0.25)],
        "Median":   [np.median(H_hat_RV_list), np.median(H_hat_IV_list)],
        "Mean":     [np.mean(H_hat_RV_list), np.mean(H_hat_IV_list)],
        "75% quantile": [np.quantile(H_hat_RV_list, 0.75), np.quantile(H_hat_IV_list, 0.75)],
        "95% quantile": [np.quantile(H_hat_RV_list, 0.95), np.quantile(H_hat_IV_list, 0.95)],
        "Max.":     [np.max(H_hat_RV_list), np.max(H_hat_IV_list)],
    }

    df = pd.DataFrame(
        stats,
        index=["Realized volatility", "Instantaneous volatility"]
    ).T

    return df


# ============================================================
# (1) Simulate fractional OU volatility model
# ============================================================
def simulate_example7_fOUSV(
    T: float,
    L: int,
    m: int,
    H: float,
    s0: float = 1.0,
    sigma0: float = 1.0,
    Y0: float = 0.0,
    gamma: float = 1.0,
    theta: float = 1.0,
    seed: int | None = None,
    fbm_path_fn=None,  # function(T, n_steps, H, seed) -> array length n_steps+1
):
    """
    Example 7 (Fractional OU-SV):
      dY_t = -gamma Y_t dt + theta dB^H_t
      sigma_t = sigma0 * exp(Y_t)
      dlogS_t = -0.5 sigma_t^2 dt + sigma_t dB_t
    with B independent of B^H.

    Simulate on fine grid with n_steps = L*m.

    Requires an fBM path generator fbm_path_fn returning B^H on the grid.

    Returns:
      t_fine, sigma_fine, logS_fine, Y_fine, BH_fine, dt_fine
    """
    if fbm_path_fn is None:
        raise ValueError("Provide fbm_path_fn(T, n_steps, H, seed) -> fBM path of length n_steps+1")

    rng = np.random.default_rng(seed)
    n_steps = L * m
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # Brownian increments for price (independent of fBM)
    dB = np.sqrt(dt) * rng.standard_normal(n_steps)

    # Fractional Brownian motion path + increments
    # Use a different seed stream so you don't correlate it with dB by accident
    BH = fbm_path_fn(T, n_steps, H, None if seed is None else seed + 10_000)
    dBH = np.diff(BH)  # length n_steps

    # fOU recursion (discrete-time)
    Y = np.empty(n_steps + 1)
    Y[0] = Y0
    a = np.exp(-gamma * dt)
    for i in range(n_steps):
        Y[i + 1] = a * Y[i] + theta * dBH[i]

    sigma = sigma0 * np.exp(Y)

    logS = np.empty(n_steps + 1)
    logS[0] = np.log(s0)
    for i in range(n_steps):
        logS[i + 1] = logS[i] - 0.5 * sigma[i] ** 2 * dt + sigma[i] * dB[i]

    return t, sigma, logS, Y, BH, dt

# ------------------------------------------------------------
# Figure 14 plotting (Price, RV, IV across H values)
# ------------------------------------------------------------
def plot_figure14(
    t_coarse: np.ndarray,
    Hs: list[float],
    S_coarse_paths: list[np.ndarray],
    RV_coarse_paths: list[np.ndarray],
    IV_coarse_paths: list[np.ndarray],
):
    """
    Reproduce Fig. 14 style:
      Row 1: Price S_t
      Row 2: Realized volatility RV_t (Eq 12, window = m fine points, non-overlapping here)
      Row 3: Spot / instantaneous volatility sigma_t

    Each column corresponds to one H in Hs.
    All paths must be on the same coarse grid t_coarse and have length len(t_coarse).
    """
    nH = len(Hs)
    fig, axes = plt.subplots(
        nH, 3,
        figsize=(16, 2.5*nH),
        sharex=True,
        sharey=False
    )

    # If nH == 1, axes is 1D; normalize to 2D indexing
    if nH == 1:
        axes = np.array(axes).reshape(3, 1)

    for j, H in enumerate(Hs):
        S  = S_coarse_paths[j]
        RV = RV_coarse_paths[j]
        IV = IV_coarse_paths[j]

        assert len(S)  == len(t_coarse)
        assert len(RV) == len(t_coarse)
        assert len(IV) == len(t_coarse)

        # Column title = H
        axes[j, 0].set_title(f"H = {H:.2f}")

        # Row 1: price
        axes[j, 0].plot(t_coarse, S, linewidth=0.9, color="black")
        if j == 0:
            axes[j, 0].set_ylabel("Price $S_t$")

        # Row 2: realized volatility (unscaled RV)
        axes[j, 1].plot(t_coarse, RV, linewidth=0.9, color="black")
        if j == 0:
            axes[j, 1].set_ylabel(r"Realized vol $RV_t$")

        # Row 3: instantaneous volatility
        axes[j, 2].plot(t_coarse, IV, linewidth=0.9, color="black")
        if j == 0:
            axes[j, 2].set_ylabel(r"Spot vol $\sigma_t$")
        axes[j, 2].set_xlabel("t")

    plt.tight_layout()
    plt.show()
    return fig