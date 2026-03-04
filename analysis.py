from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from events import SimulationResult, EventType
from state import LOBState
from intensities import IntensityModel


_KIND_MAP = {0:"L", 1:"C", 2:"M"}
KindStr = Literal["L", "C", "M"]


def times_and_sales(res: SimulationResult) -> pd.DataFrame:
    """
    Times & Sales style table:
        - one row per event
        - includes event time, signed level, side, type, dq, q_before, q_after
    """
    kinds_str = np.array([_KIND_MAP[int(k)] for k in res.kinds], dtype=object)
    side = np.where(res.levels < 0, "BID", "ASK")
    dq = np.where(kinds_str == "L", 1, -1)
    
    df = pd.DataFrame(
        {
            "t"         : res.times,
            "side"      : side,
            "level"     : res.levels,
            "dist"      : np.abs(res.levels),
            "type"      : kinds_str,
            "dq"        : dq,
            "q_before"  : res.q_before,
            "q_after"   : res.q_after
        }
    )
    return df


def build_book_snapshots(
    state0: LOBState,
    res: SimulationResult,
    *,
    dt: float = 0.01,
    T: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Times & Sales style table:
        - one row per event
        - includes event time, signed level, side, type, dq, q_before, q_after
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    
    if T is None:
        T = float(res.times[-1] if res.n_events() > 0 else 0.0)
    T = float(T)
    
    t_grid = np.arange(0.0, T+1e-12, dt, dtype=float)
    M = t_grid.shape[0]
    L = state0.n_levels
    
    q_grid = np.empty((M, L), dtype=np.int32)
    q = state0.q.astype(np.int32).copy()
    
    # event pointer
    e = 0
    nE = res.n_events()
    
    # Map level -> state index once
    lvl_to_idx = state0.level_to_index
    
    # For speed: pre-extract arrays
    ev_t = res.times
    ev_lvl = res.levels
    ev_kind = res.kinds
    
    for m, tg in enumerate(t_grid):
        # apply all events with time <= tg
        while e<nE and ev_t[e]<=tg:
            lvl = int(ev_lvl[e])
            j = lvl_to_idx[lvl]
            k = int(ev_kind[e])
            # L => +1, C/M => -1
            q[j] += 1 if k == 0 else -1
            e += 1
        q_grid[m, :] = q
    return t_grid, q_grid


def plot_lob_heatmap(
    t_grid: np.ndarray,
    q_grid: np.ndarray,
    levels: np.ndarray,
    *,
    title: str = "LOB evolution (queue sizes)",
) -> None:
    """
    Heatmap: y-axis = signed levels, x-axis = time, color = queue size
    """
    # Put levels on y-axis in the natural order [-K..-1, +1..+K]
    # q_grid is already aligned with `levels`
    fig, ax = plt.subplots()
    im = ax.imshow(
        q_grid.T,
        aspect="auto",
        origin="lower",
        extent=[t_grid[0], t_grid[-1], 0, len(levels)],
    )

    # y ticks as actual signed levels
    ax.set_yticks(np.arange(len(levels)) + 0.5)
    ax.set_yticklabels([str(int(l)) for l in levels])

    ax.set_xlabel("time")
    ax.set_ylabel("level (signed)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="queue size")
    plt.show()


def plot_best_levels(t_grid: np.ndarray, q_grid: np.ndarray, levels: np.ndarray) -> None:
    """
    Plot best bid and best ask (closest nonzero queues to 0) over time.
    """
    # indices for bid and ask in this ordering
    bid_mask = levels < 0
    ask_mask = levels > 0

    best_bid = np.full_like(t_grid, fill_value=np.nan, dtype=float)
    best_ask = np.full_like(t_grid, fill_value=np.nan, dtype=float)

    # closest to 0 are -1 for bids, +1 for asks
    bid_levels = levels[bid_mask]          # [-K..-1]
    ask_levels = levels[ask_mask]          # [1..K]
    bid_q = q_grid[:, bid_mask]
    ask_q = q_grid[:, ask_mask]

    # best bid: highest (closest to 0) => last column in bid_levels corresponds to -1
    # find, for each t, the closest-to-0 level with q>0
    for i in range(len(t_grid)):
        nzb = np.where(bid_q[i, :] > 0)[0]
        if nzb.size > 0:
            best_bid[i] = bid_levels[nzb.max()]  # closest to 0 among bids
        nza = np.where(ask_q[i, :] > 0)[0]
        if nza.size > 0:
            best_ask[i] = ask_levels[nza.min()]  # closest to 0 among asks

    plt.figure()
    plt.plot(t_grid, best_bid, label="best bid level")
    plt.plot(t_grid, best_ask, label="best ask level")
    plt.xlabel("time")
    plt.ylabel("signed level")
    plt.title("Best bid/ask levels over time")
    plt.legend()
    plt.show()


def reconstruct_states_at_events(state0, res):
    """
    Returns:
        q_path: (n_events+1, 2K) queue sizes after each event (including t=0 initial)
        t_path: (n_events+1,) times (t=0 plus event times)
    """
    nE = res.n_events()
    q_path = np.empty((nE + 1, state0.n_levels), dtype=np.int32)
    t_path = np.empty(nE + 1, dtype=float)

    q = state0.q.astype(np.int32).copy()
    q_path[0] = q
    t_path[0] = 0.0

    lvl_to_idx = state0.level_to_index

    for k in range(nE):
        lvl = int(res.levels[k])
        j = lvl_to_idx[lvl]
        kind = int(res.kinds[k])
        q[j] += 1 if kind == 0 else -1
        q_path[k + 1] = q
        t_path[k + 1] = float(res.times[k])

    return t_path, q_path


def animate_lob_with_tns(
    state0,
    res,
    *,
    step: int = 5,               # frame stride in events
    last_n_events: int = 30,      # number of rows shown in table
    title: str = "LOB ladder + Times & Sales",
):
    """
    Animated Plotly figure:
        - Left: bar chart of queue sizes vs signed level
        - Right: times & sales table (last_n_events up to the frame)
    """
    levels = state0.levels.astype(int)
    t_path, q_path = reconstruct_states_at_events(state0, res)
    df_tns = times_and_sales(res)

    # pre-split bid/ask indices for style
    bid_mask = levels < 0
    ask_mask = levels > 0

    def make_frame(k):
        # frame uses state after event index k (k from 0..nE)
        qk = q_path[k]
        tk = t_path[k]

        # LOB ladder traces
        bar_bid = go.Bar(
            x=levels[bid_mask],
            y=qk[bid_mask],
            name="BID queues",
        )
        bar_ask = go.Bar(
            x=levels[ask_mask],
            y=qk[ask_mask],
            name="ASK queues",
        )

        # Times & Sales slice (events up to k-1)
        end_evt = max(0, k)  # number of events included
        sub = df_tns.iloc[max(0, end_evt - last_n_events):end_evt].copy()

        # Format table columns (keep it compact)
        if len(sub) == 0:
            header = ["t", "side", "level", "type", "q_before", "q_after"]
            cells = [[""], [""], [""], [""], [""], [""]]
        else:
            sub["t"] = sub["t"].map(lambda x: f"{x:.4f}")
            header = ["t", "side", "level", "type", "q_before", "q_after"]
            cells = [
                sub["t"].tolist(),
                sub["side"].tolist(),
                sub["level"].tolist(),
                sub["type"].tolist(),
                sub["q_before"].tolist(),
                sub["q_after"].tolist(),
            ]

        table = go.Table(
            header=dict(values=header),
            cells=dict(values=cells),
        )

        frame = go.Frame(
            name=str(k),
            data=[bar_bid, bar_ask, table],
            layout=go.Layout(
                title_text=f"{title} — t={tk:.4f}, events={k}"
            ),
        )
        return frame

    # Base figure with subplots (bar + table)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("LOB ladder (x=level, y=queue size)", f"Times & Sales (last {last_n_events})"),
        horizontal_spacing=0.08
    )

    # initial frame (k=0)
    init = make_frame(0)
    for tr in init.data:
        fig.add_trace(tr, row=1, col=1 if tr.type != "table" else 2)

    # x/y axis config
    fig.update_xaxes(title_text="signed level", row=1, col=1)
    fig.update_yaxes(title_text="queue size", row=1, col=1)
    fig.update_xaxes(type="category", row=1, col=1)  # keep integer ticks as categories
    fig.update_layout(barmode="group", template="plotly_dark", height=600, width=1200)

    # frames
    nE = res.n_events()
    frame_indices = list(range(0, nE + 1, step))
    frames = [make_frame(k) for k in frame_indices]
    fig.frames = frames

    # slider
    sliders = [{
        "steps": [
            {
                "method": "animate",
                "args": [[str(k)], {"mode": "immediate", "frame": {"duration": 40, "redraw": True}}],
                "label": str(k),
            }
            for k in frame_indices
        ],
        "x": 0.1, "y": -0.12, "len": 0.8,
        "currentvalue": {"prefix": "event: "}
    }]

    # play/pause
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.5, "y": -0.18, "xanchor": "center",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                    "args": [None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                    "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ],
        }],
        sliders=sliders,
    )

    return fig


@dataclass(frozen=True)
class EmpiricalIntensityEstimate:
    """
    For each level:
        occupancy[level][n] = total time queue stayed at size n
        counts[level][kind][n] = number of events of 'kind' occurring when pre-size was n
        lambda_hat[level][kind][n] = counts / occupancy  (0 if occupancy=0)
        pi_emp[level][n] = occupancy / total_time
    """
    n_max: int
    total_time: float
    occupancy: Dict[int, np.ndarray]
    counts: Dict[int, Dict[KindStr, np.ndarray]]
    lambda_hat: Dict[int, Dict[KindStr, np.ndarray]]
    pi_emp: Dict[int, np.ndarray]


def estimate_empirical_intensities(
    state0: LOBState,
    res: SimulationResult,
    *,
    n_max: int = 50,
    T: Optional[float] = None,
) -> EmpiricalIntensityEstimate:
    """
    Empirical estimator:
        $\hat{lambda}(n) = N_events_at_state_n / time_spent_in_state_n$

    We do it level-by-level using only events affecting that level.
    Between two events at the same level, q_i is constant.

    Parameters
    - n_max: maximum queue size bin (values > n_max are clipped into n_max)
    - T: horizon; if None uses last event time
    """
    if n_max <= 0:
        raise ValueError("n_max must be >= 1")

    if res.n_events() == 0:
        raise ValueError("SimulationResult has no events")

    if T is None:
        T = float(res.times[-1])
    T = float(T)
    if T <= 0:
        raise ValueError("T must be > 0")

    levels = state0.levels.astype(int).tolist()
    lvl_to_idx = state0.level_to_index

    # init containers
    occupancy: Dict[int, np.ndarray] = {lvl: np.zeros(n_max + 1, dtype=float) for lvl in levels}
    counts: Dict[int, Dict[KindStr, np.ndarray]] = {
        lvl: {k: np.zeros(n_max + 1, dtype=np.int64) for k in ("L", "C", "M")} for lvl in levels
    }

    # per-level current queue and last time
    q_curr = {lvl: int(state0.get(lvl)) for lvl in levels}
    t_last = {lvl: 0.0 for lvl in levels}

    # process global events in chronological order
    for k in range(res.n_events()):
        t = float(res.times[k])
        lvl = int(res.levels[k])
        kind = _KIND_MAP[int(res.kinds[k])]

        # time spent in current state since last event at this level
        dt = t - t_last[lvl]
        if dt < -1e-12:
            raise ValueError("Non-monotone times detected")

        q0 = q_curr[lvl]
        qbin = min(max(q0, 0), n_max)
        occupancy[lvl][qbin] += max(dt, 0.0)

        # count event at pre-state q0
        counts[lvl][kind][qbin] += 1

        # update queue size
        if kind == "L":
            q_curr[lvl] = q0 + 1
        else:
            # C or M
            q_curr[lvl] = max(q0 - 1, 0)

        t_last[lvl] = t

    # add tail time from last event at level to horizon T
    for lvl in levels:
        dt = T - t_last[lvl]
        if dt > 0:
            q0 = q_curr[lvl]
            qbin = min(max(q0, 0), n_max)
            occupancy[lvl][qbin] += dt

    # compute lambda_hat and empirical stationary distribution pi_emp
    lambda_hat: Dict[int, Dict[KindStr, np.ndarray]] = {lvl: {} for lvl in levels}
    pi_emp: Dict[int, np.ndarray] = {}

    for lvl in levels:
        occ = occupancy[lvl]
        total = occ.sum()
        pi = occ / total if total > 0 else np.zeros_like(occ)
        pi_emp[lvl] = pi

        for kind in ("L", "C", "M"):
            lam = np.zeros_like(occ, dtype=float)
            mask = occ > 0
            lam[mask] = counts[lvl][kind][mask] / occ[mask]
            lambda_hat[lvl][kind] = lam

    return EmpiricalIntensityEstimate(
        n_max=n_max,
        total_time=T,
        occupancy=occupancy,
        counts=counts,
        lambda_hat=lambda_hat,
        pi_emp=pi_emp,
    )


def plot_intensities_model1(
    est: EmpiricalIntensityEstimate,
    intensity: IntensityModel,
    *,
    K: int,
    n_max: Optional[int] = None,
    use_empirical: bool = True,
    title_suffix: str = "Model I",
) -> None:
    """
    Plots lambda^L(n), lambda^C(n), lambda^M(n) for i=1..K.
    If use_empirical=True: plot empirical estimates (solid) + model intensity (dashed).
    If use_empirical=False: only model intensity (solid).
    """
    n_max = est.n_max if n_max is None else int(n_max)
    n = np.arange(0, n_max + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

    panels = [  ("L", "Limit order insertion intensity"),
                ("C", "Limit order cancellation intensity"),
                ("M", "Market order arrival intensity")]

    for ax, (kind, ttl) in zip(axes, panels):
        for i in range(1, K + 1):
            lvl = +i  # use ask side; by symmetry it represents ±i
            # model curve
            if kind == "L":
                model_curve = np.array([intensity.lambda_L(lvl, int(x)) for x in n], dtype=float)
            elif kind == "C":
                model_curve = np.array([intensity.lambda_C(lvl, int(x)) for x in n], dtype=float)
            else:
                model_curve = np.array([intensity.lambda_M(lvl, int(x)) for x in n], dtype=float)

            label_i = ["First limit", "Second limit", "Third limit"][i - 1] if K <= 3 else f"Limit {i}"

            if use_empirical:
                emp_curve = est.lambda_hat[lvl][kind][: n_max + 1]
                ax.plot(n, emp_curve, marker="o", markersize=2, linewidth=1.2, label=label_i)
                #ax.plot(n, model_curve, linestyle="--", linewidth=1.2)
            else:
                ax.plot(n, model_curve, marker="o", markersize=2, linewidth=1.2, label=label_i)

        ax.set_title(f"{ttl}, {title_suffix}")
        ax.set_xlabel("Queue Size (bins)")
        ax.set_ylabel("Intensity (events per unit time)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)

    plt.tight_layout()
    plt.show()


def stationary_birth_death(
    lam_plus: np.ndarray,   # length n_max+1, lam_plus[n] = rate from n -> n+1
    lam_minus: np.ndarray,  # length n_max+1, lam_minus[n] = rate from n -> n-1 (lam_minus[0] unused / 0)
) -> np.ndarray:
    """
    Truncated stationary distribution on {0,...,n_max}:
        pi[n] ∝ prod_{j=1..n} lam_plus[j-1] / lam_minus[j]
    Then normalized.
    """
    n_max = len(lam_plus) - 1
    pi = np.zeros(n_max + 1, dtype=float)
    pi[0] = 1.0
    for n in range(1, n_max + 1):
        denom = lam_minus[n]
        if denom <= 0:
            pi[n] = 0.0
        else:
            pi[n] = pi[n - 1] * (lam_plus[n - 1] / denom)

    s = pi.sum()
    return pi / s if s > 0 else pi


def plot_invariant_distributions_model1(
    est: EmpiricalIntensityEstimate,
    intensity: IntensityModel,
    *,
    K: int,
    n_max: Optional[int] = None,
    title: str = "Model I invariant distributions",
) -> None:
    """
    For i=1..K (using +i by symmetry):
        - Empirical distribution from occupancy time
        - Model I stationary distribution implied by intensity model
        - Poisson benchmark: constant birth/death rates equal to empirical averages
    """
    n_max = est.n_max if n_max is None else int(n_max)
    n = np.arange(0, n_max + 1)

    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4), sharey=True)
    if K == 1:
        axes = [axes]

    for idx, i in enumerate(range(1, K + 1)):
        ax = axes[idx]
        lvl = +i

        # empirical pi from occupancy
        pi_emp = est.pi_emp[lvl][: n_max + 1]

        # Model I implied pi
        lam_plus = np.array([intensity.lambda_L(lvl, int(x)) for x in n], dtype=float)
        lam_minus = np.array([0.0] + [
            float(intensity.lambda_C(lvl, int(x)) + intensity.lambda_M(lvl, int(x)))
            for x in n[1:]
        ], dtype=float)
        pi_model = stationary_birth_death(lam_plus, lam_minus)

        # Poisson benchmark: constant rates = empirical average intensities
        # avg rate = total count / total time (per kind); deaths are C+M
        occ = est.occupancy[lvl][: n_max + 1]
        total_time = occ.sum() if occ.sum() > 0 else 1.0
        lamL_bar = est.counts[lvl]["L"][: n_max + 1].sum() / total_time
        lamD_bar = (est.counts[lvl]["C"][: n_max + 1].sum() + est.counts[lvl]["M"][: n_max + 1].sum()) / total_time

        lam_plus_p = np.full(n_max + 1, lamL_bar, dtype=float)
        lam_minus_p = np.full(n_max + 1, lamD_bar, dtype=float)
        lam_minus_p[0] = 0.0
        pi_pois = stationary_birth_death(lam_plus_p, lam_minus_p)

        # plot
        ax.plot(n, pi_emp, marker="o", markersize=2, linewidth=1.2, label="Empirical estimation")
        ax.plot(n, pi_model, marker="o", markersize=2, linewidth=1.2, label="Model I")
        ax.plot(n, pi_pois, marker="o", markersize=2, linewidth=1.2, label="Poisson model")

        ax.set_title(["First limit", "Second limit", "Third limit"][i - 1] if K <= 3 else f"Limit {i}")
        ax.set_xlabel("Queue Size (bins)")
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Distribution")
        ax.legend(frameon=True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()