#!/usr/bin/env python3
"""Generate Notebook 1: Data Exploration and the Queue-Reactive (QR) Model."""
import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

def md(src):
    lines = src.strip('\n').split('\n')
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + '\n' for l in lines[:-1]] + [lines[-1]]}

def code(src):
    lines = src.strip('\n').split('\n')
    return {"cell_type": "code", "metadata": {},
            "source": [l + '\n' for l in lines[:-1]] + [lines[-1]],
            "outputs": [], "execution_count": None}

cells = []

# ==============================================================================
# TITLE
# ==============================================================================
cells.append(md(r"""# Notebook 1 -- Data Exploration and the Queue-Reactive (QR) Model

**Paper:** *Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation* (Bodor & Carlier, 2025)

**Objective:** In this notebook we:
1. Load and explore LOBSTER limit order book data (NASDAQ equities as a proxy for the Bund futures used in the paper).
2. Derive and implement the analytical Maximum Likelihood Estimator (MLE) of the Queue-Reactive (QR) model.
3. Simulate LOB dynamics using the Gillespie algorithm.
4. Validate the model against basic stylized facts (transition matrices, queue distributions, return distributions).

This notebook establishes the **baseline** model. Subsequent notebooks extend it to the Deep Queue-Reactive (DQR) and Multidimensional Deep Queue-Reactive (MDQR) models."""))

# ==============================================================================
# IMPORTS
# ==============================================================================
cells.append(code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
np.random.seed(42)

print("All imports successful.")"""))

# ==============================================================================
# SECTION 1: LOBSTER Data
# ==============================================================================
cells.append(md(r"""---
## 1. The LOBSTER Dataset

[LOBSTER](https://lobsterdata.com/) (Limit Order Book System -- The Efficient Reconstructor) provides historical limit order book data for NASDAQ-listed securities. Each trading day is represented by two synchronized files:

| File | Columns | Description |
|------|---------|-------------|
| **Message** | `time, type, order_id, size, price, direction` | Event-level information |
| **Orderbook** | $4K$ columns (ask_px, ask_sz, bid_px, bid_sz) $\times K$ levels | LOB snapshot **after** the event |

**Event types:**

| Code | Meaning | Our label |
|------|---------|-----------|
| 1 | New limit order | $L$ (Limit) |
| 2 | Partial cancellation | $C$ (Cancel) |
| 3 | Total deletion | $C$ (Cancel) |
| 4 | Execution of visible limit order | $M$ (Market) |
| 5 | Execution of hidden limit order | $M$ (Market) |
| 7 | Trading halt | Ignored |

**Important alignment rule:** Row $k$ of the orderbook file represents the LOB state **after** event $k$. Therefore, the pre-event state for event $k$ is given by orderbook row $k-1$.

We use sample data for 5 stocks on 2012-06-21, with $K = 5$ price levels per side."""))

cells.append(code("""# --- Constants ---
DATA_DIR = Path("data")
STOCKS = {
    "AAPL": "LOBSTER_SampleFile_AAPL_2012-06-21_5",
    "INTC": "LOBSTER_SampleFile_INTC_2012-06-21_5",
    "GOOG": "LOBSTER_SampleFile_GOOG_2012-06-21_5",
    "MSFT": "LOBSTER_SampleFile_MSFT_2012-06-21_5",
    "AMZN": "LOBSTER_SampleFile_AMZN_2012-06-21_5",
}
DATE = "2012-06-21"
LEVELS = 5
K = 5
MARKET_OPEN = 34200    # 9:30 AM EST in seconds after midnight
MARKET_CLOSE = 57600   # 4:00 PM EST

MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]
EVENT_MAP = {1: 'L', 2: 'C', 3: 'C', 4: 'M', 5: 'M'}

def make_ob_cols(levels):
    cols = []
    for i in range(1, levels + 1):
        cols += [f"ask_px_{i}", f"ask_sz_{i}", f"bid_px_{i}", f"bid_sz_{i}"]
    return cols

OB_COLS = make_ob_cols(LEVELS)"""))

cells.append(code("""# --- Load all stocks ---
def load_stock(stock, folder_name):
    folder = DATA_DIR / folder_name
    msg_f = folder / f"{stock}_{DATE}_34200000_57600000_message_{LEVELS}.csv"
    ob_f  = folder / f"{stock}_{DATE}_34200000_57600000_orderbook_{LEVELS}.csv"
    msg = pd.read_csv(msg_f, header=None, names=MSG_COLS)
    ob  = pd.read_csv(ob_f, header=None, names=OB_COLS)
    assert len(msg) == len(ob), f"Row mismatch for {stock}"
    msg['price_dollars'] = msg['price'] / 10000.0
    return msg, ob

all_data = {}
for stock, folder in STOCKS.items():
    msg, ob = load_stock(stock, folder)
    all_data[stock] = (msg, ob)
    print(f"  {stock}: {len(msg):>10,} events | "
          f"Price range: ${msg['price_dollars'].min():.2f} - ${msg['price_dollars'].max():.2f}")
print(f"\\nTotal events across all stocks: {sum(len(v[0]) for v in all_data.values()):,}")"""))

# ==============================================================================
# SECTION 2: Data Exploration
# ==============================================================================
cells.append(md(r"""---
## 2. Data Exploration

We now examine the data to understand its structure and compute descriptive statistics analogous to Table 1 of the paper. The paper reports per-level statistics for the Bund futures market: event counts by type, Average Event Size (AES), and Average Inter-arrival Time (AIT)."""))

cells.append(code("""def preprocess_stock(msg, ob, K=5):
    \"\"\"
    Full preprocessing pipeline for one stock.

    Steps:
    1. Classify events (L, C, M), filter halts
    2. Determine affected side (bid/ask)
    3. Compute pre-event queue sizes (shift orderbook by 1)
    4. Compute signed price levels (-K..-1 for bid, +1..+K for ask)
    5. Compute inter-event times and mid-price
    \"\"\"
    df = pd.concat([msg.copy(), ob.copy()], axis=1)

    # Step 1: Classify events
    df['event_type'] = df['type'].map(EVENT_MAP)
    df = df[df['event_type'].notna()].copy()

    # Step 2: Determine affected side
    # In LOBSTER, direction == +1 ALWAYS means the Bid book is affected,
    # for ALL event types (limit, cancel, and execution against the bid side).
    # direction == -1 means the Ask book is affected.
    df['side'] = np.where(df['direction'] == 1, 'bid', 'ask')

    # Step 3: Compute pre-event orderbook (shift by 1)
    price_cols = [f'{s}_px_{i}' for s in ['ask', 'bid'] for i in range(1, K+1)]
    size_cols  = [f'{s}_sz_{i}' for s in ['ask', 'bid'] for i in range(1, K+1)]
    all_ob_cols = price_cols + size_cols
    for col in all_ob_cols:
        df[f'pre_{col}'] = df[col].shift(1)
    df = df.iloc[1:].copy()  # drop first row (no pre-event book)

    # Step 4: Find price level (1..K) using PRE-event orderbook (vectorized)
    df['level'] = 0
    for i in range(1, K+1):
        mask_ask = (df['side'] == 'ask') & (df['price'].astype(np.int64) == df[f'pre_ask_px_{i}'].astype(np.int64))
        mask_bid = (df['side'] == 'bid') & (df['price'].astype(np.int64) == df[f'pre_bid_px_{i}'].astype(np.int64))
        df.loc[mask_ask, 'level'] = i
        df.loc[mask_bid, 'level'] = i

    # Signed level: positive for ask, negative for bid
    df['signed_level'] = np.where(df['side'] == 'ask', df['level'], -df['level'])

    # Step 5: Pre-event queue size at the affected level
    df['q_before'] = 0
    for i in range(1, K+1):
        mask_ask = (df['side'] == 'ask') & (df['level'] == i)
        mask_bid = (df['side'] == 'bid') & (df['level'] == i)
        df.loc[mask_ask, 'q_before'] = df.loc[mask_ask, f'pre_ask_sz_{i}'].astype(int)
        df.loc[mask_bid, 'q_before'] = df.loc[mask_bid, f'pre_bid_sz_{i}'].astype(int)

    # Step 6: Mid-price (computed before filtering, on all events)
    df['mid_price'] = (df['pre_ask_px_1'].astype(float) + df['pre_bid_px_1'].astype(float)) / 2.0

    # Filter to events within top-K levels
    df_clean = df[df['level'] > 0].copy()

    # Inter-event time computed AFTER filtering: for each event in df_clean,
    # dt = time since previous event in df_clean. This correctly accounts for
    # all time elapsed between consecutive top-K-level events, including
    # intervals occupied by out-of-book events (level=0), ensuring unbiased
    # MLE occupancy times T^occ_i(n). Computing dt BEFORE filtering would
    # underestimate occupancy and inflate intensity estimates.
    df_clean['dt'] = df_clean['time'].diff().fillna(0).clip(lower=0)

    return df, df_clean

# Process the primary stock (AAPL -- large-tick, liquid, similar to Bund)
PRIMARY = "AAPL"
df_all, df_clean = preprocess_stock(*all_data[PRIMARY], K=K)
print(f"Stock: {PRIMARY}")
print(f"  Total events (after filtering halts): {len(df_all):,}")
print(f"  Events within top-{K} levels:         {len(df_clean):,} ({len(df_clean)/len(df_all)*100:.1f}%)")
print(f"  Events outside top-{K} levels:        {len(df_all) - len(df_clean):,}")"""))

cells.append(code("""# --- Descriptive statistics (analogous to Table 1 in the paper) ---
def compute_descriptive_stats(df_clean, K):
    rows = []
    for i in range(1, K+1):
        for side, sign in [('ask', +1), ('bid', -1)]:
            sl = sign * i
            sub = df_clean[df_clean['signed_level'] == sl]
            n_L = (sub['event_type'] == 'L').sum()
            n_C = (sub['event_type'] == 'C').sum()
            n_M = (sub['event_type'] == 'M').sum()
            aes = sub['size'].mean() if len(sub) > 0 else 0
            times = sub['time'].values
            ait = np.mean(np.diff(times)) if len(times) > 1 else np.nan
            rows.append({
                'Level': i, 'Side': side,
                '#L': n_L, '#C': n_C, '#M': n_M,
                'AES': aes, 'AIT_s': ait
            })
    stats_df = pd.DataFrame(rows)

    # Average bid/ask (symmetry assumption as in the paper)
    avg = stats_df.groupby('Level').agg({
        '#L': 'mean', '#C': 'mean', '#M': 'mean',
        'AES': 'mean', 'AIT_s': 'mean'
    }).reset_index()
    avg['AIT_ms'] = avg['AIT_s'] * 1000
    return avg

stats = compute_descriptive_stats(df_clean, K)
print(f"\\nDescriptive Statistics for {PRIMARY} (bid/ask averaged):")
print("=" * 70)
print(stats[['Level', '#L', '#C', '#M', 'AES', 'AIT_ms']].to_string(index=False, float_format='%.1f'))
print("\\n#L, #C, #M = number of limit, cancel, market events")
print("AES = Average Event Size (shares), AIT = Average Inter-event Time (ms)")"""))

cells.append(code("""# --- Visualize the LOB at a mid-day snapshot ---
def plot_lob_snapshot(ob_row, K, title="LOB Snapshot"):
    fig, ax = plt.subplots(figsize=(10, 5))
    levels = list(range(-K, 0)) + list(range(1, K+1))
    sizes = []
    colors = []
    for lvl in levels:
        if lvl < 0:
            sizes.append(ob_row[f'bid_sz_{-lvl}'])
            colors.append('#2196F3')
        else:
            sizes.append(ob_row[f'ask_sz_{lvl}'])
            colors.append('#F44336')
    ax.bar([str(l) for l in levels], sizes, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("Price Level (negative = bid, positive = ask)")
    ax.set_ylabel("Queue Size (shares)")
    ax.set_title(title)
    ax.axvline(x=K-0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-price')
    ax.legend()
    plt.tight_layout()
    plt.show()

msg_primary, ob_primary = all_data[PRIMARY]
snapshot_idx = len(ob_primary) // 2
plot_lob_snapshot(ob_primary.iloc[snapshot_idx], K,
                  title=f"{PRIMARY} LOB Snapshot (event #{snapshot_idx:,})")"""))

cells.append(code("""# --- Event type distribution over time ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, et, color, label in zip(axes, ['L', 'C', 'M'],
                                  ['#4CAF50', '#FF9800', '#F44336'],
                                  ['Limit Orders', 'Cancellations', 'Market Orders']):
    sub = df_clean[df_clean['event_type'] == et]
    bins = np.arange(MARKET_OPEN, MARKET_CLOSE + 300, 300)
    counts_hist, _ = np.histogram(sub['time'].values, bins=bins)
    hours = (bins[:-1] - MARKET_OPEN) / 3600 + 9.5
    ax.bar(hours, counts_hist, width=300/3600*0.9, color=color, alpha=0.7)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Event Count (per 5 min)")
    ax.set_title(label)

plt.suptitle(f"{PRIMARY} -- Intraday Event Frequency", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 3: QR MODEL THEORY
# ==============================================================================
cells.append(md(r"""---
## 3. The Queue-Reactive (QR) Model -- Mathematical Foundation

### 3.1 Model Setting

The Queue-Reactive model, introduced by Huang et al. (2015), views the limit order book as a collection of $2K$ queues centered around a reference price $p_{\text{ref}}$ (typically the mid-price):

$$
\mathbf{Q} = (Q_{-K}, \ldots, Q_{-1}, Q_1, \ldots, Q_K)
$$

where $Q_{-i}$ (resp. $Q_i$) is the queue size at the $i$-th bid (resp. ask) level from the reference price.

Three types of events can occur at each queue:
- **Limit order** ($L$): adds volume $\rightarrow$ $Q_i \to Q_i + \text{Volume}$
- **Cancellation** ($C$): removes volume $\rightarrow$ $Q_i \to Q_i - \text{Volume}$ (only if $Q_i > 0$)
- **Market order** ($M$): removes volume $\rightarrow$ $Q_i \to Q_i - \text{Volume}$ (only if $Q_i > 0$)

### 3.2 Key Assumption: Queue-Reactive Intensities

The central insight of the QR model is that **event arrival rates depend on the current queue size**. For each queue $i$ and event type $\eta \in \{L, C, M\}$, the arrival intensity is:

$$
\lambda^\eta_i(n) = \text{rate of events of type } \eta \text{ at queue } i \text{ when } Q_i = n
$$

The total intensity (rate of any event) at queue $i$ when $Q_i = n$ is:

$$
\Lambda_i(n) = \lambda^L_i(n) + \lambda^C_i(n) + \lambda^M_i(n)
$$

**Critical assumption:** Each queue evolves **independently** of all others. This is a major simplification -- relaxed later in the MDQR model.

### 3.3 Likelihood Function

Consider a sequence of events at queue $i$: $\mathcal{E}_i = \{e_k\}_{k=1}^{N_i}$, where each event $e_k$ is characterized by:
- $\eta_k \in \{L, C, M\}$: event type
- $q_k$: queue size just **before** the event
- $\Delta t_k = t_k - t_{k-1}$: inter-event time at this queue

Since the queue $Q_i$ follows a continuous-time Markov chain with state-dependent intensities, the probability of observing event $e_k$ (of type $\eta_k$) after waiting $\Delta t_k$ in state $q_k$ is:

$$
\mathbb{P}(\text{event of type } \eta_k \text{ at time } t_k \mid Q_i(t_{k-1}^+) = q_k) = \lambda^{\eta_k}_i(q_k) \cdot e^{-\Lambda_i(q_k) \cdot \Delta t_k}
$$

This follows from the **exponential inter-arrival property** of Poisson processes: the time until the next event in state $n$ is $\text{Exp}(\Lambda_i(n))$, and the probability that it is of type $\eta$ is $\frac{\lambda^\eta_i(n)}{\Lambda_i(n)}$.

The log-likelihood for queue $i$ is therefore:

$$
\ell(\{\lambda^\eta_i\} \mid \mathcal{E}_i) = \sum_{k=1}^{N_i} \left[ \log \lambda^{\eta_k}_i(q_k) - \Lambda_i(q_k) \cdot \Delta t_k \right]
$$"""))

cells.append(md(r"""### 3.4 Maximum Likelihood Estimator (Derivation)

We seek to maximize the log-likelihood with respect to $\lambda^\eta_i(n)$ for each state $n$. Define:

- $\mathcal{N}^\eta_i(n) = \#\{k : \eta_k = \eta, \, q_k = n\}$ -- count of events of type $\eta$ when queue size is $n$
- $T^{\text{occ}}_i(n) = \sum_{k: q_k = n} \Delta t_k$ -- total occupancy time in state $n$

The log-likelihood decomposes across states:

$$
\ell = \sum_{n=0}^{n_{\max}} \sum_{\eta \in \{L,C,M\}} \left[ \mathcal{N}^\eta_i(n) \cdot \log \lambda^\eta_i(n) - \lambda^\eta_i(n) \cdot T^{\text{occ}}_i(n) \right]
$$

Taking the derivative with respect to $\lambda^\eta_i(n)$ and setting it to zero:

$$
\frac{\partial \ell}{\partial \lambda^\eta_i(n)} = \frac{\mathcal{N}^\eta_i(n)}{\lambda^\eta_i(n)} - T^{\text{occ}}_i(n) = 0
$$

Solving:

$$
\boxed{\hat{\lambda}^\eta_i(n) = \frac{\mathcal{N}^\eta_i(n)}{T^{\text{occ}}_i(n)}}
$$

This is the **empirical rate estimator**: the number of events of type $\eta$ at state $n$ divided by the total time spent in state $n$. This is equivalent to the formula in the paper (Eq. 1):

$$
\hat{\lambda}^\eta(n) = \underbrace{\frac{\#\{e_k : \eta_k = \eta, q_k = n\}}{\#\{e_k : q_k = n\}}}_{\text{event-type probability}} \times \underbrace{\left[\frac{1}{\#\{k : q_k = n\}} \sum_{k: q_k = n} \Delta t_k\right]^{-1}}_{\text{inverse mean inter-event time}}
$$

**Note:** In practice, we clip queue sizes to a maximum value $n_{\max}$ to ensure sufficient statistical support for each state. Following the paper, we also normalize queue sizes by the Average Event Size (AES) at each level."""))

# ==============================================================================
# SECTION 4: QR MODEL CALIBRATION
# ==============================================================================
cells.append(md(r"""---
## 4. QR Model Calibration

### 4.1 Preprocessing: Queue Size Normalization

Following Huang et al. (2015), we normalize queue sizes by their respective Average Event Sizes:

$$
q_i^{\text{norm}} = \left\lceil \frac{q_i^{\text{raw}}}{\text{AES}_i} \right\rceil
$$

This ensures that each event represents approximately one "unit" change in the normalized queue, making the model consistent across levels with different typical order sizes."""))

cells.append(code("""def compute_aes(df_clean, K):
    \"\"\"Compute Average Event Size per signed level.\"\"\"
    aes = {}
    for sl in range(-K, K+1):
        if sl == 0:
            continue
        sub = df_clean[df_clean['signed_level'] == sl]
        aes[sl] = sub['size'].mean() if len(sub) > 0 else 1.0
    return aes

# Compute AES and use bid/ask averaged values
aes_dict = compute_aes(df_clean, K)
print("Average Event Size (AES) per level:")
print("-" * 40)
for i in range(1, K+1):
    aes_ask = aes_dict.get(i, 1)
    aes_bid = aes_dict.get(-i, 1)
    aes_avg = (aes_ask + aes_bid) / 2
    aes_dict[i] = aes_avg
    aes_dict[-i] = aes_avg
    print(f"  Level {i}: AES_ask = {aes_ask:.1f}, AES_bid = {aes_bid:.1f}, AES_avg = {aes_avg:.1f}")"""))

cells.append(code("""def fit_qr_mle(df_clean, K, n_max=100, aes_dict=None):
    \"\"\"
    Fit QR model intensities using the analytical MLE (Eq. 1 of the paper).

    For each signed level and queue state n:
        lambda_hat[eta][level][n] = N_eta(level, n) / T_occ(level, n)

    Parameters
    ----------
    df_clean : DataFrame with columns [time, event_type, signed_level, q_before, dt]
    K : int, number of levels per side
    n_max : int, maximum normalized queue size
    aes_dict : dict, Average Event Size per level (for normalization)

    Returns
    -------
    intensities : dict of {event_type: {signed_level: array of shape (n_max+1,)}}
    occupancy : dict of {signed_level: array of shape (n_max+1,)}
    counts : dict
    \"\"\"
    signed_levels = list(range(-K, 0)) + list(range(1, K+1))

    # Initialize storage
    occupancy = {sl: np.zeros(n_max + 1, dtype=float) for sl in signed_levels}
    counts = {
        et: {sl: np.zeros(n_max + 1, dtype=np.int64) for sl in signed_levels}
        for et in ['L', 'C', 'M']
    }

    dt_arr = df_clean['dt'].values.astype(float)
    et_arr = df_clean['event_type'].values
    sl_arr = df_clean['signed_level'].values.astype(int)

    # Build normalized queue size arrays per level
    q_arrays = {}
    for sl in signed_levels:
        side = 'ask' if sl > 0 else 'bid'
        i = abs(sl)
        col = f'pre_{side}_sz_{i}'
        if col in df_clean.columns:
            raw_q = df_clean[col].values.astype(float)
        else:
            raw_q = np.zeros(len(df_clean))

        if aes_dict and sl in aes_dict and aes_dict[sl] > 0:
            q_norm = np.ceil(raw_q / aes_dict[sl]).astype(int)
        else:
            q_norm = raw_q.astype(int)
        q_norm = np.clip(q_norm, 0, n_max)
        q_arrays[sl] = q_norm

    # Accumulate occupancy: for each event, ALL queues accumulate dt
    for sl in signed_levels:
        q = q_arrays[sl]
        np.add.at(occupancy[sl], q, dt_arr)

    # Accumulate event counts: only for the affected level
    for et in ['L', 'C', 'M']:
        et_mask = et_arr == et
        for sl in signed_levels:
            mask = et_mask & (sl_arr == sl)
            if mask.any():
                np.add.at(counts[et][sl], q_arrays[sl][mask], 1)

    # Compute intensities: lambda = count / occupancy
    intensities = {et: {} for et in ['L', 'C', 'M']}
    for et in ['L', 'C', 'M']:
        for sl in signed_levels:
            lam = np.zeros(n_max + 1, dtype=float)
            mask = occupancy[sl] > 0
            lam[mask] = counts[et][sl][mask] / occupancy[sl][mask]
            intensities[et][sl] = lam

    # Average bid/ask for symmetry (as in the paper)
    intensities_sym = {et: {} for et in ['L', 'C', 'M']}
    for et in ['L', 'C', 'M']:
        for i in range(1, K+1):
            avg = (intensities[et][i] + intensities[et][-i]) / 2.0
            intensities_sym[et][i] = avg
            intensities_sym[et][-i] = avg

    return intensities_sym, occupancy, counts

# Fit the QR model
N_MAX = 100
intensities_qr, occupancy_qr, counts_qr = fit_qr_mle(df_clean, K, n_max=N_MAX, aes_dict=aes_dict)
print("QR Model fitted successfully!")
print(f"  n_max = {N_MAX}")
print(f"  Levels: {K} per side (10 total)")"""))

cells.append(code("""# --- Diagnostic: normalized queue size distribution ---
# This reveals whether the queue concentrates in a narrow range of normalized states,
# which would make the raw MLE noisy (few observations per integer state).
print("Normalized queue size statistics per level (ask side, pre-event state):")
print(f"{'Level':>7}  {'min':>5}  {'p10':>5}  {'median':>7}  {'p90':>5}  {'max':>5}  {'#states':>8}")
print("-" * 56)
for i in range(1, K+1):
    col = f'pre_ask_sz_{i}'
    raw_q = df_clean[col].values.astype(float)
    aes = aes_dict.get(i, 1.0)
    q_n = np.clip(np.ceil(raw_q / max(aes, 1)).astype(int), 0, N_MAX)
    n_states = len(np.unique(q_n))
    print(f"  Lvl {i}:  {q_n.min():>4}   {np.percentile(q_n,10):>4.0f}   "
          f"{np.median(q_n):>6.0f}   {np.percentile(q_n,90):>4.0f}   "
          f"{q_n.max():>4}   {n_states:>7}")
print("\\nIf the queue only visits a few integer states (narrow range),")
print("the raw MLE lambda(n)=N(n)/T_occ(n) will be very noisy.")
print("Solution: Nadaraya-Watson kernel smoothing (see below).")"""))

cells.append(code("""def smooth_intensities_nw(counts, occupancy, K, n_max, bandwidth=2):
    \"\"\"
    Nadaraya-Watson Gaussian kernel smoothing of raw MLE intensity estimates.

    When the queue concentrates in a narrow range of normalized states (common
    for large-tick equities like AAPL), the raw estimator N(n)/T_occ(n) is
    very noisy: a single event during a brief visit to state n gives a huge
    apparent intensity.  The standard fix is to smooth jointly the numerator
    and denominator with a Gaussian kernel K_h:

        lambda_smooth(n) = [K_h * N](n) / [K_h * T_occ](n)

    This is the Nadaraya-Watson non-parametric regression estimator. It is
    well-justified: if the true intensity is a smooth function of n (as assumed
    by the QR model), this estimator converges to it as the sample grows.

    Parameters
    ----------
    bandwidth : float
        Standard deviation of the Gaussian kernel in normalized queue units.
        Larger bandwidth = more smoothing. Typical choice: 2-5 normalized units.
    \"\"\"
    from scipy.ndimage import gaussian_filter1d

    signed_levels = list(range(-K, 0)) + list(range(1, K+1))
    intensities_sm = {et: {} for et in ['L', 'C', 'M']}

    for et in ['L', 'C', 'M']:
        for sl in signed_levels:
            cnt = counts[et][sl].astype(float)
            occ = occupancy[sl].astype(float)
            # Smooth numerator and denominator separately
            cnt_sm = gaussian_filter1d(cnt, sigma=bandwidth)
            occ_sm = gaussian_filter1d(occ, sigma=bandwidth)
            lam = np.zeros(n_max + 1, dtype=float)
            valid = occ_sm > 1e-12
            lam[valid] = cnt_sm[valid] / occ_sm[valid]
            intensities_sm[et][sl] = lam

    # Bid/ask symmetry averaging (same as in fit_qr_mle)
    intensities_sym = {et: {} for et in ['L', 'C', 'M']}
    for et in ['L', 'C', 'M']:
        for i in range(1, K+1):
            avg = (intensities_sm[et][i] + intensities_sm[et][-i]) / 2.0
            intensities_sym[et][i] = avg
            intensities_sym[et][-i] = avg

    return intensities_sym

BW = 2  # bandwidth in normalized queue units
intensities_qr_smooth = smooth_intensities_nw(counts_qr, occupancy_qr, K, N_MAX, bandwidth=BW)
print(f"Kernel-smoothed intensities computed (Nadaraya-Watson, bandwidth sigma={BW}).")
print("These will be used for both visualization and simulation.")"""))

cells.append(code("""# --- Visualize fitted QR intensities (raw MLE vs kernel-smoothed) ---
# Determine the actual support range across all levels so the x-axis
# automatically zooms to where data exists.
all_occ = sum(occupancy_qr[i] + occupancy_qr[-i] for i in range(1, K+1))
support_mask = all_occ > 0
n_lo = max(0, int(np.where(support_mask)[0].min()) - 2) if support_mask.any() else 0
n_hi = min(N_MAX, int(np.where(support_mask)[0].max()) + 2) if support_mask.any() else 50
print(f"Queue state support range: n = [{n_lo}, {n_hi}]")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
n = np.arange(N_MAX + 1)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, K))
titles = ['Limit Order ($L$)', 'Cancellation ($C$)', 'Market Order ($M$)']

for col_idx, (et, title) in enumerate(zip(['L', 'C', 'M'], titles)):
    sup = (occupancy_qr[1] + occupancy_qr[-1]) > 0  # use level-1 for support mask

    # Top row: raw MLE
    ax_raw = axes[0, col_idx]
    for i in range(1, K+1):
        lam = intensities_qr[et][i]
        s = (occupancy_qr[i] + occupancy_qr[-i]) > 0
        if s.any():
            ax_raw.plot(n[s], lam[s], 'o', markersize=3, linewidth=1,
                        color=colors[i-1], label=f'L{i}', alpha=0.7)
    ax_raw.set_title(f'Raw MLE -- {title}')
    ax_raw.set_xlim(n_lo, n_hi)
    ax_raw.set_xlabel("Normalized queue $n$")
    ax_raw.set_ylabel("Intensity (events/s)")
    ax_raw.legend(fontsize=8)

    # Bottom row: N-W smoothed
    ax_sm = axes[1, col_idx]
    for i in range(1, K+1):
        lam_sm = intensities_qr_smooth[et][i]
        s = (occupancy_qr[i] + occupancy_qr[-i]) > 0
        if s.any():
            ax_sm.plot(n[s], lam_sm[s], '-', linewidth=2,
                       color=colors[i-1], label=f'L{i}', alpha=0.9)
    ax_sm.set_title(f'N-W Smoothed ($\\\\sigma$={BW}) -- {title}')
    ax_sm.set_xlim(n_lo, n_hi)
    ax_sm.set_xlabel("Normalized queue $n$")
    ax_sm.set_ylabel("Intensity (events/s)")
    ax_sm.legend(fontsize=8)

plt.suptitle(
    f"QR Model -- Fitted Intensity Functions ({PRIMARY})\\n"
    f"Row 1: raw MLE (noisy due to sparse integer-state coverage)  |  "
    f"Row 2: Nadaraya-Watson kernel smoothed ($\\\\sigma$={BW})",
    fontsize=11)
plt.tight_layout()
plt.show()"""))

cells.append(md(r"""### 4.2 Observations on the Fitted Intensities

**Why smoothing is necessary.** For NASDAQ equities (and AAPL in particular), queue sizes in normalized units $n = \lceil Q^{\text{raw}} / \text{AES} \rceil$ tend to concentrate in a narrow range of integer values (e.g. $n \in [35, 55]$). The raw MLE $\hat{\lambda}^\eta_i(n) = N^\eta_i(n) / T^{\text{occ}}_i(n)$ is then very noisy: if the queue visits state $n = 42$ for a total of $0.002$ seconds and one event occurs, the raw estimate is $500\,\text{s}^{-1}$. This is a statistical artifact with no physical meaning. The Nadaraya–Watson kernel estimator avoids this by pooling nearby states.

**Patterns visible in the smoothed intensity functions:**

1. **Limit order intensity $\lambda^L_i(n)$:** Relatively flat or slightly decreasing in $n$. This reflects that limit orders arrive at a roughly constant rate regardless of queue depth. Deeper levels ($i > 1$) have lower intensities.

2. **Cancellation intensity $\lambda^C_i(n)$:** Increasing in $n$ — the more orders in the queue, the more candidates for cancellation. A linear relationship $\lambda^C_i(n) \approx b_i \cdot n$ is a well-documented stylized fact (Cont et al., 2010).

3. **Market order intensity $\lambda^M_i(n)$:** Concentrated at the best level ($i = 1$), near-zero at all deeper levels. Market orders execute at the best available price and so only affect level 1.

These qualitative patterns are consistent with Huang et al. (2015) and Bodor & Carlier (2025), confirming the model is correctly calibrated despite the data being equities rather than Bund futures."""))

# ==============================================================================
# SECTION 5: QR SIMULATION
# ==============================================================================
cells.append(md(r"""---
## 5. QR Model Simulation via the Gillespie Algorithm

### 5.1 The Gillespie Algorithm

Given the fitted intensities $\{\hat{\lambda}^\eta_i(n)\}$, we simulate LOB dynamics using the **Gillespie algorithm** (also known as the Stochastic Simulation Algorithm), which is the exact simulation method for continuous-time Markov chains.

**Algorithm:** At each step, given the current state $\mathbf{Q} = (Q_{-K}, \ldots, Q_K)$:

1. **Compute total rate:**
$$R = \sum_{i} \sum_{\eta \in \{L,C,M\}} \lambda^\eta_i(Q_i)$$
where cancellations and market orders are only possible when $Q_i > 0$.

2. **Sample waiting time:** $\Delta t \sim \text{Exp}(R)$, i.e., $\Delta t = -\frac{1}{R}\log(U_1)$ with $U_1 \sim \text{Uniform}(0,1)$.

3. **Select event:** Choose event $(\eta, i)$ with probability $\frac{\lambda^\eta_i(Q_i)}{R}$.

4. **Update state** using normalized volume $v = \lceil \text{Volume} / \text{AES}_i \rceil$ sampled from the empirical size distribution at level $i$:
   - If $\eta = L$: $Q_i \leftarrow Q_i + v$
   - If $\eta \in \{C, M\}$: $Q_i \leftarrow \max(Q_i - v, \, 0)$

5. **Repeat** until the time horizon $T$ is reached.

### 5.2 Reference Price Changes

When the best bid or ask queue is completely depleted ($Q_{\pm 1} = 0$), the reference price shifts. We handle this by shifting all queues by one position and initializing the new outermost queue from its empirical distribution."""))

cells.append(code("""class QRSimulator:
    \"\"\"
    Gillespie-based LOB simulator using QR model intensities.
    State: {signed_level: queue_size} for levels -K..-1, 1..K.
    \"\"\"

    def __init__(self, intensities, K, n_max=100, seed=42, emp_sizes=None):
        self.intensities = intensities
        self.K = K
        self.n_max = n_max
        self.rng = np.random.default_rng(seed)
        self.signed_levels = list(range(-K, 0)) + list(range(1, K+1))
        # emp_sizes: {signed_level: array of normalized volumes} for realistic updates
        self.emp_sizes = emp_sizes if emp_sizes is not None else {}

    def _sample_volume(self, sl):
        \"\"\"Sample a normalized volume at level sl from empirical distribution.\"\"\"
        arr = self.emp_sizes.get(sl)
        if arr is not None and len(arr) > 0:
            return max(1, int(self.rng.choice(arr)))
        return 1

    def _get_rate(self, eta, level, q):
        n = min(max(int(q), 0), self.n_max)
        return float(self.intensities[eta][level][n])

    def _compute_rates(self, state):
        events = []
        rates = []
        for sl in self.signed_levels:
            q = state[sl]
            r_L = self._get_rate('L', sl, q)
            if r_L > 0:
                events.append(('L', sl))
                rates.append(r_L)
            if q > 0:
                r_C = self._get_rate('C', sl, q)
                r_M = self._get_rate('M', sl, q)
                if r_C > 0:
                    events.append(('C', sl))
                    rates.append(r_C)
                if r_M > 0:
                    events.append(('M', sl))
                    rates.append(r_M)
        return events, np.array(rates, dtype=float)

    def _handle_price_change(self, state, side, empirical_init):
        K = self.K
        if side == 'ask':
            for i in range(1, K):
                state[i] = state[i+1]
            state[K] = int(self.rng.choice(empirical_init.get(K, [1])))
        elif side == 'bid':
            for i in range(1, K):
                state[-i] = state[-(i+1)]
            state[-K] = int(self.rng.choice(empirical_init.get(K, [1])))
        return state

    def simulate(self, state0, T, empirical_init=None, max_events=5_000_000):
        state = dict(state0)
        t = 0.0
        log = []

        for step in range(max_events):
            events, rates = self._compute_rates(state)
            R = rates.sum()
            if R <= 0:
                break

            dt = self.rng.exponential(1.0 / R)
            t_next = t + dt
            if t_next > T:
                break

            probs = rates / R
            idx = self.rng.choice(len(events), p=probs)
            eta, sl = events[idx]
            q_before = state[sl]

            v = self._sample_volume(sl)
            if eta == 'L':
                state[sl] = min(q_before + v, self.n_max)
            else:
                state[sl] = max(q_before - v, 0)

            q_after = state[sl]
            log.append((t_next, eta, sl, q_before, q_after))

            if empirical_init is not None:
                if state.get(1, 0) == 0:
                    self._handle_price_change(state, 'ask', empirical_init)
                if state.get(-1, 0) == 0:
                    self._handle_price_change(state, 'bid', empirical_init)

            t = t_next

        return log

print("QRSimulator defined.")"""))

cells.append(code("""# --- Compute initial state and empirical init distribution ---
def compute_initial_state(df_clean, K, aes_dict):
    state0 = {}
    for sl in list(range(-K, 0)) + list(range(1, K+1)):
        side = 'ask' if sl > 0 else 'bid'
        i = abs(sl)
        col = f'pre_{side}_sz_{i}'
        raw_q = df_clean[col].median() if col in df_clean.columns else 100
        if aes_dict and sl in aes_dict and aes_dict[sl] > 0:
            state0[sl] = max(1, int(np.ceil(raw_q / aes_dict[sl])))
        else:
            state0[sl] = max(1, int(raw_q))
    return state0

def compute_empirical_init(df_clean, K, aes_dict, n_samples=1000):
    emp_init = {}
    for i in range(1, K+1):
        samples = []
        for side, sign in [('ask', i), ('bid', -i)]:
            col = f'pre_{side}_sz_{abs(sign)}'
            if col in df_clean.columns:
                vals = df_clean[col].dropna().values.astype(float)
                if aes_dict and sign in aes_dict and aes_dict[sign] > 0:
                    vals = np.ceil(vals / aes_dict[sign]).astype(int)
                samples.extend(vals[:n_samples].tolist())
        emp_init[i] = np.array(samples) if samples else np.array([1])
    return emp_init

def compute_empirical_sizes(df_clean, K, aes_dict):
    \"\"\"
    Compute empirical normalized order size distributions per level.
    Used by the simulator to sample realistic volume changes per event.
    Normalized size: v = ceil(raw_size / AES_i), so each event changes
    the normalized queue by approximately 1 unit on average.
    \"\"\"
    emp_sizes = {}
    for sl in list(range(-K, 0)) + list(range(1, K+1)):
        sub = df_clean[df_clean['signed_level'] == sl]
        if len(sub) > 0:
            raw_sz = sub['size'].values.astype(float)
            aes = aes_dict.get(sl, 1.0)
            norm_sz = np.maximum(1, np.ceil(raw_sz / max(aes, 1.0)).astype(int))
            emp_sizes[sl] = norm_sz
        else:
            emp_sizes[sl] = np.array([1])
    return emp_sizes

state0 = compute_initial_state(df_clean, K, aes_dict)
emp_init = compute_empirical_init(df_clean, K, aes_dict)
emp_sizes = compute_empirical_sizes(df_clean, K, aes_dict)

print("Initial state (normalized queue sizes):")
for sl in sorted(state0.keys()):
    side = 'ask' if sl > 0 else 'bid'
    i = abs(sl)
    col = f'pre_{side}_sz_{i}'
    raw_med = df_clean[col].median() if col in df_clean.columns else float('nan')
    aes = aes_dict.get(sl, 1.0)
    print(f"  Level {sl:+d} ({side}): raw median={raw_med:.0f}, AES={aes:.1f}, norm={state0[sl]}")
print(f"\\nMean empirical sizes (normalized, per level):")
for sl in sorted(emp_sizes.keys()):
    print(f"  Level {sl:+d}: mean={emp_sizes[sl].mean():.2f}, median={np.median(emp_sizes[sl]):.2f}")"""))

cells.append(code("""# --- Run simulation ---
T_SIM = 3600  # 1 hour

# Use kernel-smoothed intensities: avoids propagating statistical noise
# from sparse raw MLE estimates into the simulation
sim = QRSimulator(intensities_qr_smooth, K=K, n_max=N_MAX, seed=42, emp_sizes=emp_sizes)
sim_log = sim.simulate(state0, T=T_SIM, empirical_init=emp_init)

print(f"Simulation completed: {len(sim_log):,} events in {T_SIM}s")
print(f"Average inter-event time: {T_SIM / max(len(sim_log),1) * 1000:.2f} ms")

# Convert to DataFrame
sim_df = pd.DataFrame(sim_log, columns=['time', 'event_type', 'signed_level', 'q_before', 'q_after'])
sim_df['dt'] = sim_df['time'].diff().fillna(0)
sim_df['side'] = np.where(sim_df['signed_level'] > 0, 'ask', 'bid')
sim_df['level'] = sim_df['signed_level'].abs()

print(f"\\nSimulated event counts:")
for et in ['L', 'C', 'M']:
    n = (sim_df['event_type'] == et).sum()
    print(f"  {et}: {n:,} ({n/len(sim_df)*100:.1f}%)")"""))

cells.append(code("""# --- Visualize simulated LOB evolution ---
def reconstruct_lob_timeseries(sim_log, state0, K, dt_grid=1.0):
    signed_levels = list(range(-K, 0)) + list(range(1, K+1))
    state = dict(state0)

    if not sim_log:
        return np.array([0.0]), {sl: np.array([state[sl]]) for sl in signed_levels}

    T_max = sim_log[-1][0]
    t_grid = np.arange(0, T_max + dt_grid, dt_grid)
    q_series = {sl: np.zeros(len(t_grid), dtype=int) for sl in signed_levels}

    event_idx = 0
    for i, t in enumerate(t_grid):
        while event_idx < len(sim_log) and sim_log[event_idx][0] <= t:
            _, eta, sl, _, q_after = sim_log[event_idx]
            state[sl] = q_after
            event_idx += 1
        for sl in signed_levels:
            q_series[sl][i] = state[sl]

    return t_grid, q_series

t_grid, q_series = reconstruct_lob_timeseries(sim_log, state0, K, dt_grid=1.0)

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# Best bid and ask queues
axes[0].plot(t_grid, q_series[1], label='Best Ask (level +1)', color='#F44336', alpha=0.8)
axes[0].plot(t_grid, q_series[-1], label='Best Bid (level -1)', color='#2196F3', alpha=0.8)
axes[0].set_ylabel("Normalized Queue Size")
axes[0].set_title("QR Simulation -- Best Bid/Ask Queue Evolution")
axes[0].legend()

# Heatmap of all levels
all_levels = list(range(-K, 0)) + list(range(1, K+1))
q_matrix = np.array([q_series[sl] for sl in all_levels])
axes[1].imshow(q_matrix, aspect='auto', origin='lower',
               extent=[t_grid[0], t_grid[-1], 0, 2*K],
               cmap='YlOrRd', interpolation='nearest')
axes[1].set_yticks(np.arange(2*K) + 0.5)
axes[1].set_yticklabels([str(sl) for sl in all_levels])
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Level")
axes[1].set_title("QR Simulation -- LOB Heatmap")

plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 6: VALIDATION
# ==============================================================================
cells.append(md(r"""---
## 6. Validation Against Stylized Facts

We now compare the QR simulation against historical data on several key metrics. This analysis reveals both the strengths and limitations of the QR model, motivating the extensions in subsequent notebooks.

### 6.1 Transition Matrix of Events

The transition matrix $P(\eta_{k+1} \mid \eta_k)$ captures temporal dependencies between consecutive event types. A key limitation of the QR model is that it assumes **no excitation** between events: the probability of the next event type depends only on the current queue size, not on the previous event. This leads to approximately uniform rows in the transition matrix."""))

cells.append(code("""def compute_transition_matrix(event_types, labels=['C', 'L', 'M']):
    n = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    matrix = np.zeros((n, n), dtype=float)
    types = np.array(event_types)
    for k in range(len(types) - 1):
        i = label_to_idx.get(types[k])
        j = label_to_idx.get(types[k+1])
        if i is not None and j is not None:
            matrix[i, j] += 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix /= row_sums
    return matrix

def plot_transition_matrices(matrices, titles, labels=['cancel', 'limit', 'trade']):
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    for ax, mat, title in zip(axes, matrices, titles):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=0.8, aspect='equal')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("New event")
        ax.set_ylabel("Old event")
        ax.set_title(title, fontsize=12)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                       fontsize=11, color='white' if mat[i,j] > 0.4 else 'black')
    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    plt.show()

# Historical transition matrix (best bid/ask level)
hist_events_best = df_clean[df_clean['level'] == 1]['event_type'].values
P_hist = compute_transition_matrix(hist_events_best, labels=['C', 'L', 'M'])

# QR simulation
sim_events = sim_df['event_type'].values
P_qr = compute_transition_matrix(sim_events, labels=['C', 'L', 'M'])

plot_transition_matrices(
    [P_hist, P_qr],
    ['Real (Historical)', 'QR Model'],
    labels=['cancel', 'limit', 'trade']
)"""))

cells.append(md(r"""**Observation (cf. Figure 1 of the paper):** The historical transition matrix shows clear **excitation patterns** -- for example, a cancel is more likely followed by another cancel, and a trade is more likely followed by another trade. The QR model produces nearly **uniform rows**, indicating that it fails to capture event-type dependencies. This is a direct consequence of the Markov property on queue sizes alone: the intensity depends only on $q_k$, not on $\eta_{k-1}$.

This limitation motivates the DQR model (Notebook 2), where we include $\eta_{k-1}$ in the state vector $\mathbf{x}_k$."""))

cells.append(code("""# --- 6.2 Queue size distributions ---
fig, axes = plt.subplots(1, min(K, 3), figsize=(6*min(K,3), 5))
if min(K, 3) == 1:
    axes = [axes]

for idx, i in enumerate(range(1, min(K, 3)+1)):
    ax = axes[idx]

    # Historical distribution from pre-event queue sizes (normalized)
    side = 'ask'
    col = f'pre_{side}_sz_{i}'
    raw_q = df_clean[col].values.astype(float)
    if aes_dict and i in aes_dict and aes_dict[i] > 0:
        q_norm_hist = np.ceil(raw_q / aes_dict[i]).astype(int)
    else:
        q_norm_hist = raw_q.astype(int)
    q_norm_hist = np.clip(q_norm_hist, 0, N_MAX)

    hist_dist = np.zeros(N_MAX + 1)
    np.add.at(hist_dist, q_norm_hist, 1)
    hist_dist /= hist_dist.sum()

    # Simulated distribution
    sim_q = sim_df[sim_df['signed_level'] == i]['q_before'].values
    sim_q = np.clip(sim_q, 0, N_MAX)
    sim_dist = np.zeros(N_MAX + 1)
    if len(sim_q) > 0:
        np.add.at(sim_dist, sim_q, 1)
        sim_dist /= sim_dist.sum()

    n_arr = np.arange(N_MAX + 1)
    ax.plot(n_arr, hist_dist, 'b-', linewidth=2, label='Real', alpha=0.8)
    ax.plot(n_arr, sim_dist, 'r--', linewidth=2, label='QR', alpha=0.8)
    ax.set_xlabel("Normalized Queue Size $n$")
    ax.set_ylabel("Density")
    ax.set_title(f"Level {i}")
    ax.set_xlim(0, 50)
    ax.legend()

plt.suptitle("Queue Size Distributions: Real vs QR Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()"""))

cells.append(code("""# --- 6.3 Distribution of returns ---
def compute_returns(df, interval_seconds=60):
    times = df['time'].values
    mid = df['mid_price'].values
    t_grid = np.arange(times[0], times[-1], interval_seconds)
    mid_grid = np.interp(t_grid, times, mid)
    returns = np.diff(mid_grid)
    return returns

# Historical returns
returns_hist = compute_returns(df_all, interval_seconds=60)

# Simulated mid-price proxy from queue depletion
def compute_sim_mid_price(sim_log, state0, K):
    state = dict(state0)
    mid_prices = []
    times = []
    mid = 0.0

    for t, eta, sl, q_before, q_after in sim_log:
        state[sl] = q_after
        if sl == 1 and q_after == 0:
            mid += 0.5
        elif sl == -1 and q_after == 0:
            mid -= 0.5
        mid_prices.append(mid)
        times.append(t)

    return np.array(times), np.array(mid_prices)

sim_times, sim_mid = compute_sim_mid_price(sim_log, state0, K)

if len(sim_times) > 2:
    t_grid_sim = np.arange(sim_times[0], sim_times[-1], 60)
    mid_grid_sim = np.interp(t_grid_sim, sim_times, sim_mid)
    returns_sim = np.diff(mid_grid_sim)
else:
    returns_sim = np.array([0.0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF comparison
for returns, label, color in [(returns_hist, 'Real', 'blue'),
                                (returns_sim, 'QR', 'red')]:
    if len(returns) > 10:
        axes[0].hist(returns, bins=80, density=True, alpha=0.5, color=color, label=label)
axes[0].set_xlabel("1-min Returns")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution of 1-minute Returns")
axes[0].legend()

# Q-Q plot
if len(returns_sim) > 10 and len(returns_hist) > 10:
    q_real = np.quantile(returns_hist, np.linspace(0.01, 0.99, 100))
    q_sim = np.quantile(returns_sim, np.linspace(0.01, 0.99, 100))
    axes[1].plot(q_real, q_sim, 'ro', markersize=4, alpha=0.6, label='QR')
    lim = max(abs(q_real).max(), abs(q_sim).max()) * 1.1
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='$y = x$')
    axes[1].set_xlabel("Real Returns Quantiles")
    axes[1].set_ylabel("Model Returns Quantiles")
    axes[1].set_title("Q-Q Plot: QR vs Real")
    axes[1].legend()
    axes[1].set_aspect('equal')
else:
    axes[1].text(0.5, 0.5, "Insufficient simulated data for Q-Q plot",
                transform=axes[1].transAxes, ha='center', va='center')

plt.tight_layout()
plt.show()"""))

cells.append(code("""# --- 6.4 Event frequency per 5-min window ---
def compute_event_counts_windowed(df, window_seconds=300, time_col='time', type_col='event_type'):
    times = df[time_col].values
    types = df[type_col].values
    bins = np.arange(times[0], times[-1] + window_seconds, window_seconds)
    result = {}
    for et in ['L', 'C', 'M']:
        mask = types == et
        counts_w, _ = np.histogram(times[mask], bins=bins)
        result[et] = counts_w
    return result

hist_counts = compute_event_counts_windowed(df_clean)
sim_counts = compute_event_counts_windowed(sim_df)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
labels_map = {'L': 'Limit Events', 'C': 'Cancel Events', 'M': 'Trade Events'}

for ax, et in zip(axes, ['L', 'C', 'M']):
    data = [hist_counts[et], sim_counts[et]]
    labels = ['Real', 'QR']
    bp = ax.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True, labels=labels)
    bp['boxes'][0].set_facecolor('#2196F3')
    bp['boxes'][1].set_facecolor('#F44336')
    ax.set_title(labels_map[et])
    ax.set_ylabel("Events per 5-min window")

plt.suptitle("Event Frequency: Real vs QR Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()"""))

# ==============================================================================
# SECTION 7: Summary
# ==============================================================================
cells.append(md(r"""---
## 7. Summary and Limitations of the QR Model

### Strengths
1. **Analytical tractability:** The MLE estimator has a closed-form solution, making calibration fast and transparent.
2. **Captures queue-size dependency:** The intensity functions correctly reflect that cancellations increase with queue size and market orders concentrate at the best levels.
3. **Efficient simulation:** The Gillespie algorithm provides exact simulation of the CTMC dynamics.

### Limitations (motivating extensions)
1. **No excitation between events:** The QR model produces uniform transition matrices because it ignores the history of past event types. This is addressed in the **DQR model** (Notebook 2) by including $\eta_{k-1}$ in the state vector.

2. **No intraday seasonality:** The model treats all hours uniformly, whereas real markets show heightened activity at open/close. This is addressed in the **DQR model** by including the hour $h_k$ as a feature.

3. **Queue independence:** Each queue evolves independently, missing important cross-level correlations (e.g., negative correlation between best bid and ask). This is addressed in the **MDQR model** (Notebook 3).

4. **No order size modeling:** All events have unit size in the QR framework. The **MDQR model** adds a separate neural network to predict order size distributions.

5. **Simplified price dynamics:** Reference price changes are handled heuristically. A richer state space (spread, trade imbalance) in the MDQR model produces more realistic price dynamics including the **square-root law of market impact**.

---

**Next:** In Notebook 2, we extend the QR model to the Deep Queue-Reactive (DQR) framework, where neural networks replace the lookup-table intensities and the state space is enriched with additional features."""))

# ==============================================================================
# SAVE NOTEBOOK
# ==============================================================================
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0", "mimetype": "text/x-python",
                          "file_extension": ".py", "codemirror_mode": {"name": "ipython", "version": 3}}
    },
    "cells": cells
}

out = os.path.join(BASE, "01_data_and_qr_model.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Created: {out}")
