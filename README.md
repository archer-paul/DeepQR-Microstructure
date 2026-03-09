# Deep Queue-Reactive Framework for Limit Order Book Simulation

This repository provides a comprehensive implementation of the Deep Queue-Reactive (DQR) and Multidimensional Deep Queue-Reactive (MDQR) models for simulating high-frequency Limit Order Book (LOB) dynamics. The framework generalizes the classical Queue-Reactive model using neural point processes to capture high-dimensional dependencies and market stylized facts, following the research by Bodor and Carlier (2025).

## 1. Project Overview

The core objective of this project is to simulate realistic LOB trajectories by parameterising event arrival intensities as functions of the book's state. We transition from independent queue modeling (QR) to a unified multidimensional system (MDQR) that incorporates cross-level interactions and conditional order size distributions.

The implementation utilizes LOBSTER data for NASDAQ equities, specifically focusing on Intel (INTC) to approximate the "large-tick" regime characteristic of Bund futures analyzed in the foundational paper.

### Core Technical Contributions
- **Neural Intensity Parameterisation:** Employment of Multi-Layer Perceptrons (MLPs) with Softplus activations to estimate event arrival intensities in high-dimensional state spaces.
- **Multidimensional State Representation:** Relaxation of the independence assumption between price levels to model joint queue dynamics.
- **Conditional Size Modeling (SizeNet):** Implementation of a categorical distribution model to reproduce empirical order size patterns conditional on event type and level.
- **Empirical Validation Suite:** Comprehensive benchmarking against market stylized facts, including the square-root law of market impact and cross-level correlations.

---

## 2. Theoretical Framework and Implementation

The project is structured into three progressive stages, each detailed in a dedicated Jupyter notebook.

### Step 1: Queue-Reactive (QR) Baseline
The foundational model represents the LOB as a set of independent queues. Event arrivals (Limit, Cancel, Market) are modeled as Poisson processes with intensities $\lambda^\eta(q)$ dependent on the local queue size $q$.
- **Notebook:** `01_data_and_qr_model.ipynb`
- **Methodology:** Maximum Likelihood Estimation (MLE) of arrival intensities and simulation via the Gillespie (Discrete Event Simulation) algorithm.
- **Enhancement:** Refinement of inter-event time ($\Delta t$) calculations by segmenting intervals according to mid-price shifts, preventing gap inflation.

### Step 2: Deep Queue-Reactive (DQR) Extension
The DQR model replaces lookup tables with neural architectures, enabling the inclusion of exogenous features into the state vector $\mathbf{x}_k$.
- **Notebook:** `02_dqr_model.ipynb`
- **Feature Engineering:** Integration of intraday seasonality (hour embeddings) and recent event excitation.
- **Analysis:** Demonstration of how neural parameterisation captures the "U-shaped" activity profile and event clustering.

### Step 3: Multidimensional Deep Queue-Reactive (MDQR) Model
The MDQR framework represents the LOB as a unified system, jointly modeling all monitored price levels ($K=5$ levels on each side).
- **Notebook:** `03_mdqr_model.ipynb`
- **Architecture:** A joint intensity network for $3 \times 2K$ event types and a separate `SizeNet` for order size distributions.
- **Loss Function:** Negative Log-Likelihood (NLL) derived from the compensator of the point process, optimized using Adam with Cosine Annealing.
- **Stylized Facts Verification:**
    - **Market Impact:** Validation of the concave price response and power-law scaling.
    - **Queue Correlations:** Capturing negative correlations between best bid/ask volumes and positive correlations within the same side of the book.
    - **Distributional Accuracy:** Alignment of simulated returns with historical excess kurtosis and heavy tails.

---

## 3. Project Architecture

The codebase is organized into modular components to support scalability:

- **mle/**: Modules for LOBSTER data parsing, state reconstruction, and MLE calibration.
- **models/**: PyTorch implementations of the DQR and MDQR neural architectures.
- **simulator.py**: Discrete-event simulation engine based on the Gillespie algorithm.
- **state.py**: Object-oriented representation of the LOB state and event types.
- **lobster.py**: Specialized utilities for NASDAQ LOBSTER data processing.
- **analysis.py**: Statistical tools for validating stylized facts and generating visualizations.

---

## 4. Setup and Usage

### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA)
- Scientific Stack: pandas, numpy, matplotlib, scipy, scikit-learn

### Instructions
1. **Data Placement:** Ensure LOBSTER message and orderbook CSV files are located in the `data/` directory.
2. **Execution:** Run the notebooks in sequential order (01, 02, 03) to perform data preprocessing, model calibration, and simulation validation.

---

## 5. References

- Bodor, H., & Carlier, L. (2025). *Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation*. arXiv:2501.08822.
- Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). *Simulating and analyzing order book dynamics: the queue-reactive model*. Journal of Statistical Physics.
