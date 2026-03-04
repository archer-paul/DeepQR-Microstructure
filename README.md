# Deep Learning Meets Queue-Reactive: A Framework for Realistic LOB Simulation

![LOB Simulation Header](LOB_Simulation.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

## 1. Project Overview

This repository provides a comprehensive implementation of the Deep Queue-Reactive (DQR) and Multidimensional Deep Queue-Reactive (MDQR) frameworks for simulating Limit Order Book (LOB) dynamics. The work is based on the research by Bodor and Carlier (2025), which generalizes the classical Queue-Reactive model using deep learning architectures to capture high-dimensional dependencies and market stylized facts.

The project demonstrates the transition from a lookup-table based point process (QR) to a neural network parameterization (DQR) and finally to a multidimensional joint model (MDQR) that incorporates cross-level interactions and realistic order size distributions. We utilize LOBSTER data for NASDAQ equities, specifically focusing on INTC as a representative large-tick asset to align with the characteristics of the Bund futures analyzed in the original paper.

### Key Contributions
- Neural Intensity Parameterization: Employing Multi-Layer Perceptrons (MLPs) to estimate event arrival intensities in high-dimensional state spaces.
- Multidimensional State Space: Relaxing the independence assumption between price levels to model cross-queue interactions.
- Conditional Order Size Modeling: Implementing a categorical distribution model via neural networks to reproduce empirical order size patterns.
- Empirical Validation: Extensive benchmarking against market stylized facts including the square-root law of market impact and inter-queue correlations.

---

## 2. Theoretical Framework and Implementation

The implementation is structured into three progressive stages, each detailed in a dedicated Jupyter Notebook.

### Step 1: Queue-Reactive (QR) Baseline
The foundational model follows the approach of Huang et al. (2015), where the LOB is represented as a set of independent queues. Event arrivals (Limit, Cancel, Market) are modeled as Poisson processes with intensities $\lambda^\eta(q)$ dependent on the local queue size $q$.
- Notebook: 01_data_and_qr_model.ipynb
- Methodology: Maximum Likelihood Estimation (MLE) of arrival intensities and simulation using the Gillespie (Discrete Event Simulation) algorithm.
- Asset Selection: Transition from AAPL to INTC to better approximate the "large-tick" regime (where the spread is predominantly one tick), ensuring compatibility with the paper's findings on the Bund futures market.

![QR Model Intensities](QR%20Model%20--%20Fitted%20Intensity%20Functions%20INTC.png)
*Figure 1: Fitted arrival intensities for INTC across different event types and price levels.*

### Step 2: Deep Queue-Reactive (DQR) Extension
The DQR model replaces lookup tables with neural networks, enabling the inclusion of exogenous and historical features into the state vector $x_k$.
- Notebook: 02_dqr_model.ipynb
- Feature Engineering: Integration of intraday seasonality (hour of the day) and event excitation (type of the preceding event) into the intensity estimation.
- Analysis: Demonstrates how the DQR model captures the "U-shaped" activity profile and the clustering of events, significantly improving log-likelihood and next-event prediction accuracy over the baseline.

### Step 3: Multidimensional Deep Queue-Reactive (MDQR) Model
The MDQR framework represents the LOB as a unified multidimensional system, jointly modeling all monitored price levels.
- Notebook: 03_mdqr_model.ipynb
- Model Architecture: A joint intensity network for $3 \times 2K$ event types and a separate categorical network for order size distribution $\{1, \dots, 200\}$.
- Stylized Facts Verification:
    - Market Impact: Validation of the concave price response and the square-root law of impact.
    - Queue Correlations: Capturing the negative correlation between best bid and ask volumes and positive correlations within the same side of the book.
    - Distributional Accuracy: Alignment of simulated queue sizes and returns with empirical distributions.

---

## 3. Project Architecture

The codebase is organized into modular components to support scalability and reproducibility:

- mle/: Modules for LOBSTER data parsing, state reconstruction, and MLE calibration.
- models/: PyTorch implementations of the DQR and MDQR neural architectures.
- simulator.py/: Discrete-event simulation engine based on the Gillespie algorithm.
- state.py/: Object-oriented representation of the Limit Order Book state and event types.
- intensities.py/: Implementation of intensity functions (analytical, DQR, and MDQR).
- analysis.py/: Statistical tools for validating stylized facts and generating comparative visualizations.

---

## 4. Setup and Usage

### Prerequisites
- Python 3.10+
- PyTorch (compatible with CPU or CUDA)
- Scientific Stack: pandas, numpy, matplotlib, scipy, scikit-learn

### Instructions
1. Clone the repository.
2. Data Placement: Ensure LOBSTER message and orderbook CSV files are located in the data/ directory.
3. Execution: Run the notebooks in sequential order (01, 02, 03) to perform data preprocessing, model calibration, and simulation validation.

---

## 5. Academic Context

- Institution: Imperial College London
- Module: Market Microstructure
- Supervision: Professor Mathieu Rosenbaum

### Authors
- Paul Archer (CID: 06054057)
- Thibault Marty (CID: 06055275)
- Rebecca Laïk (CID: 01776263)

---

## 6. References
- Bodor, H., & Carlier, L. (2025). Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation. arXiv:2501.08822.
- Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). Simulating and analyzing order book dynamics: the queue-reactive model. Journal of Statistical Physics.
- LOBSTER Data: https://lobsterdata.com/
