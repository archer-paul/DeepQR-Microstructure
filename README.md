# Deep Learning Meets Queue-Reactive: A Framework for Realistic LOB Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 1. Project Overview
This repository implements the **Deep Queue-Reactive (DQR)** and **Multidimensional Deep Queue-Reactive (MDQR)** frameworks for simulating Limit Order Book (LOB) dynamics. The project is based on the research paper by **Hamza Bodor and Laurent Carlier (2025)**: *"Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation"*.

The project is structured around three main Jupyter Notebooks that guide through the data processing, baseline modeling, and the final deep learning implementations.

### Academic Context
*   **Institution:** Imperial College London
*   **Department:** Department of Mathematics / Finance
*   **Module:** Market Microstructure
*   **Supervised by:** Professor Mathieu Rosenbaum

### Authors
*   **Paul Archer** (CID: 06054057)
*   **Thibault Marty** (CID: 06055275)
*   **Rebecca Laïk** (CID: 01776263)

---

## 2. Core Notebooks

The project is divided into three primary steps:

1.  **[01_data_and_qr_model.ipynb](01_data_and_qr_model.ipynb)**: 
    *   Loading and reconstructing LOBSTER data (AAPL, AMZN, etc.).
    *   Implementation of the **Queue-Reactive (QR)** model (Huang et al., 2015).
    *   Maximum Likelihood Estimation (MLE) of arrival intensities $\lambda^{\eta}(q)$.

2.  **[02_dqr_model.ipynb](02_dqr_model.ipynb)**: 
    *   Implementation of the **Deep Queue-Reactive (DQR)** model.
    *   Transition from local queue sizes to a rich state vector $x_k$.
    *   Neural Network parameterization of intensity functions.

3.  **[03_mdqr_model.ipynb](03_mdqr_model.ipynb)**: 
    *   Implementation of the **Multidimensional Deep Queue-Reactive (MDQR)** framework.
    *   Joint modeling of event intensities and order sizes across all levels.
    *   Verification of stylized facts (Market Impact, Correlations, Return Distributions).

---

## 3. Theoretical Background

### Baseline: The Queue-Reactive (QR) Model
The QR model represents the LOB as a set of independent queues. For each event $\eta \in \{L, C, M\}$, the arrival intensity $\lambda^{\eta}(q)$ is a function of the local queue size $q$:
$$ \mathcal{L}(\{\lambda^\eta\} \mid \mathcal{E}) = \prod_{k=1}^{N} \exp\left(-\Lambda(q_k)\Delta t_k\right)\lambda^{\eta_k}(q_k) $$

### The DQR Extension
The **Deep Queue-Reactive** model conditions intensities on a rich state vector $x_k$ via neural networks:
$$ \lambda^\eta_\theta(x_k) = \text{NN}_\theta(x_k) $$

### Multidimensional MDQR
The **MDQR** models the LOB as a unified multidimensional system, jointly optimizing intensity networks and order size classification models.

---

## 4. Supporting Architecture

While the notebooks are the main entry points, the following modules provide the underlying logic:

*   **mle/**: Statistical Calibration and LOBSTER IO.
*   **models/**: Neural Network definitions (DQR/MDQR).
*   **simulator.py**: Discrete-event simulator engine.
*   **analysis.py**: Tools for heatmaps and stylized facts validation.
*   **state.py**: Real-time state representation of the LOB.

---

## 5. Usage & Setup

### Prerequisites
*   Python 3.10+
*   `pip install torch pandas numpy matplotlib plotly`

### Installation
1. Clone the repository.
2. Ensure LOBSTER data is placed in the `data/` folder.
3. Open the notebooks in order (01 -> 02 -> 03) to follow the implementation.

---

## 6. License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 7. References
*   Bodor, H., & Carlier, L. (2025). *Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation*. arXiv:2501.08822.
*   Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). *Simulating and analyzing order book dynamics: the queue-reactive model*. Journal of Statistical Physics.
