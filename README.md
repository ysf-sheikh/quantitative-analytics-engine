# Quantitative Finance & Credit Risk Analytics Engine

A Python portfolio demonstrating quantitative methods developed for the JPMorgan Chase job simulation, spanning asset pricing, derivative valuation, credit default classification, and algorithmic risk bucket optimization.

---

## Summary

This repository contains two production-oriented financial analytics modules designed to automate commodity valuation and optimize consumer lending credit risk management:
1. **Commodity Derivatives Desk:** Implements a continuous pricing engine for natural gas using structural interpolation/extrapolation models, wrapped inside an event-driven asset storage contract valuation simulator.
2. **Credit Risk & Portfolio Management Desk:** Builds an end-to-end default evaluation system that calculates Expected Loss ($EL$) using a Scikit-Learn pipeline and optimizes credit scorecard binning via a custom Dynamic Programming (DP) Log-Likelihood quantizer.

---

## Module Details & Methodology

### 1. Commodity Pricing & Asset Storage Valuation

#### **Continuous Forward Curve Engine (`interpolate_extrapolate.py`)**
To transform discrete monthly market price snapshots into a continuous pricing model, this engine fits a **Natural Cubic Spline** across historical periods. This ensures smooth pricing transitions for mid-month contract evaluation without sharp breaks. 

For future contracts, a **Sinusoidal Ordinary Least Squares (OLS)** regression is deployed to capture systemic linear drift alongside annual commodity seasonality (e.g., winter heating demand spikes):

$$Price(t) = \beta_0 + \beta_1 t + \beta_2 \sin\left(\frac{2\pi t}{365.25}\right) + \beta_3 \cos\left(\frac{2\pi t}{365.25}\right)$$

#### **Event-Driven Contract Pricer (`gas_storage_valuation.py`)**
Valuing physical gas storage contracts requires simulating complex operational real-world constraints. Instead of an inefficient calendar-day loop, this engine utilizes an **optimized event-driven architecture** that processes injection/withdrawal milestones chronologically, dynamically adjusting for:
* Time-dilated, compounding daily storage costs.
* Volumetric injection and withdrawal flow-rate limits.
* Max physical inventory capacity overflows and stockout underflows.

#### **Visualizing the Curve Engine:**
<img width="1200" height="600" alt="Curve Engine" src="https://github.com/user-attachments/assets/4b4ca41e-d57a-43a7-8a13-0b131ae9eef0" />

---

### 2. Credit Risk Analytics & Algorithmic Quantization

#### **Expected Loss Evaluation Engine (`credit_risk_model.py`)**
Predicts a borrower's Probability of Default ($PD$) using a robust classification pipeline. Features are engineered to measure systemic leverage (Debt-to-Income and Loan-to-Income ratios). 

Models are benchmarked using **ROC-AUC** metrics via a stratified data split to handle default imbalances. While ensemble methods (Random Forest) were evaluated, **Logistic Regression** paired with a `StandardScaler` pipeline was selected for deployment to maintain complete feature interpretability and comply with Basel III framework standards. Expected Loss ($EL$) is computed via:

$$EL = PD \times LGD \times EAD$$

Where $EAD$ is the Exposure at Default (outstanding balance) and $LGD$ is the Loss Given Default ($1 - \text{Recovery Rate}$).

#### **Optimal FICO Quantizer (`fico_rating_map.py`)**
Instead of using arbitrary equal-width or quantile binning, this engine applies a custom **Dynamic Programming (DP)** algorithm to discretize continuous FICO scores into structural credit ratings by maximizing **Binomial Log-Likelihood**:

$$\ln \mathcal{L} = \sum_{i=1}^{M} \left[ k_i \ln(p_i) + (n_i - k_i) \ln(1 - p_i) \right]$$

Where $n_i$ represents total borrowers, $k_i$ observed defaults, and $p_i = \frac{k_i}{n_i}$ the internal default rate in bucket $i$. The DP matrix architecture resolves optimal boundaries in $O(B \cdot N^2)$ time, avoiding combinatorial brute-force overhead.

#### **Visualizing Credit Risk Brackets:**
<img width="800" height="500" alt="Fico Rating" src="https://github.com/user-attachments/assets/52e50edf-a341-4695-8ad8-c2aeffe48e7f" />

---

## Setup

### Prerequisites
Ensure you have Python 3.9+ installed. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
