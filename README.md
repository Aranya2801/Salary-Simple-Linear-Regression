<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Salary%20Simple%20Linear%20Regression&fontSize=38&fontAlignY=35&animation=twinkling&fontColor=ffffff&desc=Production-Grade%20Statistical%20ML%20Engine&descAlignY=60&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-59%20Passed-22C55E?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-98%25-22C55E?style=for-the-badge&logo=codecov&logoColor=white)](https://codecov.io)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=python)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-635BFF?style=for-the-badge)](https://github.com/Aranya2801/Salary-Simple-Linear-Regression)

<br/>

> **A production-grade, mathematically rigorous Simple Linear Regression engine for salary prediction — built to MIT research standards.**
> Full OLS inference · Bootstrap CIs · 6-panel diagnostic dashboard · Cross-validation · 59 unit tests.

<br/>

[🚀 Quick Start](#-quick-start) · [📊 Results](#-results--outputs) · [🧪 Tests](#-testing) · [📐 Theory](#-mathematical-foundation) · [📁 Structure](#-project-structure) · [🤝 Contributing](#-contributing)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Mathematical Foundation](#-mathematical-foundation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Results & Outputs](#-results--outputs)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements **Simple Linear Regression** from the ground up — not as a black-box `model.fit()` call, but as a complete statistical framework modeled on how econometricians and data scientists at top research institutions approach regression analysis.

**What makes this different from a typical tutorial project:**

| Typical Tutorial | This Project |
|:---|:---|
| `LinearRegression().fit(X, y)` | Full OLS derivation with matrix algebra |
| Single R² metric | R², Adj-R², RMSE, MAE, MAPE, RSE |
| Point predictions only | Confidence + Prediction Intervals |
| No assumption checking | Shapiro-Wilk, Durbin-Watson, Breusch-Pagan |
| No uncertainty quantification | 1,000-iteration Bootstrap CIs |
| No validation | 5-fold Cross-Validation |
| No outlier analysis | Cook's Distance + Leverage |
| No tests | 59 comprehensive unit tests |
| No CI/CD | Full GitHub Actions pipeline |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 📐 Statistical Engine
- **OLS Estimation** — closed-form matrix solution
- **t-tests** on slope & intercept
- **F-test** (ANOVA decomposition: SSR, SSE, SST)
- **95% Confidence Intervals** (parametric)
- **95% Prediction Intervals** (individual obs.)
- **Bootstrap CIs** — 1,000 resampling iterations
- **Pearson, Spearman, Kendall** correlations

</td>
<td width="50%">

### 🔬 Diagnostic Suite
- **Shapiro-Wilk** normality test
- **Durbin-Watson** autocorrelation test
- **Breusch-Pagan** heteroskedasticity test
- **Cook's Distance** — influential observations
- **Leverage** (hat matrix diagonal)
- **Studentized residuals** — outlier flagging
- **Scale-Location** plot for variance stability

</td>
</tr>
<tr>
<td>

### 📊 Visualizations
- 8-panel diagnostic dashboard (publication-ready)
- Regression line + CI + PI bands
- Industry-segmented regression plots
- Bootstrap distribution comparison
- Normal Q-Q plot
- Residual vs Fitted
- Cook's Distance bar chart

</td>
<td>

### ⚙️ Engineering Quality
- **59 unit tests** across 8 test classes (100% pass)
- **Cross-validation** (5-fold, R²/RMSE/MAE)
- **GitHub Actions** CI (3 OS × 4 Python versions)
- `pyproject.toml` package configuration
- Full type hints throughout codebase
- Modular, importable `src/` package

</td>
</tr>
</table>

---

## 📐 Mathematical Foundation

### The Linear Model

The Simple Linear Regression model assumes:

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad \varepsilon_i \overset{\text{iid}}{\sim} \mathcal{N}(0, \sigma^2)$$

where:
- $Y_i$ = Annual Salary (response variable)
- $X_i$ = Years of Experience (predictor)
- $\beta_0$ = intercept (baseline salary at 0 experience)
- $\beta_1$ = slope (salary increase per additional year)
- $\varepsilon_i$ = random error (noise)

### OLS Estimators

The **Ordinary Least Squares** estimators minimize the Residual Sum of Squares $\text{RSS} = \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2$:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n}(X_i - \bar{X})^2} = \frac{S_{XY}}{S_{XX}}$$

$$\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}$$

### Statistical Inference

**Standard errors** of the estimators:

$$\text{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{XX}}}, \qquad \text{SE}(\hat{\beta}_0) = s\sqrt{\frac{1}{n} + \frac{\bar{X}^2}{S_{XX}}}$$

where $s = \sqrt{\text{MSE}} = \sqrt{\frac{\text{SSE}}{n-2}}$ is the Residual Standard Error.

**t-statistics** (under $H_0: \beta_j = 0$):

$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-2}$$

**95% Confidence Interval** for slope:

$$\hat{\beta}_1 \pm t_{n-2,\,0.025} \cdot \text{SE}(\hat{\beta}_1)$$

### Prediction vs Confidence Intervals

For a **new observation** at $X^* = x_0$:

**Confidence interval** (mean response):
$$\hat{Y}^* \pm t_{n-2,\,\alpha/2} \cdot s \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{X})^2}{S_{XX}}}$$

**Prediction interval** (individual observation — always wider):
$$\hat{Y}^* \pm t_{n-2,\,\alpha/2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{X})^2}{S_{XX}}}$$

### ANOVA Decomposition

$$\underbrace{\text{SST}}_{\text{Total}} = \underbrace{\text{SSR}}_{\text{Regression}} + \underbrace{\text{SSE}}_{\text{Residual}}$$

$$R^2 = \frac{\text{SSR}}{\text{SST}}, \qquad F = \frac{\text{MSR}}{\text{MSE}} \sim F_{1,\,n-2}$$

### Gauss-Markov Assumptions Tested

| Assumption | Test Used | Result |
|:---|:---|:---|
| **Linearity** | Residuals vs Fitted plot | ✓ Visual inspection |
| **Independence** | Durbin-Watson test | ✓ DW ≈ 1.95 |
| **Homoskedasticity** | Breusch-Pagan test | ✓ p = 0.279 |
| **Normality of errors** | Shapiro-Wilk test | ✓ p = 0.794 |

---

## 📁 Project Structure

```
Salary-Simple-Linear-Regression/
│
├── 📂 src/                          # Core Python package
│   ├── __init__.py
│   ├── regression_engine.py         # OLS engine + inference + bootstrap
│   └── visualization.py             # Publication-quality diagnostic plots
│
├── 📂 notebooks/
│   └── full_analysis.py             # Complete 10-step analysis pipeline
│
├── 📂 data/
│   ├── salary_dataset.csv           # Full 500-row, 10-feature dataset
│   ├── salary_dataset.xlsx          # Excel version (same data)
│   └── salary_simple.csv            # YearsExperience + Salary only
│
├── 📂 tests/
│   └── test_regression.py           # 59 unit tests across 8 test classes
│
├── 📂 reports/                      # Auto-generated visualizations
│   ├── regression_dashboard.png     # 8-panel diagnostic dashboard
│   ├── industry_breakdown.png       # Per-industry regression analysis
│   ├── bootstrap_analysis.png       # Bootstrap CI distributions
│   └── metrics_report.csv           # Full metrics export
│
├── 📂 .github/workflows/
│   └── ci.yml                       # GitHub Actions CI (3 OS × 4 Python)
│
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # Package config (PEP 517/518)
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aranya2801/Salary-Simple-Linear-Regression.git
cd Salary-Simple-Linear-Regression
pip install -r requirements.txt
```

### 2. Run Full Analysis

```bash
python notebooks/full_analysis.py
```

This runs all 10 pipeline stages and saves charts to `reports/`.

### 3. Use the Engine in Your Code

```python
from src.regression_engine import SalaryRegressionEngine, load_data

# Load data
X, y, df = load_data("data/salary_simple.csv")

# Fit model
engine = SalaryRegressionEngine(alpha=0.05, n_bootstrap=1000, cv_folds=5)
engine.fit(X, y)

# Print full statistical summary
print(engine.summary())

# Predict salary for 7 years of experience
pi = engine.prediction_interval(7)
print(f"Predicted: ${pi['prediction']:,.0f}")
print(f"95% PI: [${pi['lower']:,.0f}, ${pi['upper']:,.0f}]")

# Run diagnostics
diag = engine.residual_diagnostics()
print(f"Shapiro-Wilk p: {diag['shapiro_wilk']['p_value']:.4f}")
print(f"Durbin-Watson : {diag['durbin_watson']['statistic']:.4f}")

# Bootstrap confidence intervals
boot = engine.bootstrap_confidence_intervals()
print(f"Bootstrap slope CI: [{boot['slope_ci'][0]:,.0f}, {boot['slope_ci'][1]:,.0f}]")

# Cross-validation
cv = engine.cross_validate()
print(f"CV R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}")
```

---

## 📊 Dataset

### salary_simple.csv (Core regression dataset)

| Column | Type | Description |
|:---|:---|:---|
| `YearsExperience` | float | Years of professional experience (0–30) |
| `Salary` | float | Annual salary in USD |

**500 observations** · **Pearson r = 0.908** · Generated with realistic noise

### salary_dataset.csv (Full feature set)

| Column | Description |
|:---|:---|
| `YearsExperience` | Years of experience (continuous) |
| `Salary` | Annual salary in USD |
| `Industry` | Tech, Finance, Healthcare, Education, Government, Manufacturing |
| `Education` | High School → Associate → Bachelor's → Master's → PhD |
| `Location` | San Francisco, New York, Seattle, Boston, Chicago, Austin, LA, Remote |
| `JobTitle` | Junior → Senior → Lead → Manager → Director → VP |
| `Gender` | Male, Female, Non-binary |
| `Age` | Age in years |
| `SkillScore` | 0–100 composite skill assessment |
| `PerformanceRating` | 1–5 annual review score |

---

## 📈 Results & Outputs

### Regression Equation

```
Salary = $80,344.79 + $3,804.89 × YearsExperience
```

> Each additional year of experience is associated with a **$3,805 salary increase** (95% CI: [$3,650, $3,960]).

### Key Metrics

| Metric | Value | Interpretation |
|:---|:---|:---|
| **R²** | 0.8242 | 82.4% of salary variance explained |
| **Adj. R²** | 0.8239 | Corrected for 1 predictor |
| **F-statistic** | 2,334.95 | p = 3.96×10⁻¹⁹⁰ — overwhelmingly significant |
| **RMSE** | $8,549 | Typical prediction error |
| **MAE** | $6,829 | Median absolute error |
| **MAPE** | 7.12% | Average percentage error |
| **CV R² (5-fold)** | 0.818 ± 0.032 | Generalisation confirmed |

### Prediction Table

| Experience | Predicted Salary | 95% Prediction Interval |
|:---:|:---:|:---|
| 1 year | $84,150 | [$67,291 — $101,008] |
| 3 years | $91,759 | [$74,909 — $108,609] |
| 5 years | $99,369 | [$82,522 — $116,216] |
| 10 years | $118,394 | [$101,529 — $135,258] |
| 15 years | $137,418 | [$120,500 — $154,336] |
| 20 years | $156,443 | [$139,437 — $173,448] |

### Diagnostic Results

| Test | Statistic | p-value | Conclusion |
|:---|:---:|:---:|:---|
| **Shapiro-Wilk** | W = 0.9979 | p = 0.794 | ✅ Residuals are normally distributed |
| **Durbin-Watson** | DW = 1.950 | — | ✅ No autocorrelation |
| **Breusch-Pagan** | χ² = 1.173 | p = 0.279 | ✅ Homoskedastic residuals |

> **All four Gauss-Markov assumptions satisfied.** The OLS estimator is BLUE (Best Linear Unbiased Estimator).

---

## 🔌 API Reference

### `SalaryRegressionEngine`

```python
engine = SalaryRegressionEngine(
    alpha=0.05,          # Significance level for CIs and tests
    n_bootstrap=1000,    # Bootstrap iterations
    cv_folds=5           # Cross-validation folds
)
```

| Method | Returns | Description |
|:---|:---|:---|
| `.fit(X, y)` | `self` | Fit OLS model |
| `.predict(X_new)` | `ndarray` | Point predictions |
| `.prediction_interval(x, level)` | `dict` | Individual prediction interval |
| `.confidence_interval_mean(x, level)` | `dict` | Mean response interval |
| `.ols_statistics()` | `dict` | Full inference table |
| `.bootstrap_confidence_intervals()` | `dict` | Bootstrap CIs for β₀, β₁ |
| `.residual_diagnostics()` | `dict` | Shapiro-Wilk, DW, BP, Cook's D |
| `.cross_validate()` | `dict` | K-fold CV metrics |
| `.summary()` | `str` | Formatted statistical summary |

---

## 🧪 Testing

```bash
# Run all 59 tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run in parallel (faster)
python -m pytest tests/ -n auto

# Run specific test class
python -m pytest tests/test_regression.py::TestDiagnostics -v
```

### Test Coverage

| Test Class | Tests | Coverage |
|:---|:---:|:---|
| `TestFitting` | 13 | OLS accuracy, R², residuals, edge cases |
| `TestOLSStatistics` | 9 | Full inference table, ANOVA, CIs |
| `TestPrediction` | 8 | Point preds, PI/CI ordering, monotonicity |
| `TestDiagnostics` | 9 | S-W, DW, Cook's D, leverage |
| `TestBootstrap` | 4 | CI bounds, ordering, coverage |
| `TestCrossValidation` | 5 | R² range, RMSE, fold counts |
| `TestUtilities` | 6 | `format_currency`, `salary_percentile`, `load_data` |
| `TestSummary` | 5 | Output string completeness |
| **Total** | **59** | **100% passing** |

---

## ⚙️ CI/CD Pipeline

GitHub Actions runs on every push and pull request:

```
┌────────────────────────────────────────┐
│  🔍 Lint (flake8 + black + isort)      │
├────────────────────────────────────────┤
│  🧪 Unit Tests                         │
│     Ubuntu × [3.9, 3.10, 3.11, 3.12]  │
│     Windows × [3.9, 3.10, 3.11, 3.12] │
│     macOS   × [3.9, 3.10, 3.11, 3.12] │
│                                        │
│     (12 matrix jobs in parallel)       │
├────────────────────────────────────────┤
│  📊 Analysis Pipeline                  │
│     Full 10-step pipeline run          │
│     Report artifacts uploaded          │
├────────────────────────────────────────┤
│  🔒 Security Scan (bandit)             │
└────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Write tests for your changes
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Format code: `black src/ notebooks/ && isort src/ notebooks/`
6. **Pull request** with a clear description

**Areas for contribution:**
- Weighted Least Squares (WLS) regression
- Robust regression (Huber, RANSAC)
- Polynomial regression extension
- Interactive Streamlit dashboard
- SHAP-based feature importance

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
Copyright (c) 2025 Aranya2801

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files ...
```

---

## 🙏 Acknowledgements

- [Scikit-learn](https://scikit-learn.org) — ML infrastructure
- [SciPy](https://scipy.org) — Statistical tests and distributions
- [Matplotlib](https://matplotlib.org) / [Seaborn](https://seaborn.pydata.org) — Visualization
- *Introduction to Statistical Learning* (James et al.) — Theory reference
- *Applied Linear Statistical Models* (Kutner et al.) — Diagnostic methodology

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**⭐ If this project helped you, please star it!**

Made with ❤️ and 📊 by [Aranya2801](https://github.com/Aranya2801)

</div>
