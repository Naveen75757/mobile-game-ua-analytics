# Mobile Game UA Analytics

## Business Question

How should a mobile game measure and optimize user acquisition (UA) spend when attribution is unreliable?

## Key Findings

_TBD — populate as analyses are completed._

## The Three Analyses

1. **pLTV Model** — Predict long-term player value from early behavioral signals to guide bidding and budget allocation.
2. **Cookie Cats A/B Test** — Randomized experiment analysis (frequentist + Bayesian) on a progress-gate placement change, framed as an incrementality read with a go/no-go recommendation: retention is the multiplier on every UA dollar spent, so a retention-damaging product change is a hidden tax on acquisition efficiency.
3. **Marketing Mix Model** — Adstock (carryover) + saturation (diminishing returns) + regression-based channel ROI on a structurally equivalent multi-channel spend dataset, standing in for real mobile UA spend data (Meta/Google UAC/Apple Search Ads), which is commercially sensitive and not publicly available. Demonstrates an attribution-agnostic alternative to user-level tracking, including catching and correcting a collinearity-driven sanity-check failure in the naive regression.

## AI-Assisted Workflow

_TBD — document how Claude/AI tooling was used throughout the analysis (e.g., code generation, exploratory analysis, report drafting)._

## Stack

- **Data / Analysis**: pandas, numpy, scipy, statsmodels, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter
- **App/Dashboard**: Streamlit
- **AI**: Anthropic (Claude)
