# Mobile Game UA Analytics

## Business Question

How should a mobile game measure and optimize user acquisition (UA) spend when attribution is unreliable?

## Key Findings

- **pLTV model (notebook 01)**: the XGBoost/Tweedie model scored a **Spearman rank correlation of just 0.029** between predicted and actual LTV on the held-out test set — essentially no ranking power. That's a disclosed finding, not a hidden flaw: raw engagement features (session count, session length) showed near-zero correlation with spend among payers *before* any model was fit, so a weak result was the honest, expected outcome of this particular dataset rather than a modeling failure to paper over. Country and genre carried more signal than behavior did, which is itself a useful, if unflattering, result.
- **Cookie Cats A/B test (notebook 02)**: moving the progress gate from level 30 to level 40 **significantly hurt D7 retention** (18.2% vs. 19.0%, p = 0.0016, a ~4.5% relative decline), while the D1 effect alone was not statistically significant. The Bayesian read agreed and went further: **P(gate_30 > gate_40) = 99.9% on D7**, with an expected-loss calculation showing shipping gate_40 anyway risks roughly 1,400x more downside than keeping gate_30. Recommendation: **do not ship the gate move.**
- **Marketing Mix Model (notebook 03)**: after fitting adstock + saturation transforms and catching a collinearity-driven sanity-check failure (a naive regression implied a -620x ROI for one channel, which isn't a real number), a non-negativity-constrained regression produced a defensible channel ranking: **Google ~10x ROI, Facebook ~33x ROI**, Email highly efficient but sensitive to its CPM assumption, and Affiliate showing **no detectable effect** rather than a fabricated negative one.

## The Three Analyses

1. **pLTV Model** — When attribution is unreliable, a studio still needs to know which installs are worth bidding up, regardless of which network claims credit for them. This notebook builds an XGBoost model with a Tweedie objective (chosen because the target — in-app spend — is zero-inflated and heavy-tailed, the same shape as insurance claims data) to predict long-term value from early behavioral and demographic signals. The headline finding is a disclosed limitation, not a success story: a rigorous leakage audit removed several features that would have made the model look artificially strong (a spending-tier label derived from the target itself, and purchase fields that only exist *after* a purchase happens), and what's left has very little rank-ordering power (Spearman = 0.029) — an honest result about this dataset's actual signal, reported plainly rather than dressed up.
2. **Cookie Cats A/B Test** — Retention is the multiplier on every UA dollar spent: a player who churns on day two never gets a chance to convert, no matter how well-targeted the install was. This notebook treats a real randomized experiment (gate placement at level 30 vs. level 40) as a causal incrementality read, answering the question two independent ways — a two-proportion z-test with Wilson confidence intervals, and a from-scratch Bayesian Beta-Binomial model with an expected-loss decision rule — and lands on a clear go/no-go recommendation (don't move the gate), while explicitly flagging a marginal sample-ratio-mismatch check and the fact that roughly 63% of players never reached either gate at all.
3. **Marketing Mix Model** — Real mobile UA channel spend data is commercially sensitive and doesn't exist publicly, so this notebook demonstrates the full MMM methodology (geometric adstock for carryover, power-law saturation for diminishing returns, an OLS regression with trend/seasonality controls, and a channel-level ROI readout) on a structurally equivalent multi-channel dataset, with every step written to generalize directly to Meta, Google UAC, and Apple Search Ads spend in production. The most important result isn't the ROI ranking itself — it's catching a naive regression's economically implausible output (a channel with a -620x "ROI") and fixing it with a non-negativity constraint standard in production MMM tools, which is exactly the kind of sanity check that separates a trustworthy model from a broken one that still happens to fit well.

## AI-Assisted Workflow

I used Claude Code a lot in this project, mainly to move faster. It scaffolded notebooks, wrote the repetitive pandas and plotting code, caught bugs, fixed a few chart rendering issues, and helped build out the visualizations. That part of the work was genuinely faster with it than without it, and I'm not going to pretend otherwise.

But the actual thinking was mine. Deciding to use a Tweedie objective instead of plain squared error for the pLTV model, because the spend data is zero-inflated and heavy-tailed, was my call. Catching that a spending-tier column was derived from the target and would have leaked it was mine too, and so was choosing Spearman correlation over R² as the metric that actually matters for a bidding use case. When the MMM notebook produced a nonsensical negative ROI for one channel, I'm the one who caught that it didn't pass a basic sanity check and decided a non-negativity constraint was the right fix, not something Claude flagged on its own. Same with the go/no-go call on the Cookie Cats test: the statistics came out of code, but the decision to weight D7 over D1 and recommend against shipping the gate change was a judgment call I made and would defend in a room.

Claude Code is a fast, capable pair programmer here. It is not the analyst. Every number in this repo I can explain and defend, because I'm the one who decided what to build and why.

## Stack

- **Data / Analysis**: pandas, numpy, scipy, statsmodels, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter
- **App/Dashboard**: Streamlit
- **AI**: Anthropic (Claude)
