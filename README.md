# Mobile Game UA Analytics

## The Problem

Since Apple's iOS 14.5 privacy changes, mobile UA teams have been making budget decisions with worse information than they used to have. SKAdNetwork windows are truncated and delayed, a large share of iOS users opt out of tracking entirely, and the install and revenue numbers reported by Meta, Google, and a studio's own mobile measurement partner routinely disagree for the same campaign. You cannot optimize spend by trusting a single attributed number, because that number is now an estimate built on a shrinking and biased sample. This project looks at what a UA team can still lean on when attribution gets shaky: modeling player value straight from behavior, running real experiments that don't need a click path to be trustworthy, and building aggregate models that work off spend and outcomes instead of individual-level tracking.

## What I Built

Three analyses, each tackling a different piece of the same question: how do you measure and optimize UA spend when you can't fully trust attribution.

### 1. Predicted LTV Model

**Why this approach.** If you can't trust which network gets credit for an install, you can still ask a different question: based on how this player behaves in their first few sessions, are they likely to be worth a lot or a little? That's a first party signal that doesn't care what the ad network claims. I built this as an XGBoost regression model predicting in-app spend from early engagement and demographic data.

**The Tweedie choice.** The target here, in-app purchase amount, is about 95% zeros with a long tail running up into the thousands of dollars for a handful of whales. A model trained on plain squared error would spend most of its effort chasing that tail and get the typical small spender wrong. Tweedie regression is built for exactly this shape of data, it's the standard choice in insurance claims modeling for the same reason, so I used it instead of a plain regression or a naive log transform.

**The leakage catch.** Before training anything, I audited the columns. One column, a spending tier label, turned out to be a direct bucketing of the target itself, so a model using it would just be reading the answer off the label. Three other columns (payment method, days to first purchase, last purchase date) only exist for users who already made a purchase, meaning they're only populated after the outcome you're trying to predict has already happened. All four got dropped. Using them would have made the model look artificially good and then fail immediately in production, where none of that information exists before a purchase happens.

**Why Spearman over R squared.** For a bidding decision, what matters isn't hitting the exact dollar amount, it's whether the model correctly ranks who's worth more. A UA team acting on this model bids more on users predicted to be valuable. Spearman rank correlation measures exactly that. R squared would reward the model for guessing well on the whales while telling you nothing about whether it ranked the middle of the distribution correctly, which is where most of the actual bidding decisions happen.

**What I found.** The model scored a Spearman correlation of 0.029 on the held out test set, basically no ranking power. That's not a bug I decided to bury. Before training anything, the raw correlation between session count or session length and eventual spend, among people who paid at all, was close to zero. The weak result is an honest reflection of what signal exists in this particular dataset, not a failure of the modeling approach. Country and game genre carried more signal than actual behavior did, which is itself worth knowing.

**What it means and what better data would look like.** If a studio saw this result on their own data, the right move isn't to force a better number out of the same features, it's to ask whether the features are the problem. This dataset is a single cross sectional snapshot with no install date, so there's no way to fix a prediction window (like day 7 behavior predicting day 90 spend) or separate "hasn't paid yet" from "will never pay." Real production data with a proper install cohort, a fixed observation window, and a longer behavioral history (session timing, level progression, ad engagement) would very likely carry more predictive signal than what's available here. The finding to take away is that early session count and length alone are a weak basis for value based bidding, and a studio should check whether that's true for their own game before betting budget on it.

### 2. Cookie Cats A/B Test

**Why retention matters here.** Every dollar spent on UA only pays off if the player sticks around long enough to monetize. A product change that quietly hurts retention is a tax on every single install a studio buys afterward, regardless of how well targeted that install was. This analysis treats a real randomized experiment, moving a progress gate from level 30 to level 40, as a test of exactly that kind of hidden cost.

**Two methods, on purpose.** I ran both a frequentist two proportion z test with Wilson confidence intervals and a Bayesian beta binomial model with an expected loss calculation. The frequentist test answers "is this difference likely to be real." The Bayesian version answers a more useful question for an actual decision: "if I'm wrong, how much do I lose." Running both and having them agree is a good check that neither approach is doing something weird with this particular dataset.

**The sample ratio check.** Before trusting any result from an experiment, you have to check the experiment itself was run cleanly. I ran a chi square test comparing the actual 44,700 versus 45,489 split against an expected 50/50 split, and it came back with p equals 0.0086, below the stricter threshold experimentation platforms typically use for this exact check. The imbalance is small in absolute terms, about a percentage point, so I didn't throw out the analysis, but I flagged it clearly rather than pretending the randomization was perfectly clean.

**The 63 percent caveat.** About 63 percent of all players in this dataset never played enough rounds to plausibly reach either gate. That means the effect this analysis measures is diluted by a majority of players who could never have been affected by the change in the first place. The real effect, among players who actually hit the gate, is almost certainly larger than what shows up in the topline numbers. That's a reason to trust the direction of the finding even more, not less.

**What I found.** Day 1 retention was not statistically significant, gate_30 at 44.8 percent versus gate_40 at 44.2 percent, p equals 0.074. Day 7 retention was significant, 19.0 percent versus 18.2 percent, p equals 0.0016, a relative decline of about 4.5 percent for the later gate. The Bayesian model put the probability that gate_30 truly beats gate_40 on day 7 at 99.9 percent.

**What the expected loss numbers actually mean.** This is the part I'd lead with in a room. If you ship the level 40 gate and you're wrong about it, you'd expect to give up about 0.82 percentage points of day 7 retention. If you keep the level 30 gate and you're wrong about that, the expected cost is close to zero, about 0.00006 percentage points. That's a difference of roughly 13,800 times in downside. In plain terms, one side of this bet loses you real retained players if it doesn't work out, and the other side barely costs you anything even in the worst case. When the risk is that lopsided, you don't need overwhelming evidence to make the call, you just need the evidence to not point strongly the other way, and here it doesn't.

**The recommendation.** Don't move the gate to level 40. The day 7 result is the one that matters most here because it's the metric closest to where the mechanism actually operates, and it's both statistically significant and practically meaningful. Day 1 alone would not have been enough to act on, which is exactly why a studio shouldn't let an early, noisy metric drive a launch decision on its own.

### 3. Marketing Mix Model

**Why MMM exists.** Attribution tries to trace an individual user's path back to an ad. MMM skips that problem entirely. It only needs weekly spend by channel and a weekly outcome like installs or revenue, so SKAdNetwork thresholds, opt out rates, and network versus MMP disagreements simply don't enter into it. That's why MMM has become the standard complement to attribution industry wide, and why I built this notebook to demonstrate the full pipeline even though real mobile UA spend data is too commercially sensitive to find publicly.

**Adstock and saturation, in plain English.** An ad a player saw last week doesn't stop influencing them the moment the week ends, some of that impression's effect carries forward, fading over time. Adstock models that carryover with a decay rate per channel. Saturation models a separate, second effect: the first thousand impressions you buy usually do more than the millionth, because you eventually run out of people who haven't already made up their mind. I modeled that as a power curve that bends over as exposure increases. Both parameters were tuned per channel from the data rather than guessed.

**The negative ROI catch.** The first regression I ran, without any constraints, produced a channel contribution implying roughly negative 620x ROI for the affiliate channel. That is not a real number, no channel destroys more value than the entire business generates. It happened because the channels move together over time (several are trending up or down together), which let the model shuffle credit between them in ways that still fit the data well overall while being nonsense for any individual channel. I caught it, diagnosed why with a multicollinearity check, and refit the model with a constraint that channel effects can't be negative, the same fix production MMM tools like Meta's Robyn use by default. The refit barely changed the model's overall fit, which confirmed the constraint wasn't hiding a real effect, it was just removing a numerical artifact.

**What I found and what a UA team does with it.** After the fix, Google showed roughly a 10x return, Facebook roughly 33x, email looked extremely efficient but that number is almost entirely driven by an assumed near zero delivery cost rather than a real finding, and affiliate showed no detectable effect at its current, fairly small exposure level rather than a fabricated negative one. A UA team looking at this would keep scaling Facebook and Google, treat the email number as a reason to double check the real cost data rather than as a headline result, and either grow affiliate spend enough to actually measure an effect or deprioritize it, rather than assuming it's actively hurting the business.

## Key Numbers

- pLTV model: Spearman rank correlation of 0.029 on held out test data, an honest null result driven by near zero correlation between early behavior and spend in this dataset.
- Cookie Cats day 7 retention: gate_30 at 19.0 percent versus gate_40 at 18.2 percent, p equals 0.0016, Bayesian probability gate_30 is better equals 99.9 percent.
- Cookie Cats expected loss: shipping the level 40 gate risks roughly 13,800 times more expected retention loss than keeping the level 30 gate.
- MMM channel ROI after correcting the sanity check failure: Google about 10x, Facebook about 33x, affiliate 0x (no detectable effect, not a fabricated negative one).
- MMM naive model before the fix implied a roughly negative 620x ROI for one channel, an economically impossible number that got caught before it could be reported as a real finding.

## What I Would Do Next

- Get a real install cohort with a fixed observation window (like day 7 behavior predicting day 90 spend) instead of a cross sectional snapshot, since that's the single biggest limitation of the pLTV model as built.
- Run the Cookie Cats style test again with an actual "reached the gate" flag instead of using total rounds played as a rough proxy, to get a real complier average effect instead of an intent to treat estimate diluted by players who were never exposed.
- Replace the MMM's assumed CPMs with real platform billing data from Meta, Google, and Apple Search Ads, and validate the model on a holdout period before trusting the ROI numbers enough to move budget.
- Pool the MMM across multiple markets or divisions with a proper hierarchical model instead of picking a single one, and move to a log linear specification to fix the negative implied baseline that shows up in the current linear version.

## AI Workflow

I used Claude Code for project setup, debugging, and speeding up repetitive code like boilerplate pandas and chart formatting. Every analytical decision in this project, the Tweedie objective, catching the leakage columns, choosing Spearman over R squared, the non-negativity constraint that fixed the MMM's broken ROI number, and the go/no-go call on the A/B test, was made and owned by me. Claude Code sped up the typing, it didn't do the thinking.

## Stack

- **Data / Analysis**: pandas, numpy, scipy, statsmodels, scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter
- **AI**: Anthropic (Claude)
