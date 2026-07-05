"""
Reproduces the exact modeling logic already verified in notebooks 01, 02, and 03,
and writes three clean, small CSVs to bq_exports/ for upload to BigQuery.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.optimize import lsq_linear
import statsmodels.api as sm

RANDOM_STATE = 42

# =============================================================================
# 1. pltv_predictions.csv — notebook 01's test set, actual vs. predicted LTV
# =============================================================================

df = pd.read_excel('data/mobile_game_inapp_purchases.xlsx')
df['LTV'] = df['InAppPurchaseAmount'].fillna(0.0)
df['is_payer'] = (df['LTV'] > 0).astype(int)

feature_cols = ['Age', 'Gender', 'Country', 'Device', 'GameGenre', 'SessionCount', 'AverageSessionLength']
X = df[feature_cols].copy()
y = df['LTV'].copy()

X_train, X_test, y_train, y_test, payer_train, payer_test = train_test_split(
    X, y, df['is_payer'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['is_payer'],
)

age_median = X_train['Age'].median()

def preprocess(X_part, age_median):
    X_out = X_part.copy()
    X_out['Age'] = X_out['Age'].fillna(age_median)
    for c in ['Gender', 'Country', 'Device', 'GameGenre']:
        X_out[c] = X_out[c].fillna('Missing').astype('category')
    return X_out

X_train_p = preprocess(X_train, age_median)
X_test_p = preprocess(X_test, age_median)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_p, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=payer_train,
)

tweedie_model = xgb.XGBRegressor(
    objective='reg:tweedie', tweedie_variance_power=1.5, n_estimators=500, max_depth=3,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    enable_categorical=True, early_stopping_rounds=30, eval_metric='rmse', random_state=RANDOM_STATE,
)
tweedie_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

pred_tweedie = tweedie_model.predict(X_test_p)

pltv_export = pd.DataFrame({
    'user_id': df.loc[X_test.index, 'UserID'].values,
    'actual_ltv': y_test.values.round(2),
    'predicted_ltv': pred_tweedie.round(2),
})
pltv_export['residual'] = (pltv_export['predicted_ltv'] - pltv_export['actual_ltv']).round(2)
pltv_export['is_payer'] = (pltv_export['actual_ltv'] > 0).astype(int)
pltv_export['predicted_decile'] = (
    pd.qcut(pltv_export['predicted_ltv'].rank(method='first'), 10, labels=False) + 1
)

pltv_export.to_csv('bq_exports/pltv_predictions.csv', index=False)
print('pltv_predictions.csv:', pltv_export.shape)
print(pltv_export.head(3))
print()

# =============================================================================
# 2. ab_test_summary.csv — notebook 02's retention rates by group
# =============================================================================

cookie_cats = pd.read_csv('data/cookie_cats.csv')

def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return center - half_width, center + half_width

rows = []
for ver in ['gate_30', 'gate_40']:
    sub = cookie_cats[cookie_cats['version'] == ver]
    n = len(sub)
    x1 = sub['retention_1'].sum()
    x7 = sub['retention_7'].sum()
    ci1 = wilson_ci(x1, n)
    ci7 = wilson_ci(x7, n)
    rows.append({
        'version': ver,
        'n_users': n,
        'retention_1_rate': round(x1 / n, 4),
        'retention_1_ci_low': round(ci1[0], 4),
        'retention_1_ci_high': round(ci1[1], 4),
        'retention_7_rate': round(x7 / n, 4),
        'retention_7_ci_low': round(ci7[0], 4),
        'retention_7_ci_high': round(ci7[1], 4),
    })

ab_export = pd.DataFrame(rows)
ab_export.to_csv('bq_exports/ab_test_summary.csv', index=False)
print('ab_test_summary.csv:', ab_export.shape)
print(ab_export)
print()

# =============================================================================
# 3. mmm_channel_roi.csv — notebook 03's channel ROI summary
# =============================================================================

raw = pd.read_csv('data/Sample Media Spend Data.csv')
raw['date'] = pd.to_datetime(raw['Calendar_Week'])
division = raw[raw['Division'] == 'A'].sort_values('date').reset_index(drop=True)
division['t'] = np.arange(len(division))

CHANNELS = ['Google_Impressions', 'Facebook_Impressions', 'Email_Impressions', 'Affiliate_Impressions']

def adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] + decay * out[i - 1]
    return out

def saturate(x, alpha):
    return np.power(x, alpha)

def partial_corr_vs_trend(x, y, t):
    T = np.column_stack([np.ones_like(t), t])
    bx, *_ = np.linalg.lstsq(T, x, rcond=None)
    by, *_ = np.linalg.lstsq(T, y, rcond=None)
    return np.corrcoef(x - T @ bx, y - T @ by)[0, 1]

decays = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
t = division['t'].values.astype(float)
sales = division['Sales'].values.astype(float)

best_params = {}
for c in CHANNELS:
    x = division[c].values.astype(float)
    best_abs, best_pair = -1.0, None
    for d in decays:
        ad = adstock(x, d)
        for al in alphas:
            r = partial_corr_vs_trend(saturate(ad, al), sales, t)
            if not np.isnan(r) and abs(r) > best_abs:
                best_abs, best_pair = abs(r), (d, al)
    best_params[c] = best_pair

transformed = {}
for c in CHANNELS:
    d, alpha = best_params[c]
    transformed[c] = saturate(adstock(division[c].values.astype(float), d), alpha)

X_mmm = pd.DataFrame({'t': division['t'].astype(float)})
for c in CHANNELS:
    X_mmm[c] = transformed[c]
month_dummies = pd.get_dummies(division['date'].dt.month, prefix='month', drop_first=True).astype(float)
X_full = sm.add_constant(pd.concat([X_mmm, month_dummies], axis=1))
y_mmm = division['Sales'].astype(float)

cols = list(X_full.columns)
lower_bounds = np.full(len(cols), -np.inf)
upper_bounds = np.full(len(cols), np.inf)
for c in CHANNELS:
    lower_bounds[cols.index(c)] = 0.0

bounded_fit = lsq_linear(X_full.values, y_mmm.values, bounds=(lower_bounds, upper_bounds), method='bvls')
coefs = pd.Series(bounded_fit.x, index=cols)
contributions = {c: coefs[c] * X_mmm[c].values for c in CHANNELS}

ILLUSTRATIVE_CPM = {
    'Google_Impressions': 8.00,
    'Facebook_Impressions': 7.00,
    'Email_Impressions': 0.10,
    'Affiliate_Impressions': 15.00,
}

roi_rows = []
for c in CHANNELS:
    d, alpha = best_params[c]
    impressions = division[c].sum()
    spend = impressions / 1000 * ILLUSTRATIVE_CPM[c]
    contribution = contributions[c].sum()
    roi = contribution / spend if spend > 0 else np.nan
    roi_rows.append({
        'channel': c.replace('_Impressions', ''),
        'adstock_decay': d,
        'saturation_alpha': alpha,
        'impressions': int(impressions),
        'assumed_spend_usd': round(spend, 2),
        'contribution_usd': round(contribution, 2),
        'roi': round(roi, 2),
    })

mmm_export = pd.DataFrame(roi_rows)
mmm_export.to_csv('bq_exports/mmm_channel_roi.csv', index=False)
print('mmm_channel_roi.csv:', mmm_export.shape)
print(mmm_export)
