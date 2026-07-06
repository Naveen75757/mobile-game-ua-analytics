"""
Generates a simulated weekly UA KPI table (12 weeks x 3 channels) for the
Looker Studio dashboard. This is synthetic data, not derived from any notebook
analysis -- unlike the other bq_exports/ CSVs, which reproduce real notebook results.
"""
import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_WEEKS = 12
WEEK_START = pd.Timestamp('2026-04-06')  # a Monday
weeks = [WEEK_START + pd.Timedelta(weeks=i) for i in range(N_WEEKS)]

# Per-channel baseline assumptions -- kept plausible and distinct per channel
# (Apple Search Ads: lower volume, higher intent, better economics; Meta: high
# volume, broad reach, thinner margins; Google UAC: in between).
CHANNEL_PROFILES = {
    'Meta': {
        'installs_range': (3000, 8000),
        'cac_range': (3.0, 6.5),
        'roas_range': (0.8, 1.8),
        'd1_range': (0.33, 0.42),
        'd7_range': (0.15, 0.20),
    },
    'Google': {
        'installs_range': (2000, 6000),
        'cac_range': (2.5, 5.5),
        'roas_range': (1.0, 2.0),
        'd1_range': (0.36, 0.45),
        'd7_range': (0.17, 0.22),
    },
    'Apple Search Ads': {
        'installs_range': (800, 2500),
        'cac_range': (4.0, 8.0),
        'roas_range': (1.5, 2.5),
        'd1_range': (0.40, 0.50),
        'd7_range': (0.20, 0.25),
    },
}

rows = []
for week in weeks:
    for channel, p in CHANNEL_PROFILES.items():
        installs = int(rng.integers(*p['installs_range']))
        cac = round(float(rng.uniform(*p['cac_range'])), 2)
        spend_usd = round(installs * cac, 2)

        roas = round(float(rng.uniform(*p['roas_range'])), 2)
        revenue_usd = round(spend_usd * roas, 2)

        # Recompute CAC/ROAS from the rounded dollar columns so they're exact
        # derived values (spend/installs, revenue/spend), not independently
        # drawn numbers that only approximately match.
        cac_exact = round(spend_usd / installs, 2)
        roas_exact = round(revenue_usd / spend_usd, 2)

        d1_retention = round(float(rng.uniform(*p['d1_range'])), 3)
        d7_retention = round(float(rng.uniform(*p['d7_range'])), 3)
        d7_retention = min(d7_retention, d1_retention - 0.05)  # d7 must stay below d1

        # 30-day LTV per install: tied to this week's revenue-per-install
        # economics (so channels with strong ROAS also show strong LTV), plus
        # a multiplier for continued monetization beyond the attribution week.
        revenue_per_install = revenue_usd / installs
        ltv_30d = round(revenue_per_install * float(rng.uniform(1.15, 1.45)), 2)

        rows.append({
            'week': week.date().isoformat(),
            'channel': channel,
            'installs': installs,
            'spend_usd': spend_usd,
            'revenue_usd': revenue_usd,
            'cac': cac_exact,
            'roas': roas_exact,
            'd1_retention': d1_retention,
            'd7_retention': d7_retention,
            'ltv_30d': ltv_30d,
        })

kpi_export = pd.DataFrame(rows)
kpi_export.to_csv('bq_exports/kpi_summary.csv', index=False)

print('kpi_summary.csv:', kpi_export.shape)
print(kpi_export.head(6))
print()
print('Sanity check -- ranges:')
print('CAC:', kpi_export['cac'].min(), '-', kpi_export['cac'].max())
print('ROAS:', kpi_export['roas'].min(), '-', kpi_export['roas'].max())
print('D7 retention:', kpi_export['d7_retention'].min(), '-', kpi_export['d7_retention'].max())
print('D1 >= D7 always:', (kpi_export['d1_retention'] >= kpi_export['d7_retention']).all())
