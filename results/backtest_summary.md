# Backtest on archived Kalshi prices

Markets: 3,216 contracts over 67 days (2026-07-07 → 2026-09-11), 96,480 contract-hours, 52,412 with a two-sided book (spread ≤ 0.20).

## Information: who knows more, the model or the market?

| forecaster | Brier | log-loss | n |
| :-- | --: | --: | --: |
| model | 0.1407 | 0.4335 | 52,412 |
| market | 0.1204 | 0.3716 | 52,412 |
| blend | 0.1254 | 0.3870 | 52,412 |

## P&L of the simple rule

One contract per signal, first signal per contract-day, held to settlement, taker fees.

| margin | trades | hit rate | avg price | P&L total | P&L / trade | 95% CI (total) | daily Sharpe |
| --: | --: | --: | --: | --: | --: | :--: | --: |
| 0.02 | 2410 | 0.389 | 0.40 | $-68.02 | $-0.028 | [-109.7, -27.1] | -7.71 |
| 0.05 | 2177 | 0.382 | 0.39 | $-52.45 | $-0.024 | [-94.3, -13.0] | -6.03 |
| 0.10 | 1839 | 0.378 | 0.37 | $-15.52 | $-0.008 | [-55.9, +23.7] | -1.79 |

## P&L by station (margin 0.05)

| station | trades | P&L total | P&L / trade |
| :-- | --: | --: | --: |
| KAUS | 268 | $-16.32 | $-0.061 |
| KDEN | 283 | $+2.17 | $+0.008 |
| KHOU | 263 | $-11.88 | $-0.045 |
| KLAX | 265 | $+2.39 | $+0.009 |
| KMDW | 300 | $-1.48 | $-0.005 |
| KMIA | 245 | $+5.42 | $+0.022 |
| KNYC | 263 | $-24.39 | $-0.093 |
| KPHL | 290 | $-8.36 | $-0.029 |