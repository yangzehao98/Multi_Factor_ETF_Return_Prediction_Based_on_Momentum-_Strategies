# NYU Courant Mathematics in Finance Capstone Project

## Sector ETF Returns Prediction and Factor Momentum

##### Contributors: Soo Han Kim, Ju Hyung Kang, Zehao Yang

### (1) Data Obtaining and Cleaning

- Original source for 11 SPDR Sector ETF prices and Fama-French returns history were obtained from Bloomberg
- Macro data of treasury bonds and gold were retrieved from yfinance
- data_cleaning.ipynb: cross checks Sector ETF prices with yahoo finance and fill in for missing dates

### (2) Data Characteristics Research

- etf_ff_corrs.ipynb: investigates correlations of lagged daily & monthly ETF and FF returns with future daily & monthly ETF returns

### (3) Prediction Models

- etf_rets_pred_A_B.ipynb: crafts momentum factor portfolios from lagged daily ETF and FF returns and supplies the portfolio returns to fit OLS with Ridge regularization on future daily ETF returns

- etf_rets_pred_A_B_C.ipynb: investigates daily & monthly lagged correlations after removing market from ETF returns for ETF, FF, and macro returns

### (4) Backtests

- All backtests are conducted on models using features (X) & targets (y) created from ETF residual returns (after subtracting market effects)

- Beta on backtesting period were calculated on an expanding window basis, up until 2 days earlier

- daily_results.ipynb: backtest results based on predictions with lagged daily returns OLS

- etf_pf_backtests.ipynb: backtest results based on predictions with lagged monthly cumulative returns (as in Falck et al.); tried equal-weight, weighting proportionately with predictions, and equal-vol targeting

### (5) Results

#### (1) Correlations

- Daily Lagged Returns Correlations Example

![Daily Corelations Example](results/daily_correlations.png)

- Monthly Lagged Returns Correlations Example

![Monthly Correlations Example](results/monthly_correlations.png)

- Monthly Lagged Cumulative Returns Statistical Significance Example

![Monthly Cumulative Correlation Significance Example](results/monthly_cumulative_corr_significance.png)

In the above chart, values are the maximum of the correlation divided by statistical significance threshold at 95% confidence and 1.

#### (2) Fitting

- Return period definition

We used daily returns until T-1 close for predictions conducted at timestep T. For monthly returns, resampling from daily returns were done in a way that respects this restriction. Beta calculations for
forming market-neutral portfolios on test set were done by calculating beta with returns up to T-1 close.

- Feature Construction

For daily lagged returns, we used lags for auto & cross-correlations that were statistically significant. For lagged monthly cumulative returns, we used (m, n) pairs, i.e. the lag and holding period, that were the highest for auto & cross-correlations among the statistically significant pairs per asset.

- In-sample Results (using Lagged Monthly Cumulative Returns features)

| ETF | XLC | XLRE | XLY | XLP |  XLE |   XLF |  XLV  |   XLI  |  XLB | XLK | XLU |
|-----|-----| ---- | ---- | ---- | ---| ------| ------| -------| -----| ----| --- |
| R2 | 0.822| 0.402 | 0.177 | 0.082| 0.155| 0.203| 0.149| 0.191| 0.160 | 0.213| 0.139|
| Return-sign Accuracy (in-sample) | 0.694 | 0.783 | 0.655 | 0.624 | 0.608 | 0.590 | 0.632 | 0.635 | 0.592 | 0.568 | 0.641 |
| Return-sign Accuracy (out-of-sample) | 0.571 | 0.857 | 0.857 | 0.571 | 0.714 | 0.429 | 0.571 | 0.857 | 0.429 | 0.714 | 0.571 |

- Backtesting Out-of-sample

We used model predicted returns to form weights for the ETFs and then market-neutralized the portfolio by taking appropriate positions in SPY through expanding window beta calculation. Equal weighting only utilized the sign of the return predictions while prediction-proportionate weighting scaled these weights by the magnitude of the return predictions as well.

#### (3) Backtest Results on Test Data

- Summary Table

| Features   | Model | Weighting  | Sharpe |
|-----------|-----|----------| ---------- |
| Daily     | OLS  |   Equal       | 0.58 |
| Monthly Cumulative       |  OLS   | Equal   | 3.01 |
| Monthly Cumulative       |  OLS   | Equal-vol   | 2.83 |
| Monthly Cumulative       |  OLS   | Prediction-proportionate   | 3.09 |

S&P 500 achieved a Sharpe of 2.3 on the same period.

- Equity Curves

![Equal-weighted Constituents' Cumulative Returns](results/ew_cum_rets_constituents_equity_curve.png)

![Equal-weighted Cumulative Returns and SP500](results/ew_cum_rets_vs_sp.png)