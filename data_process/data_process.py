import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import numpy as np
from utils.utils import *

def get_etf_returns():
    """
    Retrieve ETF price data and compute daily returns.

    This function reads ETF price data from a CSV file, calculates the daily percentage returns,
    converts the index to pandas datetime format, and returns the resulting DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing daily returns of ETFs. The index is of datetime type, and
        each column represents an ETF identified by its ticker symbol.

    Notes
    -----
    - The function expects the ETF price data to be located at '../data/etf_prices/etf_prices.csv'.
    - The CSV file should have a 'Date' column to be used as the index.
    - Returns are calculated as the percentage change between consecutive trading days.
    """
    # Read ETF price data and change them into returns
    etf_rets = pd.read_csv('../data/etf_prices/etf_prices.csv', index_col='Date').pct_change()

    # Change index into pandas datetime
    etf_rets.index = pd.to_datetime(etf_rets.index)

    return etf_rets

def get_fama_french_returns(start_date="1998-12-22", end_date="2024-12-01"):
    """
    Retrieve and process Fama-French and additional factor returns.

    This function fetches daily Fama-French 5 factors, Momentum factor,
    Short-Term Reversal factor, and Long-Term Reversal factor from the Fama-French
    data library. It processes and concatenates these factors into a single DataFrame
    of returns.

    Parameters
    ----------
    start_date : str or datetime, optional
        The start date for fetching the factor data. Default is "1998-12-22".
    end_date : str or datetime, optional
        The end date for fetching the factor data. Default is "2024-12-01".

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing daily returns for the Fama-French 5 factors,
        Momentum factor ('UMD'), Short-Term Reversal factor ('ST_Reversal'),
        and Long-Term Reversal factor ('LT_Reversal'). The index is of datetime type,
        and each column represents a specific factor.

    Notes
    -----
    - The function retrieves data using `pandas_datareader`'s `DataReader` from the 'famafrench' source.
    - The 'Mkt-RF' (Market minus Risk-Free) and 'RF' (Risk-Free rate) columns are excluded from the
      Fama-French 5 factors.
    - All factor returns are converted from percentages to decimal form by dividing by 100.
    - Missing data is dropped to ensure a clean dataset.
    """
    # Retrieve the Fama-French 5 factors
    ff5_data = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start_date, end_date)
    umd_data = web.DataReader("F-F_Momentum_Factor_daily", "famafrench", start_date, end_date)
    st_reversal_data = web.DataReader("F-F_ST_Reversal_Factor_daily", "famafrench", start_date, end_date)
    lt_reversal_data = web.DataReader("F-F_LT_Reversal_Factor_daily", "famafrench", start_date, end_date)

    # Get returns
    ff5_rets = ff5_data[0].drop(columns=['Mkt-RF', 'RF'])
    umd_rets = umd_data[0]
    st_reversal_rets = st_reversal_data[0]
    lt_reversal_rets = lt_reversal_data[0]

    # Concat returns
    ff_rets = pd.concat([ff5_rets, umd_rets, st_reversal_rets, lt_reversal_rets], axis=1)

    # Drop na
    ff_rets.dropna(inplace=True)

    # Clean column names
    ff_rets.columns = [column.strip() for column in ff_rets.columns]

    return ff_rets / 100 # Divide by 100 since the unit is percentages

def get_macro_returns(start_date="1998-12-22", end_date="2024-12-01"):
    """
    Retrieve and compute daily returns for selected macroeconomic indicators.

    This function downloads daily closing prices for various macroeconomic indicators,
    including the S&P 500, Gold Futures, and Treasury Yield indices. It then calculates
    the daily percentage returns for each indicator.

    Parameters
    ----------
    start_date : str or datetime, optional
        The start date for fetching the macroeconomic data. Default is "1998-12-22".
    end_date : str or datetime, optional
        The end date for fetching the macroeconomic data. Default is "2024-12-01".

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing daily returns for the following macroeconomic indicators:
        - 'SP500': S&P 500 (ticker: "SPY")
        - 'Gold': Gold Futures (ticker: "GC=F")
        - '3yr_Treasury': 3-Month Treasury Yield Index (ticker: "^IRX")
        - '10yr_Treasury': 10-Year Treasury Yield Index (ticker: "^TNX")
        - '30yr_Treasury': 30-Year Treasury Yield Index (ticker: "^TYX")

        The index is of datetime type, and each column represents a specific macroeconomic indicator.

    Notes
    -----
    - The function uses the `yfinance` library to download data for each ticker.
    - If data retrieval for a ticker fails, an error message is printed, and the ticker is skipped.
    - Only the 'Close' price is used to compute daily returns.
    - Missing data is dropped to ensure a complete dataset of returns.
    """
    # Macro tickers to download
    tickers = {
        'SP500': "SPY",
        "Gold": "GC=F",  # Gold Futures
        "3yr_Treasury": "^IRX",  # 3-Month Treasury Yield Index (Approximating short-term rates)
        "10yr_Treasury": "^TNX",  # 10-Year Treasury Yield Index
        "30yr_Treasury": "^TYX",  # 30-Year Treasury Yield Index
    }

    # Retrieve data for all tickers
    macro_data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            macro_data[name] = df
            print(f"Successfully retrieved data for {name}")
        except Exception as e:
            print(f"Failed to retrieve data for {name}: {e}")

    # Get only close data from the retrieved data
    macro_dfs = []
    for name, df in macro_data.items():
        macro_dfs.append(df['Close'])
    macro_df = pd.concat(macro_dfs, axis=1)
    macro_df.columns = list(macro_data.keys())

    # Get returns from the close data
    macro_rets = macro_df.pct_change()

    # Drop na
    macro_rets.dropna(inplace=True)

    return macro_rets

def get_return_data():
    """
    Aggregate ETF, Fama-French, and macroeconomic returns into a unified DataFrame.

    This function combines daily returns from ETFs, Fama-French factors, and selected
    macroeconomic indicators into a single DataFrame. It merges the datasets on their
    datetime indices to ensure alignment of return periods.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing daily returns from:
        - ETFs (columns starting with 'X')
        - Fama-French factors and additional factors
        - Macroeconomic indicators

        The index is of datetime type, representing each trading day.

    Notes
    -----
    - The function internally calls `get_etf_returns()`, `get_fama_french_returns()`,
      and `get_macro_returns()` to obtain the respective return datasets.
    - Merging is performed using a left join based on the date index to ensure that all
      ETF returns are retained.
    - Missing values resulting from the merge are preserved as NaNs.
    """
    # Get ETF returns
    etf_rets = get_etf_returns()

    # Get Fama French returns
    fama_french_returns = get_fama_french_returns()

    # Get Macro returns
    macro_rets = get_macro_returns()

    # Merge returns
    rets = pd.merge(etf_rets, fama_french_returns, how='left', left_index=True, right_index=True)
    rets = pd.merge(rets, macro_rets, how='left', left_index=True, right_index=True)

    return rets

def get_beta(rets, period=252*2, exponential_weight=True):
    """
    Calculate rolling beta coefficients for ETFs relative to the S&P 500.

    This function computes the beta coefficients for each ETF in the provided returns DataFrame
    using a rolling window approach. Beta is calculated as the covariance between ETF returns and
    S&P 500 returns divided by the variance of S&P 500 returns. Optionally, exponential weighting
    can be applied to the returns within each rolling window.

    Parameters
    ----------
    rets : pandas.DataFrame
        A DataFrame containing daily returns for ETFs and the S&P 500.
        - ETF columns should start with 'X'.
        - Must include a column named 'SP500' representing the market benchmark.
    period : int, optional
        The number of trading days to include in each rolling window for beta calculation.
        Default is 504 (approximately two trading years).
    exponential_weight : bool, optional
        If True, applies exponential weighting to the returns within each rolling window.
        If False, uses equal weighting. Default is True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the rolling beta coefficients for each ETF. The index matches
        the input `rets` DataFrame's index, and each column corresponds to an ETF.
        Beta values are NaN for dates where insufficient data is available.

    Notes
    -----
    - The function shifts the returns by one day to avoid look-ahead bias.
    - Rows with more than two missing values across ETFs are excluded from beta calculations.
    - If exponential weighting is enabled, the `exp_weight` function is applied to the returns.
      This function must be defined elsewhere in the codebase.
    - The computed beta coefficients are saved to '../data/etf_prices/beta.csv'.
    - The function handles missing data by dropping NaNs within each rolling window before computation.
    - Beta is only calculated if the number of non-NaN observations within the window equals the specified period.
    """
    # Get the length of available data points
    bias_free_rets = rets.shift(1).copy()
    bias_free_rets = bias_free_rets[bias_free_rets.isna().sum(axis=1) < 3]

    # Get ETF columns
    etfs = [col for col in rets.columns if col.startswith('X')]

    # Set beta
    beta = bias_free_rets.loc[:, etfs].copy()
    for i, (idx, row) in enumerate(bias_free_rets.iterrows()):
        if i < period:
            beta.loc[idx, etfs] = np.nan
        else:
            beta_series = pd.Series(index=etfs, data=np.nan)
            for etf in etfs:
                # Get index number of the date index
                idx_num = bias_free_rets.index.get_loc(idx)

                # Set temporary df
                rets_temp = bias_free_rets.iloc[idx_num - period:idx_num].loc[:, [etf, 'SP500']].dropna()

                # Skip if not enough data
                if len(rets_temp) < period:
                    continue

                # Set ETF returns
                etf_ret = rets_temp[etf]
                if exponential_weight:
                    etf_ret = exp_weight(etf_ret)

                # Set market returns and variance
                market_ret = rets_temp['SP500']
                if exponential_weight:
                    market_ret = exp_weight(market_ret)
                market_var = (market_ret ** 2).sum()

                # Compute ETF/market covariance
                etf_mkt_cov = (market_ret * etf_ret).sum()

                # Compute beta
                beta_val = etf_mkt_cov / market_var

                # Eliminate market effect
                beta_series.loc[etf] = beta_val

            # Save beta
            beta.loc[idx, beta_series.index] = beta_series

    # Save beta as csv
    beta.to_csv('../data/etf_prices/beta.csv')

    return beta

def get_cov(rets, idx, period=21*6):
    # Get the length of available data points
    bias_free_rets = rets.shift(1).copy().dropna(axis=0, how='all')

    # Get ETF columns
    etfs = [col for col in rets.columns if col.startswith('X')]

    # Get rid of ETF that does not have full data
    rets_trunc = bias_free_rets.loc[:idx, etfs]
    if len(rets_trunc) > period:
        rets_trunc = rets_trunc.iloc[-period:]
    else:
        print("Not enough data points for covariance calculation.")
        return None
    rets_trunc.dropna(axis=1, inplace=True)

    covariance = rets_trunc.cov()

    def is_positive_definite(matrix):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
        except:
            print(idx)
            print(rets_trunc)
            print(covariance)
            return False
        return np.all(eigenvalues > 0)

    if not is_positive_definite(covariance):
        print("Covariance matrix is not positive definite.")
        return None

    return covariance


if __name__ == '__main__':
    returns = get_return_data()
    print(returns.head())
    returns.to_csv('../data/returns/returns.csv')

    # Read returns
    rets = pd.read_csv('../data/returns/returns.csv', index_col='Date')
    rets.index = pd.to_datetime(rets.index)

    get_beta(rets)

    cov = get_cov(rets, pd.to_datetime('2005-05-24'))

    print(cov)