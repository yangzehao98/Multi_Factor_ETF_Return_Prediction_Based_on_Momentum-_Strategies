import pandas as pd
import pickle
import numpy as np
# from docutils.nodes import target
# from networkx import total_spanning_tree_weight
from model.feature_engineering import *
from model.fit import *
import copy
from utils.utils import *
import matplotlib.pyplot as plt
from evaluation.evaluation import *
from data_process.data_process import *
from scipy.optimize import minimize
from tqdm import tqdm

def rolling_train_weights(rets, betas, start_date='2024-08-01', significance_level=0.05, shift=0, rebalance=21):
    # Initiate market weight
    market_weights = pd.Series(name='SP500')

    weights_matrix = rets.loc[betas.index, betas.columns]
    weights_matrix = weights_matrix.loc[start_date:]

    total_features = rets.columns

    # Allocate equal weight
    for i, (idx, row) in tqdm(enumerate(weights_matrix.iterrows()), total=len(weights_matrix), desc="Calculating weights"):
        if i % rebalance == shift:
            etfs = [col for col in total_features if col.startswith('X')]
            feature_set = etfs + ['SP500', 'RMW', 'Mom', 'ST_Rev', '3yr_Treasury']

            data = construct_features(rets.loc[:, feature_set], betas, idx - pd.DateOffset(days=1), significance_level=significance_level,save=False, plot=False)

            temp_data = copy.deepcopy(data)

            feature_set.remove('SP500')

            for etf in etfs:
                all_columns = temp_data[etf]['train'].columns
                filtered_columns = [item for item in all_columns if item.startswith(tuple(feature_set))]
                temp_train = temp_data[etf]['train'].loc[:, filtered_columns].copy()
                temp_data[etf]['train'] = temp_train
                temp_test = temp_data[etf]['test'].loc[:, filtered_columns].copy()
                temp_data[etf]['test'] = temp_test

            fit_result = fit(temp_data, save=False)

            etfs = weights_matrix.columns

            # Set test data
            test = pd.concat([fit_result[etf]['test'][f'{etf}_pred'] for etf in etfs], axis=1)
            test.columns = etfs

            if test.empty:
                row.loc[:] = np.nan
                print(f"Test set is empty for {idx}.")
                continue

            etf_predicted_rets = test.iloc[0]

            # Get betas
            beta = betas.loc[idx, weights_matrix.columns]

            # Skip if beta data is not available
            if beta.isna().sum() > 2:
                row.loc[:] = 0
                continue

            # Drop na
            beta.dropna(inplace=True)

            # Get the direction of the predicted returns
            row.loc[beta.index] = np.sign(etf_predicted_rets.loc[beta.index])

            # Calculate market weight
            market_weight = (-beta * etf_predicted_rets.loc[beta.index]).sum()

            # Make total absolute weight equals 1
            feature_weight = np.abs(row).sum()
            total_weight = feature_weight + np.abs(market_weight)
            if total_weight == 0:
                continue
            else:
                row.loc[:] = row / total_weight
                market_weight /= total_weight

            # Add market weight
            market_weights.loc[idx] = market_weight
        else:
            row.loc[:] = np.nan

    # Merge market weights into returns df
    ptf_weights = pd.concat([weights_matrix, market_weights], axis=1)

    # Fill na positions
    ptf_weights.ffill(inplace=True, axis=0)
    ptf_weights.fillna(0, inplace=True)

    return ptf_weights


def equal_weight_ptf(etf_predicted_rets, betas, shift=0, rebalance=21):
    """
    Allocate portfolio weights based on equal weighting of predicted ETF returns and corresponding betas.

    This function assigns portfolio weights to ETFs and the market (SP500) based on predicted returns
    and their beta coefficients. The allocation is performed periodically (every 21 trading days,
    typically representing one month) with an optional shift. For each allocation period:

    - Predicted returns are converted to directional signals using the sign function.
    - Weights are calculated by multiplying the betas with these directional signals.
    - The total absolute weight is normalized to sum to 1, ensuring a fully invested portfolio.
    - A separate weight for the market (SP500) is also calculated and included in the portfolio.

    Parameters
    ----------
    etf_predicted_rets : pandas.DataFrame
        A DataFrame containing predicted returns for each ETF. The index represents dates, and
        each column corresponds to an ETF ticker symbol.

    betas : pandas.DataFrame
        A DataFrame containing beta coefficients for each ETF relative to the market (SP500).
        The index should align with `etf_predicted_rets`, and columns correspond to ETF ticker symbols.

    shift : int, optional
        An integer representing the shift in allocation periods. This determines the specific day
        within each 21-day window when weights are recalculated. For example, a `shift` of 0
        recalculates weights on the first day of each window. Default is 0.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the allocated portfolio weights for each ETF and the market (SP500).
        The index matches `etf_predicted_rets`, and columns include all ETFs plus 'SP500'.
        Weights are forward-filled to maintain consistency between allocation periods.

    Notes
    -----
    - The function allocates weights only on days that satisfy the condition `i % 21 == shift`,
      where `i` is the row index. Other days will have NaN weights initially, which are later
      forward-filled.
    - If an ETF has more than two missing beta values (`NaN`), its weight is set to 0 for that
      allocation period.
    - The market weight ('SP500') is calculated separately and included as an additional column
      in the returned DataFrame.
    - After allocation, all weights are normalized so that the sum of absolute weights equals 1.
      This ensures a fully invested portfolio without leverage.
    - The function assumes that the `betas` DataFrame includes a 'SP500' column representing the
      market benchmark.
    - The input DataFrames (`etf_predicted_rets` and `betas`) must have aligned indices for
      accurate weight allocation.
    """

    # Initiate market weight
    market_weights = pd.Series(name='SP500')

    # Allocate equal weight
    for i, (idx, row) in enumerate(etf_predicted_rets.iterrows()):
        if i % rebalance == shift:
            # Get betas
            beta = betas.loc[idx, etf_predicted_rets.columns]

            # Skip if beta data is not available
            if beta.isna().sum() > 2:
                row.loc[:] = 0
                continue

            # Drop na
            beta.dropna(inplace=True)

            # Get the direction of the predicted returns
            row.loc[beta.index] = np.sign(row.loc[beta.index])

            # Calculate market weight
            market_weight = (-beta * row.loc[beta.index]).sum()

            # Make total absolute weight equals 1
            feature_weight = np.abs(row).sum()
            total_weight = feature_weight + np.abs(market_weight)
            if total_weight == 0:
                continue
            else:
                row.loc[:] = row / total_weight
                market_weight /= total_weight

            # Add market weight
            market_weights.loc[idx] = market_weight
        else:
            row.loc[:] = np.nan

    # Merge market weights into returns df
    ptf_weights = pd.concat([etf_predicted_rets, market_weights], axis=1)

    # Fill na positions
    ptf_weights.ffill(inplace=True, axis=0)
    ptf_weights.fillna(0, inplace=True)

    return ptf_weights


def optimized_weight_ptf(etf_predicted_rets,
                         marketless_rets,
                         betas,
                         shift=0,
                         rebalance=21,
                         period=21*6,
                         mode='sharpe'):
    # Initiate market weight
    market_weights = pd.Series(name='SP500')

    # Allocate equal weight
    for i, (idx, row) in enumerate(etf_predicted_rets.iterrows()):
        if i % rebalance == shift:
            # Get betas
            beta = betas.loc[idx, etf_predicted_rets.columns]

            # Skip if beta data is not available
            if beta.isna().sum() > 2:
                row.loc[:] = 0
                continue

            # Drop na
            beta.dropna(inplace=True)

            # Get covariance matrix
            cov_matrix = get_cov(marketless_rets, idx, period=period)

            # Skip if covariance matrix is not available
            if cov_matrix is None:
                print("Covariance matrix for ETF {} not calculated".format(idx))
                row.loc[:] = 0
                continue

            num_assets = len(cov_matrix)

            # Get the direction of the predicted returns
            expected_returns = row.loc[cov_matrix.index]

            # Get initial weight
            init_weights = np.sign(expected_returns) / num_assets

            # Sharpe Ratio objective function (negative for minimization)
            def objective_func(weights, expected_returns, cov_matrix, mode=mode):
                # Check sanity
                if len(weights) == 0 or len(expected_returns) == 0 or cov_matrix.size == 0:
                    raise ValueError("Input arrays are empty.")

                # Portfolio return (Use log transformed returns)
                portfolio_return = np.dot(weights, np.log1p(expected_returns))

                # Portfolio volatility
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

                # Sharpe ratio
                sharpe_ratio = portfolio_return - portfolio_variance

                # Mode
                if mode == "sharpe":
                    return -sharpe_ratio  # Negate to maximize Sharpe Ratio
                elif mode == "min_variance":
                    return portfolio_variance

            # Constraints: absolute weights sum to 1
            target_return = 0.002
            constraints = [
                # Cash Neutral
                # {'type': 'eq', 'fun': lambda w: np.sum(w) + np.sum(w * beta[cov_matrix.index])},
                # Exposure = 1
                {'type': 'eq', 'fun': lambda w: np.sum(abs(w)) + np.sum(abs(w * beta[cov_matrix.index])) - 1},
                # Portfolio return should be higher than target return
                {'type': 'ineq', 'fun': lambda w: np.dot(w, expected_returns) - target_return},
            ]

            # Bounds
            bounds = [(-1, 1) for _ in range(num_assets)]

            # Optimize the portfolio for maximum Sharpe Ratio
            result = minimize(objective_func, init_weights, args=(expected_returns, cov_matrix, mode),
                              method='SLSQP', bounds=bounds, constraints=constraints)

            # Extract optimized weights
            optimized_weights = result.x

            # Set optimized weights
            row.loc[cov_matrix.index] = optimized_weights

            # Calculate market weight
            market_weight = (-beta[cov_matrix.index] * row.loc[cov_matrix.index]).sum()

            # Add market weight
            market_weights.loc[idx] = market_weight
        else:
            row.loc[:] = np.nan

    # Merge market weights into returns df
    ptf_weights = pd.concat([etf_predicted_rets, market_weights], axis=1)

    # Fill na positions
    ptf_weights.ffill(inplace=True, axis=0)
    ptf_weights.fillna(0, inplace=True)

    return ptf_weights


def get_ptf_returns(weights, rets, trading_cost=0.02):
    """
    Calculate portfolio returns based on allocated weights and asset returns.

    This function computes the daily returns of a portfolio by applying the provided weights to
    the corresponding asset returns. It handles synchronization of dates between weights and
    returns, ensures that only valid trading days are considered, and incorporates market
    returns for comparison.

    Parameters
    ----------
    weights : pandas.DataFrame
        A DataFrame containing portfolio weights for each asset and the market ('SP500').
        The index represents dates, and columns correspond to asset ticker symbols plus 'SP500'.
        Weights should be aligned with the `rets` DataFrame's index.

    rets : pandas.DataFrame
        A DataFrame containing daily returns for each asset and the market ('SP500').
        The index represents dates, and columns correspond to asset ticker symbols plus 'SP500'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the portfolio's daily returns and the market return ('market').
        The index represents dates, and columns include all assets, 'ptf' (portfolio return),
        and 'market'.

    Notes
    -----
    - The function identifies the last date where all asset weights are zero and starts
      calculations from the subsequent trading day to avoid periods with no investment.
    - Returns are shifted by one day (`rets.shift(-1)`) to align the portfolio weights
      with the next day's returns, simulating real-time investment decisions.
    - Missing weights are forward-filled to maintain consistent allocations over time.
    - The portfolio return ('ptf') is calculated as the sum of the product of weights and
      asset returns for each day.
    - The market return is extracted separately and merged into the final DataFrame for
      performance comparison.
    - Any remaining missing values in the portfolio returns are filled with zeros.
    """
    # Identify rows where all values are 0
    all_zero_mask = (weights == 0).all(axis=1)

    last_zero_index = all_zero_mask.index[0]
    # Find the index of the last all-zero row
    for idx, item in all_zero_mask.items():
        if item == False:
            break
        last_zero_index = idx

    # Remove days without trading
    weights = weights.loc[last_zero_index:]

    # Sync the returns
    rets_synced = rets.shift(-1)
    rets_used = rets_synced.loc[weights.index, weights.columns]

    # Match the index
    weights = weights.reindex(rets_used.index).ffill()

    # Calculate trading cost and turnover (proxy)
    trading_cost_matrix = weights.copy()

    trades_sum = 0
    for i, (idx, row) in enumerate(trading_cost_matrix.iterrows()):
        if i == 0:
            row[:] = weights.iloc[i] * trading_cost
        else:
            trades_sum += np.abs(weights.iloc[i] - weights.iloc[i - 1]).sum()
            row[:] = np.abs(weights.iloc[i] - weights.iloc[i - 1]) * trading_cost

    # Calculate turnover
    annualized_turnover = trades_sum / len(weights) * 252 / 2

    # Calculate the performance
    ptf_rets = rets_used * weights
    ptf_rets.fillna(0, inplace=True)

    # Subtract trading cost
    ptf_rets -= trading_cost_matrix

    # Calculate portfolio return
    ptf_rets['ptf'] = ptf_rets.sum(axis=1)

    # Add market return
    market_return = rets.shift(-1)['SP500']
    market_return.rename('market', inplace=True)
    ptf_rets = pd.merge(ptf_rets, market_return, how='left', left_index=True, right_index=True)
    ptf_rets.dropna(axis=0, inplace=True)

    return ptf_rets, annualized_turnover


def backtest(rets, betas, fit_result, weight_scheme='equal', shift=0, rebalance=21, period=21*12, features=['Mom', 'ST_Rev', '3yr_Treasury'], trading_cost=0.0009):
    """
    Perform backtesting of a portfolio strategy based on predicted ETF returns and allocated weights.

    This function evaluates the performance of a trading strategy by applying allocated weights
    to predicted ETF returns within a specified backtesting framework. The process involves:

    1. Selecting ETF and market return data.
    2. Extracting training and testing predicted returns from the `fit_result`.
    3. Allocating portfolio weights using the specified weighting scheme (currently supports 'equal').
    4. Calculating portfolio returns for both training and testing periods.
    5. Accounting for trading costs during the allocation process.

    Parameters
    ----------
    rets : pandas.DataFrame
        A DataFrame containing daily returns for ETFs and the market ('SP500').
        The index represents dates, and columns correspond to ETF ticker symbols plus 'SP500'.

    betas : pandas.DataFrame
        A DataFrame containing beta coefficients for each ETF relative to the market (SP500).
        The index should align with `rets`, and columns correspond to ETF ticker symbols.

    fit_result : dict
        A nested dictionary containing predicted returns for each ETF, split into training and testing sets.
        The structure should be:

            fit_result = {
                'ETF_Ticker_1': {
                    'train': pandas.DataFrame with a column 'ETF_Ticker_1_pred',
                    'test': pandas.DataFrame with a column 'ETF_Ticker_1_pred'
                },
                'ETF_Ticker_2': {
                    'train': pandas.DataFrame with a column 'ETF_Ticker_2_pred',
                    'test': pandas.DataFrame with a column 'ETF_Ticker_2_pred'
                },
                ...
            }

    weight_scheme : str, optional
        The method used to allocate portfolio weights. Currently supports:

        - 'equal': Equal weighting based on the sign of predicted returns and betas.

        Default is 'equal'.

    shift : int, optional
        An integer representing the shift in allocation periods, used when `weight_scheme` is 'equal'.
        This determines the specific day within each 21-day window when weights are recalculated.
        Default is 0.

    trading_cost : float, optional
        The transaction cost per trade, represented as a decimal (e.g., 0.0009 for 0.09%).
        This cost is applied to the portfolio during weight allocation to simulate realistic trading conditions.
        Default is 0.0009.

    Returns
    -------
    dict
        A dictionary containing the backtested portfolio returns for both training and testing periods:

            {
                'train': pandas.DataFrame with portfolio and market returns for the training set,
                'test': pandas.DataFrame with portfolio and market returns for the testing set
            }

    Notes
    -----
    - The function currently supports only the 'equal' weighting scheme. Additional schemes can be
      implemented and integrated as needed.
    - Trading costs are considered during the weight allocation process to provide a more accurate
      assessment of strategy performance.
    - The function assumes that the `fit_result` dictionary is properly structured and contains
      non-overlapping training and testing predictions for each ETF.
    - Portfolio returns are calculated using the `get_ptf_returns` function, which aligns weights with
      returns and computes the portfolio's daily performance.
    - The backtest results include both the portfolio's returns ('ptf') and the market's returns ('market')
      for comparative analysis.
    - Ensure that the `equal_weight_ptf` and `get_ptf_returns` functions are properly defined and
      accessible within the scope of this function.
    """

    # Get ETF columns
    etfs = [col for col in rets.columns if col.startswith('X')]

    # rets_copy
    rets_copy = rets.copy()

    # Set asset returns
    rets = rets[etfs + ['SP500']]

    # Set train data
    train = pd.concat([fit_result[etf]['train'][f'{etf}_pred'] for etf in etfs], axis=1)
    train.columns = etfs
    train = train[train.isna().sum(axis=1) < 3]

    # Set test data
    test = pd.concat([fit_result[etf]['test'][f'{etf}_pred'] for etf in etfs], axis=1)
    test.columns = etfs

    if weight_scheme == 'equal':
        train_weight = equal_weight_ptf(train, betas, shift=shift, rebalance=rebalance).fillna(0)
        test_weight = equal_weight_ptf(test, betas, shift=shift, rebalance=rebalance).fillna(0)
    elif (weight_scheme == 'sharpe_optimize') | (weight_scheme == 'min_variance_optimize'):
        # Calculate available returns
        # Set betas
        betas_trunc = betas[betas.isna().sum(axis=1) < 3].copy()

        # Slice based on beta index
        rets_available = rets.loc[betas_trunc.index, betas_trunc.columns].copy()

        # Set available returns
        mask = ~betas_trunc.isna()
        rets_available = rets_available.where(mask)

        ### 1. Eliminate market returns from etf returns using beta
        market_returns = rets.loc[betas_trunc.index, 'SP500'].copy()
        market_ret_adjust = betas_trunc.mul(market_returns, axis=0)
        rets_available -= market_ret_adjust

        marketless_returns = rets_available

        if weight_scheme == 'sharpe_optimize':
            train_weight = optimized_weight_ptf(train, marketless_returns, betas,
                                                shift=shift, rebalance=rebalance,
                                                period=period, mode='sharpe').fillna(0)
            test_weight = optimized_weight_ptf(test, marketless_returns, betas,
                                               shift=shift, rebalance=rebalance,
                                               period=period, mode='sharpe').fillna(0)
        elif weight_scheme == 'min_variance_optimize':
            train_weight = optimized_weight_ptf(train, marketless_returns, betas,
                                                shift=shift, rebalance=rebalance,
                                                period=period, mode='min_variance').fillna(0)
            test_weight = optimized_weight_ptf(test, marketless_returns, betas,
                                               shift=shift, rebalance=rebalance,
                                               period=period, mode='min_variance').fillna(0)
    elif weight_scheme == 'rolling':
        test_weight = rolling_train_weights(rets_copy, betas, start_date='2024-01-01', significance_level=0.05, shift=shift, rebalance=21)

    ### Backtest train

    # Calculate train portfolio returns
    if weight_scheme != 'rolling':
        train_ptf_rets, train_turnover = get_ptf_returns(train_weight, rets_copy, trading_cost=trading_cost)
    else:
        train_ptf_rets = None

    # Calculate test portfolio returns
    test_ptf_rets, test_turnover = get_ptf_returns(test_weight, rets_copy, trading_cost=trading_cost)

    return {'train': train_ptf_rets, 'test': test_ptf_rets, 'train_turnover': train_turnover, 'test_turnover': test_turnover}


if __name__ == '__main__':
    # Read returns
    rets = pd.read_csv('../data/returns/returns.csv', index_col='Date')
    rets.index = pd.to_datetime(rets.index)

    # Read betas
    betas = pd.read_csv('../data/etf_prices/beta.csv', index_col='Date')
    betas.index = pd.to_datetime(betas.index)

    # # Load fit result
    # with open("../data/model/fit_result.pkl", "rb") as file:
    #     fit_result = pickle.load(file)
    #
    # shift = 0
    #
    # print('\nshift: ' + str(shift))
    # result = backtest(rets, betas, fit_result, weight_scheme='sharpe_optimize', shift=shift)
    #
    # print(f'Train Evaluation')
    # evaluate(result['train'], data_name=f'train_shift_{shift}', weight_scheme='sharpe_optimize')
    #
    # print(f'Test Evaluation')
    # evaluate(result['test'], data_name=f'test_shift_{shift}', weight_scheme='sharpe_optimize')

    weights = rolling_train_weights(rets, betas, start_date='2024-01-01', shift=0, rebalance=21)
    print(weights)