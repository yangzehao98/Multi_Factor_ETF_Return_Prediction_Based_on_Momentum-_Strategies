import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os

def evaluate(ptf_rets,
             turnover,
             data_name='backtest',
             annualization_factor=252,
             weight_scheme='equal',
             rebalance=21,
             ):
    """
    Evaluate portfolio and market performance.

    Parameters:
    - ptf_rets (pd.DataFrame): DataFrame containing portfolio and market returns with columns 'ptf' and 'market'.
    - data_name (str): Identifier for the dataset (e.g., 'train', 'test') to customize plot titles and filenames.
    - annualization_factor (int): Number of periods in a year (default is 252 for daily data).

    Returns:
    - metrics_df (pd.DataFrame): DataFrame containing performance metrics.
    """
    # Get the root directory of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check if required columns exist
    required_cols = ['ptf', 'market']
    for col in required_cols:
        if col not in ptf_rets.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column.")

    # Calculate cumulative returns
    cum_return = (1 + ptf_rets).cumprod()

    # Prepare the directory for saving plots
    plot_dir = os.path.join(project_root,f'data/plots/backtest/{weight_scheme}/{rebalance}')
    os.makedirs(plot_dir, exist_ok=True)

    # Calculate annualized returns
    periods = len(cum_return)
    annual_return_ptf = cum_return['ptf'].iloc[-1] ** (annualization_factor / periods) - 1
    annual_return_market = cum_return['market'].iloc[-1] ** (annualization_factor / periods) - 1

    # Calculate annualized volatility
    volatility_ptf = ptf_rets['ptf'].std() * np.sqrt(annualization_factor)
    volatility_market = ptf_rets['market'].std() * np.sqrt(annualization_factor)

    # Calculate Sharpe Ratio (assuming risk-free rate is 0)
    sharpe_ptf = annual_return_ptf / volatility_ptf
    sharpe_market = annual_return_market / volatility_market

    # Calculate Sortino Ratio
    downside_ptf = ptf_rets['ptf'][ptf_rets['ptf'] < 0]
    downside_market = ptf_rets['market'][ptf_rets['market'] < 0]
    sortino_ptf = annual_return_ptf / (downside_ptf.std() * np.sqrt(annualization_factor)) if not downside_ptf.empty else np.nan
    sortino_market = annual_return_market / (downside_market.std() * np.sqrt(annualization_factor)) if not downside_market.empty else np.nan

    # Calculate Maximum Drawdown
    rolling_max_ptf = cum_return['ptf'].cummax()
    drawdown_ptf = (cum_return['ptf'] / rolling_max_ptf) - 1
    max_drawdown_ptf = drawdown_ptf.min()

    rolling_max_market = cum_return['market'].cummax()
    drawdown_market = (cum_return['market'] / rolling_max_market) - 1
    max_drawdown_market = drawdown_market.min()

    # Compile metrics into a dictionary
    metrics = {
        'Annual Return': {
            'Portfolio': annual_return_ptf,
            'Market': annual_return_market
        },
        'Annual Volatility': {
            'Portfolio': volatility_ptf,
            'Market': volatility_market
        },
        'Sharpe Ratio': {
            'Portfolio': sharpe_ptf,
            'Market': sharpe_market
        },
        'Sortino Ratio': {
            'Portfolio': sortino_ptf,
            'Market': sortino_market
        },
        'Maximum Drawdown': {
            'Portfolio': max_drawdown_ptf,
            'Market': max_drawdown_market
        },
        'Turnover': {
            'Portfolio': turnover,
            'Market': 0
        }
    }

    # Convert metrics dictionary to a DataFrame for tabular presentation
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df[['Portfolio', 'Market']]  # Ensure consistent column order

    # Display the metrics table
    print("\nPerformance Metrics:")
    print(metrics_df.round(4).to_string())

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    other_columns = cum_return.columns.drop(['ptf', 'market'])
    for col in other_columns:
        plt.plot(cum_return.index, cum_return[col], alpha=0.3, label=col)
    plt.plot(cum_return.index, cum_return['market'], color='blue', linewidth=2, label='Market')
    plt.plot(cum_return.index, cum_return['ptf'], color='red', linewidth=2, label='Portfolio')

    # Market Ptf correlation
    mkt_ptf_corr = np.corrcoef(ptf_rets['ptf'], ptf_rets['market'])[0][1]

    plt.title(f'{data_name.capitalize()} Cumulative Returns [Sharpe: {np.round(sharpe_ptf,2)} (mkt: {np.round(sharpe_market,2)}), Drawdown: {np.round(max_drawdown_ptf, 2)}, Corr:{np.round(mkt_ptf_corr, 2)}]')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True)
    # Adjust Layout to Make Room for the Legend
    plt.tight_layout(rect=[0, 0, 1, 1])  # Left, Bottom, Right, Top
    cum_return_plot_path = os.path.join(plot_dir, f'{data_name}_cumulative_returns.png')
    plt.savefig(cum_return_plot_path)
    plt.close()

    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    plt.plot(drawdown_ptf.index, drawdown_ptf, label='Portfolio Drawdown')
    plt.plot(drawdown_market.index, drawdown_market, label='Market Drawdown')
    plt.title(f'{data_name.capitalize()} Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    drawdown_plot_path = os.path.join(plot_dir, f'{data_name}_drawdowns.png')
    plt.savefig(drawdown_plot_path)
    plt.close()

    return metrics_df
