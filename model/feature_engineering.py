import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import KMeans
import os



def lead_lag_correlations(rets, holding_n, lag_n, save=True, plot=True):
    """
    Compute lead-lag correlations between ETF returns and various features across different holding and lag periods.

    This function takes a DataFrame of returns (`rets`) including multiple ETFs and other features (e.g., SMB, HML, etc.),
    and computes the correlation of each ETF's one-month forward return with the rolling holding-period returns of each feature
    at various lags. It applies Fisher's Z-transformation to aggregate multiple sampled correlations, computes significance
    (Z-scores), and p-values. Optionally, it plots heatmaps of correlations, p-values, and significances.

    Parameters
    ----------
    rets : pandas.DataFrame
        A DataFrame of daily returns indexed by date, with columns representing ETFs and other features.
        Columns starting with 'X' are considered ETF columns.
    holding_n : int
        The maximum number of holding periods (in months) for which to compute correlations.
    lag_n : int
        The maximum number of lag periods (in months) for which to compute correlations.
    plot : bool, optional
        Whether to plot the correlation, p-value, and significance heatmaps for each ETF-feature combination.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing:
        - 'correlations': dict of dict of numpy arrays
            correlations[etf][feature] is a holding_n x lag_n array of correlation coefficients.
        - 'significances': dict of dict of numpy arrays
            significances[etf][feature] is a holding_n x lag_n array of Z-scores for significance.
        - 'p_values': dict of dict of numpy arrays
            p_values[etf][feature] is a holding_n x lag_n array of p-values.

    Notes
    -----
    - The function internally saves computed correlations, p-values, and significances as pickle files.
    - The correlation computation uses a rolling window of one month (21 days) and aggregates by applying non-overlapping
      sampling. Fisher's Z-transformation is used to aggregate correlation estimates from multiple sub-samples.
    - The significance (Z-score) and p-values are derived from the aggregated Z statistic.
    - Heatmaps are saved as .png files in designated directories for each ETF-feature pair if plot is True.
    """

    # Initialize variables
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    one_month_days = 21
    correlations = {}
    significances = {}
    p_values = {}

    # Get ETF columns
    etfs = [col for col in rets.columns if col.startswith('X')]

    for etf in etfs:
        # Initialize variables
        correlations[etf] = {}
        significances[etf] = {}
        p_values[etf] = {}

        for feature in rets.columns:
            correlations[etf][feature] = np.zeros((holding_n, lag_n))
            significances[etf][feature] = np.zeros((holding_n, lag_n))
            p_values[etf][feature] = np.zeros((holding_n, lag_n))

        rets_temp = rets.copy()
        y = rets_temp[[etf]].copy()

        # Compute y (one-month forward return)
        y[etf] = (1 + y[etf]).rolling(window=one_month_days).apply(lambda x: x.prod(), raw=True) - 1
        y[etf] = y[etf].shift(-one_month_days)
        y.dropna(inplace=True)

        # Compute X's
        for lag in range(1, lag_n + 1):
            for holding in range(1, holding_n + 1):
                # Compute holding period return
                X = (1 + rets_temp).rolling(window=one_month_days * holding).apply(lambda x: x.prod(), raw=True) - 1

                # Apply lag to holding period return
                X = X.shift(one_month_days * lag)

                # Merge y and X
                etf_df = pd.concat([y, X], axis=1)
                etf_df.dropna(subset=[etf], inplace=True)
                etf_df = etf_df[etf_df.isna().sum(axis=1) <= 2]
                etf_df.dropna(axis=1, inplace=True)

                # Calculate sample correlations
                sum_w = 0
                sum_wz = 0
                sample_cnt = 0
                for i in range(one_month_days * holding):
                    # Non-overlapping returns sampling
                    sample = etf_df[i::one_month_days * holding]

                    # Calculate sample correlation
                    r = sample.corr().loc[:, [etf]].iloc[1:, [0]]

                    # Get number of data points and get weight from it
                    n = len(sample)

                    # Skip for bad samples
                    if n < 4:
                        continue

                    w = n - 3

                    # Transform to Fisher's Z-transformation
                    z = 1 / 2 * np.log((1 + r) / (1 - r))

                    # Add up to sums
                    sum_wz += w * z
                    sum_w += w
                    sample_cnt += 1

                # Skip for bad samples
                if sum_w <= 0:
                    # Save correlation and significance
                    for feature in etf_df.columns:
                        correlations[etf][feature][holding - 1, lag - 1] = 0
                        significances[etf][feature][holding - 1, lag - 1] = 0
                        p_values[etf][feature][holding - 1, lag - 1] = 1
                    continue

                # Calculate z_bar
                z_bar = sum_wz / sum_w

                # Calculate the standard error of the weighted mean
                se_z_bar = 1 / np.sqrt(sum_w)

                # Calculate z score
                z_score = z_bar / se_z_bar

                # Calculate r_bar
                r_bar = (np.exp(2 * z_bar) - 1) / (np.exp(2 * z_bar) + 1)

                # Get p values
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
                p_value_df = z_score.copy()
                p_value_df.loc[:,etf] = p_value[:,0]

                # Save correlation and significance
                for feature in z_score.index:
                    correlations[etf][feature][holding - 1, lag - 1] = float(r_bar.loc[feature].iloc[0])
                    significances[etf][feature][holding - 1, lag - 1] = float(z_score.loc[feature].iloc[0])
                    p_values[etf][feature][holding - 1, lag - 1] = float(p_value_df.loc[feature].iloc[0])
    if save:

        # Save the result in pickles
        with open(os.path.join(project_dir, "data/pickles/corrs.pkl"), "wb") as file:
            pickle.dump(correlations, file)

        with open(os.path.join(project_dir, "data/pickles/p_values.pkl"), "wb") as file:
            pickle.dump(p_values, file)

        with open(os.path.join(project_dir, "data/pickles/sigs.pkl"), "wb") as file:
            pickle.dump(significances, file)

        print("Saved correlation, p_values, and significances to pickle files.")

    if plot:
        # Plot results
        for etf in etfs:
            features = p_values[etf].keys()

            for feature in features:
                corr = correlations[etf][feature]
                # Plot p values heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                annot_kws={"size": 10}, linewidths=0.5)
                plt.xlabel("Lag Period", fontsize=10)
                plt.ylabel("Holding Period", fontsize=10)
                plt.title(f"Correlation Heat Map [ETF: {etf}, Feature: {feature}]")
                plt.savefig(os.path.join(project_dir, f"data/plots/correlations/correlation_ETF_{etf}_Feature_{feature}"))
                plt.close()

                p_vals = p_values[etf][feature]
                # Plot heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(p_vals, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                annot_kws={"size": 10}, linewidths=0.5)
                plt.xlabel("Lag Period", fontsize=10)
                plt.ylabel("Holding Period", fontsize=10)
                plt.title(f"p-Value Heat Map [ETF: {etf}, Feature: {feature}]")
                plt.savefig(os.path.join(project_dir, f"data/plots/p_values/p_value_ETF_{etf}_Feature_{feature}"))
                plt.close()

                sig = significances[etf][feature]
                # Plot heatmap
                levels = np.maximum(sig / 1.96,
                                  np.ones_like(sig))
                plt.figure(figsize=(10, 8))
                sns.heatmap(levels, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                annot_kws={"size": 10}, linewidths=0.5)
                plt.xlabel("Lag Period", fontsize=10)
                plt.ylabel("Holding Period", fontsize=10)
                plt.title(f"Significance Heat Map [ETF: {etf}, Feature: {feature}]")
                plt.savefig(os.path.join(project_dir, f"data/plots/significances/significance_ETF_{etf}_Feature_{feature}"))
                plt.close()

                print("Saved plots.")

    return {'correlations': correlations, 'significances': significances, 'p_values': p_values}


def extract_lag_features(correlations, significances, significance_level=0.05, save=True, plot=True):
    """
    Extract the most significant holding-lag feature combinations from correlations and significances.

    This function takes the correlation and significance results from `lead_lag_correlations()` and identifies
    combinations of holding and lag periods that yield significantly high z-scores for each ETF-feature pair.
    It uses K-Means clustering to group holding-lag combinations based on their standardized values. The top
    combinations per cluster are selected based on their Z-scores, and only those exceeding a specified
    significance threshold are retained.

    Parameters
    ----------
    correlations : dict
        A dictionary of correlation results as produced by `lead_lag_correlations()`.
    significances : dict
        A dictionary of significance (Z-score) results as produced by `lead_lag_correlations()`.
    significance_level : float, optional
        Significance level used to filter combinations. The default is 0.01.
    plot : bool, optional
        Whether to plot a heatmap of K-Means clustering results with annotated correlation values. Default is True.

    Returns
    -------
    dict
        A dictionary of extracted lag features of the form:
        lag_features[etf][feature] = list of (holding_period, lag_period) tuples.

    Notes
    -----
    - The function saves the extracted lag features as a pickle file.
    - K-Means is used to cluster holding-lag combinations into two clusters.
    - Only those combinations that have a Z-score above the critical value associated with `significance_level`
      are considered significant.
    - A heatmap of clusters with annotated correlations is saved for each ETF-feature pair if plot is True.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lag_features = {}

    # Get ETF columns
    etfs = [col for col in significances.keys() if col.startswith('X')]

    for etf in etfs:
        # Initialize lag features
        lag_features[etf] = {}

        # Get feature names
        features = significances[etf].keys()

        for feature in features:
            # Get z_scores
            z_scores = significances[etf][feature]

            # Get correlations
            corr = correlations[etf][feature]

            # Get holding/lag periods
            holding_periods = z_scores.shape[0]
            lag_periods = z_scores.shape[1]

            # Make the list of periods
            holding_periods = np.arange(1, holding_periods + 1)
            lag_periods = np.arange(1, lag_periods + 1)

            # Reformat the z score matrix
            data = []
            for i, hp in enumerate(holding_periods):
                for j, lp in enumerate(lag_periods):
                    z_score = z_scores[i, j]
                    corr_val = corr[i, j]
                    data.append([hp, lp, z_score, corr_val])

            # Form a data frame
            df = pd.DataFrame(data, columns=['HoldingPeriod', 'LagPeriod', 'ZScore', 'Correlation'])

            # Standardize the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[['HoldingPeriod', 'LagPeriod', 'Correlation']])

            # Apply K-Means
            k = 2  # Specify the number of clusters
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            df['Cluster'] = clusters

            if plot:
                # Pivot the DataFrame to create a matrix for the heatmap
                cluster_matrix = df.pivot(index='HoldingPeriod', columns='LagPeriod', values='Cluster')

                # Pivot the DataFrame to create a matrix for z-scores
                corr_matrix = df.pivot(index='HoldingPeriod', columns='LagPeriod', values='Correlation')

                plt.figure(figsize=(10, 8))

                # Create a custom colormap to distinguish clusters
                cmap = sns.color_palette("tab20", n_colors=len(df['Cluster'].unique()))

                # Plot the heatmap
                sns.heatmap(cluster_matrix, annot=corr_matrix.round(2), fmt='.2f', cmap=cmap, cbar_kws={'label': 'Cluster Label'})

                # Set axis labels and title
                plt.title(f'KMeans Clustering Heatmap [ETF_{etf}_Feature_{feature}]')
                plt.xlabel('Lag Period')
                plt.ylabel('Holding Period')

                # Save the plot
                plt.savefig(os.path.join(project_dir, f'data/plots/feature_clustering/kmeans_clustering_ETF_{etf}_Feature_{feature}.png'))
                plt.close()

            # Filter out noise points labeled as -1
            df_clusters = df[~df['Cluster'].isna()]

            # Group by clusters and find the combination with the highest z-score
            top_combinations = df_clusters.loc[df_clusters.groupby('Cluster')['ZScore'].idxmax()]

            # Filter clusters if not significant
            z_sig = norm.ppf(1 - significance_level / 2)
            top_combinations = top_combinations[top_combinations['ZScore'] > z_sig]

            if not top_combinations.empty:
                # top_combinations = top_combinations.sort_values(by='ZScore', ascending=False).iloc[:10]
                lag_features[etf][feature] = list(zip(top_combinations["HoldingPeriod"], top_combinations["LagPeriod"]))

    if save:
        # Save the result in pickles
        with open(os.path.join(project_dir, "data/features/features.pkl"), "wb") as file:
            pickle.dump(lag_features, file)
        print("Saved features.")

    return lag_features


def construct_features(rets, betas, train_end, significance_level=0.05, save=False, plot=False):
    """
    Construct a dataset of features for predictive modeling by:
    1. Adjusting ETF returns to remove market returns based on given betas.
    2. Using `lead_lag_correlations()` to compute correlations and significances between ETFs and various features.
    3. Using `extract_lag_features()` to select the most significant holding-lag feature combinations.
    4. Building a final dataset that includes one-month forward returns of ETFs (as the target, y) and
       the selected holding-lag features (X).

    Parameters
    ----------
    rets : pandas.DataFrame
        A DataFrame of daily returns indexed by date, with columns representing ETFs and other features.
        The column 'SP500' should represent the market benchmark.
    betas : pandas.DataFrame
        A DataFrame of betas indexed by date, with columns corresponding to the ETFs. Betas represent sensitivity
        of ETF returns to the market returns.
    start_date : str or datetime
        The start date of the analysis period.
    end_date : str or datetime
        The end date of the analysis period.
    train_end : str or datetime
        The end date of the training period. Observations after this date are considered test data.

    Returns
    -------
    dict
        A dictionary of ETF datasets, where each key is an ETF and each value is a dictionary containing:
        - 'train': DataFrame with training data (y and selected features)
        - 'test': DataFrame with test data (y and selected features)

    Notes
    -----
    - The function first adjusts ETF returns by removing the component explained by the market (SP500) using the given betas.
    - It then calls `lead_lag_correlations()` on the training subset to compute correlations and significances.
    - It calls `extract_lag_features()` to find the most predictive holding-lag combinations.
    - Finally, it constructs and returns datasets with these features, splitting into train and test sets.
    - The resulting dataset is saved as a pickle file.
    - Features are named using the format "feature(lag,holding)".
    """
    # Set variables
    one_month_days = 21
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Get ETF columns
    etfs = [col for col in rets.columns if col.startswith('X')]

    # Set betas
    betas = betas[betas.isna().sum(axis=1) < 3]

    # Slice based on beta index
    rets_available = rets.loc[betas.index, betas.columns].copy()

    # Set available returns
    mask = ~betas.isna()
    rets_available = rets_available.where(mask)

    ### 1. Eliminate market returns from etf returns using beta
    market_returns = rets.loc[betas.index, 'SP500'].copy()
    market_ret_adjust = betas.mul(market_returns, axis=0)
    rets_available -= market_ret_adjust

    rets_all = rets.copy()
    rets_all.loc[rets_available.index, rets_available.columns] = rets_available
    result = lead_lag_correlations(rets_all.loc[:train_end, :], holding_n=12, lag_n=12, save=save, plot=plot)


    # ### 2. Calculate correlation
    # result = lead_lag_correlations(rets_available.loc[:train_end], holding_n=12, lag_n=12, save=False, plot=False)

    ### 3. Extract features
    corrs = result['correlations']
    sigs = result['significances']
    lag_features = extract_lag_features(corrs, sigs, significance_level=significance_level, save=save, plot=plot)

    # lag_features = {'XLC': {'XLC': [(1, 9), (6, 1)], 'XLY': [(1, 3), (12, 12), (2, 8)], 'XLP': [(8, 5), (11, 10), (5, 10)], 'XLE': [(11, 12), (3, 12)], 'XLV': [(1, 10), (8, 3), (1, 6)], 'XLI': [(2, 4), (8, 2)], 'XLB': [(1, 9), (1, 2)], 'XLK': [(6, 8), (12, 1), (1, 1)], 'XLU': [(5, 10), (12, 9), (9, 5)], 'SMB': [(2, 4)], 'RMW': [(10, 12), (3, 11)], 'CMA': [(6, 11), (3, 11)], 'Mom': [(5, 10), (8, 10), (8, 5)], 'ST_Rev': [(5, 6), (4, 6)], 'SP500': [(1, 9), (3, 4)], 'Gold': [(6, 9), (12, 2), (2, 2)], '3yr_Treasury': [(10, 9), (5, 10)], '30yr_Treasury': [(1, 6), (1, 9)]}, 'XLY': {'XLY': [(1, 6), (7, 12)], 'XLP': [(12, 5), (1, 12), (3, 4)], 'XLE': [(9, 12), (1, 7)], 'XLF': [(5, 3), (3, 5)], 'XLV': [(10, 1), (12, 8), (2, 8)], 'XLI': [(1, 12), (8, 9)], 'XLB': [(2, 9), (10, 8), (2, 2)], 'XLK': [(1, 1), (12, 8), (1, 9)], 'XLU': [(1, 2)], 'SMB': [(4, 7), (10, 6)], 'HML': [(1, 12), (1, 7)], 'RMW': [(9, 6), (2, 12), (2, 6)], 'CMA': [(1, 12), (1, 4)], 'Mom': [(5, 1), (1, 8), (1, 4)], 'ST_Rev': [(2, 10), (12, 10), (2, 5)], 'LT_Rev': [(1, 12), (1, 4)], 'SP500': [(1, 10)], 'Gold': [(1, 1), (11, 4), (6, 9)], '3yr_Treasury': [(7, 9), (3, 11)], '10yr_Treasury': [(1, 9)]}, 'XLP': {'XLY': [(11, 11), (2, 1)], 'XLP': [(1, 7), (1, 10)], 'XLE': [(1, 5), (1, 10)], 'XLF': [(2, 1)], 'XLV': [(3, 7)], 'XLI': [(7, 12), (1, 5)], 'XLB': [(1, 10), (1, 5)], 'XLK': [(1, 4), (8, 11), (2, 8)], 'XLU': [(2, 6), (1, 7)], 'SMB': [(2, 12), (2, 1)], 'HML': [(1, 5)], 'RMW': [(2, 10)], 'CMA': [(1, 5)], 'Mom': [(3, 5)], 'ST_Rev': [(1, 3)], 'LT_Rev': [(2, 1)], 'SP500': [(4, 1), (8, 12), (3, 8)], 'Gold': [(1, 7), (11, 12), (2, 6)], '3yr_Treasury': [(4, 11), (5, 1)], '10yr_Treasury': [(1, 1), (2, 8)], '30yr_Treasury': [(1, 1)]}, 'XLE': {'XLY': [(2, 11), (1, 11)], 'XLP': [(3, 3)], 'XLE': [(1, 6), (9, 9), (1, 10)], 'XLF': [(1, 1), (8, 11), (1, 11)], 'XLV': [(1, 9)], 'XLI': [(6, 1), (12, 6), (1, 8)], 'XLB': [(7, 4), (9, 3), (6, 12)], 'XLK': [(2, 12), (1, 2)], 'XLU': [(6, 1), (1, 6)], 'SMB': [(10, 5), (11, 11), (2, 11)], 'HML': [(9, 11), (9, 3), (1, 6)], 'RMW': [(7, 4), (1, 5)], 'CMA': [(8, 2), (9, 9), (3, 7)], 'Mom': [(2, 3), (1, 7)], 'ST_Rev': [(1, 12), (11, 12)], 'LT_Rev': [(4, 6), (7, 8), (2, 7)], 'SP500': [(4, 3), (9, 12), (5, 12)], 'Gold': [(2, 2), (10, 12), (1, 9)], '3yr_Treasury': [(8, 1), (2, 6)], '10yr_Treasury': [(12, 5), (3, 10), (2, 5)], '30yr_Treasury': [(4, 3), (10, 10), (3, 10)]}, 'XLF': {'XLY': [(1, 6), (9, 7), (2, 2)], 'XLP': [(2, 8), (12, 8)], 'XLE': [(2, 2)], 'XLF': [(1, 7)], 'XLV': [(1, 1), (9, 9), (2, 7)], 'XLI': [(12, 12), (1, 3)], 'XLB': [(3, 1)], 'XLK': [(5, 1), (9, 8), (2, 10)], 'XLU': [(2, 3)], 'SMB': [(1, 6), (1, 11), (3, 1)], 'HML': [(5, 6), (10, 6)], 'RMW': [(2, 8), (9, 8), (1, 4)], 'CMA': [(4, 6), (1, 7)], 'Mom': [(12, 6), (1, 9), (1, 5)], 'ST_Rev': [(4, 12), (8, 1), (5, 3)], 'LT_Rev': [(3, 6), (1, 6)], 'SP500': [(3, 2)], 'Gold': [(1, 2)], '3yr_Treasury': [(8, 4), (2, 1)], '10yr_Treasury': [(4, 10), (2, 1)], '30yr_Treasury': [(1, 1)]}, 'XLV': {'XLY': [(1, 11), (12, 11), (10, 2)], 'XLP': [(8, 3), (11, 7), (1, 10)], 'XLE': [(1, 1), (1, 5)], 'XLF': [(9, 1), (1, 8)], 'XLV': [(2, 12), (5, 12)], 'XLI': [(1, 4), (10, 11), (1, 1)], 'XLB': [(5, 4), (4, 5)], 'XLK': [(1, 6)], 'XLU': [(2, 6)], 'SMB': [(3, 11), (2, 4)], 'HML': [(1, 10), (5, 1)], 'RMW': [(7, 5), (2, 10)], 'CMA': [(1, 10), (7, 1)], 'Mom': [(4, 5), (3, 5)], 'ST_Rev': [(1, 9)], 'LT_Rev': [(5, 10), (7, 1), (5, 1)], 'SP500': [(3, 11), (12, 10), (3, 4)], 'Gold': [(2, 10)], '3yr_Treasury': [(3, 3)], '10yr_Treasury': [(10, 12), (7, 1)], '30yr_Treasury': [(10, 12), (1, 5)]}, 'XLI': {'XLY': [(4, 1), (6, 3), (2, 3)], 'XLP': [(1, 7)], 'XLE': [(2, 1)], 'XLF': [(2, 10), (12, 10), (9, 3)], 'XLV': [(7, 5)], 'XLI': [(1, 3)], 'XLB': [(1, 11), (2, 1)], 'XLK': [(10, 1), (7, 12), (1, 12)], 'XLU': [(1, 11), (6, 1)], 'SMB': [(1, 2), (9, 10), (11, 1)], 'HML': [(4, 8), (9, 6)], 'RMW': [(12, 12), (1, 9)], 'CMA': [(6, 8), (3, 9), (2, 1)], 'Mom': [(1, 1)], 'ST_Rev': [(1, 10), (12, 10), (1, 3)], 'LT_Rev': [(3, 7), (9, 1), (2, 6)], 'SP500': [(1, 3), (8, 8), (1, 6)], 'Gold': [(1, 9), (7, 12), (3, 1)], '3yr_Treasury': [(1, 2), (9, 9), (5, 8)], '10yr_Treasury': [(1, 1), (10, 1)], '30yr_Treasury': [(10, 1), (1, 1)]}, 'XLB': {'XLY': [(3, 2), (12, 10), (6, 10)], 'XLP': [(2, 7)], 'XLE': [(2, 9), (9, 6)], 'XLF': [(2, 10), (8, 9)], 'XLV': [(8, 5), (4, 5)], 'XLI': [(1, 10)], 'XLB': [(11, 6), (9, 10), (2, 1)], 'XLK': [(12, 1), (1, 12)], 'XLU': [(1, 6), (7, 12)], 'SMB': [(5, 10), (8, 10), (2, 1)], 'HML': [(3, 9), (9, 6)], 'RMW': [(2, 8), (12, 8)], 'CMA': [(4, 7), (9, 6), (2, 1)], 'Mom': [(2, 8)], 'ST_Rev': [(10, 3), (3, 10), (2, 7)], 'LT_Rev': [(6, 6), (11, 5), (1, 9)], 'SP500': [(6, 10), (1, 3)], 'Gold': [(1, 2), (7, 12), (1, 9)], '3yr_Treasury': [(9, 2), (2, 2)], '10yr_Treasury': [(1, 1), (9, 5)], '30yr_Treasury': [(1, 1)]}, 'XLRE': {'XLY': [(2, 8), (2, 3), (1, 3)], 'XLP': [(1, 11), (1, 6)], 'XLE': [(1, 1)], 'XLF': [(2, 1)], 'XLV': [(2, 6), (3, 5)], 'XLI': [(1, 11), (2, 4)], 'XLB': [(6, 9)], 'XLRE': [(9, 3), (2, 3)], 'XLK': [(11, 3), (8, 11), (3, 11)], 'XLU': [(1, 11), (1, 7)], 'SMB': [(8, 3), (2, 4)], 'HML': [(2, 1)], 'RMW': [(1, 11)], 'CMA': [(1, 2)], 'Mom': [(3, 12), (1, 7)], 'ST_Rev': [(5, 6), (8, 10), (2, 9)], 'LT_Rev': [(1, 9), (1, 1)], 'SP500': [(9, 3), (8, 8), (2, 3)], 'Gold': [(9, 6), (7, 9), (1, 4)], '3yr_Treasury': [(1, 3)], '10yr_Treasury': [(2, 8), (9, 1)], '30yr_Treasury': [(9, 1), (2, 8)]}, 'XLK': {'XLY': [(1, 8)], 'XLP': [(12, 3), (4, 12), (1, 2)], 'XLE': [(8, 11), (7, 11)], 'XLF': [(9, 12), (1, 3)], 'XLV': [(5, 4), (12, 10)], 'XLI': [(2, 10), (10, 9), (5, 2)], 'XLB': [(1, 8), (12, 8)], 'XLK': [(3, 9), (12, 12), (3, 3)], 'XLU': [(1, 7), (12, 11)], 'SMB': [(3, 5)], 'HML': [(5, 12), (1, 7)], 'RMW': [(2, 6), (4, 12), (2, 2)], 'CMA': [(5, 12)], 'Mom': [(1, 8), (8, 12), (7, 2)], 'ST_Rev': [(12, 1), (11, 8), (1, 2)], 'LT_Rev': [(5, 12)], 'SP500': [(1, 5)], 'Gold': [(2, 10), (8, 10)], '3yr_Treasury': [(5, 12), (10, 11), (1, 1)], '10yr_Treasury': [(1, 7), (10, 7)], '30yr_Treasury': [(5, 12), (10, 7)]}, 'XLU': {'XLY': [(5, 1), (11, 10), (1, 1)], 'XLP': [(2, 2)], 'XLE': [(6, 1), (1, 10)], 'XLF': [(3, 1), (10, 8), (6, 1)], 'XLV': [(12, 11), (2, 11)], 'XLI': [(1, 1), (9, 9), (2, 3)], 'XLB': [(8, 8), (1, 3)], 'XLK': [(2, 11), (1, 12)], 'XLU': [(3, 6)], 'SMB': [(10, 3), (12, 9), (5, 10)], 'HML': [(4, 8), (10, 8), (3, 1)], 'RMW': [(1, 11)], 'CMA': [(3, 1), (12, 9), (1, 10)], 'Mom': [(2, 5)], 'ST_Rev': [(12, 11), (3, 2)], 'LT_Rev': [(12, 9), (2, 9)], 'SP500': [(6, 3), (9, 7), (5, 11)], 'Gold': [(10, 5), (1, 3)], '3yr_Treasury': [(1, 10), (5, 1)], '10yr_Treasury': [(1, 4), (7, 12), (1, 1)], '30yr_Treasury': [(1, 1), (10, 9), (1, 9)]}}

    ### 4. Construct data set with y and features
    data_set = {}

    for etf in etfs:
        rets_temp = rets.copy()
        y = rets_temp[[etf]].copy()

        # Compute y (one-month forward return)
        y[etf] = (1 + y[etf]).rolling(window=one_month_days).apply(lambda x: x.prod(), raw=True) - 1
        y[etf] = y[etf].shift(-one_month_days)
        y.dropna(inplace=True)

        # Compute X
        X_li = []
        for feature in lag_features[etf].keys():
            holding_lag_combinations = lag_features[etf][feature]

            for holding_lag_combination in holding_lag_combinations:
                holding, lag = holding_lag_combination

                # Compute X: holding period return
                X = rets[[feature]].copy()
                X.loc[:, feature] = (1 + X[feature]).rolling(window=one_month_days * holding).apply(lambda x: x.prod(), raw=True) - 1

                # Apply lag
                X.loc[:, feature] = X[feature].shift(one_month_days * lag)

                # Rename feature
                X.rename(columns={feature: f'{feature}({lag},{holding})'}, inplace=True)

                # Append X to X_li
                X_li.append(X)

        # Divide train test
        df = pd.concat([y] + X_li, axis=1).dropna()
        train_idx = len(df.loc[:train_end]) - 1 - one_month_days
        data_set[etf] = {'train' : df.iloc[:train_idx],
                         'test' : df.iloc[train_idx + 1 + one_month_days:],}

    # Save the result in pickles
    if save:
        with open(os.path.join(project_dir, "data/features/dataset.pkl"), "wb") as file:
            pickle.dump(data_set, file)
        print("Saved dataset")

    return data_set


if __name__ == '__main__':
    # Read returns
    rets = pd.read_csv('../data/returns/returns.csv', index_col='Date')
    rets.index = pd.to_datetime(rets.index)

    # Read betas
    betas = pd.read_csv('../data/etf_prices/beta.csv', index_col='Date')
    betas.index = pd.to_datetime(betas.index)

    data = construct_features(rets, betas, '2023-12-31', save=True, plot=True)
    print(data)

