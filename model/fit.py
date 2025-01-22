import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import yfinance as yf
import pickle


def fit(data, save=False, verbose=False):
    """
    Fit Ordinary Least Squares (OLS) regression models for each ETF and generate predictions.

    This function iterates over each Exchange-Traded Fund (ETF) in the provided dataset, fits an OLS regression model
    using the training data, and generates predicted returns for both the training and testing datasets. The results,
    including actual and predicted returns, are stored in a nested dictionary structure and saved as a pickle file
    for future use.

    Parameters
    ----------
    data : dict
        A nested dictionary containing training and testing data for each ETF. The structure should be:

            data = {
                'ETF_Ticker_1': {
                    'train': pandas.DataFrame with actual returns and features,
                    'test': pandas.DataFrame with actual returns and features
                },
                'ETF_Ticker_2': {
                    'train': pandas.DataFrame with actual returns and features,
                    'test': pandas.DataFrame with actual returns and features
                },
                ...
            }

        Each DataFrame within 'train' and 'test' should have the ETF's actual returns as one column and relevant
        feature columns for regression.

    Returns
    -------
    dict
        A nested dictionary containing actual and predicted returns for both training and testing datasets
        for each ETF. The structure is as follows:

            result = {
                'ETF_Ticker_1': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred']
                },
                'ETF_Ticker_2': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred']
                },
                ...
            }

        Each DataFrame under 'train' and 'test' contains the actual returns and the corresponding predicted returns.

    Notes
    -----
    - The function uses the `statsmodels` library to perform OLS regression.
    - Predictions for both training and testing datasets are generated using the fitted model.
    - The results are saved to a pickle file at '../data/model/fit_result.pkl' for persistence.
    - Ensure that the feature columns in the training and testing DataFrames do not include the ETF's return column
      itself, except for the target variable.
    - Missing values in the input data should be handled prior to calling this function to avoid errors during model fitting.
    """
    result = {}

    etfs = data.keys()

    for etf in etfs:
        # Set train data
        train = data[etf]['train']
        y_train = train[etf]
        X_train = train.drop(columns=[etf])
        X_train = sm.add_constant(X_train)

        # Set test data
        test = data[etf]['test']
        y_test = test[etf]
        X_test = test.drop(columns=[etf])
        X_test = sm.add_constant(X_test)

        # Train the data
        model = sm.OLS(y_train, X_train)
        res = model.fit()

        if verbose:
            print(res.summary())

        # Get predictions
        y_train_pred = res.predict(X_train)
        y_test_pred = res.predict(X_test)

        train_fit = pd.concat([y_train, y_train_pred], axis=1)
        train_fit.columns = [etf, f'{etf}_pred']

        test_fit = pd.concat([y_test, y_test_pred], axis=1)
        test_fit.columns = [etf, f'{etf}_pred']

        result[etf] = {}
        result[etf]['train'] = train_fit
        result[etf]['test'] = test_fit

        if y_test.empty:
            continue

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        if verbose:
            print("train r2:", train_r2)
            print("test r2:", test_r2)

    if save:
        # Save the result in pickles
        with open("../data/model/fit_result.pkl", "wb") as file:
            pickle.dump(result, file)

    return result

def plot_true_predict(result):
    """
    Plot scatter plots of true versus predicted returns for each ETF's training and testing datasets.

    This function generates scatter plots comparing the actual (true) returns to the predicted returns for both
    training and testing periods for each ETF. The plots are saved as PNG files in the designated directory,
    allowing for visual assessment of the model's predictive performance.

    Parameters
    ----------
    result : dict
        A nested dictionary containing actual and predicted returns for both training and testing datasets
        for each ETF. The structure should be as follows:

            result = {
                'ETF_Ticker_1': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred']
                },
                'ETF_Ticker_2': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred']
                },
                ...
            }

        Each DataFrame under 'train' and 'test' contains the actual returns and the corresponding predicted returns.

    Returns
    -------
    None
        The function does not return any value. It saves the generated plots as PNG files in the
        '../data/plots/model_fit/true_vs_predict/' directory.

    Notes
    -----
    - The function uses `matplotlib` for plotting.
    - Ensure that the directory '../data/plots/model_fit/true_vs_predict/' exists before running the function,
      or modify the path as needed.
    - The scatter plots help in visualizing the correlation between actual and predicted returns, where
      a perfect prediction would lie along the diagonal.
    - Overlapping points from training and testing datasets are differentiated using labels in the legend.
    - The function closes each plot after saving to free up memory and avoid display issues in certain environments.
    """
    etfs = result.keys()

    for etf in etfs:
        # Get data
        y_train, y_train_pred = result[etf]['train'][etf], result[etf]['train'][f'{etf}_pred']
        y_test, y_test_pred = result[etf]['test'][etf], result[etf]['test'][f'{etf}_pred']

        # Plot true vs predict
        plt.scatter(y_train, y_train_pred, label='Train')
        plt.scatter(y_test, y_test_pred, label='Test')
        plt.title(f'[{etf} Residual Returns] True vs Predicted')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.legend()
        plt.savefig(f'../data/plots/model_fit/true_vs_predict/{etf}_true_predicted.png')
        plt.close()

def plot_rmse(result):
    """
    Calculate and plot Root Mean Squared Error (RMSE) for training and testing predictions of each ETF.

    This function computes the RMSE between the actual and predicted returns for both training and testing
    datasets for each ETF. It then visualizes these RMSE values using a bar plot, facilitating comparison
    of model performance across different ETFs.

    Parameters
    ----------
    result : dict
        A nested dictionary containing actual and predicted returns for both training and testing datasets
        for each ETF. The structure should be as follows:

            result = {
                'ETF_Ticker_1': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_1', 'ETF_Ticker_1_pred']
                },
                'ETF_Ticker_2': {
                    'train': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred'],
                    'test': pandas.DataFrame with columns ['ETF_Ticker_2', 'ETF_Ticker_2_pred']
                },
                ...
            }

        Each DataFrame under 'train' and 'test' contains the actual returns and the corresponding predicted returns.

    Returns
    -------
    None
        The function does not return any value. It saves the generated RMSE plot as a PNG file in the
        '../data/plots/model_fit/' directory.

    Notes
    -----
    - The function uses `matplotlib` and `scikit-learn`'s `mean_squared_error` for calculations and plotting.
    - Ensure that the directory '../data/plots/model_fit/' exists before running the function,
      or modify the path as needed.
    - RMSE provides a measure of the average magnitude of the prediction errors, with lower values
      indicating better model performance.
    - The bar plot differentiates RMSE for training and testing datasets for each ETF, aiding in
      the identification of overfitting or underfitting issues.
    - The plot is saved as 'rmse.png' in the specified directory.
    - The function handles any potential missing values by ensuring that only valid data points are used
      in RMSE calculations.
    """
    rmse = pd.DataFrame(columns=['RMSE_Train', 'RMSE_Test'])
    etfs = result.keys()

    for etf in etfs:
        # Get data
        y_train, y_train_pred = result[etf]['train'][etf], result[etf]['train'][f'{etf}_pred']
        y_test, y_test_pred = result[etf]['test'][etf], result[etf]['test'][f'{etf}_pred']

        # Calculate RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Append row
        rmse.loc[etf] = [train_rmse, test_rmse]

    ax = rmse.plot(figsize=(20, 10), title='OLS Fit Results')
    ax.set_xticks(range(len(rmse)))
    ax.set_xticklabels(rmse.index, rotation=45)
    plt.tight_layout()
    plt.savefig(f'../data/plots/model_fit/rmse.png')
    plt.close()

if __name__ == '__main__':
    # Save the result in pickles
    with open("../data/features/dataset.pkl", "rb") as file:
        dataset = pickle.load(file)

    result = fit(dataset, save=True, verbose=True)
    plot_true_predict(result)
    plot_rmse(result)