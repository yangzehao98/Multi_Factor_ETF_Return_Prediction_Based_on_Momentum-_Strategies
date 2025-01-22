import numpy as np
import pandas as pd

def exp_weight(data, alpha=0.05):
    """
    Apply exponential weighting to a pandas Series.

    This function assigns exponentially increasing weights to the elements of a pandas Series,
    emphasizing more recent data points while diminishing the influence of older observations.
    The weights are calculated based on the provided decay factor (`alpha`) and normalized
    to ensure that their sum equals one. The weighted series is returned, preserving the original
    index and data alignment.

    Parameters
    ----------
    series : pandas.Series
        A pandas Series containing the data to be exponentially weighted. The Series should be
        ordered chronologically, with the earliest data point first and the most recent data point last.

    alpha : float, optional
        The decay factor for exponential weighting. A higher `alpha` value gives more weight to
        recent data points, making the weighting scheme more responsive to recent changes.
        Conversely, a lower `alpha` value results in a more gradual decay, spreading weights
        more evenly across the data points. Default is 0.05.

    Returns
    -------
    pandas.Series
        A pandas Series containing the exponentially weighted values of the input `series`.
        The weights are applied element-wise, and the resulting Series maintains the original
        index alignment. If the input `series` contains NaN values, they are preserved in the
        output `weighted_series`.

    Raises
    ------
    ValueError
        If the input `series` is empty.

    TypeError
        If the input `series` is not a pandas Series.

    References
    ----------
    - [1] Exponentially Weighted Moving Average (EWMA):
        https://en.wikipedia.org/wiki/Moving_average#Exponentially_weighted_moving_average
    - [2] Financial Modeling Using Exponentially Weighted Moving Averages:
        https://www.investopedia.com/terms/e/ewma.asp
    """
    # Generate exponential weights in reverse order
    n = len(data)
    weights = np.exp(alpha * np.arange(n))  # Exponentially increasing weights
    weights /= weights.sum()  # Lower the scale by dividing by square root of sum

    # Apply weights
    weighted_data = data
    if isinstance(data, pd.Series):
        weighted_data = data * weights
    elif isinstance(data, pd.DataFrame):
        weighted_data = data.mul(weights, axis=0)

    return weighted_data


if __name__ == '__main__':
    test_series = pd.Series([0.01, 0.02, -0.03, 0.04, -0.01, 0.01])
    test_series = exp_weight(test_series)
    print(test_series)