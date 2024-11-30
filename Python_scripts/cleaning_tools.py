import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.ticker as ticker
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# The data cleaning module 

# Detect the outliers for one series in the original dataframe  
def detect_outliers_iqr(data, lower_quantil = 0.25, upper_quantil = 0.75):
    """
    Detects outliers in a pandas Series using the IQR method.
    
    Parameters:
        data (pd.Series): Input data series to check for outliers.
        
    Returns:
        pd.Series: Outliers detected in the data.
        tuple: Lower and upper bounds for the outliers.
    """
    # Calculate Q1, Q3, and IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return outliers, (lower_bound, upper_bound)


# Replace the wanted values with the new values, based on the common 
# indices they have
def replace_outliers(data, column, index, new_values):
    """
    Replace the values that laying in the given index with the 
    new values

    Args:
        data (dataframe): the original data
        column (df.column): the column in which the old values are to be replaced with
        index (array): the indeces for targeting the rows/elements in the column
        new_values (_type_): the new values used to replace the old values
    """
    # TO DO
    

# project the date into corresponding season for Delhi
def assign_season(date_series):
    """
    Function to assign a season to each date in a pandas datetime series.
    Seasons are based on the Indian climate.
    
    Args:
        date_series (pd.Series): A pandas series of datetime objects.

    Returns:
        pd.Series: A pandas series of season names corresponding to the input dates.
    """

# extract the season part from the date attribute
def assign_season(date_series):
    """
    Function to assign a season to each date in a pandas datetime series.
    Seasons are based on the Indian climate.
    
    Args:
        date_series (pd.Series): A pandas series of datetime objects.

    Returns:
        pd.Series: A pandas series of season names corresponding to the input dates.
    """
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:  # Winter: December, January, February
            return "Winter"
        elif month in [3, 4, 5]:  # Spring: March, April, May, June
            return "Spring"
        elif month in [6, 7, 8]:  # Summer: July, August, September
            return "Summer"
        elif month in [9, 10, 11]:  # Autumn: October, November
            return "Autumn"
        else:
            return None  # Fallback, though not expected

    return date_series.apply(get_season)



# doing tests on the series to check if it is stationary
# before transforming the original data, doing tests on y to check its stationary
def analyze_time_series(series, seasonal_period=60, alpha=0.05, lags=10):
    """
    Perform ADF, KPSS, number of differences, number of seasonal differences, 
    and Ljung-Box tests on a time series.
    
    Args:
        series (pd.Series): The time series data.
        seasonal_period (int): Seasonal period for the unitroot_nsdiffs.
        alpha (float): Significance level for stationarity tests.
        lags (int): Number of lags for the Ljung-Box test.
    
    Returns:
        pd.DataFrame: Results of the tests in a structured DataFrame.
    """
    def unitroot_ndiffs(series, alpha=alpha):
        diff_count = 0
        while adfuller(series, autolag='AIC')[1] > alpha:
            series = series.diff().dropna()
            diff_count += 1
        return diff_count

    def unitroot_nsdiffs(series, seasonal_period=seasonal_period, alpha=alpha):
        diff_count = 0
        while adfuller(series, autolag='AIC')[1] > alpha:
            series = series.diff(periods=seasonal_period).dropna()
            diff_count += 1
        return diff_count

    # Run ADF Test
    adf_result = adfuller(series, autolag='AIC')
    adf_p_value = adf_result[1]

    # Run KPSS Test
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_p_value = kpss_result[1]

    # Calculate number of differences
    ndiffs = unitroot_ndiffs(series)

    # Calculate number of seasonal differences
    nsdiffs = unitroot_nsdiffs(series)

    # Create a DataFrame summarizing the results
    summary = pd.DataFrame({
        "Variable": [series.name] * 4,
        "Test": ["ADF Test", "KPSS Test", "Unit Root Differences", "Seasonal Differences"],
        "P-Value": [adf_p_value, kpss_p_value, ":)", ":)"],
        "Test Statistic": [adf_result[0], kpss_result[0], ndiffs, nsdiffs]
    })

    return summary

# doing the stationary tests on a series of columns
def analyze_time_series_more(dataframe, cols):
    """Doing the stationary tests on a series of columns and return a 
    concated dataframe for easy checking

    Args:
        dataframe (pd.dataframe): the time series dataframe
        cols (sequence): the name of columns in the dataframe
    """
    # drop the rows where containing NaN values
    dataframe = dataframe.dropna()
    
    # doing the tests
    stationary_tests = [analyze_time_series(dataframe[col]) for col in cols]
    stationary_tests_all = pd.concat(stationary_tests, axis=0, ignore_index=True)

    return stationary_tests_all


# quickly check the 4 variables before and after the transformation
def _check_the_eight(dataframe, cols, plot_type = "line"):
    fig, axes = plt.subplots(
        4,2, figsize = (14, 9), sharex=True, dpi = 300
        )
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))  # Show only 6 ticks

    if plot_type == "line":
        for ax, col in zip(axes.flatten(), cols):
            sns.lineplot(data = train_data, x = "date", y = dataframe[col], ax = ax)
            ax.grid()
    elif plot_type == "ACF":
        for ax, col in zip(axes.flatten(), cols):
            custom_plot_acf(dataframe[col], ax = ax)
            ax.grid()
    elif plot_type == "PACF":
        for ax, col in zip(axes.flatten(), cols):
            custom_plot_pacf(dataframe[col], ax = ax)
            ax.grid()

    return fig, ax

def custom_plot_acf(series, lags=10, ax=None):
    """
    Plot the ACF for a given time series using object-oriented matplotlib.

    Args:
        series (pd.Series or np.array): The time series data.
        lags (int): Number of lags to include in the plots.
        ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.

    Returns:
        ax: The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))  # Create new figure and axis if not provided

    # Plot ACF
    plot_acf(series, lags=lags, ax=ax)
    ax.set_ylim(-1.1, 1.1)  # Set y-axis limits
    ax.set_title(f"ACF of {series.name}")
    

    plt.tight_layout()
    return ax

def custom_plot_pacf(series, lags=10, ax=None):
    """
    Plot the PACF for a given time series using object-oriented matplotlib.

    Args:
        series (pd.Series or np.array): The time series data.
        lags (int): Number of lags to include in the plots.
        ax (matplotlib.axes.Axes, optional): Axis object to plot on. Defaults to None.

    Returns:
        ax: The matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))  # Create new figure and axis if not provided

    # Plot PACF.
    plot_pacf(series, lags=lags, ax=ax, method="ywm")
    ax.set_ylim(-1.1, 1.1)  # Set y-axis limits
    ax.set_title(f"PACF of {series.name}")
    
    plt.tight_layout()
    return ax

# test 
if __name__ == "__main__":
    data = pd.read_csv("group work/Yishu Liu/DailyDelhiClimateTrain.csv")
    # test the season generating function
    season = assign_season(data.date)
    print(season)
