# The data processing module
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns 
import kagglehub
import time
from functools import wraps
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.ticker as ticker
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import OneHotEncoder


def read_data():
    """Read the data from Kaggle into Python environment

    Returns:
        tuple: (traindata, testdata)
    """
    # download latest version
    path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
    file_names = os.listdir(path)
    file_paths = [os.path.join(path, file_name) for file_name in file_names]
    test_data = pd.read_csv(file_paths[0])
    train_data = pd.read_csv(file_paths[1])
    return train_data, test_data

def check_missing_data(dataframe):
    """Check if there are any missing data in the dataframe

    Args:
        dataframe (pd.dataframe): data

    Returns:
        array: number of missing data, 0 means no missing values
    """
    missing_count = dataframe.isna().sum(axis = 0)
    print(missing_count)

def convert_datetime():
    """Convert the Date column in train and test data into datetime data type
    """
    for data in [train_data, test_data]:
        data["date"] = pd.to_datetime(data["date"])

    print(train_data.info())
    print("***" * 10)
    print(test_data.info()) 
    
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

def create_dummy_seasons(dataframe):
    """Create dummy varaibles for seasons

    Returns:
        dataframe: the df having 3 dummy variables
    """
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform the data
    encoded_data = encoder.fit_transform(dataframe[['season']])

    # Take out the dummy variale names
    dummy_varialbes = encoder.get_feature_names_out()

    # Assign the values into the train data
    for i, col in enumerate(dummy_varialbes[:-1]):
        dataframe[col] = encoded_data[:, i]
    

# Detect the outliers for one series in the original dataframe  
def detect_outliers_iqr(data, lower_quantil = 0.05, upper_quantil = 0.95):
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


# Replace the wanted values with the new values, based on the common indices they have
def replace_outliers(data, column, outlier = None, moving_window = 120):
    """
    Replace the values that laying in the given index with the 
    moving average values

    Args:
        data (dataframe): the original data
        column (df.column): the column in which the old values are to be replaced with
        outlier (series): the series object from detect_outlier_iqr function
        new_values (_type_): the new values used to replace the old values
    """
    # create the default outlier
    outlier = detect_outliers_iqr(data[column])
    
    # replace the extreme values with the moving average value on that place
    new_values = data[column].rolling(window = 120).mean()

    # indices for outliers
    outlier_index = outlier[0].index.values
    # check the outliers
    print(f"The outliers: \n {data[column][outlier_index][:5]}")
    # replace the outliers
    data[column][outlier_index] = new_values.iloc[outlier_index]
    # check the modified data
    print(f"The new values: \n {data.iloc[outlier_index][column][:5]}") # now, it looks normal in the meanpressure variable
    
    


document_image_path = "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/documents/images"

def _save_plot(fig, pre_fix, director_path = document_image_path, dpi = 100):
    """Save the current fig object from matplotlib

    Args:
        director_path (string): _description_
        pre_fix (string): _description_
    """
    if pre_fix != None:
        fig.savefig(f"{director_path}/{pre_fix}.png", dpi = dpi)
    else:
        fig.savefig(f"{director_path}/_nan_.png", dpi = dpi)
        
    
def plot_box_and_hist(data, columns, figsize=(16, 8), dpi = 100, bins=15, pre_fix = None):
    """
    Creates box plots and histograms for specified columns in a DataFrame.
    
    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        columns (list): List of column names to plot.
        figsize (tuple): Size of the overall figure (default is (16, 8)).
        bins (int): Number of bins for the histograms (default is 15).
        pre_fix: the prefix used to name the saved plot file
    """
    # Number of columns to plot
    num_cols = len(columns)
    
    # Create a figure with 2 rows and n columns
    fig, axes = plt.subplots(2, num_cols, figsize=figsize)  # 2 rows, num_cols columns
    fig.tight_layout(pad=4)

    # Loop through the columns and create box plots and histograms
    for i, col in enumerate(columns):
        # Box Plot (Row 1)
        box_ax = axes[0, i]
        box_ax.boxplot(data[col].dropna())  # Drop NaN values to avoid errors
        box_ax.set_title(f"Box Plot: {col}")
        box_ax.set_ylabel("Values")
        
        # Histogram (Row 2)
        hist_ax = axes[1, i]
        hist_ax.hist(data[col].dropna(), bins=bins, color="skyblue", edgecolor="black")
        hist_ax.set_title(f"Histogram: {col}")
        hist_ax.set_xlabel("Values")
        hist_ax.set_ylabel("Frequency")

    # Show the plots
    plt.show()
    _save_plot(fig, pre_fix, dpi = dpi)
    

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
    train_data, test_data = read_data()
    missing_count = check_missing_data(train_data)
    print(missing_count)
    convert_datetime()
    for data in [train_data, test_data]:
        data["season"] = assign_season(data["date"])
        create_dummy_seasons(data)
        print(data.head())
    # add the row_number here as variable - "t"
    train_data["time"] = train_data.index.values + 1 # starts from 1
    print(train_data["time"])
    test_data["time"] = test_data.index.values + 1 + 1462 # starts from the last day in train data
    print(test_data["time"])
    
    # replace outliers
    replace_outliers(train_data, "meanpressure")
    
    # check the distribution on the variables
    variable_names = train_data.columns.values[1:5]
    plot_box_and_hist(train_data, variable_names, pre_fix="box and hist plots", dpi = 300)
    
    
