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
    # download the latest version
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

    _save_plot(fig, pre_fix, dpi = dpi)
    

# test 
if __name__ == "__main__":
    train_data, test_data = read_data()
    
    #check the original distribution of them
    variable_names = train_data.columns.values[1:5]
    plot_box_and_hist(train_data, variable_names, dpi = 300, pre_fix="raw_train_data")
    plot_box_and_hist(test_data, variable_names, dpi = 300, pre_fix="raw_test_data")

    # save the original data as raw_train, raw_test
    store_path = "/Users/gufeng/2024_Fall/dasc6510/Final report for time series/data"
    for data, name in zip([train_data, test_data], ["train_data", "test_data"]):
        data.to_csv(store_path + f"/raw_{name}.csv")
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
    replace_outliers(test_data, "meanpressure")
    
    # store the processed data
    for data, name in zip([train_data, test_data], ["train_data", "test_data"]):
        data.to_csv(store_path + f"/{name}.csv")

    # check the distribution on the variables
    plot_box_and_hist(train_data, variable_names, pre_fix="processed_train_data_box_and_hist_plots", dpi = 300)
    plot_box_and_hist(test_data, variable_names, pre_fix="processed_test_data_box_and_hist_plots", dpi = 300)


    """
    I have tryed to finish the modeling work with Python, due to having not used 
    Python for modeling for nearly six months and the lack of time,
    i have to turn to R language to finish the following work, 
    which breaks the current consistent workflow.
    """
    
