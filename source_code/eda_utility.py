# Data processing
import numpy as np
import pandas as pd

# Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy.stats import skew


def count_columns_by_dtype(df):
    """
    Count the number of columns in the input DataFrame grouped by their data type.

    Args:
        df (pd.DataFrame): The input DataFrame with columns of various data types.

    Returns:
        pd.DataFrame: A DataFrame containing the data types and their corresponding counts in the input DataFrame.
    """
    # Group the columns in the DataFrame by their data type and count the columns in each group
    col_counts = df.dtypes.groupby(df.dtypes).count()

    # Create a DataFrame from the column counts
    col_counts_df = pd.DataFrame({'Data Type': col_counts.index.astype(str), 'Count': col_counts.values})

    return col_counts_df


def summarize_numeric_columns(df):
    """
    Create a summary DataFrame for numeric columns in the input DataFrame.
    
    The summary includes the column type, count of non-NaN values, count of NaN values, count of zero values,
    mean, standard deviation, minimum, 25th percentile, median, 75th percentile, maximum, number of lower outliers,
    number of upper outliers, and skewness.

    Args:
        df (pd.DataFrame): The input DataFrame with numeric columns to summarize.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each numeric column.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64']).columns

    # Create a list to store the results for each numeric column
    results_list = []

    # Iterate over each numeric column in the DataFrame
    for col in numeric_df:
        col_dtype = str(df[col].dtype)

        # Determine the types of values in the column
        non_nan_count = df[col].count()
        nan_count = df[col].isna().sum()
        zero_count = len(df[df[col] == 0])

        # If there are non-NaN values in the column, calculate the statistics
        if non_nan_count > 0:
            non_nan_values = df[col].dropna()
            mean = non_nan_values.mean()
            std = non_nan_values.std()
            minimum = non_nan_values.min()
            percentile25 = np.percentile(non_nan_values, 25)
            median = np.percentile(non_nan_values, 50)
            percentile75 = np.percentile(non_nan_values, 75)
            maximum = non_nan_values.max()
            num_upper_outliers = len(df[df[col] > percentile75 + 1.5*(percentile75-percentile25)])
            num_lower_outliers = len(df[df[col] < percentile25 - 1.5*(percentile75-percentile25)])
            skewness = skew(non_nan_values)

        # Add the results to the results list for this column
        results_list.append([col, col_dtype, non_nan_count, nan_count, zero_count, mean, std, minimum, percentile25, median, percentile75, maximum, num_lower_outliers, num_upper_outliers, skewness])

    # Create the result DataFrame from the list of results
    numeric_summary_df = pd.DataFrame(results_list, columns=["Column", "Column Type", "Non-NaN Count", "NaN Count", "Zero Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max", "Num Lower Outliers", "Num Upper Outliers", "Skew"])
    return numeric_summary_df


def summarize_categorical_columns(df):
    """
    Create a summary DataFrame for categorical columns in the input DataFrame.
    
    The summary includes the column type, count of non-None values, count of None values, count of empty values,
    number of unique values, mode, and mode occurrences.

    Args:
        df (pd.DataFrame): The input DataFrame with categorical columns to summarize.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each categorical column.
    """
    # Select the columns of interest
    cols_of_interest = df.select_dtypes(include=['O', 'bool', 'category']).columns

    # Create a list to store the results for each column
    results_list = []

    # Iterate over each column of interest
    for col in cols_of_interest:
        # Determine the counts of non-None, None, and empty values in the column
        non_none_count = df[col].count()
        none_count = df[col].isna().sum()
        empty_count = len(df[df[col] == ""])
        
        # Calculate the number of unique values and the mode of the column
        num_unique = len(df[col].unique())
        mode_value = df[col].mode().iloc[0]
        mode_count = df[col].value_counts()[mode_value]
        
        # Add the results to the results list for this column
        results_list.append([col, str(df[col].dtype), non_none_count, none_count, empty_count, num_unique, mode_value, mode_count])

    # Create the result DataFrame from the list of results
    categorical_summary_df = pd.DataFrame(results_list, columns=["Column", "Column Type", "Non-None Count", "None Count", "Empty Count", "Num Unique", "Mode", "Mode Occurrences"])

    return categorical_summary_df



def plot_histograms_and_boxplots(df):
    """
    Plot histograms with KDE curves and mean lines, as well as horizontal boxplots for numeric columns in a DataFrame.

    The function iterates through the int64 and float64 columns in the input DataFrame, and for each column, it creates
    a histogram with a Kernel Density Estimation (KDE) curve and a mean line. Additionally, it creates a horizontal boxplot.
    NaN values are excluded from the data before plotting.

    Args:
        df (pd.DataFrame): The input DataFrame with numeric columns to visualize.

    Returns:
        None
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in num_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        
        # Exclude NaN values from the column
        non_nan_data = df[col].dropna()
        
        # Plotting the histogram
        ax1.hist(non_nan_data, bins=20, color='blue', alpha=0.7, rwidth=0.85, density=True)
        ax1.set_title(f'Histogram of {col}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        
        # Plotting the KDE curve (distribution line)
        sns.kdeplot(non_nan_data, ax=ax1, color='red')
        
        # Plotting the mean line
        mean_value = np.mean(non_nan_data)
        ax1.axvline(mean_value, color='green', linestyle='--', label='Mean')
        ax1.legend()

        # Plotting the boxplot
        ax2.boxplot(non_nan_data, vert=False)
        ax2.set_title(f'Boxplot of {col}')
        ax2.set_xlabel('Value')

        plt.tight_layout()
        plt.show()


def convert_binary_to_boolean(df):
    """
    Convert int64 columns containing only 1 and 0 to boolean columns in the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing int64 columns.

    Returns:
        pd.DataFrame: A DataFrame with int64 columns containing only 1 and 0 converted to boolean columns.
    """
    int64_cols = df.select_dtypes(include=['int64']).columns

    for col in int64_cols:
        unique_values = df[col].unique()
        if set(unique_values) == {0, 1} or set(unique_values) == {0} or set(unique_values) == {1}:
            df[col] = df[col].astype(bool)

    return df
