import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fortinet_hacker_rank_interview.python_question.src.config import PrintStyle, COLORS


def data_type_discovery(dataframe: pd.DataFrame):
    """
    This function discovers and prints the type of each feature
    :param dataframe: the data that is used
    :return:
    """
    columns = list(dataframe.columns)
    categorical_variables = []
    integer_variables = []
    decimal_variables = []
    for col in columns:
        if dataframe[col].dtypes in ['O']:
            categorical_variables.append(col)
        elif dataframe[col].dtypes in ['float64', 'float32']:
            decimal_variables.append(col)
        elif dataframe[col].dtypes in ['int64', 'int32']:
            integer_variables.append(col)
        else:
            print(f"Warning: data type not handled: {dataframe[col].dtypes}")

    print(f"Features of Object type are: {categorical_variables}. Those features are called categorical variables")
    print(f"Features of Integer type are: {integer_variables}.")
    print(f"Features of Decimal type are: {decimal_variables}. Those variables are numerical values")


def categorical_variable_analysis(dataframe: pd.DataFrame, cat_var_list: list):
    """
    This function generates a report of analysis of each categorical variable in the provided list
    :param dataframe: The original data to be analyzed
    :param cat_var_list: The list of categorical variables
    :return:
    """
    for cat_var in cat_var_list:
        print(f"{PrintStyle.BOLD}Analysis on '{cat_var}' variable:{PrintStyle.END}")
        print(f"1- Size of {cat_var} variable is : {dataframe[cat_var].count()}")
        print(f"2- Detailed unique values of {cat_var} variable is : {dataframe[cat_var].value_counts()}")
        print(
            f"3- Frequency of each unique value of {cat_var} variable is : {dataframe[cat_var].value_counts(normalize=True) * 100}")
        print(f"4- Bar plot of {cat_var} variable is :")
        dataframe[cat_var].value_counts(normalize=True).plot.bar(title=cat_var, color=COLORS[:])
        plt.show()


def numerical_variable_analysis(dataframe: pd.DataFrame, num_var_list: list):
    """
    This function generates a report of analysis of each categorical variable in the provided list
    :param dataframe: The original data to be analyzed
    :param num_var_list: The list of numerical variables
    :return:
    """
    for num_var in num_var_list:
        print(f"{PrintStyle.BOLD}Analysis on '{num_var}' variable:{PrintStyle.END}")
        plt.figure(1)
        plt.subplot(121)
        sns.histplot(dataframe[num_var]);
        plt.subplot(122)
        dataframe[num_var].plot.box(figsize=(16, 5))
        plt.show()


def bivariate_analysis(dataframe: pd.DataFrame, cat_var_list: list, target: str):
    """
    This function generates a report of analysis of each categorical variable in the provided list
    :param target: The target column
    :param dataframe: The original data to be analyzed
    :param cat_var_list: The list of categorical variables
    :return:
    """
    for cat_var in cat_var_list:
        print(f"{PrintStyle.BOLD}Bi-variate Analysis on '{cat_var}' variable versus the target '{target}':"
              f"{PrintStyle.END}")
        relationship = pd.crosstab(dataframe[cat_var], dataframe[target])
        relationship.div(relationship.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4))
        plt.show()


def define_bins_and_groups(dataframe: pd.DataFrame, variable: str) -> tuple:
    if variable == 'ApplicantIncome':
        bins = [0, 2500, 4000, 6000, 81000]
        group = ['Low', 'Average', 'High', 'Very high']
    elif variable == 'CoapplicantIncome':
        bins = [0, 1000, 3000, 42000]
        group = ['Low', 'Average', 'High']
    elif variable == 'Total_Income':
        bins = [0, 2500, 4000, 6000, 81000]
        group = ['Low', 'Average', 'High', 'Very high']
    elif variable == 'LoanAmount':
        bins = [0, 100, 200, 700]
        group = ['Low', 'Average', 'High']
    else:
        print(f"Warning: the variable is not found {variable}")
        min_val = dataframe[variable].min()
        average_val = dataframe[variable].mean()
        max_val = dataframe[variable].max()

        bins = [0, min_val, average_val, max_val]
        group = ['Low', 'Average', 'High']
    return bins, group


def bins_plot_bivariate_between_num_and_target_variable(dataframe: pd.DataFrame, variable: str, target):
    bins, group = define_bins_and_groups(dataframe, variable)
    dataframe[f'{variable}_bin'] = pd.cut(dataframe[variable], bins, labels=group)
    variable_bin = pd.crosstab(dataframe[f'{variable}_bin'], dataframe[target])
    variable_bin.div(variable_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
    plt.xlabel(variable)
    P = plt.ylabel('Percentage')


def fill_empty_values_with_frequently_used(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function fills the empty values in the given column of the given dataframe with the frequently used
    value. The mode of a set of values is the value that appears most often. It can be multiple values.
    :param dataframe: The data stored as a dataframe format
    :param column: The column name that we want to fill the empty values with the frequently used value
    :return: dataframe with no empty values in the given column
    """
    dataframe[column].fillna(dataframe[column].mode()[0], inplace=True)
    return dataframe


def fill_empty_values_with_median(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function fills the empty values in the given column of the given dataframe with the median value
    :param dataframe: The data stored as a dataframe format
    :param column: The column name that we want to fill the empty values with the median value
    :return: dataframe with no empty values in the given column
    """
    dataframe[column].fillna(dataframe[column].median(), inplace=True)
    return dataframe


def feature_log_transformation(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This function will create a new feature based on the given column.This feature is simply the log transformation
    of the given data in the given column
    :param dataframe: The data stored as a dataframe format
    :param column:  The column name that we want to make a log transformation
    :return: dataframe with new feature
    """
    dataframe[column+'_log'] = np.log(dataframe[column])
    return dataframe


def drop_column_from_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Drops the column col_name from the dataframe
    :param df: The dataframe having the data
    :param col_name: The column col_name that will be removed from the dataframe
    :return: dataframe without the column col_name
    """
    df.drop(col_name, axis=1, inplace=True)
    return df


def ordinal_encoding_to_replace_categorical_features(df: pd.DataFrame, col_name: str, mapper: dict) -> pd.DataFrame:
    """
    Assigns numeric values to the col_name (ordinal variable). This technique is used for categorical variables where
    order matters
    :param df: The dataframe having the data
    :param col_name: The column col_name that will be filled by numeric values
    :param mapper: e.g mapper = {'very low': 1, 'medium low': 2, 'low': 3,
            'medium high': 4, 'high': 5, 'very high': 6,
            'unknown': 7}
    :return: dataframe with col_name having numeric values
    """
    # we can use : df[col_name].cat.codes
    df[col_name].replace(mapper, inplace=True)
    return df


def one_hot_encoding_to_replace_categorical_features(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """
    This function replace the categorical column or feature using the one hot encoding. One-Hot Encoding is the process
     of creating dummy variables. This technique is used for categorical variables where order does not matter.
     After creating the dummy columns, we drop one of the newly created columns produced by one-hot encoding
    :param df: The dataframe having the data
    :param col_list: The list of categorical columns
    :return:df: the dataframe with numeric column
    """
    for col in col_list:
        df = pd.get_dummies(data=df, columns=[col])
        # delete one of the created column, e.g. the latest column
        # The Dummy Variable Trap occurs when different  input variables perfectly predict each other –
        # leading to multi-collinearity.
        drop_column_from_df(df, df.columns[-1])
    return df

