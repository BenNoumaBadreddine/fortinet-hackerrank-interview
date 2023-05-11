import pandas as pd


def load_data(data_path: str = '../../data/Loan-Approval-Prediction.csv') -> pd.DataFrame:
    """
    This function import the data to be used in the project
    :param data_path: the path to the data file
    :return: a dataframe having the data
    """

    return pd.read_csv(data_path)
