from fortinet_hacker_rank_interview.python_question.src.loading_data import load_data
from fortinet_hacker_rank_interview.python_question.src.util import fill_empty_values_with_frequently_used, \
    fill_empty_values_with_median, ordinal_encoding_to_replace_categorical_features, feature_log_transformation, \
    one_hot_encoding_to_replace_categorical_features, drop_column_from_df


def data_preprocessing_workflow():

    # load the data
    data = load_data()

    # fill missing data
    missing_values_categorical_variable = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History",
                                           "Loan_Amount_Term"]
    for column in missing_values_categorical_variable:
        fill_empty_values_with_frequently_used(data, column)

    missing_values_numerical_variable = ["LoanAmount"]
    for column in missing_values_numerical_variable:
        fill_empty_values_with_median(data, column)

    # deal with categorical variables
    col_name = 'Education'
    mapper = {'Graduate': 1, 'Not Graduate': 2}
    mapper_2 = {'Y': 1, 'N': 0}
    target = 'Loan_Status'
    data = ordinal_encoding_to_replace_categorical_features(data, col_name, mapper)
    data = ordinal_encoding_to_replace_categorical_features(data, target, mapper_2)

    col_list = ['Gender', 'Married', 'Property_Area', 'Dependents', 'Self_Employed']
    data = one_hot_encoding_to_replace_categorical_features(data, col_list)

    # feature transformation: log transformation
    column_to_log_transform = ["LoanAmount"]
    for column in column_to_log_transform:
        data = feature_log_transformation(data, column)

    # features creation
    # 1- Total_income = applicant_income + coapplicant_income
    data["total_income"] = data["ApplicantIncome"]+data["CoapplicantIncome"]
    data = feature_log_transformation(data, "total_income")
    # 2- EMI = LoanAmount / Loan_Amount_Term
    data["EMI"] = data["LoanAmount"]/data["Loan_Amount_Term"]
    # 3- Balance_Income = TotalIncome - EMI*1000
    data["Balance_Income"] = data["total_income"]-data["EMI"]

    # data = data.drop(["Loan_ID"], axis=1)
    data = drop_column_from_df(data, "Loan_ID")
    return data
