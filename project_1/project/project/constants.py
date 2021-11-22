'''
Creator: Muhammad Hassaan Rafique
Date: 19-11-2021
'''


def get_cat_cols_list():
    '''
    returns the categorical list of columns for
    encoder_helper in churn_library.

    input: None.
    output: list of strings.
    '''

    return [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]


def get_eda_cols_list():
    '''
    returns a list of columns required
    for perform_eda function.

    input: none.
    output: list of strings.
    '''

    return ['Churn', 'Customer_Age', 'Marital_Status',
            'Total_Trans_Ct', 'corr']


def get_feature_engineering_cols_list():
    '''
    returns a list of columns required on which
    we run the feature_engineering function.

    input: None.
    output: list of strings
    '''

    return ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn']
