'''
Creator: Muhammad Hassaan Rafique
Date: 19-11-2021
'''

import logging
from churn_library import ChurnPredictor

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df_ = import_data
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_.shape[0] > 0
        assert df_.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda(perform_eda):
    """
        test perform eda function
    """
    try:
        cust_df = perform_eda
        assert cust_df.shape[0] > 0
        assert cust_df.shape[1] > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: The eda function cannot return data\
                    cannot be empty")
        raise err

    try:
        assert cust_df.shape[0] > 0
        assert cust_df.shape[1] > 0
    except ArithmeticError as err:
        logging.error(
            "Testing perform_eda: The file dosen't appear to have rows and columns"
        )
        raise err
    try:
        assert "Churn" in cust_df.columns
    except AssertionError as err:
        logging.error(
            "Testing perform_ed: The data dosen't appeat to have the churn column"
        )
        raise err


def test_encoder_helper(encoder_helper):
    """
    test encoder helper
    """
    try:
        cust_df = encoder_helper
        assert cust_df.shape[0] > 0
        assert cust_df.shape[1] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns."
        )
        raise err

    try:
        assert set(
            [
                "Gender_Churn",
                "Education_Level_Churn",
                "Marital_Status_Churn",
                "Income_Category_Churn",
                "Card_Category_Churn",
            ]
        ) <= set(cust_df.columns)

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The data dosen'nt appear to \
                have the different churn columns created."
        )
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    test perform_feature_engineering
    """
    try:
        cust_df = perform_feature_engineering
        assert cust_df.shape[0] > 0
        assert cust_df.shape[1] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The file doesn't appear to have rows and columns."
        )
        raise err


def test_train_models(train_models):
    """
    test train_models
    """
    try:
        model_data_dict = train_models
        assert len(model_data_dict) == 7
        for _, arr_ in enumerate(model_data_dict):
            assert len(arr_) > 0
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models: The number of dataframes returned should be 7 and all of these\
            should not be empty"
        )
        raise err


if __name__ == "__main__":

    churn_predictor = ChurnPredictor("./data/bank_data.csv")
    test_import(churn_predictor.import_data())
    test_eda(churn_predictor.perform_eda())
    test_encoder_helper(churn_predictor.encoder_helper())
    test_train_models(churn_predictor.train_models())
