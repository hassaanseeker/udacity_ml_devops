from churn_library import ChurnPredictor

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

    
    

    churn_predictor = ChurnPredictor(pth=r"./data/bank_data.csv")

    # print(churn_predictor.import_data())
    churn_predictor.perform_eda()
    churn_predictor.classification_report_image()