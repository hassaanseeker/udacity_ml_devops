'''
Creator: Muhammad Hassaan Rafique
Date: 19-11-2021
'''

import shap
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from constants import get_cat_cols_list, get_eda_cols_list, get_feature_engineering_cols_list

class ChurnPredictor():
    '''
    class for doing eda, training and results for
    churn prediction.
    '''

    def __init__(self, pth):
        '''
        '''
        self.pth = pth

    def import_data(self):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''

        customer_df = pd.read_csv(self.pth, index_col=0)

        return customer_df

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''

        plt_cols_list = get_eda_cols_list()

        cust_df = self.import_data()
        cust_df['Churn'] = cust_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        for col in plt_cols_list:
            plt.figure(figsize=(20, 10))
            file_name = '_hist.png'

            if col == 'Total_Trans_Ct':
                sns.distplot(cust_df[col])
                file_name = '_dist.png'
            elif col == 'Marital_Status':
                cust_df[col].value_counts('normalize').plot(kind='bar')
            elif col == 'corr':
                file_name = '_heatmap.png'
                sns.heatmap(
                    cust_df.corr(),
                    annot=False,
                    cmap='Dark2_r',
                    linewidths=2)
            else:
                cust_df[col].hist()

            plt.savefig('./images/eda/' + col + file_name)

        return cust_df

    def encoder_helper(self):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe.
                category_lst: list of columns that contain categorical features.
                response: dataframe after categorical columns transformation with mean.

        output:
                df: pandas dataframe with new columns for
        '''

        cln_cust_df = self.perform_eda()
        category_lst = get_cat_cols_list()

        for col in category_lst:

            col_list = []
            col_groups = cln_cust_df.groupby(col).mean()['Churn']

            for val in cln_cust_df[col]:
                col_list.append(col_groups.loc[val])
            cln_cust_df[col + '_Churn'] = col_list
        print(cln_cust_df.columns)

        return cln_cust_df

    def perform_feature_engineering(self):
        '''
        function that splits the data into training and test with
        dependent and independent variables.

        input:
                df: pandas dataframe
                response: four pandas dataframe split into train and test.

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        keep_cols = get_feature_engineering_cols_list()

        trsfrmd_cust_df = self.encoder_helper()

        y_values = trsfrmd_cust_df['Churn']
        x_df = pd.DataFrame()

        x_df[keep_cols] = trsfrmd_cust_df[keep_cols]

        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y_values, test_size=0.3, random_state=42)

        return x_train, x_test, y_train, y_test

    def classification_report_image(self):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''

        def save_sns_img(clf_report, image_name):

            plt.figure(figsize=(20, 10))

            sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
            plt.savefig('./images/results/' + image_name)

        train_results_dict = self.train_models()

        # scores
        save_sns_img(
            classification_report(
                train_results_dict['y_test'],
                train_results_dict['y_test_preds_rf'],
                output_dict=True),
            'clf_report_test_rf.png')
        save_sns_img(
            classification_report(
                train_results_dict['y_train'],
                train_results_dict['y_train_preds_rf'],
                output_dict=True),
            'clf_report_train_rf.png')
        save_sns_img(
            classification_report(
                train_results_dict['y_test'],
                train_results_dict['y_test_preds_lr'],
                output_dict=True),
            'clf_report_test_lr.png')
        save_sns_img(
            classification_report(
                train_results_dict['y_train'],
                train_results_dict['y_train_preds_lr'],
                output_dict=True),
            'clf_report_train_lr.png')

    def feature_importance_plot(self, model, x_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)
        plt.figure(figsize=(20, 10))
        shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
        plt.savefig(output_pth)

    def save_roc_plot(self, x_test, y_test, *args):
        '''
        creates and stores the roc curve plot in images results folder.
        input:
                x_test: pandas dataframe test dataset for independent variable.
                y_test: an array of test dataset for dependent variable.
                args[0]: best model.
                args[1]: image name.
        output:
                None
        '''

        plt.figure(figsize=(15, 8))
        rfc_disp = plot_roc_curve(args[0], x_test, y_test, ax=plt.gca(), alpha=0.8)
        rfc_disp.plot(ax=plt.gca(), alpha=0.8)
        plt.savefig('./images/results/' + args[1])
        print('All results saved.')

    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''

        x_train, x_test, y_train, y_test = self.perform_feature_engineering()

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        self.feature_importance_plot(
            cv_rfc.best_estimator_,
            x_data=x_train,
            output_pth='./images/results/feature_importance.png')
        self.save_roc_plot(
            x_test,
            y_test,
            cv_rfc.best_estimator_,
            'roc_plot.png')

        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

        return {'y_train': y_train,
                'y_test': y_test,
                'y_train_preds_lr': y_train_preds_lr,
                'y_test_preds_lr': y_test_preds_lr,
                'y_train_preds_rf': y_train_preds_rf,
                'y_test_preds_rf': y_test_preds_rf,
                'best_estimator': cv_rfc.best_estimator_
                }
