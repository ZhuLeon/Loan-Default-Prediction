import os
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from keras.models import load_model
from keras.utils import to_categorical
from src.Loan_Model import loan_model
from sklearn.preprocessing import StandardScaler


def process_na(input_data):
    """
    Set a list of default values to replace NaNs with
    Values are based on the median if numerical and most frequent if categorical
    :param input_data:
    :return: Dataframe
    """
    na_dict = {'loan_amnt': 12000,
               'term': '36 months',
               'int_rate': '13.49%',
               'installment': 356.76,
               'grade': 'C',
               'emp_length': '5 years',
               'home_ownership': 'RENT',
               'annual_inc': 67000,
               'verification_status': 'Not Verified',
               'purpose': 'debt_consolidation',
               'dti': 17.85,
               'delinq_2yrs': 0,
               'earliest_cr_line': 'Jul-02',
               'open_acc': 11,
               'pub_rec': 0,
               'revol_bal': 10381.5,
               'revol_util': '45.5%',
               'total_acc': 23,
               'initial_list_status': 'w',
               'application_type': 'Individual',
               'mort_acc': 1,
               'pub_rec_bankruptcies': 0}

    missing_values = ((input_data.isna().sum()) / len(input_data.index)).sort_values(ascending=False)
    missing_values = list(missing_values[missing_values > 0].index)
    for item in missing_values:
        input_data[item].fillna(na_dict[item], inplace=True)

    return input_data


def process_input(input_data):
    # Transform Data
    input_data['log_annual_inc'] = input_data['annual_inc'].apply(lambda x: np.log(x + 1))
    input_data.drop('annual_inc', axis=1, inplace=True)

    input_data['delinq_2yrs'].values[input_data['delinq_2yrs'] > 1] = 2

    input_data['earliest_cr_line'] = input_data['earliest_cr_line'].apply(lambda s: int(s[-2:]))
    input_data.loc[input_data['earliest_cr_line'] > 20, 'earliest_cr_line'] += 1900
    input_data.loc[input_data['earliest_cr_line'] < 20, 'earliest_cr_line'] += 2000

    input_data['emp_length'].replace('< 1 year', 0, inplace=True)
    input_data['emp_length'].replace('1 year', 1, inplace=True)
    input_data['emp_length'].replace('2 years', 2, inplace=True)
    input_data['emp_length'].replace('3 years', 3, inplace=True)
    input_data['emp_length'].replace('4 years', 4, inplace=True)
    input_data['emp_length'].replace('5 years', 5, inplace=True)
    input_data['emp_length'].replace('6 years', 6, inplace=True)
    input_data['emp_length'].replace('7 years', 7, inplace=True)
    input_data['emp_length'].replace('8 years', 8, inplace=True)
    input_data['emp_length'].replace('9 years', 9, inplace=True)
    input_data['emp_length'].replace('10+ years', 10, inplace=True)

    input_data['home_ownership'].replace(['NONE', 'ANY'], 'MORTGAGE', inplace=True)

    input_data['log_installment'] = input_data['installment'].apply(lambda x: np.log(x + 1))
    input_data.drop('installment', axis=1, inplace=True)

    input_data['int_rate'] = input_data['int_rate'].str.rstrip('%').astype('float')

    input_data.loc[input_data['mort_acc'] > 9, 'mort_acc'] = 10

    input_data['pub_rec'] = pd.cut(input_data['pub_rec'], [0, 0.9, 25],
                                   labels=['None', 'At least one'], include_lowest=True)

    input_data['pub_rec_bankruptcies'] = pd.cut(input_data['pub_rec_bankruptcies'], [0, 0.9, 25],
                                                labels=['None', 'At least one'], include_lowest=True)

    input_data['log_revol_bal'] = input_data['revol_bal'].apply(lambda x: np.log(x + 1))
    input_data.drop('revol_bal', axis=1, inplace=True)

    input_data['revol_util'] = input_data['revol_util'].str.rstrip('%').astype('float')

    input_data['term'] = input_data['term'].apply(lambda s: np.int8(s.split()[0]))

    return input_data


def create_dummies(input_data):
    """
    What the function does and why
    :param input_data:
    :return:
    (optional) sample usage of function
    """
    # Create dummy variables for categorical variables
    # None of the numerical values have NA so we dont need a special one for that? (reference)
    input_data = pd.get_dummies(input_data, columns=['grade', 'home_ownership', 'verification_status', 'purpose',
                                                     'initial_list_status', 'application_type', 'pub_rec',
                                                     'pub_rec_bankruptcies'], drop_first=False)

    # Columns list should be equal to the columns the model uses
    # dummy_frame = pd.DataFrame(columns=['term', 'int_rate', 'emp_length', 'dti', 'delinq_2yrs',
    #                                     'earliest_cr_line', 'open_acc', 'revol_util', 'total_acc', 'mort_acc',
    #                                     'log_annual_inc', 'log_installment', 'log_revol_bal', 'grade_B',
    #                                     'grade_C', 'grade_D', 'home_ownership_OWN', 'home_ownership_RENT',
    #                                     'verification_status_Source Verified', 'verification_status_Verified',
    #                                     'purpose_house', 'purpose_small_business',
    #                                     'application_type_Joint App'])
    dummy_frame = pd.DataFrame(columns=['loan_amnt', 'term', 'int_rate', 'emp_length', 'dti', 'delinq_2yrs',
                                        'earliest_cr_line', 'open_acc', 'revol_util', 'total_acc', 'mort_acc',
                                        'log_annual_inc', 'log_installment', 'log_revol_bal', 'grade_B',
                                        'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
                                        'home_ownership_OWN', 'home_ownership_RENT',
                                        'verification_status_Source Verified', 'verification_status_Verified',
                                        'purpose_credit_card', 'purpose_debt_consolidation',
                                        'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
                                        'purpose_medical', 'purpose_moving', 'purpose_other',
                                        'purpose_renewable_energy', 'purpose_small_business',
                                        'purpose_vacation', 'initial_list_status_w',
                                        'application_type_Joint App', 'pub_rec_At least one',
                                        'pub_rec_bankruptcies_At least one'])
    input_data = input_data.reindex(columns=dummy_frame.columns, fill_value=0)

    return input_data


def print_prediction(input_data, prod_model, output):
    output = pd.DataFrame(output, columns=['Prediction'])
    output['Prediction'].replace(0, 'Fully Paid', inplace=True)
    output['Prediction'].replace(1, 'Default', inplace=True)
    print(output)
    p_output = pd.DataFrame(prod_model.predict_proba(input_data))
    p_output.columns = ['Fully Paid', 'Default']
    p_output = p_output.multiply(100).round(0).astype(int).astype(str) + '%'
    print(p_output)


def main():
    model_prod_path = Path('../data/model_nn.h5')
    if not os.path.exists(model_prod_path):
        print('Model not found. Building model...')
        loan_model()
        print('Build complete.')
    # Create input DataFrame
    input_data = pd.read_csv(Path('../data/input.csv'), header=0)
    input_data['loan_status'].replace('Charged Off', 'Default', inplace=True)
    keep_list = ['annual_inc', 'application_type', 'dti', 'delinq_2yrs', 'earliest_cr_line',
                 'emp_length', 'home_ownership', 'initial_list_status', 'installment', 'int_rate',
                 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec',
                 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'grade',
                 'term', 'total_acc', 'verification_status']
    drop_list = [col for col in input_data.columns if col not in keep_list]
    input_data.drop(labels=drop_list, axis=1, inplace=True)
    if np.sort(input_data.columns.values) != np.sort(keep_list):
        raise Exception('Columns not correct. Please recheck data')
    input_data.to_csv(Path('../data/input_clean.csv'), index=False)
    test = input_data.loc[:, input_data.columns == 'loan_status']
    test['loan_status'].replace('Default', 1, inplace=True)
    input_data = process_na(input_data)
    input_data = process_input(input_data)
    input_data = create_dummies(input_data)
    # Load the Model
    scaler = load('..\\data\\scaler.joblib')
    input_data = scaler.transform(input_data.astype('float64'))
    prod_model = load_model('..\\data\\model_nn.h5')
    output = prod_model.predict_classes(input_data)
    print_prediction(input_data, prod_model, output)

    score = prod_model.evaluate(input_data, to_categorical(test))
    print("Accuracy: %.2f%%" % (score[1]*100))


if __name__ == "__main__":
    main()
