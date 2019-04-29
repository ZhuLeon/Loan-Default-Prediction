from src.main import process_input
from src.main import create_dummies
import pandas as pd
from joblib import load


def test_prediction_default():
    """
    Unit test to showcase functionality of handling missing values
    """
    test_data = pd.DataFrame(columns=['loan_amnt', 'term', 'int_rate', 'installment',
                                      'grade', 'emp_length', 'home_ownership', 'annual_inc',
                                      'verification_status', 'purpose', 'dti', 'delinq_2yrs',
                                      'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
                                      'revol_util', 'total_acc', 'initial_list_status',
                                      'application_type', 'mort_acc', 'pub_rec_bankruptcies'])

    test_data.loc[0] = pd.Series({'loan_amnt': 25000,
                                  'term': '60 months',
                                  'int_rate': '12.00%',
                                  'installment': 600,
                                  'grade': 'C',
                                  'emp_length': '< 1 year',
                                  'home_ownership': 'RENT',
                                  'annual_inc': 45000,
                                  'verification_status': 'Not Verified',
                                  'purpose': 'small_business',
                                  'dti': 14.00,
                                  'delinq_2yrs': 2,
                                  'earliest_cr_line': 'Jul-06',
                                  'open_acc': 10,
                                  'pub_rec': 1,
                                  'revol_bal': 18000,
                                  'revol_util': '70%',
                                  'total_acc': 15,
                                  'initial_list_status': 'w',
                                  'application_type': 'Individual',
                                  'mort_acc': 0,
                                  'pub_rec_bankruptcies': 1})

    test_data = process_input(test_data)
    test_data = create_dummies(test_data)
    prod_model = load('..\\..\\data\\ensemble_model.joblib')
    output = prod_model.predict(test_data)

    assert output[0] == 1
