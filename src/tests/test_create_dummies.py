from src.main import process_input
from src.main import create_dummies
import pandas as pd


def test_create_dummies_columns():
    """
    Integration test to showcase functionality of creating dummy variables
    """
    test_data = pd.DataFrame(columns=['loan_amnt', 'term', 'int_rate', 'installment',
                                      'grade', 'emp_length', 'home_ownership', 'annual_inc',
                                      'verification_status', 'purpose', 'dti', 'delinq_2yrs',
                                      'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
                                      'revol_util', 'total_acc', 'initial_list_status',
                                      'application_type', 'mort_acc', 'pub_rec_bankruptcies'])

    test_data.loc[0] = pd.Series({'loan_amnt': 12000,
                                  'term': '36 months',
                                  'int_rate': '10.00%',
                                  'installment': 350,
                                  'grade': 'A',
                                  'emp_length': '< 1 year',
                                  'home_ownership': 'RENT',
                                  'annual_inc': 10000,
                                  'verification_status': 'Source Verified',
                                  'purpose': 'major_purchase',
                                  'dti': 15.23,
                                  'delinq_2yrs': 0,
                                  'earliest_cr_line': 'Jul-17',
                                  'open_acc': 3,
                                  'pub_rec': 0,
                                  'revol_bal': 1000,
                                  'revol_util': '50%',
                                  'total_acc': 1,
                                  'initial_list_status': 'w',
                                  'application_type': 'Individual',
                                  'mort_acc': 0,
                                  'pub_rec_bankruptcies': 1})

    test_data = process_input(test_data)
    test_data = create_dummies(test_data)
    actual = list(test_data.columns.values)
    expected = ['term', 'int_rate', 'emp_length', 'dti', 'delinq_2yrs',
                'earliest_cr_line', 'open_acc', 'revol_util', 'total_acc', 'mort_acc',
                'log_annual_inc', 'log_installment', 'log_revol_bal', 'grade_B',
                'grade_C', 'grade_D', 'home_ownership_OWN', 'home_ownership_RENT',
                'verification_status_Source Verified', 'verification_status_Verified',
                'purpose_house', 'purpose_small_business', 'application_type_Joint App']

    assert actual == expected
