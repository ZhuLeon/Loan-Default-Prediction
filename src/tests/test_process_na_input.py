from src.main import process_na
import pandas as pd
import numpy as np


def test_process_na_input_contains_nan():
    """
    Unit test to showcase functionality of handling missing values
    """
    test_data = pd.DataFrame(columns=['loan_amnt', 'term', 'int_rate', 'installment',
                                      'grade', 'emp_length', 'home_ownership', 'annual_inc',
                                      'verification_status', 'purpose', 'dti', 'delinq_2yrs',
                                      'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_bal',
                                      'revol_util', 'total_acc', 'initial_list_status',
                                      'application_type', 'mort_acc', 'pub_rec_bankruptcies'])

    test_data.loc[0] = pd.Series({'loan_amnt': np.nan,
                                  'term': np.nan,
                                  'int_rate': np.nan,
                                  'installment': np.nan,
                                  'grade': np.nan,
                                  'emp_length': np.nan,
                                  'home_ownership': np.nan,
                                  'annual_inc': np.nan,
                                  'verification_status': np.nan,
                                  'purpose': np.nan,
                                  'dti': np.nan,
                                  'delinq_2yrs': np.nan,
                                  'earliest_cr_line': np.nan,
                                  'open_acc': np.nan,
                                  'pub_rec': np.nan,
                                  'revol_bal': np.nan,
                                  'revol_util': np.nan,
                                  'total_acc': np.nan,
                                  'initial_list_status': np.nan,
                                  'application_type': np.nan,
                                  'mort_acc': np.nan,
                                  'pub_rec_bankruptcies': np.nan})
    test_data = process_na(test_data)
    actual = test_data.isnull().values.any()
    expected = False
    
    assert actual == expected
