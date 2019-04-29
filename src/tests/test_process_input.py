from src.main import process_input
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np


def test_process_input_calculation():
    """
    Unit test to showcase functionality of data transformation
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
    test_data['pub_rec'] = test_data['pub_rec'].astype('object')
    test_data['pub_rec_bankruptcies'] = test_data['pub_rec_bankruptcies'].astype('object')

    expected = pd.DataFrame().reindex_like(test_data)
    expected.loc[0] = [25000, 60, 12.00, 'C', 0, 'RENT', 'Not Verified', 'small_business',
                       14.0, 2, 2006, 10, 'At least one', 70.0, 15, 'w', 'Individual', 0,
                       'At least one', np.log(45001), np.log(601), np.log(18001)]

    assert_frame_equal(test_data, expected, check_dtype=False)
