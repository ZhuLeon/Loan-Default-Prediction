from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, auc, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import load_model
from xgboost import XGBClassifier
from xgboost import plot_importance


def plot_var(col_name, title, continuous, dataset, x1limit=False, x2limit=False, x1l=0, x1u=0, x2l=0, x2u=0):
    """
    Plot a variable against the response variable loan status
    - col_name is the variable name in the dataframe
    - title is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    - dataset is the pandas dataframe containing the dataset
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), dpi=90)

    # Plot without loan status
    if continuous:
        sns.distplot(dataset.loc[dataset[col_name].notnull(), col_name], kde=False, ax=ax1)
    else:
        sns.countplot(dataset[col_name], order=sorted(dataset[col_name].unique()),
                      color='#4f81bd', saturation=1, ax=ax1)
    if x1limit:
        ax1.set_xlim([x1l, x1u])
    ax1.set_xlabel(title)
    ax1.set_ylabel('Frequency')
    ax1.set_title(title)

    # Plot with loan status
    if continuous:
        sns.boxplot(x=col_name, y='loan_status', data=dataset, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(title + ' by Loan Status')
    else:
        charge_off_rates = dataset.groupby(col_name)['loan_status'].value_counts(normalize=True).loc[:, 'Default']
        sns.barplot(x=charge_off_rates.index, y=charge_off_rates.values, color='#4f81bd', saturation=1, ax=ax2)
        ax2.set_ylabel('Fraction of Loans Default')
        ax2.set_title('Default Rate by ' + title)
    if x2limit:
        ax2.set_xlim([x2l, x2u])
    ax2.set_xlabel(title)

    plt.tight_layout()


def loan_model():
    # # Preliminary Data Analysis
    # read in the entire raw dataset
    dataset = pd.read_csv(Path('../data/2017Half_clean.csv'), header=0)
    # pd.set_option('display.max_columns', None)
    print(dataset.head())

    # dataset['loan_status'].replace('Charged Off', 'Default', inplace=True)
    # # Create a list of columns with missing values for reference later
    # missing_values = ((dataset.isna().sum()) / len(dataset.index)).sort_values(ascending=False)
    #
    # # Drop columns with missing value >= 15%
    # drop_list = sorted(list(missing_values[missing_values > 0.15].index))
    # len(drop_list)
    # dataset.drop(labels=drop_list, axis=1, inplace=True)
    #
    # # drop columns that are irrelevant like date, or has too many unique values to be useful (job_title),
    # # or does not have any information (out_prncp, pymnt_plan), or information after the loan has been given.
    # keep_list = ['annual_inc', 'application_type', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    #              'emp_length', 'home_ownership', 'initial_list_status', 'installment', 'int_rate',
    #              'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec',
    #              'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'grade',
    #              'term', 'total_acc', 'verification_status']
    # drop_list = [col for col in dataset.columns if col not in keep_list]
    # dataset.drop(labels=drop_list, axis=1, inplace=True)
    # dataset.to_csv(Path('../data/2017Half_clean.csv'), index=False)

    # Look at how many of each response variable we have
    print(dataset['loan_status'].value_counts(dropna=False))
    print(dataset['loan_status'].value_counts(normalize=True, dropna=False))
    print(dataset.shape)

    # Create a list of columns with missing values for reference later
    missing_values = ((dataset.isna().sum()) / len(dataset.index)).sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    print(missing_values*100)

    # Look at the data type of each column
    print(dataset.dtypes)

    # # Exploratory Data Analysis

    # ### Annual Income (Numerical)
    # The self-reported annual income provided by the borrower during registration
    print(dataset.groupby('loan_status')['annual_inc'].describe())
    plot_var('annual_inc', 'Annual Income $', continuous=True, dataset=dataset,
             x1limit=True, x2limit=True, x1l=-1, x1u=350000, x2l=-1, x2u=200000)
    # Because the plot is right skewed we take a log transform of the annual income
    dataset['log_annual_inc'] = dataset['annual_inc'].apply(lambda x: np.log(x + 1))
    dataset.drop('annual_inc', axis=1, inplace=True)
    print(dataset.groupby('loan_status')['log_annual_inc'].describe())
    plot_var('log_annual_inc', 'Log Annual Income', continuous=True, dataset=dataset,
             x1limit=True, x2limit=True, x1l=9, x1u=14, x2l=9, x2u=14)
    # It seems that those with higher annual income are less risk

    # ### Application Type (Categorical)
    # Indicates whether the loan is an individual application or a joint application with two co-borrowers
    print(dataset.groupby('loan_status')['application_type'].value_counts(dropna=False))
    plot_var('application_type', 'Application Type', continuous=False, dataset=dataset)
    # There does not seem to be a strong correlation between risk and application_type.
    # Both risk around the same percentage.

    # ### Debt to Income Ratio (Numerical)
    # A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding
    # mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income
    # dummy = dataset.loc[dataset['dti'].notnull() & (dataset['dti']<60), 'dti']
    print(dataset.groupby('loan_status')['dti'].describe())
    plot_var('dti', 'Debt To Income Ratio', continuous=True, dataset=dataset,
             x1limit=True, x2limit=True, x1l=0, x1u=40, x2l=0, x2u=60)
    # sns.distplot(dataset.loc[dataset['dti'].notnull() & (dataset['dti']<60), 'dti'], kde=False)
    # plt.xlabel('Debt-to-income Ratio')
    # plt.ylabel('Count')
    # plt.title('Debt-to-income Ratio')
    # It seems that a higher debt to income ratio has a higher probability of charged off

    # ### Delinquency (Categorical)
    # The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
    dataset['delinq_2yrs'].values[dataset['delinq_2yrs'] > 1] = 2
    print(dataset.groupby('loan_status')['delinq_2yrs'].value_counts(dropna=False))
    plot_var('delinq_2yrs', 'Number of Delinquencies', continuous=False, dataset=dataset)
    # There seems to be an increasing trend in charge-off rate and increasing number of delinquencies but it is small.
    # Whether or not this is significant remains to be seen.

    # ### Earliest Credit Line (Numerical)
    # The month the borrower's earliest reported credit line was opened.
    # dataset['earliest_cr_line'] = pd.to_datetime(dataset['earliest_cr_line'], format='%b-%y').dt.year
    dataset['earliest_cr_line'] = dataset['earliest_cr_line'].apply(lambda s: int(s[-2:]))
    dataset.loc[dataset['earliest_cr_line'] > 20, 'earliest_cr_line'] += 1900
    dataset.loc[dataset['earliest_cr_line'] < 20, 'earliest_cr_line'] += 2000
    print(dataset.groupby('loan_status')['earliest_cr_line'].describe())
    plot_var('earliest_cr_line', 'Earliest Credit Line', continuous=True, dataset=dataset)
    # Surprisingly, when a person first obtained credit seems irrelevant.

    # ### Employment Length (Categorical)
    # Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means
    # ten or more years.
    dataset['emp_length'].replace('< 1 year', 0, inplace=True)
    dataset['emp_length'].replace('1 year', 1, inplace=True)
    dataset['emp_length'].replace('2 years', 2, inplace=True)
    dataset['emp_length'].replace('3 years', 3, inplace=True)
    dataset['emp_length'].replace('4 years', 4, inplace=True)
    dataset['emp_length'].replace('5 years', 5, inplace=True)
    dataset['emp_length'].replace('6 years', 6, inplace=True)
    dataset['emp_length'].replace('7 years', 7, inplace=True)
    dataset['emp_length'].replace('8 years', 8, inplace=True)
    dataset['emp_length'].replace('9 years', 9, inplace=True)
    dataset['emp_length'].replace('10+ years', 10, inplace=True)
    print(dataset.groupby('loan_status')['emp_length'].value_counts(dropna=False).sort_index())
    plot_var('emp_length', 'Employment Length by Year', continuous=False, dataset=dataset)
    # Doesn't seem to be much of a significant pattern here

    # ### Home Ownership (Categorical)
    # The home ownership status provided by the borrower during registration or obtained from the credit report.
    # dataset['home_ownership'].replace(['NONE', 'ANY'], 'NaN', inplace=True)
    dataset['home_ownership'].replace(['NONE', 'ANY'], 'MORTGAGE', inplace=True)
    print(dataset.groupby('loan_status')['home_ownership'].value_counts(dropna=False))
    plot_var('home_ownership', 'Home Ownership', continuous=False, dataset=dataset)
    # dataset['home_ownership'].replace('NaN', np.nan, inplace=True)
    # Interestingly those with a mortgage are more likely to pay off loans and those who rent are the least likely

    # ### Initial List Status (Categorical)
    # The initial listing status of the loan. Possible values are – W, F
    print(dataset.groupby('loan_status')['initial_list_status'].value_counts(dropna=False))
    plot_var('initial_list_status', 'Initial List Status', continuous=False, dataset=dataset)
    # Theres does not seem to be much information gained from Initial List status

    # ### Installment (Numerical)
    # The monthly payment owed by the borrower if the loan originates.
    print(dataset.groupby('loan_status')['installment'].describe())
    plot_var('installment', 'Installment', continuous=True, dataset=dataset)
    # since the plot is left skewed, we take the log transformation
    dataset['log_installment'] = dataset['installment'].apply(lambda x: np.log(x + 1))
    dataset.drop('installment', axis=1, inplace=True)
    print(dataset.groupby('loan_status')['log_installment'].describe())
    plot_var('log_installment', 'Log Installment', continuous=True, dataset=dataset)
    # It seems those with higher installments are more likely to be charged off

    # ### Interest Rate (Numerical)
    # Interest Rate on the loan
    dataset['int_rate'] = dataset['int_rate'].str.rstrip('%').astype('float')
    print(dataset.groupby('loan_status')['int_rate'].describe())
    plot_var('int_rate', 'Interest Rate', continuous=True, dataset=dataset)
    # There seems to be a much higher interest rate on average for loans that charge off

    # ### Loan Amount (Numerical)
    # The listed amount of the loan applied for by the borrower.
    # If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
    print(dataset.groupby('loan_status')['loan_amnt'].describe())
    plot_var('loan_amnt', 'Loan Amount', continuous=True, dataset=dataset)
    # It seems charged off loans have a higher loan amount

    # ### Mortgage Accounts (Numerical)
    # Number of mortgage accounts
    dataset.loc[dataset['mort_acc'] > 9, 'mort_acc'] = 10
    print(dataset.groupby('loan_status')['mort_acc'].value_counts(dropna=False))
    plot_var('mort_acc', 'Mortgage Accounts', continuous=True, dataset=dataset)
    # Currently there does not seem to be a significant difference

    # ### Open Account (Numerical)
    # The number of open credit lines in the borrower's credit file.
    print(dataset.groupby('loan_status')['open_acc'].describe())
    plot_var('open_acc', 'Open Credit Lines', continuous=True, dataset=dataset,
             x1limit=True, x2limit=True, x1l=0, x1u=40, x2l=0, x2u=30)
    # Does not seem to be a good indicator of risk

    # ### Public Record (Categorical)
    # Number of derogatory public records
    dataset['pub_rec'] = pd.cut(dataset['pub_rec'], [0, 0.9, 25], labels=['None', 'At least one'], include_lowest=True)
    print(dataset.groupby('loan_status')['pub_rec'].value_counts(dropna=False))
    plot_var('pub_rec', 'Public Records', continuous=False, dataset=dataset)
    # Loan default rate does not seem to change much by derogatory public records

    # ### Public Record of Bankruptcies (Categorical)
    # Number of public record bankruptcies
    dataset['pub_rec_bankruptcies'] = pd.cut(dataset['pub_rec_bankruptcies'], [0, 0.9, 25],
                                             labels=['None', 'At least one'], include_lowest=True)
    print(dataset.groupby('loan_status')['pub_rec_bankruptcies'].value_counts(dropna=False))
    plot_var('pub_rec_bankruptcies', 'Bankruptcies', continuous=False, dataset=dataset)
    # Loan default rate does not seem to change much by public bankruptcies records

    # ### Purpose (Categorical)
    # A category provided by the borrower for the loan request
    dataset.groupby('purpose')['loan_status'].value_counts(normalize=True).loc[:, 'Default'].sort_values()

    # ### Revolving Balance (Numerical)
    # Total credit revolving balance
    print(dataset.groupby('loan_status')['revol_bal'].describe())
    plot_var('revol_bal', 'Revolving Balance in $', continuous=True, dataset=dataset,
             x1limit=True, x2limit=True, x1l=0, x1u=80000, x2l=0, x2u=40000)
    # Seems like the data is heavily right skewed with a large range due to
    # large outliers so we take the log transformation
    dataset['log_revol_bal'] = dataset['revol_bal'].apply(lambda x: np.log(x + 1))
    dataset.drop('revol_bal', axis=1, inplace=True)
    print(dataset.groupby('loan_status')['log_revol_bal'].describe())
    plot_var('log_revol_bal', 'Log Revolving Balance in $', continuous=True, dataset=dataset)
    # There is not much difference in the two categories for revolving balances

    # ### Revolving Utility (Numerical)
    # Revolving line utilization rate, or the amount of credit the borrower
    # is using relative to all available revolving credit.
    dataset['revol_util'] = dataset['revol_util'].str.rstrip('%').astype('float')
    print(dataset.groupby('loan_status')['revol_util'].describe())
    plot_var('revol_util', 'Revolving Utility in %', continuous=True, dataset=dataset)
    # It seems those with a lower revolving utility are more likely to pay off their loans

    # ### Grade (Categorical)
    # LendingClub assigned loan grade. The higher the letter, the safer the loan.
    plot_var('grade', 'Grade', continuous=False, dataset=dataset)
    # There seems to be a strong trend between charge off rate and deteriorating grade

    # ### Term (Categorical)
    # The number of payments on the loan. Values are in months and can be either 36 or 60
    # dataset['term'].replace('36 months', 36, inplace=True)
    # dataset['term'].replace('60 months', 60, inplace=True)
    dataset['term'] = dataset['term'].apply(lambda s: np.int8(s.split()[0]))
    print(dataset.groupby('loan_status')['term'].value_counts(dropna=False))
    plot_var('term', 'Term (months)', continuous=False, dataset=dataset)
    # Loan Duration or how long to maturity seems to be important and a good indicator of risk of default. 
    # A longer duration has a higher risk that the loan will not be repaid.  

    # ### Total Accounts (Numerical)
    # The total number of credit lines currently in the borrower's credit file
    print(dataset.groupby('loan_status')['total_acc'].describe())
    plot_var('total_acc', 'Number of Total Accounts', continuous=True, dataset=dataset)
    # There does not seem to be a significant difference in charge off rate depending on the total account number

    # ### Verification Status (Categorical)
    # Indicates if income was verified, not verified, or if the income source was verified.
    print(dataset.groupby('loan_status')['verification_status'].value_counts(dropna=False))
    plot_var('verification_status', 'Verification Status', continuous=False, dataset=dataset)
    # There seems to be a strong linear trend between charged off rate and verification status.
    # Surprisingly, loans with a status of verified have a higher chance of becoming charged off.

    # # Preliminary Model Design

    # ### Create dummy variables
    # 1(negative class) means charged-off and 0(positive class) means fully paid and create dummy variables 
    # for all categorical variables
    dataset['loan_status'].replace('Default', 1, inplace=True)
    dataset['loan_status'].replace('Fully Paid', 0, inplace=True)

    dataset = pd.get_dummies(dataset, columns=['grade', 'home_ownership', 'verification_status', 'purpose',
                                               'initial_list_status', 'application_type', 'pub_rec',
                                               'pub_rec_bankruptcies'], drop_first=True)
    print(dataset.head())

    # ### Split the data
    dataset.dropna(inplace=True)
    # dataset.fillna(lambda x: x.median())
    y = dataset.loc[:, dataset.columns == 'loan_status']
    x = dataset.loc[:, dataset.columns != 'loan_status']
    # y = dataset['loan_status']
    # X = dataset.drop(columns=['loan_status'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0,
                                                        stratify=dataset['loan_status'])
    
    x_train.drop('total_pymnt', axis=1, inplace=True)
    test_pymnts = x_test.loc[:, x_test.columns == 'total_pymnt']
    x_test.drop('total_pymnt', axis=1, inplace=True)

    # x_train = x_train.reset_index()
    # y_train = y_train.reset_index()

    # # Feature Selection and Model Fitting

    # ### Logistic Regression Model

    # #### Base
    model_lr_base = LogisticRegression(penalty="l2",
                                       C=0.5,
                                       fit_intercept=True,
                                       random_state=0,
                                       max_iter=10000,
                                       solver='lbfgs')
    model_lr_base.fit(x_train, y_train.values.ravel())

    # #### Cost Sensitive Method
    model_lr = LogisticRegression(penalty="l2",
                                  C=0.5,
                                  fit_intercept=True,
                                  class_weight='balanced',
                                  random_state=0,
                                  max_iter=10000,
                                  solver='lbfgs')
    model_lr = model_lr.fit(x_train, y_train.values.ravel())

    import statsmodels.api as sm
    logit_model = sm.Logit(y, x)
    result = logit_model.fit()
    result.summary2()

    # We remove all variables with p-value less than 0.05
    print(result.pvalues[result.pvalues > 0.05])
    keep_list = list(result.pvalues[result.pvalues < 0.05].index)
    keep_list.append('loan_status')
    drop_list = [col for col in dataset.columns if col not in keep_list]
    x_train_lr = x_train.drop(labels=drop_list, axis=1)
    x_test_lr = x_test.drop(labels=drop_list, axis=1)
    model_lr2 = LogisticRegression(penalty="l2", C=0.5, fit_intercept=True, class_weight='balanced',
                                   random_state=0, max_iter=10000, solver='lbfgs')
    model_lr2.fit(x_train_lr, y_train.values.ravel())

    # #### Sampling Method - SMOTE
    x_train_lr_sm, y_train_sm = SMOTE(random_state=1).fit_resample(x_train, y_train.values.ravel())
    model_lr_smote = LogisticRegression(penalty="l2",
                                        C=0.5,
                                        fit_intercept=True,
                                        random_state=0,
                                        max_iter=10000,
                                        solver='lbfgs')
    model_lr_smote.fit(x_train_lr_sm, y_train_sm)

    # #### Sampling Method - ADASYN
    x_train_lr_as, y_train_as = ADASYN(random_state=1).fit_sample(x_train, y_train.values.ravel())
    model_lr_adasyn = LogisticRegression(penalty="l2",
                                         C=0.5, fit_intercept=True,
                                         random_state=0,
                                         max_iter=10000,
                                         solver='lbfgs')
    model_lr_adasyn.fit(x_train_lr_as, y_train_as)

    # ### Logistic Regression Results

    # #### Base
    # Make predictions and determine the error
    y_pred_lr = model_lr_base.predict(x_test)
    print("Accuracy: %.2f%%" % (model_lr_base.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    # predict probabilities
    prob = model_lr_base.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    preds = prob[:,1]
    # calculate pr curve
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    # calculate auc
    print('PR-AUC: ', auc(recall_lr, precision_lr))
    # if you got an AUROC of <=0.5, it just means you need to invert the predictions
    # because Scikit-Learn is misinterpreting the positive class. AUROC should be >= 0.5.

    # #### Cost Sensitive Method
    y_pred_lr = model_lr.predict(x_test)
    print("Accuracy: %.2f%%" % (model_lr.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    prob = model_lr.predict_proba(x_test)
    preds = prob[:,1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    y_pred_lr2 = model_lr2.predict(x_test_lr)
    print("Accuracy: %.2f%%" % (model_lr2.score(x_test_lr, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr2))
    print('F1 Score:', f1_score(y_test, y_pred_lr2))
    print(classification_report(y_test, y_pred_lr2))
    prob = model_lr2.predict_proba(x_test_lr)
    preds = prob[:,1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    # manually picked
    important_indices = ['log_annual_inc',
                         'dti',
                         'home_ownership_OWN',
                         'home_ownership_RENT',
                         'log_installment',
                         'int_rate',
                         'loan_amnt',
                         'revol_util',
                         'grade_B',
                         'grade_C',
                         'grade_D',
                         'grade_E',
                         'grade_F',
                         'grade_G',
                         'term',
                         'verification_status_Source Verified',
                         'verification_status_Verified']
    x_train_lr = x_train.loc[:, important_indices]
    x_test_lr = x_test.loc[:, important_indices]
    model_lr3 = LogisticRegression(penalty="l2", C=0.5, fit_intercept=True, class_weight='balanced',
                                   random_state=0, max_iter=10000, solver='lbfgs')
    model_lr3 = model_lr3.fit(x_train_lr, y_train.values.ravel())
    y_pred_lr = model_lr3.predict(x_test_lr)
    print("Accuracy: %.2f%%" % (model_lr3.score(x_test_lr, y_test) * 100))
    confusion_matrix(y_test, y_pred_lr)
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    prob = model_lr3.predict_proba(x_test_lr)
    preds = prob[:,1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    # #### Sampling Method - SMOTE
    y_pred_lr = model_lr_smote.predict(x_test)
    print("Accuracy: %.2f%%" % (model_lr_smote.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    prob = model_lr_smote.predict_proba(x_test)
    preds = prob[:, 1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    # #### Sampling Method - ADASYN
    y_pred_lr = model_lr_adasyn.predict(x_test)
    print("Accuracy: %.2f%%" % (model_lr_adasyn.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))
    prob = model_lr_adasyn.predict_proba(x_test)
    preds = prob[:, 1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    # ### Random Forest Model
    model_rf_path = Path('../data/model_rf.joblib')
    if os.path.exists(model_rf_path):  # list of features files does not exist dont save model
        model_rf = load(model_rf_path)
    else:
        n_trees = [50, 100, 250, 500, 1000, 1500, 2500]
        rf_dict = dict.fromkeys(n_trees)
        for num in n_trees:
            print(num)
            rf = RandomForestClassifier(n_estimators=num,
                                        min_samples_leaf=30,
                                        oob_score=True,
                                        random_state=100,
                                        class_weight='balanced',
                                        n_jobs=-1)
            rf.fit(x_train, y_train.values.ravel())
            rf_dict[num] = rf

        oob_error_list = [None] * len(n_trees)

        for i in range(len(n_trees)):
            oob_error_list[i] = 1 - rf_dict[n_trees[i]].oob_score_

        plt.plot(n_trees, oob_error_list, 'bo', n_trees, oob_error_list, 'k')
        plt.xlabel('Number of Trees')
        plt.ylabel('Out of Bag Error')
        model_rf = rf_dict[500]

        # calculate permutation feature importance
        result = permutation_importance(model_rf, x_train, y_train, n_repeats=10, random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=x_train.columns[perm_sorted_idx])
        ax1.set_xlabel('Feature Importance')
        fig.tight_layout()
        plt.show()

    # manually picked
    important_indices = ['log_annual_inc',
                         'dti',
                         'home_ownership_OWN',
                         'home_ownership_RENT',
                         'log_installment',
                         'int_rate',
                         'loan_amnt',
                         'revol_util',
                         'grade_B',
                         'grade_C',
                         'grade_D',
                         'grade_E',
                         'grade_F',
                         'grade_G',
                         'term',
                         'verification_status_Source Verified',
                         'verification_status_Verified']
    x_train_rf = x_train.loc[:, important_indices]
    x_test_rf = x_test.loc[:, important_indices]

    # #### Base Model
    model_rf_base_path = Path('../data/model_rf_base.joblib')
    if os.path.exists(model_rf_base_path):
        model_rf_base = load(model_rf_base_path)
    else:
        model_rf_base = RandomForestClassifier(n_estimators=500,
                                               min_samples_leaf=30,
                                               oob_score=True,
                                               random_state=100,
                                               n_jobs=-1)
        model_rf_base.fit(x_train_rf, y_train.values.ravel())
        dump(model_rf_base, model_rf_base_path)

    # #### Class Sensetive Method
    model_rf_path = Path('../data/model_rf.joblib')
    if os.path.exists(model_rf_path):
        model_rf = load(model_rf_path)
    else:
        model_rf = RandomForestClassifier(n_estimators=500,
                                          min_samples_leaf=30,
                                          oob_score=True,
                                          random_state=100,
                                          class_weight='balanced',
                                          n_jobs=-1)
        model_rf.fit(x_train_rf, y_train.values.ravel())
        dump(model_rf, model_rf_path)

    # #### Sampling - SMOTE Method
    model_rf_smote_path = Path('../data/model_rf_smote.joblib')
    if os.path.exists(model_rf_smote_path):
        model_rf_smote = load(model_rf_smote_path)
    else:
        x_train_rf_sm, y_train_sm = SMOTE(random_state=1).fit_resample(x_train_rf, y_train.values.ravel())
        model_rf_smote = RandomForestClassifier(n_estimators=500,
                                                min_samples_leaf=30,
                                                oob_score=True,
                                                random_state=100,
                                                n_jobs=-1)
        model_rf_smote.fit(x_train_rf_sm, y_train_sm)
        dump(model_rf_smote, model_rf_smote_path)

    # #### Sampling - ADASYN Method
    model_rf_adasyn_path = Path('../data/model_rf_adasyn.joblib')
    if os.path.exists(model_rf_adasyn_path):
        model_rf_adasyn = load(model_rf_adasyn_path)
    else:
        x_train_rf_as, y_train_as = ADASYN(random_state=1).fit_sample(x_train_rf, y_train.values.ravel())
        model_rf_adasyn = RandomForestClassifier(n_estimators=500,
                                                 min_samples_leaf=30,
                                                 oob_score=True,
                                                 random_state=100,
                                                 n_jobs=-1)
        model_rf_adasyn.fit(x_train_rf_as, y_train_as)
        dump(model_rf_adasyn, model_rf_adasyn_path)

    # ### Random Forest Results

    # #### Base
    # Make predictions and determine the error
    y_pred_rf = model_rf_base.predict(x_test_rf)
    print("Accuracy: %.2f%%" % (model_rf_base.score(x_test_rf, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_rf))
    print('F1 Score:', f1_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    # predict probabilities
    prob = model_rf_base.predict_proba(x_test_rf)
    # keep probabilities for the positive outcome only
    preds = prob[:, 1]
    # calculate pr curve
    precision_rf, recall_rf, threshold = precision_recall_curve(y_test, preds)
    # calculate auc
    print('PR-AUC: ', auc(recall_rf, precision_rf))

    # #### Cost Sensitive Methods
    y_pred_rf = model_rf.predict(x_test_rf)
    print("Accuracy: %.2f%%" % (model_rf.score(x_test_rf, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_rf))
    print('F1 Score:', f1_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    prob = model_rf.predict_proba(x_test_rf)
    preds = prob[:, 1]
    precision_rf, recall_rf, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_rf, precision_rf))

    # #### Sampling Methods - SMOTE
    y_pred_rf = model_rf_smote.predict(x_test_rf)
    print("Accuracy: %.2f%%" % (model_rf_smote.score(x_test_rf, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_rf))
    print('F1 Score:', f1_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    prob = model_rf_smote.predict_proba(x_test_rf)
    preds = prob[:, 1]
    precision_rf, recall_rf, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_rf, precision_rf))

    # #### Sampling Methods - ADASYN
    y_pred_rf = model_rf_adasyn.predict(x_test_rf)
    print("Accuracy: %.2f%%" % (model_rf_adasyn.score(x_test_rf, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_rf))
    print('F1 Score:', f1_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))
    prob = model_rf_adasyn.predict_proba(x_test_rf)
    preds = prob[:, 1]
    precision_rf, recall_rf, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_rf, precision_rf))

    # ### Neural Network Model
    scaler = StandardScaler()
    scaler.fit(x_train.astype('float64'))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train_nn = scaler.transform(x_train.astype('float64'))
    x_test_nn = scaler.transform(x_test.astype('float64'))
    dump(scaler, Path('../data/scaler.joblib'))

    # #### Base
    model_nn_base = Sequential()
    # Input layer
    model_nn_base.add(Dense(20, activation='relu', input_shape=(39,)))
    # Hidden layer
    model_nn_base.add(Dense(15, activation='relu'))
    model_nn_base.add(Dense(4, activation='relu'))
    # Output layer
    model_nn_base.add(Dense(2, activation='sigmoid'))
    print(model_nn_base.output_shape)

    model_nn_base.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
    model_nn_base.fit(x_train_nn, to_categorical(y_train), epochs=5, batch_size=10, verbose=1)

    # #### Cost Sensitive Method
    model_nn_path = Path('../data/model_nn.h5')
    if os.path.exists(model_nn_path):
        model_nn = load_model(str(model_nn_path))
    else:
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(y_train.values.ravel()),
                                                          y_train.values.ravel())
        model_nn = Sequential()
        # Input layer
        model_nn.add(Dense(20, activation='relu', input_shape=(39,)))
        # Hidden layer
        model_nn.add(Dense(15, activation='relu'))
        model_nn.add(Dense(4, activation='relu'))
        # Output layer
        model_nn.add(Dense(2, activation='sigmoid'))
        print(model_nn.output_shape)

        model_nn.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
        model_nn.fit(x_train_nn, to_categorical(y_train), epochs=5, batch_size=10, class_weight=class_weights,
                     verbose=1)

        model_nn.save(str(model_nn_path))
    print(model_nn.summary())

    # #### Sampling Method - SMOTE
    if os.path.exists(Path('../data/model_nn_smote.h5')):
        model_nn_smote = load_model(str(Path('../data/model_nn_smote.h5')))
    else:
        x_train_nn_sm, y_train_sm = SMOTE(random_state=1).fit_sample(x_train_nn, y_train.values.ravel())
        model_nn_smote = load_model(str(model_nn_path))
        model_nn_smote.fit(x_train_nn_sm, to_categorical(y_train_sm), epochs=5, batch_size=10, verbose=0)
        model_nn_smote.save(str(Path('../data/model_nn_smote.h5')))

    # #### Sampling Method - ADASYN
    if os.path.exists(Path('../data/model_nn_adasyn.h5')):
        model_nn_adasyn = load_model(str(Path('../data/model_nn_adasyn.h5')))
    else:
        x_train_nn_as, y_train_as = ADASYN(random_state=1).fit_sample(x_train_nn, y_train.values.ravel())
        model_nn_adasyn = load_model(str(model_nn_path))
        model_nn_adasyn.fit(x_train_nn_as, to_categorical(y_train_as), epochs=5, batch_size=10, verbose=0)
        model_nn_adasyn.save(str(Path('../data/model_nn_adasyn.h5')))

    # ### Neural Network Results

    # #### Base
    y_pred_nn = model_nn_base.predict_classes(x_test_nn)
    score = model_nn_base.evaluate(x_test_nn, to_categorical(y_test))
    print("Accuracy: %.2f%%" % (score[1] * 100))
    print(confusion_matrix(y_test, y_pred_nn))
    print('F1 Score:', f1_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))
    # predict probabilities
    prob = model_nn_base.predict(x_test_nn)
    # keep probabilities for the positive outcome only
    preds = prob[:, 1]
    # calculate pr curve
    precision_nn, recall_nn, threshold = precision_recall_curve(y_test, preds)
    # calculate auc
    print('PR-AUC: ', auc(recall_nn, precision_nn))

    # #### Cost Sensentive Method
    y_pred_nn = model_nn.predict_classes(x_test_nn)
    score = model_nn.evaluate(x_test_nn, to_categorical(y_test))
    print("Accuracy: %.2f%%" % (score[1] * 100))
    print(confusion_matrix(y_test, y_pred_nn))
    print('F1 Score:', f1_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))
    prob = model_nn.predict(x_test_nn)
    preds = prob[:, 1]
    precision_nn, recall_nn, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_nn, precision_nn))

    # #### Sampling Method - SMOTE
    y_pred_nn = model_nn_smote.predict_classes(x_test_nn)
    score = model_nn_smote.evaluate(x_test_nn, to_categorical(y_test))
    print("Accuracy: %.2f%%" % (score[1] * 100))
    print(confusion_matrix(y_test, y_pred_nn))
    print('F1 Score:', f1_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))
    prob = model_nn_smote.predict(x_test_nn)
    preds = prob[:, 1]
    precision_nn, recall_nn, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_nn, precision_nn))

    # #### Sampling Method - ADASYN
    y_pred_nn = model_nn_adasyn.predict_classes(x_test_nn)
    score = model_nn_adasyn.evaluate(x_test_nn, to_categorical(y_test))
    print("Accuracy: %.2f%%" % (score[1] * 100))
    print(confusion_matrix(y_test, y_pred_nn))
    print('F1 Score:', f1_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))
    prob = model_nn_adasyn.predict(x_test_nn)
    preds = prob[:, 1]
    precision_nn, recall_nn, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_nn, precision_nn))

    # ### XGBoost Model
    model_xgb = XGBClassifier(max_depth=3,
                              learning_rate=0.05,
                              n_estimators=300,
                              objective='binary:logistic',
                              subsample=0.8,
                              random_state=42)
    eval_set = [(x_train, y_train.values.ravel()), (x_test, y_test.values.ravel())]
    eval_metric = ['auc', 'error']
    model_xgb.fit(x_train, y_train.values.ravel(), eval_metric=eval_metric, eval_set=eval_set,
                  verbose=False)

    results = model_xgb.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), dpi=90)
    ax1.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax1.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax1.legend()
    ax1.set_ylabel('AUC')
    ax1.set_title('XGBoost PR-AUC')

    ax2.plot(x_axis, results['validation_0']['error'], label='Train')
    ax2.plot(x_axis, results['validation_1']['error'], label='Test')
    ax2.legend()
    ax2.set_ylabel('Classification Error')
    ax2.set_title('XGBoost Classification Error')
    plt.show()

    plot_importance(model_xgb)
    plt.show()

    # from sklearn.feature_selection import SelectFromModel
    # from sklearn.metrics import accuracy_score
    # thresholds = np.sort(model_xgb.feature_importances_)
    # for thresh in thresholds:
    #     # select features using threshold
    #     selection = SelectFromModel(model_xgb, threshold=thresh, prefit=True)
    #     select_x_train = selection.transform(x_train)
    #     # train model
    #     selection_model = XGBClassifier(scale_pos_weight=class_weights[1])
    #     selection_model.fit(select_x_train, y_train.values.ravel())
    #     # eval model
    #     select_x_test = selection.transform(x_test)
    #     y_pred_xgb = selection_model.predict(select_x_test)
    #     predictions = [round(value) for value in y_pred_xgb]
    #     accuracy = accuracy_score(y_test, predictions)
    #     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh,
    #                                                   select_x_train.shape[1],
    #                                                   accuracy * 100))

    important_indices = ['log_annual_inc',
                         'dti',
                         'home_ownership_OWN',
                         'home_ownership_RENT',
                         'log_installment',
                         'int_rate',
                         'loan_amnt',
                         'revol_util',
                         'grade_B',
                         'grade_C',
                         'grade_D',
                         'grade_E',
                         'grade_F',
                         'grade_G',
                         'term',
                         'verification_status_Source Verified',
                         'verification_status_Verified']
    x_train_xgb = x_train.loc[:, important_indices]
    x_test_xgb = x_test.loc[:, important_indices]

    # #### Base Model
    model_xgb_base_path = Path('../data/model_xgb_base.joblib')
    if os.path.exists(model_xgb_base_path):
        model_xgb_base = load(model_xgb_base_path)
    else:
        model_xgb_base = XGBClassifier(max_depth=3,
                                       learning_rate=0.05,
                                       n_estimators=300,
                                       objective='binary:logistic',
                                       subsample=0.8,
                                       random_state=42)
        model_xgb_base.fit(x_train_xgb, y_train.values.ravel())
        dump(model_xgb_base, model_xgb_base_path)

    # #### Cost Sensitive Method
    model_path_xgb = Path('../data/model_xgb.joblib')
    if os.path.exists(model_path_xgb):
        model_xgb = load(model_path_xgb)
    else:
        neg_weight = y_train['loan_status'].value_counts(dropna=False)[0] / \
                     y_train['loan_status'].value_counts(dropna=False)[1]
        model_xgb = XGBClassifier(max_depth=3,
                                  learning_rate=0.05,
                                  n_estimators=300,
                                  objective='binary:logistic',
                                  subsample=0.8,
                                  scale_pos_weight=neg_weight,
                                  random_state=42)
        model_xgb.fit(x_train_xgb, y_train.values.ravel())
        dump(model_xgb, model_path_xgb)

    # #### Sampling Method - SMOTE
    model_xgb_smote_path = Path('../data/model_xgb_smote.joblib')
    if os.path.exists(model_xgb_smote_path):
        model_xgb_smote = load(model_xgb_smote_path)
    else:
        x_train_xgb_sm, y_train_sm = SMOTE(random_state=1).fit_resample(x_train_xgb, y_train.values.ravel())
        x_train_xgb_sm = pd.DataFrame(x_train_xgb_sm, columns=x_test_xgb.columns)
        y_train_sm = pd.DataFrame(y_train_sm, columns=y_train.columns)
        model_xgb_smote = XGBClassifier(max_depth=3,
                                        learning_rate=0.01,
                                        n_estimators=300,
                                        objective='binary:logistic',
                                        subsample=0.8,
                                        random_state=42)
        model_xgb_smote.fit(x_train_xgb_sm, y_train_sm.values.ravel())
        dump(model_xgb_smote, model_xgb_smote_path)

    # #### Sampling Method - ADASYN
    model_xgb_adasyn_path = Path('../data/model_xgb_adasyn.joblib')
    if os.path.exists(model_xgb_adasyn_path):
        model_xgb_adasyn = load(model_xgb_adasyn_path)
    else:
        x_train_xgb_as, y_train_as = ADASYN(random_state=1).fit_sample(x_train_xgb, y_train.values.ravel())
        x_train_xgb_as = pd.DataFrame(x_train_xgb_as, columns=x_test_xgb.columns)
        y_train_as = pd.DataFrame(y_train_as, columns=y_train.columns)
        model_xgb_adasyn = XGBClassifier(max_depth=3,
                                         learning_rate=0.01,
                                         n_estimators=300,
                                         objective='binary:logistic',
                                         subsample=0.8,
                                         random_state=42)
        model_xgb_adasyn.fit(x_train_xgb_as, y_train_as.values.ravel())
        dump(model_xgb_adasyn, model_xgb_adasyn_path)

    # ### XGBoost Results
    # Make predictions and determine the error
    y_pred_xgb = model_xgb_base.predict(x_test_xgb)
    print("Accuracy: %.2f%%" % (model_xgb_base.score(x_test_xgb, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_xgb))
    print('F1 Score:', f1_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    # predict probabilities
    prob = model_xgb_base.predict_proba(x_test_xgb)
    # keep probabilities for the positive outcome only
    preds = prob[:, 1]
    # calculate pr curve
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test, preds)
    # calculate auc, equivalent to roc_auc_score()?
    print('PR-AUC: ', auc(recall_xgb, precision_xgb))

    # #### Cost Sensitive Method
    y_pred_xgb = model_xgb.predict(x_test_xgb)
    print("Accuracy: %.2f%%" % (model_xgb.score(x_test_xgb, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_xgb))
    print('F1 Score:', f1_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    prob = model_xgb.predict_proba(x_test_xgb)
    preds = prob[:, 1]
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_xgb, precision_xgb))

    # #### Sampling Method - SMOTE
    y_pred_xgb = model_xgb_smote.predict(x_test_xgb)
    print("Accuracy: %.2f%%" % (model_xgb_smote.score(x_test_xgb, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_xgb))
    print('F1 Score:', f1_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    prob = model_xgb_smote.predict_proba(x_test_xgb)
    preds = prob[:, 1]
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_xgb, precision_xgb))

    # #### Sampling Method - ADASYN
    y_pred_xgb = model_xgb_adasyn.predict(x_test_xgb)
    print("Accuracy: %.2f%%" % (model_xgb_adasyn.score(x_test_xgb, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_xgb))
    print('F1 Score:', f1_score(y_test, y_pred_xgb))
    print(classification_report(y_test, y_pred_xgb))
    prob = model_xgb_adasyn.predict_proba(x_test_xgb)
    preds = prob[:, 1]
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test, preds)
    print('PR-AUC: ', auc(recall_xgb, precision_xgb))

    # ### Support Vector Machine Model
    # from sklearn.svm import SVC
    # from sklearn.model_selection import GridSearchCV

    # model_svm_path = Path('../data/model_svm.joblib')
    # if os.path.exists(model_svm_path):
    #     model_svm = load(model_svm_path)
    # else:
    #     # parameter_candidates = [
    #     #     {'C': [1, 10, 100], 'kernel': ['linear'], 'class_weight': ['balanced']},
    #     #     {'C': [1, 10, 100], 'gamma': [10, 0, 0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced']}
    #     # ]
    #     #
    #     # clf = GridSearchCV(estimator=SVC(), param_grid = parameter_candidates, n_jobs=-1, cv=5)
    #     # clf.fit(x_train, y_train.values.ravel())
    #     # print('Best score for data:', clf.best_score_)
    #     # print('Best C:', clf.best_estimator_.C)
    #     # print('Best Kernel:', clf.best_estimator_.kernel)
    #     # print('Best Gamma:', clf.best_estimator_.gamma)
    #     model_svm = SVC(kernel='rbf', C=1, gamma=0.05, class_weight='balanced', random_state=0)
    #     model_svm.fit(x_train, y_train.values.ravel())
    #     dump(model_svm, model_svm_path)
    # y_pred_svm = model_svm.predict(x_test)
    # print('Accuracy of SVM on test set: {:.3f}'.format(model_svm.score(x_test, y_test)))
    # # 0.741 C=1 gamma = 0.05
    # # 0.777 C=1 gamma = 1
    # print("Accuracy: %.2f%%" % (model_svm.score(x_test, y_test) * 100))
    # print(confusion_matrix(y_test, y_pred_svm))

    # ### Ensemble Model

    # Random Forest Model
    output_rf = model_rf.predict_proba(x_test_rf)
    output_rf2 = model_rf.predict(x_test_rf)

    # Neural Network Model
    output_nn = model_nn_adasyn.predict(x_test_nn)
    output_nn2 = model_nn_adasyn.predict_classes(x_test_nn)

    # XGBoost Model
    output_xgb = model_xgb.predict_proba(x_test_xgb)
    output_xgb2 = model_xgb.predict(x_test_xgb)

    # Ensemble output
    # Avg
    output = (output_rf + output_nn + output_xgb) / 3

    # Majority vote
    output2 = pd.DataFrame({'rf': output_rf2, 'nn': output_nn2, 'xgb': output_xgb2})
    output2['Prediction'] = output2.sum(axis=1)
    output2.loc[output2['Prediction'] < 2, 'Prediction'] = 0
    output2.loc[output2['Prediction'] >= 2, 'Prediction'] = 1

    # #### Stacking
    # Obtain the base first-level model predictions
    y_pred_stacking_rf = model_rf.predict(x_train_rf)
    y_pred_stacking_nn = model_nn_adasyn.predict_classes(x_train_nn)
    y_pred_stacking_xgb = model_xgb.predict(x_train_xgb)
    x_train_stacking = pd.DataFrame({'rf': y_pred_stacking_rf, 'nn': y_pred_stacking_nn, 'xgb': y_pred_stacking_xgb})
    y_train_stacking = pd.DataFrame(y_train.copy())
    x_test_stacking = pd.DataFrame({'rf': output_rf2, 'nn': output_nn2, 'xgb': output_xgb2})
    y_test_stacking = pd.DataFrame(y_test.copy())

    # Fit the Meta Learner from first level predictions
    neg_weight = y_train_stacking['loan_status'].value_counts(dropna=False)[0] / \
                 y_train_stacking['loan_status'].value_counts(dropna=False)[1]
    model_stacking_xgb = XGBClassifier(max_depth=3,
                                       learning_rate=0.05,
                                       n_estimators=300,
                                       objective='binary:logistic',
                                       subsample=0.8,
                                       scale_pos_weight=neg_weight,
                                       random_state=42)
    model_stacking_xgb.fit(x_train_stacking, y_train_stacking.values.ravel())

    model_stacking_lr = LogisticRegression(penalty="l2", C=0.5, fit_intercept=True, class_weight='balanced',
                                           random_state=0, max_iter=10000, solver='lbfgs')
    model_stacking_lr.fit(x_train_stacking, y_train_stacking.values.ravel())

    # ### Ensembled Results
    p_output = pd.DataFrame(output.copy())
    p_output.columns = ['Fully Paid', 'Default']
    p_output['Prediction'] = p_output.idxmax(axis=1)
    p_output['Fully Paid'] = p_output['Fully Paid'].multiply(100).round(0).astype(int).astype(str) + '%'
    p_output['Default'] = p_output['Default'].multiply(100).round(0).astype(int).astype(str) + '%'

    y_test2 = pd.DataFrame(y_test.copy())
    y_test2.columns = ['Prediction']
    y_test2['Prediction'].replace(0, 'Fully Paid', inplace=True)
    y_test2['Prediction'].replace(1, 'Default', inplace=True)
    print('Averaging Method')
    accuracy = accuracy_score(y_true=y_test2, y_pred=p_output['Prediction'])
    print("Accuracy: %.2f%%" % (accuracy * 100))
    print(confusion_matrix(y_test2, p_output['Prediction']))
    print(classification_report(y_test2, p_output['Prediction']))

    # Majority
    output2['Prediction'].replace(0, 'Fully Paid', inplace=True)
    output2['Prediction'].replace(1, 'Default', inplace=True)

    print('\nMajority Method')
    accuracy = accuracy_score(y_true=y_test2, y_pred=output2['Prediction'])
    print("Accuracy: %.2f%%" % (accuracy * 100))
    print(confusion_matrix(y_test2, output2['Prediction']))
    print(classification_report(y_test2, output2['Prediction']))

    # Predict the test
    y_pred_xgb = model_stacking_xgb.predict(x_test_stacking)
    print("Accuracy: %.2f%%" % (model_stacking_xgb.score(x_test_stacking, y_test_stacking) * 100))
    print(confusion_matrix(y_test_stacking, y_pred_xgb))
    print('F1 Score:', f1_score(y_test_stacking, y_pred_xgb))
    print(classification_report(y_test_stacking, y_pred_xgb))
    # predict probabilities
    prob = model_stacking_xgb.predict_proba(x_test_stacking)
    # keep probabilities for the positive outcome only
    preds = prob[:, 1]
    # calculate pr curve
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test_stacking, preds)
    # calculate auc, equivalent to roc_auc_score()?
    print('PR-AUC: ', auc(recall_xgb, precision_xgb))

    y_pred_lr = model_stacking_lr.predict(x_test_stacking)
    print("Accuracy: %.2f%%" % (model_stacking_lr.score(x_test_stacking, y_test_stacking) * 100))
    print(confusion_matrix(y_test_stacking, y_pred_lr))
    print('F1 Score:', f1_score(y_test_stacking, y_pred_lr))
    print(classification_report(y_test_stacking, y_pred_lr))
    prob = model_stacking_lr.predict_proba(x_test_stacking)
    preds = prob[:, 1]
    precision_lr, recall_lr, threshold = precision_recall_curve(y_test_stacking, preds)
    print('PR-AUC: ', auc(recall_lr, precision_lr))

    # predict probabilities
    model_rf = load(model_rf_path)
    model_nn = load_model(str(Path('../data/model_nn_adasyn.h5')))
    model_xgb = load(model_path_xgb)
    pos_pred_rf = model_rf.predict_proba(x_test_rf)
    pos_pred_nn = model_nn.predict(x_test_nn)
    pos_pred_xgb = model_xgb.predict_proba(x_test_xgb)
    # keep probabilities for the positive outcome only
    pos_pred_rf = pos_pred_rf[:, 1]
    pos_pred_nn = pos_pred_nn[:, 1]
    pos_pred_xgb = pos_pred_xgb[:, 1]
    # calculate roc curve
    fpr_rf, tpr_rf, threshold = roc_curve(y_test, pos_pred_rf)
    fpr_nn, tpr_nn, threshold = roc_curve(y_test, pos_pred_nn)
    fpr_xgb, tpr_xgb, threshold = roc_curve(y_test, pos_pred_xgb)
    # # calculate auc, equivalent to roc_auc_score()?
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), dpi=90)
    ax1.set_title('Receiver Operating Characteristic')
    ax1.plot(fpr_rf, tpr_rf, 'r', label='RF AUC = %0.2f' % roc_auc_rf)
    ax1.plot(fpr_nn, tpr_nn, 'b', label='NN AUC = %0.2f' % roc_auc_nn)
    ax1.plot(fpr_xgb, tpr_xgb, color='yellow', label='XGB AUC = %0.2f' % roc_auc_xgb)
    ax1.legend(loc='lower right')
    ax1.plot([0, 1], [0, 1], 'g--')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_xlabel('False Positie Rate')

    precision_rf, recall_rf, threshold = precision_recall_curve(y_test, pos_pred_rf)
    precision_nn, recall_nn, threshold = precision_recall_curve(y_test, pos_pred_nn)
    precision_xgb, recall_xgb, threshold = precision_recall_curve(y_test, pos_pred_xgb)
    pr_auc_rf = auc(recall_rf, precision_rf)
    pr_auc_nn = auc(recall_nn, precision_nn)
    pr_auc_xgb = auc(recall_xgb, precision_xgb)
    ax2.set_title('Precision-Recall Curve')
    ax2.plot(recall_rf, precision_rf, 'r', label='RF AUC = %0.2f' % pr_auc_rf)
    ax2.plot(recall_nn, precision_nn, 'b', label='NN AUC = %0.2f' % pr_auc_nn)
    ax2.plot(recall_xgb, precision_xgb, 'y', label='XGB AUC = %0.2f' % pr_auc_xgb)
    ax2.legend(loc='upper right')
    ax2.plot([0, 1], [0.4, 0.4], 'r--')
    ax2.set_ylabel('Precision')
    ax2.set_xlabel('Recall')
    plt.show()

    accu_metrics = pd.DataFrame(p_output.copy())
    accu_metrics['Actual'] = y_test2.to_numpy()
    accu_metrics['category'] = accu_metrics[['Fully Paid', 'Default']].max(axis=1)
    accu_metrics['category'] = accu_metrics['category'].str.rstrip('%').astype('float')
    temp = pd.DataFrame(x_test['loan_amnt'])
    temp = temp.join(test_pymnts)
    temp.reset_index(inplace=True, drop=True)
    accu_metrics = accu_metrics.join(temp)
    accu_metrics['PL'] = ((accu_metrics['total_pymnt'] - accu_metrics['loan_amnt']) / abs(
        accu_metrics['loan_amnt'])) * 100
    accu_metrics['proba_range'] = pd.cut(accu_metrics['category'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                         right=False,
                                         labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60',
                                                 '60-70', '70-80', '80-90', '90-100'], include_lowest=True)
    accu_metrics['correct'] = 0
    accu_metrics.loc[accu_metrics['Prediction'] == accu_metrics['Actual'], 'correct'] = 1
    accu_metric = pd.DataFrame(accu_metrics.groupby('proba_range')['correct'].sum())
    accu_metric['total'] = accu_metrics.groupby(['proba_range']).count()['correct']
    accu_metric['Accuracy'] = accu_metric['correct'] / accu_metric['total'] * 100
    accu_metric.dropna(inplace=True)
    accu_metric.reset_index(inplace=True)

    returns_orgi = pd.DataFrame(accu_metrics.groupby('proba_range')['PL'].sum())
    returns_orgi['rows'] = accu_metrics.groupby(['proba_range']).count()['correct']
    returns_orgi['Avg_pct_return'] = returns_orgi['PL'] / returns_orgi['rows']
    returns_new = accu_metrics.loc[accu_metrics['Prediction'] == 'Fully Paid']
    returns_new = pd.DataFrame(returns_new.groupby('proba_range')['PL'].sum())
    returns_new['rows'] = accu_metrics.groupby(['proba_range']).count()['correct']
    returns_new['Avg_pct_return'] = returns_new['PL'] / returns_new['rows']
    returns_chart = pd.DataFrame()
    returns_chart['avg_improvement'] = returns_new['Avg_pct_return'] - returns_orgi['Avg_pct_return']
    returns_chart.dropna(inplace=True)
    returns_chart.reset_index(inplace=True)

    # This is for get rid of the groupby no data issue
    accu_metric.to_csv('../data/delme.csv', index=False)
    accu_metric = pd.read_csv(Path('../data/delme.csv'), header=0)
    returns_chart.to_csv('../data/delme2.csv')
    returns_chart = pd.read_csv(Path('../data/delme2.csv'), header=0)

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 3), dpi=90)
    sns.barplot(x='proba_range', y='total', data=accu_metric, color=sns.xkcd_rgb['windows blue'], ax=ax1)
    ax1.set_xlabel('Prediction Probability Range')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Ranges')
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    sns.barplot(x='proba_range', y="avg_improvement", data=returns_chart, color=sns.xkcd_rgb['windows blue'], ax=ax2)
    ax2.set_xlabel('Prediction Probability Range')
    ax2.set_ylabel('Percent')
    ax2.set_title('Average Percent Return Improvement')
    # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    plt.show()

    print('Overall return without model:',
          ((accu_metrics['total_pymnt'].sum() / accu_metrics['loan_amnt'].sum()) - 1) * 100)

    total_cost = accu_metrics.loc[accu_metrics['Prediction'] == 'Fully Paid']['loan_amnt'].sum()
    total_return = accu_metrics.loc[accu_metrics['Prediction'] == 'Fully Paid']['total_pymnt'].sum()
    print('Overall return with model:', ((total_return / total_cost) - 1) * 100)

    orig_roi = accu_metrics['total_pymnt'].sum() - accu_metrics['loan_amnt'].sum()
    new_roi = total_return - total_cost
    pcnt_chng = ((new_roi - orig_roi) / abs(orig_roi)) * 100
    print('Percent improvement:', pcnt_chng)
