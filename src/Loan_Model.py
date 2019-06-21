import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.ensemble import VotingClassifier
from joblib import dump, load


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
    print(missing_values)

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
    # It seems that those with an earlier credit line are more likely to be less risk

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
    # It seems longer term loans have a higher likelihood of being charged off

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
    # 1 means charged-off and 0 means fully paid and create dummy variables for all categorical variables
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
    # x_train = x_train.reset_index()
    # y_train = y_train.reset_index()


    # # Feature Selection and Model Fitting

    # ### Logistic Regression Model
    model_lr = LogisticRegression(penalty="l2",
                                  C=0.5,
                                  fit_intercept=True,
                                  class_weight='balanced',
                                  random_state=0,
                                  max_iter=10000,
                                  solver='lbfgs')
    model_lr = model_lr.fit(x_train, y_train.values.ravel())
    y_pred_lr = model_lr.predict(x_test)
    # df_coefs = pd.DataFrame(model.coef_[0], index=x.columns, columns = ['Coefficient'])
    # df_coefs

    print("Accuracy: %.2f%%" % (model_lr.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))

    import statsmodels.api as sm
    logit_model = sm.Logit(y, x)
    result = logit_model.fit()
    print(result.summary2())

    # We remove all variables with p-value less than 0.05
    print(result.pvalues[result.pvalues > 0.05])
    keep_list = list(result.pvalues[result.pvalues < 0.05].index)
    keep_list.append('loan_status')
    drop_list = [col for col in dataset.columns if col not in keep_list]
    x_train_lr = x_train.drop(labels=drop_list, axis=1)
    x_test_lr = x_test.drop(labels=drop_list, axis=1)
    model_lr2 = LogisticRegression(penalty="l2", C=0.5, fit_intercept=True, class_weight='balanced',
                                   random_state=0, max_iter=10000, solver='lbfgs')
    model_lr2 = model_lr2.fit(x_train_lr, y_train.values.ravel())
    y_pred_lr = model_lr2.predict(x_test_lr)

    # ### Logistic Regression Results
    print("Accuracy: %.2f%%" % (model_lr2.score(x_test_lr, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_lr))
    print('F1 Score:', f1_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    # ### Random Forest Model
    model_rf_path = Path('../data/rf_1000.joblib')
    if os.path.exists(model_rf_path):
        forest = load(model_rf_path)
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
        # Save model to file
        dump(rf_dict[1000], model_rf_path)
        forest = rf_dict[1000]
    y_pred_rf = forest.predict(x_test)

    # ### Random Forest Results
    print("Accuracy: %.2f%%" % (forest.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_rf))
    print('F1 Score:', f1_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    feature_list = list(x.columns)
    # Get numerical feature importances
    importances = list(forest.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    # New random forest with only >= 0.05 important variables
    rf_most_important = RandomForestClassifier(n_estimators=1000,
                                               min_samples_leaf=30,
                                               oob_score=True,
                                               random_state=100,
                                               class_weight='balanced',
                                               n_jobs=2)
    # Extract the important features
    important_indices = ['int_rate',
                         'revol_util',
                         'dti',
                         'log_installment',
                         'loan_amnt',
                         'mort_acc',
                         'log_annual_inc',
                         'log_revol_bal']
    x_train_i = x_train.loc[:, important_indices]
    x_test_i = x_test.loc[:, important_indices]
    # Train the random forest
    rf_most_important.fit(x_train_i, y_train.values.ravel())
    # Make predictions and determine the error
    predictions = rf_most_important.predict(x_test_i)
    print("Accuracy: %.2f%%" % (rf_most_important.score(x_test_i, y_test) * 100))

    # ### Neural Network Model
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.utils import to_categorical
    from keras.models import load_model
    from sklearn.utils import class_weight

    scaler = StandardScaler()
    scaler.fit(x_train.astype('float64'))
    StandardScaler(copy=True, with_mean=True, with_std=True)
    x_train_nn = scaler.transform(x_train.astype('float64'))
    x_test_nn = scaler.transform(x_test.astype('float64'))
    dump(scaler, Path('../data/scaler.joblib'))

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train.values.ravel()),
                                                      y_train.values.ravel())

    model_nn_path = Path('../data/model_nn.h5')
    if os.path.exists(model_nn_path):
        model_nn = load_model(str(model_nn_path))
    else:
        model_nn = Sequential()
        # Input layer
        model_nn.add(Dense(20, activation='relu', input_shape=(39,)))
        # Hidden layer
        model_nn.add(Dense(15, activation='relu'))
        model_nn.add(Dense(4, activation='relu'))
        # Output layer
        model_nn.add(Dense(2, activation='sigmoid'))
        model_nn.output_shape

        model_nn.compile(loss='binary_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
        model_nn.fit(x_train_nn, to_categorical(y_train), epochs=5, batch_size=10, class_weight=class_weights,
                     verbose=1)

        model_nn.save(str(model_nn_path))
    y_pred_nn = model_nn.predict_classes(x_test_nn)
    print(model_nn.summary())
    # y_pred_nn = [round(x[0] for x in y_pred_nn)]

    # ### Neural Network Results
    score = model_nn.evaluate(x_test_nn, to_categorical(y_test))
    print("Accuracy: %.2f%%" % (score[1] * 100))
    print(confusion_matrix(y_test, y_pred_nn))
    print('F1 Score:', f1_score(y_test, y_pred_nn))
    print(classification_report(y_test, y_pred_nn))
    NN_probs = pd.DataFrame(model_nn.predict(x_test_nn))
    print(NN_probs.head())

    # ### XGBoost Model
    from xgboost import XGBClassifier
    from xgboost import plot_importance
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train.values.ravel()),
                                                      y_train.values.ravel())
    model_xgb = XGBClassifier(scale_pos_weight=class_weights[1])  # random state?
    model_xgb.fit(x_train, y_train.values.ravel())
    y_pred_xgb = model_xgb.predict(x_test)

    # ### XGBoost Results
    print("Accuracy: %.2f%%" % (model_xgb.score(x_test, y_test) * 100))
    print(confusion_matrix(y_test, y_pred_xgb))
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

    # Extract the two most important features
    model_path_xgb = Path('../data/model_xgb.joblib')
    if os.path.exists(model_path_xgb):
        model_xgb = load(model_path_xgb)
    else:
        important_indices = ['dti',
                             'int_rate',
                             'revol_util',
                             'log_installment',
                             'open_acc',
                             'log_revol_bal',
                             'mort_acc',
                             'loan_amnt',
                             'total_acc',
                             'log_annual_inc']
        x_train_xgb = x_train.loc[:, important_indices]
        x_test_xgb = x_test.loc[:, important_indices]

        model_xgb = XGBClassifier(scale_pos_weight=class_weights[1])
        model_xgb.fit(x_train_xgb, y_train.values.ravel())
        y_pred_xgb = model_xgb.predict(x_test_xgb)
        print("Accuracy: %.2f%%" % (model_xgb.score(x_test_xgb, y_test) * 100))
        print(confusion_matrix(y_test, y_pred_xgb))
        plot_importance(model_xgb)
        plt.show()
        dump(model_xgb, model_path_xgb)

    # ### Support Vector Machine Model
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    model_svm_path = Path('../data/model_svm.joblib')
    if os.path.exists(model_svm_path):
        model_svm = load(model_svm_path)
    else:
        # parameter_candidates = [
        #     {'C': [1, 10, 100], 'kernel': ['linear'], 'class_weight': ['balanced']},
        #     {'C': [1, 10, 100], 'gamma': [10, 0, 0.001, 0.0001], 'kernel': ['rbf'], 'class_weight': ['balanced']}
        # ]
        #
        # clf = GridSearchCV(estimator=SVC(), param_grid = parameter_candidates, n_jobs=-1, cv=5)
        # clf.fit(x_train, y_train.values.ravel())
        # print('Best score for data:', clf.best_score_)
        # print('Best C:', clf.best_estimator_.C)
        # print('Best Kernel:', clf.best_estimator_.kernel)
        # print('Best Gamma:', clf.best_estimator_.gamma)
        model_svm = SVC(kernel='rbf', C=1, gamma=0.05, class_weight='balanced', random_state=0)
        model_svm.fit(x_train, y_train.values.ravel())
        dump(model_svm, model_svm_path)
    print('Accuracy of SVM on test set: {:.3f}'.format(model_svm.score(x_test, y_test)))
    # 0.741 C=1 gamma = 0.05
    # 0.777 C=1 gamma = 1

    # ### Ensembled Model
    # RF_probs = pd.DataFrame(forest.predict_proba(x_test))
    # LR_probs = pd.DataFrame(lr_model.predict_proba(x_test))
    # probs
    ensemble_model_path = Path('../data/ensemble_model.joblib')
    if os.path.exists(ensemble_model_path):
        ensemble = load(ensemble_model_path)
    else:
        estimators = [('log_reg', model_lr), ('rf', forest)]
        ensemble = VotingClassifier(estimators, voting='soft', n_jobs=-1)
        ensemble.fit(x_train, y_train.values.ravel())
        # Save model to file
        dump(ensemble, ensemble_model_path)
    y_pred_ensemble = ensemble.predict(x_test)

    # ### Ensembled Results
    print(confusion_matrix(y_test, y_pred_ensemble))
    print('Accuracy of ensemble on test set: {:.3f}'.format(ensemble.score(x_test, y_test.values.ravel())))
