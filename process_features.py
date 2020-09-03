import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === YEAR & RATING ===
# Year                              Categorical     Extract from LoanDate                       2009, 2010, ..., 2018
# Rating                            Categorical     Fill NA, Downsample 'AA' to 'A'             A, B, C, D, E, F, HR, -
# Rating2                           Categorical     Fill NA, Combine
#                                                   CreditScoreEsEquifaxRisk (ES),              A, B, C, D, -
#                                                   CreditScoreFiAsiakasTietoRiskGrade (FI),    RL1, RL2, RL3, RL4, RL1x, RL2x, RL3x, RL4x, RL5x, -
#                                                   CreditScoreEeMini (EE)                      600, 700, 800, 900, 1000, -
#
# === PERSON ===
# Gender                            Categorical     Fill NA,                                    Male, Female, '-'
# Age                               Numeric         Downsample <18 => 18 and > 70 => 70
# Education                         Categorical     Fill NA                                     Basic education, Vocational education, Primary education, Secondary education, Higher education
# HomeOwnershipType                 Categorical     Fill NA, Downsample                         Council house, Living with parents, Mortgage, Other, Owner, Tenant, -
# ApplicationSignedHour             Numeric         Transform hours +- from 18_00               -12, ..., 0, ... 12
# ApplicationSignedWeekday          Numeric
# EmploymentDurationCurrentEmployer Categorical     Fill NA, Downsample                         1-, 1-5, 5+, TrialPeriod, Retiree, Other
#
# === LOAN AND INCOME ===
# LoanDuration                      Numeric
# Amount                            Numeric
# VerificationType                  Categorical     Fill NA, Downsample                         Income unverified, Income verified, Income and expenses verified
# IncomeTotal                       Numeric
# IncomeUnknown                     Categorical                                                 True, False
# LiabilitiesTotal                  Numeric
# LiabilitiesUnknown                Categorical                                                 True, False
# LiabilitiesToIncome               Numeric
# LiabilitiesToIncomeUnknown        Categorical                                                 True, False
# LoansToIncome                     Numeric
# LoansToIncomeUnknown              Categorical                                                 True, False
# PaymentToIncome                   Numeric
# ExistingLiabilities               Numeric                                                     0, 1, 2, ...
#
# === PAYMENT HISTORY ===
# PreviousNumber_Intervals          Numeric                                                     0, 1, 2, ...
# NoHistory                         Categorical                                                 True, False
# PreviousNumber_Current            Numeric                                                     0, 1, 2, ...
# PreviousNumber_Default            Numeric                                                     0, 1, 2, ...
# PreviousNumber_Repaid             Numeric                                                     0, 1, 2, ...
# PreviousNumber_Rescheduled        Numeric                                                     0, 1, 2, ...


def process_features(fr, verbose=True):

    # === YEAR & RATING ===

    # Year: 2009, 2010, ..., 2018
    fr['Year'] = pd.to_datetime(fr['LoanDate']).dt.year

    fr['Rating'].fillna('-', inplace=True)
    fr.loc[fr['Rating'] == 'AA', 'Rating'] = 'A'
    fr['Rating'] = pd.Categorical(fr['Rating'])
    if verbose: print(fr['Rating'].value_counts(dropna=False).sort_index())

    # CreditScoreEsEquifaxRisk: A, B, C, D
    fr['CreditScoreEsEquifaxRisk'].fillna('-', inplace=True)
    fr['CreditScoreEsEquifaxRisk'] = fr['CreditScoreEsEquifaxRisk'].map({'-': '-', 'AAA': 'A', 'AA': 'A', 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'})
    fr['CreditScoreEsEquifaxRisk'] = pd.Categorical(fr['CreditScoreEsEquifaxRisk'])
    if verbose: print(fr['CreditScoreEsEquifaxRisk'].value_counts(dropna=False).sort_index())

    # CreditScoreFiAsiakasTietoRiskGrade: RL1, RL2, RL3, RL4, RL1x, RL2x, RL3x, RL4x, RL5x, -
    fr.loc[fr['CreditScoreFiAsiakasTietoRiskGrade'] == 'RL0', 'CreditScoreFiAsiakasTietoRiskGrade'] = 'RL1'
    numeric = ~fr['CreditScoreFiAsiakasTietoRiskGrade'].isin(['RL1', 'RL2', 'RL3', 'RL4', 'RL5'])
    fr.loc[numeric, 'CreditScoreFiAsiakasTietoRiskGrade'] = fr.loc[numeric, 'CreditScoreFiAsiakasTietoRiskGrade'].astype(float)
    fr.loc[fr['CreditScoreFiAsiakasTietoRiskGrade'] == 6.0, 'CreditScoreFiAsiakasTietoRiskGrade'] = 5.0
    fr.loc[fr['CreditScoreFiAsiakasTietoRiskGrade'] == 7.0, 'CreditScoreFiAsiakasTietoRiskGrade'] = 5.0
    fr['CreditScoreFiAsiakasTietoRiskGrade'] = fr['CreditScoreFiAsiakasTietoRiskGrade'].astype(str)
    fr.loc[fr['CreditScoreFiAsiakasTietoRiskGrade'] == 'nan', 'CreditScoreFiAsiakasTietoRiskGrade'] = '-'
    fr['CreditScoreFiAsiakasTietoRiskGrade'] = fr['CreditScoreFiAsiakasTietoRiskGrade'].map({'RL1': 'RL1', 'RL2': 'RL2', 'RL3': 'RL3', 'RL4': 'RL4', 'RL5': 'RL4', '-': '-',
                                                                                             '1.0': 'RL1x', '2.0': 'RL2x', '3.0': 'RL3x', '4.0': 'RL4x', '5.0': 'RL5x'})
    fr['CreditScoreFiAsiakasTietoRiskGrade'] = pd.Categorical(fr['CreditScoreFiAsiakasTietoRiskGrade'].fillna('-'))
    if verbose: print(fr['CreditScoreFiAsiakasTietoRiskGrade'].value_counts(dropna=False).sort_index())

    # CreditScoreEeMini: 600, 700, 800, 900, 1000, -
    fr.loc[fr['CreditScoreEeMini'] < 600, 'CreditScoreEeMini'] = 600
    fr['CreditScoreEeMini'].fillna('-', inplace=True)
    fr['CreditScoreEeMini'] = pd.Categorical(fr['CreditScoreEeMini'])
    if verbose: print(fr['CreditScoreEeMini'].value_counts(dropna=False).sort_index())

    # Rating2: ES:A, ES:B, ES:C, ES:D, ES:-, FI:RL1, FI:RL2, FI:RL3, FI:RL4, FI:RL1x, FI:RL2x, FI:RL3x, FI:RL4x, FI:RL5x, FI:-,  EE:600, EE:700, EE:800, EE:900, EE:1000, EE:-
    fr['Rating2'] = np.where(fr['Country'] == 'ES', fr['CreditScoreEsEquifaxRisk'].astype(str), np.where(fr['Country'] == 'FI', fr['CreditScoreFiAsiakasTietoRiskGrade'].astype(str), fr['CreditScoreEeMini'].astype(str)))
    fr['Rating2'] = fr['Country'].str.cat(fr['Rating2'], sep=':')
    fr['Rating2'].replace({'EE:-': 'EE:1000.0', 'ES:-': 'ES:A', 'FI:-':'FI:RL3'}, inplace=True)
    if verbose: print(fr['Rating2'].value_counts(dropna=False).sort_index())


    # === PERSON ===

    # Gender: Male, Female
    fr.loc[~fr['Gender'].isin([0, 1]), 'Gender'] = '-'
    fr['Gender'] = fr['Gender'].map({0: 'Male', 1: 'Female', '-': '-'})
    fr['Gender'] = pd.Categorical(fr['Gender'])
    if verbose: print(fr['Gender'].value_counts(dropna=False).sort_index())

    # Age: 18, 19, ..., 70
    fr.loc[(fr['Age'] < 18), 'Age'] = 18
    fr.loc[(fr['Age'] > 70), 'Age'] = 70
    if verbose: print(fr['Age'].value_counts(dropna=False).sort_index())

    # Education: Basic education, Vocational education, Primary education, Secondary education, Higher education
    fr.loc[~fr['Education'].isin([1, 2, 3, 4, 5]), 'Education'] = '-'  # np.nan
    fr['Education'] = fr['Education'].map({1: 'Primary education', 2: 'Basic education',
                                           3: 'Vocational education', 4: 'Secondary education',
                                           5: 'Higher education', '-': 'Secondary education'})
    fr['Education'] = pd.Categorical(fr['Education'])
    if verbose: print(fr['Education'].value_counts(dropna=False).sort_index())

    # HomeOwnershipType: Council house, Living with parents, Mortgage, Other, Owner, Tenant, -
    fr.loc[~fr['HomeOwnershipType'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 'HomeOwnershipType'] = '-'
    fr.loc[fr['HomeOwnershipType'].isnull(), 'HomeOwnershipType'] = '-'
    fr['HomeOwnershipType'] = fr['HomeOwnershipType'].map({0: 'Homeless', 1: 'Owner', 2: 'Living with parents',
                                                           3: 'Tenant pre-furnished property',
                                                           4: 'Tenant, unfurnished property',
                                                           5: 'Council house', 6: 'Joint tenant', 7: 'Joint ownership',
                                                           8: 'Mortgage', 9: 'Owner with encumbrance', 10: 'Other',
                                                           '-': 'Other'})
    fr['HomeOwnershipType'] = pd.Categorical(fr['HomeOwnershipType'].map({
        'Owner': 'Owner',
        'Tenant pre-furnished property': 'Tenant',
        'Living with parents': 'Living with parents',
        'Mortgage': 'Mortgage',
        'Tenant, unfurnished property': 'Tenant',
        'Joint ownership': 'Owner',
        'Joint tenant': 'Tenant',
        'Council house': 'Council house',
        'Owner with encumbrance': 'Owner',
        'Other': 'Other',
        'Homeless': 'Other',
        '-': '-'
    }))
    if verbose: print(fr['HomeOwnershipType'].value_counts(dropna=False).sort_index())

    # ApplicationSignedHour: Midnight, Not Midnight
    #fr['ApplicationSignedHour'] = pd.Categorical(pd.cut(fr['ApplicationSignedHour'], bins=[0, 6, 24], right=True, include_lowest=True, labels=['Midnight', 'Not Midnight']).astype(str))
    fr['ApplicationSignedHour'] = ((fr['ApplicationSignedHour'] - 6) % 24 - 12) # Hours before or after 18:00
    if verbose: print(fr['ApplicationSignedHour'].value_counts(dropna=False).sort_index())

    # ApplicationSignedWeekday: 1, 2, ..., 7
    if verbose: print(fr['ApplicationSignedWeekday'].value_counts(dropna=False).sort_index())

    # EmploymentDurationCurrentEmployer: 1 -, 1 - 5, 5 +, TrialPeriod, Retiree, Other
    fr.loc[fr['EmploymentDurationCurrentEmployer'].isnull(), 'EmploymentDurationCurrentEmployer'] = '-'
    fr['EmploymentDurationCurrentEmployer'] = pd.Categorical(fr['EmploymentDurationCurrentEmployer'].map({
        'MoreThan5Years': '5+',
        'UpTo1Year': '1-',
        'UpTo5Years': '1-5,',
        'UpTo2Years': '1-5,',
        'UpTo3Years': '1-5,',
        'UpTo4Years': '1-5,',
        'Retiree': 'Retiree',
        'Other': 'Other',
        'TrialPeriod': 'TrialPeriod',
        '-': '-'
    }))
    if verbose: print(fr['EmploymentDurationCurrentEmployer'].value_counts(dropna=False).sort_index())


    # === LOAN AND INCOME ===

    # LoanDuration
    if verbose: print(fr['LoanDuration'].value_counts(dropna=False).sort_index())

    # Amount
    if verbose:
        plt.figure()
        np.log(fr['Amount']).hist(bins=100).set_title('Amount (log)')

    # VerificationType: Income unverified, Income verified, Income and expenses verified
    fr.loc[~fr['VerificationType'].isin([1, 2, 3, 4]), 'VerificationType'] = 1
    fr['VerificationType'] = fr['VerificationType'].map({1: 'Income unverified',
                                                         2: 'Income unverified, cross-referenced by phone',
                                                         3: 'Income verified', 4: 'Income and expenses verified',
                                                         '-': '-'})
    fr['VerificationType'] = pd.Categorical(fr['VerificationType'].map({
        'Income and expenses verified': 'Income and expenses verified',
        'Income unverified': 'Income unverified',
        'Income verified': 'Income verified',
        'Income unverified, cross-referenced by phone': 'Income unverified'
    }))
    if verbose: print(fr['VerificationType'].value_counts(dropna=False).sort_index())

    # IncomeTotal
    # IncomeUnknown: True, False
    fr.loc[fr['IncomeTotal'] <= 0, 'IncomeTotal'] = np.nan
    fr['IncomeUnknown'] = pd.Categorical(fr['IncomeTotal'].isnull())
    fr['IncomeTotal'].fillna(fr['IncomeTotal'].mean(), inplace=True)
    if verbose:
        plt.figure()
        np.log(fr.loc[fr['IncomeUnknown'] == False, 'IncomeTotal']).hist(bins=100).set_title('IncomeTotal (log)')
        print(fr['IncomeUnknown'].value_counts(dropna=False).sort_index())

    # LiabilitiesTotal
    # LiabilitiesUnknown: True, False
    fr.loc[fr['LiabilitiesTotal'] <= 0, 'LiabilitiesTotal'] = np.nan
    fr['LiabilitiesUnknown'] = pd.Categorical(fr['LiabilitiesTotal'].isnull())
    fr['LiabilitiesTotal'].fillna(fr['LiabilitiesTotal'].mean(), inplace=True)
    if verbose:
        plt.figure()
        np.log(fr.loc[fr['LiabilitiesUnknown'] == False, 'LiabilitiesTotal']).hist(bins=100).set_title('LiabilitiesTotal (log)')
        print(fr['LiabilitiesUnknown'].value_counts(dropna=False).sort_index())

    # LiabilitiesToIncome
    # LiabilitiesToIncomeUnknown: True, False
    missing_dti = pd.to_datetime(fr['LoanDate']) >= pd.Timestamp('2017-06-01')
    fr['LiabilitiesToIncome'] = np.where(missing_dti, fr['LiabilitiesTotal']/fr['IncomeTotal'], fr['DebtToIncome']/100.0)
    fr.loc[fr['LiabilitiesToIncome'] > 0.7, 'LiabilitiesToIncome'] = 0.7
    fr['LiabilitiesToIncomeUnknown'] = pd.Categorical((fr['IncomeUnknown']) | (fr['LiabilitiesUnknown']) | fr['LiabilitiesToIncome'].isnull())
    fr['LiabilitiesToIncome'].fillna(fr['LiabilitiesToIncome'].mean(), inplace=True)
    fr['LiabilitiesToIncomeMaxed'] = pd.Categorical(fr['LiabilitiesToIncome'] == 0.7)
    if verbose:
        plt.figure()
        fr.loc[(fr['LiabilitiesToIncomeUnknown'] == False) & (fr['LiabilitiesToIncomeMaxed'] == False), 'LiabilitiesToIncome'].hist(bins=100).set_title('LiabilitiesToIncome')
        print(fr['LiabilitiesToIncomeUnknown'].value_counts(dropna=False).sort_index())
        print(fr['LiabilitiesToIncomeMaxed'].value_counts(dropna=False).sort_index())

    # LoansToIncome
    # LoansToIncomeUnknown: True, False
    fr['LoansToIncome'] = fr['AmountOfPreviousLoansBeforeLoan']/fr['IncomeTotal']
    fr.loc[fr['LoansToIncome'] > 20.0, 'LoansToIncome'] = 20.0
    fr['LoansToIncomeUnknown'] = pd.Categorical((fr['LoansToIncome'] == 0.0) | fr['IncomeUnknown'])
    if verbose:
        plt.figure()
        fr.loc[fr['LoansToIncomeUnknown'] == False, 'LoansToIncome'].hist(bins=100).set_title('LoansToIncome')
        print(fr['LoansToIncomeUnknown'].value_counts(dropna=False).sort_index())

    # PaymentToIncome
    I = fr['Interest'] / 12 / 100
    n = fr['LoanDuration']
    M = fr['Amount']
    P = M * I * (1 + I) ** n / ((1 + I) ** n - 1)
    fr['PaymentToIncome'] = P / fr['IncomeTotal']
    fr.loc[fr['PaymentToIncome'] > 0.5, 'PaymentToIncome'] = 0.5
    if verbose:
        plt.figure()
        fr['PaymentToIncome'].hist(bins=100).set_title('PaymentToIncome')

    # ExistingLiabilities: 0, 1, 2, ...
    if verbose: print(fr['ExistingLiabilities'].value_counts(dropna=False).sort_index())

    ## IncomeFromPrincipalEmployer (True, Some, False)
    ## IncomeFromPension (True, False), IncomeFromFamilyAllowance (True, False), IncomeFromSocialWelfare (True, False),
    ## IncomeFromLeavePay (True, False), IncomeFromChildSupport (True, False), IncomeOther (True, False)
    #fr['IncomeFromPrincipalEmployer'] = fr['IncomeFromPrincipalEmployer'] / fr['IncomeTotal']
    #fr['IncomeFromPrincipalEmployer'] = pd.Categorical(np.where(fr['IncomeFromPrincipalEmployer'] == 0, 'False', np.where(fr['IncomeFromPrincipalEmployer'] == 1, 'True', 'Some')))
    #fr['IncomeFromPension'] = fr['IncomeFromPension'] / fr['IncomeTotal']
    #fr['IncomeFromPension'] = pd.Categorical(fr['IncomeFromPension'] > 0)
    #fr['IncomeFromFamilyAllowance'] = fr['IncomeFromFamilyAllowance'] / fr['IncomeTotal']
    #fr['IncomeFromFamilyAllowance'] = pd.Categorical(fr['IncomeFromFamilyAllowance'] > 0)
    #fr['IncomeFromSocialWelfare'] = fr['IncomeFromSocialWelfare'] / fr['IncomeTotal']
    #fr['IncomeFromSocialWelfare'] = pd.Categorical(fr['IncomeFromSocialWelfare'] > 0)
    #fr['IncomeFromLeavePay'] = fr['IncomeFromLeavePay'] / fr['IncomeTotal']
    #fr['IncomeFromLeavePay'] = pd.Categorical(fr['IncomeFromLeavePay'] > 0)
    #fr['IncomeFromChildSupport'] = fr['IncomeFromChildSupport'] / fr['IncomeTotal']
    #fr['IncomeFromChildSupport'] = pd.Categorical(fr['IncomeFromChildSupport'] > 0)
    #fr['IncomeOther'] = fr['IncomeOther'] / fr['IncomeTotal']
    #fr['IncomeOther'] = pd.Categorical(fr['IncomeOther'] > 0)
    #if verbose: print(fr['IncomeFromPrincipalEmployer'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeFromPension'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeFromFamilyAllowance'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeFromSocialWelfare'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeFromLeavePay'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeFromChildSupport'].value_counts(dropna=False).sort_index())
    #if verbose: print(fr['IncomeOther'].value_counts(dropna=False).sort_index())

    # PreviousNumber_Intervals: 0, 1, 2, ...
    if verbose: print(fr['PreviousNumber_Intervals'].value_counts(dropna=False).sort_index())

    # PreviousNumber: 0, 1, 2, ...
    fr['PreviousNumber'] = fr['PreviousNumber_Current'] + fr['PreviousNumber_Default'] + fr['PreviousNumber_Repaid']
    if verbose: print(fr['PreviousNumber'].value_counts(dropna=False).sort_index())

    # NoHistory: True, False
    fr['NoHistory'] = pd.Categorical(fr['PreviousNumber'] == 0)
    if verbose: print(fr['NoHistory'].value_counts(dropna=False).sort_index())

    # PreviousNumber Current / Default / Repaid
    if verbose: print(fr['PreviousNumber_Current'].value_counts(dropna=False).sort_index())
    if verbose: print(fr['PreviousNumber_Default'].value_counts(dropna=False).sort_index())
    if verbose: print(fr['PreviousNumber_Repaid'].value_counts(dropna=False).sort_index())

    # PreviousNumber Rescheduled
    if verbose: print(fr['PreviousNumber_Rescheduled'].value_counts(dropna=False).sort_index())

    return(fr)

def process_start(verbose=True):
    print("Reading...")
    loans = pd.read_csv('datas/LoanData.csv', usecols=['LoanId', 'UserName', 'LoanDate', 'Rating', 'Country', 'CreditScoreEsEquifaxRisk', 'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEeMini',
                                              'Gender', 'Age', 'Education', 'HomeOwnershipType', 'EmploymentDurationCurrentEmployer', 'ApplicationSignedHour', 'ApplicationSignedWeekday',
                                              'LoanDuration', 'Interest', 'Amount', 'VerificationType', 'ExistingLiabilities', 'IncomeTotal', 'LiabilitiesTotal', 'DebtToIncome', 'AmountOfPreviousLoansBeforeLoan'
                                              ])
    print(len(loans))

    print("Adding user histories...")
    dates = pd.read_csv('datas/UserHistories.csv')
    dates = loans[['LoanId', 'LoanDate', 'UserName']].merge(dates)
    dates = dates[dates['Date'] <= dates['LoanDate']]
    dates = dates.groupby('LoanId')['inc_intervals', 'inc_current', 'inc_default', 'inc_repaid', 'inc_rescheduled'].agg('sum')
    dates['inc_current'] = dates['inc_current'] - 1
    dates.rename(columns={'inc_intervals': 'PreviousNumber_Intervals', 'inc_current': 'PreviousNumber_Current',
                          'inc_default': 'PreviousNumber_Default', 'inc_repaid': 'PreviousNumber_Repaid',
                          'inc_rescheduled': 'PreviousNumber_Rescheduled'}, inplace=True)
    dates.reset_index(inplace=True)
    loans = loans.merge(dates)
    print(len(loans))

    print("Processing...")
    loans = process_features(loans, verbose=verbose)
    print(len(loans))

    print("Writing...")
    cols = ['LoanId',
        'Interest', 'Year', 'Rating', 'Rating2',  # Year & Rating
        'Gender', 'Age', 'Education', 'HomeOwnershipType', 'ApplicationSignedHour', 'ApplicationSignedWeekday', 'EmploymentDurationCurrentEmployer', # Person
        'LoanDuration', 'Amount', 'VerificationType', 'IncomeTotal', 'LiabilitiesTotal', # Loan and Income
        'LiabilitiesToIncome', 'LiabilitiesToIncomeUnknown', 'LoansToIncome', 'PaymentToIncome', 'ExistingLiabilities',
        'PreviousNumber_Intervals', 'NoHistory', 'PreviousNumber', 'PreviousNumber_Current', 'PreviousNumber_Default', 'PreviousNumber_Repaid', 'PreviousNumber_Rescheduled' # Payment history
    ]
    loans[cols].to_csv('datas/LoanData2.csv', index=False)

if __name__ == '__main__':
    process_start()