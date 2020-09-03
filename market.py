import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import requests
import datetime
import json
import zipfile

from urllib import request
from collections import OrderedDict
from models import ProfitModel

from process_features import process_features

def get_id(authorization):
    r = requests.post('http://api.bondora.com/api/v1/secondarymarket/98b03d31-ed1b-4338-96d5-aa1c002cdd8a',
                      headers={'Authorization': ('Bearer %s' % authorization)})
    j = r.json()
    return j

def get_primary(authorization, download=True):
    dt = pd.Timestamp(datetime.datetime.today())
    listedon_from = dt.strftime('%Y-%m-%d')
    listedon_to = dt.strftime('%Y-%m-%d')

    if download:
        n = 1
        payload = [[]]
        payloads = []
        while len(payload) > 0:
            print("Fetching %d..." % n)
            r = requests.get('http://api.bondora.com/api/v1/auctions',
                              params={'request.pageSize': 1000, 'request.pageNr': n},
                              headers={'Authorization': ('Bearer %s' % authorization)})
            j = r.json()
            payload = j['Payload']
            payloads.extend(payload)
            n += 1
            time.sleep(1.1)

        payloads = pd.DataFrame(payloads)
        payloads.rename(columns={'AppliedAmount': 'Amount'}, inplace=True)
        payloads['LoanDate'] = dt
        payloads['AmountOfPreviousLoansBeforeLoan'] = 0
        payloads['ExistingLiabilities'] = 0

        cols = ['LoanId', 'UserName', 'Rating', 'Country', 'CreditScoreEsEquifaxRisk',
         'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEeMini',
         'Gender', 'Age', 'Education', 'HomeOwnershipType', 'EmploymentDurationCurrentEmployer',
         'ApplicationSignedHour', 'ApplicationSignedWeekday',
         'LoanDuration', 'Interest', 'Amount', 'VerificationType', 'ExistingLiabilities', 'IncomeTotal',
         'LiabilitiesTotal', 'DebtToIncome', 'AmountOfPreviousLoansBeforeLoan'
         ]
        payloads = payloads[cols]
        payloads.to_csv('datas/LoanPrimary.csv', index=False)
    else:
        payloads = pd.read_csv('datas/LoanPrimary.csv', parse_dates=['LoanDate'])

    return payloads

def get_secondary(authorization, listedon_from='', listedon_to='', download=True):
    dt2 = pd.Timestamp(datetime.datetime.today())
    dt1 = dt2 - pd.Timedelta(1, 'D')
    listedon_from = listedon_from if listedon_from != '' else dt1.strftime('%Y-%m-%d')
    listedon_to = listedon_to if listedon_to != '' else dt2.strftime('%Y-%m-%d')
    print(listedon_from, listedon_to)

    if download:
        n = 1
        payload = [[]]
        payloads = []
        error = False
        while len(payload) > 0 and error == False:
            print("Fetching %d..." % n)
            r = requests.post('http://api.bondora.com/api/v1/secondarymarket',
                              params={'request.pageSize': 1000, 'request.pageNr': n, 'request.loanStatusCode': 2,
                                      'request.listedOnDateFrom': listedon_from, 'request.listedOnDateTo': listedon_to},
                              headers={'Authorization': ('Bearer %s' % authorization)})
            j = r.json()
            try:
                payload = j['Payload']
                payloads.extend(payload)
            except KeyError:
                print(j)
                error = True
            n += 1
            time.sleep(1.1)

        cols = ['Id', 'LoanPartId', 'AuctionId', 'UserName', 'SignedDate', 'LateAmountTotal', 'Interest', 'LoanStatusCode',
                'NextPaymentNr', 'NrOfScheduledPayments', 'PrincipalRemaining', 'Price']
        payloads = pd.DataFrame(payloads)[cols]
        payloads['SignedDate'] = pd.to_datetime(pd.to_datetime(payloads['SignedDate']).dt.date)
        payloads.rename(columns={'SignedDate': 'LoanDate', 'NextPaymentNr': 'n_survived', 'NrOfScheduledPayments': 'n_total'},
                      inplace=True)
        payloads['n_survived'] = payloads['n_survived'] - 1
        payloads.to_csv('datas/LoanSecondary.csv', index=False)
    else:
        payloads = pd.read_csv('datas/LoanSecondary.csv', parse_dates=['LoanDate'])

    return payloads


def buy_secondary(authorization, payload):

    url = "https://api.bondora.com/api/v1/secondarymarket/buy"

    headers = {
        'Accept': "application/json, text/json, application/xml, text/xml",
        'Content-Type': "application/json",
        'Authorization': ('Bearer %s' % authorization)
    }
    response = requests.post(url, data=payload, headers=headers)
    print(response.text)

def get_data(download=True):

    if download:
        print('Downloading...')
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'cashmoney')]
        request.install_opener(opener)
        request.urlretrieve('https://www.bondora.com/marketing/media/LoanData.zip', 'datas/LoanData.zip')
        with zipfile.ZipFile("datas/LoanData.zip", "r") as zip_ref: zip_ref.extractall("datas/")

    print("Reading...")
    loans = pd.read_csv('datas/LoanData.csv', usecols=['LoanId', 'UserName', 'DefaultDate', 'Status', 'LoanDate', 'Rating', 'Country', 'CreditScoreEsEquifaxRisk', 'CreditScoreFiAsiakasTietoRiskGrade', 'CreditScoreEeMini',
                                              'Gender', 'Age', 'Education', 'HomeOwnershipType', 'EmploymentDurationCurrentEmployer', 'ApplicationSignedHour', 'ApplicationSignedWeekday',
                                              'LoanDuration', 'Interest', 'Amount', 'VerificationType', 'ExistingLiabilities', 'IncomeTotal', 'LiabilitiesTotal', 'DebtToIncome', 'AmountOfPreviousLoansBeforeLoan'
                                              ])
    print(len(loans))

    print("Adding user histories...")
    dates = pd.read_csv('datas/UserHistories.csv')
    dates = loans[['LoanId', 'UserName']].merge(dates)
    dates = dates.groupby('LoanId')['inc_intervals', 'inc_current', 'inc_default', 'inc_repaid', 'inc_rescheduled'].agg('sum')
    dates['inc_current'] = dates['inc_current'] - 1
    dates.rename(columns={'inc_intervals': 'PreviousNumber_Intervals', 'inc_current': 'PreviousNumber_Current', 'inc_default': 'PreviousNumber_Default', 'inc_repaid': 'PreviousNumber_Repaid', 'inc_rescheduled': 'PreviousNumber_Rescheduled'}, inplace=True)
    dates.reset_index(inplace=True)
    loans = loans.merge(dates)
    print(len(loans))

    print("Processing...")
    loans = process_features(loans, verbose=False)
    print(len(loans))

    cols = ['LoanId', 'UserName',  'DefaultDate', 'Status', 'LoanDate', 'Interest',
        'Year', 'Rating', 'Rating2',  # Year & Rating
        'Gender', 'Age', 'Education', 'HomeOwnershipType', 'ApplicationSignedHour', 'ApplicationSignedWeekday', 'EmploymentDurationCurrentEmployer', # Person
        'LoanDuration', 'Amount', 'VerificationType', 'IncomeTotal', 'LiabilitiesTotal', # Loan and Income
        'LiabilitiesToIncome', 'LiabilitiesToIncomeUnknown', 'LoansToIncome', 'PaymentToIncome', 'ExistingLiabilities',
        'PreviousNumber_Intervals', 'NoHistory', 'PreviousNumber', 'PreviousNumber_Current', 'PreviousNumber_Default', 'PreviousNumber_Repaid', 'PreviousNumber_Rescheduled' # Payment history
    ]
    return(loans[cols])


def train_model(subset=20000):

    print("Reading...")
    loans = pd.read_csv('datas/LoanData2.csv')
    intervals = pd.read_csv('datas/df_Y.csv')
    residuals = pd.read_csv('datas/df_R.csv')

    print("Filtering...")
    #loans = loans[(loans['Rating'] != '-') & (loans['Year'] > 2012)]
    loanids = np.random.choice(loans['LoanId'], subset, replace=False)
    loans = loans[loans['LoanId'].isin(loanids)]
    intervals = intervals[(intervals['T'] <= 80) & (intervals['LoanId'].isin(loanids))]
    residuals = residuals[(residuals['T'] <= 80) & (residuals['LoanId'].isin(loanids))]
    print("Loans:", len(loans))
    print("Intervals:", len(intervals))
    print("Residuals:", len(residuals))

    # Numeric and categorical columns used for prediction
    col_numeric = ['Interest', 'Age', 'ApplicationSignedHour', 'ApplicationSignedWeekday', 'LoanDuration', 'Amount',
                      'IncomeTotal', 'LiabilitiesTotal', 'LiabilitiesToIncome', 'LoansToIncome', 'PaymentToIncome',
                      'ExistingLiabilities', 'PreviousNumber_Intervals', 'PreviousNumber', 'PreviousNumber_Current',
                      'PreviousNumber_Default', 'PreviousNumber_Repaid', 'PreviousNumber_Rescheduled']
    col_classes = ['Year', 'Rating', 'Rating2', 'Gender', 'Education', 'HomeOwnershipType',
                      'EmploymentDurationCurrentEmployer', 'VerificationType', 'LiabilitiesToIncomeUnknown', 'NoHistory']
    all_classes = OrderedDict([(col, np.sort(loans[col].unique())) for col in col_classes])
    T_classes = np.sort(intervals['T'].unique())

    # Use data set as the training set
    print("Training...")
    model = ProfitModel(col_numeric, col_classes, all_classes, T_classes)
    model.fit(loans, intervals, residuals)

    return(model)

