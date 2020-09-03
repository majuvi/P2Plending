import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

def process_start(verbose=True):

    cols = ['UserName', 'LoanId', 'Status', 'LoanDate', 'MonthlyPaymentDay', 'FirstPaymentDate', 'MaturityDate_Last',
            'ReportAsOfEOD', 'ContractEndDate', 'DefaultDate', 'ReScheduledOn']

    loans = pd.read_csv('datas/LoanData.csv', sep=',', usecols=cols)
    loans['LoanDate'] = pd.to_datetime(loans['LoanDate'])
    loans['FirstPaymentDate'] = pd.to_datetime(loans['FirstPaymentDate'])
    loans['MaturityDate_Last'] = pd.to_datetime(loans['MaturityDate_Last'])
    loans['ReportAsOfEOD'] = pd.to_datetime(loans['ReportAsOfEOD'])
    loans['ContractEndDate'] = pd.to_datetime(loans['ContractEndDate'])
    loans['DefaultDate'] = pd.to_datetime(loans['DefaultDate'])
    loans['ReScheduledOn'] = pd.to_datetime(loans['ReScheduledOn'])

    dates = []
    i = 1
    n = len(loans['UserName'].unique())
    for username, userloans in loans.groupby('UserName'):
        if (i % 100 == 0) and verbose:
            print(i, "/", n, "(", username, ")")
        payment_dates = []
        for asdf, loan in userloans.iterrows():
            m = loan['MonthlyPaymentDay']
            P0 = loan['LoanDate']
            Ps = loan['FirstPaymentDate']
            Pe = loan['MaturityDate_Last']
            Pf = loan['DefaultDate'] if not pd.isnull(loan['DefaultDate']) else (
                loan['ContractEndDate'] if loan['Status'] == 'Repaid' else loan['ReportAsOfEOD'])
            s, e = pd.Timestamp(year=Ps.year, month=Ps.month, day=1), pd.Timestamp(year=Pe.year, month=Pe.month, day=1)
            dm = pd.date_range(s, e, freq='MS') + pd.Timedelta(m - 1, 'D') + 0 * BusinessDay()
            payment_dates.extend(dm[(dm > P0) & (dm <= Pf)])
        loan_dates = userloans['LoanDate']
        default_dates = userloans.loc[~userloans['DefaultDate'].isnull(), 'DefaultDate']
        repaid_dates = userloans.loc[
            userloans['DefaultDate'].isnull() & (userloans['Status'] == 'Repaid'), 'ContractEndDate']
        rescheduled_dates = userloans.loc[~userloans['ReScheduledOn'].isnull(), 'ReScheduledOn']
        dates.extend([(username, t, 1, 0, 0, 0, 0) for t in payment_dates] +
                     [(username, t, 0, 1, 0, 0, 0) for t in loan_dates] +
                     [(username, t, 0, -1, 1, 0, 0) for t in default_dates] +
                     [(username, t, 0, -1, 0, 1, 0) for t in repaid_dates] +
                     [(username, t, 0, 0, 0, 0, 1) for t in rescheduled_dates])
        i += 1
    dates = pd.DataFrame(dates, columns=['UserName', 'Date', 'inc_intervals', 'inc_current', 'inc_default', 'inc_repaid', 'inc_rescheduled'])
    dates.sort_values(['UserName', 'Date'], inplace=True)

    #print(dates.groupby('UserName').agg('sum'))

    dates.to_csv('datas/UserHistories.csv', index=False)

if __name__ == '__main__':
    process_start()
