import numpy as np
import pandas as pd
from pandas.tseries.offsets import BusinessDay

def process_start(verbose=True):
    cols = ['LoanId', 'Status', 'LoanDate', 'FirstPaymentDate', 'ContractEndDate', 'ReportAsOfEOD', 'DefaultDate', 'MonthlyPaymentDay', 'EAD1']
    loans = pd.read_csv('datas/LoanData.csv', sep=',', usecols=cols)
    loans = loans[~((loans['Status'] == 'Repaid') & (loans['ContractEndDate'].isnull()))]
    loans['LoanDate'] = pd.to_datetime(loans['LoanDate'])
    loans['FirstPaymentDate'] = pd.to_datetime(loans['FirstPaymentDate'])
    loans['ContractEndDate'] = pd.to_datetime(loans['ContractEndDate'])
    loans['ReportAsOfEOD'] = pd.to_datetime(loans['ReportAsOfEOD'])
    loans['DefaultDate'] = pd.to_datetime(loans['DefaultDate'])

    payments = pd.read_csv('datas/RepaymentsData.csv', sep=',')
    payments['Date'] = pd.to_datetime(payments['Date'])

    # Processing:
    #   with default date: take DefaultDate interval, status = 1
    #   no default date & repaid: take ContractEndDate interval, status = 0
    #   no default date & current: take ReportAsOfEOD interval, status = 0
    #   no default date & late: take ReportAsOfEOD interval, status = 0

    # For some, monthly payment date is the date of first payment, for some it is monthly payment day, first interval is approx uniformly 30-60 days
    df_T = []
    df_Y = []
    df_M = []
    i = 1
    for asdf, loan in loans.iterrows():
        loanid = loan['LoanId']

        if (i % 100 == 0) and verbose:
            print(i, "/", len(loans), "(", loanid, ")")

        P0 = loan['LoanDate']
        Ps = loan['FirstPaymentDate']
        Pe = loan['ReportAsOfEOD']
        m = loan['MonthlyPaymentDay']#Ps.day

        s, e = pd.Timestamp(year=Ps.year, month=Ps.month, day=1), pd.Timestamp(year=Pe.year, month=Pe.month, day=1)
        payment_dates = pd.date_range(s, e, freq='MS') + pd.Timedelta(m-1,'D') + 0*BusinessDay()
        payment_dates = payment_dates[(payment_dates > P0) & (payment_dates < Pe)]
        index = pd.Series(pd.DatetimeIndex([P0]).append(payment_dates))
        intervals = pd.DataFrame({'interval': np.arange(1, len(index)+1), 'start': index, 'end': index.shift(-1).fillna(Pe)})

        dt = loan['DefaultDate']
        if not pd.isnull(dt):
            shift = 3
            t = dt
        elif loan['Status'] == 'Repaid':
            shift = 1
            t = loan['ContractEndDate']
        else:
            shift = 3
            t = loan['ReportAsOfEOD']

        T = intervals.loc[(t >= intervals['start']) & (t <= intervals['end']), 'interval'].iloc[0]
        T = T - shift
        if T >= 1:
            idx = np.arange(T) + 1
            seq = np.zeros_like(idx)
            if not pd.isnull(dt):
                seq[-1] = 1
            df_Y.extend([(loanid, t, y) for t, y in zip(idx, seq)])

        df_payments = payments.loc[payments['loan_id'] == loanid, ['Date', 'PrincipalRepayment', 'InterestRepayment', 'LateFeesRepayment']]
        if len(payment_dates) > 1:
            df_payments = df_payments.set_index('Date').sort_index().sum(axis=1)
            df_payments.index = pd.cut(df_payments.index, bins=index)
            df_payments = df_payments.groupby(level=0).agg('sum')
            idx = np.arange(1,len(index))
            seq = df_payments.values
            dts = (dt <= index.iloc[1:]).values
            df_M.extend([(loanid, t, p, s) for t, p, s in zip(idx, seq, dts)])

        i += 1


    print("Samples:")
    dfs = pd.DataFrame(df_Y, columns=['LoanId', 'T', 'Y'])
    dfs.to_csv('datas/df_Y.csv', index=False)
    print(len(dfs['LoanId'].unique()), "/", len(loans['LoanId'].unique()))

    dfs = pd.DataFrame(df_M, columns=['LoanId', 'T', 'M', 'Default'])
    dfs.to_csv('datas/df_M.csv', index=False)
    print(len(dfs['LoanId'].unique()), "/", len(loans['LoanId'].unique()))

    dfs['T'] = dfs.groupby('LoanId')['Default'].agg('cumsum').astype(int)
    dfs = dfs[dfs['Default']].copy()
    dfs = loans.loc[loans['EAD1'] > 400, ['LoanId', 'EAD1']].merge(dfs)
    dfs['Y'] = dfs['M'] / dfs['EAD1']
    dfs = dfs[['LoanId', 'T', 'Y']]
    dfs.to_csv('datas/df_R.csv', index=False)
    print(len(dfs['LoanId'].unique()), "/", len(loans['LoanId'].unique()))

    # Mean Cumulative Function
    #print(dfs.groupby('T')['Repayment'].mean())

if __name__ == '__main__':
    process_start()


