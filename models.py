import numpy as np
import pandas as pd

from scipy.optimize import newton, bisect
from scipy.linalg import toeplitz

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn import metrics

class Model(object):

    def __init__(self, col_numeric, col_classes, all_classes, T_classes=None):
        self.T_stratify = not T_classes is None

        # Numeric, categorical and resulting categorical 'feature=value' columns
        self.col_numeric = col_numeric
        self.col_classes = col_classes
        self.col_onehots = ['%s=%s' % (feature, category) for feature in all_classes for category in all_classes[feature]]

        # Feature transformers
        self.quants = preprocessing.QuantileTransformer()
        self.onehot = preprocessing.OneHotEncoder(categories=list(all_classes.values()))

        if not self.T_stratify:
            self.col_t = ['T'] #['First', 'T']
            self.times = preprocessing.MinMaxScaler()
        else:
            self.col_t = ['T=%s' % T for T in T_classes]
            self.times = preprocessing.OneHotEncoder(categories=[T_classes])

        # All columns
        self.col = self.col_numeric + self.col_onehots + self.col_t

    def _encode_X(self, loans, fit):
        # Continuous & Categorical features
        if fit:
            X1 = self.quants.fit_transform(loans[self.col_numeric])
            X2 = self.onehot.fit_transform(loans[self.col_classes]).toarray()
        else:
            X1 = self.quants.transform(loans[self.col_numeric])
            X2 = self.onehot.transform(loans[self.col_classes]).toarray()

        X1 = pd.DataFrame(X1, index=loans.index, columns=self.col_numeric)
        X2 = pd.DataFrame(X2, index=loans.index, columns=self.col_onehots)
        X = pd.concat([X1, X2], axis=1)
        return(X)

    def _encode_T(self, intervals, fit):
        # Loan features
        if fit:
            T = self.times.fit_transform(intervals[['T']].astype(float))
        else:
            T = self.times.transform(intervals[['T']].astype(float))

        T = T.toarray() if self.T_stratify else T
        T = pd.DataFrame(T, index=intervals.index, columns=self.col_t)
        return(T)


    def encode_features(self, loans, intervals, fit=True):

        # Transform into encoded features
        X = self._encode_X(loans, fit)
        T = self._encode_T(intervals, fit)

        # Combine into data frame
        X['LoanId'] = loans['LoanId']
        T['LoanId'] = intervals['LoanId']
        XT = X.merge(T)

        return(XT)

    def fit(self, loans, intervals):
        pass

    def predict(self, loans, intervals):
        pass


class DefaultModel(Model):

    def fit(self, loans, intervals, C=1.0):
        XT = self.encode_features(loans, intervals) # LoanId, X1, ..., Xd, T1, ..., Tk
        X = XT[self.col].values
        Y = intervals['Y'].values
        self.clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, C=C)
        self.clf.fit(X, Y)

    def predict(self, loans, intervals):
        XT = self.encode_features(loans, intervals, fit=False) # LoanId, X1, ..., Xd, T1, ..., Tk
        X = XT[self.col].values
        P = self.clf.predict_proba(X)[:, 1]
        return(P)

    def coefs(self):
        return pd.Series(np.array(self.clf.coef_.flat), index=pd.Categorical(self.col))

    def risk(self, loans):
        X = self._encode_X(loans, fit=False)
        # Time-invariant risk score
        col = self.col_numeric + self.col_onehots
        risk = pd.Series(np.dot(X[col], self.coefs()[col]), index=loans['LoanId'].values)
        return(risk)

    # https://www.ibm.com/developerworks/community/blogs/jfp/entry/Fast_Computation_of_AUC_ROC_score?lang=en
    def _fast_auc(self, Y, P, split=False):
        y_true = np.asarray(Y)
        y_true = y_true[np.argsort(P)]
        nfalse = 0
        auc = 0
        n = len(y_true)
        for i in range(n):
            y_i = y_true[i]
            nfalse += (1 - y_i)
            auc += y_i * nfalse
        npairs = nfalse * (n - nfalse)
        return auc / npairs if not split else (auc, npairs)

    # Compute standard AUC
    def auc(self, Y, P):
        return self._fast_auc(Y, P)

    # Compute time-stratified AUC
    def auc_stratified(self, Y, P, T):
        fr = pd.DataFrame({'Y': np.array(Y), 'P': np.array(P), 'T': np.array(T)})
        nc, n = 0, 0
        for T, s in fr.groupby('T'):
            if len(s['Y'].unique()) > 1:
                nc_, n_ = self._fast_auc(s['Y'], s['P'], split=True)
                nc += nc_
                n += n_
        return nc/n

    # Compute time-varying AUC
    def auc_t(self, Y, P, T):
        fr = pd.DataFrame({'Y': np.array(Y), 'P': np.array(P), 'T': np.array(T)})
        auc = fr.groupby('T').apply(lambda s: metrics.roc_auc_score(s['Y'], s['P']) if len(s['Y'].unique()) > 1 else np.nan)
        return auc

class LGDModel(Model):

    def fit(self, loans, intervals, C=1.0):
        XT = self.encode_features(loans, intervals) # LoanId, X1, ..., Xd, T1, ..., Tk
        X = XT[self.col].values
        Y = intervals['Y'].values
        self.clf = Ridge(random_state=0, max_iter=1000, alpha=1.0/C)
        self.clf.fit(X, Y)

    def predict(self, loans, intervals):
        XT = self.encode_features(loans, intervals, fit=False) # LoanId, X1, ..., Xd, T1, ..., Tk
        X = XT[self.col].values
        P = self.clf.predict(X)
        return(P)

    # Compute standard MSE
    def mse(self, Y, P):
        mse = metrics.mean_squared_error(Y, P)
        return mse

    # Compute time-varying MSE
    def mse_t(self, Y, P, T):
        fr = pd.DataFrame({'Y': np.array(Y), 'P': np.array(P), 'T': np.array(T)})
        mse = fr.groupby('T').apply(lambda s: metrics.mean_squared_error(s['Y'], s['P']) if len(s['Y'].unique()) > 1 else np.nan)
        return mse

    def coefs(self):
        return pd.Series(np.array(self.clf.coef_.flat), index=pd.Categorical(self.col))

    def lgd(self, loans, discount_rate=0.10, verbose=False):
        i = 1
        lgds = []
        tmax = len(self.col_t) if self.T_stratify else self.times.data_max_[0]
        intervals = pd.DataFrame({'T': np.arange(tmax) + 1, 'LoanId': ''})
        for idx, loan in loans.iterrows():
            if (i % 100 == 0) & verbose:
                print(i, "/", len(loans), "(", loan['LoanId'], ")")
            intervals['LoanId'] = loan['LoanId']
            loan = loan.to_frame().transpose()
            pt = self.predict(loan, intervals)
            pt[pt < 0.00] = 0.00
            dt = (1 + discount_rate / 12) ** (np.arange(tmax) + 1)
            lgd = (pt / dt).sum() - 1.0
            lgds.append(lgd)
            i += 1
        lgds = pd.Series(lgds, index=loans['LoanId'].values)
        return (lgds)

class ProfitModel(object):

    def __init__(self, col_numeric, col_classes, all_classes, T_classes=None):
        self.haz = DefaultModel(col_numeric, col_classes, all_classes, T_classes)
        self.lgd = LGDModel(col_numeric, col_classes, all_classes, T_classes)
        self.risks = preprocessing.QuantileTransformer()

    def fit(self, loans, intervals, residuals, C1=1.0, C2=1.0):
        self.haz.fit(loans, intervals, C=C1)
        self.lgd.fit(loans, residuals, C=C2)
        beta = self.haz.risk(loans)
        risk = self.risks.fit(beta.values.reshape(-1, 1))

    def credit_risk(self, loans):
        beta = self.haz.risk(loans)
        risk = self.risks.transform(beta.values.reshape(-1, 1))
        risk = pd.Series(risk.flat, beta.index)
        return(risk)

    def loss_given_default(self, loans):
        lgds = self.lgd.lgd(loans)
        return(lgds)

    def _predict_loan(self, loan, tmin, tmax, classify=True):
        intervals = pd.DataFrame({'T': np.arange(tmin, tmax+1), 'LoanId': ''})
        intervals['LoanId'] = loan['LoanId']
        loan = loan.to_frame().transpose()
        if classify:
            pt = self.haz.predict(loan, intervals)
        else:
            pt = self.lgd.predict(loan, intervals)
        return pt

    def predict(self, loans, t=60, classify=True, verbose=False):
        intervals = []
        i = 1
        for idx, loan in loans.iterrows():
            if (i % 100 == 0) & verbose:
                print(i, "/", len(loans), "(", loan['LoanId'], ")")
            pt = self._predict_loan(loan, 1, t, classify)
            intervals.append(pt)
            i += 1
        intervals = pd.DataFrame(intervals, columns=np.arange(1, t+1), index=loans['LoanId'].values)
        return(intervals)

    def _compute_cashflow(self, ht, rt, I):

        n = len(ht)
        P = I * (1 + I) ** n / ((1 + I) ** n - 1)
        t = np.arange(n) + 1
        Bs = []
        B = 1
        for x in t:
            B = B * (1 + I)
            Bs.append(B)
            B = B - P
        Bs = np.array(Bs)

        S = ((1 - ht).cumprod()).values
        f = ((1 - ht).cumprod().shift(1).fillna(1) * ht).values
        rt = rt.values

        nr = len(rt)
        rtx = np.concatenate([[0.00], rt[:-1]])
        R = toeplitz(np.zeros(n), rtx)
        R = np.roll(R, 2, axis=1)
        R = np.triu(R, 3)
        R = (f * Bs).reshape(-1, 1) * R
        R = R.sum(axis=0)

        C = np.concatenate([P * S, [0.00] * (nr - n)])
        return(C, R)

    def predict_cashflow(self, loans, t=60, verbose=False):
        ts = np.arange(t) + 1
        cashflows = []
        i = 1
        for idx, loan in loans.iterrows():
            loanid = loan['LoanId']
            if (i % 100 == 0) & verbose:
                print(i, "/", len(loans), "(", loanid, ")")
            interest = loan['Interest']
            tmin = loan['n_survived'] + 1
            tmax = loan['n_total']
            haz = pd.Series(self._predict_loan(loan, tmin, tmax, classify=True))
            lgd = pd.Series(self._predict_loan(loan, 1, t, classify=False))
            pt, rt  = self._compute_cashflow(haz, lgd, interest)
            rt[rt < 0.00] = 0.00
            ct = pt + rt
            dt = (1 + interest) ** ts
            dcf = (ct / dt).sum()
            if dcf > 1.0:
                ct = ct / dcf
            cashflows.append(ct)
            i += 1
        cashflows = pd.DataFrame(cashflows, columns=ts, index=loans['LoanId'].values)
        return(cashflows)

    def predict_stats(self, loans, t=60, verbose=False):
        ts = np.arange(t) + 1
        hazs = []
        lgds = []
        cashflow_current = []
        cashflow_default = []
        i = 1
        for idx, loan in loans.iterrows():
            loanid = loan['LoanId']
            if (i % 100 == 0) & verbose:
                print(i, "/", len(loans), "(", loanid, ")")
            interest = loan['Interest']
            tmin = loan['n_survived'] + 1
            tmax = loan['n_total']
            haz = pd.Series(self._predict_loan(loan, tmin, tmax, classify=True))
            lgd = pd.Series(self._predict_loan(loan, 1, t, classify=False))
            pt, rt  = self._compute_cashflow(haz, lgd, interest)
            rt[rt < 0.00] = 0.00
            ct = pt + rt
            dt = (1 + interest) ** ts
            dcf = (ct / dt).sum()
            if dcf > 1.0:
                pt = pt / dcf
                rt = rt / dcf
            haz_ = np.zeros(t)*np.nan
            haz_[0:len(haz)] = haz
            lgd_ = lgd.values
            lgd_[lgd_ < 0.00] = 0.00
            hazs.append(haz_)
            lgds.append(lgd_)
            cashflow_current.append(pt)
            cashflow_default.append(rt)
            i += 1
        hazs = pd.DataFrame(hazs, columns=ts, index=loans['LoanId'].values)
        lgds = pd.DataFrame(lgds, columns=ts, index=loans['LoanId'].values)
        cashflow_current = pd.DataFrame(cashflow_current, columns=ts, index=loans['LoanId'].values)
        cashflow_default = pd.DataFrame(cashflow_default, columns=ts, index=loans['LoanId'].values)
        return(hazs, lgds, cashflow_current, cashflow_default)

    def predict_profit(self, cashflows, verbose=False):
        profits = []
        i = 1
        for loanid, cashflow in cashflows.iterrows():
            if (i % 100 == 0) & verbose:
                print(i, "/", len(cashflows), "(", loanid, ")")
            t = np.arange(len(cashflow)) + 1
            dcf = lambda x: (np.exp(-x*t)*cashflow).sum() - 1
            if dcf(-0.99) < 0:
                z = -1.00
            if dcf(1.00) > 0:
                z = 1.00
            else:
                z = bisect(dcf, -0.99, 1.00, maxiter=3000)
            profit = (np.exp(z)**12 - 1)*100
            # print("profit: %.2f%%" % profit)
            profits.append(profit)
            i += 1
        profits = pd.Series(profits, index=cashflows.index)
        return profits

