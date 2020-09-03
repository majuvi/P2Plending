import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_payments(payments, followup=True, alpha=0.2):
    payments = payments.copy()
    mean_payment = payments.groupby('T')['Y'].agg('mean')

    followups = payments.groupby('LoanId').agg('max').reset_index()
    followups = followups.sort_values('T', ascending=False)
    followups['Sample'] = np.arange(1, len(followups) + 1)
    ids_to_idx = dict(zip(followups['LoanId'], followups['Sample']))
    payments['Sample'] = payments['LoanId'].map(ids_to_idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [10, 1]}, sharex=True)

    ax1.scatter(payments['T'], payments['Sample'], s=payments['Y'] * 50, c='tab:blue')
    if followup:
        ax1.hlines(followups['Sample'], 0, followups['T'], color='grey', zorder=-10, alpha=alpha)
        ax1.scatter(np.zeros_like(followups['T']), followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
        ax1.scatter(followups['T'], followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
    ax1.set_title('Loan Payments')
    ax1.set_ylabel('Loan')
    ax1.set_ylim(0.5, len(followups) + 0.5)

    ax2.scatter(mean_payment.index, np.zeros_like(mean_payment), s=mean_payment * 50, c='tab:blue')
    ax2.set_xlabel('Month')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlim(0, 120)
    ax2.get_yaxis().set_visible(False)
    ax2.set_title('Average Payments')

def plot_models(payments, intervals, residuals, alpha=0.2):
    from matplotlib.lines import Line2D
    payments = payments.copy()
    intervals = intervals.copy()
    residuals = residuals.copy()

    mean_payment = payments.groupby('T')['Y'].agg('mean')
    mean_default = intervals.groupby('T')['Y'].agg('mean')
    mean_recovery = residuals.groupby('T')['Y'].agg('mean')

    followups = payments.groupby('LoanId').agg('max').reset_index()
    followups = followups.sort_values('T', ascending=False)
    followups['Sample'] = np.arange(1, len(followups) + 1)
    ids_to_idx = dict(zip(followups['LoanId'], followups['Sample']))
    payments['Sample'] = payments['LoanId'].map(ids_to_idx)
    intervals['Sample'] = intervals['LoanId'].map(ids_to_idx)
    residuals['Sample'] = residuals['LoanId'].map(ids_to_idx)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, gridspec_kw={'height_ratios': [10, 1]}, figsize=(14,4), sharex=True)

    ax1.scatter(payments['T'], payments['Sample'], s=payments['Y'] * 50, c=payments['Default'].map({False: 'tab:blue', True: 'tab:orange'}))
    followups = payments.groupby('Sample')['T'].agg('max').reset_index()
    ax1.hlines(followups['Sample'], 0, followups['T'], color='grey', zorder=-10, alpha=alpha)
    ax1.scatter(np.zeros_like(followups['T']), followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
    ax1.scatter(followups['T'], followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scheduled', markerfacecolor='tab:blue', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Recovery', markerfacecolor='tab:orange', markersize=5)]
    ax1.legend(handles=legend_elements, title='Payment')
    ax1.set_title('Payments')

    ax2.scatter(intervals['T'], intervals['Sample'], s=5, c=intervals['Y'].map({False: 'tab:blue', True: 'tab:red'}))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='False', markerfacecolor='tab:blue', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='True', markerfacecolor='tab:red', markersize=5)]
    ax2.legend(handles=legend_elements, title='Default')
    ax2.set_title('Defaults')

    ax3.scatter(residuals['T'], residuals['Sample'], s=residuals['Y'] * 50, c='tab:orange')
    followups = residuals.groupby('Sample')['T'].agg('max').reset_index()
    ax3.hlines(followups['Sample'], 0, followups['T'], color='grey', zorder=-10, alpha=alpha)
    ax3.scatter(np.zeros_like(followups['T']), followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
    ax3.scatter(followups['T'], followups['Sample'], marker='|', color='grey', label='', zorder=-10, alpha=alpha)
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Recovery', markerfacecolor='tab:orange', markersize=5)]
    ax3.legend(handles=legend_elements, title='Payment')
    ax3.set_title('Recoveries')

    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(0, len(ids_to_idx) + 1)
        ax.set_xlim(0, 80)
        ax.set_ylabel('Loan')

    ax4.scatter(mean_payment.index, np.zeros_like(mean_payment), s=mean_payment * 10, c='tab:blue')
    ax4.set_title('Average Payments')

    ax5.scatter(mean_default.index, np.zeros_like(mean_default), s=mean_default * 10, c='tab:red')
    ax5.set_title('Average Defaults')

    ax6.scatter(mean_recovery.index, np.zeros_like(mean_recovery), s=mean_recovery * 10, c='tab:orange')
    ax6.set_title('Average Recoveries')

    for ax in [ax4, ax5, ax6]:
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, 80)
        ax.get_yaxis().set_visible(False)
        ax.set_xlabel('Followup')
