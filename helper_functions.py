#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string

from matplotlib import gridspec
from matplotlib.colors import ListedColormap
import pylab as plt
from pylab import *
import scipy
import sklearn.metrics
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def norm_data(data):
    """
    Norm data by mean and std.

    Input: arraylike structure, if 2D rows represent different
    channels
    Return: arraylike structure
    """
    if len(np.shape(data)) == 2:
        data = data.copy()
        data -= np.mean(data, 1)[:, np.newaxis]
        data /= np.std(data, 1)[:, np.newaxis]

    if len(np.shape(data)) == 1:
        data = data.copy()
        data -= np.mean(data)
        data /= np.std(data)
    return data


def wd(mat):
    """Return matrix without diagonal."""
    return mat - diag(diag(mat))


def C_from_G(G, diag_Y='ones'):
    """Calculate covariance matrix from connectivity assuming linear model, return array."""
    G = matrix(G)
    n_nodes = G.shape[0]
    Id = matrix(eye(n_nodes))
    if diag_Y == 'ones':
        Y = Id
    else:
        Y = matrix(diag(diag_Y))
    return array((Id-G).I * Y * ((Id-G).I).H)


def B_from_C(C):
    """Calculate initial condition from covariance matrix, return matrix."""
    eig_vals, eig_vecs = linalg.eig(inv(C))
    D = matrix(diag(sqrt(eig_vals)))
    eig_vecs = matrix(eig_vecs)
    B = eig_vecs * D * eig_vecs.H
    return B


def B_from_CI(CI):
    """Calculate initial condition from inverse covariance matrix, return matrix."""
    eig_vals, eig_vecs = linalg.eig(CI)
    D = matrix(diag(sqrt(eig_vals)))
    eig_vecs = matrix(eig_vecs)
    B = eig_vecs * D * eig_vecs.H
    return B


def B_from_G(G, chol=False):
    """Calculate initial condition from connectivity matrix, return matrix."""
    C = matrix(C_from_G(G))
    if chol:
        return cholesky(C.I).H
    eig_vals, eig_vecs = linalg.eig(C.I)
    D = matrix(diag(sqrt(eig_vals)))
    eig_vecs = matrix(eig_vecs)
    B = eig_vecs * D * eig_vecs.H
    return B


def get_performance(G, G_est):
    """Calculate area under the roc curve for given network G and estimation G_est (leaves out diagonal entries).

    Input:
    G -- arraylike, true connectivity matrix
    G_est -- arraylike, estimated connectivity matrix
    Returns:
    auc -- float, Area under the ROC curve
    prs -- float, Precision-Recall Score
    """
    G = array(abs(wd(G)) > 0)
    G_est = array(abs(wd(G_est)))
    auc = sklearn.metrics.roc_auc_score(G.flatten(), G_est.flatten())
    prs = sklearn.metrics.average_precision_score(G.flatten(), G_est.flatten())
    return auc, prs


def calc_G_percentile(G, percentile):
    """Set all values but 'percentile'-biggest to zero, return thresholded G (arraylike) and threshold value (float)."""
    if len(np.shape(G)) == 3:
        G_treshold = np.copy(abs(G))  # initialize matrix
        min_value = np.zeros(np.shape(G)[0])  # initialize min_value array
        for timebin in range(np.shape(G)[0]):
            min_value[timebin] = scipy.stats.scoreatpercentile(abs(G[timebin].flatten(
                )), 100.-percentile)  # calculates minimal value which should not be set to zero
            # set all values smaller than min_value to zero
            G_treshold[timebin][abs(G[timebin]) < min_value[timebin]] = 0
    if len(np.shape(G)) == 2:  # same for one timebin
        G_treshold = np.copy(G)
        min_value = scipy.stats.scoreatpercentile(
            abs(G.flatten()), 100.-percentile)
        G_treshold[abs(G) < min_value] = 0
    return G_treshold, min_value


def plot_networks(G, G_est, C, percentile):
    """Plot original G, G_est and C with false connections marked and ROC and PR curves."""
    G = abs(array(G))
    G_est = abs(array(G_est))
    f, ax = plt.subplots(2, 2, figsize=(14, 12.0))
    f.subplots_adjust(left=0.03, right=.9, bottom=0.07, top=.9)
    G_est_TP, res = calc_TP_FP_FN(G, G_est, percentile)
    auc, prs = get_performance(G, G_est)
    pcc = corrcoef(G.flatten(), G_est.flatten())[0, 1]
    print('AUC: ' + str(auc))
    print('PRS: ' + str(prs))
    print('PCC: ' + str(pcc))
    print('True positives: ' + str(res[0]))
    print('True negatives: ' + str(res[3]))
    print('False positives: ' + str(res[1]))
    print('Flase negatives: ' + str(res[2]))
    C_TP, res_C = calc_TP_FP_FN(G, C - np.diag(np.diag(C)), percentile)
    colors = ['#67a9cf', 'white', 'black', '#ef8a62']
    cmap = ListedColormap(colors, 'indexed')
    ax[0, 0].spines['top'].set_visible(True)
    ax[0, 0].spines['right'].set_visible(True)
    ax[0, 1].spines['top'].set_visible(True)
    ax[0, 1].spines['right'].set_visible(True)
    ax[1, 0].spines['top'].set_visible(True)
    ax[1, 0].spines['right'].set_visible(True)
    ax[1, 0].set_xticklabels([])
    ax[0, 1].set_yticklabels([])
    ax[0, 0].matshow(G, interpolation='None', cmap=cmap, vmin=-1, vmax=2)
    ax[0, 0].set_title(r'True Connectivity Matrix', fontsize=25, y=1.15)
    ax[0, 0].set_ylabel('Nodes', fontsize=25)
    ax[0, 0].set_xlabel('Nodes', fontsize=25)
    ax[0, 1].set_xlabel('Nodes', fontsize=25)
    ax[1, 0].set_xlabel('Nodes', fontsize=25)
    ax[1, 0].set_ylabel('Nodes', fontsize=25)
    ax[0, 0].text(-0.1, 1.05, string.ascii_uppercase[0],
                  transform=ax[0, 0].transAxes, size=30, weight='bold')
    ax[0, 1].text(-0.1, 1.05, string.ascii_uppercase[1],
                  transform=ax[0, 1].transAxes, size=30, weight='bold')
    ax[1, 0].text(-0.1, 1.05, string.ascii_uppercase[2],
                  transform=ax[1, 0].transAxes, size=30, weight='bold')
    ax[1, 1].text(-0.1, 1.05, string.ascii_uppercase[3],
                  transform=ax[1, 1].transAxes, size=30, weight='bold')
    im = ax[0, 1].matshow(G_est_TP, interpolation='None',
                          cmap=cmap, vmin=-1, vmax=2)
    ax[0, 1].set_title(r'Estimated Connectivity Matrix', fontsize=25, y=1.15)
    ax[1, 0].matshow(C_TP, interpolation='None', cmap=cmap, vmin=-1, vmax=2)  # Ignore PEP8Bear
    ax[1, 0].set_title(r"Covariance Matrix", fontsize=25, y=1.)
    cbar_ax = f.add_axes([0.95, 0.28, 0.01, 0.45])
    cbar = f.colorbar(im, cax=cbar_ax, ax=[
                      ax, [0, 0], ax[0, 1], ax[1, 0]], cmap=cmap, ticks=[-.6, 0.1, .9, 1.6])
    cbar.ax.set_yticklabels(['FN', 'TN', 'TP', 'FP'])
    fpr, tpr, thresh = roc_curve(G.flatten(), G_est.flatten())
    fpr_c, tpr_c, thresh_c = roc_curve(G.flatten(), abs(C).flatten())
    pres, recall, thresh2 = precision_recall_curve(
        G.flatten(), G_est.flatten())
    pres_c, recall_c, thresh2_c = precision_recall_curve(
        G.flatten(), abs(C).flatten())
    ax[1, 1].plot(fpr, tpr, label=r'ROC Curve $G_{est}$')
    ax[1, 1].plot(recall, pres, label=r'PR Curve $G_{est}$')
    ax[1, 1].plot(fpr_c, tpr_c, label=r'ROC Curve $C$')
    ax[1, 1].plot(recall_c, pres_c, label=r'PR Curve $C$')
    ax[1, 1].set_title(r'Performance Measures', fontsize=25, y=1.0)
    ax[1, 1].set_xlabel('False Positive Rate / Recall', fontsize=25)
    ax[1, 1].set_ylabel('True Positive Rate / Precision', fontsize=25)
    ax[1, 1].set_xlim(-.02, 1.)
    ax[1, 1].set_ylim(0.5, 1.05)
    ax[1, 1].legend(loc=8)
    ax[1, 1].yaxis.set_ticks_position('left')
    ax[1, 1].xaxis.set_ticks_position('bottom')


def calc_TP_FP_FN(G, G_est, percentile):
    """Find TP, TN, FP, FN in G_est.

    True positives are encoded TP = 1, true negatives TN = 0, false positives FP = 2 and false negatives FN = -1.
    Input:
    G -- arraylike, true connectivity matrix.
    G_est -- arraylike, estimated connectivity matrix.
    percentile -- float, percentage of strongest connections which are considered.
    Return:
    G_est_TP -- arraylike, estimated connectivity with TP,TN, FP, FN encoded as described above.
    [TP, FP, FN, TN] -- list, number of true posivtives, false positives, false negatives, true negatives.
    """
    Gest, minval1 = calc_G_percentile(G_est, percentile)
    N = np.shape(G)[0]
    Gest_TP = np.zeros((N, N))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for row in range(N):
        for col in range(N):
            if G[row, col] == 0 and abs(Gest[row, col]) >= 1e-12:
                Gest_TP[row, col] = 2
                FP += 1
            if abs(G[row, col]) >= 1e-12 and Gest[row, col] == 0:
                Gest_TP[row, col] = -1
                FN += 1
            if abs(G[row, col]) >= 1e-12 and abs(Gest[row, col]) >= 1e-12:
                Gest_TP[row, col] = 1
                TP += 1
            if abs(G[row, col]) <= 1e-12 and abs(Gest[row, col]) <= 1e-12:
                Gest_TP[row, col] = 0
                TN += 1

    return Gest_TP, [TP, FP, FN, TN]


def plot_results(xs, auc_means, auc_stds, prs_means, prs_stds, corrcoef_means, corrcoef_stds, variable):
    """Plot results of estimation.

    Input:
    xs -- arraylike, variable on x-axis.
    auc_means -- arraylike, AUC values, (means if there are more than 1 realizations).
    auc_stds -- arraylike, standard deviation of AUC values if there are more than 1 realizations, if only 1 realization is done, use zeros(len(xs)).
    prs_means -- arraylike, PRS values, (means if there are more than 1 realizations).
    prs_stds -- arraylike, standard deviation of PRS values if there are more than 1 realizations, if only 1 realization is done, use zeros(len(xs)).
    corrcoef_means -- arraylike, correlation coeffieents, (means if there are more than 1 realizations).
    corrcoef_stds -- arraylike, standard deviation of correlation coefficients, if there are more than 1 realizations, if only 1 realization is done, use zeros(len(xs)).
    variable -- string, label for x-axis.
    """
    if auc_stds == None:
        auc_stds = np.zeros(len(auc_means))
        prs_stds = np.zeros(len(auc_means))
        corrcoef_stds = np.zeros(len(auc_means))
    ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.plot(xs, auc_means, label='AUC', color='#1b9e77')
    plt.plot(xs, prs_means, label='Precision Recall score', color='#d95f02')
    plt.plot(xs, corrcoef_means, label='Correlation Coefficient', color='#2C12A6')
    plt.fill_between(xs, auc_means - auc_stds, auc_means +
                     auc_stds, facecolor='#1b9e77', alpha=0.3)
    plt.fill_between(xs, prs_means - prs_stds, prs_means +
                     prs_stds, facecolor='#d95f02', alpha=0.3)
    plt.fill_between(xs, corrcoef_means - corrcoef_stds,
                     corrcoef_means + corrcoef_stds, facecolor='#2C12A6', alpha=0.3)
    plt.legend(loc=4)
    plt.xlabel(variable)
    plt.ylabel('Score')
    plt.xlim(xs[0], xs[-1])
    plt.ylim(0.5, 1.02)
    plt.show()
