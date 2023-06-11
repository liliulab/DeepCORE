#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Pramod Bharadwaj Chandrashekar, Li Liu
@email: pchandrashe3@wisc.edu, liliu@asu.edu
"""

import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as stats


def get_cdf_pval(data):
    """ Function for guassian mixture of dat and computing pvalues """
    cdf_pvals = []
    for i in range(0, len(data)):
        mn_samp = np.mean(data[i, :])
        sd_samp = np.std(data[i, :])

        kcl = KMeans(n_clusters=2, random_state=0).fit(np.reshape(data[i], [-1, 1]))
        cluster_1_id = np.where(kcl.labels_ == 0)[0]
        c1_mn, c1_sd = np.mean(data[i, cluster_1_id]), np.std(data[i, cluster_1_id])
        cdf_pval_1 = np.reshape(1.0 - stats.norm.cdf(data[i, :], c1_mn, c1_sd), [-1, 1])

        cluster_2_id = np.where(kcl.labels_ == 1)[0]
        c2_mn, c2_sd = np.mean(data[i, cluster_2_id]), np.std(data[i, cluster_2_id])
        cdf_pval_2 = np.reshape(1.0 - stats.norm.cdf(data[i, :], c2_mn, c2_sd), [-1, 1])

        cdf_pval_3 = np.reshape(1.0 - stats.norm.cdf(data[i, :], mn_samp, sd_samp), [-1, 1])

        cdf_pvals.append(np.concatenate((cdf_pval_1, cdf_pval_2, cdf_pval_3), axis=1))
    return cdf_pvals

def get_important_bins(pval_data):
    """ Fetch important bins based on pvalues"""
    imp_bins = []
    # Bonferroni Corrected pvals check
    if len(np.where(pval_data*200 < 0.05)[0]) > 0:
        imp_bins = np.where(pval_data*200 < 0.05)[0]
    # Normal pval check
    elif len(np.where(pval_data < 0.05)[0]):
        imp_bins = np.where(pval_data < 0.05)[0]
    # Top 10 bins
    else:
        sorted_bins = np.argsort(pval_data)
        imp_bins = sorted_bins[0:20]
        #imp_bins = np.argpartition(pval_data, 10)
    return imp_bins
