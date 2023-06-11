#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Pramod Bharadwaj Chandrashekar
@email: pchandrashe3@wisc.edu
"""

import math
import numpy as np
import pandas as pd
import DeepCORE_attention_util as dau

def match_enhancer_location(gene_info, enhancer_info):
    """
    Function to extract all the locations flanking TSS of a gene contianing an enhancer.
    Input: gene_info - A series that contains start, chromosome start and end loc, and strand info.
           enhancer_info - A dataframe containng all the known location of enhancer.s
    """

    gene_info_start = gene_info['upstream_start']
    gene_info_end = gene_info['downstream_end']
    if gene_info['strand'] == '-':
        gene_info_start = gene_info['downstream_start']
        gene_info_end = gene_info['upstream_end']

    gene_encr = enhancer_info[(enhancer_info['chrom'] == gene_info['chromosome_name']) &
                              (((enhancer_info['start'] >= gene_info_start) &
                                (enhancer_info['end'] <= gene_info_end)) |
                               ((enhancer_info['start'] <= gene_info_start) &
                                (enhancer_info['end'] <= gene_info_end) &
                                (enhancer_info['end'] >= gene_info_start)) |
                               ((enhancer_info['start'] >= gene_info_start) &
                                (enhancer_info['end'] <= gene_info_end) &
                                (enhancer_info['start'] <= gene_info_end)) |
                               ((enhancer_info['start'] <= gene_info_start) &
                                (enhancer_info['end'] >= gene_info_end)))]

    encr_info = np.reshape(np.zeros([10000]), [1, -1])
    encr_bins = []
    if not gene_encr.empty:
        for _, row in gene_encr.iterrows():
            if row['start'] < gene_info_start:
                st = 0
            else:
                st = row['start'] - gene_info_start # Start of the prom

            if row['end'] > gene_info_end:
                ed = 9999
            else:
                ed = st + (row['end'] - row['start']) # Total number of positions
            encr_info[0, st:ed] += 1

        bins = []
        for i in range(0, 10000, 50):
            bins.append(np.sum(encr_info[0, i:(i+50)]))

        encr_bins = np.where(np.asanyarray(bins) > 0)[0]

        print(gene_info['gene_id'], encr_bins)
    return gene_encr, encr_bins


def match_promoter_location(gene_info, promoter_info):
    """
    Function to extract all the locations flanking TSS of a gene contianing an promoter.
    Input: gene_info - A series that contains start, chromosome start and end loc, and strand info.
           promoter_info - A dataframe containng all the known location of promoter.s
    """

    gene_info_start = gene_info['upstream_start']
    gene_info_end = gene_info['downstream_end']
    if gene_info['strand'] == '-':
        gene_info_start = gene_info['downstream_start']
        gene_info_end = gene_info['upstream_end']

    gene_prom = promoter_info[(promoter_info['chrom'] == gene_info['chromosome_name']) &
                              (promoter_info['strand'] == gene_info['strand']) &
                              (((promoter_info['start'] >= gene_info_start) &
                                (promoter_info['end'] <= gene_info_end)) |
                               ((promoter_info['start'] <= gene_info_start) &
                                (promoter_info['end'] <= gene_info_end) &
                                (promoter_info['end'] >= gene_info_start)) |
                               ((promoter_info['start'] >= gene_info_start) &
                                (promoter_info['end'] <= gene_info_end) &
                                (promoter_info['start'] <= gene_info_end)) |
                               ((promoter_info['start'] <= gene_info_start) &
                                (promoter_info['end'] >= gene_info_end)))]

    prom_info = np.reshape(np.zeros([10000]), [1, -1])
    prom_bins = []
    if not gene_prom.empty:
        for _, row in gene_prom.iterrows():
            if row['start'] < gene_info_start:
                st = 0
            else:
                st = row['start'] - gene_info_start # Start of the prom

            if row['end'] > gene_info_end:
                ed = 9999
            else:
                ed = st + (row['end'] - row['start']) # Total number of positions
            prom_info[0, st:ed] += 1

        bins = []
        for i in range(0, 10000, 50):
            bins.append(np.sum(prom_info[0, i:(i+50)]))

        prom_bins = np.where(np.asanyarray(bins) > 0)[0]
        print(gene_info['gene_id'], prom_bins)

    return gene_prom, prom_bins


def match_promoter_loc(gene_info, promoter_info):
    """
    Function to extract all the locations flanking TSS of a gene contianing an promoters.
    Input: gene_info - A series that contains start, chromosome start and end loc, and strand info.
           promoter_info - A dataframe containng all the known location of promoters.
    """
    if gene_info['strand'] == '+':
        gene_prom = promoter_info[(promoter_info['chrom'] == gene_info['chromosome_name']) &
                                  (promoter_info['start'] >= (gene_info['upstream_end']-1000)) &
                                  (promoter_info['end'] <= (gene_info['upstream_end']+1000)) &
                                  (promoter_info['strand'] == gene_info['strand'])]
    else:
        gene_prom = promoter_info[(promoter_info['chrom'] == gene_info['chromosome_name']) &
                                  (promoter_info['start'] >= (gene_info['upstream_start']-1000)) &
                                  (promoter_info['end'] <= (gene_info['upstream_start']+1000)) &
                                  (promoter_info['strand'] == gene_info['strand'])]

    prom_bins = []
    if not gene_prom.empty:
        for _, row in gene_prom.iterrows():
            if gene_info['strand'] == '+':
                st = row['start'] - gene_info['upstream_start'] # Start of the prom
                ed = st + (row['end'] - row['start']) # Total number of positions
            else:
                st = abs(row['start'] - gene_info['downstream_start']) # Start of the prom
                ed = st + (row['end'] - row['start']) # Total number of positions

        print(st, ed)
        prom_bins.append(int(math.ceil(st/50)))
        prom_bins.append(int(math.ceil(ed/50)))

    return gene_prom


def match_attentionn_regelem_bins(attn_list, elem_list, reg_elem):
    """
    Function to check how many attended bins contain a given regulatory element.
    Input: attn_list - A list of bins which has high attention pvalues.
           elem_list - A list of bins that contains a regulatory element(Promoter/Enahncer).
           reg_elem - A string which tells what regulatory element we are using.
    Output: a_re - bins with both attn and reg elem.
            a_nre - bins with attn but no reg elem.
            na_re - bins with no attn but has reg elem.
            na_nre - bins with no attn and no reg elem.
    """
    a_re = list(set(attn_list).intersection(elem_list)) #
    a_nre = list(set(attn_list) - set(elem_list))
    na_re = list(set(elem_list) - set(attn_list))

    all_bin_list = range(0, 200)
    if reg_elem == 'promoter':
        all_bin_list = range(80, 120)
    elif reg_elem == 'enhancer':
        all_bin_list = list(range(0, 80)) + list(range(120, 200))

    na_nre = list(set(all_bin_list) - (set(attn_list).union(elem_list)))
    return a_re, a_nre, na_re, na_nre


def match_attn_loc(gene, attn_info):
    gene_attn = np.zeros([10000])

    # Highly significant attention bins
    attn_info = np.reshape(attn_info, [1, -1])
    attn_pvals = np.asarray(dau.get_cdf_pval(attn_info))
    attn_pvals[np.isnan(attn_pvals)] = 1
    attn_pvals = np.min(attn_pvals, axis=2)
    imp_bins = dau.get_important_bins(np.reshape(attn_pvals, [-1]))

    # Attention values for each position of the gene
    attn_info = np.reshape(attn_info, [-1])
    for idx, val in enumerate(attn_info):
        if idx == 0:
            bin_st, bin_end = 0, 98
        else:
            bin_st = bin_st + 50
            bin_end = bin_end + 50
        gene_attn[bin_st:min(bin_end, 10000)] += attn_info[idx]

    return gene_attn, list(imp_bins), attn_pvals


def combine_multiple_tissue_results(cell_types, results_folder, file_names):
    """ """
    cntr = 0
    all_dat = []
    for ctype in cell_types:
        dat = pd.read_csv(results_folder + ctype + '_' + file_names)
        dat = dat.drop('Unnamed: 0', 1)
        dat['tissue'] = ctype

        if cntr == 0:
            all_dat = dat
        else:
            all_dat = all_dat.append(dat, ignore_index=True)
        cntr = cntr + 1
    return all_dat
