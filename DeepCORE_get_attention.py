#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Pramod Bharadwaj Chandrashekar, Li Liu
@email: pchandrashe3@wisc.edu, liliu@asu.edu
"""

import pickle as pk
import pandas as pd
import deepclan_regelem_util as dru
import DeepCORE_data_util as ddu

mdl_fldr = 'model/samp/' # Point to the model folder
pred_info_file = mdl_fldr + 'test_pred_info.pkl' # Point to the predicted file: train, valid, or test
data_file = 'demo_5hm.csv' # Point to the input file

gene_info, labels = ddu.process_data(data_file, 'All', 1, 'percentile')
gene_info = gene_info.reset_index(drop=True)

with open(pred_info_file, 'r') as pred_file:
    g_info, attn, _, _ = pk.load(pred_file)

g_info = g_info.reset_index(drop=True)

genes_attn = pd.DataFrame(columns=['gene_id', 'chromosome_name', 'transcript_start', 'transcript_end',
                                   'upstream_start', 'upstream_end', 'downstream_start',
                                   'downstream_end', 'attention_bins'])
for ix, row in g_info.iterrows():
    gene_id = row['gene_id']
    gene = gene_info[gene_info['gene_id'] == gene_id]
    gene = gene.iloc[0]

    gene_attn, attn_bins, attn_pvals = dru.match_attn_loc(gene, attn[ix].copy())

    attn_bins_str = '|'.join(map(str, attn_bins))

    ginfo = gene_info.loc[gene_info.gene_id == gene_id, :].squeeze()

    genes_attn = genes_attn.append({'gene_id': gene_id, 'chromosome_name': row['chromosome_name'],
                                    'transcript_start': row['transcript_start'],
                                    'transcript_end': row['transcript_end'],
                                    'upstream_start': ginfo['upstream_start'],
                                    'upstream_end': ginfo['upstream_end'],
                                    'downstream_start': ginfo['downstream_start'],
                                    'downstream_end': ginfo['downstream_end'],
                                    'attention_bins': attn_bins_str}, ignore_index=True)

gene_attn.to_csv('test_prioritized_attention_bins.csv', index=False)