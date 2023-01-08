import pandas
import numpy as np

import os
import pickle

from scipy import stats
import scipy.stats as st
import pandas as pd
import math
from tqdm import tqdm


df_summary = pd.read_csv('csv_results_summary.csv', low_memory=False)
df_summary = df_summary.drop(columns=['severity', 'perturb_type'])

df_here = df_summary.copy()

table_stats = open('table_stats.txt', 'w')
models = ['UNet++', 'DeepLabV3+', 'CE-Net']

for model in models:
    table_stats.write(model + ':\n')
    df_here = df_summary[df_summary['Segmentation Model'] == model]
    table_stats.write('perturb, od_iou, oc_iou\n')
    u = df_here['Perturbation Type'].unique()
    # print(u)
    u[0] = 'orig'
    for perturb in u:
        if perturb == 'orig':
            df_here2 = df_here[df_here['Perturbation Type'].isnull()]
        elif perturb in ['1', '10', '11']:
            df_here2 = df_here[df_here['Perturbation Type'] == perturb]
        else:
            df_here2 = df_here[df_here['Perturbation Type'] == perturb]
            df_here2 = df_here2[df_here2['Severity'] == 3]
        # bg_iou = df_here2[df_here2['class']==0].jacc.mean()
        od_iou = df_here2[df_here2['class']==1].jacc.mean()
        oc_iou = df_here2[df_here2['class']==2].jacc.mean()
        od_iou = round(od_iou, 3)
        oc_iou = round(oc_iou, 3)
        od_std = round(df_here2[df_here2['class']==1].jacc.std(), 3)
        oc_std = round(df_here2[df_here2['class']==2].jacc.std(), 3)
        if perturb == '1': perturb='de-illumination (d1)'
        elif perturb == '10': perturb='de-spot (d1)'
        elif perturb == '11': perturb='de-spot+de-illumination (d1)'
        table_stats.write(f'{perturb}, {od_iou} +/- {od_std}, {oc_iou} +/- {oc_std}\n')
    allp = df_here[(df_here['Perturbation Type'].notnull()) & ((df_here['Severity'] == 0) | (df_here['Severity'] == 3))]
    od_iou = allp[allp['class']==1].jacc.mean()
    oc_iou = allp[allp['class']==2].jacc.mean()
    od_iou = round(od_iou, 3)
    oc_iou = round(oc_iou, 3)
    od_std = round(allp[allp['class']==1].jacc.std(), 3)
    oc_std = round(allp[allp['class']==2].jacc.std(), 3)
    table_stats.write(f'all_perturbs, {od_iou} +/- {od_std}, {oc_iou} +/- {oc_std}\n')
    table_stats.write('\n')

table_stats.close()