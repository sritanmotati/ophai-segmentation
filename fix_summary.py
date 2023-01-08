import pandas
import numpy as np

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats as st
import pandas as pd
from tqdm import tqdm

list_of_Results = [
                    ('UNet++', 'orig', '../results/unetpp/orig/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'), 
                    ('UNet++', 'd1', '../results/unetpp/d1/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'), 
                    ('UNet++', 'd2', '../results/unetpp/d2/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'), 

                    ('DeepLabV3+', 'orig', '../results/deeplabv3plus/orig/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'), 
                    ('DeepLabV3+', 'd1', '../results/deeplabv3plus/d1/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'), 
                    ('DeepLabV3+', 'd2', '../results/deeplabv3plus/d2/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),               
                    
                    ('CE-Net', 'orig', '../results/cenet/orig/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_orig.csv'), 
                    ('CE-Net', 'd1', '../results/cenet/d1/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d1.csv'), 
                    ('CE-Net', 'd2', '../results/cenet/d2/individual_results.csv', 
                    '../ObservedDomain_Sritan/data_AroundDisc_256_seg_joint_test_d2.csv'),

                    ]



df0 = pd.read_csv(list_of_Results[1][2] )

columns = df0.columns.to_list()
columns.append('Segmentation Model')
columns.append('Perturbation Type')
columns.append('Severity')
columns.append('InDomain')
columns.append('OutDomain')

df_summary = pd.read_csv('csv_results_summary.csv')

counter_index=-1
for csvtuples in list_of_Results: 
#csvtuples = list_of_Results[4]

    df_result = pd.read_csv(csvtuples[2] )
    df_source_test = pd.read_csv(csvtuples[3] )
    try:
        df_source_test= df_source_test.drop(columns='index')
    except:
        s=1
    
    
        
    df_result[ ['path','imageID'] ] = df_result['FullFileName'].str.split('images', expand=True)
    df_result.imageID = df_result.imageID.str[1:]
    
    df_result_merged = df_result.merge(df_source_test, how='outer', on='imageID')
    
    
    if 'RIGA' in csvtuples[1]:
        try:
            df_result_merged = df_result_merged.loc[df_result_merged.Unet_TrainedOver!='RIGA']
        except:
            s=1
        
    columns_result = df_result_merged.columns
            
    for IND in tqdm(df_result_merged.index):
        counter_index = counter_index+1
        if 'd1' in csvtuples[1]:
            try:        
                df_summary.loc[counter_index, 'Perturbation Type'] = df_result_merged.perturb_type[IND]
            except:
                df_summary.loc[counter_index, 'Perturbation Type'] = 'N/A'

df_summary.to_csv('csv_results_summary.csv', index=False) 
