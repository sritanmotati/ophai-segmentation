# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 23:33:56 2023

@author: Moha-Cate
"""

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

sns.set_style("whitegrid")


# print('################# Notice:')
# print('################# Notice:')
# print('################# Notice:')
# print('seaborn version: ', sns.__version__)
# print('if the colors in legends are mismatched, make sure to have seaborn verion 0.11.0')
# print('################# Notice end')


        
df_summary = pd.read_csv('csv_results_summary.csv', low_memory=False)        

df_summary = df_summary.drop(columns=['severity', 'perturb_type'])

try:
    os.mkdir('./figures')
except:
    s=1 
        
#%%        
        
df_here = df_summary.copy()

df_here=df_here.loc[df_here['class']==1]

# df_here_indomain = df_here.loc[df_here['InDomain']==1]
# df_here_outdomain = df_here.loc[df_here['InDomain']==0]
# df_here_indomain_orig = df_here.loc[df_here['Severity']==0]
# df_here_outdomain_orig = df_here.loc[df_here['Severity']==0]

df_here_orig = df_here.loc[df_here['Severity']==0].loc[pd.isna(df_here['Perturbation Type'])]
df_here_d1 = df_here.loc[df_here['Severity']==0].loc[pd.notna(df_here['Perturbation Type'])]
df_here_d2 = df_here.loc[df_here['Severity'] > 0]

######################
x="Segmentation Model"
y="jacc"
# hue="Method"
# hue_order = ['MWen','no-change'] #hue_order=hue_order,  hue=hue,
#fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
plt.figure(1, figsize=(4, 5), dpi=300)
ax_orig = sns.boxplot(x=x, y=y, data=df_here_orig, 
                 showfliers = False,
                 showmeans=True,
                 meanprops = {"marker": "x", "markerfacecolor": "white", 
                              "markeredgecolor": "white"},
                palette=sns.color_palette("husl")) #  flierprops = dict(markerfacecolor = '0.50', markersize = 1)
ax_orig.set(ylim=(-.05, 1))
# plt.legend(bbox_to_anchor=(0.16, 1), loc="lower left", borderaxespad=0. , ncol=3)
#ax_in_orig.set(xticklabels=[])
ax_orig.set(ylabel= 'IOU')
ax_orig.text(0.45, 1.1, "Without Perturbation", color='black', fontsize=12)
# ax_orig.set(title="Without Perturbation")
plt.savefig('./figures/IOU_orig.png', dpi=300, bbox_inches = 'tight')


######################
x="Segmentation Model"
y="jacc"
col_category = 'Perturbation Type'

df_here_d1.loc[df_here_d1['Perturbation Type']=='11', 'Perturbation Type'] = 'De-illumination + De-spot'
df_here_d1.loc[df_here_d1['Perturbation Type']=='10', 'Perturbation Type'] = 'De-spot'
df_here_d1.loc[df_here_d1['Perturbation Type']=='1', 'Perturbation Type'] = 'De-illumination'

plt.figure(3, figsize=(4, 5), dpi=300)
ax3 = sns.catplot(x=x, y=y,
                col = col_category, showmeans=True,
                data=df_here_d1, kind="box",
                showfliers = False, height=4, aspect=1 ,           
                 meanprops = {"marker": "x", "markerfacecolor": "white", 
                              "markeredgecolor": "white"},
                palette=sns.color_palette("husl"))
#ax3.set(ylim=(-.5, 10))
ax3.set_titles("{col_name}")
ax3.set(ylabel= 'IOU')
# ax3.fig.suptitle("Perturbation Types", fontweight='bold', fontsize=11)
ax3.fig.text(0.45, 1, "Perturbation Types", color='black', fontsize=12)
# plt.tight_layout()
plt.savefig('./figures/IOU_d1.png', dpi=300, bbox_inches = 'tight')


######################
x="Segmentation Model"
y="jacc"
hue="Severity"
hue_order = [1,2,3,4,5]
col_category = 'Perturbation Type'

for label in df_here_d2['Perturbation Type'].unique().tolist():
    df_here_d2.loc[df_here_d2['Perturbation Type']==label, 'Perturbation Type'] = label.replace('_', ' ').title()

plt.figure(3, figsize=(4, 5), dpi=300)
ax3 = sns.catplot(x=x, y=y,
                hue=hue, col = col_category, showmeans=True, col_wrap=4,
                data=df_here_d2, kind="box", hue_order=hue_order, 
                showfliers = False, height=4, aspect=1 ,           
                 meanprops = {"marker": "x", "markerfacecolor": "white", 
                              "markeredgecolor": "white"},
                palette=sns.color_palette("husl")
                )
#ax3.set(ylim=(-.5, 10))
ax3.set_titles("{col_name}")
ax3.set(ylabel= 'IOU')
# ax3.fig.suptitle("Perturbation Types", fontweight='bold', fontsize=11)
ax3.fig.text(0.45, 1, "Perturbation Types", color='black', fontsize=12)
# plt.tight_layout()
plt.savefig('./figures/IOU_d2.png', dpi=300, bbox_inches = 'tight')

CREATE_INDIV = False

if CREATE_INDIV:
    for label in df_here_d2['Perturbation Type'].unique().tolist():
        plt.figure(3, figsize=(4, 5), dpi=300)
        ax3 = sns.catplot(x=x, y=y,
                        hue=hue, col = col_category, showmeans=True,
                        data=df_here_d2.loc[df_here_d2['Perturbation Type']==label], kind="box", hue_order=hue_order, 
                        showfliers = False, height=4, aspect=1 ,           
                        meanprops = {"marker": "x", "markerfacecolor": "white", 
                                    "markeredgecolor": "white"},
                        palette=sns.color_palette("husl")
                        )
        #ax3.set(ylim=(-.5, 10))
        ax3.set_titles("{col_name}")
        ax3.set(ylabel= 'IOU')
        # ax3.fig.suptitle("Perturbation Types", fontweight='bold', fontsize=11)
        ax3.fig.text(0.45, 1, "Perturbation Types", color='black', fontsize=12)
        # plt.tight_layout()
        flabel = label.replace(' ', '_')
        plt.savefig(f'./figures/d2_figs/IOU_{flabel}_d2.png', dpi=300, bbox_inches = 'tight')


    
################


plt.close('all')
#%% find the images to show

img_size = 256
binary = False

import torch
from models.attnet import AttNet
from models.cenet import CENet
from models.deeplabv3plus import DeepLabV3Plus
from models.doubleunet import DoubleUnet
from models.mnet import MNet
from models.mobilenet_unet import MobileNetUnet
from models.resnet_unet import ResNetUnet
from models.resunet import ResUnet
from models.unet import Unet
from models.unetpp import UnetPlusPlus

from utils.data_utils import * 

img_size = (img_size, img_size)

torch_models = ['cenet']
polar_models = ['mnet']

list_models = ['UNet++', 'DeepLabV3+', 'CE-Net']
list_model_names = ['unetpp', 'deeplabv3plus', 'cenet']
list_saved_models = ['../models/unetpp/unetpp_model.h5', 
                     '../models/deeplabv3plus/deeplabv3plus_model.h5',
                     '../models/cenet/cenet_model.pth']
thresholds_jacc = [0.35, 0.35, 0]
prefer = ['contrast', 'Hue_minus', 'Hue_minus']

fig_jacc, axs_jacc = plt.subplots(len(list_models), 5, figsize=(7.5, 7.5), dpi=300)

repeat = ()

for counter_model in range(len(list_models)):
    model_name=list_model_names[counter_model]
    df_here_method = df_summary.loc[df_summary['Segmentation Model'] == list_models[counter_model]]
    flag_done_thisMethod_jacc = False
    
    model = {
        'attnet': AttNet,
        'cenet': CENet,
        'deeplabv3plus': DeepLabV3Plus,
        'doubleunet': DoubleUnet,
        'mnet': MNet,
        'mobilenet_unet': MobileNetUnet,
        'resnet_unet': ResNetUnet,
        'resunet': ResUnet,
        'unet': Unet,
        'unetpp': UnetPlusPlus
    }[model_name]((img_size[0],img_size[1],3), 2 if binary else 3) # only important for unet models, SOTA models have their own size/n_channels and this will be disregarded
    
    if model_name not in torch_models:
        model.load(list_saved_models[counter_model])
    else:
        model.load(list_saved_models[counter_model], cpu=True)

    maskIDs = df_here_method.maskID.unique() if counter_model != 2 else [repeat[0]]
    for maskID in maskIDs:
        if flag_done_thisMethod_jacc:
            continue
        else:
            rec = df_here_method.loc[df_here_method.maskID==maskID]
            # print(rec[['maskID', 'Perturbation Type', 'jacc', 'class']])
            rec_noPert = rec.loc[rec['Perturbation Type'].isnull()]
            # print(rec_noPert[['maskID', 'Perturbation Type', 'jacc', 'class']])
            jacc_origin = rec_noPert.jacc.to_numpy()
            # rec_Pert = rec.loc[(rec['Perturbation Type'].notnull())] #  & (rec['Severity'] == 5)
            rec_Pert = rec.loc[(rec['Perturbation Type'] == prefer[counter_model])] if counter_model != 2 else repeat[1]
            # if counter_model > 0: print(rec_Pert)
            # print(rec_Pert[['maskID', 'Perturbation Type', 'jacc', 'class']])
            # idx = rec_Pert.index if counter_model != 2 else rec.loc[(rec['Perturbation Type'] == repeat[1])].loc[(rec['Severity'] == repeat[2])].index
            for IND in rec_Pert.index:
                # print(jacc_origin.shape, rec_Pert.jacc[IND])
                if counter_model==2 or jacc_origin[IND%3]-rec_Pert.jacc[IND] > thresholds_jacc[counter_model]:
                    img_orig, mask_gt_cat = process_pair(rec_noPert.FullFileName.iloc[0], rec_noPert.MaskFileName.iloc[0], img_size, binary=binary, polar=(model_name in polar_models), channelsFirst=(model_name in torch_models))
                    img_perturbed, _ = process_pair(rec_Pert.FullFileName[IND], rec_noPert.MaskFileName.iloc[0], img_size, binary=binary, polar=(model_name in polar_models), channelsFirst=(model_name in torch_models))
                    tensor = np.expand_dims(img_orig, axis=0)
                    tensor_perturbed = np.expand_dims(img_perturbed, axis=0)
                    
                    pred_orig = model.predict(tensor)
                    pred_orig = np.squeeze(pred_orig)
                    
                    pred_perturebed = model.predict(tensor_perturbed)
                    pred_perturebed = np.squeeze(pred_perturebed)
                    
                    if model_name in torch_models:
                      pred_orig = np.moveaxis(pred_orig, 0, -1)
                      pred_perturebed = np.moveaxis(pred_perturebed, 0, -1)
                      img_orig = np.moveaxis(img_orig, 0, -1)
                      img_perturbed = np.moveaxis(img_perturbed, 0, -1)
                      mask_gt_cat = np.moveaxis(mask_gt_cat, 0, -1)
                               
                    mask_gt = np.argmax(mask_gt_cat, axis=-1)
        
                    im_pred = np.argmax(pred_orig, axis=-1) 
                    im_pred_perturebed = np.argmax(pred_perturebed, axis=-1)        
        
                    axs_jacc[counter_model, 0].imshow(img_orig)
                    axs_jacc[counter_model, 0].set_title('Orig. Image')
                    axs_jacc[counter_model, 0].set_xticks([])
                    axs_jacc[counter_model, 0].set_yticks([])
                    axs_jacc[counter_model, 1].imshow(mask_gt)
                    axs_jacc[counter_model, 1].set_title('Ground Truth')
                    axs_jacc[counter_model, 1].set_xticks([])
                    axs_jacc[counter_model, 1].set_yticks([])
                    axs_jacc[counter_model, 2].imshow(im_pred)
                    axs_jacc[counter_model, 2].set_title('Pred. Orig.')
                    axs_jacc[counter_model, 2].set_xticks([])
                    axs_jacc[counter_model, 2].set_yticks([])
                    axs_jacc[counter_model, 2].set_xlabel( list_models[counter_model] ) 
                    axs_jacc[counter_model, 3].imshow(img_perturbed)
                    axs_jacc[counter_model, 3].set_title('Pert. Image')
                    axs_jacc[counter_model, 3].set_xticks([])
                    axs_jacc[counter_model, 3].set_yticks([])
                    #axs[counter_model, 3].yaxis.set_label_position("right")
                    axs_jacc[counter_model, 3].set_ylabel( 'Severity: ' + str(int(rec_Pert.loc[IND, 'Severity'])) ) 
                    axs_jacc[counter_model, 3].set_xlabel(rec_Pert.loc[IND, 'Perturbation Type'].replace('_', ' ').title())
                    axs_jacc[counter_model, 4].imshow(im_pred_perturebed)            
                    axs_jacc[counter_model, 4].set_title('Pred. Pert.')
                    axs_jacc[counter_model, 4].set_xticks([])
                    axs_jacc[counter_model, 4].set_yticks([])
                    axs_jacc[counter_model, 4].set_xlabel( list_models[counter_model] ) 
        
                    flag_done_thisMethod_jacc=True
                    if counter_model==1: repeat = (maskID, rec_Pert.loc[IND:IND+1, rec_Pert.columns])
                    break


fig_jacc.tight_layout() 
# plt.show()

fig_jacc.savefig('./figures/fundus_byJACC_models.png', dpi=300, bbox_inches = 'tight')         