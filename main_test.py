import argparse

parser = argparse.ArgumentParser(description='Test model (Keras).')
parser.add_argument('--model-name', help='Name of model to train.', choices=['attnet', 'cenet', 'deeplabv3plus', 'doubleunet', 'mnet', 'mobilenet_unet', 'resnet_unet', 'resunet', 'unet', 'unetpp'])
parser.add_argument('--name-csv-test', help='Name of the CSV file with testing dataset information.', required=True)
parser.add_argument('--data-dir', help='Path to the folder with the CSV files and image subfolders.', required=True)
parser.add_argument('--path-model', help='Path to the saved model.', required=True)
parser.add_argument('--img-size', type=int, help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--path-save-results', help='Path to the folder where results and predictions will be saved.', required=True)
parser.add_argument('--save-masks', type=bool, help='Whether you want to save all predicted masks for test set or not.', default=False)
parser.add_argument('--binary', type=bool, help='Whether the segmentation masks are binary (True) or multi-class (False).', default=False)

args = parser.parse_args()

from utils.metrics import *
import numpy as np
import os
from PIL import Image
from utils.data_utils import *
import pandas as pd
import os.path as osp
from tqdm import tqdm

test_path = os.path.join(args.data_dir, args.name_csv_test)
img_size = (args.img_size, args.img_size)
df_test = pd.read_csv(test_path)[['imageID', 'imageDIR', 'segDIR', 'maskID']].values.tolist()

test_paths = []
for r in df_test:
    if os.path.exists(os.path.join(args.data_dir, r[1]).replace('\\', '/')):
        img_path = os.path.join(args.data_dir, r[1], r[0]).replace('\\', '/')
        mask_path = os.path.join(args.data_dir, r[2], r[3]).replace('\\', '/')
        test_paths.append((img_path, mask_path))
    else:
        img_path = osp.join(osp.split(args.data_dir)[0], r[1], r[0]).replace('\\', '/')
        mask_path = osp.join(osp.split(args.data_dir)[0], r[2], r[3]).replace('\\', '/')
        test_paths.append((img_path, mask_path))

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

sp = args.path_save_results
if not os.path.isdir(sp): os.makedirs(sp)

if args.save_masks and not os.path.isdir(osp.join(sp,'preds')): os.makedirs(osp.join(sp,'preds'))

img_size = (args.img_size, args.img_size)
n_classes = 2 if args.binary else 3
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
}[args.model_name]((img_size[0],img_size[1],3), n_classes) # only important for unet models, SOTA models have their own size/n_channels and this will be disregarded

torch_models = ['cenet']
polar_models = ['mnet']

_, _, test_gen = get_gens(img_size, [], test_paths, 1, val_size=0, binary=args.binary, polar=(args.model_name in polar_models), channelsFirst=(args.model_name in torch_models))

model.load(args.path_model)

class_stats = [{'acc':0., 'dice':0., 'jacc':0.,
                    'tpr':0., 'fpr':0., 'fnr':0., 'tnr':0., 
                    'precision':0., 'recall':0., 'f1':0.} for i in range(n_classes)]

df= pd.DataFrame(columns=['FullFileName', 'MaskFileName', 'class','acc','dice',
                          'tpr','fpr','tnr','fnr','precision','recall','f1' ])

counter = -1
for i in tqdm(range(len(test_paths))):
    x, y = next(test_gen)
    pred = model.predict(x)[0]
    y = y[0]
    if args.model_name in torch_models:
      pred = np.moveaxis(pred, 0, -1)
      y = np.moveaxis(y, 0, -1)
    im_pred = np.argmax(pred, axis=2) if pred.shape[-1] > 1 else pred.reshape(pred.shape[:-1])
    if args.save_masks:
      im = Image.fromarray((im_pred * 255).astype(np.uint8))
      im_name = f'pred_{i}.jpg'
      im.save(os.path.join('preds', im_name))
    im_gt = np.argmax(y, axis=-1)
    
    acc = accuracy_multilabel(im_gt, im_pred, y.shape[-1])
    dice = dice_coef_multilabel(im_gt, im_pred, y.shape[-1])
    jacc = jaccard_coef_multilabel(im_gt, im_pred, y.shape[-1])
    tfstats = tf_stats_multiclass(im_gt, im_pred, y.shape[-1])
  
    for c in range(y.shape[-1]):
        
        class_stats[c]['acc'] += acc[c]
        class_stats[c]['dice'] += dice[c]
        class_stats[c]['jacc'] += jacc[c]

        counter = counter+1         
        
        df.loc[counter, 'FullFileName'] = test_paths[i][0]
        df.loc[counter, 'MaskFileName'] = test_paths[i][1]
        df.loc[counter, 'class'] = c
        df.loc[counter, 'acc'] = acc[c]
        df.loc[counter, 'dice'] = dice[c]
        df.loc[counter, 'jacc'] = jacc[c]
        
        for k in ['tpr','fpr','fnr','tnr','precision','recall','f1']:
            class_stats[c][k] += tfstats[c][k]            
            df.loc[counter, k] = tfstats[c][k]
       

for i in range(n_classes):
    for k in class_stats[i]:
        class_stats[i][k] /= len(test_paths)

df_overall= pd.DataFrame(columns=['class','acc','dice',
                          'tpr','fpr','tnr','fnr','precision','recall','f1' ])
for c in range(n_classes):
    df_overall.loc[c, 'class'] = c
    for k in ['acc','dice','jacc','tpr','fpr','fnr','tnr','precision','recall','f1']:
        df_overall.loc[c, k] = class_stats[c][k]

# out_write = open(os.path.join(sp, args.saveNameCSV[:-4]+'_overal.csv'), 'w')
# n = out_write.write(output)
# out_write.close()

df.to_csv(os.path.join(sp, 'individual_results.csv'), index=False)
df_overall.to_csv(os.path.join(sp, 'overall_results.csv'), index=False)

# class_stats = [{'acc':0, 'dice':0, 'jacc':0, 'tp':0, 'fp':0, 'fn':0, 'tn':0, 'precision':0, 'recall':0, 'f1':0} for i in range(3)]; acc = 0

# for i in tqdm(range(len(test_paths))):
#   x, y = next(test_gen)
#   pred = model.predict(x)[0]
#   y = y[0]
#   if args.model_name in torch_models:
#     pred = np.moveaxis(pred, 0, -1)
#     y = np.moveaxis(y, 0, -1)
#   im_pred = np.argmax(pred, axis=2)
#   if args.save_masks:
#     im = Image.fromarray((im_pred * 255).astype(np.uint8))
#     im_name = f'pred_{i}.jpg'
#     im.save(os.path.join(sp,'preds', im_name))
#   gt = np.argmax(y, axis=-1)
#   n_classes=y.shape[-1]
#   acc = accuracy_multilabel(gt, im_pred, n_classes)
#   dice = dice_coef_multilabel(gt, im_pred, n_classes)
#   jacc = jaccard_coef_multilabel(gt, im_pred, n_classes)
#   tfstats = tf_stats_multiclass(gt, im_pred, n_classes)
#   for c in range(y.shape[-1]):
#     class_stats[c]['acc'] += acc[c]
#     class_stats[c]['dice'] += dice[c]
#     class_stats[c]['jacc'] += jacc[c]
#     for k in ['tp','fp','fn','tn','precision','recall','f1']:
#       class_stats[c][k] += tfstats[c][k]

# for i in range(y.shape[-1]):
#     for k in class_stats[i]:
#         class_stats[i][k] /= len(test_paths)

# output = 'class,accuracy,dice,iou,true_positive,false_positive,true_negative,false_negative,precision,recall,f1\n'

# # class_names = ['Rest of eye', 'Optic cup', 'Optic disk'] # 3 classes
# for c in range(y.shape[-1]):
#     output += f"{c},"
#     output += f"{class_stats[c]['acc']},"
#     output += f"{class_stats[c]['dice']},"
#     output += f"{class_stats[c]['jacc']},"
#     output += f"{class_stats[c]['tp']},"
#     output += f"{class_stats[c]['fp']},"
#     output += f"{class_stats[c]['tn']},"
#     output += f"{class_stats[c]['fn']},"
#     output += f"{class_stats[c]['precision']},"
#     output += f"{class_stats[c]['recall']},"
#     output += f"{class_stats[c]['f1']}\n"

# out_write = open(osp.join(sp,'results.csv'), 'w')
# n = out_write.write(output)
# out_write.close()