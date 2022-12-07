import argparse

parser = argparse.ArgumentParser(description='Test model (Keras).')
parser.add_argument('--model_name', help='Name of model to train.', choices=['attnet', 'cenet', 'deeplabv3plus', 'doubleunet', 'mnet', 'mobilenet_unet', 'resnet_unet', 'resunet', 'unet', 'unetpp'])
parser.add_argument('--name_csv_test', help='Name of the CSV file with testing dataset information.', required=True)
parser.add_argument('--data_dir', help='Path to the folder with the CSV files and image subfolders.', required=True)
parser.add_argument('--path_trained_model', help='Path to the saved model.', required=True)
parser.add_argument('--img_size', type=int, help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--path_save_results', help='Path to the folder where results and predictions will be saved.', required=True)

args = parser.parse_args()

from metrics import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from utils.data_utils import *
import pandas as pd

test_path = os.path.join(args.data_dir, args.name_csv_test)
img_size = (args.img_size, args.img_size)
df_test = pd.read_csv(test_path).drop(['ind', 'Fovea_X', 'Fovea_Y', 'patientID'], axis=1).values.tolist()
test_paths = []
for r in df_test:
    test_paths.append((os.path.join(args.data_dir, r[1], r[0]), os.path.join(args.data_dir, r[2], r[0].replace('tif', 'png').replace('jpg', 'png'))))

from models.attnet import AttNet
from models.cenet import CENet
from models.deeplabv3plus import DeepLabV3Plus
from models.doubleunet import DoubleUNet
from models.mnet import MNet
from models.mobilenet_unet import MobileNetUNet
from models.resnet_unet import ResNetUNet
from models.resunet import ResUNet
from models.unet import UNet
from models.unetpp import UnetPlusPlus

from utils.data_utils import *

sp = args.path_save_results
if not os.path.isdir(sp): os.mkdir(sp)
os.chdir(sp)

if not os.path.isdir('predictions'): os.mkdir('predictions')

img_size = (args.img_size, args.img_size)
model = {
    'attnet': AttNet,
    'cenet': CENet,
    'deeplabv3plus': DeepLabV3Plus,
    'doubleunet': DoubleUNet,
    'mnet': MNet,
    'mobilenet_unet': MobileNetUNet,
    'resnet_unet': ResNetUNet,
    'resunet': ResUNet,
    'unet': UNet,
    'unetpp': UnetPlusPlus
}[args.model_name]((img_size[0],img_size[1],3), 3) # only important for unet models, SOTA models have their own size/n_channels and this will be disregarded

_, _, test_gen = model.get_gens([], test_paths, 1, val_size=0)

model.load(args.path_trained_model)

class_stats = [{'dice':0, 'jacc':0, 'tp':0, 'fp':0, 'fn':0, 'tn':0, 'precision':0, 'recall':0, 'f1':0} for i in range(3)]; acc = 0

torch_models = ['cenet']

for i in range(len(test_paths)):
  x, y = next(test_gen)
  pred = model.predict(x)[0]
  y = y[0]
  if args.model_name in torch_models:
    pred = np.moveaxis(pred, 0, -1)
    y = np.moveaxis(y, 0, -1)
  im_pred = np.argmax(pred, axis=2) if pred.shape[-1] > 1 else pred.reshape(pred.shape[:-1])
  im = Image.fromarray((im_pred * 255).astype(np.uint8))
  im_name = f'pred_{i}.jpg'
  im.save(os.path.join('predictions', im_name))
  gt = y
  acc += accuracy(gt, pred)
  dice = dice_coef_multilabel(gt, pred, y.shape[-1])
  jacc = jaccard_coef_multilabel(gt, pred, y.shape[-1])
  tfstats = tf_stats_multiclass(gt, pred, y.shape[-1])
  for c in range(y.shape[-1]):
    class_stats[c]['dice'] += dice[c]
    class_stats[c]['jacc'] += jacc[c]
    for k in ['tp','fp','fn','tn','precision','recall','f1']:
      class_stats[c][k] += tfstats[c][k]

acc /= len(test_paths)
for i in range(y.shape[-1]):
    for k in class_stats[i]:
        class_stats[i][k] /= len(test_paths)

output = f'Accuracy: {acc}\n\n'

# class_names = ['Rest of eye', 'Optic cup', 'Optic disk'] # 3 classes
for c in range(y.shape[-1]):
    output += f'CLASS {c}:\n'
    output += '--------------------------\n'
    output += f"Dice score: {class_stats[c]['dice']}\n"
    output += f"IoU (Jaccard score): {class_stats[c]['jacc']}\n"
    output += f"True positives: {class_stats[c]['tp']}\n"
    output += f"False positives: {class_stats[c]['fp']}\n"
    output += f"True negatives: {class_stats[c]['tn']}\n"
    output += f"False negatives: {class_stats[c]['fn']}\n"
    output += f"Precision: {class_stats[c]['precision']}\n"
    output += f"Recall: {class_stats[c]['recall']}\n"
    output += f"F1 score: {class_stats[c]['f1']}\n\n"

print(output)

out_write = open('results.txt', 'w')
n = out_write.write(output)
out_write.close()