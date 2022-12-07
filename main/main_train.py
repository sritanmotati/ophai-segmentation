import argparse

parser = argparse.ArgumentParser(description='Train model (Keras).')
parser.add_argument('--model_name', help='Name of model to train.', choices=['attnet', 'cenet', 'deeplabv3plus', 'doubleunet', 'mnet', 'mobilenet_unet', 'resnet_unet', 'resunet', 'unet', 'unetpp'])
parser.add_argument('--name_csv_train', help='Name of the CSV file with training dataset information.', required=True)
parser.add_argument('--data_dir', help='Path to the folder with the CSV files and image subfolders.', required=True)
parser.add_argument('--path_save_model', help='Path to the folder where model will be saved.', required=True)
parser.add_argument('--img_size', type=int, help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--epochs', type=int, help='Number of epochs for the model to train.', default=20)
parser.add_argument('--batch_size', type=int, help='Batch size for the model during training.', default=4)

args = parser.parse_args()

import numpy as np 
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import pandas as pd

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

train_path = os.path.join(args.data_dir, args.name_csv_train)
img_size = (args.img_size, args.img_size)

dataset_dir = args.data_dir

df_train = pd.read_csv(train_path).drop(['ind', 'Fovea_X', 'Fovea_Y', 'patientID'], axis=1).values.tolist()

train_paths = []
for r in df_train:
    train_paths.append((os.path.join(dataset_dir, r[1], r[0]), os.path.join(dataset_dir, r[2], r[0].replace('tif', 'png').replace('jpg', 'png'))))

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

val_size=0.1
train_gen, val_gen, _ = model.get_gens(train_paths, [], args.batch_size, val_size=val_size)
history = model.train(train_gen, val_gen, int(len(train_paths)*(1-val_size))//args.batch_size, int(len(train_paths)*val_size)//args.batch_size)

sp = args.path_save_model
if not os.path.isdir(sp): os.mkdir(sp)
os.chdir(sp)
torch_models = ['cenet']
model.save(f'{args.model_name}_model' + ['.h5', '.pth'][args.model_name in torch_models])

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig(f'{args.model_name}_loss.png')