import argparse

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('--model-name', help='Name of model to train.', choices=['attnet', 'cenet', 'deeplabv3plus', 'doubleunet', 'mnet', 'mobilenet_unet', 'resnet_unet', 'resunet', 'unet', 'unetpp'], required=True)
parser.add_argument('--name-csv-train', help='Name of the CSV file with training dataset information.', required=True)
parser.add_argument('--data-dir', help='Path to the folder with the CSV files and image subfolders.', required=True)
parser.add_argument('--path-save', help='Path to the folder where model will be saved.', required=True)
parser.add_argument('--img-size', type=int, help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--batch-size', type=int, help='Batch size for the model during training.', default=4)
parser.add_argument('--binary', type=bool, help='Whether the segmentation masks are binary (True) or multi-class (False).', default=False)

args = parser.parse_args()

import os
import os.path as osp
import matplotlib.pyplot as plt
import pandas as pd

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

train_path = osp.join(args.data_dir, args.name_csv_train)
img_size = (args.img_size, args.img_size)

dataset_dir = args.data_dir

df_train = pd.read_csv(train_path)[['imageID', 'imageDIR', 'segDIR']].values.tolist()

train_paths = []
for r in df_train:
    img_path = osp.join(osp.split(dataset_dir)[0], r[1], r[0])
    mask_path = osp.join(osp.split(dataset_dir)[0], r[2], r[0])
    train_paths.append((img_path, mask_path))

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
}[args.model_name]((img_size[0],img_size[1],3), 1 if args.binary else 3) # only important for unet models, SOTA models have their own size/n_channels and this will be disregarded

torch_models = ['cenet']
polar_models = ['mnet']

val_size=0.1
train_gen, val_gen, _ = get_gens(img_size, train_paths, [], args.batch_size, val_size=val_size, binary=args.binary, polar=(args.model_name in polar_models), channelsFirst=(args.model_name in torch_models))
train_len = int(len(train_paths)*(1-val_size))
val_len = len(train_paths) - train_len

# models needing extra config
if args.model_name == 'attnet':
    model.set_config_params(args.batch_size, train_len, val_len)

history = model.train(train_gen, val_gen, train_len//args.batch_size, val_len//args.batch_size)

sp = args.path_save
if not os.path.isdir(sp): os.makedirs(sp)
os.chdir(sp)
model.save(f'{args.model_name}_model' + ['.h5', '.pth'][args.model_name in torch_models])

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig(f'{args.model_name}_loss.png')

out_write = open('losses.csv', 'w')
out_write.write('epoch,loss,val_loss')
for i in range(len(history['loss'])):
    n = out_write.write(f"{i},{history['loss'][i]},{history['val_loss'][i]}")
out_write.close()