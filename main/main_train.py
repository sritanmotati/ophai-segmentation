import argparse

parser = argparse.ArgumentParser(description='Train model (Keras).')
parser.add_argument('--model_name', help='Name of model to train.', choices=['unet', 'attnet', 'deeplabv3plus', 'doubleunet', 'mobilenet_unet', 'resnet_unet', 'resunet'])
parser.add_argument('--path_npy', help='Path to the folder with the NPY files for the X_train and y_train data.', required=True)
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

from models.attnet import AttNet
from models.deeplabv3plus import DeepLabV3Plus
from models.doubleunet import DoubleUNet
from models.mobilenet_unet import MobileNetUNet
from models.resnet_unet import ResNetUNet
from models.resunet import ResUNet
from models.unet import UNet

X_train = np.load(os.path.join(args.path_npy, 'X_train.npy'))
y_train = np.load(os.path.join(args.path_npy, 'y_train.npy'))

model = {
    'attnet': AttNet,
    'deeplabv3plus': DeepLabV3Plus,
    'doubleunet': DoubleUNet,
    'mobilenet_unet': MobileNetUNet,
    'resnet_unet': ResNetUNet,
    'resunet': ResUNet,
    'unet': UNet
}[args.model_name](X_train[0].shape, 3)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model.summary()

num_epochs = args.epochs
bs = args.batch_size

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(X_train, tf.keras.utils.to_categorical(y_train), epochs=num_epochs, validation_split=0.1, batch_size=bs, callbacks=[callback])

sp = args.path_save_model
if not os.path.isdir(sp): os.mkdir(sp)
os.chdir(sp)
model.save(f'{args.model_name}_model.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy During Training')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig(f'{args.model_name}_acc.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss During Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()
plt.savefig(f'{args.model_name}_loss.png')