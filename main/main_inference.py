import argparse

parser = argparse.ArgumentParser(description='Make inference using UNet model.')
parser.add_argument('--path_img', help='Path to the image.', required=True)
parser.add_argument('--path_trained_model', help='Path to the saved model.', required=True)
parser.add_argument('--path_save', help='Path to the folder where prediction will be saved.', required=True)
parser.add_argument('--img_size', type=int, help='Size to which the image should be reshaped (one number, i.e. 256 or 512).', required=True)

args = parser.parse_args()

import tensorflow as tf
import os
import skimage.io as io
from PIL import Image
import numpy as np

img = io.imread(args.path_img)
l,r = 0,img.shape[1]-1
threshold = 100
while l < img.shape[1] and not any([x[0] > threshold for x in img[:, l]]):
    l += 1
while r >= 0 and not any([x[0] > threshold for x in img[:, r]]):
    r -= 1
t,b = 0,img.shape[0]-1
while t < img.shape[0] and not any([x[0] > threshold for x in img[t]]):
    t += 1
while b >= 0 and not any([x[0] > threshold for x in img[b]]):
    b -= 1
img = img[t:b, l:r]
img = Image.fromarray(img).resize(args.img_size)
img = np.asarray(img)

model = tf.keras.models.load_model(args.path_trained_model)

pred = model.predict(np.array([img]))[0]
im_pred = np.argmax(pred, axis=2)
im = Image.fromarray((im_pred * 255).astype(np.uint8))
im_name = f'pred_{os.path.splitext(args.path_img)[0].split("/")[-1]}.jpg'
im.save(os.path.join(args.path_save, im_name))

print('Prediction saved!')