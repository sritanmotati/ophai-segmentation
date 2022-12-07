import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.keras.preprocessing import image
import cv2

def cf(img):
    return np.moveaxis(img, -1, 0)

def np2torch(arr):
    new_arr = []
    for i in range(arr.shape[0]):
        new_arr.append(cf(arr[i]))
    return torch.tensor(np.array(new_arr), dtype=torch.float32)

def crop_disc(img, mask):
    l,r = 0,mask.shape[1]-1
    while l < mask.shape[1] and not any([x>0 for x in mask[:, l]]):
        l += 1
    while r >= 0 and not any([x>0 for x in mask[:, r]]):
        r -= 1
    t,b = 0,mask.shape[0]-1
    while t < mask.shape[0] and not any([x>0 for x in mask[t]]):
        t += 1
    while b >= 0 and not any([x>0 for x in mask[b]]):
        b -= 1
    buf=30
    end = lambda n,d : mask.shape[d]-1-n
    img = img[t-(buf if t>buf else t):b+(buf if end(b,0)>buf else mask.shape[0]-1), l-(buf if l>buf else l):r+(buf if end(r,1)>buf else mask.shape[1]-1)]
    mask = mask[t-(buf if t>buf else t):b+(buf if end(b,0)>buf else mask.shape[0]-1), l-(buf if l>buf else l):r+(buf if end(r,1)>buf else mask.shape[1]-1)]
    return img, mask

def process_pair(x,y,img_size,crop=False,channelsFirst=False,binary=False,polar=False):
    img = io.imread(x)
    mask = io.imread(y)
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
    mask = mask[t:b, l:r]
    if crop:
        img, mask = crop_disc(img, mask)
    img = Image.fromarray(img).resize(img_size)
    img = np.asarray(img).astype('float32')
    mask = Image.fromarray(mask).resize(img_size)
    mask = np.asarray(mask)
    mask[mask<100] = 0
    mask[mask>200] = 2
    mask[mask>2] = 1
    if binary:
        mask[mask>1] = 1
        mask = np.expand_dims(mask, axis=-1)
    else:
        mask = tf.keras.utils.to_categorical(mask).astype('float32')
    if polar:
        SHAPE = img_size[0]
        img = rotate(cv2.linearPolar(img, (SHAPE / 2, SHAPE / 2), SHAPE / 2, cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
        mask = rotate(cv2.linearPolar(mask, (SHAPE / 2, SHAPE / 2), SHAPE / 2, cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = mask * 2.
        mask = np.rint(mask)
    if channelsFirst:
        img = cf(img)
        mask = cf(mask)
    return img, mask

def fundus_gen(paths, batch_size, img_size, crop=False,channelsFirst=False,binary=False,polar=False):
    while True:
        batch_paths = np.random.choice(a=paths, size=batch_size)
        batch_img = []
        batch_mask = []
        for img_path, mask_path in batch_paths:
            img, mask = process_pair(img_path, mask_path, img_size, crop=crop,channelsFirst=channelsFirst,binary=binary,polar=polar)
            batch_img.append(img)
            batch_mask.append(mask)
        batch_img = np.array(batch_img) / 255.
        batch_mask = np.array(batch_mask)
        yield (batch_img, batch_mask)

def viz(x,y, title=''):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(x); ax2.imshow(y)
    if title: fig.suptitle(title)
    plt.show()

def eval_pred(y_true, y_pred):
    fig, (a1, a2) = plt.subplots(nrows=1, ncols=2)
    a1.imshow(y_true)
    a1.set_title('True mask')
    a2.imshow(y_pred)
    a2.set_title('Predicted mask')
    plt.show()