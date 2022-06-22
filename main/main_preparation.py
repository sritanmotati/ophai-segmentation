import argparse

parser = argparse.ArgumentParser(description='Load data for fundus segmentation (SegLoc, OPHAI).')
parser.add_argument('--name_csv_train', help='Name of the CSV file with training dataset information.', required=True)
parser.add_argument('--name_csv_test', help='Name of the CSV file with testing dataset information.', required=True)
parser.add_argument('--data_dir', help='Path to the folder with the CSV files and image subfolders.', required=True)
parser.add_argument('--path_save_npy', help='Path to the folder where NumPy files of the dataset will be saved.', required=True)
parser.add_argument('--img_size', type=int,  help='Size to which the images should be reshaped (ex. 256 or 512).', required=True)

args = parser.parse_args()

import numpy as np 
import os
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

train_path = os.path.join(args.data_dir, args.name_csv_train); test_path = os.path.join(args.data_dir, args.name_csv_test); dataset_dir = args.data_dir
img_size = (args.img_size, args.img_size)

df_train = pd.read_csv(train_path).drop(['ind', 'Fovea_X', 'Fovea_Y', 'patientID'], axis=1).values.tolist()
df_test = pd.read_csv(test_path).drop(['ind', 'Fovea_X', 'Fovea_Y', 'patientID'], axis=1).values.tolist()

train_paths = []
for r in df_train:
    train_paths.append((os.path.join(dataset_dir, r[1], r[0]), os.path.join(dataset_dir, r[2], r[0].replace('tif', 'png').replace('jpg', 'png'))))

test_paths = []
for r in df_test:
    test_paths.append((os.path.join(dataset_dir, r[1], r[0]), os.path.join(dataset_dir, r[2], r[0].replace('tif', 'png').replace('jpg', 'png'))))

def viz(x,y, title=''):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(x); ax2.imshow(y)
    if title: fig.suptitle(title)
    plt.show()

def process_pair(x,y):
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
    img = Image.fromarray(img).resize(img_size)
    img = np.asarray(img)
    mask = Image.fromarray(mask).resize(img_size)
    mask = np.asarray(mask)
    return img, mask

X_train, y_train, X_test, y_test = [], [], [], []
for index, p in enumerate(train_paths):
    x,y = p
    img,mask = process_pair(x,y)
    mask = mask.tolist()
    for i,r in enumerate(mask):
        for j,c in enumerate(r):
            if c in range(100,200): mask[i][j] = 1
            elif c < 100: mask[i][j] = 0
            else: mask[i][j] = 2
    mask = np.array(mask)
    if index == 0: viz(img, mask)
    X_train.append(img)
    y_train.append(mask)
    print(f'{index+1}/{len(train_paths)} TRAIN')
  
for index, p in enumerate(test_paths):
    x,y = p
    img,mask = process_pair(x,y)
    mask = mask.tolist()
    for i,r in enumerate(mask):
        for j,c in enumerate(r):
            if c in range(100,200): mask[i][j] = 1
            elif c < 100: mask[i][j] = 0
            else: mask[i][j] = 2
    mask = np.array(mask)
    if index == 0: viz(img, mask)
    X_test.append(img)
    y_test.append(mask)
    print(f'{index+1}/{len(test_paths)} TEST')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

np.save(os.path.join(args.path_save_npy, 'X_train.npy'), X_train)
np.save(os.path.join(args.path_save_npy, 'X_test.npy'), X_test)
np.save(os.path.join(args.path_save_npy, 'y_train.npy'), y_train)
np.save(os.path.join(args.path_save_npy, 'y_test.npy'), y_test)