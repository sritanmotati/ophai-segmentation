import argparse

parser = argparse.ArgumentParser(description='Test model (Keras).')
parser.add_argument('--path_npy', help='Path to the folder with the NPY files for the X_test and y_test data.', required=True)
parser.add_argument('--path_trained_model', help='Path to the saved model.', required=True)
parser.add_argument('--path_save_results', help='Path to the folder where results and predictions will be saved.', required=True)

args = parser.parse_args()

from metrics import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image

X_test = np.load(os.path.join(args.path_npy, 'X_test.npy'))
y_test = np.load(os.path.join(args.path_npy, 'y_test.npy'))

model = tf.keras.models.load_model(args.path_trained_model)

sp = args.path_save_results
if not os.path.isdir(sp): os.mkdir(sp)
os.chdir(sp)

if not os.path.isdir('predictions'): os.mkdir('predictions')

def eval_pred(x_true, y_true, y_pred):
  y_pred = np.argmax(y_pred, axis=2)
  fig, (a1, a2, a3) = plt.subplots(nrows=1, ncols=3)
  a1.imshow(x_true)
  a1.set_title('Input')
  a2.imshow(y_true)
  a2.set_title('True mask')
  a3.imshow(y_pred)
  a3.set_title('Predicted mask')
  plt.show()

preds = []
for i in range(X_test.shape[0]):
  pred = model.predict(np.array([X_test[i]]))[0]
  im_pred = np.argmax(pred, axis=2)
  preds.append(im_pred)
  im = Image.fromarray((im_pred * 255).astype(np.uint8))
  im_name = f'pred_{i}.jpg'
  im.save(os.path.join('predictions', im_name))

class_stats = [{'dice':0, 'jacc':0, 'tp':0, 'fp':0, 'fn':0, 'tn':0, 'precision':0, 'recall':0, 'f1':0} for i in range(3)]; acc = 0
for i,pred in enumerate(preds):
  gt = y_test[i]
  acc += accuracy(gt, pred)
  dice = dice_coef_multilabel(gt, pred, 3)
  jacc = jaccard_coef_multilabel(gt, pred, 3)
  tfstats = tf_stats_multiclass(gt, pred, 3)
  for c in range(3):
    class_stats[c]['dice'] += dice[c]
    class_stats[c]['jacc'] += jacc[c]
    for k in ['tp','fp','fn','tn','precision','recall','f1']:
      class_stats[c][k] += tfstats[c][k]

acc /= len(preds)
for i in range(3):
    for k in class_stats[i]:
        class_stats[i][k] /= len(preds)

output = f'Accuracy: {acc}\n\n'

class_names = ['Rest of eye', 'Optic cup', 'Optic disk']
for c in range(3):
    output += f'CLASS {c} - {class_names[c]}:\n'
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