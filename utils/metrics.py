import numpy as np

# Pixel-wise accuracy
def accuracy(y_true, y_pred):
    return np.sum((y_true==y_pred).astype(int)) / y_true.size

# Accuracy for all classes
def accuracy_multilabel(y_true, y_pred, numLabels):
    accs = []
    for index in range(numLabels):
        ac= accuracy((y_true==index).astype(int), (y_pred==index).astype(int))
        accs.append(ac)
    return accs

# One vs. rest dice coefficient
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 1e-7
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# Dice coefficients for all classes
def dice_coef_multilabel(y_true, y_pred, numLabels):
    dices = []
    for index in range(numLabels):
        dc= dice_coef((y_true==index).astype(int), (y_pred==index).astype(int))
        dices.append(dc)
    return dices

# Jaccard score / intersection over union (IoU) for one vs. rest
def jaccard_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    smooth = 1e-7
    return intersection / (union + smooth)

# Jaccard score / IoU for all classes
def jaccard_coef_multilabel(y_true, y_pred, numLabels):
    jcs = []
    for i in range(numLabels):
        jc = jaccard_coef((y_true==i).astype(int), (y_pred==i).astype(int))
        jcs.append(jc)
    return jcs

# True positives, true negatives, false positives, false negatives, precision, recall, F1 score (for one class vs. rest)
def tf_stats(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    TP = ((y_pred == 1) & (y_true == 1)).sum() / y_true.shape[0]
    FP = ((y_pred == 1) & (y_true == 0)).sum() / y_true.shape[0]
    TN = ((y_pred == 0) & (y_true == 0)).sum() / y_true.shape[0]
    FN = ((y_pred == 0) & (y_true == 1)).sum() / y_true.shape[0]
    smooth = 1e-7
    precision = TP/(TP+FP+smooth)
    recall = TP/(TP+FN+smooth)
    f1 = 2*TP / (2*TP+FP+FN+smooth)
    return {'tp':TP, 'fp':FP, 'fn':FN, 'tn':TN, 'precision':precision, 'recall':recall, 'f1':f1}

# Above stats for each class (list of dictionaries)
def tf_stats_multiclass(y_true, y_pred, numLabels):
    stats = []
    for i in range(numLabels):
        tf = tf_stats((y_true==i).astype(int), (y_pred==i).astype(int))
        stats.append(tf)
    return stats