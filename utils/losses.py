from tensorflow.keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_multi(y_true, y_pred):
    score0 = dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    score1 = dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])
    try:
        score2 = dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])
        return (score0+score1+score2) / 3.
    except:
        return (score0+score1) / 2.

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_multi_loss(y_true, y_pred):
    return 1-dice_coef_multi(y_true, y_pred)