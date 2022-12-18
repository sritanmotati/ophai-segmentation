import random
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.layers import Lambda, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, Dropout, BatchNormalization, concatenate, Activation
import tensorflow.keras
from tensorflow.keras.layers import Add, Multiply
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import math
from skimage import morphology, measure
import cv2
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import h5py
import numpy as np
from utils.data_utils import *
from utils.losses import *

def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    create dir
    :param dir_name: 文件夹列表
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(u'[INFO] Dir "%s" exists, deleting.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(u'[INFO] Dir "%s" not exists, creating.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def genMasks(masks, channels):
    """
    为groundtruth生成相反的掩膜
    generate masks for groundtruth
    :param masks:  groundtruth图
    :param channels: 通道数
    :return:
    """
    assert (len(masks.shape) == 4)
    im_h = masks.shape[2]
    im_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], channels, im_h * im_w))
    new_masks = np.empty((masks.shape[0], im_h * im_w, channels + 1))

    new_masks[:, :, 0:channels] = masks[:, 0:channels, :].transpose(0, 2, 1)
    maskTotal = np.ma.array(new_masks[:, :, 0], mask=new_masks[:, :, 0]).mask
    for index in range(channels):
        mask = new_masks[:, :, index]
        m = np.ma.array(new_masks[:, :, index], mask=mask)
        maskTotal = maskTotal | m.mask

    new_masks[:, :, channels] = 1 - maskTotal
    return new_masks


def gray2binary(image, threshold=0.5):
    """
    灰度图二值化
    :param image: 灰度图
    :param threshold: 二值化阈值
    :return: 二值图
    """
    image = (image >= threshold) * 1
    return image


def colorize(img, gt, prob):
    image = np.copy(img)
    if np.max(gt) > 1:
        gt = gt/255.
    gtlist = [np.where(gt >= 0.5)]
    problist = [np.where(prob == 1)]
    gtx = gtlist[0][0]
    gty = gtlist[0][1]
    for index in range(gtx.shape[0]):           # gt区域标为绿色
        image[gtx[index], gty[index], 0] = 0
        image[gtx[index], gty[index], 1] = 1
        image[gtx[index], gty[index], 2] = 0

    probx = problist[0][0]
    proby = problist[0][1]
    for index in range(probx.shape[0]):
        if image[probx[index], proby[index], 1] != 1:    # 预测错误区域标为红色
            image[probx[index], proby[index], 0] = 1
            image[probx[index], proby[index], 1] = 0
            image[probx[index], proby[index], 2] = 0
        else:  # 预测正确区域标为蓝色
            image[probx[index], proby[index], 0] = 0
            image[probx[index], proby[index], 1] = 0
            image[probx[index], proby[index], 2] = 1
    return image


def visualize(image, subplot):
    """
    将多张大小相同的图片拼接
    :param image: 图片列表
    :param subplot: 行列数[row,col]
    :return: 拼接图
    """
    row = subplot[0]
    col = subplot[1]
    height, width = image[0].shape[:2]
    result = np.zeros((height*row, width*col, 3))

    total_image = len(image)
    index = 0
    for i in range(row):
        for j in range(col):
            row_index = i*height
            col_index = j*width
            if index < total_image:
                try:  # 单通道灰度图与3通道彩色图单独处理
                    result[row_index:row_index+height,
                           col_index:col_index+width, :] = image[index]*255
                except:
                    result[row_index:row_index + height,
                           col_index:col_index + width, 0] = image[index]*255
                    result[row_index:row_index + height,
                           col_index:col_index + width, 1] = image[index]*255
                    result[row_index:row_index + height,
                           col_index:col_index + width, 2] = image[index]*255
            index = index+1
    result = result.astype(np.uint8)
    return result


def connectTable(image, min_size, connect):
    label_image = measure.label(image)
    dst = morphology.remove_small_objects(
        label_image, min_size=min_size, connectivity=connect)
    return dst, measure.regionprops(dst)


def countWhite(image):  # 统计二值图中白色区域面积
    return np.count_nonzero(image)


def imgResize(image, scale):
    dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def postprocess(probResult, probImage):
    dst, regionprops = connectTable(probResult, 3000, 1)
    result = np.zeros_like(probResult)
    prob = np.zeros_like(probImage)
    candidates = []  # 被选择区域集
    probResult = probResult.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    probResult = cv2.morphologyEx(probResult, cv2.MORPH_CLOSE, kernel)
    probResult = cv2.morphologyEx(probResult, cv2.MORPH_CLOSE, kernel)

    for region in regionprops:  # 循环得到每一个连通区域属性集
        minr, minc, maxr, maxc = region.bbox
        area = (maxr - minr) * (maxc - minc)  # 候选区域面积  area of selected patch

        if math.fabs((maxr - minr) / (maxc - minc)) > 1.3 or math.fabs((maxr - minr) / (maxc - minc)) < 0.8 or area * 4/3.1415926 < countWhite(probResult[minr:maxr, minc:maxc]):
            # 剔除细、长区域和太过夸张的内凹型、外凸形 delete area which too small or big or wide etc
            continue
    # 筛选过的区域与已选择区域合
        candidates.append(region.bbox)
    select_minr = 0
    select_maxr = 0
    select_minc = 0
    select_maxc = 0
    w_h_ratio = 0
    # 从原图中切割选择的区域  cut selected patch from origin image
    for candi in range(len(candidates)):
        minr, minc, maxr, maxc = candidates[candi]
        if math.fabs(w_h_ratio-1.0) > math.fabs((maxr - minr) / (maxc - minc)-1.0):
            select_minr = minr
            select_maxr = maxr
            select_minc = minc
            select_maxc = maxc
    result[select_minr:select_maxr,
           select_minc:select_maxc] = probResult[select_minr:select_maxr, select_minc:select_maxc]
    prob[select_minr:select_maxr,
         select_minc:select_maxc] = probImage[select_minr:select_maxr, select_minc:select_maxc]

    if np.max(prob) == 0:
        prob = probImage
    return result.astype(np.uint8), prob


def get_test_patches(img, config, rl=False):
    """
    将待分割图预处理后，分割成patch
    :param img: 待分割图
    :param config: 配置文件
    :return:
    """
    test_img = []

    test_img.append(img)
    test_img = np.asarray(test_img)

    test_img_adjust = img_process(test_img, rl=rl)  # 预处理
    test_imgs = paint_border(test_img_adjust, config)  # 将图片补足到可被完美分割状态

    test_img_patch = extract_patches(test_imgs, config)  # 依顺序分割patch

    return test_img_patch, test_imgs.shape[1], test_imgs.shape[2], test_img_adjust


def paint_border(imgs, config):
    """
    将图片补足到可被完美分割状态
    :param imgs:  预处理后的图片
    :param config: 配置文件
    :return:
    """
    assert (len(imgs.shape) == 4)
    img_h = imgs.shape[1]  # height of the full image
    img_w = imgs.shape[2]  # width of the full image
    # leftover on the h dim
    leftover_h = (img_h - config.patch_height) % config.stride_height
    # leftover on the w dim
    leftover_w = (img_w - config.patch_width) % config.stride_width
    full_imgs = imgs  # 设置成None时 一些stride情况下会报错，比如stride=1
    if (leftover_h != 0):  # change dimension of img_h
        tmp_imgs = np.zeros(
            (imgs.shape[0], img_h+(config.stride_height-leftover_h), img_w, imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):  # change dimension of img_w
        tmp_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_w+(
            config.stride_width - leftover_w), full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:imgs.shape[1],
                 0:img_w, 0:full_imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    print("new full images shape: \n" + str(full_imgs.shape))
    return full_imgs


def extract_patches(full_imgs, config):
    """
    按顺序分割patch
    :param full_imgs: 补足后的图片
    :param config: 配置文件
    :return: 分割后的patch
    """
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image

    assert ((img_h-config.patch_height) % config.stride_height ==
            0 and (img_w-config.patch_width) % config.stride_width == 0)
    N_patches_img = ((img_h-config.patch_height)//config.stride_height+1)*((img_w -
                                                                            config.patch_width)//config.stride_width+1)  # // --> division between integers
    N_patches_tot = N_patches_img*full_imgs.shape[0]

    patches = np.empty((N_patches_tot, config.patch_height,
                       config.patch_width, full_imgs.shape[3]))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h-config.patch_height)//config.stride_height+1):
            for w in range((img_w-config.patch_width)//config.stride_width+1):
                patch = full_imgs[i, h*config.stride_height:(h*config.stride_height)+config.patch_height, w*config.stride_width:(
                    w*config.stride_width)+config.patch_width, :]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches


def pred_to_patches(pred, config):
    """
    将预测的向量 转换成patch形态
    :param pred: 预测结果
    :param config: 配置文件
    :return: Tensor [-1，patch_height,patch_width,seg_num+1]
    """
    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)

    # (Npatches,height*width)
    pred_images = np.empty((pred.shape[0], pred.shape[1], config.seg_num+1))
    pred_images[:, :, 0:config.seg_num+1] = pred[:, :, 0:config.seg_num+1]
    pred_images = np.reshape(
        pred_images, (pred_images.shape[0], config.patch_height, config.patch_width, config.seg_num+1))
    return pred_images


def img_process(data, rl=False):
    """
    预处理图片
    :param data: 输入图片
    :param rl: 原始图片是否预处理过
    :return: 预处理结果
    """
    assert (len(data.shape) == 4)
    data = data.transpose(0, 3, 1, 2)
    if rl == False:  # 原始图片是否已经预处理过
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros(
                [data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            train_img = dataset_normalized(train_img)  # 归一化
            train_img = clahe_equalized(train_img)  # 限制性直方图归一化
            train_img = adjust_gamma(train_img, 1.2)  # gamma校正
            train_img = train_img/255.  # reduce to 0-1 range
            train_imgs[:, index, :, :] = train_img[:, 0, :, :]

    else:
        train_imgs = np.zeros(data.shape)
        for index in range(data.shape[1]):
            train_img = np.zeros(
                [data.shape[0], 1, data.shape[2], data.shape[3]])
            train_img[:, 0, :, :] = data[:, index, :, :]
            train_img = dataset_normalized(train_img)
            train_imgs[:, index, :, :] = train_img[:, 0, :, :]/255.

    train_imgs = train_imgs.transpose(0, 2, 3, 1)
    return train_imgs


def recompone_overlap(preds, config, img_h, img_w):
    """
    将patch拼成原始图片
    :param preds: patch块
    :param config: 配置文件
    :param img_h:  原始图片 height
    :param img_w:  原始图片 width
    :return:  拼接成的图片
    """
    assert (len(preds.shape) == 4)  # 4D arrays

    patch_h = config.patch_height
    patch_w = config.patch_width
    N_patches_h = (img_h-patch_h)//config.stride_height+1
    N_patches_w = (img_w-patch_w)//config.stride_width+1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " + str(N_patches_h))
    print("N_patches_w: " + str(N_patches_w))
    print("N_patches_img: " + str(N_patches_img))
    #assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0]//N_patches_img
    print("According to the dimension inserted, there are " + str(N_full_imgs) +
          " full images (of " + str(img_h)+"x" + str(img_w) + " each)")
    # itialize to zero mega array with sum of Probabilities
    full_prob = np.zeros((N_full_imgs, img_h, img_w, preds.shape[3]))
    full_sum = np.zeros((N_full_imgs, img_h, img_w, preds.shape[3]))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h-patch_h)//config.stride_height+1):
            for w in range((img_w-patch_w)//config.stride_width+1):
                full_prob[i, h*config.stride_height:(h*config.stride_height)+patch_h, w*config.stride_width:(
                    w*config.stride_width)+patch_w, :] += preds[k]
                full_sum[i, h*config.stride_height:(h*config.stride_height)+patch_h,
                         w*config.stride_width:(w*config.stride_width)+patch_w, :] += 1
                k += 1

    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob/full_sum
    print('using avg')
    return final_avg


# ============================================================
# ========= PRE PROCESSING FUNCTIONS ========================#
# ============================================================

# ==== histogram equalization
def histo_equalized(imgs):
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized = cv2.equalizeHist(np.array(imgs, dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
            np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs

# https://github.com/DeepTrial/Optic-Disc-Unet/blob/master/perception/models/AttnUnet.py


# from keras.layers.core import Layer, InputSpec
add = Add
multiply = Multiply


def _MiniUnet(input, shape):
    x1 = Conv2D(shape, (3, 3), strides=(1, 1),
                padding="same", activation="relu")(input)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)

    x2 = Conv2D(shape*2, (3, 3), strides=(1, 1),
                padding="same", activation="relu")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)

    x3 = Conv2D(shape * 3, (3, 3), strides=(1, 1),
                padding="same", activation="relu")(pool2)

    x = concatenate([UpSampling2D(size=(2, 2))(x3), x2], axis=3)
    x = Conv2D(shape*2, (3, 3), strides=(1, 1),
               padding="same", activation="relu")(x)

    x = concatenate([UpSampling2D(size=(2, 2))(x), x1], axis=3)
    x = Conv2D(shape, (3, 3), strides=(1, 1),
               padding="same", activation="sigmoid")(x)
    return x


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(
        x, repnum, axis=3), arguments={'repnum': rep})(tensor)
    return my_repeat


def AttnGatingBlock(x, g, inter_shape):
    shape_x = K.int_shape(x)  # 32
    shape_g = K.int_shape(g)  # 16

    theta_x = Conv2D(inter_shape, (2, 2), strides=(
        2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3), strides=(
        shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), padding='same')(phi_g)  # 16

    concat_xg = Add()([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(
        shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    # my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
    # upsample_psi=my_repeat([upsample_psi])
    upsample_psi = expend_as(upsample_psi, shape_x[3])

    y = Multiply()([upsample_psi, x])

    # print(K.is_keras_tensor(upsample_psi))

    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn


def UnetGatingSignal(input, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UnetConv2D(input, outdim, is_batchnorm=False):
    shape = K.int_shape(input)
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    if is_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def UnetConv2DPro(input, outdim):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    attn_shortcut = _MiniUnet(input, outdim)

    merge = Multiply()([attn_shortcut, x])
    result = Add()([merge, x])
    return result


def build_model(config):
    inputs = Input((config['patch_height'], config['patch_width'], 3))
    conv = Conv2D(16, (3, 3), padding='same')(inputs)  # 'valid'
    conv = LeakyReLU(alpha=0.3)(conv)

    conv1 = UnetConv2D(conv, 32, is_batchnorm=True)  # 32 128
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = UnetConv2D(pool1, 32, is_batchnorm=True)  # 32 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 64, is_batchnorm=True)  # 64 32
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True)  # 64 16
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    center = UnetConv2D(pool4, 128, is_batchnorm=True)  # 128 8

    gating = UnetGatingSignal(center, is_batchnorm=True)
    attn_1 = AttnGatingBlock(conv4, gating, 128)
    up1 = Concatenate(axis=3)([Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same', activation="relu")(center), attn_1])

    gating = UnetGatingSignal(up1, is_batchnorm=True)
    attn_2 = AttnGatingBlock(conv3, gating, 64)
    up2 = Concatenate(axis=3)([Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same', activation="relu")(up1), attn_2])

    gating = UnetGatingSignal(up2, is_batchnorm=True)
    attn_3 = AttnGatingBlock(conv2, gating, 32)
    up3 = Concatenate(axis=3)([Conv2DTranspose(32, (3, 3), strides=(
        2, 2), padding='same', activation="relu")(up2), attn_3])

    up4 = Concatenate(axis=3)([Conv2DTranspose(32, (3, 3), strides=(
        2, 2), padding='same', activation="relu")(up3), conv1])

    conv8 = Conv2D(config['seg_num'] + 1, (1, 1),
                   activation='relu', padding='same')(up4)
    conv8 = Reshape(
        (config['patch_height'] * config['patch_width'], config['seg_num']+1))(conv8)
    ############
    act = Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=act)
    return model


class DataGenerator():
    """
    load image (Generator)
    """

    def __init__(self, train_gen, val_gen, config):
        self.tgen = train_gen
        self.vgen = val_gen
        self.config = config
    # self.train_img=img_process(Xt)
    # self.train_gt=yt/1.
    # self.val_img=img_process(Xv)
    # self.val_gt=yv/1.

    def _CenterSampler(self, attnlist, class_weight, Nimgs):
        """
        围绕目标区域采样
        :param attnlist:  目标区域坐标
        :return: 采样的坐标
        """
        class_weight = class_weight / np.sum(class_weight)
        p = random.uniform(0, 1)
        psum = 0
        for i in range(class_weight.shape[0]):
            psum = psum + class_weight[i]
            if p < psum:
                label = i
                break
        if label == class_weight.shape[0] - 1:
            i_center = random.randint(0, Nimgs - 1)
            x_center = random.randint(
                0 + int(self.config['patch_width'] / 2), self.config['width'] - int(self.config['patch_width'] / 2))
            # print "x_center " +str(x_center)
            y_center = random.randint(
                0 + int(self.config['patch_height'] / 2), self.config['height'] - int(self.config['patch_height'] / 2))
        else:
            t = attnlist[label]
            cid = random.randint(0, t[0].shape[0] - 1)
            i_center = t[0][cid]
            y_center = t[1][cid] + random.randint(0 - int(
                self.config['patch_width'] / 2), 0 + int(self.config['patch_width'] / 2))
            x_center = t[2][cid] + random.randint(0 - int(
                self.config['patch_width'] / 2), 0 + int(self.config['patch_width'] / 2))

        if y_center < self.config['patch_width'] / 2:
            y_center = self.config['patch_width'] / 2
        elif y_center > self.config['height'] - self.config['patch_width'] / 2:
            y_center = self.config['height'] - self.config['patch_width'] / 2

        if x_center < self.config['patch_width'] / 2:
            x_center = self.config['patch_width'] / 2
        elif x_center > self.config['width'] - self.config['patch_width'] / 2:
            x_center = self.config['width'] - self.config['patch_width'] / 2

        return i_center, x_center, y_center

    def _genDef(self, gen, class_weight):
        """
        图片取块生成器模板
        :param train_imgs: 原始图
        :param train_masks:  原始图groundtruth
        :param attnlist:  目标区域list
        :return:  取出的训练样本
        """
        while 1:
            for t in range(int(self.config['subsample'] * self.config['total_train'] / self.config['batch_size'])):
                X = np.zeros([self.config['batch_size'], self.config['patch_height'],
                             self.config['patch_width'], 3])
                Y = np.zeros([self.config['batch_size'], self.config['patch_height']
                             * self.config['patch_width'], self.config['seg_num'] + 1])
                x,y = next(gen)
                x=img_process(x)
                y=y/1.
                attnlist = [np.where(y[:, 0, :, :] == np.max(y[:, 0, :, :]))]
                for j in range(self.config['batch_size']):
                    [i_center, x_center, y_center] = self._CenterSampler(
                        attnlist, class_weight, self.config['batch_size'])  # print(i_center, x_center, y_center)
                    patch = x[i_center, int(y_center - self.config['patch_height'] / 2):int(y_center + self.config['patch_height'] / 2), int(
                        x_center - self.config['patch_width'] / 2):int(x_center + self.config['patch_width'] / 2), :]
                    patch_mask = y[i_center, int(y_center - self.config['patch_height'] / 2):int(y_center + self.config['patch_height'] / 2), int(
                        x_center - self.config['patch_width'] / 2):int(x_center + self.config['patch_width'] / 2), :]
                    # print(patch.shape, patch_mask.shape)
                    X[j, :, :, :] = patch
                    Y[j, :, :] = genMasks(np.reshape(patch_mask, [
                                          1, self.config['seg_num'], self.config['patch_height'], self.config['patch_width']]), self.config['seg_num'])
                yield (X, Y)

    def train_gen(self):
        """
        训练样本生成器
        """
        class_weight = [1.0, 0.0]
        return self._genDef(self.tgen, class_weight)

    def val_gen(self):
        """
        验证样本生成器
        """
        class_weight = [1.0, 0.0]
        return self._genDef(self.vgen, class_weight)

    def visual_patch(self):
        gen = self.train_gen()
        (X, Y) = next(gen)
        image = []
        mask = []
        print("[INFO] Visualize Image Sample...")
        for index in range(self.config.batch_size):
            image.append(X[index])
            mask.append(np.reshape(Y, [self.config.batch_size, self.config.patch_height,
                        self.config.patch_width, self.config.seg_num+1])[index, :, :, 0])
        if self.config.batch_size % 4 == 0:
            row = self.config.batch_size/4
            col = 4
        else:
            if self.config.batch_size % 5 != 0:
                row = self.config.batch_size // 5+1
            else:
                row = self.config.batch_size // 5
            col = 5
        imagePatch = visualize(image, [row, col])
        maskPatch = visualize(mask, [row, col])
        cv2.imwrite(self.config.checkpoint+"image_patch.jpg", imagePatch)
        cv2.imwrite(self.config.checkpoint +
                    "groundtruth_patch.jpg", maskPatch)


class AttNet:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        self.config = {
            "exp_name": "OpticDisc",
            "epochs": 100,
            # "batch_size": batch_size,
            "patch_height": 128,
            "patch_width": 128,
            "subsample": 200,
            # "total_train": total_train,
            # "total_val": total_val,
            "train_datatype": "jpg",
            "val_datatype": "jpg",
            "test_datatype": "jpg",
            "test_gt_datatype": "jpg",
            "height": self.shape[0],
            "width": self.shape[1],
            "stride_height": 20,
            "stride_width": 20,
            "seg_num": 1,
        }
        self.model = build_model(self.config)

    def summary(self):
        self.model.summary()
    
    def set_config_params(self, batch_size, total_train, total_val):
        self.config['batch_size'] = batch_size
        self.config['total_train'] = total_train
        self.config['total_val'] = total_val
    
    def train(self, train_gen, val_gen, train_steps, val_steps):
        gen = DataGenerator(train_gen, val_gen, self.config)
        callbacks = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10)
        self.model.compile(optimizer='adam', loss=dice_coef_multi_loss if self.n_classes>1 else dice_coef_loss)
        return self.model.fit_generator(gen.train_gen(), steps_per_epoch=self.config['subsample'] * train_steps, epochs=100, validation_data=gen.val_gen(), validation_steps=val_steps*self.config['subsample'], callbacks=callbacks)

    #TODO: add prediction code for attnet (patching)
    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def get_model(self):
        return self.model
