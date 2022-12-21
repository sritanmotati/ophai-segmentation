from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from utils.data_utils import *
from utils.losses import *

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def resnetunet(input_shape, n_classes=3):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    # s1.trainable = False
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    # s2.trainable = False
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    # s3.trainable = False
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)
    # s4.trainable = False

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)
    # b1.trainable = False

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    """ Output """
    outputs = Conv2D(n_classes, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model

class ResNetUnet:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        self.model = resnetunet(self.shape, self.n_classes)

    def summary(self):
        self.model.summary()
    
    def train(self, train_gen, val_gen, train_steps, val_steps):
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.compile(optimizer='adam', loss=dice_coef_multi_loss)
        return self.model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100, validation_data=val_gen, validation_steps=val_steps, callbacks=callbacks).history

    def predict(self, x):
        return self.model.predict(x, verbose=0)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)

    def get_model(self):
        return self.model