import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from utils.data_utils import *

def down_block(x, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x

def up_block(x, filters, concat, dropout):
    x = Conv2DTranspose(filters, (3, 3), strides = (2, 2), padding = 'same')(x)
    x = concatenate([x, concat])
    x = Dropout(dropout)(x)
    x = down_block(x, filters)
    return x
    
def unet(input_size, n_classes, n_filters = 64, dropout = 0.1):
    inputs = tf.keras.Input(shape=input_size)
    c1 = down_block(inputs, n_filters)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = down_block(p1, n_filters * 2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = down_block(p2, n_filters * 4)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = down_block(p3, n_filters * 8)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = down_block(p4, n_filters * 16)
    
    c6 = up_block(c5, n_filters * 8, c4, dropout)
    c7 = up_block(c6, n_filters * 4, c3, dropout)
    c8 = up_block(c7, n_filters * 2, c2, dropout)
    c9 = up_block(c8, n_filters, c1, dropout)
    
    outputs = Conv2D(n_classes, 1, activation = 'softmax')(c9)

    model = Model(inputs = inputs, outputs = outputs)
    return model

class Unet:
    def __init__(self, shape, n_classes):
        self.shape = shape
        self.n_classes = n_classes
        self.model = unet(self.shape, self.n_classes)

    def summary(self):
        self.model.summary()

    def train(self, train_gen, val_gen, train_steps, val_steps):
        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100, validation_data=val_gen, validation_steps=val_steps, callbacks=callbacks).history

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def get_model(self):
        return self.model