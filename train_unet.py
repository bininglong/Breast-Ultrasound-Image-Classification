import time
import keras
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.regularizers import l1, l2

from data_utils import *


hyper_params = {"dr": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], "lr": 0.001, "us": False, "reg": None, "bn": True, "num": 50}


def UNet(f, dr_rates, us=False, reg=None, bn=True):
    inputs = tf.keras.layers.Input((128, 128, 3))

    init_filters = f

    if reg == None:
        kernel_regularizer=None
    elif reg == "l1":
        kernel_regularizer=l1(0.001)
    elif reg == "l2":
        kernel_regularizer=l2(0.001)

    # Contraction path
    c1 = tf.keras.layers.Conv2D(filters=init_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(inputs)
    if bn:
        c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation("relu")(c1)
    c1 = tf.keras.layers.Conv2D(filters=init_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c1)
    if bn:
        c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation("relu")(c1)
    
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dr_rates[0])(p1)

    c2 = tf.keras.layers.Conv2D(filters=init_filters*2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(p1)
    if bn:
        c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation("relu")(c2)
    c2 = tf.keras.layers.Conv2D(filters=init_filters*2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c2)
    if bn:
        c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation("relu")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = tf.keras.layers.Dropout(dr_rates[1])(p2)

    c3 = tf.keras.layers.Conv2D(filters=init_filters*4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(p2)
    if bn:
        c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation("relu")(c3)
    c3 = tf.keras.layers.Conv2D(filters=init_filters*4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c3)
    if bn:
        c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation("relu")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = tf.keras.layers.Dropout(dr_rates[2])(p3)

    c4 = tf.keras.layers.Conv2D(filters=init_filters*8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(p3)
    if bn:
        c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation("relu")(c4)
    c4 = tf.keras.layers.Conv2D(filters=init_filters*8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c4)
    if bn:
        c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation("relu")(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)
    p4 = tf.keras.layers.Dropout(dr_rates[3])(p4)

    c5 = tf.keras.layers.Conv2D(filters=init_filters*16, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(p4)
    if bn:
        c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation("relu")(c5)
    c5 = tf.keras.layers.Conv2D(filters=init_filters*16, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c5)
    if bn:
        c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation("relu")(c5)
    c5 = tf.keras.layers.Dropout(dr_rates[4])(c5)

    # Expansive path
    if us:
        u6 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(c5)
    else:
        u6 = tf.keras.layers.Conv2DTranspose(init_filters*8, (3, 3), strides=(2, 2), padding="same")(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Dropout(dr_rates[5])(u6)
    c6 = tf.keras.layers.Conv2D(filters=init_filters*8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(u6)
    if bn:
        c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation("relu")(c6)
    c6 = tf.keras.layers.Conv2D(filters=init_filters*8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c6)
    if bn:
        c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation("relu")(c6)

    if us:
        u7 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(c6)
    else:
        u7 = tf.keras.layers.Conv2DTranspose(init_filters*4, (3, 3), strides=(2, 2), padding="same")(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Dropout(dr_rates[6])(u7)
    c7 = tf.keras.layers.Conv2D(filters=init_filters*4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(u7)
    if bn:
        c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation("relu")(c7)
    c7 = tf.keras.layers.Conv2D(filters=init_filters*4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c7)
    if bn:
        c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation("relu")(c7)

    if us:
        u8 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(c7)
    else:
        u8 = tf.keras.layers.Conv2DTranspose(init_filters*2, (3, 3), strides=(2, 2), padding="same")(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Dropout(dr_rates[7])(u8)
    c8 = tf.keras.layers.Conv2D(filters=init_filters*2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(u8)
    if bn:
        c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation("relu")(c8)
    c8 = tf.keras.layers.Conv2D(filters=init_filters*2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c8)
    if bn:
        c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation("relu")(c8)

    if us:
        u9 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")(c8)
    else:
        u9 = tf.keras.layers.Conv2DTranspose(init_filters, (3, 3), strides=(2, 2), padding="same")(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Dropout(dr_rates[8])(u9)
    c9 = tf.keras.layers.Conv2D(filters=init_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(u9)
    if bn:
        c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation("relu")(c9)
    c9 = tf.keras.layers.Conv2D(filters=init_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same", kernel_regularizer=kernel_regularizer)(c9)
    if bn:
        c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation("relu")(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

    UNet_model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    UNet_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return UNet_model


if __name__ == "__main__":
    x_train, x_val, _, y_train, y_val, _, _, _, _ = read_data("data128")

    UNet_model = UNet(f=16, dr_rates=hyper_params["dr"], us=hyper_params["us"], reg=hyper_params["reg"], bn=hyper_params["bn"])
    optimizer = keras.optimizers.Adam(learning_rate=hyper_params["lr"])
    UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    with open("training_output_unet.txt", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Test run at {current_time}\n", file=f)

        start_time = time.time()
        results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=hyper_params["num"])
        training_time = time.time() - start_time

        print("Loss", results.history["loss"][0])
        print("Validation loss", results.history["val_loss"][0])

        print(f"Loss: {results.history['loss'][0]}", file=f)
        print(f"Validation loss: {results.history['val_loss'][0]}", file=f)
        print(f"Training time: {training_time}", file=f)

        UNet_model.save("UNet_model.h5")
