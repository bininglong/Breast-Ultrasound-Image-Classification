import keras
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.regularizers import l1, l2

from data_utils import *


def UNet(f, dr_rates, us=False, reg=None, bn=True):
    if reg == None:
        kernel_regularizer=None
    elif reg == "l1":
        kernel_regularizer=l1(0.001)
    elif reg == "l2":
        kernel_regularizer=l2(0.001)

    init_filters = f

    inputs = tf.keras.layers.Input((128, 128, 3))

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

    UNet_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return UNet_model

def tune(x_train, x_val, y_train, y_val):
    num_loop = 5

    # Regularization
    reg_choices = [None, "l1", "l2"]

    # Dropout rate
    same005 = [0.05 for x in range(9)]
    same01 = [0.1 for x in range(9)]
    same02 = [0.2 for x in range(9)]
    diff = [0.1, 0.1, 0.2, 0.2, 0.3, 0.2, 0.2, 0.1, 0.1]
    zero = [0 for x in range(9)]
    dropout_rates = [same005, same01, same02, diff, zero]

    # Upsampling
    us_choices = [False, True]

    # Batch normalization
    bn_choices = [True, False]

    # Learning rate
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    
    best_configuration = dict(val_loss=float("inf"), reg=reg_choices[0], dr=dropout_rates[0], us=us_choices[0], bn=bn_choices[0], lr=learning_rates[2], epoch_num=30)

    with open("unet_tuning_output.txt", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Run at {current_time}\n", file=f)

        for i in range(num_loop):
            print(f"------------------------ Loop #{i+1} ------------------------\n", file=f)

            # Test regularization
            for reg in reg_choices:
                UNet_model = UNet(f=16, dr_rates=best_configuration["dr"], us=best_configuration["us"], reg=reg, bn=best_configuration["bn"])
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_configuration["epoch_num"], verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (reg={reg})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["reg"] = reg

                    UNet_model.save("UNet_model.h5")

            print(f"Test regularization - best configuration: {best_configuration}\n", file=f)

            # Test dropout rate
            for dr in dropout_rates:
                UNet_model = UNet(f=16, dr_rates=dr, us=best_configuration["us"], reg=best_configuration["reg"], bn=best_configuration["bn"])
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_configuration["epoch_num"], verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (dr={dr})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["dr"] = dr

                    UNet_model.save("UNet_model.h5")

            print(f"Test dropout rate - best configuration: {best_configuration}\n", file=f)

            # Test upsampling
            for us in us_choices:
                UNet_model = UNet(f=16, dr_rates=best_configuration["dr"], us=us, reg=best_configuration["reg"], bn=best_configuration["bn"])
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_configuration["epoch_num"], verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (us={us})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["us"] = us

                    UNet_model.save("UNet_model.h5")

            print(f"Test upsampling - best configuration: {best_configuration}\n", file=f)

            # Test batch normalization
            for bn in bn_choices:
                UNet_model = UNet(f=16, dr_rates=best_configuration["dr"], us=best_configuration["us"], reg=best_configuration["reg"], bn=bn)
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_configuration["epoch_num"], verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (bn={bn})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["bn"] = bn

                    UNet_model.save("UNet_model.h5")

            print(f"Test batch normalization - best configuration: {best_configuration}\n", file=f)

            # Test learning rate
            for lr in learning_rates:
                UNet_model = UNet(f=16, dr_rates=best_configuration["dr"], us=best_configuration["us"], reg=best_configuration["reg"], bn=best_configuration["bn"])
                optimizer = keras.optimizers.Adam(learning_rate=lr)
                UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=best_configuration["epoch_num"], verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (lr={lr})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["lr"] = lr

                    UNet_model.save("UNet_model.h5")

            print(f"Test learning rate - best configuration: {best_configuration}\n", file=f)

            # Test number of epochs
            UNet_model = UNet(f=16, dr_rates=best_configuration["dr"], us=best_configuration["us"], reg=best_configuration["reg"], bn=best_configuration["bn"])
            optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
            UNet_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

            for epoch_num in range(1, 101):
                UNet_results = UNet_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, verbose=0)
                val_loss, _ = UNet_model.evaluate(x_val, y_val, verbose=0)
                print(f"val_loss={val_loss} (epoch_num={epoch_num})")

                if val_loss < best_configuration["val_loss"]:
                    print("*** best_configuration updated")
                    best_configuration["val_loss"] = val_loss
                    best_configuration["loss"] = UNet_results.history["loss"]
                    best_configuration["epoch_num"] = epoch_num

                    UNet_model.save("UNet_model.h5")

            print(f"Test number of epochs - best configuration: {best_configuration}\n", file=f)

    return best_configuration


if __name__ == "__main__":
    x_train, x_val, _, y_train, y_val, _, _, _, _ = read_data("data128")

    best_configuration = tune(x_train, x_val, y_train, y_val)
    print(f"Best configuration: {best_configuration}")
