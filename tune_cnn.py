import keras
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.applications import DenseNet201

from data_utils import *


def CNN(num_layers, num_units, dr):
    img_input = keras.layers.Input(shape=(128, 128, 1))
    mask_input = keras.layers.Input(shape=(128, 128, 1))
    img_conc = keras.layers.Concatenate()([img_input, mask_input, mask_input])    
    pretrained_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=img_conc)

    for layer in pretrained_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(pretrained_model.output)

    for _ in range(num_layers):
        x = tf.keras.layers.Dense(num_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dr)(x)
    predictions = tf.keras.layers.Dense(3, activation="softmax")(x)

    CNN_model = tf.keras.Model(inputs=[img_input, mask_input], outputs=predictions)

    return CNN_model

def tune(x_train, x_val, z_train_onehot, z_val_onehot):
    num_loop = 5

    # Number of layers
    numbers_layers = [3, 4, 5, 6, 7, 8]

    # Number of units
    numbers_units = [16, 32, 64, 128, 256]

    # Dropout rate
    dropout_rates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Learning rate
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]

    best_configuration = dict(val_acc=float(0), num_layers=numbers_layers[0], num_units=numbers_units[2], dr=dropout_rates[0], lr=learning_rates[2], epoch_num=10)

    with open("cnn_tuning_output.txt", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Run at {current_time}\n", file=f)

        UNet_model = tf.keras.models.load_model("UNet_model.h5")

        for i in range(num_loop):
            print(f"------------------------ Loop #{i+1} ------------------------\n", file=f)

            # Test number of layers
            for num_layers in numbers_layers:
                CNN_model = CNN(num_layers=num_layers, num_units=best_configuration["num_units"], dr=best_configuration["dr"])
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
                CNN_results = CNN_model.fit([x_train[..., 0:1], UNet_model.predict(x_train)], z_train_onehot, validation_data=([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot), epochs=best_configuration["epoch_num"], verbose=0)
                _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
                print(f"val_acc={val_acc} (num_layers={num_layers})")

                if val_acc > best_configuration["val_acc"]:
                    print("*** best_configuration updated")
                    best_configuration["val_acc"] = val_acc
                    best_configuration["loss"] = CNN_results.history["loss"]
                    best_configuration["val_loss"] = CNN_results.history["val_loss"]
                    best_configuration["num_layers"] = num_layers

                    CNN_model.save("CNN_model.h5")

            print(f"Test number of layers - best configuration: {best_configuration}\n", file=f)

            # Test number of units
            for num_units in numbers_units:
                CNN_model = CNN(num_layers=best_configuration["num_layers"], num_units=num_units, dr=best_configuration["dr"])
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
                CNN_results = CNN_model.fit([x_train[..., 0:1], UNet_model.predict(x_train)], z_train_onehot, validation_data=([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot), epochs=best_configuration["epoch_num"], verbose=0)
                _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
                print(f"val_acc={val_acc} (num_units={num_units})")

                if val_acc > best_configuration["val_acc"]:
                    print("*** best_configuration updated")
                    best_configuration["val_acc"] = val_acc
                    best_configuration["loss"] = CNN_results.history["loss"]
                    best_configuration["val_loss"] = CNN_results.history["val_loss"]
                    best_configuration["num_units"] = num_units

                    CNN_model.save("CNN_model.h5")

            print(f"Test number of units - best configuration: {best_configuration}\n", file=f)

            # Test dropout rate
            for dr in dropout_rates:
                CNN_model = CNN(num_layers=best_configuration["num_layers"], num_units=best_configuration["num_units"], dr=dr)
                optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
                CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
                CNN_results = CNN_model.fit([x_train[..., 0:1], UNet_model.predict(x_train)], z_train_onehot, validation_data=([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot), epochs=best_configuration["epoch_num"], verbose=0)
                _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
                print(f"val_acc={val_acc} (dr={dr})")

                if val_acc > best_configuration["val_acc"]:
                    print("*** best_configuration updated")
                    best_configuration["val_acc"] = val_acc
                    best_configuration["loss"] = CNN_results.history["loss"]
                    best_configuration["val_loss"] = CNN_results.history["val_loss"]
                    best_configuration["dr"] = dr

                    CNN_model.save("CNN_model.h5")

            print(f"Test dropout rate - best configuration: {best_configuration}\n", file=f)

            # Test learning rate
            for lr in learning_rates:
                CNN_model = CNN(num_layers=best_configuration["num_layers"], num_units=best_configuration["num_units"], dr=best_configuration["dr"])
                optimizer = keras.optimizers.Adam(learning_rate=lr)
                CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
                CNN_results = CNN_model.fit([x_train[..., 0:1], UNet_model.predict(x_train)], z_train_onehot, validation_data=([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot), epochs=best_configuration["epoch_num"], verbose=0)
                _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
                print(f"val_acc={val_acc} (lr={lr})")

                if val_acc > best_configuration["val_acc"]:
                    print("*** best_configuration updated")
                    best_configuration["val_acc"] = val_acc
                    best_configuration["loss"] = CNN_results.history["loss"]
                    best_configuration["val_loss"] = CNN_results.history["val_loss"]
                    best_configuration["lr"] = lr

                    CNN_model.save("CNN_model.h5")

            print(f"Test learning rate - best configuration: {best_configuration}\n", file=f)

            # Test number of epochs
            CNN_model = CNN(num_layers=best_configuration["num_layers"], num_units=best_configuration["num_units"], dr=best_configuration["dr"])
            optimizer = keras.optimizers.Adam(learning_rate=best_configuration["lr"])
            CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            for epoch_num in range(1, 101):
                CNN_results = CNN_model.fit([x_train[..., 0:1], UNet_model.predict(x_train)], z_train_onehot, validation_data=([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot), epochs=1, verbose=0)
                _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
                print(f"val_acc={val_acc} (epoch_num={epoch_num})")

                if val_acc > best_configuration["val_acc"]:
                    print("*** best_configuration updated")
                    best_configuration["val_acc"] = val_acc
                    best_configuration["loss"] = CNN_results.history["loss"]
                    best_configuration["val_loss"] = CNN_results.history["val_loss"]
                    best_configuration["epoch_num"] = epoch_num

                    CNN_model.save("CNN_model.h5")

            print(f"Test number of epochs - best configuration: {best_configuration}\n", file=f)

    return best_configuration


if __name__ == "__main__":
    x_train, x_val, x_test, _, _, _, z_train, z_val, z_test = read_data("data128")
    z_train_onehot = onehot_encode(z_train)
    z_val_onehot = onehot_encode(z_val)
    z_test_onehot = onehot_encode(z_test)

    best_configuration = tune(x_train, x_val, z_train_onehot, z_val_onehot)
    print(f"Best configuration: {best_configuration}")

    UNet_model = tf.keras.models.load_model("UNet_model.h5")
    CNN_model = tf.keras.models.load_model("CNN_model.h5")

    _, val_acc = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
    _, test_acc = CNN_model.evaluate([x_test[..., 0:1], UNet_model.predict(x_test)], z_test_onehot, verbose=0)

    with open("cnn_tuning_output.txt", "a") as f:
        print(f"val_acc={val_acc}, test_acc={test_acc}\n", file=f)
