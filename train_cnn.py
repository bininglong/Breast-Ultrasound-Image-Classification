import time
import keras
import numpy as np
import tensorflow as tf

from datetime import datetime
from keras.applications import DenseNet201

from data_utils import *


hyper_params = {"dr": 0.05, "lr": 0.001, "num_layers": 3, "num_units": 64, "num": 20}


def CNN(num_layers, num_units, dr_rate):
    img_input = keras.layers.Input(shape=(128, 128, 1))
    mask_input = keras.layers.Input(shape=(128, 128, 1))
    img_conc = keras.layers.Concatenate()([img_input, mask_input, mask_input])    
    pretrained_model = DenseNet201(weights="imagenet", include_top=False, input_tensor=img_conc)

    for layer in pretrained_model.layers:
        layer.trainable = False

    x = tf.keras.layers.Flatten()(pretrained_model.output)

    for _ in range(num_layers):
        x = tf.keras.layers.Dense(num_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dr_rate)(x)
    predictions = tf.keras.layers.Dense(3, activation="softmax")(x)

    CNN_model = tf.keras.Model(inputs=[img_input, mask_input], outputs=predictions)

    return CNN_model


if __name__ == "__main__":
    x_train, x_val, x_test, _, _, _, z_train, z_val, z_test = read_data("data128")
    z_train_onehot = onehot_encode(z_train)
    z_val_onehot = onehot_encode(z_val)
    z_test_onehot = onehot_encode(z_test)

    UNet_model = tf.keras.models.load_model("UNet_model.h5")

    CNN_model = CNN(num_layers=hyper_params["num_layers"], num_units=hyper_params["num_units"], dr_rate=hyper_params["dr"])
    optimizer = keras.optimizers.Adam(learning_rate=hyper_params["lr"])
    CNN_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    with open("training_output_cnn.txt", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Test run at {current_time}\n", file=f)

        masks_train_pred = UNet_model.predict(x_train)
        masks_val_pred = UNet_model.predict(x_val)

        start_time = time.time()
        results = CNN_model.fit([x_train[..., 0:1], masks_train_pred], z_train_onehot, validation_data=([x_val[..., 0:1], masks_val_pred], z_val_onehot), epochs=hyper_params["num"])
        training_time = time.time() - start_time

        start_time = time.time()
        predictions = CNN_model.predict([x_test[..., 0:1][0:1], UNet_model.predict(x_test[0:1])])
        inference_time = time.time() - start_time

        _, accuracy_val = CNN_model.evaluate([x_val[..., 0:1], UNet_model.predict(x_val)], z_val_onehot, verbose=0)
        _, accuracy_test = CNN_model.evaluate([x_test[..., 0:1], UNet_model.predict(x_test)], z_test_onehot, verbose=0)
        print("Validation accuracy:", accuracy_val)
        print("Test accuracy:", accuracy_test)
        print("Training time", training_time)
        print("Inference time", inference_time)

        print(f"Validation accuracy: {accuracy_val}", file=f)
        print(f"Test accuracy: {accuracy_test}", file=f)
        print(f"Training time: {training_time}", file=f)
        print(f"Inference time: {inference_time}", file=f)

        CNN_model.save("CNN_model.h5")
