import keras
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_utils import *


if __name__ == "__main__":
    _, x_val, x_test, _, _, _, _, z_val, z_test = read_data("data128")
    z_val = encode(z_val)
    z_test = encode(z_test)

    UNet_model = keras.models.load_model("UNet_model.h5")
    CNN_model = keras.models.load_model("CNN_model.h5")

    pred_val_f = UNet_model.predict(x_val)
    pred_val = CNN_model.predict([x_val[...,0:1],pred_val_f])

    accuracy_val = metrics.accuracy_score(z_val, np.argmax(pred_val, axis=1))

    pred_test_f = UNet_model.predict(x_test)
    pred_test = CNN_model.predict([x_test[...,0:1],pred_test_f])
    accuracy_test = metrics.accuracy_score(z_test, np.argmax(pred_test, axis=1))

    conf_mat_val = confusion_matrix(z_val, np.argmax(pred_val, axis=1))
    conf_mat_test = confusion_matrix(z_test, np.argmax(pred_test, axis=1))

    print(f"Validation accuracy of the saved model: {accuracy_val}")
    print(f"Test accuracy of the saved model: {accuracy_test}")

    print("Confusion matrix (val):")
    print(conf_mat_val)
    ConfusionMatrixDisplay.from_predictions(z_val, np.argmax(pred_val, axis=1))

    print("Confusion matrix (test):")
    print(conf_mat_test)
    ConfusionMatrixDisplay.from_predictions(z_test, np.argmax(pred_test, axis=1))

    matrix = metrics.confusion_matrix(z_test, np.argmax(pred_test, axis=1))

    print("Test report")
    print(classification_report(z_test, np.argmax(pred_test, axis=1), digits=6))
    print("Test accuracy:", matrix.diagonal()/matrix.sum(axis=1))
