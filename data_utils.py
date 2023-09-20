import numpy as np


def read_data(data_path):
    x_train = np.load(data_path+"/x_train.npy")
    x_val = np.load(data_path+"/x_val.npy")
    x_test = np.load(data_path+"/x_test.npy")
    y_train = np.load(data_path+"/y_train.npy")
    y_val = np.load(data_path+"/y_val.npy")
    y_test = np.load(data_path+"/y_test.npy")
    z_train = np.load(data_path+"/z_train.npy")
    z_val = np.load(data_path+"/z_val.npy")
    z_test = np.load(data_path+"/z_test.npy")

    return x_train, x_val, x_test, y_train, y_val, y_test, z_train, z_val, z_test

def encode(x):
    x[x=="normal"] = 0
    x[x=="benign"] = 1
    x[x=="malignant"] = 2
    x = x.astype(int)

    return x

def onehot_encode(x):
    x = encode(x)
    x_onehot = np.zeros((x.size, x.max()+1))
    x_onehot[np.arange(x.size),x] = 1

    return x_onehot
