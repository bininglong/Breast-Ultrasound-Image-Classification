import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_data(shape):
    # Define an object to store image and mask for training
    data_obj = {"img": [], "mask": [], "label": []}

    def PrepareData(data_obj, img_path, shape):
        # 1: Store image names
        # Get all the file names
        fileNames = os.listdir(img_path)

        # Remove any duplicate elements, store unique names
        names = []

        for i in range(len(fileNames)):
            names.append(fileNames[i].split(")")[0])

        names = list(set(names))

        # 2: Load images, append to data_obj
        for i in range(len(names)):
            img = plt.imread(img_path + "/" + names[i] + ").png")
            if "normal" in names[i]:
                mask = np.array(Image.open(img_path + "/" + names[i] + ")_mask.png").convert("L"))
            else:
                mask = plt.imread(img_path + "/" + names[i] + ")_mask.png")

            # Resize the images
            img = cv2.resize(img, (shape, shape)) 
            mask = cv2.resize(mask, (shape, shape))

            data_obj["img"].append(img)
            data_obj["mask"].append(mask)
            
            # Generate ground-truth labels for categories
            if "normal" in names[i]:
                label = "normal"
            elif "benign" in names[i]:
                label = "benign"
            elif "malignant" in names[i]:
                label = "malignant"
            
            data_obj["label"].append(label)

        return data_obj

    data_obj = PrepareData(data_obj, "Dataset_BUSI_with_GT/normal", shape)
    data_obj = PrepareData(data_obj, "Dataset_BUSI_with_GT/malignant", shape)
    data_obj = PrepareData(data_obj, "Dataset_BUSI_with_GT/benign", shape)

    return data_obj

def split_data(data_obj, shape):
    # Group the images by class
    classes = {"benign": [], "malignant": [], "normal": []}
    for img, mask, label in zip(data_obj["img"], data_obj["mask"], data_obj["label"]):
        classes[label].append((img, mask, label))

    # Shuffle the images in each class
    for class_name in classes:
        random.shuffle(classes[class_name])

    x_train, y_train, z_train = [], [], []
    x_val, y_val, z_val = [], [], []
    x_test, y_test, z_test = [], [], []

    # train/val/test - 80/10/10
    # Grab images from each class
    for class_name, images in classes.items():
        num_images = len(images)
        split_idx1 = int(num_images * 0.8)
        split_idx2 = int(num_images * 0.9)

        for i, (img, mask, label) in enumerate(images):
            if i < split_idx1:
                x_train.append(img)
                y_train.append(mask)
                z_train.append(label)
            elif i < split_idx2:
                x_val.append(img)
                y_val.append(mask)
                z_val.append(label)
            else:
                x_test.append(img)
                y_test.append(mask)
                z_test.append(label)

    x_train, y_train, z_train = np.array(x_train), np.array(y_train), np.array(z_train)
    x_val, y_val, z_val = np.array(x_val), np.array(y_val), np.array(z_val)
    x_test, y_test, z_test = np.array(x_test), np.array(y_test), np.array(z_test)

    if not os.path.exists("data"+str(shape)):
        os.makedirs("data"+str(shape))

    # Save the data to disk
    np.save(f"data{shape}/x_train.npy", x_train)
    np.save(f"data{shape}/x_val.npy", x_val)
    np.save(f"data{shape}/x_test.npy", x_test)
    np.save(f"data{shape}/y_train.npy", y_train)
    np.save(f"data{shape}/y_val.npy", y_val)
    np.save(f"data{shape}/y_test.npy", y_test)
    np.save(f"data{shape}/z_train.npy", z_train)
    np.save(f"data{shape}/z_val.npy", z_val)
    np.save(f"data{shape}/z_test.npy", z_test)


if __name__ == "__main__":
    shape = 128
    data_obj = load_data(shape)
    split_data(data_obj, shape)
