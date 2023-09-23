# Breast Ultrasound Image Classification

<!-- This repository accompanies the paper ["*A Two-Stage Neural Network Model for Breast Ultrasound Image Classification*"]. -->

## Dependencies

All the dependencies are packaged in [`environment.yml`](/environment.yml) that can be installed using [conda](https://docs.conda.io/).
```bash
conda env create -f environment.yml     # install dependencies
conda activate buic                     # activate encironment
```

## Data

The *Breast Ultrasound Images* dataset can be downloaded [here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

Before running the experiments, please download the dataset and unzip it to this directory.

## Usage

### Prepare Dataset

Please run the script [`prepare_data.py`](/prepare_data.py). 

This script performs the following tasks: 
1. Loads and resize images to $128 \times 128$ pixels.
2. Randomly shuffles the data.
3. Splits the data into training, validation, and test sets (ratio: $80:10:10$) while preserving class proportions.
4. Saves each prepared set as a NumPy array in folder `data128`.

### Optimize Configuration

Scripts [`tune_unet.py`](/tune_unet.py) and [`tune_cnn.py`](/tune_cnn.py) explore a range of hyperparameters and model structures to identify the optimal settings for the U-Net and CNN model, respectively.

The optimal configurations will be written to the output `.txt` files.

### Training

Scripts [`train_unet.py`](/train_unet.py) and [`train_cnn.py`](/train_cnn.py) train the U-Net and CNN model using the optimal configuration, respectively.

The best models will be saved under the same directory.

### Evaluation

Script [`test.py`](/test.py) evaluates the saved models and prints the confusion matrices and classification report (accuracy, precision, recall and F1-score) on the validation set and the test set.
