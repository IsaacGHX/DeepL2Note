from tensorflow import keras
import keras as k
from keras import layers
from keras.datasets import mnist, imdb
from keras.preprocessing.image_dataset import image_dataset_from_directory


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorboard
from keras.callbacks import TensorBoard
import random

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("tensorflow_version: "
      , tf.__version__)

tensorboard_callback = TensorBoard(log_dir="E:\TensorBoard_logs", histogram_freq=1)



def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model
