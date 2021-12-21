import glob
import cv2
import os
import pathlib
import numpy as np
from PIL.Image import Image
from keras_preprocessing.image import load_img, img_to_array
from matplotlib import image
from matplotlib.image import imread
from numpy import resize
from pretty_midi import pretty_midi
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.train_generator, self.val_generator = self.create_generators()

    def create_generators(self):
        train_dir = "midi_images\\images"
        label_dir = "midi_images\\labels"
        train_filenames = []
        label_filenames = []
        for subdir, dirs, files in os.walk(train_dir):
            for file in files:
                train_filenames.append(os.path.join(subdir, file))
        for subdir, dirs, files in os.walk(label_dir):
            for file in files:
                label_filenames.append(os.path.join(subdir, file))
        shuffled_train_filenames, shuffled_label_filenames = shuffle(train_filenames, label_filenames)
        X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
            shuffled_train_filenames, shuffled_label_filenames, test_size=0.25, random_state=8)

        return DataGenerator(X_train_filenames, y_train, self.config.trainer.batch_size,
                             self.config.data_loader.seq_size), DataGenerator(X_val_filenames,
                                                                              y_val,
                                                                              self.config.trainer.batch_size,
                                                                              self.config.data_loader.seq_size)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size, seq_size):
        self.seq_size = seq_size
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        return np.expand_dims(np.array(
            [cv2.imread(x, 0).T for x in
             batch_x]) / 255, axis=1), np.squeeze(np.array(
            [cv2.imread(y, 0).T for y in batch_y]) / 255, axis=1)
