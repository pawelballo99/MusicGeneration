import os
import random
from os import path
from urllib.request import urlopen
from io import BytesIO
import pretty_midi
from zipfile import ZipFile
import glob
import numpy as np
import pathlib
from time import time
from PIL import Image as im
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow import clip_by_value
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, \
    BatchNormalization, Input, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.backend import _to_tensor
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, LSTM
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.layers.pooling import MaxPooling3D, MaxPooling2D
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits
from tensorflow.python.ops.numpy_ops import log

SIZE_DATASET = 50
FPS = 10
SEQ_SIZE = 80

random.seed(50)


def download_MAESTRO(dataset):
    data_dir = pathlib.Path('midi/maestro-v3.0.0')
    if not data_dir.exists():
        http_response = urlopen("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0"
                                "-midi.zip")
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path='midi/')
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    piano_rolls = []
    for file in filenames[:dataset]:
        mid = pretty_midi.PrettyMIDI(file)
        frame = np.transpose(mid.get_piano_roll(fs=FPS)).astype(bool).astype(float)
        piano_rolls.append(frame)
        print('Number of songs:', len(piano_rolls))
    return piano_rolls


def get_train_and_target(piano_rolls):
    x_train = []
    y_target = []
    piano_rolls = np.vstack(piano_rolls)

    data = im.fromarray(piano_rolls[:1024][:]*255)
    if data.mode != 'L':
        data = data.convert('L')
    data.save('data_vizualization.png')

    for i in range(piano_rolls.shape[0] - SEQ_SIZE):
        x_train.append(piano_rolls[i:SEQ_SIZE + i, :])
        y_target.append(piano_rolls[SEQ_SIZE + i, :])
    return np.asarray(x_train), np.asarray(y_target)


def bc(target, output, from_logits=False):
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = clip_by_value(output, _epsilon, 1 - _epsilon)
        output = log(output / (1 - output))

    return sigmoid_cross_entropy_with_logits(labels=target,
                                             logits=output)


def get_model():
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=135, kernel_size=2, activation='relu'),
                              input_shape=(None, SEQ_SIZE, 128)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
    model.add(TimeDistributed(Conv1D(filters=135, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
    model.add(TimeDistributed(Conv1D(filters=135, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(128))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='sigmoid'))
    model.compile(optimizer='RMSprop', loss=bc, metrics=['accuracy'], )
    model.summary()

    return model


if __name__ == "__main__":
    x = download_MAESTRO(SIZE_DATASET)
    train, target = get_train_and_target(x)
    train = np.expand_dims(train, axis=1)

    print('\n------ Finished preprocessing ------\n')
    mlb = MultiLabelBinarizer()
    mlb.fit(target)

    model = get_model()

    print('\n------ Model created ------\n')
    if path.isfile("best_model.hdf5"):
        try:
            model.load_weights("best_model.hdf5")
            print('\n------ Weights loaded ------\n')
        except:
            os.remove("best_model.hdf5")
            pass

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    print('\n------ Start training ------\n')
    model.fit(train, target, epochs=150, verbose=1, callbacks=[checkpoint])
