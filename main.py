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
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, \
    BatchNormalization, Input, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.layers.pooling import MaxPooling3D, MaxPooling2D

SIZE_DATASET = 25
BATCH_SIZE = 32
FPS = 10
SEQ_SIZE = 70

random.seed(50)


def generate_song(array_song):
    array_song = array_song.T
    new_midi = pretty_midi.PrettyMIDI()
    midi_list = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    for i in range(array_song.shape[1]):
        time = 0
        for j in range(array_song.shape[0]):
            if array_song[j][i] != 0 and j != array_song.shape[0] - 1:
                time += 1
            else:
                if time != 0:
                    print("pitch:" + str(i))
                    print("time:" + str(time) + "\n")
                    note = pretty_midi.Note(
                        velocity=100, pitch=i, start=(j - time) / FPS, end=j / FPS)
                    midi_list.notes.append(note)
                    time = 0
    new_midi.instruments.append(midi_list)
    print('\n------ Writing MIDI ------\n')
    new_midi.write('generated_song.midi')


def download_MAESTRO(dataset):
    data_dir = pathlib.Path('midi/maestro-v3.0.0')
    if not data_dir.exists():
        http_response = urlopen("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0"
                                "-midi.zip")
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path='midi/')
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    piano_rolls = []
    random.shuffle(filenames)
    for file in filenames[:dataset]:
        mid = pretty_midi.PrettyMIDI(file)
        piano_rolls.append(np.transpose(mid.get_piano_roll(fs=FPS)).astype(bool).astype(int))
        print('Number of songs:', len(piano_rolls))
    return piano_rolls


def get_train_and_target(piano_rolls):
    x_train = []
    y_target = []
    for sng in piano_rolls:
        for i in range(sng.shape[0] - SEQ_SIZE):
            x_train.append(sng[i:SEQ_SIZE + i])
            y_target.append(sng[SEQ_SIZE + i].T)
    x_train, y_target = x_train[:(len(x_train) - len(x_train) % BATCH_SIZE)], y_target[:(
            len(y_target) - len(y_target) % BATCH_SIZE)]
    x_train, y_target = sklearn.utils.shuffle(x_train, y_target)
    return np.expand_dims(np.asarray(x_train), axis=[2, 4]), np.asarray(y_target)


def get_model():
    model = Sequential()
    model.add(Input(shape=(SEQ_SIZE, 1, 128, 1)))
    model.add(ConvLSTM2D(filters=128, kernel_size=(1, 3), return_sequences=True))
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), return_sequences=False))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='sigmoid'))

    return model


if __name__ == "__main__":
    x = download_MAESTRO(SIZE_DATASET)
    train, target = get_train_and_target(x)
    print('\n------ Finished preprocessing ------\n')

    mlb = MultiLabelBinarizer()
    mlb.fit(target)

    model = get_model()

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.03), metrics=['accuracy'], )

    model.summary()

    print('\n------ Model created ------\n')

    if path.isfile("best_model.hdf5"):
        model.load_weights("best_model.hdf5")
        print('\n------ Weights loaded ------\n')

    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='accuracy', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    print('\n------ Start training ------\n')
    model.fit(train, target, batch_size=BATCH_SIZE, epochs=64, verbose=1, callbacks=[checkpoint])
