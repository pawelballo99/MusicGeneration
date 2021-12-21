import numpy as np
import os
import pathlib
import pretty_midi
from PIL import Image
from data_loader.data_loader import DataLoader
from model.model import Model
from utils.config import process_config

length_song = 50


def generate_song(array_song, fps):
    array_song = array_song.T
    new_midi = pretty_midi.PrettyMIDI()
    midi_list = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    # data = im.fromarray(array_song * 255)
    # if data.mode != 'L':
    #     data = data.convert('L')
    # data.save('generated_song.png')

    print('\n------ Writing MIDI ------\n')
    for i in range(array_song.shape[0]):
        tm = 0
        for j in range(array_song.shape[1]):
            if array_song[i][j] != 0 and j != array_song.shape[0] - 1:
                tm += 1
            else:
                if tm != 0:
                    print("pitch:" + str(i))
                    print("time:" + str(tm) + "\n")
                    note = pretty_midi.Note(
                        velocity=100, pitch=i, start=(j - tm) / fps, end=j / fps)
                    midi_list.notes.append(note)
                    tm = 0
    new_midi.instruments.append(midi_list)
    dir = "generated_midi\\"
    if not os.path.exists(dir):
        os.mkdir(dir)
    new_midi.write(dir + 'generated_song.midi')
    data = Image.fromarray(array_song * 255)
    if data.mode != 'L':
        data = data.convert('L')
    data.save(dir + 'generated_song.png')


def get_path():
    best_h5 = 0

    dir = "experiments\\gen1\\checkpoint"
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            value = int(file.split("_")[3].split('.')[1])
            if best_h5 < value:
                best_h5 = value
                idx = files.index(file)
    return dir + "\\" + files[idx]


if __name__ == "__main__":
    config = process_config()
    model = Model(config).model
    start_frame = np.random.randint(0, 2, size=(config.data_loader.seq_size, 128))
    #start_frame = np.squeeze(DataLoader(config).train_generator.__getitem__(2)[0][0], axis=0)
    vanish_proof = 0.2
    vanish_inc = 1.0001
    for i in range(length_song * config.data_loader.fps):
        y = model.predict(
            np.expand_dims(start_frame[i:config.data_loader.seq_size + i, :], axis=[0, 1])) + vanish_proof * vanish_inc
        for i in range(y.shape[1]):
            if y[0, i] >= 0.5:
                y[0, i] = 1
            else:
                y[0, i] = 0
        start_frame = np.concatenate((start_frame, y), axis=0)

    generate_song(start_frame, config.data_loader.fps)
