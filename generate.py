import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import pretty_midi
from PIL import Image
from model.model import Model
from utils.config import process_config

length_song = 32


def generate_song(array_song, fps):
    new_midi = pretty_midi.PrettyMIDI()
    midi_list = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    for i in range(array_song.shape[0]):
        tm = 0
        for j in range(array_song.shape[1]):
            if array_song[i][j] != 0 and j != array_song.shape[0] - 1:
                tm += 1
            else:
                if tm != 0:
                    note = pretty_midi.Note(
                        velocity=100, pitch=i, start=(j - tm) / fps, end=j / fps)
                    midi_list.notes.append(note)
                    tm = 0
    new_midi.instruments.append(midi_list)
    new_midi.write("generated_song.midi")


def get_image(model, start_frame):
    frame = start_frame
    bias = 0.25
    for i in range(length_song * config.data_loader.fps):
        r = frame[i:config.data_loader.seq_size + i, :]
        y = model.predict(
            np.expand_dims(r, axis=[0, 1])) + bias
        bias *= 1.0017
        for j in range(y.shape[1]):
            if y[0, j] >= 0.5:
                y[0, j] = 1
            else:
                y[0, j] = 0
        frame = np.concatenate((frame, y), axis=0)

    return frame.T


def get_file():
    images_path = os.path.join("example_midi_images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    filenames = []
    for r, d, f in os.walk(images_path):
        for file in f:
            filenames.append(os.path.join(r, file))
    random_image = filenames[random.randint(0, len(filenames) - 1)]
    return np.squeeze(np.array([cv2.imread(random_image, 0).T]) / 255, axis=0)


if __name__ == "__main__":
    print("Configuration loading...")
    config = process_config(generation=True)

    print("Model loading...")
    model = Model(config, generation=True).model

    print("Generating...")
    start_frame = get_file()
    image = get_image(model, start_frame)

    print("Saving image...")
    data = Image.fromarray(image * 255)
    if data.mode != 'L':
        data = data.convert('L')
    data.save('generated_song.png')

    print("Saving midi...")
    generate_song(image, config.data_loader.fps)
    print("Done")
