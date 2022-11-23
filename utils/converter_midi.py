import os
import sys
import music21
from PIL import Image as im
import pretty_midi
from tqdm import tqdm

from config import process_config

config = process_config()

majors = dict(
    [("A-", 4), ("G#", 4), ("A", 3), ("A#", 2), ("B-", 2), ("B", 1), ("C", 0), ("C#", -1), ("D-", -1), ("D", -2),
     ("D#", -3), ("E-", -3), ("E", -4), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])
minors = dict(
    [("G#", 1), ("A-", 1), ("A", 0), ("A#", -1), ("B-", -1), ("B", -2), ("C", -3), ("C#", -4), ("D-", -4), ("D", -5),
     ("D#", 6), ("E-", 6), ("E", 5), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])


def midi_to_png(midi_file):
    # score = music21.converter.parse(midi_file)
    # key1 = score.analyze('key')
    # if not (key1.tonic.name == "C" and key1.mode == "major") and not (key1.tonic.name == "A" and key1.mode == "minor"):
    #     try:
    #         if key1.mode == "major":
    #             half_steps = majors[key1.tonic.name]
    #         else:
    #             half_steps = minors[key1.tonic.name]
    #         new_score = score.transpose(half_steps)
    #         os.remove(midi_file)
    #         new_score.write('midi', midi_file)
    #         print(str(midi_file) + " was converted from " + key1.tonic.name + " " + key1.mode)
    #         mid = pretty_midi.PrettyMIDI(os.path.join(r, midi_file))
    #         frame = (mid.get_piano_roll(fs=config.data_loader.fps)).astype(bool).astype(float)
    #         save_png(frame)
    #     except Exception as e:
    #         print(str(midi_file) + " was removed with exeception " + str(e))
    #         if os.path.exists(midi_file):
    #             os.remove(midi_file)
    # else:
    mid = pretty_midi.PrettyMIDI(os.path.join(r, midi_file))
    frame = (mid.get_piano_roll(fs=config.data_loader.fps)).astype(bool).astype(float)
    save_midi_as_png(frame)


def save_midi_as_png(midi_file):
    mid = pretty_midi.PrettyMIDI(os.path.join(r, midi_file))
    frame = (mid.get_piano_roll(fs=config.data_loader.fps)).astype(bool).astype(float)
    for j in range(frame.shape[0] - config.data_loader.seq_size):
        img_x = im.fromarray(frame[:, j:config.data_loader.seq_size + j] * 255)
        img_y = im.fromarray(frame[:, config.data_loader.seq_size + j] * 255)
        if img_x.mode != 'RGB':
            img_x = img_x.convert('RGB')
        if img_y.mode != 'RGB':
            img_y = img_y.convert('RGB')
        x_folder_path = images_path + "\\" + filenames[i].split('\\')[-1].split('.')[0]
        y_folder_path = labels_path + "\\" + filenames[i].split('\\')[-1].split('.')[0]
        if not os.path.exists(x_folder_path):
            os.mkdir(x_folder_path)
        if not os.path.exists(y_folder_path):
            os.mkdir(y_folder_path)
        img_x.save(x_folder_path + "\\" + filenames[i].split('\\')[-1].split('.')[0] + "_" + str(j) + ".png")
        img_y.save(y_folder_path + "\\" + filenames[i].split('\\')[-1].split('.')[0] + "_" + str(j) + ".png")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "dataset")
    images_path = os.path.join(root, "midi_images\\images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    labels_path = os.path.join(root, "midi_images\\labels")
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    filenames = []
    for r, d, f in os.walk(path):
        for file in f:
            filenames.append(os.path.join(r, file))
    for i in tqdm(range(len(filenames))):
        save_midi_as_png(filenames[i])
