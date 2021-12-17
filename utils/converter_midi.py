import os
import sys
import music21
from PIL import Image as im
import pretty_midi
from tqdm import tqdm

from config import process_config

config = process_config()

majors = dict(
    [("A-", 4), ("A", 3), ("B-", 2), ("B", 1), ("C", 0), ("D-", -1), ("D", -2), ("E-", -3), ("E", -4), ("F", -5),
     ("G-", 6), ("G", 5)])
minors = dict(
    [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4),
     ("G-", 3), ("G", 2)])


def midi_to_png(file):
    score = music21.converter.parse(file)
    key1 = score.analyze('key')
    if not (key1.tonic.name == "C" and key1.mode == "major") and not (key1.tonic.name == "A" and key1.mode == "minor"):
        try:
            if key1.mode == "major":
                halfSteps = majors[key1.tonic.name]
            else:
                halfSteps = minors[key1.tonic.name]
            os.remove(file)
            score.transpose(halfSteps).write('midi', file)
            mid = pretty_midi.PrettyMIDI(os.path.join(r, file))
            frame = (mid.get_piano_roll(fs=8)).astype(bool).astype(float)
            save_png(frame)
        except:
            print(str(file) + " was removed")
            if os.path.exists(file):
                os.remove(file)
    else:
        mid = pretty_midi.PrettyMIDI(os.path.join(r, file))
        frame = (mid.get_piano_roll(fs=config.data_loader.fps)).astype(bool).astype(float)
        save_png(frame)


def save_png(frame):
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
        midi_to_png(filenames[i])
