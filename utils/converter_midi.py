import os
from PIL import Image as im
import pretty_midi
from tqdm import tqdm
from config import process_config

config = process_config(generation=True)


def midi_to_png(file):
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
