import collections
from urllib.request import urlopen
from io import BytesIO
import pretty_midi
from zipfile import ZipFile
import datetime
import glob
import numpy as np
import pathlib
import pandas as pd
import random
import tensorflow as tf
import os

SIZE_DATASET = 10


def midis_to_notes(midi_file: str) -> [pd.DataFrame]:
    dataframes_midis = []
    for file in midi_file[:SIZE_DATASET]:
        pm = pretty_midi.PrettyMIDI(file)
        instrument = pm.instruments[0]
        notes = collections.defaultdict(list)

        sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
        prev_start = sorted_notes[0].start
        for note in sorted_notes:
            start = note.start
            end = note.end
            notes['pitch'].append(pretty_midi.note_number_to_name(note.pitch))
            notes['start'].append(start)
            notes['end'].append(end)
            notes['step'].append(start - prev_start)
            notes['duration'].append(end - start)
            prev_start = start

        dataframes_midis.append(pd.DataFrame({name: np.array(value) for name, value in notes.items()}))
        print(len(dataframes_midis))
    return dataframes_midis


def download_MAESTRO():
    data_dir = pathlib.Path('midi/maestro-v3.0.0')
    if not data_dir.exists():
        http_response = urlopen("https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0"
                                "-midi.zip")
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path='midi/')
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    random.shuffle(filenames)
    print('Number of files:', len(filenames))
    return filenames


if __name__ == "__main__":
    midis = download_MAESTRO()
    x = midis_to_notes(midis)
    print('kkk')
