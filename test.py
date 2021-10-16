import os
import random
from os import path
from urllib.request import urlopen
from io import BytesIO
import pretty_midi
from zipfile import ZipFile
import glob
import music21
import numpy as np
import pathlib

majors = dict(
    [("A-", 4), ("A", 3), ("B-", 2), ("B", 1), ("C", 0), ("D-", -1), ("D", -2), ("E-", -3), ("E", -4), ("F", -5),
     ("G-", 6), ("G", 5)])
minors = dict(
    [("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("D-", -4), ("D", -5), ("E-", 6), ("E", 5), ("F", 4),
     ("G-", 3), ("G", 2)])

data_dir = pathlib.Path('midi/maestro-v3.0.0')
filenames = glob.glob(str(data_dir / '**/*.mid*'))
piano_rolls = []
for file in filenames:
    score = music21.converter.parse(file)
    key1 = score.analyze('key')
    print(key1)
    if not (key1.tonic.name == "C" and key1.mode == "major") and not (key1.tonic.name == "A" and key1.mode == "minor"):
        try:
            if key1.mode == "major":
                halfSteps = majors[key1.tonic.name]

            else:
                halfSteps = minors[key1.tonic.name]
            os.remove(file)
            newscore = score.transpose(halfSteps)
            key1 = newscore.analyze('key')
            print("newkey:" + str(key1))
            newscore.write('midi', file)
        except:
            os.remove(file)
