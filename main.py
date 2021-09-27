import pretty_midi
import os

def monophonic(stream):
    try:
        length = len(instrument.partitionByInstrument(stream).parts)
    except:
        length = 0
    return length == 1

if __name__=="__main__":
    data_dir = os.path.abspath(os.getcwd())+ '\\midi\\'
    songList = os.listdir(data_dir)
    originalScores = []
    for song in songList:
        midi_pretty_format = pretty_midi.PrettyMIDI(os.getcwd()+ '\\midi\\' + song)
        piano_midi = midi_pretty_format.instruments[0] # Get the piano channels
        piano_roll = piano_midi.get_piano_roll(fs=5)
        #score = converter.parse(data_dir+song)
        """ if(monophonic(score)):
            originalScores.append(score.chordify())
            print(str(len(originalScores))) """

    """ originalChords = [[] for _ in originalScores]
    originalDurations = [[] for _ in originalScores]
    originalKeys = []

    for i, song in enumerate(originalScores):
        originalKeys.append(str(song.analyze('key')))
        for element in song:
            originalChords[i].append('.'.join(str(n) for n in element.pitches))
            originalDurations[i].append(element.duration.quarterLength)
        print(str(i)) """

    print('kkk')