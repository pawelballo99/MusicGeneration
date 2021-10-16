from main import *
import numpy as np

length_song = 100


def generate_song(array_song):
    array_song = array_song.T
    new_midi = pretty_midi.PrettyMIDI()
    midi_list = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))

    data = im.fromarray(array_song * 255)
    if data.mode != 'L':
        data = data.convert('L')
    data.save('generated_song.png')

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
                        velocity=100, pitch=i, start=(j - tm) / FPS, end=j / FPS)
                    midi_list.notes.append(note)
                    tm = 0
    new_midi.instruments.append(midi_list)
    print('\n------ Saving MIDI ------\n')
    new_midi.write('generated_song.midi')


if __name__ == "__main__":
    start_frame = download_MAESTRO(15)[12][:SEQ_SIZE]

    print('\n------ Creating model ------\n')
    model = get_model()

    if path.isfile("best_model.hdf5"):
        print('\n------ Loading weights ------\n')
        model.load_weights("best_model.hdf5")

        print('\n------ Generating MIDI ------\n')
        vanish_proof = 0.2
        vanish_inc = 1.001
        for i in range(length_song * FPS):
            y = model.predict(np.expand_dims(start_frame[i:SEQ_SIZE + i, :], axis=[0, 1])) + 0.12
            for i in range(y.shape[1]):
                if y[0, i] >= 0.5:
                    y[0, i] = 1
                else:
                    y[0, i] = 0
            start_frame = np.concatenate((start_frame, y), axis=0)

        generate_song(start_frame)
    else:
        print('\nTrain model first!!!\n')
    print('F')
