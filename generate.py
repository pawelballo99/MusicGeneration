from main import *
import numpy as np

length_song = 70  # in seconds

if __name__ == "__main__":
    start_frame = download_MAESTRO(1)[0][:SEQ_SIZE]

    print('\n------ Creating model ------\n')
    model = get_model()

    if path.isfile("best_model.hdf5"):
        print('\n------ Loading weights ------\n')
        model.load_weights("best_model.hdf5")

        print('\n------ Generating MIDI ------\n')
        for i in range(length_song * FPS):
            y = model.predict(np.expand_dims(start_frame[i:SEQ_SIZE + i, :], axis=[0, 2, 4]))
            for i in range(y.shape[1]):
                if y[0, i] >= 0.5 * max(y[0, :]):
                    y[0, i] = 1
                else:
                    y[0, i] = 0
            start_frame = np.concatenate((start_frame, y), axis=0)

        generate_song(start_frame)
    else:
        print('\nTrain model first!!!\n')
    print('F')
