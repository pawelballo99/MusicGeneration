import os
import pickle


def save_as_pickle(a, path, filename):
    with open(os.path.join(path, filename), 'wb') as handle:
        pickle.dump(a, handle)
    print("Save " + filename + " successfully.")


def load_pickle(path_to_obj):
    file = open(path_to_obj, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj
