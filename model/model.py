import tensorflow as tf
from tensorflow import clip_by_value
from tensorflow.python.keras.backend import _to_tensor, log, epsilon
from tensorflow.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization, \
    Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits

import tensorflow.python.keras.backend as K


def bc(target, output, from_logits=False):
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = clip_by_value(output, _epsilon, 1 - _epsilon)
        output = log(output / (1 - output))

    return sigmoid_cross_entropy_with_logits(labels=target,
                                             logits=output)


def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


class Model:
    def __init__(self, config, generation=False):
        self.generation = generation
        self.model = Sequential()
        self.config = config
        self.build_model()

    def save(self):
        if not self.generation:
            if self.model is None:
                raise Exception("You have to build the model first.")
            self.model.save_weights(self.config.checkpoint_dir)

    def load(self):
        if self.model is None:
            raise Exception("You have to build the model first.")
        self.model.load_weights("weights\\save_weights.h5")

    def build_model(self):
        self.model.add(TimeDistributed(Conv1D(filters=140, kernel_size=2, activation='relu'),
                                       input_shape=(None, self.config.data_loader.seq_size, 128)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model.add(TimeDistributed(Conv1D(filters=140, kernel_size=2, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model.add(TimeDistributed(Conv1D(filters=140, kernel_size=2, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(256, return_sequences=True))
        self.model.add(LSTM(128))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='sigmoid', dtype='float32'))
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate)
        self.model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(opt) if len(
            tf.config.list_physical_devices('GPU')) > 0 else 'Adam', loss=bc,
                           metrics=['accuracy', f1_metric])
        if not self.generation:
            self.model.summary()
        else: self.load()
