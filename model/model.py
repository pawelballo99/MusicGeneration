import tensorflow as tf
from tensorflow import clip_by_value
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.backend import _to_tensor, log, epsilon
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import TimeDistributed, Conv1D, MaxPooling1D, Flatten, LSTM, BatchNormalization, \
    Dense, Dropout, Bidirectional
from tensorflow.python.keras.models import Sequential
from tensorflow.python.ops.nn_impl import sigmoid_cross_entropy_with_logits


class Model:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def save(self):
        if self.model is None:
            raise Exception("You have to build the model first.")
        self.model.save_weights(self.config.checkpoint_dir)

    def load(self):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(self.config.checkpoint_dir))
        self.model.load_weights(self.config.checkpoint_dir)

    def bc(self, target, output, from_logits=False):
        if not from_logits:
            _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
            output = clip_by_value(output, _epsilon, 1 - _epsilon)
            output = log(output / (1 - output))

        return sigmoid_cross_entropy_with_logits(labels=target,
                                                 logits=output)

    def build_model(self):
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'),
                                       input_shape=(None, self.config.data_loader.seq_size, 128)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
        self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
        self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1, strides=None)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(BatchNormalization())
        self.model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3)))
        self.model.add(Bidirectional(LSTM(256, dropout=0.3)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.model.learning_rate)
        self.model.compile(optimizer=tf.keras.mixed_precision.LossScaleOptimizer(inner_optimizer=opt) if len(tf.config.list_physical_devices('GPU')) > 0 else 'Adam', loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
