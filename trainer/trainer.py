import os

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from time import time
from tensorflow.keras.callbacks import CSVLogger


class ModelTrainer:
    def __init__(self, model, data, config):
        if len(tf.config.list_physical_devices('GPU')) > 0:
            mixed_precision.set_global_policy('mixed_float16')
        self.model = model.model
        self.train_gen, self.val_gen = data.train_generator, data.val_generator
        self.config = config
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        history_path = "experiments\\history"
        if not os.path.exists(history_path):
            os.makedirs(history_path)
        csv_logger = CSVLogger(history_path + "//model_history_log.csv", append=True)
        self.callbacks = [
            csv_logger,
            TensorBoard(log_dir="experiments/logs/{}".format(time())),
            ModelCheckpoint(
                filepath=self.config.callbacks.checkpoint_dir_val_acc + "\\max_val_accuracy.h5",
                monitor="val_accuracy",
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="h5",
            ),
            ModelCheckpoint(
                period=5,
                filepath="experiments\\epoch" + "\\epoch{epoch:08d}.h5",
                save_freq='epoch',
                save_format="h5",
            ),
            ModelCheckpoint(
                filepath=self.config.callbacks.checkpoint_dir_acc + "\\max_accuracy.h5",
                monitor="accuracy",
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="h5",
            ),
            ModelCheckpoint(
                filepath="experiments\\min_loss" + "\\min_loss.h5",
                monitor="loss",
                mode=self.config.callbacks.checkpoint_mode,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="h5",
            )
        ]

    def train(self):
        history = self.model.fit(self.train_gen, epochs=self.config.trainer.num_epochs,
                                 steps_per_epoch=self.train_gen.__len__(),
                                 verbose=self.config.trainer.verbose_training,
                                 validation_data=self.val_gen,
                                 callbacks=self.callbacks)
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
