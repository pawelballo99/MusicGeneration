import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from time import time

mixed_precision.set_global_policy('mixed_float16')


class ModelTrainer:
    def __init__(self, model, data, config):
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
        self.callbacks = [
            TensorBoard(log_dir="logs/{}".format(time())),
            ReduceLROnPlateau(verbose=1),
            ModelCheckpoint(
                filepath=self.config.callbacks.checkpoint_dir + "\\" + self.config.exp.name + "_{epoch:02d}_{accuracy:.4f}_{val_accuracy:.4f}.h5",
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
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
        print(self.model.summary())
