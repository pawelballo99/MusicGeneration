from utils.config import process_config
from model.model import Model
from data_loader.data_loader import DataLoader
from trainer.trainer import ModelTrainer
import sys
import os
import h5py


def main():
    try:
        config = process_config()
        data = DataLoader(config)
        model = Model(config)
        trainer = ModelTrainer(model, data, config)
        trainer.train()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
