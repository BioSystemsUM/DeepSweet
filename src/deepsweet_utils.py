import json
import os
import sys

import pandas
import tensorflow as tf
from Datasets.Datasets import Dataset
from loaders.Loaders import CSVLoader
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import Session
import tensorflow.compat.v1.keras.backend as K


class DeviceUtils:

    @staticmethod
    def gpu_setup(device):

        tf.random.set_seed(123)

        environment_name = sys.executable.split('/')[-3]
        print('Environment:', environment_name)
        os.environ[environment_name] = str(123)
        os.environ['PYTHONHASHSEED'] = str(123)
        os.environ['TF_DETERMINISTIC_OPS'] = 'False'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = device

        G = tf.Graph()
        session = Session(graph=G,
                          config=config)

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[int(device)], 'GPU')
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)

        K.set_session(session)

        import dgl

        import torch as th

        u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])

        g = dgl.graph((u, v))

        cuda_g = g.to('cuda:' + str(device))  # accepts any device objects from backend framework


class IO:

    @staticmethod
    def load_dataset_with_features(dataset_path: str) -> Dataset:
        """
        This operation is exclusive to datasets generated with DeepMol

        :param dataset_path: dataset path string
        :return: Dataset object from DeepMol
        """
        pandas_dset = pandas.read_csv(dataset_path)
        columns = pandas_dset.columns[3:]

        loader = CSVLoader(dataset_path,
                           features_fields=list(columns),
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()

        return dataset

    @staticmethod
    def load_dataset(dataset_path: str) -> Dataset:
        loader = CSVLoader(dataset_path,
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()

        return dataset

    @staticmethod
    def load_json_config(path: str) -> dict:
        features_to_keep = os.path.join(path)
        f = open(features_to_keep, )

        data = json.load(f)

        return data
