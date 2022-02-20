import json
import os
import sys

import pandas
import tensorflow as tf
from Datasets.Datasets import Dataset
from compoundFeaturization import deepChemFeaturizers
from compoundFeaturization.rdkitDescriptors import TwoDimensionDescriptors
from compoundFeaturization.rdkitFingerprints import RDKFingerprint, MorganFingerprint, AtomPairFingerprint
from loaders.Loaders import CSVLoader
from rdkit.Chem import MolFromSmiles
from scalers.sklearnScalers import MinMaxScaler
from standardizer.CustomStandardizer import CustomStandardizer
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import Session
import tensorflow.compat.v1.keras.backend as K

from generate_features_rnn import RNNFeatureGenerator


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

class PipelineUtils:

    @staticmethod
    def standardize_dataset(dataset):
        standardisation_params = {
            'REMOVE_ISOTOPE': True,
            'NEUTRALISE_CHARGE': True,
            'REMOVE_STEREO': False,
            'KEEP_BIGGEST': True,
            'ADD_HYDROGEN': False,
            'KEKULIZE': True,
            'NEUTRALISE_CHARGE_LATE': True}

        CustomStandardizer(params=standardisation_params).standardize(dataset)

        return dataset

    @staticmethod
    def featurize_dataset_ml(dataset, featurization_method, model_folder_path):
        fingerprints = None
        if "rdk" in featurization_method:
            fingerprints = RDKFingerprint()
        elif "ecfp8" in featurization_method:
            fingerprints = MorganFingerprint(radius=4, chiral=True)
        elif "ecfp4" in featurization_method:
            fingerprints = MorganFingerprint(chiral=True)
        elif "atompair" in featurization_method:
            fingerprints = AtomPairFingerprint(includeChirality=True)
        elif "2d" in featurization_method:
            fingerprints = TwoDimensionDescriptors()

        fingerprints.featurize(dataset)

        if "2d" in model_folder_path:
            scaler = MinMaxScaler()
            scaler.load_scaler(os.path.join(model_folder_path, "scaler"))
            scaler.transform(dataset)

        return dataset

    @staticmethod
    def filter_valid_sequences(models_folder_path, dataset):

        selected_ids = []
        not_valid_molecules = []

        f = open(os.path.join(models_folder_path, "BiLSTM", "input_params.json"), )
        input_params = json.load(f)
        unique_chars = input_params["unique_chars"]
        char_to_int = input_params["char_to_int"]
        length = input_params["max_len"]
        rnn_feat_gen = RNNFeatureGenerator(unique_chars, char_to_int, length)

        for i, mol_smiles in enumerate(dataset.mols):
            try:
                mol = MolFromSmiles(mol_smiles)
                encoding = rnn_feat_gen.smiles_encoder(mol_smiles)
                if mol is not None and encoding is not None:
                    selected_ids.append(dataset.ids[i])
                else:
                    not_valid_molecules.append(mol_smiles)
            except:
                not_valid_molecules.append(mol_smiles)

        dataset.select(selected_ids, axis=0)
        return dataset, not_valid_molecules

    @staticmethod
    def featurize_dataset_dl(dataset_folder_path, dataset):
        if "GAT" in dataset_folder_path or "GCN" in dataset_folder_path:
            descriptor = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
            descriptor.featurize(dataset)
        elif "TextCNN" in dataset_folder_path:
            descriptor = deepChemFeaturizers.RawFeat()
            descriptor.featurize(dataset)
        elif "GraphConv" in dataset_folder_path:
            descriptor = deepChemFeaturizers.ConvMolFeat()
            descriptor.featurize(dataset)
        elif "LSTM" in dataset_folder_path or "BiLSTM":
            f = open(os.path.join(dataset_folder_path, "input_params.json"), )
            hyperparams = json.load(f)

            descriptor = RNNFeatureGenerator(hyperparams["unique_chars"],
                                             hyperparams["char_to_int"],
                                             hyperparams["max_len"])

            descriptor.featurize(dataset)

        return dataset

    @staticmethod
    def select_features(dataset, feature_selection_method, model_folder_path):
        f = open(os.path.join(model_folder_path, "feature_selection_config.json"), )
        fs = json.load(f)

        if feature_selection_method != "all":
            features_to_keep = sorted(fs[feature_selection_method])
            dataset.select_features(features_to_keep)

        return dataset
