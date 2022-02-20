import json
import os
from typing import Union, List

from Datasets.Datasets import NumpyDataset
from rdkit.Chem import MolFromSmiles, Mol
from standardizer.CustomStandardizer import CustomStandardizer

from deepsweet_models import PreBuiltModel
from generate_features_rnn import RNNFeatureGenerator
from model_construction import Model

import numpy as np


class Ensemble:

    def __init__(self, list_of_models: Union[List[Model], List[PreBuiltModel]]):
        self.list_of_models = list_of_models

    @staticmethod
    def filter_valid_sequences(dataset, dataset_folder_path):

        selected_ids = []
        not_valid_molecules = []

        f = open(os.path.join(dataset_folder_path, "BiLSTM", "input_params.json"), )
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
                    not_valid_molecules.append(mol_smiles)
            except:
                pass

        dataset.select(selected_ids, axis=0)
        return dataset, not_valid_molecules

    def predict(self, molecules: Union[List[Mol], List[str]], models_folder_path: str):

        dataset = NumpyDataset(molecules)
        standardisation_params = {
            'REMOVE_ISOTOPE': True,
            'NEUTRALISE_CHARGE': True,
            'REMOVE_STEREO': False,
            'KEEP_BIGGEST': True,
            'ADD_HYDROGEN': False,
            'KEKULIZE': True,
            'NEUTRALISE_CHARGE_LATE': True}

        CustomStandardizer(params=standardisation_params).standardize(dataset)
        dataset, not_valid_molecules = self.filter_valid_sequences(dataset, models_folder_path)
        all_predictions = []

        for i, model in enumerate(self.list_of_models):

            if isinstance(model, Model):
                predictions = model.predict(dataset)
            elif isinstance(model, PreBuiltModel):
                predictions = model.predict(dataset, standardize=False)
            else:
                raise Exception("Please insert a list of valid models")

            predictions = [i[1] for i in predictions]
            all_predictions.append(predictions)

        def mean_minus_variance(a):
            mean = np.mean(a)
            std = np.std(a)
            return mean - std

        final_all_predictions = np.empty(shape=len(dataset.ids))
        for prediction in all_predictions:
            final_all_predictions = np.column_stack((final_all_predictions, prediction))

        final_all_predictions = np.delete(final_all_predictions, 0, 1)
        mean_predictions = np.apply_along_axis(mean_minus_variance, 1, final_all_predictions)

        return mean_predictions, dataset, not_valid_molecules
