import json
import os
from typing import Union, List

from Datasets.Datasets import NumpyDataset
from rdkit.Chem import MolFromSmiles, Mol
from standardizer.CustomStandardizer import CustomStandardizer

from deepsweet_models import PreBuiltModel
from deepsweet_utils import PipelineUtils
from generate_features_rnn import RNNFeatureGenerator
from model_construction import Model

import numpy as np


class Ensemble:

    def __init__(self, list_of_models: Union[List[Model], List[PreBuiltModel]], models_folder_path: str):
        self.list_of_models = list_of_models
        self.models_folder_path = models_folder_path

    def predict(self, molecules: Union[List[Mol], List[str]], ids=None):

        if ids is None:
            ids = [i for i in range(len(molecules))]

        dataset = NumpyDataset(molecules, ids=np.array(ids))
        standardisation_params = {
            'REMOVE_ISOTOPE': True,
            'NEUTRALISE_CHARGE': True,
            'REMOVE_STEREO': False,
            'KEEP_BIGGEST': True,
            'ADD_HYDROGEN': False,
            'KEKULIZE': True,
            'NEUTRALISE_CHARGE_LATE': True}

        CustomStandardizer(params=standardisation_params).standardize(dataset)
        dataset, not_valid_molecules = PipelineUtils.filter_valid_sequences(self.models_folder_path, dataset)
        all_predictions = []

        for i, model in enumerate(self.list_of_models):

            if isinstance(model, Model):
                predictions, dataset = model.predict(dataset)
            elif isinstance(model, PreBuiltModel):
                predictions, dataset = model.predict(dataset, standardize=False)
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
