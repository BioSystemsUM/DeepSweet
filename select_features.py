import json
import os

import pandas as pd
from featureSelection.baseFeatureSelector import BorutaAlgorithm, SelectFromModelFS, KbestFS
from loaders.Loaders import CSVLoader
from sklearn.ensemble import RandomForestClassifier

from pipelines import Step


class FeatureSelector(Step):

    def __init__(self, dataset_path, models_folder_path, estimator=None):
        self.dataset_path = dataset_path
        self.models_folder_path = models_folder_path
        self.estimator = estimator
        super().__init__()

    @staticmethod
    def run_boruta(estimator, dataset):
        return BorutaAlgorithm(estimator=estimator, max_iter=100, support_weak=True).select_features(dataset)

    @staticmethod
    def run_model_fs(estimator, dataset):
        return SelectFromModelFS(estimator=estimator).select_features(dataset)

    @staticmethod
    def run_model_kbest_fs(dataset, k):
        return KbestFS(k=k).select_features(dataset)

    @staticmethod
    def construct_pipeline_json(features_to_keep, feature_selection_method):
        json_dict = {feature_selection_method: [int(i) for i in features_to_keep]}

        return json_dict

    def select_with_all_methods(self):
        pandas_dset = pd.read_csv(self.dataset_path)
        columns = list(pandas_dset.columns[3:])

        if self.estimator is None:
            self.estimator = RandomForestClassifier(bootstrap=False, class_weight={0: 2.0, 1: 1.0},
                                                    criterion='entropy', max_depth=80, min_samples_leaf=2,
                                                    n_estimators=900, n_jobs=-1)

        loader = CSVLoader(self.dataset_path,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()

        dataset = self.run_model_fs(self.estimator, dataset)
        features = dataset.features2keep
        json_dict = self.construct_pipeline_json(features, "SelectFromModelFS")

        pandas_dset = pd.read_csv(self.dataset_path)
        columns = list(pandas_dset.columns[3:])

        loader = CSVLoader(self.dataset_path,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()
        dataset = self.run_model_kbest_fs(dataset, 100)

        features = dataset.features2keep
        json_dict2 = self.construct_pipeline_json(features, "KbestFS")
        json_dict.update(json_dict2)

        pandas_dset = pd.read_csv(dataset)
        columns = list(pandas_dset.columns[3:])

        loader = CSVLoader(self.dataset_path,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()

        dataset = self.run_boruta(self.estimator, dataset)
        features_select_from_model = dataset.features2keep
        json_dict2 = self.construct_pipeline_json(features_select_from_model, "Boruta")
        json_dict.update(json_dict2)

        with open(os.path.join(self.models_folder_path, "feature_selection_config.json"), 'w',
                  encoding='utf8') as json_file:
            json.dump(json_dict, json_file)

    def run(self):
        self.select_with_all_methods()
