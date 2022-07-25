import json
import os
from abc import ABC

import joblib
import tensorflow
import torch
from compoundFeaturization import deepChemFeaturizers
from compoundFeaturization.rdkitDescriptors import TwoDimensionDescriptors
from compoundFeaturization.rdkitFingerprints import RDKFingerprint, MorganFingerprint, AtomPairFingerprint
from deepchem.models import torch_models, TextCNNModel
from deepchem.models.layers import DTNNEmbedding, Highway
from loaders.Loaders import CSVLoader
from models.DeepChemModels import DeepChemModel
from pandas import DataFrame
from scalers.sklearnScalers import MinMaxScaler

from metrics.Metrics import Metric
from metrics.metricsFunctions import roc_auc_score, precision_score, f1_score
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

import pandas as pd

from deepsweet_utils import PipelineUtils


class Report(ABC):

    def __init__(self):
        pass


class ResultsReport(Report):

    def __init__(self, device, train_dataset_path, test_dataset_path, blend_set_path):
        super().__init__()
        self.device = device
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.blend_set_path = blend_set_path

    @staticmethod
    def generate_features_ml_dnn(dataset_folder_path, dataset, features_to_select):
        fingerprints = None
        if "rdk" in dataset_folder_path:
            fingerprints = RDKFingerprint()
        elif "ecfp8" in dataset_folder_path:
            fingerprints = MorganFingerprint(radius=4, chiral=True)
        elif "ecfp4" in dataset_folder_path:
            fingerprints = MorganFingerprint(chiral=True)
        elif "atompair" in dataset_folder_path:
            fingerprints = AtomPairFingerprint(includeChirality=True)
        elif "2d" in dataset_folder_path:
            fingerprints = TwoDimensionDescriptors()

        fingerprints.featurize(dataset)

        if "2d" in dataset_folder_path:
            scaler = MinMaxScaler()
            scaler.load_scaler(os.path.join(dataset_folder_path, "scaler"))
            scaler.transform(dataset)

        dataset.select(features_to_select, axis=1)
        return dataset

    def predict_on_blend_test_set(self, file, dataset_folder_path, features_to_select):

        if "dnn" not in file:
            model = joblib.load(os.path.join(dataset_folder_path, file))

        else:
            from models.kerasModels import KerasModel
            model = keras.models.load_model(os.path.join(dataset_folder_path, file))
            classifier = KerasClassifier(model)
            classifier.model = model
            model = KerasModel(len)
            model.model = classifier

        blend_dataset = "blend_test_set.csv"

        loader = CSVLoader(blend_dataset,
                           mols_field='smiles',
                           labels_fields='y')
        blend_test_dataset = loader.create_dataset()

        blend_test_dataset = self.generate_features_ml_dnn(dataset_folder_path, blend_test_dataset, features_to_select)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(balanced_accuracy_score), Metric(recall_score), Metric(f1_score)]

        results = model.evaluate(blend_test_dataset, metrics)

        roc_auc_score_ = results["roc_auc_score"]
        precision_score_ = results["precision_score"]
        accuracy_score_ = results["balanced_accuracy_score"]
        recall_score_ = results["recall_score"]
        f1_score_ = results["f1_score"]

        return [roc_auc_score_, precision_score_, accuracy_score_, recall_score_, f1_score_]

    def predict_for_all_features(self, descriptor, file, models_folder_path, blend=True):

        if descriptor == "2d":
            features_to_keep = [i for i in range(208)]
        else:
            features_to_keep = [i for i in range(2048)]

        scores = self.predict_on_test_set(file, models_folder_path, features_to_keep)
        if blend:
            blend_scores = self.predict_on_blend_test_set(file, models_folder_path, features_to_keep)
        else:
            blend_scores = None

        return scores, blend_scores

    @staticmethod
    def predict_on_test_set(file, dataset_folder_path, features_to_select):
        if "dnn" not in file:
            model = joblib.load(os.path.join(dataset_folder_path, file))

        else:
            model = keras.models.load_model(os.path.join(dataset_folder_path, file))
            classifier = KerasClassifier(model)
            classifier.model = model
            from models.kerasModels import KerasModel
            model = KerasModel(len)
            model.model = classifier

        dataset_path = os.path.join(dataset_folder_path, "train_dataset.csv")
        pandas_dset = pd.read_csv(dataset_path)
        columns = pandas_dset.columns[3:]
        columns = list(columns[features_to_select])

        test_dataset = os.path.join(dataset_folder_path, "test_dataset.csv")

        loader = CSVLoader(test_dataset,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')
        test_dataset = loader.create_dataset()

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(balanced_accuracy_score), Metric(recall_score), Metric(f1_score)]

        results = model.evaluate(test_dataset, metrics)

        roc_auc_score_ = results["roc_auc_score"]
        precision_score_ = results["precision_score"]
        accuracy_score_ = results["balanced_accuracy_score"]
        recall_score_ = results["recall_score"]
        f1_score_ = results["f1_score"]

        return [roc_auc_score_, precision_score_, accuracy_score_, recall_score_, f1_score_]

    def predict_for_feature_selection_method(self, file, method, models_folder_path, blend=True):

        features_to_keep = os.path.join(models_folder_path, "feature_selection_config.json")
        f = open(features_to_keep, )

        data = json.load(f)

        features_to_keep = sorted(data[method])
        scores = self.predict_on_test_set(file, models_folder_path, features_to_keep)
        if blend:
            blend_scores = self.predict_on_blend_test_set(file, models_folder_path, features_to_keep)
        else:
            blend_scores = None

        return scores, blend_scores

    def predict_for_feature_selection_methods(self, file, models_folder_path, blend=True):
        scores, blend_scores, feature_selection_method = None, None, None
        if "KbestFS" in file:
            feature_selection_method = "KbestFS"
            scores, blend_scores = self.predict_for_feature_selection_method(file, feature_selection_method,
                                                                             models_folder_path, blend)

        elif "SelectFromModelFS" in file:
            feature_selection_method = "SelectFromModelFS"
            scores, blend_scores = self.predict_for_feature_selection_method(file, feature_selection_method,
                                                                             models_folder_path, blend)

        elif "Boruta" in file:
            feature_selection_method = "Boruta"
            scores, blend_scores = self.predict_for_feature_selection_method(file, feature_selection_method,
                                                                             models_folder_path, blend)

        return scores, blend_scores, feature_selection_method

    def load_model_accordingly(self, file, dataset_folder_path):

        from deepchem.models import GraphConvModel

        def graphconv_builder(graph_conv_layers, dense_layer_size, dropout, learning_rate, task_type, batch_size=256,
                              epochs=100):
            graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, dense_layer_size=dense_layer_size,
                                   dropout=dropout, batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
            return DeepChemModel(graph, epochs=epochs, use_weights=False, model_dir=None)

        model = None
        if "GAT" in file:
            model = torch.load(os.path.join(dataset_folder_path, file))
            model.eval()
            model.to(self.device)

            f = open(os.path.join(dataset_folder_path, "GAT_hyperparameters.json"), )
            best_hyperparams = json.load(f)

            new_model = torch_models.GATModel(1, **best_hyperparams)
            new_model.model = model
            model = DeepChemModel(new_model)

        elif "GCN" in file:
            model = torch.load(os.path.join(dataset_folder_path, "GCN.h5"))
            model.eval()
            model.to(self.device)

            f = open(os.path.join(dataset_folder_path, "GCN_hyperparameters.json"), )
            best_hyperparams = json.load(f)

            new_model = torch_models.GCNModel(1, **best_hyperparams)
            new_model.model = model

            model = DeepChemModel(new_model)

        elif "TextCNN" in file:
            tensorflow_model = tensorflow.keras.models.load_model(os.path.join(dataset_folder_path, "TextCNN.h5"),
                                                                  custom_objects={"DTNNEmbedding": DTNNEmbedding,
                                                                                  "Highway": Highway})

            f = open(os.path.join(dataset_folder_path, "input_params.json"), )
            input_params = json.load(f)

            new_model = TextCNNModel(n_tasks=1, char_dict=input_params["char_dict"],
                                     seq_length=input_params["length"])

            new_model.model = tensorflow_model

            model = DeepChemModel(new_model)

        elif "LSTM" in file or "BiLSTM" in file:
            from models.kerasModels import KerasModel
            lstm_model = tensorflow.keras.models.load_model(os.path.join(dataset_folder_path, file))

            keras_model = KerasClassifier(lstm_model)
            keras_model.model = lstm_model
            model = KerasModel(len)
            model.model = keras_model

        elif "GraphConv" in file:
            f = open(os.path.join(dataset_folder_path, "GraphConv_hyperparameters.json"), )
            best_hyperparams = json.load(f)

            train_dataset = CSVLoader(dataset_path=file,
                                      id_field="ids",
                                      mols_field='mols',
                                      labels_fields='y')
            train_dataset = train_dataset.create_dataset()

            featurizer = deepChemFeaturizers.ConvMolFeat()
            featurizer.featurize(train_dataset)

            graph = graphconv_builder(**best_hyperparams)
            graph.fit(train_dataset)
            graph.model.model.load_weights(os.path.join(dataset_folder_path, "GraphConv.h5"))

        return model

    def run_all_ml(self, models_folder_path, output_file_path, blend=True):
        if blend:
            results = DataFrame(columns=["descriptor", "feature selection", "algorithm", "test_roc_auc_score",
                                         "test_precision_score", "test_ner_score", "test_recall_score", "test_f1_score",
                                         "blend_roc_auc_score", "blend_precision_score", "blend_ner_score",
                                         "blend_recall_score", "blend_f1_score"])
        else:
            results = DataFrame(columns=["descriptor", "feature selection", "algorithm", "test_roc_auc_score",
                                         "test_precision_score", "test_ner_score", "test_recall_score", "test_f1_score",
                                         ])

        folders = os.listdir(models_folder_path)
        for folder_path in folders:
            if ".csv" not in folder_path and ".keep" not in folder_path:
                model_folder_path = os.path.join(models_folder_path, folder_path)
                files = os.listdir(model_folder_path)
                for file in files:
                    if "hyperparameter" not in file and "model" in file:
                        descriptor = folder_path
                        algorithm = None
                        try:
                            if "all" in file:
                                feature_selection_method = "none"
                                scores, blend_scores = self.predict_for_all_features(descriptor, file, model_folder_path,
                                                                                     blend)
                            else:
                                scores, blend_scores, feature_selection_method = \
                                    self.predict_for_feature_selection_methods(file, model_folder_path, blend)

                            if "rf" in file:
                                algorithm = "rf"

                            elif "svm" in file:
                                algorithm = "svm"

                            elif "dnn" in file:
                                algorithm = "dnn"

                            last_id = results.shape[0]
                            results.at[last_id, "descriptor"] = descriptor
                            results.at[last_id, "feature selection"] = feature_selection_method
                            results.at[last_id, "algorithm"] = algorithm
                            results.at[last_id, 3:8] = scores
                            if blend:
                                results.at[last_id, 8:13] = blend_scores

                        except ValueError as e:
                            print(f"WARNING: {file} does not work for test set")

        results.to_csv(output_file_path, index=False)

    def predict_on_both_sets(self, file, dataset_folder_path):
        model = self.load_model_accordingly(file, dataset_folder_path)

        test_dataset = self.test_dataset_path
        blend_set = self.blend_set_path

        loader = CSVLoader(test_dataset,
                           mols_field='mols',
                           labels_fields='y')

        test_dataset = loader.create_dataset()

        test_dataset = PipelineUtils.featurize_dataset_dl(dataset_folder_path, test_dataset)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(balanced_accuracy_score), Metric(recall_score), Metric(f1_score),
                   Metric(confusion_matrix)]

        results = model.evaluate(test_dataset, metrics)

        roc_auc_score_ = results["roc_auc_score"]
        precision_score_ = results["precision_score"]
        accuracy_score_ = results["balanced_accuracy_score"]
        recall_score_ = results["recall_score"]
        f1_score_ = results["f1_score"]

        test_set_results = [roc_auc_score_, precision_score_, accuracy_score_, recall_score_, f1_score_]

        loader = CSVLoader(blend_set,
                           mols_field='smiles',
                           labels_fields='y')
        blend_dataset = loader.create_dataset()
        blend_dataset = PipelineUtils.featurize_dataset_dl(dataset_folder_path, blend_dataset)
        results = model.evaluate(blend_dataset, metrics)

        roc_auc_score_ = results["roc_auc_score"]
        precision_score_ = results["precision_score"]
        accuracy_score_ = results["balanced_accuracy_score"]
        recall_score_ = results["recall_score"]
        f1_score_ = results["f1_score"]

        blend_set_results = [roc_auc_score_, precision_score_, accuracy_score_, recall_score_, f1_score_]

        return test_set_results, blend_set_results

    def run_all_dl(self, models_folder_path, output_file_path):

        results = DataFrame(columns=["algorithm", "test_roc_auc_score",
                                     "test_precision_score", "test_ner_score", "test_recall_score", "test_f1_score",
                                     "blend_roc_auc_score", "blend_precision_score", "blend_ner_score",
                                     "blend_recall_score", "blend_f1_score"])

        folders = os.listdir(models_folder_path)
        for folder_path in folders:
            if ".csv" not in folder_path:
                models_folder_path = os.path.join(models_folder_path, folder_path)
                files = os.listdir(models_folder_path)
                algorithm = False
                for file in files:
                    if "hyperparameter" not in file and ".h5" in file \
                            and "dnn" not in file:
                        go = False
                        if "TextCNN" in file:
                            algorithm = "TextCNN"
                            go = True

                        elif "BiLSTM" in file:
                            algorithm = "BiLSTM"
                            go = True

                        elif "LSTM" in file:
                            algorithm = "LSTM"
                            go = True

                        elif "GAT" in file:
                            algorithm = "GAT"
                            go = True

                        elif "GCN" in file:
                            algorithm = "GCN"
                            go = True

                        elif "GraphConv" in file:
                            algorithm = "GraphConv"
                            go = True

                        if go:
                            test_set_results, blend_set_results = self.predict_on_both_sets(file, models_folder_path)

                            last_id = results.shape[0]
                            results.at[last_id, "algorithm"] = algorithm
                            results.at[last_id, 1:6] = test_set_results
                            results.at[last_id, 6:11] = blend_set_results

        results.to_csv(output_file_path, index=False)
