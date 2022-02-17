import json
import os
from unittest import TestCase

from compoundFeaturization import deepChemFeaturizers

from deepsweet_utils import IO
from model_construction import SVM, RF, DNN, GAT, GCN, TextCNN


class TestModelConstruction(TestCase):

    def test_svm(self):
        model = SVM()
        model.define_model_hyperparameters()
        model.save("../resources/test_data/2d/svm_model")

    def test_load_svm_and_predict(self):
        model = SVM()
        model.load("../resources/models/2d/all_svm_model")
        test_dataset = IO.load_dataset_with_features("../resources/models/2d/test_dataset.csv")

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_svm_fs_and_predict(self):
        model = SVM()
        model.load("../resources/models/2d/Boruta_svm_model")

        test_dataset = IO.load_dataset_with_features("../resources/models/2d/test_dataset.csv")
        features = IO.load_json_config("../resources/models/2d/feature_selection_config.json")

        features_to_keep = sorted(features["Boruta"])
        test_dataset.select_features(features_to_keep)
        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_rf_and_predict(self):
        model = RF()
        model.load("../resources/models/2d/all_rf_model")
        test_dataset = IO.load_dataset_with_features("../resources/models/2d/test_dataset.csv")

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_dnn_and_predict(self):
        test_dataset = IO.load_dataset_with_features("../resources/models/2d/test_dataset.csv")
        model = DNN(test_dataset)
        model.load("../resources/models/2d/all_dnn_model.h5")

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_GAT_and_predict(self):
        test_dataset = IO.load_dataset("../resources/models/test_dataset.csv")

        descriptor = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
        descriptor.featurize(test_dataset)

        f = open(os.path.join("../resources/models/GAT/", "GAT_hyperparameters.json"), )
        hyperparams = json.load(f)
        model = GAT()
        model.load("../resources/models/GAT/GAT.h5", **hyperparams)

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_GCN_and_predict(self):
        test_dataset = IO.load_dataset("../resources/models/test_dataset.csv")

        descriptor = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
        descriptor.featurize(test_dataset)

        f = open(os.path.join("../resources/models/GCN/", "GCN_hyperparameters.json"), )
        hyperparams = json.load(f)
        model = GCN()
        model.load("../resources/models/GCN/GCN.h5", **hyperparams)

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_TextCNN_and_predict(self):
        test_dataset = IO.load_dataset("../resources/models/test_dataset.csv")

        descriptor = deepChemFeaturizers.RawFeat()
        descriptor.featurize(test_dataset)

        f = open(os.path.join("../resources/models/TextCNN/", "input_params.json"), )
        hyperparams = json.load(f)

        model = TextCNN(hyperparams["char_dict"], hyperparams["length"])
        model.load("../resources/models/TextCNN/TextCNN.h5")

        y_predict = model.predict(test_dataset)
        print(y_predict)
