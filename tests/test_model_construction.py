import json
import os
from unittest import TestCase

from compoundFeaturization import deepChemFeaturizers

from deepsweet_models import DeepSweetSVM, DeepSweetRF, DeepSweetDNN, DeepSweetGAT, DeepSweetGraphConv, \
    DeepSweetTextCNN, DeepSweetGCN
from deepsweet_utils import IO
from ensemble import Ensemble
from generate_features_rnn import RNNFeatureGenerator
from model_construction import SVM, RF, DNN, GAT, GCN, TextCNN, BiLSTM, GraphConv


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

    def test_load_BiLSTM_and_predict(self):
        test_dataset = IO.load_dataset("../resources/models/test_dataset.csv")

        f = open(os.path.join("../resources/models/BiLSTM/", "input_params.json"), )
        hyperparams = json.load(f)

        descriptor = RNNFeatureGenerator(hyperparams["unique_chars"],
                                         hyperparams["char_to_int"],
                                         hyperparams["max_len"])
        descriptor.featurize(test_dataset)

        model = BiLSTM(test_dataset)
        model.load("../resources/models/BiLSTM/BiLSTM.h5")

        y_predict = model.predict(test_dataset)
        print(y_predict)

    def test_load_graphconv_and_predict(self):
        train_dataset = IO.load_dataset("../resources/models/train_dataset.csv")
        featurizer = deepChemFeaturizers.ConvMolFeat()
        featurizer.featurize(train_dataset)

        f = open(os.path.join("../resources/models/GraphConv/", "GraphConv_hyperparameters.json"), )
        best_hyperparams = json.load(f)

        test_dataset = IO.load_dataset("../resources/models/test_dataset.csv")
        featurizer.featurize(test_dataset)

        model = GraphConv(test_dataset)
        model.load("../resources/models/GraphConv/GraphConv.h5", **best_hyperparams)

        y_predict = model.predict(test_dataset)
        print(y_predict)


class TestPreBuiltModels(TestCase):

    def setUp(self) -> None:
        self.models_folder_path = "../resources/models/"
        self.molecules = ["CN1CCC[C@H]1C2=CN=CC=C2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

    def test_svm(self):
        svm = DeepSweetSVM(self.models_folder_path, "2d", "Boruta")
        print(svm.predict(self.molecules))

    def test_rf(self):
        rf = DeepSweetRF(self.models_folder_path, "2d", "SelectFromModelFS")
        print(rf.predict(self.molecules))

    def test_dnn(self):
        rf = DeepSweetDNN(self.models_folder_path, "ecfp4", "KbestFS")
        print(rf.predict(self.molecules))

    def test_dnn_all(self):
        rf = DeepSweetDNN(self.models_folder_path, "ecfp4", "all")
        print(rf.predict(self.molecules))

    def test_GAT(self):
        rf = DeepSweetGAT(self.models_folder_path)
        print(rf.predict(self.molecules))

    def test_GraphConv(self):
        rf = DeepSweetGraphConv(self.models_folder_path)
        print(rf.predict(self.molecules))

    def test_TextCNN(self):
        rf = DeepSweetTextCNN(self.models_folder_path)
        print(rf.predict(self.molecules))

    def test_ensemble(self):
        list_of_models = [DeepSweetTextCNN(self.models_folder_path),
                          DeepSweetGCN(self.models_folder_path),
                          DeepSweetDNN(self.models_folder_path, "rdk", "all")]
        ensemble = Ensemble(list_of_models, self.models_folder_path)
        print(ensemble.predict(self.molecules))
