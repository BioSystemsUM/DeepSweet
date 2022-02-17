from unittest import TestCase

from compoundFeaturization import deepChemFeaturizers
from deepchem.models import TextCNNModel

from deepsweet_utils import IO
from generate_features_rnn import RNNFeatureGenerator
from hyperparameter_optimisation import SklearnKerasHyperparameterOptimiser, EndToEndHyperparameterOptimiser

from model_construction import RF, SVM, DNN, GAT, BiLSTM, LSTM, GCN, TextCNN, GraphConv


class TestHyperparameterOptimization(TestCase):

    def setUp(self) -> None:
        dataset_path = "../resources/test_data/2d/train_dataset_Boruta.csv"
        self.dataset = IO.load_dataset_with_features(dataset_path)

        self.rf = RF()
        self.svm = SVM()
        self.dnn = DNN(self.dataset)

        self.train_dataset = IO.load_dataset("../resources/test_data/train_dataset.csv")
        self.test_dataset = IO.load_dataset("../resources/test_data/test_dataset.csv")

        self.gat = GAT()
        self.gcn = GCN()

        self.full_dataset = self.train_dataset.merge([self.test_dataset])
        self.full_dataset.ids = self.full_dataset.mols
        char_dict, length = TextCNNModel.build_char_dict(self.full_dataset)
        self.text_cnn = TextCNN(char_dict, length)
        self.graph_conv = GraphConv(self.dataset)

    def test_rf_optimization(self):
        optimiser_RF = SklearnKerasHyperparameterOptimiser(self.rf, self.dataset, 3,
                                                           "roc_auc",
                                                           1,
                                                           123,
                                                           "../resources/test_data/2d/",
                                                           "Boruta_rf_model")

        optimiser_RF.optimise()
        optimiser_RF.save_results()

    def test_svm_optimization(self):
        optimiser_SVM = SklearnKerasHyperparameterOptimiser(self.svm, self.dataset, 3,
                                                            "roc_auc",
                                                            1,
                                                            123,
                                                            "../resources/test_data/2d/",
                                                            "Boruta_svm_model")

        optimiser_SVM.optimise()
        optimiser_SVM.save_results()

    def test_dnn_optimization(self):
        optimiser_dnn = SklearnKerasHyperparameterOptimiser(self.dnn, self.dataset, 3,
                                                            "roc_auc",
                                                            1,
                                                            123,
                                                            "../resources/test_data/atompair_fp/",
                                                            "Boruta_dnn_model.h5")

        optimiser_dnn.optimise()
        optimiser_dnn.save_results()

    def test_gat_optimization(self):
        featurizer = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
        optimiser_gat = EndToEndHyperparameterOptimiser(self.gat, self.train_dataset, 3,
                                                        "roc_auc",
                                                        1,
                                                        123,
                                                        featurizer,
                                                        "../resources/test_data/GAT/",
                                                        "GAT.h5")

        optimiser_gat.optimise()
        optimiser_gat.save_results()

    def test_bilstm(self):
        featurizer = RNNFeatureGenerator(self.full_dataset)
        featurizer.featurize(self.train_dataset)
        self.bilstm = BiLSTM(self.train_dataset)
        optimiser_bilstm = EndToEndHyperparameterOptimiser(self.bilstm, self.train_dataset, 3,
                                                           "roc_auc",
                                                           1,
                                                           123,
                                                           featurizer,
                                                           "../resources/test_data/BiLSTM/",
                                                           "BiLSTM.h5")

        optimiser_bilstm.optimise()
        optimiser_bilstm.save_results()
