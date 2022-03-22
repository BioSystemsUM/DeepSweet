import json
import os
from abc import ABC, abstractmethod
from typing import List, Union

from Datasets.Datasets import NumpyDataset, Dataset
from compoundFeaturization import deepChemFeaturizers
from rdkit.Chem import Mol

from deepsweet_utils import IO, PipelineUtils
from model_construction import SVM, Model, RF, DNN, GAT, GCN, BiLSTM, LSTM, TextCNN, GraphConv

import numpy as np


class PreBuiltModel(ABC):

    def __init__(self, model_folder_path):
        self._model = None
        self.model_folder_path = model_folder_path

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value: Model):
        self._model = value

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, molecules: List[str], ids: List[str] = None, standardize: bool = True):
        raise NotImplementedError


class PreBuiltMLDNN(PreBuiltModel, ABC):

    def __init__(self, model_folder_path, featurization_method, feature_selection_method):
        super().__init__(model_folder_path)
        self.model = None
        self.model_folder_path = os.path.join(self.model_folder_path, featurization_method)
        self.featurization_method = featurization_method
        self.feature_selection_method = feature_selection_method
        self.load()

    def predict(self, molecules: List[str], ids: List[str] = None, standardize: bool = True):
        if ids is None:
            ids = [i for i in range(len(molecules))]

        if not isinstance(molecules, Dataset):
            dataset = NumpyDataset(molecules, ids=np.array(ids))
        else:
            dataset = molecules

        if standardize:
            PipelineUtils.standardize_dataset(dataset)

        PipelineUtils.featurize_dataset_ml(dataset, self.featurization_method, self.model_folder_path)

        PipelineUtils.select_features(dataset, self.feature_selection_method, self.model_folder_path)

        y_predicted = self.model.predict(dataset)

        return y_predicted, dataset


class DeepSweetSVM(PreBuiltMLDNN):

    def __init__(self, model_folder_path, featurization_method, feature_selection_method):
        super().__init__(model_folder_path, featurization_method, feature_selection_method)

    def load(self):
        model_relative_path = self.feature_selection_method + "_svm_model"

        model_path = os.path.join(self.model_folder_path, model_relative_path)

        self.model = SVM()
        self.model.load(model_path)


class DeepSweetRF(PreBuiltMLDNN):

    def __init__(self, model_folder_path, featurization_method, feature_selection_method):
        super().__init__(model_folder_path, featurization_method, feature_selection_method)

    def load(self):
        model_relative_path = self.feature_selection_method + "_rf_model"

        model_path = os.path.join(self.model_folder_path, model_relative_path)

        self.model = RF()
        self.model.load(model_path)


class DeepSweetDNN(PreBuiltMLDNN):

    def __init__(self, model_folder_path, featurization_method, feature_selection_method):
        super().__init__(model_folder_path, featurization_method, feature_selection_method)

    def load(self):
        model_relative_path = self.feature_selection_method + "_dnn_model.h5"

        model_path = os.path.join(self.model_folder_path, model_relative_path)

        self.model = DNN(None, False)
        self.model.load(model_path)


class PreBuiltEntToEnd(PreBuiltModel, ABC):

    def __init__(self, model_folder_path, method):
        super().__init__(model_folder_path)
        self.model = None
        self.model_folder_path = os.path.join(model_folder_path, method)
        self.load()

    def predict(self, molecules: Union[List[str], List[Mol], Dataset], ids: List[str] = None, standardize: bool = True):
        if ids is None:
            ids = [i for i in range(len(molecules))]

        if not isinstance(molecules, Dataset):
            dataset = NumpyDataset(molecules, ids=np.array(ids))
        else:
            dataset = molecules

        if standardize:
            PipelineUtils.standardize_dataset(dataset)

        PipelineUtils.featurize_dataset_dl(self.model_folder_path, dataset)

        y_predicted = self.model.predict(dataset)

        return y_predicted, dataset


class DeepSweetGAT(PreBuiltEntToEnd):

    def __init__(self, model_folder_path, device):
        self.device = device
        super().__init__(model_folder_path, "GAT")

    def load(self):
        model_path = os.path.join(self.model_folder_path, "GAT.h5")
        f = open(os.path.join(self.model_folder_path, "GAT_hyperparameters.json"), )
        hyperparams = json.load(f)
        f.close()
        self.model = GAT(self.device)
        self.model.load(model_path, **hyperparams)


class DeepSweetGCN(PreBuiltEntToEnd):

    def __init__(self, model_folder_path, device):
        self.device = device
        super().__init__(model_folder_path, "GCN")

    def load(self, **kwargs):
        model_path = os.path.join(self.model_folder_path, "GCN.h5")
        f = open(os.path.join(self.model_folder_path, "GCN_hyperparameters.json"), )
        hyperparams = json.load(f)
        f.close()
        self.model = GCN(self.device)
        self.model.load(model_path, **hyperparams)


class DeepSweetBiLSTM(PreBuiltEntToEnd):

    def __init__(self, model_folder_path):
        super().__init__(model_folder_path, "BiLSTM")

    def load(self, **kwargs):
        model_path = os.path.join(self.model_folder_path, "BiLSTM.h5")
        self.model = BiLSTM(None, False)
        self.model.load(model_path)


class DeepSweetLSTM(PreBuiltEntToEnd):

    def __init__(self, model_folder_path):
        super().__init__(model_folder_path, "LSTM")

    def load(self, **kwargs):
        model_path = os.path.join(self.model_folder_path, "LSTM.h5")
        self.model = LSTM(None, False)
        self.model.load(model_path)


class DeepSweetTextCNN(PreBuiltEntToEnd):

    def __init__(self, model_folder_path):
        super().__init__(model_folder_path, "TextCNN")

    def load(self, **kwargs):
        model_path = os.path.join(self.model_folder_path, "TextCNN.h5")
        f = open(os.path.join("../resources/models/TextCNN/", "input_params.json"), )
        hyperparams = json.load(f)
        f.close()

        self.model = TextCNN(hyperparams["char_dict"], hyperparams["length"])
        self.model.load(model_path)


class DeepSweetGraphConv(PreBuiltEntToEnd):

    def __init__(self, model_folder_path):
        super().__init__(model_folder_path, "GraphConv")

    def load(self, **kwargs):
        model_path = os.path.join(self.model_folder_path, "GraphConv.h5")

        train_dataset = IO.load_dataset(os.path.join(self.model_folder_path, "train_dataset.csv"))
        featurizer = deepChemFeaturizers.ConvMolFeat()
        featurizer.featurize(train_dataset)

        f = open(os.path.join(self.model_folder_path, "GraphConv_hyperparameters.json"), )
        best_hyperparams = json.load(f)
        f.close()

        model = GraphConv(train_dataset)
        model.load(model_path, **best_hyperparams)
