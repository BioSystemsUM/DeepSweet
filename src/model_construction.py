import json
import os
from abc import ABC, abstractmethod

import joblib
import torch
from Datasets.Datasets import Dataset
from deepchem.models import torch_models, GraphConvModel, TextCNNModel
from deepchem.models.layers import DTNNEmbedding, Highway
from models.DeepChemModels import DeepChemModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf

from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.keras.layers import Bidirectional, CuDNNLSTM
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


class Model(ABC):

    def __init__(self, model_type, construct_grid=True):
        self.model_type = model_type
        self.model = None
        self.hyperparameters_grid = {}
        self.construct_model()
        if construct_grid:
            self.construct_grid()

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def hyperparameters_grid(self):
        return self._hyperparameters_grid

    @hyperparameters_grid.setter
    def hyperparameters_grid(self, value):
        self._hyperparameters_grid = value

    @abstractmethod
    def construct_grid(self):
        raise NotImplementedError

    @abstractmethod
    def _construct_model(self):
        raise NotImplementedError

    @abstractmethod
    def _save(self, output_path):
        raise NotImplementedError

    @abstractmethod
    def save_input_params(self, output_path):
        raise NotImplementedError

    def save(self, output_path):
        if self.model is not None:
            self._save(output_path)
        else:
            raise Exception("No model was defined")

    @abstractmethod
    def load(self, file_path):
        raise NotImplementedError

    def predict(self, test_dataset: Dataset):

        y_predicted = self.model.predict(test_dataset)
        return y_predicted

    def define_model_hyperparameters(self, **hyperparameters):
        self.model = self.model_construction_function(**hyperparameters)

    def construct_model(self):
        self.model_construction_function = self._construct_model()


class SVM(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, construct_grid=True):
        model_type = "sklearn"
        super().__init__(model_type, construct_grid)

    def _save(self, output_path):
        joblib.dump(self.model, output_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)

    def _construct_model(self):
        def svm_constructor(C=1, gamma=1e-4, kernel='rbf', degree=2, class_weight=None, coef0=0.1):
            if class_weight is None:
                class_weight = {0: 1., 1: 1.}
            svm_model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, degree=degree, class_weight=class_weight,
                            coef0=coef0)

            return svm_model

        return svm_constructor

    def construct_grid(self):
        self.hyperparameters_grid = {'C': [1, 0.7, 0.5, 0.3, 0.1, 0.2, 0.25, 0.8],
                                     'gamma': ['scale', 'auto'],
                                     'degree': [2, 3],
                                     'coef0': [0.1, 0.2, 0.3, 0.4, 0.5],
                                     'class_weight': [{0: 1.0, 1: 1.0},
                                                      {0: 2.0, 1: 1.0},
                                                      {0: 1.0, 1: 2.0},
                                                      {0: 1.0, 1: 3.0}],
                                     'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}


class RF(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self):
        model_type = "sklearn"
        super().__init__(model_type)

    def _save(self, output_path):
        joblib.dump(self.model, output_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)

    def construct_grid(self):
        self.hyperparameters_grid = {'n_estimators': [100, 150, 175, 200, 300, 400, 500, 600, 800, 900, 1000],
                                     'max_features': ['sqrt', 'log2', 'auto'],
                                     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                     'min_samples_leaf': [1, 2, 3, 4, 10, 15],
                                     'min_samples_split': [2, 5, 10, 15, 20, 25, 50, 100],
                                     'criterion': ['gini', 'entropy'],
                                     'bootstrap': [True, False],
                                     'class_weight': [{0: 1.0, 1: 1.0},
                                                      {0: 2.0, 1: 1.0},
                                                      {0: 1.0, 1: 2.0},
                                                      {0: 1.0, 1: 3.0}]}

    def _construct_model(self):
        def rf_constructor(n_estimators=10, max_features='auto', bootstrap='gini',
                           min_samples_split=2, min_samples_leaf=1, max_depth=10, class_weight=None,
                           criterion='gini'):
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                              class_weight=class_weight, bootstrap=bootstrap,
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              max_depth=max_depth, criterion=criterion)
            return rf_model

        return rf_constructor


class DNN(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, train_dataset, construct_grid=True):
        model_type = "keras"
        self.train_dataset = train_dataset
        super().__init__(model_type, construct_grid)

    def _save(self, output_path):

        self.model.model.model.save(output_path)

    def load(self, file_path):

        from models.kerasModels import KerasModel
        model = tf.keras.models.load_model(file_path)
        classifier = KerasClassifier(model)
        classifier.model = model
        self.model = KerasModel(self.model_construction_function)
        self.model.model = classifier

    def construct_grid(self):
        self.hyperparameters_grid = {

            "hlayers_sizes": [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64], [512, 256, 128, 64, 28],
                              [256], [256, 128], [256, 128, 64], [256, 128, 64, 28],
                              [128], [128, 64], [128, 64, 28], [64], [64, 28], [28]],

            'learning_rate_value': [1e-4, 1e-3, 1e-2],
            'dropout': [0.0, 0.25, 0.5],
            "optimizer_name": ["Adam", "Adagrad", "Adamax", "Adadelta"],
            "batchnorm": [True, False],
            "epochs": [40],
            "batch_size": [200, 100, 50],
            "verbose": [0],
            "input_dim": [self.train_dataset.X.shape[1]]
        }

    @staticmethod
    def construct_layers(model, hlayers_sizes, initializer, l1, l2, batchnorm, dropout):

        for i in range(len(hlayers_sizes)):
            model.add(Dense(units=hlayers_sizes[i], kernel_initializer=initializer,
                            kernel_regularizer=l1_l2(l1=l1, l2=l2)))
            if batchnorm:
                model.add(BatchNormalization())

            model.add(Activation('relu'))

            if dropout > 0:
                model.add(Dropout(rate=dropout))

        return model

    @staticmethod
    def define_optimiser(model, optimizer_name, learning_rate_value):

        optimizer = None
        if optimizer_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_value,
                name=optimizer_name)

        elif optimizer_name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate_value, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad'
            )

        elif optimizer_name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate=learning_rate_value, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Adamax'
            )
        elif optimizer_name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate_value, rho=0.95, epsilon=1e-07, name='Adadelta'
            )

        if optimizer is not None:

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        else:
            raise Exception(
                "Please choose an optimizer from the following options: Adadelta, Adamax, Adagrad, Adam")

        return model

    def _construct_model(self):

        def dense_builder(input_dim=None, hlayers_sizes=None, initializer='he_normal',
                          l1=0, l2=0, dropout=0, batchnorm=True, learning_rate_value=0.001,
                          optimizer_name="Adam"):
            if hlayers_sizes is None:
                hlayers_sizes = [10]
            model = Sequential()
            model.add(Input(shape=input_dim))

            model = self.construct_layers(model, hlayers_sizes, initializer, l1, l2, batchnorm, dropout)

            model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

            model = self.define_optimiser(model, optimizer_name, learning_rate_value)

            return model

        return dense_builder


class GAT(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, device="cpu"):
        model_type = "deepchem"
        self.device = device
        super().__init__(model_type)

    def _save(self, output_path):
        torch.save(self.model.model.model, output_path)

    def load(self, file_path, **kwargs):
        model = torch.load(file_path, map_location=torch.device(self.device))
        model.eval()

        new_model = torch_models.GATModel(1, **kwargs)
        new_model.model = model
        self.model = DeepChemModel(new_model)

    def construct_grid(self):
        mode = 'classification'
        self.hyperparameters_grid = {'graph_attention_layers': [[8, 8], [16, 16], [32, 32],
                                                                [8, 8, 8], [16, 16, 16], [32, 32, 32]],
                                     'n_attention_heads': [4, 8],
                                     'predictor_hidden_feats': [256, 128, 64],
                                     'dropout': [0.0, 0.25, 0.5],
                                     'predictor_dropout': [0.0, 0.25, 0.5],
                                     'learning_rate': [1e-4, 1e-3, 1e-2],
                                     'task_type': [mode], 'epochs': [40]}

    def _construct_model(self):
        def gat_builder(graph_attention_layers, n_attention_heads, dropout, predictor_hidden_feats,
                        predictor_dropout, learning_rate, task_type, batch_size=256, epochs=100):
            gat = torch_models.GATModel(n_tasks=1, graph_attention_layers=graph_attention_layers,
                                        n_attention_heads=n_attention_heads, dropout=dropout,
                                        predictor_hidden_feats=predictor_hidden_feats,
                                        predictor_dropout=predictor_dropout,
                                        learning_rate=learning_rate, batch_size=batch_size, mode=task_type)

            return DeepChemModel(gat, epochs=epochs, use_weights=False, model_dir=None)

        return gat_builder


class GCN(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, device="cpu"):
        model_type = "deepchem"
        self.device = device
        super().__init__(model_type)

    def _save(self, output_path):
        torch.save(self.model.model.model, output_path)

    def load(self, file_path, **kwargs):
        model = torch.load(file_path, map_location=torch.device(self.device))
        model.eval()

        new_model = torch_models.GCNModel(1, **kwargs)
        new_model.model = model
        self.model = DeepChemModel(new_model)

    def construct_grid(self):
        mode = 'classification'
        self.hyperparameters_grid = {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                                           [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                                           [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                                     'predictor_hidden_feats': [256, 128, 64],
                                     'dropout': [0.0, 0.25, 0.5],
                                     'predictor_dropout': [0.0, 0.25, 0.5],
                                     'learning_rate': [1e-4, 1e-3, 1e-2],
                                     'task_type': [mode], 'epochs': [40]}

    def _construct_model(self):
        def gcn_builder(graph_conv_layers, dropout, predictor_hidden_feats, predictor_dropout, learning_rate, task_type,
                        batch_size=256, epochs=100):
            gcn = torch_models.GCNModel(n_tasks=1, graph_conv_layers=graph_conv_layers, activation=None,
                                        residual=True, batchnorm=False, predictor_hidden_feats=predictor_hidden_feats,
                                        dropout=dropout, predictor_dropout=predictor_dropout,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size, mode=task_type)

            return DeepChemModel(gcn, epochs=epochs, use_weights=False, model_dir=None)

        return gcn_builder


class GraphConv(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, train_dataset, construct_grid=True):
        model_type = "deepchem"
        self.train_dataset = train_dataset
        super().__init__(model_type, construct_grid)

    def _save(self, output_path):
        torch.save(self.model.model.model, output_path)

    def load(self, file_path, **kwargs):
        self.model = self.model_construction_function(**kwargs)
        self.model.fit(self.train_dataset)
        self.model.model.model.load_weights(file_path)

    def construct_grid(self):
        mode = 'classification'
        self.hyperparameters_grid = {'graph_conv_layers': [[32, 32], [64, 64], [128, 128],
                                                           [32, 32, 32], [64, 64, 64], [128, 128, 128],
                                                           [32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128]],
                                     'dense_layer_size': [2048, 1024, 512, 256, 128, 64, 32],
                                     'dropout': [0.0, 0.25, 0.5], 'learning_rate': [1e-4, 1e-3, 1e-2],
                                     'task_type': [mode], 'epochs': [40]}

    def _construct_model(self):
        def graphconv_builder(graph_conv_layers, dense_layer_size, dropout, learning_rate, task_type, batch_size=256,
                              epochs=100):
            graph = GraphConvModel(n_tasks=1, graph_conv_layers=graph_conv_layers, dense_layer_size=dense_layer_size,
                                   dropout=dropout, batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
            return DeepChemModel(graph, None, epochs=epochs, use_weights=False)

        return graphconv_builder


class TextCNN(Model):

    def __init__(self, char_dict, length):
        self.char_dict = char_dict
        self.length = length
        model_type = "deepchem"
        super().__init__(model_type)

    def _save(self, output_path):
        self.model.model.model.save(output_path)

    def load(self, file_path, **kwargs):
        tensorflow_model = tf.keras.models.load_model(file_path,
                                                      custom_objects={"DTNNEmbedding": DTNNEmbedding,
                                                                      "Highway": Highway})

        model = TextCNNModel(n_tasks=1, char_dict=self.char_dict, seq_length=self.length)
        model.model = tensorflow_model

        self.model = DeepChemModel(model)

    def save_input_params(self, output_folder_path):
        to_export = {"char_dict": self.char_dict,
                     "length": self.length}

        out_file = open(os.path.join(output_folder_path, "input_params.json"), "w")
        json.dump(to_export, out_file)
        out_file.close()

    def construct_grid(self):
        mode = 'classification'
        self.hyperparameters_grid = {'n_embedding': [75, 32, 64],
                                     'kernel_sizes': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                                                      # DeepChem default. Their code says " Multiple convolutional layers with different filter widths", so I'm not repeating kernel_sizes
                                                      [1, 2, 3, 4, 5, 7, 10, 15],
                                                      [3, 4, 5, 7, 10, 15],
                                                      [3, 4, 5, 7, 10],
                                                      [3, 4, 5, 7],
                                                      [3, 4, 5],
                                                      [3, 5, 7]],
                                     'num_filters': [[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160],
                                                     # DeepChem default
                                                     [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128],
                                                     [128, 128, 128, 128, 64, 64, 64, 64, 32, 32, 32, 32]],
                                     'dropout': [0.0, 0.25, 0.5], 'learning_rate': [1e-4, 1e-3, 1e-2],
                                     'char_dict': [self.char_dict], 'seq_length': [self.length],

                                     'task_type': [mode],
                                     'epochs': [40]}

    def _construct_model(self):
        def textcnn_builder(char_dict, seq_length, n_embedding, kernel_sizes, num_filters, dropout, learning_rate,
                            task_type,
                            batch_size=256, epochs=100):
            textcnn = TextCNNModel(n_tasks=1, char_dict=char_dict, seq_length=seq_length, n_embedding=n_embedding,
                                   kernel_sizes=kernel_sizes, num_filters=num_filters, dropout=dropout,
                                   batch_size=batch_size, learning_rate=learning_rate, mode=task_type)
            return DeepChemModel(textcnn, epochs=epochs, use_weights=False, model_dir=None)

        return textcnn_builder


class BiLSTM(Model):

    def __init__(self, train_dataset, construct_grid=True):
        model_type = "deepchem"
        self.train_dataset = train_dataset
        super().__init__(model_type, construct_grid)

    def _save(self, output_path):
        self.model.model.model.save(output_path)

    def load(self, file_path, **kwargs):
        model = tf.keras.models.load_model(file_path)

        from keras.wrappers.scikit_learn import KerasClassifier
        from models.kerasModels import KerasModel

        keras_model = KerasClassifier(self.model_construction_function)
        keras_model.model = model
        self.model = KerasModel(self.model_construction_function)
        self.model.model = keras_model

    def save_input_params(self, output_folder_path):
        pass

    def construct_grid(self):

        if self.train_dataset.X is None:
            raise Exception("Pre-condition violated: Please call the RNNFeatureConstructor")

        self.hyperparameters_grid = {

            "LSTM_layers": [[256], [256, 256], [256, 256, 256], [128], [128, 128], [128, 128, 128],
                            [64], [64, 64], [64, 64, 64], [256, 128], [128, 64], [256, 64], [256, 128, 64]],

            "dense_layers": [[32], [16], [8], [32, 16], [16, 8], [32, 16, 8]],
            'learning_rate_value': [1e-4, 1e-3, 1e-2],
            'dropout': [0.0, 0.25, 0.5],
            "optimizer_name": ["Adam", "Adagrad", "Adamax", "Adadelta"],
            "input_dim1": [self.train_dataset.X.shape[1]],
            "input_dim2": [self.train_dataset.X.shape[2]],
            "epochs": [40]

        }

    @staticmethod
    def construct_layers(model, input_dim1, input_dim2, LSTM_layers, dense_layers, dropout):
        for i, layer in enumerate(LSTM_layers):
            if i == 0:
                model.add(
                    Bidirectional(CuDNNLSTM(layer, input_shape=(input_dim1, input_dim2), return_sequences=True)))
            elif i == len(LSTM_layers) - 1:
                model.add(Bidirectional(CuDNNLSTM(layer, return_sequences=False)))
            else:
                model.add(Bidirectional(CuDNNLSTM(layer, return_sequences=True)))

            model.add(Dropout(dropout))

        for i, dense_layer in enumerate(dense_layers):
            model.add(Dense(dense_layer, activation='relu'))

        return model

    @staticmethod
    def define_optimizer(model, optimizer_name, learning_rate_value):

        optimizer = None
        if optimizer_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_value,
                name=optimizer_name)

        elif optimizer_name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate_value, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad'
            )

        elif optimizer_name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate=learning_rate_value, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Adamax'
            )
        elif optimizer_name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate_value, rho=0.95, epsilon=1e-07, name='Adadelta'
            )

        if optimizer is not None:

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        else:
            raise Exception(
                "Please choose an optimizer from the following options: Adadelta, Adamax, Adagrad, Adam")

    def _construct_model(self):
        def bi_lstm_builder(dropout, input_dim1, input_dim2,
                            optimizer_name, learning_rate_value, LSTM_layers=None,
                            dense_layers=None):
            if dense_layers is None:
                dense_layers = [32]
            if LSTM_layers is None:
                LSTM_layers = [256]

            model = Sequential()

            model = self.construct_layers(model, input_dim1, input_dim2, LSTM_layers, dense_layers, dropout)

            model.add(Dense(1, activation='sigmoid'))

            model = self.define_optimizer(model, optimizer_name, learning_rate_value)

            return model

        return bi_lstm_builder


class LSTM(Model):

    def save_input_params(self, output_path):
        pass

    def __init__(self, train_dataset, construct_grid=True):
        model_type = "deepchem"
        self.train_dataset = train_dataset
        super().__init__(model_type, construct_grid)

    def _save(self, output_path):
        self.model.model.model.save(output_path)

    def load(self, file_path, **kwargs):
        model = tf.keras.models.load_model(file_path)

        from keras.wrappers.scikit_learn import KerasClassifier
        from models.kerasModels import KerasModel

        keras_model = KerasClassifier(self.model_construction_function)
        keras_model.model = model
        self.model = KerasModel(self.model_construction_function)
        self.model.model = keras_model

    def construct_grid(self):
        self.hyperparameters_grid = {
            "LSTM_layers": [[256], [256, 256], [256, 256, 256], [128], [128, 128], [128, 128, 128],
                            [64], [64, 64], [64, 64, 64], [256, 128], [128, 64], [256, 64], [256, 128, 64]],
            "dense_layers": [[32], [16], [8], [32, 16], [16, 8], [32, 16, 8]],
            'learning_rate_value': [1e-4, 1e-3, 1e-2],
            'dropout': [0.0, 0.25, 0.5],
            "optimizer_name": ["Adam", "Adagrad", "Adamax", "Adadelta"],
            "input_dim1": [self.train_dataset.X.shape[1]],
            "input_dim2": [self.train_dataset.X.shape[2]],
            "epochs": [40]}

    @staticmethod
    def construct_layers(model, input_dim1, input_dim2, LSTM_layers, dense_layers, dropout):
        for i, layer in enumerate(LSTM_layers):
            if i == 0:
                model.add(
                    CuDNNLSTM(layer, input_shape=(input_dim1, input_dim2), return_sequences=True))
            elif i == len(LSTM_layers) - 1:
                model.add(CuDNNLSTM(layer, return_sequences=False))
            else:
                model.add(CuDNNLSTM(layer, return_sequences=True))

            model.add(Dropout(dropout))

        for i, dense_layer in enumerate(dense_layers):
            model.add(Dense(dense_layer, activation='relu'))

        return model

    @staticmethod
    def define_optimizer(model, optimizer_name, learning_rate_value):

        optimizer = None
        if optimizer_name == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_value,
                name=optimizer_name)

        elif optimizer_name == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(
                learning_rate=learning_rate_value, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad'
            )

        elif optimizer_name == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate=learning_rate_value, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Adamax'
            )
        elif optimizer_name == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=learning_rate_value, rho=0.95, epsilon=1e-07, name='Adadelta'
            )

        if optimizer is not None:

            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])

        else:
            raise Exception(
                "Please choose an optimizer from the following options: Adadelta, Adamax, Adagrad, Adam")

    def _construct_model(self):
        def lstm_builder(dropout, input_dim1, input_dim2,
                         optimizer_name, learning_rate_value, LSTM_layers=None,
                         dense_layers=None):
            if dense_layers is None:
                dense_layers = [32]
            if LSTM_layers is None:
                LSTM_layers = [256]

            model = Sequential()

            model = self.construct_layers(model, input_dim1, input_dim2, LSTM_layers, dense_layers, dropout)

            model.add(Dense(1, activation='sigmoid'))

            model = self.define_optimizer(model, optimizer_name, learning_rate_value)

            return model

        return lstm_builder
