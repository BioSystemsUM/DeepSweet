import os

from compoundFeaturization import deepChemFeaturizers
from deepchem.models import TextCNNModel

from deepsweet_utils import IO
from generate_features_ml import FeaturesGenerator
from generate_features_rnn import RNNFeatureGenerator
from hyperparameter_optimisation import SklearnKerasHyperparameterOptimiser, EndToEndHyperparameterOptimiser
from model_construction import RF, SVM, DNN, BiLSTM, GCN, GAT, TextCNN, GraphConv, LSTM
from pipelines import Pipeline
from select_features import FeatureSelector
from split_dataset import DatasetSplitter


def run_splitters(dataset_path, output_folder_path):
    new_pipeline = Pipeline()

    splitter = DatasetSplitter(dataset_path, "mols", "y", 0.5,
                               output_folder_path)
    new_pipeline.register(splitter)
    new_pipeline.run()


def run_ml_pipeline(pipeline_folder):
    new_pipeline = Pipeline()

    generator = FeaturesGenerator("../resources/test_data/")
    new_pipeline.register(generator)

    selector = FeatureSelector([os.path.join(pipeline_folder, "2d"),
                                os.path.join(pipeline_folder, "atompair_fp"),
                                os.path.join(pipeline_folder, "ecfp4"),
                                os.path.join(pipeline_folder, "ecfp8"),
                                os.path.join(pipeline_folder, "rdk")])

    new_pipeline.register(selector)

    feature_selection_methods = ["all", "Boruta", "SelectFromModelFS", "KbestFS"]
    ml_features = ["2d", "atompair_fp", "ecfp4", "ecfp8", "rdk"]
    new_pipeline = Pipeline()
    for ml_feature in ml_features:
        for feature_selection_method in feature_selection_methods:

            if feature_selection_method == "all":
                dataset_path = os.path.join(pipeline_folder, f"{ml_feature}/train_dataset.csv")
            else:
                dataset_path = os.path.join(pipeline_folder,
                                            f"{ml_feature}/train_dataset_{feature_selection_method}.csv")

            dataset = IO.load_dataset_with_features(dataset_path)

            rf = RF()
            optimiser = SklearnKerasHyperparameterOptimiser(rf, dataset, 3,
                                                            "roc_auc",
                                                            1,
                                                            123,
                                                            os.path.join(pipeline_folder, f"{ml_feature}/"),
                                                            f"{feature_selection_method}_rf_model")

            new_pipeline.register(optimiser)

            svm = SVM()
            optimiser = SklearnKerasHyperparameterOptimiser(svm, dataset, 3,
                                                            "roc_auc",
                                                            1,
                                                            123,
                                                            os.path.join(pipeline_folder, f"{ml_feature}/"),
                                                            f"{feature_selection_method}_svm_model")

            new_pipeline.register(optimiser)

            dnn = DNN(train_dataset=dataset)
            optimiser = SklearnKerasHyperparameterOptimiser(dnn, dataset, 3,
                                                            "roc_auc",
                                                            1,
                                                            123,
                                                            os.path.join(pipeline_folder, f"{ml_feature}/"),
                                                            f"{feature_selection_method}_dnn_model.h5")

            new_pipeline.register(optimiser)

    new_pipeline.run()


def run_dl_pipeline(pipeline_folder):
    pipeline = Pipeline()

    train_dataset = IO.load_dataset(os.path.join(pipeline_folder, "train_dataset.csv"))
    test_dataset = IO.load_dataset(os.path.join(pipeline_folder, "test_dataset.csv"))

    gat = GAT()
    featurizer = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
    optimiser_gat = EndToEndHyperparameterOptimiser(gat, train_dataset, 10,
                                                    "roc_auc",
                                                    30,
                                                    123,
                                                    featurizer,
                                                    os.path.join(pipeline_folder, "GAT/"),
                                                    "GAT.h5")

    gcn = GCN()
    optimiser_gcn = EndToEndHyperparameterOptimiser(gcn, train_dataset, 10,
                                                    "roc_auc",
                                                    30,
                                                    123,
                                                    featurizer,
                                                    os.path.join(pipeline_folder, "GCN/"),
                                                    "GCN.h5")
    pipeline.register(optimiser_gat)
    pipeline.register(optimiser_gcn)

    full_dataset = train_dataset.merge([test_dataset])
    full_dataset.ids = full_dataset.mols
    char_dict, length = TextCNNModel.build_char_dict(full_dataset)
    text_cnn = TextCNN(char_dict, length)

    train_dataset = IO.load_dataset(os.path.join(pipeline_folder, "train_dataset.csv"))
    featurizer = deepChemFeaturizers.RawFeat()
    optimiser_textcnn = EndToEndHyperparameterOptimiser(text_cnn, train_dataset, 10,
                                                        "roc_auc",
                                                        30,
                                                        123,
                                                        featurizer,
                                                        os.path.join(pipeline_folder, "TextCNN/"),
                                                        "TextCNN.h5")

    pipeline.register(optimiser_textcnn)

    graph_conv = GraphConv(train_dataset)
    train_dataset = IO.load_dataset("../resources/models/train_dataset.csv")
    featurizer = deepChemFeaturizers.ConvMolFeat()
    optimiser_graphconv = EndToEndHyperparameterOptimiser(graph_conv, train_dataset, 10,
                                                          "roc_auc",
                                                          30,
                                                          123,
                                                          featurizer,
                                                          os.path.join(pipeline_folder, "GraphConv/"),
                                                          "GraphConv.h5")
    pipeline.register(optimiser_graphconv)

    train_dataset = IO.load_dataset(os.path.join(pipeline_folder, "train_dataset.csv"))
    test_dataset = IO.load_dataset(os.path.join(pipeline_folder, "test_dataset.csv"))
    full_dataset = train_dataset.merge([test_dataset])
    featurizer = RNNFeatureGenerator(full_dataset)
    featurizer.featurize(train_dataset)
    featurizer.save_input_params(os.path.join(pipeline_folder, "BiLSTM/"))
    featurizer.save_input_params(os.path.join(pipeline_folder, "LSTM/"))

    bilstm = BiLSTM(train_dataset)
    lstm = LSTM(train_dataset)
    optimiser_bilstm = EndToEndHyperparameterOptimiser(bilstm, train_dataset, 10,
                                                       "roc_auc",
                                                       30,
                                                       123,
                                                       featurizer,
                                                       os.path.join(pipeline_folder, "BiLSTM/"),
                                                       "BiLSTM.h5")

    optimiser_lstm = EndToEndHyperparameterOptimiser(lstm, train_dataset, 10,
                                                     "roc_auc",
                                                     30,
                                                     123,
                                                     featurizer,
                                                     os.path.join(pipeline_folder, "LSTM/"),
                                                     "LSTM.h5")

    pipeline.register(optimiser_bilstm)
    pipeline.register(optimiser_lstm)

    pipeline.run()


