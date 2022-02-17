from unittest import TestCase

from deepsweet_utils import IO
from generate_features_ml import FeaturesGenerator
from hyperparameter_optimisation import SklearnKerasHyperparameterOptimiser
from model_construction import RF, SVM, DNN
from pipelines import Pipeline
from select_features import FeatureSelector
from split_dataset import DatasetSplitter


class TestRunPipeline(TestCase):

    def test_splitter_pipeline(self):
        new_pipeline = Pipeline()

        splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", "mols", "y", 0.5,
                                   "../resources/test_data/")
        new_pipeline.register(splitter)
        new_pipeline.run()

    def test_splitter_feature_generation(self):
        new_pipeline = Pipeline()

        splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", "mols", "y", 0.5,
                                   "../resources/test_data/")
        new_pipeline.register(splitter)

        generator = FeaturesGenerator("../resources/test_data/")
        new_pipeline.register(generator)

        new_pipeline.run()

    def test_splitter_feature_generation_selection(self):
        new_pipeline = Pipeline()

        splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", "mols", "y", 0.5,
                                   "../resources/test_data/")
        new_pipeline.register(splitter)

        generator = FeaturesGenerator("../resources/test_data/")
        new_pipeline.register(generator)

        selector = FeatureSelector(["../resources/test_data/2d",
                                    "../resources/test_data/atompair_fp",
                                    "../resources/test_data/ecfp4",
                                    "../resources/test_data/ecfp8",
                                    "../resources/test_data/rdk"])

        new_pipeline.register(selector)
        new_pipeline.run()

    def test_hyperparameter_optimization(self):
        dataset_path = "../resources/test_data/2d/train_dataset_Boruta.csv"
        new_pipeline = Pipeline()

        dataset = IO.load_dataset_with_features(dataset_path)

        rf = RF()
        optimiser = SklearnKerasHyperparameterOptimiser(rf, dataset, 3,
                                                        "roc_auc",
                                                        1,
                                                        123,
                                                        "../resources/test_data/2d/",
                                                        "Boruta_rf_model")

        new_pipeline.register(optimiser)
        new_pipeline.run()

    def test_run_the_whole_features_pipeline_at_once(self):
        # new_pipeline = Pipeline()
        #
        # splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", "mols", "y", 0.5,
        #                            "../resources/test_data/")
        # new_pipeline.register(splitter)
        #
        # generator = FeaturesGenerator("../resources/test_data/")
        # new_pipeline.register(generator)
        #
        # selector = FeatureSelector(["../resources/test_data/2d",
        #                             "../resources/test_data/atompair_fp",
        #                             "../resources/test_data/ecfp4",
        #                             "../resources/test_data/ecfp8",
        #                             "../resources/test_data/rdk"])
        # new_pipeline.register(selector)

        feature_selection_methods = ["Boruta", "SelectFromModelFS", "KbestFS"]
        ml_features = ["2d", "atompair_fp", "ecfp4", "ecfp8", "rdk"]
        new_pipeline = Pipeline()
        for ml_feature in ml_features:
            for feature_selection_method in feature_selection_methods:
                dataset_path = f"../resources/test_data/{ml_feature}/train_dataset_{feature_selection_method}.csv"
                dataset = IO.load_dataset_with_features(dataset_path)

                rf = RF()
                optimiser = SklearnKerasHyperparameterOptimiser(rf, dataset, 3,
                                                                "roc_auc",
                                                                1,
                                                                123,
                                                                f"../resources/test_data/{ml_feature}/",
                                                                f"{feature_selection_method}_rf_model")

                new_pipeline.register(optimiser)

                svm = SVM()
                optimiser = SklearnKerasHyperparameterOptimiser(svm, dataset, 3,
                                                                "roc_auc",
                                                                1,
                                                                123,
                                                                f"../resources/test_data/{ml_feature}/",
                                                                f"{feature_selection_method}_svm_model")

                new_pipeline.register(optimiser)

                dnn = DNN(train_dataset=dataset)
                optimiser = SklearnKerasHyperparameterOptimiser(dnn, dataset, 3,
                                                                "roc_auc",
                                                                1,
                                                                123,
                                                                f"../resources/test_data/{ml_feature}/",
                                                                f"{feature_selection_method}_dnn_model.h5")

                new_pipeline.register(optimiser)

        new_pipeline.run()
