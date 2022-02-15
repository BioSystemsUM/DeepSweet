from unittest import TestCase

from loaders.Loaders import CSVLoader

from hyperparameter_optimisation import SklearnKerasHyperparameterOptimiser
import pandas

from model_construction import RF


class TestHyperparameterOptimization(TestCase):

    def setUp(self) -> None:
        dataset_path = "../resources/test_data/2d/train_dataset_Boruta.csv"
        pandas_dset = pandas.read_csv(dataset_path)
        columns = pandas_dset.columns[3:]

        loader = CSVLoader(dataset_path,
                           features_fields=list(columns),
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()

        rf = RF()
        self.optimiser = SklearnKerasHyperparameterOptimiser(rf, dataset, 3,
                                                             "roc_auc",
                                                             1,
                                                             123,
                                                             "../resources/test_data/2d/",
                                                             "Boruta_rf_model")

    def test_fs(self):
        self.optimiser.optimise()
        self.optimiser.save_results()