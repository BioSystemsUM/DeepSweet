from unittest import TestCase

from deepsweet_models import DeepSweetRF, DeepSweetDNN, DeepSweetTextCNN
from ensemble import Ensemble


class TestEnsemble(TestCase):

    def test_ensemble(self):
        models_folder_path = "../resources/models/"
        molecules = ["CN1CCC[C@H]1C2=CN=CC=C2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

        list_of_models = []
        list_of_models.append(DeepSweetRF(models_folder_path, "2d", "SelectFromModelFS"))
        list_of_models.append(DeepSweetDNN(models_folder_path, "rdk", "all"))

        ensemble = Ensemble(list_of_models, models_folder_path)

        predictions, dataset, not_converted_molecules = ensemble.predict(molecules)