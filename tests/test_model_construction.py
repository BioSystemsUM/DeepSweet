from unittest import TestCase

from model_construction import SVM


class TestModelConstruction(TestCase):

    def test_svm(self):
        model = SVM()
        model.define_model_hyperparameters()
        model.save("../resources/test_data/2d/svm_model")

    def test_load_svm(self):
        model = SVM()
        model.load("../resources/test_data/2d/svm_model")

