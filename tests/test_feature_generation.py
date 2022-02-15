from unittest import TestCase

from generate_features_ml import FeaturesGenerator


class TestFeatureGeneration(TestCase):

    def setUp(self) -> None:
        self.generator = FeaturesGenerator("../resources/test_data/")

    def test_feature_generator(self):
        self.generator.featurize_all()