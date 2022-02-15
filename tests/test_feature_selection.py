from unittest import TestCase

from select_features import FeatureSelector


class TestFeatureSelector(TestCase):

    def setUp(self) -> None:
        self.generator = FeatureSelector(["../resources/test_data/2d"])

    def test_fs(self):
        self.generator.select_with_all_methods()