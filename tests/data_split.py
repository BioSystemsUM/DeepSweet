from unittest import TestCase

from split_dataset import DatasetSplitter


class TestSplits(TestCase):

    def setUp(self) -> None:
        self.splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", "mols", "y", 0.5,
                                        "../resources/test_data/")

    def test_splitter(self):
        self.splitter.split()
