from loaders.Loaders import CSVLoader
from splitters.splitters import SingletaskStratifiedSplitter

from pipelines import Step


class DatasetSplitter(Step):

    def __init__(self, dataset_path, smiles_field, y_label_field, train_percentage):
        super().__init__()
        self.dataset_path = dataset_path
        self.smiles_field = smiles_field
        self.y_label_field = y_label_field
        self.train_percentage = train_percentage
        self.splitter = SingletaskStratifiedSplitter()

    def split(self):
        seed_value = 123

        loader = CSVLoader(self.dataset_path,
                           mols_field=self.smiles_field,
                           labels_fields=self.y_label_field)

        dataset = loader.create_dataset()
        train_dataset, test_dataset = self.splitter.train_test_split(dataset,
                                                                     frac_train=self.train_percentage,
                                                                     seed=seed_value)

        train_dataset.save_to_csv("train_dataset.csv")
        test_dataset.save_to_csv("test_dataset.csv")

    def run(self):
        self.split()





