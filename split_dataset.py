
class DatasetSplitter():

    def __init__(self):
        self.splitter = SingletaskStratifiedSplitter()

    def split(self, dataset_path, smiles_field, y_label_field, train_percentage):
        seed_value = 123

        loader = CSVLoader(dataset_path,
                           mols_field=smiles_field,
                           labels_fields=y_label_field)

        dataset = loader.create_dataset()
        train_dataset, test_dataset = self.splitter.train_test_split(dataset,
                                                                     frac_train=train_percentage,
                                                                     seed=seed_value)

        train_dataset.save_to_csv("train_dataset.csv")
        test_dataset.save_to_csv("test_dataset.csv")


