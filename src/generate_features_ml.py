import os
import numpy as np
from compoundFeaturization.rdkitDescriptors import TwoDimensionDescriptors
from compoundFeaturization.rdkitFingerprints import MorganFingerprint, RDKFingerprint, AtomPairFingerprint
from loaders.Loaders import CSVLoader
from scalers.sklearnScalers import MinMaxScaler

from pipelines import Step


class FeaturesGenerator(Step):

    def __init__(self, models_folder_path):
        self.models_folder_path = models_folder_path
        super().__init__()

    @staticmethod
    def balance_dataset(dataset):
        if dataset.y[dataset.y == 1].shape[0] < dataset.y[dataset.y == 0].shape[0]:
            indexes = np.random.choice(dataset.ids[dataset.y == 0], int(dataset.y[dataset.y == 1].shape[0]),
                                       replace=False)
            indexes = list(indexes)
            indexes.extend(list(dataset.ids[dataset.y == 1]))
            dataset.select(indexes)
            print("Dataset balanced: %d positive values and %d negative ones" % (
                dataset.y[dataset.y == 1].shape[0], dataset.y[dataset.y == 0].shape[0]))

        elif dataset.y[dataset.y == 1].shape[0] > dataset.y[dataset.y == 0].shape[0]:
            indexes = np.random.choice(dataset.ids[dataset.y == 1], int(dataset.y[dataset.y == 0].shape[0]),
                                       replace=False)
            indexes = list(indexes)
            indexes.extend(list(dataset.ids[dataset.y == 0]))
            dataset.select(indexes)

            print("Dataset balanced: %d positive values and %d negative ones" % (
                dataset.y[dataset.y == 1].shape[0], dataset.y[dataset.y == 0].shape[0]))
        return dataset

    @staticmethod
    def featurize_test_set(folder_path, folder_path_to_save, featurize_method, columns_to_scale=None, axis=0):
        loader = CSVLoader(os.path.join(folder_path, "%s_dataset.csv" % "test"),
                           mols_field='mols',
                           labels_fields='y')

        dataset = loader.create_dataset()

        featurize_method.featurize(dataset, remove_nans_axis=axis)

        if columns_to_scale is not None:
            scaler = MinMaxScaler(clip=True)
            scaler.load_scaler(os.path.join(folder_path, folder_path_to_save, "scaler"))
            scaler.scaler_object.clip = True
            scaler.transform(dataset, columns_to_scale)

        os.makedirs(os.path.join(folder_path, folder_path_to_save), exist_ok=True)

        dataset.save_to_csv(path=os.path.join(folder_path, folder_path_to_save, "%s_dataset.csv" % "test"))

    def featurize_dataset(self, folder_path, folder_path_to_save, featurize_method, columns_to_scale=None, axis=0):

        datasets = ["test"]

        loader = CSVLoader(os.path.join(folder_path, "%s_dataset.csv" % "train"),
                           mols_field='mols',
                           labels_fields='y')

        train_dataset = loader.create_dataset()

        os.makedirs(os.path.join(folder_path, folder_path_to_save), exist_ok=True)

        featurize_method.featurize(train_dataset)
        train_dataset.remove_duplicates()

        train_dataset = self.balance_dataset(train_dataset)

        if columns_to_scale is not None:
            scaler = MinMaxScaler(clip=True)
            scaler.fit(train_dataset, columns=columns_to_scale)
            scaler.transform(train_dataset, columns=columns_to_scale)
            scaler.save_scaler(os.path.join(folder_path, folder_path_to_save, "scaler"))

        for dataset_type in datasets:
            loader = CSVLoader(os.path.join(folder_path, "%s_dataset.csv" % dataset_type),
                               mols_field='mols',
                               labels_fields='y')

            dataset = loader.create_dataset()

            featurize_method.featurize(dataset, remove_nans_axis=axis)

            if columns_to_scale is not None:
                scaler.transform(dataset, columns_to_scale)

            os.makedirs(os.path.join(folder_path, folder_path_to_save), exist_ok=True)

            dataset.save_to_csv(path=os.path.join(folder_path, folder_path_to_save, "%s_dataset.csv" % dataset_type))

        train_dataset.save_to_csv(path=os.path.join(folder_path, folder_path_to_save, "train_dataset.csv"))

    def featurize_all(self):

        descriptors = TwoDimensionDescriptors()
        self.featurize_dataset(self.models_folder_path, "2d",
                               descriptors, columns_to_scale=[i for i in range(208)])

        morgan = MorganFingerprint(chiral=True)
        self.featurize_dataset(self.models_folder_path, "ecfp4", morgan)

        morgan = MorganFingerprint(radius=4, chiral=True)
        self.featurize_dataset(self.models_folder_path, "ecfp8", morgan)

        rdk = RDKFingerprint()
        self.featurize_dataset(self.models_folder_path, "rdk", rdk)

        featurize_method = AtomPairFingerprint(nBits=2048, includeChirality=True)
        self.featurize_dataset(self.models_folder_path, "atompair_fp", featurize_method)

    def featurize_test_set_all(self):
        descriptors = TwoDimensionDescriptors()
        self.featurize_test_set(self.models_folder_path, "2d",
                                descriptors, columns_to_scale=[i for i in range(208)])

        morgan = MorganFingerprint(chiral=True)
        self.featurize_test_set(self.models_folder_path, "ecfp4", morgan)

        morgan = MorganFingerprint(radius=4, chiral=True)
        self.featurize_test_set(self.models_folder_path, "ecfp8", morgan)

        rdk = RDKFingerprint()
        self.featurize_test_set(self.models_folder_path, "rdk", rdk)

        featurize_method = AtomPairFingerprint(nBits=2048, includeChirality=True)
        self.featurize_test_set(self.models_folder_path, "atompair_fp", featurize_method)

    def run(self):
        self.featurize_all()
