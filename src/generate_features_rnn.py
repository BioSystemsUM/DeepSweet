from SmilesPE.pretokenizer import atomwise_tokenizer
import numpy as np


class RNNFeatureGenerator:

    def __init__(self, full_dataset):
        super().__init__()
        text = ''.join(full_dataset.mols)
        self.unique_chars = sorted(list(set(atomwise_tokenizer(text))))
        self.char_to_int = dict((c, i) for i, c in enumerate(self.unique_chars))
        self.int_to_char = dict((i, c) for i, c in enumerate(self.unique_chars))

        lens_smiles = []
        for index, mol in enumerate(full_dataset.mols):
            lens_smiles.append(len(atomwise_tokenizer(mol)))

        perc_of_instances = 0.999

        lens_smiles.sort()
        print('lenght of metabolite SMILES that explain {:.0%} of the instances:'.format(perc_of_instances),
              lens_smiles[int(len(lens_smiles) * perc_of_instances) - 1])

        self.max_len = lens_smiles[int(len(lens_smiles) * perc_of_instances) - 1]

    def smiles_encoder(self, smiles, maxlen):
        X = np.zeros((maxlen, len(self.unique_chars)))
        encoding = atomwise_tokenizer(smiles)
        if maxlen >= len(encoding) > 10:
            for i, c in enumerate(encoding):
                if c in self.char_to_int:
                    X[i, self.char_to_int[c]] = 1
                else:
                    return None
            return X
        return None

    def smiles_decoder(self, X):
        smi = ''
        X = X.argmax(axis=-1)
        for n, i in enumerate(X):
            if sum(X[n:]) == 0:
                break
            smi += self.int_to_char[i]
        return smi

    @staticmethod
    def balance_dataset(dataset):
        if dataset.y[dataset.y == 1].shape[0] < dataset.y[dataset.y == 0].shape[0]:
            ids_0 = dataset.ids[dataset.y == 0]
            n_1 = int(dataset.y[dataset.y == 1].shape[0])
            indexes = np.random.choice(ids_0, n_1, replace=False)
            indexes = list(indexes)
            indexes.extend(list(dataset.ids[dataset.y == 1]))
            dataset.select(indexes)
            print("Dataset balanced: %d positive values and %d negative ones" % (
                dataset.y[dataset.y == 1].shape[0], dataset.y[dataset.y == 0].shape[0]))

        elif dataset.y[dataset.y == 1].shape[0] > dataset.y[dataset.y == 0].shape[0]:
            ids_1 = dataset.ids[dataset.y == 1]
            n_0 = int(dataset.y[dataset.y == 0].shape[0])
            indexes = np.random.choice(ids_1, n_0, replace=False)
            indexes = list(indexes)
            indexes.extend(list(dataset.ids[dataset.y == 0]))
            dataset.select(indexes)

            print("Dataset balanced: %d positive values and %d negative ones" % (
                dataset.y[dataset.y == 1].shape[0], dataset.y[dataset.y == 0].shape[0]))
        return dataset

    def generate_dataset(self, dataset, balance=False):

        def encode_smiles(smiles, max_len):
            encoding = self.smiles_encoder(smiles, max_len)
            if encoding is not None:
                return encoding
            return None

        keep = []

        for i, x in enumerate(dataset.mols):
            encoding = encode_smiles(x, self.max_len)

            if encoding is not None:
                keep.append(dataset.ids[i])

        dataset.select(keep)
        size = len(dataset.y)
        print("encoding smiles")
        X = []
        for i, mol in enumerate(dataset.mols):
            smiles_encoding = encode_smiles(mol, self.max_len)
            X.append(smiles_encoding)
        X = np.array(X)
        print(X.shape)

        X = np.concatenate(X, axis=0)
        X = np.reshape(X, (size, self.max_len, len(self.unique_chars)))

        dataset.X = X
        if balance:
            print("balancing dataset")
            dataset = self.balance_dataset(dataset)

        return dataset

    def featurize(self, train_dataset):

        self.generate_dataset(train_dataset, True)
        return train_dataset
