import json
import os
from abc import ABC, abstractmethod

from pandas import DataFrame
from parameterOptimization.HyperparameterOpt import HyperparamOpt_CV

from pipelines import Step


class HyperparameterOptimiser(Step, ABC):

    def __init__(self, model, train_dataset, cv, opt_metric, n_iter_search, seed, folder_path,
                 model_output_path):
        super().__init__()
        self.model_wrapper = model
        self.cv = cv
        self.opt_metric = opt_metric
        self.n_iter = n_iter_search
        self.seed = seed
        self.train_dataset = train_dataset
        self.folder_path = folder_path
        self.model_output_path = model_output_path
        self.all_results = None
        self.best_hyperparams = None

    def save_results(self):
        result_objects = [self.model_wrapper.model, self.all_results, self.best_hyperparams]
        if all([result_object is not None for result_object in result_objects]):
            self.model_wrapper.save(os.path.join(self.folder_path, self.model_output_path))

            hyperparameters_file = self.model_output_path.replace(".h5", "")
            out_file = open(os.path.join(self.folder_path, f"{hyperparameters_file}_hyperparameters.json"), "w")
            json.dump(self.best_hyperparams, out_file)
            out_file.close()

            new_df = DataFrame(self.all_results)
            new_df.to_csv(os.path.join(self.folder_path, f"hyperparameter_opt_{hyperparameters_file}_model_results"),
                          index=False)

        else:
            raise Exception("Please optimise the model first")

    @abstractmethod
    def optimise(self):
        raise NotImplementedError


class SklearnKerasHyperparameterOptimiser(HyperparameterOptimiser):

    def __init__(self, model,
                 train_dataset,
                 cv,
                 opt_metric,
                 n_iter_search,
                 seed,
                 folder_path,
                 model_output_path):
        self.train_dataset = train_dataset
        super().__init__(model, train_dataset, cv, opt_metric, n_iter_search, seed, folder_path,
                         model_output_path)

    def optimise(self):
        optimizer = HyperparamOpt_CV(self.model_wrapper.model_construction_function)

        self.model_wrapper.model, self.best_hyperparams, self.all_results = \
            optimizer.hyperparam_search(self.model_wrapper.model_type,
                                        self.model_wrapper.hyperparameters_grid,
                                        self.train_dataset,
                                        self.opt_metric,
                                        cv=self.cv,
                                        n_iter_search=self.n_iter,
                                        seed=self.seed)

    def run(self):
        self.optimise()
        self.save_results()


class EndToEndHyperparameterOptimiser(HyperparameterOptimiser, ABC):

    def __init__(self, model, train_dataset, cv, opt_metric, n_iter_search, seed, featurization_method, folder_path,
                 model_output_path):
        self.featurization_method = featurization_method
        os.makedirs(folder_path, exist_ok=True)

        super().__init__(model, train_dataset, cv, opt_metric, n_iter_search, seed, folder_path,
                         model_output_path)

    def optimise(self):
        self.featurization_method.featurize(self.train_dataset)

        self.train_dataset.remove_duplicates()

        optimizer = HyperparamOpt_CV(self.model_wrapper.model_construction_function)

        self.model_wrapper.model, self.best_hyperparams, self.all_results = \
            optimizer.hyperparam_search(self.model_wrapper.model_type,
                                        self.model_wrapper.hyperparameters_grid,
                                        self.train_dataset,
                                        self.opt_metric,
                                        cv=self.cv,
                                        n_iter_search=self.n_iter,
                                        seed=self.seed)

    def run(self):
        self.optimise()
        self.save_results()
