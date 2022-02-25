# DeepSweet

### Table of contents:

- [Requirements](#requirements)
- [Setup GPU](#setup-gpu)
- [Prepare the dataset](#prepare-dataset)
- [Split dataset](#split-dataset)
- [Generate features](#generate-features)
- [Select features](#select-features)
- [Hyperparameter optimization](#optimize-hyperparameters)
- [Generate results](#generate-results)
- [Analyse results](#analyse-results)
- [Predict with built models](#predict-with-built-models)
- [Predict with ensemble](#predict-with-ensemble)
- [Feature explainability](#feature-explainability)
- [Repurposing PubChem molecules](#repurposing-pubchem-molecules)


## Requirements

- Anaconda or miniconda

Create a conda environment with python 3.7

- pip install rdkit-pypi
- pip install --upgrade pip
- conda install -c anaconda cudnn=8.2.1
- pip install tensorflow-gpu==2.6.2
- conda install pytorch=1.10.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
- conda install -y -c dglteam dgl=0.7.2 dgl-cuda11.3 dgllife=0.2.6
- pip install joblib==1.1.0 pillow==8.4.0 h5py==3.1.0 deepchem==2.5.0
- pip install smilespe==0.0.3
- python setup.py install
- cd DeepMol && \
    python setup.py install
- conda install -c anaconda ipython
- conda install shap
- conda install jupyter
- conda install seaborn

## Setup GPU
```python
from deepsweet_utils import DeviceUtils

# set up the gpu
DeviceUtils.gpu_setup("2")
```

## Prepare dataset

Checkout the notebook entitled "dataset_standardization_and_negative_cases_selection.ipynb" in the "notebooks" folder 
where we provide the data analysis made

## Split dataset

```python
from split_dataset import DatasetSplitter

splitter = DatasetSplitter("../resources/data/preprocessed_sweeteners_dataset.csv", 
                           "mols", "y", 0.5,
                            "../resources/test_data/")
splitter.split()
```

## Generate features

```python
from generate_features_ml import FeaturesGenerator

generator = FeaturesGenerator("../resources/test_data/")
generator.featurize_all()
```

## Select features

```python
from select_features import FeatureSelector

selector = FeatureSelector(["../resources/test_data/2d"])
selector.select_with_all_methods()
```

## Optimize hyperparameters

```python
from deepsweet_utils import IO
from hyperparameter_optimisation import SklearnKerasHyperparameterOptimiser, EndToEndHyperparameterOptimiser
from compoundFeaturization import deepChemFeaturizers
from model_construction import DNN, GAT

dataset_path = "../resources/test_data/2d/train_dataset_Boruta.csv"
dataset = IO.load_dataset_with_features(dataset_path)


# Algorithm that uses directly the generated features
dnn = DNN(dataset)
optimiser_RF = SklearnKerasHyperparameterOptimiser(model=DNN, 
                                                   train_dataset=dataset, 
                                                   cv=10,
                                                   opt_metric="roc_auc",
                                                   n_iter_search=30,
                                                   seed=123,
                                                   folder_path="../resources/test_data/2d/",
                                                   model_output_path="Boruta_rf_model")

# End-to-end algorithm example of Graph Attention Network (GAT)
gat = GAT()
featurizer = deepChemFeaturizers.MolGraphConvFeat(use_edges=True)
optimiser_gat = EndToEndHyperparameterOptimiser(model=gat, 
                                                train_dataset=dataset, 
                                                cv=10,
                                                opt_metric="roc_auc",
                                                n_iter_search=30,
                                                seed=123,
                                                featurization_method=featurizer,
                                                folder_path="../resources/test_data/GAT/",
                                                model_output_path="GAT.h5")

optimiser_gat.optimise()
optimiser_gat.save_results()

```

## Run pipeline all at once
```python
from pipelines_and_results import run_ml_pipeline, run_splitters, run_dl_pipeline

# run splitters
run_splitters("resources/data/preprocessed_sweeteners_dataset.csv", "resources/test_data/")

#run ML pipeline
run_ml_pipeline("resources/test_data/")

#run DL pipeline
run_dl_pipeline("resources/test_data/")
```

## Generate results
```python
from reports import ResultsReport

# run splitters
report_generator = ResultsReport("cuda:0", "resources/test_data/train_dataset.csv", 
              "resources/test_data/test_dataset.csv", 
              "resources/data/blend_test_set.csv")

report_generator.run_all_ml("resources/models", "resources/results/ML_DNN_results.csv")
report_generator.run_all_dl("resources/models", "resources/results/EndToEnd_results.csv")
```

## Analyse results

You can checkout the results file in the "resources/results" folder. "all_results.csv" 
enumerates the results for all metrics to all models.

## Predict with built models

```python
from deepsweet_models import DeepSweetDNN, DeepSweetGAT, DeepSweetRF, DeepSweetTextCNN

models_folder_path = "../resources/models/"
molecules = ["CN1CCC[C@H]1C2=CN=CC=C2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

# predict with DNN trained with RDK fingerprints with all features
dnn = DeepSweetDNN(models_folder_path, featurization_method="rdk", feature_selection_method="all")
predictions, dataset = dnn.predict(molecules)

# predict with RF trained with 2D descriptors using Boruta as feature selection method
dnn = DeepSweetRF(models_folder_path, featurization_method="2d", feature_selection_method="Boruta")
predictions2, dataset = dnn.predict(molecules)

# predict with TextCNN
textcnn = DeepSweetTextCNN(models_folder_path)
predictions3, dataset = textcnn.predict(molecules)

# predict with GAT
gat = DeepSweetGAT(models_folder_path)
predictions4, dataset = gat.predict(molecules)
```

## Predict with ensemble
```python
from deepsweet_models import DeepSweetDNN, DeepSweetGCN, DeepSweetRF, DeepSweetSVM, DeepSweetBiLSTM
from ensemble import Ensemble

models_folder_path = "../resources/models/"
molecules = ["CN1CCC[C@H]1C2=CN=CC=C2", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]

list_of_models = []
list_of_models.append(DeepSweetRF(models_folder_path, "2d", "SelectFromModelFS"))
list_of_models.append(DeepSweetDNN(models_folder_path, "rdk", "all"))
list_of_models.append(DeepSweetGCN(models_folder_path))
list_of_models.append(DeepSweetSVM(models_folder_path, "ecfp4", "all"))
list_of_models.append(DeepSweetDNN(models_folder_path, "atompair_fp", "SelectFromModelFS"))
list_of_models.append(DeepSweetBiLSTM(models_folder_path))

ensemble = Ensemble(list_of_models, models_folder_path)

predictions, dataset, not_converted_molecules = ensemble.predict(molecules)
```

## Feature explainability

The whole analysis on feature explainability using SHAP values is contained both in "resources/SHAP_analysis" and "notebooks/SHAP_analysis".

## Repurposing PubChem molecules

- Run the two filters on PubChem - "notebooks/run_puchem.ipynb"
- Search for strong sweeteners derivatives on the filtered molecules - "notebooks/pubchem_repurposed_molecules_analysis.ipynb".
