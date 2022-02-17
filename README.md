# DeepSweet

### Table of contents:

- [Requirements](#requirements)
- [Getting Started](#getting-started)
    - [Setup GPU](#setup-gpu)
    - [Prepare the dataset](#prepare-dataset)
    - [Split dataset](#split-dataset)
    - [Generate features](#generate-features)
    - [Select features](#select-features)
    - [Hyperparameter optimization](#optimize-hyperparameters)
    - [Analyse results](#analyse-results)
<!---    - [Predict](#)
    - [Predict with ensemble](#)
    - [Feature explainability](#)
--->


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
                                                cv=3,
                                                opt_metric="roc_auc",
                                                n_iter_search=30,
                                                seed=123,
                                                featurization_method=featurizer,
                                                folder_path="../resources/test_data/GAT/",
                                                model_output_path="GAT.h5")

optimiser_gat.optimise()
optimiser_gat.save_results()

```

## Analyse results

You can checkout the results file in the "results" folder. "all_results.csv" 
enumerates the results for all metrics to all models.