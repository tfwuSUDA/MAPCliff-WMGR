# MAPCliff-WMGR: Exploring Activity Cliffs in Molecular Activity Prediction Enhanced by Weighted Molecular Graph Representations

## Abstract
In drug discovery, accurately predicting molecular activity is crucial for identifying and optimizing molecules with desirable biological properties. A significant challenge in this field is the phenomenon of activity cliffs, where molecules with similar structures exhibit significantly divergent biological activities. This study introduces MAPCliff-WMGR, a computational framework designed to predict molecular activity under the activity cliff scenario using weighted molecular graphs. MAPCliff-WMGR consists of a core $mGraphSNN_{GAT}$ module to integrate model-specific adjustments to better handle molecular data, enabling the model to effectively predict molecular activity under the activity cliff scenario. Due to activity cliff data exhibiting characteristics of spectral bias, MAPCliff-WMGR addresses this by employing an Independent Feature Mapping (IFM) module that uses sinusoidal transformations to map features into a frequency-rich domain. Experimental results demonstrate that MAPCliff-WMGR outperforms existing models in molecular activity prediction, particularly in handling activity cliffs. Furthermore, we build the MACE-R7 platform, a richer benchmark with various response types and targets, on which our method also achieves strong results. Moreover, the model's interpretability is further demonstrated to uncover critical atoms responsible for activity cliffs through attention-based analysis and dimensionality reduction visualizations. Finally, a case study on small-molecule drugs targeting estrogen receptor alpha (ER$\alpha$) for breast cancer treatment underscores the model's ability to accurately predict activity for cliff molecules, validating its potential for virtual drug screening.

## Environment Setup

To set up the environment, please use the following command:

```bash
conda env create -f environment.yml
```

This will create a new conda environment with all the necessary dependencies listed in the environment.yml file. Once the environment is created, activate it using:

```bash
conda activate <env_name>
```

## Data Processing
For detailed data processing steps, please refer to the README file in the dataset directory.

## Train
```bash
python ./code/end2end_train.py --data_label CHEMBL218_EC50 --device 0
```

## Evaluation
```bash
python ./code/evaluation.py --data_label CHEMBL218_EC50 --graph_model_path ./checkpoints/graph_model.pth --ifm_model_path ./checkpoints/ifm_model.pth --device 0
```

## Inference
```bash
python ./code/inference.py --smiles_file ./data/test.csv --fingerprint_file ./data/test_fingerprints.csv --graph_model_path ./checkpoints/graph_model.pth --ifm_model_path ./checkpoints/ifm_model.pth --output_file ./results/predictions.csv
```

## Reproducibility
Processed data and trained checkpoints: https://pan.quark.cn/s/11d8498e256b