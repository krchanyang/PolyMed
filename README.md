# PolyMed: A Medical Dataset Addressing Disease Imbalance for Robust Automatic Diagnosis Systems
[Paper(TBA)]() | [Weights(TBA)]() | [Dataset ![DOI: 10.5281/zenodo.7866103](https://zenodo.org/badge/DOI/10.5281/zenodo.7866103.svg)](https://doi.org/10.5281/zenodo.7866103)
<br>
<br>
The **PolyMed** dataset has been developed to improve Automatic Diagnosis Systems (ADS) by addressing the limitations of existing medical case data. This dataset incorporates medical knowledge graph data and diagnosis case data to provide comprehensive evaluation, diverse disease information, effective utilization of external knowledge, and tasks closer to real-world scenarios. The data collection tools have been made publicly available to enable researchers and other interested parties to contribute additional data in a standardized format. 
## Structure
```
repo
  |——data
  |——data_stat
  |——models
  |——runners
    |——training
    |——tuning
    |——testing
  |——tools
  |——utils
```
***

## Anaconda Enironment
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3812/)

```shell
conda env create -n PolyMed -f environments.yaml python==3.8.5
```
You must install [Deep Graph Library](https://www.dgl.ai/pages/start.html) to train Graph Models.
```shell
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
```
```shell
conda activate PolyMed
```

***


## Train
Download the PolyMed and place in **data** folder:
```shell
python run_train.py \
--data_type "extend" \
--train_data_type "extend" \
--save_base_path "./experiments" \
--model_name "ML" \
--device 0 \
--seed 42 \
```

After run this code, the model saved at ``./experiments/{train_data_type}/{model_name}``.

### Arguments
* `data_type`: Data type of whole dataset. It supports "extend" and "norm"
* `train_data_type`: Specify the type of train data. It supports "norm", "extend", and "kb_extend"
  * norm: symptom-diagnosis
  * extend: norm + additional information(e.g. family history, background, underlying disease, ...)
  * kb_extend: extend + knowledge graph 
* `model_name`: Set the type of model to train. It supports "ML", "MLP", "Res", "GraphV1" and "GraphV2"
  - ML
    - LogisticRegression
    - CatBoostClassifier
    - LinearDiscriminantAnalysis
    - XGBClassifier
    - GradientBoostingClassifier
    - RandomForestClassifier
    - ExtraTreesClassifier
    - DecisionTreeClassifier
    - KNeighborsClassifier
    - GaussianNB
    - AdaBoostClassifier
    - LGBMClassifier
  - MLP
    - Simple MLP
  - Res
    - Simple MLP + Residual Block
  - Graph
    - V1 (Knowledge Search)
    - V2 (Cosine Similarity Search)
***
## Test
```shell
python run_test.py \
--data_type "extend" \
--train_data_type "extend" \
--test_data_type "unseen" \
--save_base_path "./experiments" \
--model_name "ML" \
--device 0 \
--seed 42 \
```

### Arguments
* `test_data_type`: Specify the type of train data. It supports "single", "multi", and "unseen"
  * single: The Single test dataset consists of diseases used in training process. This test dataset aim to measure the typical diagnosic ability of ADS.
  * multi: The Multi test dataset consists of multiple diseases. This test dataset aim to measure the multiple-diseases diagnostic ability of ADS.
  * unseen: The Unseen test dataset consists of diseases not used in training process. This test dataset aim to measure the unseen diseases diagnostic ability of ADS. Especially, unseen diseases requires predicting diseases by utilizing the extenal medical knowledge(PolyMed-kg).
***