# PolyMed: A Medical Dataset Addressing Disease Imbalance for Robust Automatic Diagnosis Systems
[Paper(TBA)]() | [Weights](https://drive.google.com/drive/folders/15w-i46TBs9T7QB78onARuEgnlxxC3JGS?usp=sharing) | [Dataset ![DOI: 10.5281/zenodo.7866103](https://zenodo.org/badge/DOI/10.5281/zenodo.7866103.svg)](https://doi.org/10.5281/zenodo.7866103)
<br>
<br>
The **PolyMed** dataset has been developed to improve Automatic Diagnosis Systems (ADS) by addressing the limitations of existing medical case data. This dataset incorporates medical knowledge graph data and diagnosis case data to provide comprehensive evaluation, diverse disease information, effective utilization of external knowledge, and tasks closer to real-world scenarios. The data collection tools have been made publicly available to enable researchers and other interested parties to contribute additional data in a standardized format. 
## Structure
```
repo
  |——data
    |——eng_external_medical_knowledge.json
    |——eng_test_multi.json
    |——eng_test_single.json
    |——eng_test_unseen.json
    |——eng_train.json
  |——data_stat
  |——experiments
    |——extend
    |——kb_extend
    |——norm
  |——models
  |——runners
    |——training
    |——tuning
    |——testing
  |——tools
  |——utils
```

Our baseline models have been trained on PolyMed, which are available for download in the [Dataset](https://doi.org/10.5281/zenodo.7866103) and [Weights](https://drive.google.com/drive/folders/15w-i46TBs9T7QB78onARuEgnlxxC3JGS?usp=sharing), respectively.

Download and extract these files to the following location:<br>
 *Dataset -> ./data<br>
 *weights -> ./experiments<br>

It is important to follow the file structure provided to ensure that the models can access the necessary data and weights during training and testing.
***

## Docker Image (Option 1)
```shell
docker pull kimjonghyeon/polymed
```

1. Create Container with Local Repository
```shell
docker run -it -v {$your_path}/PolyMed:/home/PolyMed --name "polymed" --gpus "device=0" kimjonghyeon/polymed
```
2. Change Directory
```shell
$ cd /home/PolyMed
```

## Anaconda Environment (Option 2)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3812/)

1. Create Anaconda Environment
```shell
conda env -n PolyMed python==3.8.5
```
2. Activate Created Environment
```shell
conda activate PolyMed
```
3. Install Requirements (CUDA 11.8)
```shell
pip install -r requirements.txt
```

## Simple Exploratory Data Analysis (EDA)
```shell
python run_data_stat.py
```
After run this code, the figure of data statistics saved at ``./data_stat``.

## Train
Download the PolyMed and place in **data** folder:
```shell
python run_train.py \
--data_type "extend" \
--train_data_type "kb_extend" \
--class_weights "True" \
--save_base_path "./experiments" \
--model_name "ML" \
--device 0 \
--seed 42
```

After run this code, the trained models saved at ``./experiments/{train_data_type}/{model_name}``.

### Arguments
* `data_type`: Data type of whole dataset. It supports "extend" and "norm"
* `train_data_type`: Specify the type of train data. It supports "norm", "extend", and "kb_extend"
  * norm: symptom-diagnosis
  * extend: norm + additional information(e.g. family history, background, underlying disease, ...)
  * kb_extend: extend + knowledge graph 
* `class_weights`: Using class weights when training the kb_extend data. (Default: False)
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

### Training Guides
* If want to train with other models instead ML, just change model_name argument to other models. (e.g. --model_name "ML" → --model_name "GraphV2")
* Must match data type arguments to train each datasets:
  - Normal Data: --data_type "norm" --train_data_type "norm"
  - Extend Data: --data_type "extend" --train_data_type "extend"
  - Knowledge Base Extend Data: --data_type "extend" --train_data_type "kb_extend"

## Test
```shell
python run_test.py \
--data_type "extend" \
--train_data_type "kb_extend" \
--test_data_type "unseen" \
--class_weights "True" \
--save_base_path "./experiments" \
--model_name "ml_tuned" \
--device 0 \
--seed 42
```

### Arguments
* `test_data_type`: Specify the type of train data. It supports "single", "multi", and "unseen"
  * single: The Single test dataset consists of diseases used in training process. This test dataset aim to measure the typical diagnosic ability of ADS.
  * multi: The Multi test dataset consists of multiple diseases. This test dataset aim to measure the multiple-diseases diagnostic ability of ADS.
  * unseen: The Unseen test dataset consists of diseases not used in training process. This test dataset aim to measure the unseen diseases diagnostic ability of ADS. Especially, unseen diseases requires predicting diseases by utilizing the extenal medical knowledge(PolyMed-kg).

### Testing Guides
Just additionally specify the type of test data following the training guides<br>
- Example1-Machine Learning test for 'norm' dataset:
   * --data_type "norm" --train_data_type "norm" --test_data_type "single" --model_name "ml_tuned"
   * --data_type "norm" --train_data_type "norm" --test_data_type "multi" --model_name "ml_tuned"
- Example2-Graph test for 'kb_extend' dataset:
   * --data_type "extend" --train_data_type "kb_extend" --test_data_type "single" --model_name "graphv1"
   * --data_type "extend" --train_data_type "kb_extend" --test_data_type "unseen" --model_name "graphv1"
   * --data_type "extend" --train_data_type "kb_extend" --test_data_type "multi" --model_name "graphv1"
- Example3-resmlp test for 'extend' dataset:
   * --data_type "extend" --train_data_type "extend" --test_data_type "single" --model_name "res"
   * --data_type "extend" --train_data_type "extend" --test_data_type "unseen" --model_name "res"
   * --data_type "extend" --train_data_type "extend" --test_data_type "multi" --model_name "res"
***
### Installation Issues
If you encounter a version collision or Not Found Error of specific library during the installation of requirements, follow these steps:<br>
1. First, install the re_env.txt file using the following command:<br>
```shell
pip install -r re_env.txt
```
2. Second, install cupy according to your specific requirements. You can find instructions for installing cupy on the official website: https://docs.cupy.dev/en/stable/install.html
3. Third, install PyTorch according to your specific requirements. You can find instructions for installing PyTorch on the official website: https://pytorch.org/get-started.
Make sure to install the correct version of PyTorch that matches your system's specifications.

4. Finally, install DGL according to your specific requirements. You can find instructions for installing DGL on the official website: https://www.dgl.ai/pages/start.html.
Make sure to install the version of DGL that matches your PyTorch version (including any CUDA or cuDNN specifications).
***
### Experiment Environment
- CPU: AMD Ryzen Threadripper PRO 5995WX (64 cores, 128 threads)
- GPU: RTX 4090 1EA
- RAM: 256GB
- OS: Ubuntu 20.04.6 LTS
- CUDA: 11.8
- CuDNN: 8.8.0

## Baseline Experiments (kb_extend)
### Single
|Model|recall@1|recall@3|recall@5|precision@1|precision@3|precision@5|f1@1|f1@3|f1@5|ndcg@1|ndcg@3|ndcg@5|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|CatBoost|0.4906|0.7734|0.8713|0.4906|0.2578|0.1743|0.4906|0.3867|0.2904|0.4906|0.6532|0.6935|
|LDA|0.4587|0.7195|0.7987|0.4587|0.2398|0.1597|0.4587|0.3597|0.2662|0.4587|0.6096|0.6426|
|XGBoost|0.4004|0.6623|0.7855|0.4004|0.2208|0.1571|0.4004|0.3311|0.2618|0.4004|0.5543|0.6049|
|GradientBoosting|0.4807|0.7437|0.8515|0.4807|0.2479|0.1703|0.4807|0.3718|0.2838|0.4807|0.6340|0.6781|
|RandomForest|0.4994|0.7646|0.8724|0.4994|0.2549|0.1745|0.4994|0.3823|0.2908|0.4994|0.6556|0.7003|
|ExtraTrees|0.4774|0.7448|0.8306|0.4774|0.2483|0.1661|0.4774|0.3724|0.2769|0.4774|0.6337|0.6691|
|MLP|0.4268|0.7041|0.8350|0.4268|0.2347|0.1670|0.4268|0.3520|0.2783|0.4268|0.5886|0.6420|
|ResMLP|0.3641|0.6271|0.7723|0.3641|0.2090|0.1545|0.3641|0.3135|0.2574|0.3641|0.5184|0.5784|
|Graphv1|0.4323|0.6854|0.8108|0.4323|0.2285|0.1622|0.4323|0.3427|0.2703|0.4323|0.5805|0.6321|
|Graphv2|0.4202|0.6601|0.7591|0.4202|0.2200|0.1518|0.4202|0.3300|0.2530|0.4202|0.5589|0.5997|

### Multi
|Model|recall@1|recall@3|recall@5|precision@1|precision@3|precision@5|f1@1|f1@3|f1@5|ndcg@1|ndcg@3|ndcg@5|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|CatBoost|0.2446|0.5301|0.6984|0.5087|0.3679|0.2918|0.3304|0.4344|0.4116|0.3035|0.5073|0.5933|
|LDA|0.2328|0.5080|0.6391|0.4856|0.3526|0.2676|0.3147|0.4162|0.3772|0.2891|0.4845|0.5520|
|XGBoost|0.1922|0.4463|0.6098|0.4002|0.3106|0.2551|0.2597|0.3663|0.3597|0.2386|0.4200|0.5034|
|GradientBoosting|0.2191|0.5191|0.6683|0.4556|0.3602|0.2789|0.2959|0.4253|0.3936|0.2718|0.4866|0.5632|
|RandomForest|0.2366|0.5172|0.6659|0.4925|0.3591|0.2782|0.3197|0.4239|0.3924|0.2936|0.4948|0.5708|
|ExtraTrees|0.2216|0.4765|0.6221|0.4637|0.3306|0.2597|0.2999|0.3904|0.3665|0.2755|0.4586|0.5335|
|MLP|0.2015|0.4751|0.6563|0.4198|0.3291|0.2743|0.2723|0.3889|0.3869|0.2501|0.4462|0.5393|
|ResMLP|0.1912|0.4200|0.5687|0.3956|0.2910|0.2371|0.2578|0.3438|0.3347|0.2368|0.4011|0.4775|
|Graphv1|0.2055|0.4842|0.6490|0.4279|0.3372|0.2715|0.2777|0.3975|0.3829|0.2551|0.4538|0.5382|
|Graphv2|0.1904|0.4022|0.5395|0.3945|0.2780|0.2242|0.2568|0.3287|0.3168|0.2359|0.3880|0.4583|

### Unseen
|Model|recall@1|recall@3|recall@5|precision@1|precision@3|precision@5|f1@1|f1@3|f1@5|ndcg@1|ndcg@3|ndcg@5|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|CatBoost|0.0335|0.0706|0.0818|0.0335|0.0235|0.0164|0.0335|0.0353|0.0273|0.0335|0.0550|0.0595|
|LDA|0.0892|0.1599|0.1729|0.0892|0.0533|0.0346|0.0892|0.0799|0.0576|0.0892|0.1304|0.1359|
|XGBoost|0.0223|0.0613|0.0874|0.0223|0.0204|0.0175|0.0223|0.0307|0.0291|0.0223|0.0435|0.0542|
|GradientBoosting|0.0539|0.1115|0.1468|0.0539|0.0372|0.0294|0.0539|0.0558|0.0489|0.0539|0.0866|0.1013|
|RandomForest|0.0390|0.1580|0.2454|0.0390|0.0527|0.0491|0.0390|0.0790|0.0818|0.0390|0.1053|0.1413|
|ExtraTrees|0.0316|0.1041|0.1468|0.0316|0.0347|0.0294|0.0316|0.0520|0.0489|0.0316|0.0720|0.0900|
|MLP|0.0130|0.0204|0.0446|0.0130|0.0068|0.0089|0.0130|0.0102|0.0149|0.0130|0.0170|0.0266|
|ResMLP|0.0074|0.0297|0.0539|0.0074|0.0099|0.0108|0.0074|0.0149|0.0180|0.0074|0.0193|0.0292|
|Graphv1|0.0260|0.0316|0.0428|0.0260|0.0105|0.0086|0.0260|0.0158|0.0143|0.0260|0.0291|0.0336|
|Graphv2|0.0223|0.0428|0.0558|0.0223|0.0143|0.0112|0.0223|0.0214|0.0186|0.0223|0.0342|0.0393|

### Weighted Arithmetic Mean (WAM)
|Model|recall@1|recall@3|recall@5|precision@1|precision@3|precision@5|f1@1|f1@3|f1@5|ndcg@1|ndcg@3|ndcg@5|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|CatBoost|0.1881|0.3653|0.4462|0.2740|0.1688|0.1212|0.2218|0.2300|0.1902|0.2120|0.3251|0.3625|
|LDA|0.2158|0.4049|0.4712|0.2970|0.1798|0.1245|0.2473|0.2474|0.1961|0.2381|0.3577|0.3886|
|XGBoost|0.1467|0.3108|0.4053|0.2164|0.1440|0.1100|0.1740|0.1960|0.1726|0.1660|0.2704|0.3136|
|GradientBoosting|0.1898|0.3834|0.4753|0.2722|0.1734|0.1268|0.2217|0.2375|0.1994|0.2123|0.3360|0.3781|
|RandomForest|0.1895|0.4153|0.5373|0.2761|0.1858|0.1410|0.2233|0.2550|0.2222|0.2135|0.3548|0.4102|
|ExtraTrees|0.1751|0.3637|0.4564|0.2582|0.1665|0.1223|0.2074|0.2271|0.1920|0.1980|0.3171|0.3599|
|MLP|0.1468|0.2989|0.4001|0.2213|0.1421|0.1105|0.1760|0.1923|0.1729|0.1675|0.2664|0.3127|
|ResMLP|0.1310|0.2735|0.3667|0.1951|0.1290|0.1015|0.1565|0.1749|0.1587|0.1491|0.2404|0.2829|
|Graphv1|0.1588|0.3071|0.3917|0.2340|0.1440|0.1079|0.1882|0.1957|0.1690|0.1796|0.2760|0.3152|
|Graphv2|0.1479|0.2813|0.3558|0.2198|0.1332|0.0987|0.1759|0.1802|0.1542|0.1677|0.2541|0.2884|