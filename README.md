## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Data

The dataset link is https://zenodo.org/records/5146275. You can download the datasets and place them in the `raw_data` directory after converting to LibCity atomized format.

Note that our model would calculate a **DTW matrix** and a **traffic pattern set** for each dataset, which is time-consuming. Therefore, we have provided DTW matrices and traffic pattern sets of all datasets in `./libcity/cache/dataset_cache/`.

## Train & Test

You can train and test **PDFormer** through the following commands for 6 datasets. Parameter configuration (**--config_file**) reads the JSON file in the root directory. If you need to modify the parameter configuration of the model, please modify the corresponding **JSON** file.

```shell
python run_model.py --task traffic_state_pred --model PDFormerDV --dataset PEMS_BAY --config_file PeMS_BAY
python run_model.py --task traffic_state_pred --model PDFormerDV --dataset METR_LA --config_file METR_LA
```

If you have trained a model as above and only want to test it, you can set it as follows (taking PeMS08 as an example, assuming the experiment ID during training is $ID):

```shell
python run_model.py --task traffic_state_pred --model PDFormer --dataset PEMS-BAY --config_file PEMS-BAY --train false --exp_id $ID
```

**Note**: By default the result recorded in the experiment log is the average of the first n steps. This is consistent with the paper (configured as **"mode": "average"** in the JSON file). If you need to get the results of each step separately, please modify the configuration of the JSON file to **"mode": "single"**.


## Reference Code

Code based on [LibCity](https://github.com/LibCity/Bigscity-LibCity) framework development, an open source library for traffic prediction.

## Cite

If you find the paper useful, please cite as following:

```
TBA
```

If you find [LibCity](https://github.com/LibCity/Bigscity-LibCity) useful, please cite as following:

```
@inproceedings{libcity,
  author    = {Jingyuan Wang and
               Jiawei Jiang and
               Wenjun Jiang and
               Chao Li and
               Wayne Xin Zhao},
  title     = {LibCity: An Open Library for Traffic Prediction},
  booktitle = {{SIGSPATIAL/GIS}},
  pages     = {145--148},
  publisher = {{ACM}},
  year      = {2021}
}
```
