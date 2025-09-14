# Decentralised Person Re-Identification with Selective Knowledge Aggregation (BMVC 2021)

Code release of paper:

[**Decentralised Person Re-Identification with Selective Knowledge Aggregation**](https://arxiv.org/pdf/2110.11384)

[Shitong Sun](https://suntongtongtong.github.io/), [Guile Wu](https://guilewu.github.io/), [Shaogang Gong](http://www.eecs.qmul.ac.uk/~sgg/)

<a href='https://arxiv.org/pdf/2110.11384'><img src='https://img.shields.io/badge/ArXiv-2312.07374-red' /></a> 


## :bulb: Highlight

Existing person re-identification (Re-ID) methods mostly follow a centralised learning paradigm which shares all training data to a collection for model learning, which is limited when data from different sources cannot be shared due to privacy concerns. To resolve this problem, two recent works have introduced decentralised (federated) Re-ID learning for constructing a globally generalised model (server) without any direct access to local training data nor shared data across different source domains (clients). However, these methods are poor on how to adapt the generalised model to maximise its performance on individual client domain Re-ID tasks having different Re-ID label spaces, due to a lack of understanding of data heterogeneity across domains. We call this poor ‘model personalisation’. In this work, we present a new Selective Knowledge Aggregation approach to decentralised person Re-ID to optimise the trade-off between model personalisation and generalisation. 
  

## Quick Start
<!-- The prompt-dialogue of varies abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). -->

<!-- The synthesized prompt-dialogue datasets of various abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). Please follow the steps below to generate datasets with LLaVA format. -->

<!-- 1. Use [SD-XL](https://github.com/crystraldo/StableLLAVA/blob/main/stable_diffusion.py) to generate images as training images. It will take ~13s to generate one image on V100.-->
<!-- python stable_diffusion.py --prompt_path dataset/animal.json --save_path train_set/animal/-->
<!-- 2. Use [data_to_llava](https://github.com/crystraldo/StableLLAVA/blob/main/data_to_llava.py) to convert dataset format for LLaVA model training. -->
<!-- ```
python data_to_llava.py --image_path train_set/ --prompt_path dataset/ --save_path train_ano/
``` -->

### 📊 Dataset Preparation


#### 🔗 Step 1: Download Dataset Files

#### 📁 Step 2: Organize Dataset Structure
Organize the dataset with the following command:
```
python ./utils/prepare.py
```

#### Example Structure
```
sourceDataset/
└── market1501/
    ├── query/
    │   ├── 0001/
    │   │   └── xxx.jpg
    │   ├── 0002/
    │   │   └── xxx.jpg
    │   └── ...
    └── gallery/
        ├── 0001/
        │   └── xxx.jpg
        └── ...
```


### 2. Extract Feature Representation of Testing Data

```
python feat_extract.py
```
Ensure the trained model is saved in ./model_save/FedReID (A trained model is saved in model_example)
You can change testing parameter setting in config_test.py
When you complete, you should find a xxx.mat file in the current directory


### 3 Prepare ids of testing data based on the format of their name

In ./evaluation/get_id.py:

+  for testing on MSMT17, change Line19 `camera[0]' to `camera[0:2]';
+  for CUHK-SYSU person search testing, change Line19 `camera[0]' to `camera[0:2]' and change Line12 `filename[0:4]' to `filename[0:5]';
+  for other benchmarks, use Line19 `camera[0]' and Line12 `filename[0:4]'.


# 4. Testing/Evaluation
```
python evaluate.py
```
When you complete, you should see the rank-1 accuracy and mAP results.
Note, this is the simplified evaluation code, you can download the complete evaluation codes
from Torchreid which supports Cython for evaluation and use format equation for computing mAP performance.
https://kaiyangzhou.github.io/deep-person-reid/


# 5. Training (If you want to train the model again)
Change training parameter setting in `config.py`

```
python train.py
```
When you complete, you should find the model and the log in ./model_save/xxx/



## Citation

If you find our work useful in your research, please consider citing:
```
@article{sun2021decentralised,
  title={Decentralised person re-identification with selective knowledge aggregation},
  author={Sun, Shitong and Wu, Guile and Gong, Shaogang},
  journal={arXiv preprint arXiv:2110.11384},
  year={2021}
}
```
