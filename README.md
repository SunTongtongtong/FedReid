# Official code for paper Decentralised Person Re-Identification with Selective Knowledge Aggregation

## Usage
# 1. Prepare datasets
```
1.1 Download the datasets to the folder 'sourceDataset' or 'targetDataset'
    or change the dataset path in ./configuration/.
    e.g. to ./sourceDataset/market1501
1.2 python ./utils/prepare.py
```
When you complete, your dataset is constructed as:
    ./sourceDataset/market1501/python/query/
                                    ....../0001/xxx.jpg
                                    ....../0002/xxx.jpg
    ./sourceDataset/market1501/python/gallery/



# 2. Extract Feature Representation of Testing Data
```
2.1 Ensure the trained model is saved in ./model_save/FedReID (A trained model is saved in model_example)
2.2 Change testing parameter setting in config_test.py
2.3 python feat_extract.py
```
When you complete, you should find a xxx.mat file in the current directory


# 3 Prepare ids of testing data based on the format of their name
```
In ./evaluation/get_id.py,
(1) for testing on MSMT17, change Line19 'camera[0]' to 'camera[0:2]';
(2) for CUHK-SYSU person search testing, change Line19 'camera[0]' to 'camera[0:2]' and change Line12 'filename[0:4]' to 'filename[0:5]';
(3) for other benchmarks, use Line19 'camera[0]' and Line12 'filename[0:4]'.
```

# 4. Testing/Evaluation
```
4.1 python evaluate.py
```
When you complete, you should see the rank-1 accuracy and mAP results.
Note, this is the simplified evaluation code, you can download the complete evaluation codes
from Torchreid which supports Cython for evaluation and use format equation for computing mAP performance.
https://kaiyangzhou.github.io/deep-person-reid/


# 5. Training (If you want to train the model again)
```
2.1 Change training parameter setting in config.py
2.2 python train.py
```
When you complete, you should find the model and the log in ./model_save/xxx/


## Citation
    @article{wu2020decentralised,
      title={Decentralised Learning from Independent Multi-Domain Labels for Person Re-Identification},
      author={Wu, Guile and Gong, Shaogang},
      journal={arXiv preprint arXiv:2006.04150},
      year={2020}
    }

