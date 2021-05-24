MODEL_NAME=$1
GPU_ID=$2
#
# Guile method to evaluate, used to calculate 4 large datasets
# example: sh test_result.sh model_SNR.03_13_21:59:10.pth 0  change --name to model dir name
# for i in 1 2 #3 4 5 6 7 8 9 10
# do 
CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py  --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/10_split_targetDataset/VIPeR/split-0/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/10_split_targetDataset/QMUL-iLIDS/split-0/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/10_split_targetDataset/GRID/split-0/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/10_split_targetDataset/PRID/split-0/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/dukemtmc-reid/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/market1501/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/cuhk03-np/pytorch --frac 1.0

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/msmt17/pytorch --frac 1.0



#only for evaluating,on viper,
# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight /homes/ss014/projects/FedReID-torchreid/model/8datasets/federated_model.pth \
# --evaluate -s market1501 -t cuhk01  -f market1501  \
# --height 256 --width 128 \
# --gpu-devices 0
