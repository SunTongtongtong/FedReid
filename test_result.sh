MODEL_NAME=$1
GPU_ID=$2
#
# Guile method to evaluate, used to calculate 4 large datasets
# example: sh test_result.sh model_SNR.03_13_21:59:10.pth 0  change --name to model dir name
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py  --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/targetDataset/VIPeR/pytorch --frac 1.0
# echo 'VIPeR result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/targetDataset/QMUL-iLIDS/pytorch --frac 1.0
# echo 'QMUL result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/targetDataset/3DPeS/pytorch --frac 1.0
# echo '3DPeS result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# # echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/targetDataset/CAVIAR/pytorch --frac 1.0
# echo 'CAVIAR result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/targetDataset/GRID/pytorch --frac 1.0
# echo 'GRID result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/dukemtmc-reid/pytorch --frac 1.0
# echo 'Duke result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/market1501/pytorch --frac 1.0
# echo 'market1501 result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'

CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/cuhk03-np/pytorch --frac 1.0
echo 'cuhk03 result'
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
# echo '--------------------------------------------'

# CUDA_VISIBLE_DEVICES=${GPU_ID} python feat_extract.py --model_name ${MODEL_NAME} --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/msmt17/pytorch --frac 1.0
# echo 'msmt result'
# CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluate.py
echo '--------------------------------------------'


#only for evaluating,on viper,
# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight /homes/ss014/projects/FedReID-torchreid/model/8datasets/federated_model.pth \
# --evaluate -s market1501 -t cuhk01  -f market1501  \
# --height 256 --width 128 \
# --gpu-devices 0
