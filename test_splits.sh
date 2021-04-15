#only for evaluating,on small datasets: viper,ilids,grid,cuhk01,prid2011, have ten splits, calculate average
#using arg.py file as configuration
MODEL_NAME=$1
GPU_ID=$2

# /homes/ss014/projects/FedReID-master/model_save/withoutExpert/model_withoutExpert.04_13_11:46:45.pth
# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight  ${MODEL_NAME}  \
# --evaluate -s market1501 -t viper  -f market1501 \
# --gpu_ids ${GPU_ID}   


#--load-weight /homes/ss014/projects/FedReID-master/model_save/AAAIFedReIDAgain/model_AAAIFedReIDAgain.04_12_22:08:50.pth \
# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight  ${MODEL_NAME}  \
# --evaluate -s market1501 -t ilids  -f market1501 \
# --gpu_ids ${GPU_ID}   

# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight  ${MODEL_NAME}  \
# --evaluate -s market1501 -t prid2011  -f market1501 \
# --gpu_ids ${GPU_ID}   

# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight  ${MODEL_NAME}  \
# --evaluate -s market1501 -t grid  -f market1501 \
# --gpu_ids ${GPU_ID}   

python test_shitong.py --root /homes/ss014/datasets/ \
--load-weight  ${MODEL_NAME}  \
--evaluate -s market1501 -t cuhk01  -f market1501 \
--gpu_ids ${GPU_ID}   

# python test_shitong.py --root /homes/ss014/datasets/ \
# --load-weight  ${MODEL_NAME}  \
# --evaluate -s market1501 -t viper  -f market1501 \
# --gpu_ids ${GPU_ID}   