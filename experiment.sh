#experiment1
# train:

#test on duke
# test on duke, no need to change get id function
python feat_extract.py  --name centralize --agg dukemtmc-reid --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/dukemtmc-reid/pytorch
python evaluate.py

# test:
# test on market1501
python feat_extract.py  --name centralize --agg market1501 --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/market1501/pytorch
python evaluate.py

# test on cuhk03
#Note: change get_id function to get_cuhk_id function
python feat_extract.py  --name centralize --agg cuhk03-np --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/cuhk03-np/pytorch
python evaluate.py

#test on msmt17
#change get_id function to get_cuhk_id function
python feat_extract.py  --name centralize --agg msmt17 --test_data_dir /homes/ss014/projects/FedReID-master/sourceDataset/msmt17/pytorch
python evaluate.py

# disentangle experiment 2
#for the expert model, model representation concat instead of summation



