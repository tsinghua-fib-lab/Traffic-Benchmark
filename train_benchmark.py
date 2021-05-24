import os
import random
import numpy as np
import torch
# import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str,default='DGCRN',help='model')
parser.add_argument('--data',type=str,default='METR-LA',help='dataset')
args = parser.parse_args()

model = args.model
data = args.data
# setproctitle.setproctitle(model + '_' + data + "@lifuxian")

def main():
    if model == 'DGCRN':
        if data == 'BJ':
            run = 'python ./methods/DGCRN_BJ/train.py --adj_data ~/NE-BJ/adj_mat_BJ_new.pkl --data ~/NE-BJ/ --num_nodes 500 --runs 3  --epochs 250 --print_every 10 --batch_size 16 --tolerance 100 --step_size1 2100 --cl_decay_steps 3500 --expid DGCRN_bj --rnn_size 64 --node_dim 100 --device cuda:5'
            os.system(run)
        elif data == 'METR-LA':
            run = 'python ./methods/DGCRN/train.py --adj_data ./data/sensor_graph/adj_mx.pkl --data ./data/METR-LA --num_nodes 207 --runs 3  --epochs 110 --print_every 10 --batch_size 64 --tolerance 100  --cl_decay_steps 4000 --expid DGCRN_metrla --device cuda:2'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python ./methods/DGCRN/train.py --adj_data ./data/sensor_graph/adj_mx_bay.pkl --data ./data/PEMS-BAY --num_nodes 325 --runs 3 --epochs 110 --print_every 10 --batch_size 64 --tolerance 100 --expid DGCRN_pemsbay  --cl_decay_steps 5500 --rnn_size 96 --device cuda:2'
            os.system(run)
    elif model == 'FNN':
        if data == 'BJ':
            run = 'CUDA_VISIBLE_DEVICES=2 python dcrnn_train_pytorch.py --config_filename=data/model/stmetanet_BJ500.yaml'
            os.system(run)
        elif data == 'METR-LA':
            run = ''
            os.system(run)
        elif data == 'PEMS-BAY':
            run = ''
            os.system(run)
    elif model == 'FC-LSTM':
        if data == 'BJ':
            run = 'CUDA_VISIBLE_DEVICES=0 python ./methods/LSTM/dcrnn_train_pytorch.py --config_filename=data/model/LSTM_BJ500.yaml '
            os.system(run)
        elif data == 'METR-LA':
            run = ''
            os.system(run)
        elif data == 'PEMS-BAY':
            run = ''
            os.system(run)
    elif model == 'DCRNN':
        if data == 'BJ':
            run = 'CUDA_VISIBLE_DEVICES=1 python ./methods/DCRNN/dcrnn_train_pytorch.py --config_filename=data/BJ/dcrnn_BJ.yaml'
            os.system(run)
        elif data == 'METR-LA':
            run = 'CUDA_VISIBLE_DEVICES=3 python ./methods/DCRNN/dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_la.yaml'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'CUDA_VISIBLE_DEVICES=1 python ./methods/DCRNN/dcrnn_train_pytorch.py --config_filename=data/model/dcrnn_bay.yaml'
            os.system(run)
    elif model == 'STGCN':
        if data == 'BJ':
            run = ''
            os.system(run)
        elif data == 'METR-LA':
            run = ''
            os.system(run)
        elif data == 'PEMS-BAY':
            run = ''
            os.system(run)
    elif model == 'Graph-WaveNet':
        if data == 'BJ':
            run = 'python ./methods/Graph-WaveNet/train.py --data ~/NE-BJ --adjdata ~/NE-BJ/adj_mat_BJ.pkl  --save ./garage/BJ_500_nodedim100 --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --device cuda:2 --batch_size 64 --epoch 200 --print_every 10'
            os.system(run)
        elif data == 'METR-LA':
            run = 'python ./methods/Graph-WaveNet/train.py --data=data/METR-LA --gcn_bool --adjtype doubletransition --addaptadj  --randomadj'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python ./methods/Graph-WaveNet/train.py --data=data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --save ./garage/pems --gcn_bool --adjtype doubletransition --addaptadj  --randomadj'
            os.system(run)
    elif model == 'ST-MetaNet':
        if data == 'BJ':
            run = 'CUDA_VISIBLE_DEVICES=1 python ./methods/ST-MetaNet/dcrnn_train_pytorch.py --config_filename=data/model/stmetanet_BJ500.yaml'
            os.system(run)
        elif data == 'METR-LA':
            run = ''
            os.system(run)
        elif data == 'PEMS-BAY':
            run = ''
            os.system(run)
    elif model == 'ASTGCN':
        if data == 'BJ':
            run = 'python ./methods/ASTGCN/train_ASTGCN_r.py --config configurations/BJ.conf'
            os.system(run)
        elif data == 'METR-LA':
            run = 'python ./methods/ASTGCN/train_ASTGCN_r.py --config configurations/METR-LA.conf'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python ./methods/ASTGCN/train_ASTGCN_r.py --config configurations/PEMS-BAY.conf'
            os.system(run)
    elif model == 'STSGCN': #mxnet-1.41-py3
        if data == 'BJ':
            run = 'python3 ./method/STSGCN/main.py --config config/BJ/individual_GLU_mask_emb.json --save'
            os.system(run)
        elif data == 'METR-LA':
            run = 'python3 ./method/STSGCN/main.py --config config/METR-LA/individual_GLU_mask_emb.json --save'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python3 ./method/STSGCN/main.py --config config/PEMS-BAY/individual_GLU_mask_emb.json --save'
            os.system(run)
    elif model == 'AGCRN':
        if data == 'BJ':
            run = 'python ./methods/AGCRN/model/Run_BJ.py --dataset_dir /data/lifuxian/NE-BJ/ --device cuda:6'
            os.system(run)
        elif data == 'METR-LA':
            run = 'python ./methods/AGCRN/model/Run_METR-LA.py --dataset_dir /data/lifuxian/DCRNN_PyTorch-pytorch_scratch/data/METR-LA --device cuda:5'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'python ./methods/AGCRN/model/Run_PEMS-BAY.py --dataset_dir /data/lifuxian/DCRNN_PyTorch-pytorch_scratch/data/PEMS-BAY --device cuda:6'
            os.system(run)
    elif model == 'GMAN': #tf-2.3-py3
        if data == 'BJ':
            run = 'CUDA_VISIBLE_DEVICES=4 python ./methods/GMAN/BJ500/train.py --batch_size 8'
            os.system(run)
        elif data == 'METR-LA':
            run = 'CUDA_VISIBLE_DEVICES=4 python ./methods/GMAN/METR/train.py'
            os.system(run)
        elif data == 'PEMS-BAY':
            run = 'CUDA_VISIBLE_DEVICES=4 python ./methods/GMAN/PeMS/train.py'
            os.system(run)
    elif model == 'MTGNN':
        if data == 'BJ':
            run = 'python ./methods/MTGNN/train_multi_step.py --adj_data ~/NE-BJ/adj_mat_BJ.pkl --data ~/NE-BJ --num_nodes 500 --runs 3 --device cuda:1 --epochs 1000 --print_every 1000 --buildA_true True --expid 80 --node_dim 80'
            os.system(run)
        elif data == 'METR-LA':
            run = ''
            os.system(run)
        elif data == 'PEMS-BAY':
            run = ''
            os.system(run)
    




if __name__ == "__main__":
    main()