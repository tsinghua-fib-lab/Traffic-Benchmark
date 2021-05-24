import torch
import numpy as np
import argparse
import time
from util import *
from trainer import Trainer

from net import DGCRN
import setproctitle
import os
import random

setproctitle.setproctitle("DGCRN@lifuxian")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10, help='number of runs')
parser.add_argument('--LOAD_INITIAL',
                    default=False,
                    type=bool,
                    help='If LOAD_INITIAL.')
parser.add_argument('--TEST_ONLY',
                    default=False,
                    type=bool,
                    help='If TEST_ONLY.')

parser.add_argument('--tolerance',
                    type=int,
                    default=100,
                    help='tolerance for earlystopping')
parser.add_argument('--OUTPUT_PREDICTION',
                    default=False,
                    type=bool,
                    help='If OUTPUT_PREDICTION.')

parser.add_argument('--cl_decay_steps',
                    default=2000,
                    type=float,
                    help='cl_decay_steps.')
parser.add_argument('--new_training_method',
                    default=False,
                    type=bool,
                    help='new_training_method.')

parser.add_argument('--rnn_size', type=int, default=64, help='rnn_size')
parser.add_argument('--hyperGNN_dim',
                    type=int,
                    default=32,
                    help='hyperGNN_dim')

parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--data',
                    type=str,
                    default='data/METR-LA',
                    help='data path')

parser.add_argument('--adj_data',
                    type=str,
                    default='data/sensor_graph/adj_mx.pkl',
                    help='adj data path')
parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')

parser.add_argument('--cl',
                    type=str_to_bool,
                    default=True,
                    help='whether to do curriculum learning')

parser.add_argument('--gcn_depth',
                    type=int,
                    default=2,
                    help='graph convolution depth')
parser.add_argument('--num_nodes',
                    type=int,
                    default=207,
                    help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')

parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--seq_in_len',
                    type=int,
                    default=12,
                    help='input sequence length')
parser.add_argument('--seq_out_len',
                    type=int,
                    default=12,
                    help='output sequence length')

parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--tanhalpha', type=float, default=3, help='adj alpha')

parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.0001,
                    help='weight decay rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')

parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./save/', help='save path')

parser.add_argument('--expid', type=str, default='1', help='experiment id')

args = parser.parse_args()
torch.set_num_threads(3)

os.makedirs(args.save, exist_ok=True)

device = torch.device(args.device)
dataloader = load_dataset(args.data, args.batch_size, args.batch_size,
                          args.batch_size)
scaler = dataloader['scaler']

predefined_A = load_adj(args.adj_data)
predefined_A = torch.tensor(predefined_A)
predefined_A = predefined_A / predefined_A.sum(-1).view(-1, 1)
predefined_A = predefined_A.to(device)


def main(runid):

    model = DGCRN(args.gcn_depth,
                  args.num_nodes,
                  device,
                  predefined_A=predefined_A,
                  dropout=args.dropout,
                  subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  middle_dim=2,
                  seq_length=args.seq_in_len,
                  in_dim=args.in_dim,
                  out_dim=args.seq_out_len,
                  layers=args.layers,
                  list_weight=[0.05, 0.95, 0.95],
                  tanhalpha=args.tanhalpha,
                  cl_decay_steps=args.cl_decay_steps,
                  rnn_size=args.rnn_size,
                  hyperGNN_dim=args.hyperGNN_dim)

    print(args)

    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip,
                     args.step_size1, args.seq_out_len, scaler, device,
                     args.cl, args.new_training_method)
    if args.LOAD_INITIAL:
        engine.model.load_state_dict(
            torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +
                       ".pth",
                       map_location='cpu'))
        print('model load success!')

    if args.TEST_ONLY:

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x,
                   y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                engine.model.eval()
                preds = engine.model(testx, ycl=testy)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        if args.OUTPUT_PREDICTION:
            pred_all = scaler.inverse_transform(yhat).cpu()
            path_savepred = args.save + 'result_pred/' + "exp" + str(
                args.expid) + "_" + str(runid)
            os.makedirs(args.save + 'result_pred/', exist_ok=True)
            np.save(path_savepred, pred_all)
            print('result of prediction has been saved, path: ' + os.getcwd() +
                  path_savepred[1:] + '.npy' + ", shape: " +
                  str(pred_all.shape))

        mae = []
        mape = []
        rmse = []
        pred = scaler.inverse_transform(yhat)
        tmae, tmape, trmse = metric(pred, realy)
        for i in [2, 5, 8, 11]:
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        return mae, mape, rmse, mae, mape, rmse, tmae, tmape, trmse

    else:
        print("start training...", flush=True)
        his_loss = []
        val_time = []
        train_time = []
        minl = 1e5
        minl_test = 1e5
        epoch_best = -1
        tolerance = args.tolerance
        count_lfx = 0
        batches_seen = 0
        for i in range(1, args.epochs + 1):
            train_loss = []
            train_mape = []
            train_rmse = []
            t1 = time.time()
            dataloader['train_loader'].shuffle()
            for iter, (x, y, ycl) in enumerate(
                    dataloader['train_loader'].get_iterator()):
                batches_seen += 1
                trainx = torch.Tensor(x).to(device)
                trainx = trainx.transpose(1, 3)
                trainy = torch.Tensor(y).to(device)
                trainy = trainy.transpose(1, 3)

                trainycl = torch.Tensor(ycl).to(device)
                trainycl = trainycl.transpose(1, 3)

                metrics = engine.train(trainx,
                                       trainy[:, 0, :, :],
                                       trainycl,
                                       idx=None,
                                       batches_seen=batches_seen)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])

            t2 = time.time()
            train_time.append(t2 - t1)

            valid_loss = []
            valid_mape = []
            valid_rmse = []

            s1 = time.time()
            for iter, (x, y) in enumerate(
                    dataloader['val_loader'].get_iterator()):
                testx = torch.Tensor(x).to(device)
                testx = testx.transpose(1, 3)
                testy = torch.Tensor(y).to(device)
                testy = testy.transpose(1, 3)
                metrics = engine.eval(testx, testy[:, 0, :, :], testy)
                valid_loss.append(metrics[0])
                valid_mape.append(metrics[1])
                valid_rmse.append(metrics[2])
            s2 = time.time()

            val_time.append(s2 - s1)
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)

            mvalid_loss = np.mean(valid_loss)
            mvalid_mape = np.mean(valid_mape)
            mvalid_rmse = np.mean(valid_rmse)
            his_loss.append(mvalid_loss)

            if (i - 1) % args.print_every == 0:
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                print(log.format(i, (s2 - s1)))
                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                                 mvalid_loss, mvalid_mape, mvalid_rmse,
                                 (t2 - t1)),
                      flush=True)

            if mvalid_loss < minl:
                torch.save(
                    engine.model.state_dict(), args.save + "exp" +
                    str(args.expid) + "_" + str(runid) + ".pth")
                minl = mvalid_loss
                epoch_best = i
                count_lfx = 0
            else:
                count_lfx += 1
                if count_lfx > tolerance:
                    break

        print("Average Training Time: {:.4f} secs/epoch".format(
            np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(
            torch.load(args.save + "exp" + str(args.expid) + "_" + str(runid) +
                       ".pth",
                       map_location='cpu'))

        print("Training finished")
        print("The valid loss on best model is {}, epoch:{}".format(
            str(round(his_loss[bestid], 4)), epoch_best))

        outputs = []
        realy = torch.Tensor(dataloader['y_val']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx, ycl=testy)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        pred = scaler.inverse_transform(yhat)
        vmae, vmape, vrmse = metric(pred, realy)

        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)[:, 0, :, :]

        for iter, (x,
                   y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx, ycl=testy)
                preds = preds.transpose(1, 3)
            outputs.append(preds.squeeze(dim=1))

        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        mae = []
        mape = []
        rmse = []

        pred = scaler.inverse_transform(yhat)
        tmae, tmape, trmse = metric(pred, realy)
        for i in [2, 5, 8, 11]:
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = metric(pred, real)
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
            mae.append(metrics[0])
            mape.append(metrics[1])
            rmse.append(metrics[2])
        return vmae, vmape, vrmse, mae, mape, rmse, tmae, tmape, trmse


if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []
    tmae = []
    tmape = []
    trmse = []
    for i in range(args.runs):

        if args.runs == 1:
            i = 2
        elif args.runs == 2:
            i += 1

        i += 3

        vm1, vm2, vm3, m1, m2, m3, tm1, tm2, tm3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        tmae.append(tm1)
        tmape.append(tm2)
        trmse.append(tm3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\nResults for ' + str(args.runs) + ' runs\n\n')

    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae), np.mean(vrmse), np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae), np.std(vrmse), np.std(vmape)))
    print('\n\n')

    print('test\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(tmae), np.mean(trmse), np.mean(tmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(tmae), np.std(trmse), np.std(tmape)))
    print('\n\n')
    print(
        'test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std'
    )
    for i in range(4):
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(
            log.format([3, 6, 9, 12][i], amae[i], armse[i], amape[i], smae[i],
                       srmse[i], smape[i]))
