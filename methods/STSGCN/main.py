# -*- coding:utf-8 -*-
import setproctitle
setproctitle.setproctitle("STSGCN@lifuxian")
import time
import json
import argparse

import numpy as np
import mxnet as mx

from utils import (construct_model, generate_data,
                   masked_mae_np, masked_mape_np, masked_mse_np)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))

net = construct_model(config)

batch_size = config['batch_size']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])

loaders = []
true_values = []
for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
    if args.test:
        x = x[: 100]
        y = y[: 100]
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    loaders.append(
        mx.io.NDArrayIter(
            x, y if idx == 0 else None,
            batch_size=batch_size,
            shuffle=(idx == 0),
            label_name='label'
        )
    )
    if idx == 0:
        training_samples = x.shape[0]
    else:
        true_values.append(y)

train_loader, val_loader, test_loader = loaders
val_y, test_y = true_values

global_epoch = 1
global_train_steps = training_samples // batch_size + 1
all_info = []
epochs = config['epochs']

mod = mx.mod.Module(
    net,
    data_names=['data'],
    label_names=['label'],
    context=ctx
)

mod.bind(
    data_shapes=[(
        'data',
        (batch_size, config['points_per_hour'], num_of_vertices, 1)
    ), ],
    label_shapes=[(
        'label',
        (batch_size, config['points_per_hour'], num_of_vertices)
    )]
)

mod.init_params(initializer=mx.init.Xavier(magnitude=0.0003))
lr_sch = mx.lr_scheduler.PolyScheduler(
    max_update=global_train_steps * epochs * config['max_update_factor'],
    base_lr=config['learning_rate'],
    pwr=2,
    warmup_steps=global_train_steps
)

mod.init_optimizer(
    optimizer=config['optimizer'],
    optimizer_params=(('lr_scheduler', lr_sch),)
)

num_of_parameters = 0
for param_name, param_value in mod.get_params()[0].items():
    # print(param_name, param_value.shape)
    num_of_parameters += np.prod(param_value.shape)
print("Number of Parameters: {}".format(num_of_parameters), flush=True)

metric = mx.metric.create(['RMSE', 'MAE'], output_names=['pred_output'])

if args.plot:
    graph = mx.viz.plot_network(net)
    graph.format = 'png'
    graph.render('graph')


def training(epochs):
    global global_epoch
    lowest_val_loss = 1e6

    tolerance = 50
    cnt_temp = 0

    for epoch in range(epochs):
        t = time.time()
        info = [global_epoch]
        train_loader.reset()
        metric.reset()
        for idx, databatch in enumerate(train_loader):
            mod.forward_backward(databatch)
            mod.update_metric(metric, databatch.label)
            mod.update()
        metric_values = dict(zip(*metric.get()))

        print('training: Epoch: %s, RMSE: %.2f, MAE: %.2f, time: %.2f s' % (
            global_epoch, metric_values['rmse'], metric_values['mae'],
            time.time() - t), flush=True)
        info.append(metric_values['mae'])

        val_loader.reset()
        prediction = mod.predict(val_loader)[1].asnumpy()
        loss = masked_mae_np(val_y, prediction, 0)
        print('validation: Epoch: %s, loss: %.2f, time: %.2f s' % (
            global_epoch, loss, time.time() - t), flush=True)
        info.append(loss)

        if loss < lowest_val_loss:

            test_loader.reset()
            prediction = mod.predict(test_loader)[1].asnumpy()
            tmp_info = []
            for idx in range(config['num_for_predict']):
                # y, x = test_y[:, : idx + 1, :], prediction[:, : idx + 1, :]
                y, x = test_y[:, idx : idx + 1, :], prediction[:, idx : idx + 1, :]
                tmp_info.append((
                    masked_mae_np(y, x, 0),
                    masked_mape_np(y, x, 0),
                    masked_mse_np(y, x, 0) ** 0.5
                ))
            mae, mape, rmse = tmp_info[-1]
            print('test: Epoch: {}, MAE: {:.2f}, MAPE: {:.2f}, RMSE: {:.2f}, '
                  'time: {:.2f}s'.format(
                    global_epoch, mae, mape, rmse, time.time() - t))
            print(flush=True)
            info.extend((mae, mape, rmse))
            info.append(tmp_info)
            all_info.append(info)
            lowest_val_loss = loss

            cnt_temp = 0
        else:
            cnt_temp += 1

        if cnt_temp < tolerance:
            global_epoch += 1
        else:
            print('earlystopping at epoch ', epoch)
            break


if args.test:
    epochs = 5
training(epochs)

the_best = min(all_info, key=lambda x: x[2])
# print('step: {}\ntraining loss: {:.2f}\nvalidation loss: {:.2f}\n'
#       'tesing: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'
#       'testing: RMSE: {:.2f}\n'.format(*the_best))
# print(the_best)

# the_best = [68, 1.411298357080995, 1.796820238713304, 2.261148341010781, 5.400548858947783, 5.2094251746191915, [(0.960964881090736, 1.8586036151815766, 1.7065738206433105), (1.231127729653431, 2.5126222235681794, 2.4261496147356), (1.4356823103392997, 3.037844077552284, 3.0128969394052794), (1.5964147147430807, 3.481093203674549, 3.4896500217794113), (1.729473497683941, 3.8699448526983558, 3.8808571468819997), (1.834626912422782, 4.16717613882355, 4.181905769364844), (1.9326727985467773, 4.442828320728456, 4.432138512797938), (2.0172858990449942, 4.6921727248928145, 4.63484797416534), (2.086977551399003, 4.89622051400139, 4.810464363080575), (2.146455060549889, 5.077811552706208, 4.952446549782243), (2.202791783490587, 5.23813464423378, 5.084228846970088), (2.261148341010781, 5.400548858947783, 5.2094251746191915)]]
print('step: {}\ntraining loss: {:.2f}\nvalidation loss: {:.2f}\n'.format(*the_best))
for i in [2,5,11]:
    print('Horizon ' + str(i+1)+':')
    print('test\tMAE\t\tMAPE\t\tRMSE')
    print(the_best[6][i])



if args.save:
    mod.save_checkpoint('STSGCN', epochs)
