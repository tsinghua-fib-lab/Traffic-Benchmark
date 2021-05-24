from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

def generate_train_val_test(df):    
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    return x, y


def main(args):
    print("Generating training data")
    df = pd.read_csv(args.traffic_df_filename, header = 0, index_col = 0)
    df.index = pd.to_datetime(df.index)
    list_index = [pd.date_range('2020-07-01 00:00:00', '2020-07-03 23:55:00', freq = '5min') ,pd.date_range('2020-07-06 00:00:00', '2020-07-10 23:55:00', freq = '5min') ,pd.date_range('2020-07-13 00:00:00', '2020-07-17 23:55:00', freq = '5min') , pd.date_range('2020-07-20 00:00:00', '2020-07-24 23:55:00', freq = '5min'), pd.date_range('2020-07-27 00:00:00', '2020-07-31 23:55:00', freq = '5min')]
    list_x = []
    list_y = []
    for idx in list_index:
        df_i = df.loc[idx]
        x, y = generate_train_val_test(df_i)
        list_x.append(x)
        list_y.append(y)
    x = np.concatenate(list_x, axis=0)
    y = np.concatenate(list_y, axis=0)
    print(x.shape, y.shape) #(6509, 12, 500, 2) (6509, 12, 500, 2)

    num_samples = x.shape[0]
    #15：3：5
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_test = round(num_samples * 5/23) #1415
    num_train = round(num_samples * 15/23) 
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="NE-BJ/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/NE-BJ.csv",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
