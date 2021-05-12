from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import setproctitle
setproctitle.setproctitle("stgcn@lifuxian")

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        data_type = args.config_filename.split('/')[-1].split('.')[0].split('_')[-1] #'bay' or 'la'
        supervisor = DCRNNSupervisor(data_type = data_type, LOAD_INITIAL = args.LOAD_INITIAL, adj_mx=adj_mx, **supervisor_config)

        if args.TEST_ONLY:
            supervisor.evaluate_test()
        else:
            supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--LOAD_INITIAL', default=False, type=bool, help='If LOAD_INITIAL.')
    parser.add_argument('--TEST_ONLY', default=False, type=bool, help='If TEST_ONLY.')
    args = parser.parse_args()
    main(args)
