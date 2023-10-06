import os
import torch
import argparse
from src.model.sparsemax_verticalMoe import SliceModel, SparseMax_VerticalSAMS
import time
import psycopg2
from src.model.factory import initialize_model
from typing import Any, List, Dict, Tuple
import json
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count_table

USER = "postgres"
HOST = "127.0.0.1"
PORT = "28814"
DB_NAME = "pg_extension"

time_dict = {
    "data_query_time": 0,
    "py_conver_to_tensor": 0,
    "tensor_to_gpu": 0,
    "py_compute": 0

}


def read_json(file_name):
    print(f"Loading {file_name}...")
    is_exist = os.path.exists(file_name)
    if is_exist:
        with open(file_name, 'r') as readfile:
            data = json.load(readfile)
        return data
    else:
        print(f"{file_name} is not exist")
        return {}


def fetch_and_preprocess(conn, batch_size, database):
    cur = conn.cursor()
    # Select rows greater than last_id
    cur.execute(f"SELECT * FROM {database}_train LIMIT {batch_size}")
    rows = cur.fetchall()
    return rows


def decode_libsvm(columns):
    map_func = lambda pair: (int(pair[0]), float(pair[1]))
    # 0 is id, 1 is label
    id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
    sample = {'id': list(id),
              'value': list(value),
              'y': int(columns[1])}
    return sample


def pre_processing(mini_batch_data: List[Tuple]):
    """
    mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
    """
    sample_lines = len(mini_batch_data)
    feat_id = []
    feat_value = []
    y = []

    for i in range(sample_lines):
        row_value = mini_batch_data[i]
        sample = decode_libsvm(row_value)
        feat_id.append(sample['id'])
        feat_value.append(sample['value'])
        y.append(sample['y'])
    feat_id = torch.LongTensor(feat_id)
    value_tensor = torch.FloatTensor(feat_value)
    y_tensor = torch.FloatTensor(y)
    return {'id': feat_id, 'value': value_tensor, 'y': y_tensor}


def fetch_data(database, batch_size):
    global time_dict
    print("Data fetching ....")
    begin_time = time.time()
    with psycopg2.connect(database=DB_NAME, user=USER, host=HOST, port=PORT) as conn:
        rows = fetch_and_preprocess(conn, batch_size, database)
    end_time = time.time()
    time_dict["data_query_time"] += end_time - begin_time
    print(f"Data fetching done {rows[0]}")

    print("Data preprocessing ....")
    begin_time = time.time()
    batch = pre_processing(rows)
    end_time = time.time()
    time_dict["py_conver_to_tensor"] += end_time - begin_time
    print("Data preprocessing done")
    return batch


def load_model(tensorboard_path: str, device: str = "cuda"):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)

    net = initialize_model(model_config)

    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path, map_location=device)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config


def if_cuda_avaiable(device):
    if "cuda" in device:
        return True
    else:
        return False


def reload_argparse(file_path: str):
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re

    return argparse.Namespace(**d)


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str,
                    help="directory to model file")
parser.add_argument('--flag', '-p', action='store_true',
                    help="wehther to print profile")
parser.add_argument('--print_net', '--b', action='store_true',
                    help="print the structure of network")

parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--dataset', type=str, default="frappe")
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--col_cardinalities_file', type=str, default="path to the stored file")

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    flag = args.flag
    device = args.device
    print(path)
    net, config = load_model(path, args.device)
    net: SparseMax_VerticalSAMS = net
    config.workload = 'random'

    print(config.workload)

    if config.net == "sparsemax_vertical_sams":
        alpha = net.sparsemax.alpha
        print(alpha)

    col_cardinalities = read_json(args.col_cardinalities_file)
    target_sql = torch.tensor([col[-1] for col in col_cardinalities]).reshape(1, -1)

    net.eval()
    net = net.to(device)
    with torch.no_grad():
        sql = target_sql.to(device)
        if config.net == "sparsemax_vertical_sams":
            subnet: SliceModel = net.tailor_by_sql(sql)
            subnet.to(device)
        else:
            subnet = net
        subnet.eval()
        target_list, y_list = [], []
        ops = 0

        for i in range(1):
            # fetch from db
            data_batch = fetch_data(args.dataset, args.batch_size)
            print("Copy to device")
            # wait for moving data to GPU
            begin = time.time()
            target = data_batch['y'].to(device)
            target_list.append(target)
            x_id = data_batch['id'].to(device)
            B = target.shape[0]
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["tensor_to_gpu"] += time.time() - begin

            print("begin to compute")
            # compute
            begin = time.time()
            y = subnet(x_id, None)
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["py_compute"] += time.time() - begin
    print(time_dict)
