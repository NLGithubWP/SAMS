import calendar
import os
import time
import json
import traceback
import orjson
from argparse import Namespace
from model_selection.shared_config import parse_config_arguments

import random
import torch
from tqdm import tqdm


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


def exception_catcher(func):
    def wrapper(encoded_str: str):
        try:
            # each functon accepts a json string
            params = json.loads(encoded_str)
            config_file = params.get("config_file")

            # Parse the config file
            args = parse_config_arguments(config_file)

            # Set the environment variables
            ts = calendar.timegm(time.gmtime())
            os.environ.setdefault("base_dir", args.base_dir)
            os.environ.setdefault("log_logger_folder_name", args.log_folder)
            os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")

            # Call the original function with the parsed parameters
            return func(params, args)
        except Exception as e:
            return orjson.dumps(
                {"Errored": traceback.format_exc()}).decode('utf-8')

    return wrapper


# Micro benchmarking filterting phaes
model = None
sliced_model = None
col_cardinalities = None


@exception_catcher
def model_inference_load_model(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities
    from model_selection.src.logger import logger
    try:
        logger.info(f"Received parameters: {params}")

        from src.data_loader import sql_attached_dataloader
        from profile_model import load_model
        # read saved col_cardinatlites file
        if col_cardinalities is None:
            col_cardinalities = read_json(params["col_cardinalities_file"])

        # {'col_cardinalities_file': '/project/TRAILS/frappe_col_cardinalities',
        #  'model_path': '/project/tensor_log/frappe/dnn_K16_alpha4',
        #  'where_cond': '{"1":266, "2":1244}',
        #  'config_file': '/project/TRAILS/internal/ml/model_selection/config.ini'}
        #
        # read the model path,
        model_path = params["model_path"]
        # get the where condition
        where_cond = json.loads(params["where_cond"])

        # generate default sql and selected sql
        target_sql = [col[-1] for col in col_cardinalities]
        for col_index, value in where_cond.items():
            target_sql[col_index] = value

        logger.info(f"target_sql is: {target_sql}")

        # if model is None:
        #     logger.info("Load model .....!")
        #     model, config = load_model(model_path)
        #     sliced_model = model.tailor_by_sql(torch.tensor(target_sql))
        #     sliced_model.eval()
        # else:
        #     logger.info("Skip Load model")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))
    return orjson.dumps({"ok": 1}).decode('utf-8')


@exception_catcher
def model_inference_compute(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities
    from model_selection.src.logger import logger
    mini_batch = json.loads(params["mini_batch"])
    logger.info(f"Received parameters: {mini_batch}")
    # begin = time.time()
    # y = mini_batch(mini_batch, None)
    # duration = time.time() - begin
    # logger.info(f"time usage for compute {len(mini_batch)} rows is {duration}")
    return orjson.dumps({"model_outputs": 1}).decode('utf-8')
