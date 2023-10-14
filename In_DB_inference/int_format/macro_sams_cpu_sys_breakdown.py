import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.0f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.35
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 12
set_tick_size = 12
colors = ['#729ECE', '#8E44AD', '#2ECC71', '#3498DB', '#F39C12']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"
datasets_result = {
    'Adult': {
        'In-Db-opt': {'diff': -0.006811057999999814, 'data_query_time_spi': 0.099216721,
                      'mem_allocate_time': 0.006704392,
                      'model_init_time': 0.007573304, 'data_query_time': 0.937413457,
                      'python_compute_time': 2.878420642,
                      'overall_query_latency': 3.830218461, 'py_conver_to_tensor': 0.01761007308959961,
                      'py_compute': 2.603079080581665,
                      'py_overall_duration': 2.63140869140625, 'py_diff': 0.010719537734985352},

        'out-DB-cpu': {'data_query_time': 0.8945093154907227, 'py_conver_to_tensor': 3.1391489505767822,
                       'tensor_to_gpu': 0.000179290771484375, 'py_compute': 0.6464982032775879,
                       'overall_query_latency': 5.005170583724976},
    },

    'Disease': {
        'In-Db-opt': {'mem_allocate_time': 0.000241846, 'data_query_time_spi': 0.092643221,
                      'python_compute_time': 4.456881872, 'overall_query_latency': 7.531777533,
                      'data_query_time': 3.067697677, 'diff': -0.0003152109999993158, 'model_init_time': 0.006882773,
                      'py_conver_to_tensor': 2.6528313159942627, 'py_compute': 0.7840120792388916,
                      'py_overall_duration': 4.027993440628052, 'py_diff': 0.5911500453948975},

        'out-DB-cpu': {'data_query_time': 0.7599310874938965, 'py_conver_to_tensor': 2.712991952896118,
                       'tensor_to_gpu': 0.0004315376281738281, 'py_compute': 0.7755249500274658,
                       'overall_query_latency': 5.472174644470215},
    },

    'Bank': {
        'In-Db-opt': {'data_query_time': 3.9064207829999997, 'python_compute_time': 4.978618743,
                      'data_query_time_spi': 0.115038494, 'mem_allocate_time': 0.000246575,
                      'overall_query_latency': 8.893539688, 'diff': -0.0003386900000013071,
                      'model_init_time': 0.008161472, 'py_conver_to_tensor': 2.878143072128296,
                      'py_compute': 0.8705038928985596, 'py_overall_duration': 4.330329895019531,
                      'py_diff': 0.8216829299926758},
        'out-DB-cpu': {'data_query_time': 0.9924757480621338, 'py_conver_to_tensor': 2.880948085784912,
                       'tensor_to_gpu': 0.00011372566223144531, 'py_compute': 0.8873722553253174,
                       'overall_query_latency': 5.279063701629639},
    },

    'AppRec': {
        'In-Db-opt':
            {'python_compute_time': 1.8740051690000001, 'overall_query_latency': 2.443160077,
             'data_query_time_spi': 0.093307434, 'diff': -0.006727488999999753, 'model_init_time': 0.00768204,
             'mem_allocate_time': 0.006639138, 'data_query_time': 0.554745379,
             'py_conver_to_tensor': 0.015256166458129883, 'py_compute': 1.6671726703643799,
             'py_overall_duration': 1.692652702331543, 'py_diff': 0.010223865509033203}
        ,
        'out-DB-cpu':
            {'data_query_time': 0.5377492904663086, 'py_conver_to_tensor': 0.01552491569519043,
             'tensor_to_gpu': 9.989738464355469e-05, 'py_compute': 1.6639730834960938,
             'load_model': 0.16022920608520508, 'overall_query_latency': 2.189664125442505},
    },
}

datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True
set_label_in_db_model_load = True

indices = []
index = 0
for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_in_db_model_load = 'Model Loading' if set_label_in_db_model_load else None
    label_in_db_data_query = 'Data Retrieval' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copying' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocessing' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_model_load = indb_med_opt["model_init_time"]
    in_db_data_copy_start_py = indb_med_opt["python_compute_time"] - indb_med_opt["py_overall_duration"]
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    in_db_data_compute = indb_med_opt["py_compute"]

    ax.bar(index + bar_width / 2, in_db_data_model_load, bar_width, color=colors[4], hatch=hatches[4], zorder=2,
           label=label_in_db_model_load,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_data_query,
           bottom=in_db_data_model_load,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_model_load,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_preprocess + in_db_data_compute, bar_width, color=colors[2],
           hatch=hatches[2], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_model_load,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width / 2, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_model_load,
           label=label_in_db_data_compute, edgecolor='black')

    # # out-db CPU
    out_db_data_query = outcpudb_med["data_query_time"]
    out_db_data_preprocess = outcpudb_med["py_conver_to_tensor"]
    out_db_data_compute = outcpudb_med["py_compute"]

    ax.bar(index + bar_width / 2, in_db_data_model_load, bar_width, color=colors[4], hatch=hatches[4], zorder=2,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           bottom=in_db_data_model_load,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=out_db_data_query+in_db_data_model_load,
           edgecolor='black')
    ax.bar(index - bar_width / 2, out_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=out_db_data_query + in_db_data_model_load + out_db_data_preprocess,
           edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False
    set_label_in_db_model_load = False

    index += 1

# legned etc
ax.set_ylabel(".", fontsize=20, color='white')
fig.text(0.01, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

ax.set_ylim(top=6000)

ax.set_xticks(indices)
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/int_format/macro.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/int_format/macro.pdf",
            bbox_inches='tight')
