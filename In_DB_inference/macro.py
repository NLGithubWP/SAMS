import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter
from brokenaxes import brokenaxes


def thousands_formatter(x, pos):
    if x >= 1e3:
        return '{:.1f}k'.format(x * 1e-3)
    else:
        return '{:.1f}'.format(x)


thousands_format = FuncFormatter(thousands_formatter)


# Helper function to load data
def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


# Set your plot parameters
bar_width = 0.25
opacity = 0.8
set_font_size = 15  # Set the font size
set_lgend_size = 12
set_tick_size = 12
colors = ['#729ECE', '#2ECC71', '#8E44AD', '#3498DB', '#F39C12']
hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


def scale_to_ms(latencies):
    result = {}
    for key, value in latencies.items():
        value = value * 1000
        result[key] = value
    return result


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"

# Collecting data for plotting
datasets_result = {
    'Frappe': {
        'In-Db-opt':
            {'data_query_time_spi': 0.012790853, 'python_compute_time': 0.520475877,
             'overall_query_latency': 0.748851507, 'data_query_time': 0.221231879, 'diff': -0.00030915300000000645,
             'model_init_time': 0.006834598, 'mem_allocate_time': 0.000235295,
             'py_conver_to_tensor': 0.24206132888793945, 'py_compute': 0.12624335289001465,
             'py_overall_duration': 0.46413350105285645, 'py_diff': 0.045828819274902344},

        'out-DB-cpu':
            {'data_query_time': 0.11190366744995117, 'py_conver_to_tensor': 0.22279024124145508,
             'tensor_to_gpu': 0.00026679039001464844, 'py_compute': 0.12605422019958496,
             'overall_query_latency': 0.6827912330627441},
        'out-DB-gpu':
            {'data_query_time': 0.1079864501953125, 'py_conver_to_tensor': 0.2427043914794922,
             'tensor_to_gpu': 19.78896689414978, 'py_compute': 0.059970855712890625,
             'overall_query_latency': 20.467687845230103},
    },

    'Adult': {
        'In-Db-opt': {'overall_query_latency': 0.920471622, 'diff': -0.0004153080000000031,
                      'data_query_time_spi': 0.014744383, 'mem_allocate_time': 0.000303075,
                      'python_compute_time': 0.55004726, 'model_init_time': 0.008975202, 'data_query_time': 0.361033852,
                      'py_conver_to_tensor': 0.2757132053375244, 'py_compute': 0.18352606201171875,
                      'py_overall_duration': 0.4639158248901367, 'py_diff': 0.0929419994354248},
        'out-DB-cpu': {'data_query_time': 0.10367798805236816, 'py_conver_to_tensor': 0.2796957492828369,
                       'tensor_to_gpu': 9.942054748535156e-05, 'py_compute': 0.18311119079589844,
                       'overall_query_latency': 0.6310579776763916},
        'out-DB-gpu': {'data_query_time': 0.09808635711669922, 'py_conver_to_tensor': 0.27641721725463867,
                       'tensor_to_gpu': 11.821438789367676, 'py_compute': 0.011312007904052734,
                       'overall_query_latency': 12.34174108505249},
    },

    'Cvd': {
        'In-Db-opt': {'python_compute_time': 0.794361108, 'model_init_time': 0.008119253,
                      'overall_query_latency': 1.122600982, 'data_query_time_spi': 0.012937076,
                      'diff': -0.0003322129999998591, 'data_query_time': 0.319788408, 'mem_allocate_time': 0.000243092,
                      'py_conver_to_tensor': 0.2250985336303711, 'py_compute': 0.21205615997314453,
                      'py_overall_duration': 0.7133233547210693, 'py_diff': 0.04838967323303223},

        'out-DB-cpu': {'data_query_time': 0.11284756660461426, 'py_conver_to_tensor': 0.2251596450805664,
                       'tensor_to_gpu': 0.00022149085998535156, 'py_compute': 0.21241354942321777,
                       'overall_query_latency': 0.6715552806854248},
        'out-DB-gpu': {'data_query_time': 0.1045682430267334, 'py_conver_to_tensor': 0.2194075584411621,
                       'tensor_to_gpu': 19.62585687637329, 'py_compute': 0.052184343338012695,
                       'overall_query_latency': 20.205774307250977},
    },

    'Bank': {
        'In-Db-opt': {'model_init_time': 0.008176494, 'python_compute_time': 0.683411322,
                      'data_query_time_spi': 0.015905248, 'data_query_time': 0.29132877,
                      'diff': -0.00032423100000000815, 'mem_allocate_time': 0.000232581,
                      'overall_query_latency': 0.983240817, 'py_conver_to_tensor': 0.30087942123413086,
                      'py_compute': 0.1244482707977295, 'py_overall_duration': 0.5877652168273926,
                      'py_diff': 0.07243752479553223},
        'out-DB-cpu': {'data_query_time': 0.14289021492004395, 'py_conver_to_tensor': 0.30756115913391113,
                       'tensor_to_gpu': 0.0002760887145996094, 'py_compute': 0.11214518547058105,
                       'overall_query_latency': 0.6774318218231201},
        'out-DB-gpu': {'data_query_time': 0.1385364532470703, 'py_conver_to_tensor': 0.29821038246154785,
                       'tensor_to_gpu': 20.349432706832886, 'py_compute': 0.045266151428222656,
                       'overall_query_latency': 21.015833854675293},
    },
}

datasets = list(datasets_result.keys())

# Plotting
fig = plt.figure(figsize=(6.4, 4.5))

# Create a broken y-axis within the fig
ax = brokenaxes(ylims=((0, 700), (12150, 12260), (19920, 20200), (20750, 21000)), hspace=.25, fig=fig, d=0)

# Initial flags to determine whether the labels have been set before
set_label_in_db_data_query = True
set_label_in_db_data_copy_start_py = True
set_label_in_db_data_preprocess = True
set_label_in_db_data_compute = True
set_label_in_db_data_others = True

indices = []
index = 0
for dataset, valuedic in datasets_result.items():
    indices.append(index)

    indb_med_opt = scale_to_ms(valuedic["In-Db-opt"])
    outgpudb_med = scale_to_ms(valuedic["out-DB-gpu"])
    outcpudb_med = scale_to_ms(valuedic["out-DB-cpu"])

    # set labesl
    label_in_db_data_query = 'Data Retrievl' if set_label_in_db_data_query else None
    label_in_db_data_copy_start_py = 'Data Copy' if set_label_in_db_data_copy_start_py else None
    label_in_db_data_preprocess = 'Data Preprocess' if set_label_in_db_data_preprocess else None
    label_in_db_data_compute = 'Model Inference' if set_label_in_db_data_compute else None
    label_in_db_data_others = 'Others' if set_label_in_db_data_others else None

    # in-db with optimizization
    in_db_data_copy_start_py = 0
    in_db_data_query = indb_med_opt["data_query_time_spi"]
    in_db_data_preprocess = indb_med_opt["py_conver_to_tensor"]
    # + indb_med_opt["python_compute_time"] \
    # - indb_med_opt["py_overall_duration"]
    in_db_data_compute = indb_med_opt["py_compute"]
    in_db_data_others = indb_med_opt["overall_query_latency"] - \
                        indb_med_opt["data_query_time"] - \
                        in_db_data_copy_start_py - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index + bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           label=label_in_db_data_query, edgecolor='black')
    # ax.bar(index, in_db_data_copy_start_py, bar_width, color=colors[1], hatch=hatches[1], zorder = 2,
    #        bottom=in_db_data_query,
    #        edgecolor='black')
    ax.bar(index + bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py,
           label=label_in_db_data_preprocess, edgecolor='black')
    ax.bar(index + bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess,
           label=label_in_db_data_compute, edgecolor='black')
    # ax.bar(index + bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_start_py + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # out-db GPU
    in_db_data_query = outgpudb_med["data_query_time"]
    in_db_data_copy_gpu = outgpudb_med["tensor_to_gpu"]
    in_db_data_preprocess = outgpudb_med["py_conver_to_tensor"]
    in_db_data_compute = outgpudb_med["py_compute"]
    in_db_data_others = outgpudb_med["overall_query_latency"] - \
                        in_db_data_query - \
                        in_db_data_copy_gpu - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index - bar_width, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_preprocess,
           label=label_in_db_data_copy_start_py,
           edgecolor='black')
    ax.bar(index - bar_width, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index -  bar_width, in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # # out-db CPU
    in_db_data_query = outcpudb_med["data_query_time"]
    in_db_data_copy_gpu = outcpudb_med["tensor_to_gpu"]
    in_db_data_preprocess = outcpudb_med["py_conver_to_tensor"]
    in_db_data_compute = outcpudb_med["py_compute"]
    in_db_data_others = outcpudb_med["overall_query_latency"] - \
                        in_db_data_query - \
                        in_db_data_copy_gpu - \
                        in_db_data_preprocess - \
                        in_db_data_compute

    ax.bar(index, in_db_data_query, bar_width, color=colors[0], hatch=hatches[0], zorder=2,
           edgecolor='black')
    ax.bar(index, in_db_data_preprocess, bar_width, color=colors[2], hatch=hatches[2], zorder=2,
           bottom=in_db_data_query,
           edgecolor='black')
    ax.bar(index, in_db_data_copy_gpu, bar_width, color=colors[1], hatch=hatches[1], zorder=2,
           bottom=in_db_data_query + in_db_data_preprocess,
           edgecolor='black')
    ax.bar(index, in_db_data_compute, bar_width, color=colors[3], hatch=hatches[3], zorder=2,
           bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess,
           edgecolor='black')
    # ax.bar(index , in_db_data_others, bar_width, color=colors[4], hatch=hatches[4], zorder = 2,
    #        bottom=in_db_data_query + in_db_data_copy_gpu + in_db_data_preprocess + in_db_data_compute,
    #        edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_in_db_data_query = False
    set_label_in_db_data_copy_start_py = False
    set_label_in_db_data_preprocess = False
    set_label_in_db_data_compute = False
    set_label_in_db_data_others = False

    index += 1

ax.set_ylabel(".", fontsize=20, color='white')
fig.text(-0.05, 0.5, 'End-to-end Time (ms)', va='center', rotation='vertical', fontsize=20)

# ax.set_ylim(top=1600)

for sub_ax in ax.axs:
    sub_ax.set_xticks(indices)
    sub_ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)

# ax.legend(fontsize=set_lgend_size - 2, ncol=2, )
ax.legend(fontsize=set_lgend_size - 2, ncol=2, loc='upper left')

# Since the yaxis formatter is tricky with brokenaxes, you might need to set it for the actual underlying axes:
for ax1 in ax.axs:
    ax1.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

ax.grid(True, zorder=1)  # grid in front of bars

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/macro.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/macro.pdf",
            bbox_inches='tight')
