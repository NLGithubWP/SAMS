import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter


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
colors = ['#729ECE', '#FFB579', '#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#8E44AD', '#C0392B']

hatches = ['/', '\\', 'x', '.', '*', '//', '\\\\', 'xx', '..', '**']


# hatches = ['', '', '', '', '']

def get_median(latencies):
    if len(latencies) > 0:
        compute_values = [entry['compute'] for entry in latencies]
        data_fetch_values = [float(entry['data_fetch']) for entry in latencies]

        median_compute = np.quantile(compute_values, 0.5)
        median_data_fetch = np.quantile(data_fetch_values, 0.5)

        return {'median_compute': median_compute*100, 'median_data_fetch': median_data_fetch*100}
    else:
        return {'median_compute': 0, 'median_data_fetch': 0}


# here run 10k rows for inference.,
# each sub-list is "compute time" and "data fetch time"
datasets_result = {
    'Frappe': {
        'In-Db': [
            {'compute': 0.17173027992248535, 'data_fetch': '0.251954086'},
            {'compute': 0.07187461853027344, 'data_fetch': '0.284413449'},
            {'compute': 0.07067489624023438, 'data_fetch': '0.28681375'},
            {'compute': 0.15250825881958008, 'data_fetch': '0.289194581'},
            {'compute': 0.0789792537689209, 'data_fetch': '0.289361299'},
            {'compute': 0.0722496509552002, 'data_fetch': '0.268448956'},
            {'compute': 0.4905436038970947, 'data_fetch': '0.3234656'},
            {'compute': 0.08709049224853516, 'data_fetch': '0.289736669'},
            {'compute': 0.07081151008605957, 'data_fetch': '0.286487382'},
            {'compute': 0.08495402336120605, 'data_fetch': '0.263012516'},
        ],
        'out-DB-gpu': [],
        'out-DB-cpu': [],
    },

    'Adult': {
        'In-Db': [
            {'compute': 0.09690213203430176, 'data_fetch': '0.352971475'},
            {'compute': 0.09694743156433105, 'data_fetch': '0.37523442'},
            {'compute': 0.05006861686706543, 'data_fetch': '0.344217125'},
            {'compute': 0.06653904914855957, 'data_fetch': '0.285121514'},
            {'compute': 0.08176016807556152, 'data_fetch': '0.312680154'},
            {'compute': 0.045957326889038086, 'data_fetch': '0.362333267'},
            {'compute': 0.05000185966491699, 'data_fetch': '0.366499086'},
            {'compute': 0.06590700149536133, 'data_fetch': '0.289758388'},
            {'compute': 0.06353759765625, 'data_fetch': '0.273682989'},
            {'compute': 0.07068490982055664, 'data_fetch': '0.375006159'}
        ],
        'out-DB-gpu': [],
        'out-DB-cpu': [],
    },

    'Cvd': {
        'In-Db': [
            {'compute': 0.19356274604797363, 'data_fetch': '0.342282083'},
            {'compute': 0.12594223022460938, 'data_fetch': '0.296606721'},
            {'compute': 0.14561080932617188, 'data_fetch': '0.338163932'},
            {'compute': 0.2633681297302246, 'data_fetch': '0.311251902'},
            {'compute': 0.10169219970703125, 'data_fetch': '0.325908049'},
            {'compute': 0.12447810173034668, 'data_fetch': '0.344278303'},
            {'compute': 0.20384788513183594, 'data_fetch': '0.279193229'},
            {'compute': 0.14433860778808594, 'data_fetch': '0.220627192'},
            {'compute': 0.3059248924255371, 'data_fetch': '0.229409827'},
            {'compute': 0.21106886863708496, 'data_fetch': '0.346429639'}
        ],
        'out-DB-gpu': [],
        'out-DB-cpu': [],
    },

    'Bank': {
        'In-Db': [
            {'compute': 0.14467668533325195, 'data_fetch': '0.464980583'},
            {'compute': 0.08016133308410645, 'data_fetch': '0.292378782'},
            {'compute': 0.09622573852539062, 'data_fetch': '0.45849089'},
            {'compute': 0.08400511741638184, 'data_fetch': '0.447140387'},
            {'compute': 0.10208964347839355, 'data_fetch': '0.291694676'},
            {'compute': 0.09836769104003906, 'data_fetch': '0.404657997'},
            {'compute': 0.08470940589904785, 'data_fetch': '0.4247317'},
            {'compute': 0.08762812614440918, 'data_fetch': '0.343508828'},
            {'compute': 0.08430290222167969, 'data_fetch': '0.314511488'},
            {'compute': 0.09432816505432129, 'data_fetch': '0.429916167'}
        ],
        'out-DB-gpu': [],
        'out-DB-cpu': [],
    },
}


# Collecting data for plotting
datasets = list(datasets_result.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6.4, 4.5))

index = np.arange(len(datasets))
# Initial flags to determine whether the labels have been set before
set_label_outdb_gpu_data = True
set_label_outdb_gpu_inference = True
set_label_outdb_cpu_data = True
set_label_outdb_cpu_inference = True
set_label_indb_data = True
set_label_indb_inference = True

for dataset, valuedic in datasets_result.items():
    indb_med = get_median(valuedic["In-Db"])
    outgpudb_med = get_median(valuedic["out-DB-gpu"])
    outcpudb_med = get_median(valuedic["out-DB-cpu"])

    # Left bar - out-db GPU
    label_outdb_gpu_data = '(Out-DB/GPU) Data Retrievl' if set_label_outdb_gpu_data else None
    label_outdb_gpu_inference = '(Out-DB/GPU) Compute' if set_label_outdb_gpu_inference else None
    ax.bar(index + bar_width , indb_med["median_data_fetch"], bar_width, color=colors[0], hatch=hatches[0],
           label=label_outdb_gpu_data, edgecolor='black')
    ax.bar(index + bar_width , indb_med["median_compute"], bar_width, color=colors[1], hatch=hatches[1],
           bottom=indb_med["median_data_fetch"], label=label_outdb_gpu_inference, edgecolor='black')

    # Right bar - out-db CPU
    label_outdb_cpu_data = '(Out-DB/CPU) Data Retrievl' if set_label_outdb_cpu_data else None
    label_outdb_cpu_inference = '(Out-DB/CPU) Compute' if set_label_outdb_cpu_inference else None
    ax.bar(index, outgpudb_med["median_data_fetch"], bar_width, color=colors[2], hatch=hatches[2],
           label=label_outdb_cpu_data, edgecolor='black')
    ax.bar(index , outgpudb_med["median_compute"], bar_width, color=colors[3], hatch=hatches[3],
           bottom=outgpudb_med["median_data_fetch"], label=label_outdb_cpu_inference, edgecolor='black')

    # Right bar - in-db
    label_indb_data = '(In-DB/CPU) Data Retrievl' if set_label_indb_data else None
    label_indb_inference = '(Out-DB/GPU) Compute' if set_label_indb_inference else None
    ax.bar(index - bar_width , outcpudb_med["median_data_fetch"], bar_width, color=colors[4], hatch=hatches[4],
           label=label_indb_data, edgecolor='black')
    ax.bar(index - bar_width , outcpudb_med["median_compute"], bar_width, color=colors[5], hatch=hatches[5],
           bottom=outcpudb_med["median_data_fetch"], label=label_indb_inference, edgecolor='black')

    # Update the flags to ensure the labels are not set again in the next iterations
    set_label_outdb_gpu_data = False
    set_label_outdb_gpu_inference = False
    set_label_outdb_cpu_data = False
    set_label_outdb_cpu_inference = False
    set_label_indb_data = False
    set_label_indb_inference = False

ax.set_ylabel('Inference Time (ms)', fontsize=20)
ax.set_xticks(index)
# ax.set_yscale('log')  # Set y-axis to logarithmic scale
ax.set_xticklabels(datasets, rotation=0, fontsize=set_font_size)
ax.legend(fontsize=set_lgend_size, loc=2)
ax.yaxis.set_major_formatter(thousands_format)

ax.tick_params(axis='y', which='major', labelsize=set_tick_size + 5)

plt.tight_layout()
fig.tight_layout()
# plt.show()
print(f"saving to ./internal/ml/model_slicing/In_DB_inference/filter_latency_memory_bar.pdf")
fig.savefig(f"./internal/ml/model_slicing/In_DB_inference/filter_latency_memory_bar.pdf",
            bbox_inches='tight')
