import itertools
import json

import matplotlib.pyplot as plt


def plot_codebleu_vs_layer(data_list, checkpoints_to_include=None, metric="codebleu_optee", submetric="eval_metric"):
    """
    Plots a specified evaluation metric versus exit_index (layer) for specified checkpoints,
    including a distinct style for the baseline. Labels checkpoints as Epoch 1-X in ascending order.

    Parameters:
    - data_list: A list of dictionaries, where each dictionary contains
                 data related to a checkpoint, including the 'checkpoint',
                 'exit_index', and the evaluation metric.
    - checkpoints_to_include: A list of checkpoint names (strings) to include in the plot.
                              If None, all checkpoints are included.
    - metric: The top-level metric to extract (e.g., "codebleu_optee", "chrf_optee", etc.).
    - submetric: The specific sub-metric to extract from within the metric (e.g., "codebleu", "score", etc.).
    """
    checkpoint_data = {}
    baseline_data = None  # Variable to store baseline data

    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})

    for data in data_list:
        checkpoint = data['checkpoint']
        exit_index = data['exit_index']
        eval_metric = data[metric][submetric] if submetric is not None else data[metric]

        if checkpoint is None:
            if baseline_data is None:
                baseline_data = {'exit_index': [], 'eval_metric': []}
            baseline_data['exit_index'].append(exit_index)
            baseline_data['eval_metric'].append(eval_metric)
            continue

        if checkpoints_to_include and checkpoint not in checkpoints_to_include:
            continue

        if checkpoint not in checkpoint_data:
            checkpoint_data[checkpoint] = {'exit_index': [], 'eval_metric': []}

        checkpoint_data[checkpoint]['exit_index'].append(exit_index)
        checkpoint_data[checkpoint]['eval_metric'].append(eval_metric)

    sorted_checkpoints = sorted(
        (ckpt for ckpt in checkpoint_data.keys() if ckpt.startswith("checkpoint-")),
        key=lambda x: int(x.split('-')[1])
    )
    checkpoint_epochs = {ckpt: f"Epoch {i + 1}" for i, ckpt in enumerate(sorted_checkpoints)}

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'x', 'D']
    colors = plt.get_cmap('tab10')

    plt.figure(figsize=(10, 6))

    line_cycler = itertools.cycle(line_styles)
    marker_cycler = itertools.cycle(markers)

    if baseline_data:
        plt.plot(baseline_data['exit_index'], baseline_data['eval_metric'],
                 linestyle='-', marker='o', color='black', label='Base Model', linewidth=2)

    for idx, checkpoint in enumerate(sorted_checkpoints):
        values = checkpoint_data[checkpoint]
        plt.plot(values['exit_index'], values['eval_metric'],
                 linestyle=next(line_cycler),
                 marker=next(marker_cycler),
                 color=colors(idx),
                 label=checkpoint_epochs[checkpoint])

    plt.xlabel('Exit Index (Layer)')
    plt.ylabel(f'{submetric} Score')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True)

    plt.tight_layout()

    plt.show()


with open("XX", 'r') as f:
    data = json.load(f)

    plot_codebleu_vs_layer(data, metric="rouge_optee", submetric="rougeL")
