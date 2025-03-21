import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 32,
    "axes.labelsize": 32,
    "legend.fontsize": 32,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})


def extract_tensorboard_scalars(log_dir, scalar_name):
    """
    Extracts a scalar from TensorBoard log files.

    Args:
        log_dir (str): Path to the directory containing TensorBoard log files.
        scalar_name (str): Name of the scalar to extract (e.g., 'train/loss').

    Returns:
        list: A tuple of (steps, values).
    """
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if scalar_name not in event_acc.scalars.Keys():
        raise ValueError(f"Scalar '{scalar_name}' not found in TensorBoard logs.")

    scalar_events = event_acc.Scalars(scalar_name)
    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    return steps, values


def plot_scalar(steps, values, title, xlabel, ylabel, save_path=None):
    """
    Plots a scalar with Matplotlib.

    Args:
        steps (list): List of steps (x-axis).
        values (list): List of values (y-axis).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str): If provided, save the plot to this path.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(steps, values, label=ylabel, color="blue", linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_scalar2(steps, values, steps2, values2, title, xlabel, ylabel, save_path=None):
    """
    Plots a scalar with Matplotlib.

    Args:
        steps (list): List of steps (x-axis).
        values (list): List of values (y-axis).
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        save_path (str): If provided, save the plot to this path.
    """
    plt.figure(figsize=(14, 8))
    plt.plot(steps, values, label="LLAMA", color="blue", linewidth=2)
    plt.plot(steps2, values2, label="OPT", color="red", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
