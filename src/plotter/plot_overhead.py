import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_mean_energies(file1_path, file2_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    mean_energies_file1 = df1.groupby('window_name')['gpu0_energy'].mean()
    std_energies_file1 = df1.groupby('window_name')['gpu0_energy'].std()

    mean_energy_full_token_file2 = df2[df2['window_name'] == 'FULL-TOKEN']['gpu0_energy'].mean()
    std_energy_full_token_file2 = df2[df2['window_name'] == 'FULL-TOKEN']['gpu0_energy'].std()

    labels_file1 = ['Full Token', 'KV', 'RL']
    means_file1 = [
        mean_energies_file1.get('FULL-TOKEN', 0),
        mean_energies_file1.get('KV', 0),
        mean_energies_file1.get('RL', 0)
    ]
    stds_file1 = [
        std_energies_file1.get('FULL-TOKEN', 0),
        std_energies_file1.get('KV', 0),
        std_energies_file1.get('RL', 0)
    ]

    label_file2 = 'Full token (all layers)'
    mean_file2 = mean_energy_full_token_file2
    std_file2 = std_energy_full_token_file2

    x = np.arange(len(labels_file1) + 1)
    means = means_file1 + [mean_file2]
    stds = stds_file1 + [std_file2]

    plt.figure(figsize=(10, 6))
    plt.bar(x[:-1], means_file1, capsize=5, color=['blue', 'orange', 'green'])
    plt.bar(x[-1], mean_file2, capsize=5, color='red')

    plt.xticks(x, labels_file1 + [label_file2])
    plt.xlabel('Window Name')
    plt.ylabel('Mean GPU Energy Consumption (Ws)')
    # plt.title('Mean Energy Consumption ')
    plt.show()
