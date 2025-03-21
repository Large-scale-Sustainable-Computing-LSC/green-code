import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

keys = [3, 5, 7, 9, 11, 13, 17, 21, 25, 27]
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})


def parse_file(file_path):
    means = []
    occurrences = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Mean of values for key"):
                try:
                    mean_part = line.split(": ")[1].split("+-")[0].strip()  # Extract mean
                    occ_part = line.split("number of occ ")[1].strip()  # Extract occurrences

                    means.append(float(mean_part))
                    occurrences.append(int(occ_part))
                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed line: {line.strip()} - Error: {e}")

    return means, occurrences


means, occurrences = parse_file("XX")
weighted_mean = np.average(means, weights=occurrences)

norm = plt.Normalize(min(occurrences), max(occurrences))
colors = cm.viridis(norm(occurrences))

x_positions = np.arange(len(keys))

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(x_positions, means, color=colors, alpha=0.8)

for x, bar, occ in zip(x_positions, bars, occurrences):
    height = bar.get_height()
    ax.text(x, height, f'{occ}', ha='center', va='bottom')

ax.axhline(weighted_mean, color='red', linestyle='--', linewidth=2, label=f'Weighted Mean = {weighted_mean:.3f}')

ax.set_xticks(x_positions)
ax.set_xticklabels(keys)

ax.set_xlabel('Exit Index')
ax.set_ylabel('% exits that lead to a correct prediction')
ax.grid(axis='y', linestyle='--', alpha=0.5)

ax.legend()

sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Number of Occurrences')

plt.show()
