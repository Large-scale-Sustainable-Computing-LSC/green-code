from matplotlib import pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})
energy_rl = "X"
energy_cls = "X"
energy_lmhead = "X"

time_rl = "X"
time_cls = "X"
time_lmhead = "X"

methods = ['RL Policy', 'Classifier', 'LMHead Confidence']

energy_means = [energy_rl[0], energy_cls[0], energy_lmhead[0]]
energy_stds = [energy_rl[1], energy_cls[1], energy_lmhead[1]]

time_means = [time_rl[0], time_cls[0], time_lmhead[0]]
time_stds = [time_rl[1], time_cls[1], time_lmhead[1]]

plt.figure(figsize=(10, 6))
plt.bar(methods, energy_means, yerr=energy_stds, capsize=5, alpha=0.7, label='Energy')
plt.ylabel('Energy (mean ± std)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("energy_plot.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(methods, time_means, yerr=time_stds, capsize=5, alpha=0.7, color='orange', label='Time')
plt.ylabel('Time (mean ± std)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("time_plot.png")
plt.show()
