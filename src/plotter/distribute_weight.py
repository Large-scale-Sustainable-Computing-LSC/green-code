import matplotlib.pyplot as plt


def distribute_combined_budget(X, n_X, Y, n_Y):
    ratio_X = 0.9
    ratio_Y = 0.9

    shares_X = [ratio_X ** i for i in range(n_X)]
    total_X = sum(shares_X)
    normalized_shares_X = [(share / total_X) * X for share in shares_X]
    shares_Y = [ratio_Y ** i for i in range(n_Y)]
    total_Y = sum(shares_Y)
    normalized_shares_Y = [(share / total_Y) * Y for share in shares_Y]

    combined_shares = normalized_shares_X + normalized_shares_Y

    return combined_shares


plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 18})

X = 0.7  # Total budget for first group
Y = 0.2  # Total budget for second group
n_X = 6
n_Y = 3

combined_shares = distribute_combined_budget(X, n_X, Y, n_Y)
combined_shares.append(0.1)  # Last layer with fixed weight 0.1

print(combined_shares)
print(len(combined_shares))
print(sum(combined_shares))
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 26,
    "axes.titlesize": 26,
    "axes.labelsize": 26,
    "legend.fontsize": 26,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "lines.linewidth": 1.0,
    "figure.dpi": 300,
})
layers = [3, 5, 7, 9, 11, 13, 17, 21, 25, 27]

x_positions = list(range(1, len(combined_shares[:len(layers)]) + 1))
colors = ['lightgreen' if i < 6 else 'lightblue' for i in range(len(layers))]

plt.figure(figsize=(12, 8))
plt.bar(x_positions, combined_shares[:len(layers)], color=colors)

plt.xlabel('Layer ')
plt.ylabel('Weight')
plt.xticks(x_positions, [lay + 1 for lay in layers], rotation=0)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.legend(handles=[plt.Line2D([0], [0], color='lightgreen', lw=4, label='First half of layers'),
                    plt.Line2D([0], [0], color='lightblue', lw=4, label='Second half of layers')],
           loc='best')
plt.show()
