import json

import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, path_to_results, contexts, num_samples, max_new, thresholds):
        self.path_to_results = path_to_results
        self.contexts = contexts
        self.num_samples = num_samples
        self.max_new = max_new
        self.thresholds = thresholds

    def plot_results(self):
        for ctx in self.contexts:
            base_results = self._load_json(
                f'eval_base_model_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')
            results_all_layers = self._load_json(
                f'eval_all_layers_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

            for thr in self.thresholds:
                results = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

                layers = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java_exit_layers.json')

                metrics_list = [base_results, results_all_layers, results]
                self.compare_codebleu_energy(metrics_list, None, base="Base Model",
                                             title=f"Context: {ctx} Threshold: {thr}")

    def _load_json(self, filename):
        with open(f"{self.path_to_results}/{filename}", 'r') as f:
            return json.load(f)

    def compare_codebleu_energy(self, metrics_list, rl_agent_energy=None, base="Base Model", title=""):
        metrics = ['codebleu']
        energy_metric = ['energy (kWh)']

        model1_codebleu = [metrics_list[0][0]['codebleu']]  # Base model CodeBLEU
        model2_codebleu = [metrics_list[1][0]['codebleu']]  # All layers CodeBLEU
        model3_codebleu = [metrics_list[2][0]['codebleu']]  # With RL Agent CodeBLEU

        model1_energy_kwh = metrics_list[0][5] * 2.77778e-7  # Base model energy
        print(metrics_list[1][5])
        model2_energy_kwh = metrics_list[1][5] * 2.77778e-7  # All layers energy
        model3_energy_kwh = metrics_list[2][5] * 2.77778e-7  # With RL Agent energy

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(metrics))
        width = 0.2

        ax.bar(x - width, model1_codebleu, width, label=base)  # Base model
        ax.bar(x, model2_codebleu, width, label='All layers (finetuned)')
        ax.bar(x + width, model3_codebleu, width, label='With RL Agent')

        ax.set_ylabel('CodeBLEU Score')
        ax.set_title('Comparison of CodeBLEU Between Three Models')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 8))
        x_energy = np.arange(len(energy_metric))

        if rl_agent_energy:
            rl_agent_energy_kwh = rl_agent_energy * 1e-3 * 2.77778e-7
            model_with_rl_energy = model3_energy_kwh + rl_agent_energy_kwh
            ax.bar(x_energy - width, model1_energy_kwh, width, label=base)
            ax.bar(x_energy, model2_energy_kwh, width, label='All layers')
            ax.bar(x_energy + width, [model_with_rl_energy], width, label='With RL Agent (Model Energy)',
                   color='lightblue')
            ax.bar(x_energy + width, [rl_agent_energy_kwh], width, bottom=[model_with_rl_energy],
                   label='RL Agent Energy', color='orange')
        else:
            ax.bar(x_energy - width, model1_energy_kwh, width, label=base)
            ax.bar(x_energy, model2_energy_kwh, width, label='All layers')
            ax.bar(x_energy + width, model3_energy_kwh, width, label='With RL Agent')

        ax.set_ylabel('Energy (kWh)')
        ax.set_title(title)
        ax.set_xticks(x_energy)
        ax.set_xticklabels(energy_metric)
        ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_layer_numbers_bar(self, relative_frq=True):
        for ctx in self.contexts:

            for thr in self.thresholds:
                layer_list = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java_exit_layers.json')

                layer_counts = {}
                for layer in layer_list:
                    if layer in layer_counts:
                        layer_counts[layer] += 1
                    else:
                        layer_counts[layer] = 1

                layer_counts = dict(sorted(layer_counts.items()))

                fig, ax = plt.subplots(figsize=(12, 8))
                x = np.arange(len(layer_counts.keys()))
                width = 0.2
                if relative_frq:
                    ax.bar(x, [val / len(layer_list) for val in layer_counts.values()], width, label='Layer Counts')
                else:
                    ax.bar(x, layer_counts.values(), width, label='Layer Counts')

                ax.set_ylabel('Relative Frequency' if relative_frq else 'Count')
                ax.set_title(f'Layer Counts for Context: {ctx} Threshold: {thr}')
                ax.set_xticks(x)
                ax.set_xticklabels(layer_counts.keys())
                ax.legend()
                fig.tight_layout()
                plt.show()

    def plot_codebleu_energy(self):
        codebleu_scores = []
        total_energies = []
        baseline_codebleu = None
        baseline_total_energy = None
        all_layers_codebleu = []
        all_layers_energy = []

        for ctx in self.contexts:
            base_results = self._load_json(
                f'eval_base_model_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

            if baseline_codebleu is None:
                baseline_codebleu = base_results[0]["codebleu"]  #
            if baseline_total_energy is None:
                baseline_total_energy = base_results[-1]["total_energy"]

            results_all_layers = self._load_json(
                f'eval_all_layers_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')
            all_layers_codebleu.append(results_all_layers[0]["codebleu"])
            all_layers_energy.append(results_all_layers[-1]["total_energy"])

            for thr in self.thresholds:
                results = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

                codebleu_score = results[0]["codebleu"]
                total_energy = results[-1]["total_energy"]

                codebleu_scores.append((thr, codebleu_score))
                total_energies.append((thr, total_energy))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot([x[0] for x in codebleu_scores], [x[1] for x in codebleu_scores], marker='o', color='b',
                 label="CodeBLEU")
        plt.axhline(y=baseline_codebleu, color='gray', linestyle='--', label="Baseline CodeBLEU")
        plt.axhline(y=all_layers_codebleu[0], color='orange', linestyle='--', label="All Layers CodeBLEU")
        plt.xlabel("Threshold")
        plt.ylabel("CodeBLEU")
        plt.title("CodeBLEU vs. Threshold")
        plt.legend()
        plt.grid(True)
        plt.ylim(0.25, 0.5)
        plt.subplot(1, 2, 2)
        plt.plot([x[0] for x in total_energies], [x[1] for x in total_energies], marker='o', color='r',
                 label="Total Energy")
        plt.axhline(y=baseline_total_energy, color='gray', linestyle='--', label="Baseline Total Energy")
        plt.axhline(y=all_layers_energy[0], color='orange', linestyle='--', label="All Layers Total Energy")
        plt.xlabel("Threshold")
        plt.ylabel("Total Energy (Joules)")
        plt.title("Total Energy vs. Threshold")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_relative_codebleu_loss_energy_gain(self):
        codebleu_losses = []
        energy_gains = []
        all_layers_codebleu = None
        all_layers_total_energy = None

        for ctx in self.contexts:
            results_all_layers = self._load_json(
                f'eval_all_layers_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

            if all_layers_codebleu is None:
                all_layers_codebleu = results_all_layers[0]["codebleu"]
                print("f all layers codebleu ", all_layers_codebleu)
            if all_layers_total_energy is None:
                all_layers_total_energy = results_all_layers[-1]["total_energy"]

            for thr in self.thresholds:
                results = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

                codebleu_score = results[0]["codebleu"]
                print("codebleu_score", codebleu_score)
                total_energy = results[-1]["total_energy"]

                codebleu_loss = (all_layers_codebleu - codebleu_score) / all_layers_codebleu * 100  # percentage
                print("codebleu_loss", codebleu_loss)
                energy_gain = (all_layers_total_energy - total_energy) / all_layers_total_energy * 100  # percentage

                codebleu_losses.append((thr, codebleu_loss))
                energy_gains.append((thr, energy_gain))

        plt.figure(figsize=(8, 6))

        plt.plot([x[0] for x in codebleu_losses], [x[1] for x in codebleu_losses],
                 marker='o', color='blue', label="CodeBleu")

        plt.plot([x[0] for x in energy_gains], [x[1] for x in energy_gains],
                 marker='o', color='green', label="Energy")

        plt.xlabel("Threshold")
        plt.ylabel("Percentage (%)")
        plt.title("Relative CodeBLEU  and Energy  vs model with all layers")
        plt.legend()
        plt.grid(True)

        plt.show()

    def plot_relative_metrics(self):
        codebleu_losses = []
        energy_gains = []
        rougeL_losses = []
        chrf_losses = []
        bleu_losses = []
        time_gains = []

        all_layers_codebleu = None
        all_layers_total_energy = None
        all_layers_rougeL = None
        all_layers_chrf = None
        all_layers_bleu = None
        all_layers_total_time = None

        for ctx in self.contexts:
            results_all_layers = self._load_json(
                f'eval_all_layers_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

            if all_layers_codebleu is None:
                all_layers_codebleu = results_all_layers[0]["codebleu"]
            if all_layers_total_energy is None:
                all_layers_total_energy = results_all_layers[-1]["total_energy"]
            if all_layers_rougeL is None:
                all_layers_rougeL = results_all_layers[2]["rougeL"]
            if all_layers_chrf is None:
                all_layers_chrf = results_all_layers[1]["score"]
            if all_layers_bleu is None:
                all_layers_bleu = results_all_layers[3]["bleu"]
            if all_layers_total_time is None:
                all_layers_total_time = results_all_layers[-1]["total_time"]

            for thr in self.thresholds:
                results = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

                codebleu_score = results[0]["codebleu"]
                total_energy = results[-1]["total_energy"]
                rougeL = results[2]["rougeL"]
                chrf = results[1]["score"]  # chrF score
                bleu = results[3]["bleu"]
                total_time = results[-1]["total_time"]

                codebleu_loss = (all_layers_codebleu - codebleu_score) / all_layers_codebleu * 100  # in percentage
                energy_gain = (all_layers_total_energy - total_energy) / all_layers_total_energy * 100  # in percentage
                rougeL_loss = (all_layers_rougeL - rougeL) / all_layers_rougeL * 100
                chrf_loss = (all_layers_chrf - chrf) / all_layers_chrf * 100
                bleu_loss = (all_layers_bleu - bleu) / all_layers_bleu * 100
                time_gain = (all_layers_total_time - total_time) / all_layers_total_time * 100

                codebleu_losses.append((thr, codebleu_loss))
                energy_gains.append((thr, energy_gain))
                rougeL_losses.append((thr, rougeL_loss))
                chrf_losses.append((thr, chrf_loss))
                bleu_losses.append((thr, bleu_loss))
                time_gains.append((thr, time_gain))

        plt.figure(figsize=(10, 6))

        plt.plot([x[0] for x in codebleu_losses], [x[1] for x in codebleu_losses],
                 marker='o', label="CodeBLEU")
        plt.plot([x[0] for x in rougeL_losses], [x[1] for x in rougeL_losses],
                 marker='o', label="ROUGE-L")
        plt.plot([x[0] for x in chrf_losses], [x[1] for x in chrf_losses],
                 marker='o', label="chrF")
        plt.plot([x[0] for x in bleu_losses], [x[1] for x in bleu_losses],
                 marker='o', label="BLEU")
        plt.plot([x[0] for x in energy_gains], [x[1] for x in energy_gains],
                 marker='o', label="Energy")
        plt.plot([x[0] for x in time_gains], [x[1] for x in time_gains],
                 marker='o', label="Time")

        plt.xlabel("Policy Network Softmax Threshold (T=4)")
        plt.ylabel("Relative Change (%)")

        plt.tight_layout()
        plt.legend(loc='upper right')

        plt.grid(True)
        plt.show()

    def plot_absolute_metrics(self):
        codebleu_changes = []
        rougeL_changes = []
        chrf_changes = []
        bleu_changes = []
        energy_changes = []
        time_changes = []

        all_layers_codebleu = None
        all_layers_total_energy = None
        all_layers_rougeL = None
        all_layers_chrf = None
        all_layers_bleu = None
        all_layers_total_time = None

        for ctx in self.contexts:
            # Load all-layers results
            results_all_layers = self._load_json(
                f'eval_all_layers_llama3b_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

            if all_layers_codebleu is None:
                all_layers_codebleu = results_all_layers[0]["codebleu"]
            if all_layers_total_energy is None:
                all_layers_total_energy = results_all_layers[-1]["total_energy"]
            if all_layers_rougeL is None:
                all_layers_rougeL = results_all_layers[2]["rougeL"]
            if all_layers_chrf is None:
                all_layers_chrf = results_all_layers[1]["score"]
            if all_layers_bleu is None:
                all_layers_bleu = results_all_layers[3]["bleu"]
            if all_layers_total_time is None:
                all_layers_total_time = results_all_layers[-1]["total_time"]

            for thr in self.thresholds:
                results = self._load_json(
                    f'eval_rl_model_llama3b_thresh_{thr}_{ctx}ctx_{self.num_samples}_samples_{self.max_new}_maxnew_java.json')

                codebleu_score = results[0]["codebleu"]
                total_energy = results[-1]["total_energy"]
                rougeL = results[2]["rougeL"]
                chrf = results[1]["score"]  # chrF score
                bleu = results[3]["bleu"]
                total_time = results[-1]["total_time"]

                codebleu_change = all_layers_codebleu - codebleu_score
                energy_change = all_layers_total_energy - total_energy
                rougeL_change = all_layers_rougeL - rougeL
                chrf_change = all_layers_chrf - chrf
                bleu_change = all_layers_bleu - bleu
                time_change = all_layers_total_time - total_time

                codebleu_changes.append((thr, codebleu_change))
                energy_changes.append((thr, energy_change))
                rougeL_changes.append((thr, rougeL_change))
                chrf_changes.append((thr, chrf_change))
                bleu_changes.append((thr, bleu_change))
                time_changes.append((thr, time_change))

        plt.figure(figsize=(10, 6))
        plt.plot([x[0] for x in codebleu_changes], [x[1] for x in codebleu_changes], marker='o',
                 label="CodeBLEU")
        plt.plot([x[0] for x in rougeL_changes], [x[1] for x in rougeL_changes], marker='o', label="ROUGE-L")
        # plt.plot([x[0] for x in chrf_changes], [x[1] for x in chrf_changes], marker='o', label="chrF Change")
        plt.plot([x[0] for x in bleu_changes], [x[1] for x in bleu_changes], marker='o', label="BLEU")

        plt.xlabel("Policy Network Softmax Threshold (T=4)")
        plt.ylabel("Absolute Change")
        plt.title("Absolute Metric Changes vs. Threshold")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot energy
        plt.figure(figsize=(10, 6))
        plt.plot([x[0] for x in energy_changes], [x[1] for x in energy_changes], marker='o', color='green',
                 label="Energy Change")
        plt.xlabel("Policy Network Softmax Threshold (T=4)")
        plt.ylabel("Absolute Change (Energy)")
        plt.title("Absolute Energy Change vs. Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot time
        plt.figure(figsize=(10, 6))
        plt.plot([x[0] for x in time_changes], [x[1] for x in time_changes], marker='o', color='orange',
                 label="Time Change")
        plt.xlabel("Policy Network Softmax Threshold (T=4)")
        plt.ylabel("Absolute Change (Time)")
        plt.title("Absolute Time Change vs. Threshold")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
