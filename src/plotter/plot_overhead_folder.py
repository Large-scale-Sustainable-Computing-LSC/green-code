import json
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.backends.backend_pdf import PdfPages


class Visualizer:
    def __init__(self, folder_path: str):

        self.folder_path = folder_path
        self.data = self._load_data()

    def _extract_params(self, filename: str) -> Dict[str, Union[float, int, str, None]]:

        params = {"thresh": None, "ctx": None, "samples": None, "baseline_type": None}

        try:
            if "_thresh_" in filename:
                thresh_value = filename.split("_thresh_")[1].split("_")[0]
                params["thresh"] = float(thresh_value)
        except (IndexError, ValueError):
            params["thresh"] = None

        try:
            if "ctx" in filename:
                ctx_value = filename.split("_")
                ctx_value = [x for x in ctx_value if "ctx" in x][0].replace("ctx", "")
                params["ctx"] = float(ctx_value)
        except (IndexError, ValueError):
            params["ctx"] = None

        try:
            if "_samples_" in filename:
                params["samples"] = int(filename.split("_samples_")[1].split("_")[0])
        except (IndexError, ValueError):
            params["samples"] = None

        if "eval_all_layers" in filename:
            params["baseline_type"] = "Full Model"
        elif "eval_base_model" in filename:
            params["baseline_type"] = "Base Model"
        else:
            params["baseline_type"] = "RL"
        print(params)
        return params

    def _load_data(self) -> pd.DataFrame:
        files = [f for f in os.listdir(self.folder_path) if f.endswith('.json')]
        records = {}

        for file in files:
            file_path = os.path.join(self.folder_path, file)

            try:
                with open(file_path, 'r') as f:
                    metrics_list = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {file}")
                continue

            params = self._extract_params(file)

            consolidated_record = {"file": file, **params}

            for metric_name, metric_value in metrics_list.items():
                consolidated_record[metric_name] = metric_value
            records[file] = consolidated_record

        df = pd.DataFrame.from_dict(records, orient='index')

        df.fillna(value={"thresh": 0.0, "ctx": 0.0, "samples": 0, "baseline_type": "Unknown"}, inplace=True)

        df.to_csv("test_overhead.csv")

        return df

    def plot_total_energy_and_total_rl_energy_over_all_thresholds(self, df, ctx, total_energy_baseline):

        df = df.loc[df["ctx"] == ctx]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x="thresh", y="total_energy", label="Total Energy")
        sns.lineplot(data=df, x="thresh", y="total_rl_energy", label="Total RL Energy")
        ax.set_title(f"Total Energy and Total RL Energy Over All Thresholds (ctx={ctx})")

        plt.axhline(y=total_energy_baseline, color='r', linestyle='-', label="Total Energy Baseline")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Energy (Joules)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_total_time_and_total_rl_time_over_all_thresholds_multiple_ctxs(self, df, ctx_list,
                                                                            base_line_totaltime_list):

        baseline_colors = ['lightgray', '#B0C4DE', '#D8BFD8', '#FFDAB9', '#E6E6FA']
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})

        fig, ax = plt.subplots(figsize=(12, 6))
        for ctx in ctx_list:
            df_ctx = df.loc[df["ctx"] == ctx]
            sns.lineplot(data=df_ctx, x="thresh", y="total_time", label=f"Total (ctx={ctx})")
            sns.lineplot(data=df_ctx, x="thresh", y="total_rl_time", label=f"Overhead (ctx={ctx})")

        for i, baseline_total_time in enumerate(base_line_totaltime_list):
            plt.axhline(
                y=baseline_total_time,
                color=baseline_colors[i % len(baseline_colors)],
                alpha=1,
                label=f"Full Model (ctx={ctx_list[i]})"
            )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Time (s)")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_total_energy_and_total_rl_energy_over_all_thresholds_multiple_ctxs(self, df, ctx_list,
                                                                                base_line_totalEnergy_list):
        baseline_colors = ['lightgray', '#B0C4DE', '#D8BFD8', '#FFDAB9', '#E6E6FA']  # Extend as needed
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 16})

        fig, ax = plt.subplots(figsize=(12, 6))
        for ctx in ctx_list:
            df_ctx = df.loc[df["ctx"] == ctx]
            sns.lineplot(data=df_ctx, x="thresh", y="total_energy", label=f"Total (ctx={ctx})")
            sns.lineplot(data=df_ctx, x="thresh", y="total_rl_energy", label=f"Overhead (ctx={ctx})")

        for i, baseline_total_energy in enumerate(base_line_totalEnergy_list):
            plt.axhline(
                y=baseline_total_energy,
                color=baseline_colors[i % len(baseline_colors)],  # Cycle through colors if more contexts than colors
                linestyle='-',
                alpha=1,
                label=f"Full Model (ctx={ctx_list[i]})"
            )

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Energy (Ws)")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()

    def print_relative_overheads(self, df, ctx, baseline_total_energy, baseline_total_time):
        df = df.loc[df["ctx"] == ctx]

        df["relative_overhead"] = 100 * df["total_rl_energy"] / df["total_energy"]
        df["relative_overhead_to_baseline"] = 100 * df["total_rl_energy"] / baseline_total_energy
        df["relative_overhead_time"] = 100 * df["total_rl_time"] / df["total_time"]
        df["relative_overhead_time_to_baseline"] = 100 * df["total_rl_time"] / baseline_total_time

        overheads = df[["thresh", "relative_overhead", "relative_overhead_to_baseline",
                        "relative_overhead_time", "relative_overhead_time_to_baseline"]]

        display(overheads)

        for index, row in overheads.iterrows():
            print(f"Threshold: {row['thresh']}")
            print(f"Relative Overhead: {row['relative_overhead']:.2f}%")
            print(f"Relative Overhead to Baseline: {row['relative_overhead_to_baseline']:.2f}%")
            print(f"Relative Overhead Time: {row['relative_overhead_time']:.2f}%")
            print(f"Relative Overhead Time to Baseline: {row['relative_overhead_time_to_baseline']:.2f}%")
            print()

    def plot_relative_overhead(self, df, ctx, baseline_total_energy, baseline_total_time):
        df = df.loc[df["ctx"] == ctx]

        df["relative_overhead"] = 100 * df["total_rl_energy"] / df["total_energy"]

        df["relative_overhead_to_baseline"] = 100 * df["total_rl_energy"] / baseline_total_energy

        df["relative_overhead_time"] = 100 * df["total_rl_time"] / df["total_time"]

        df["relative_overhead_time_to_baseline"] = 100 * df["total_rl_time"] / baseline_total_time

        #
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x="thresh", y="relative_overhead", label="Energy")
        sns.lineplot(data=df, x="thresh", y="relative_overhead_time", label="Time")

        sns.lineplot(data=df, x="thresh", y="relative_overhead_to_baseline", label="Energy (Full Model)")
        sns.lineplot(data=df, x="thresh", y="relative_overhead_time_to_baseline", label="Time (Full Model)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Relative Overhead (%)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        plt.close()
