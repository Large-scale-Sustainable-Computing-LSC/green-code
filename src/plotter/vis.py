import json
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


class VisualizationTool:
    def __init__(self, folder_path: str):
        """
        Initializes the VisualizationTool with the path to a folder containing JSON files.
        """
        self.folder_path = folder_path
        self.data = self._load_data()

    def _extract_params(self, filename: str) -> Dict[str, Union[float, int, str, None]]:
        """
        Extracts parameters (thresh, ctx, samples) from the filename.
        """
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

            if "exit_layers" in file:
                file_name_without_exit_layers = file.replace("_exit_layers", "")
                records[file_name_without_exit_layers]["mean_layers"] = np.mean(json.load(open(file_path)))

                continue

            for metrics in metrics_list:
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        consolidated_record[key] = value if value is not None else 0

            records[file] = consolidated_record

        records = {k: v for k, v in records.items() if v.get("codebleu") is not None}

        df = pd.DataFrame.from_dict(records, orient='index')

        df.fillna(value={"thresh": 0.0, "ctx": 0.0, "samples": 0, "baseline_type": "Unknown"}, inplace=True)
        df["mean_layers"].fillna(df["mean_layers"].max(), inplace=True)

        df.to_csv("test.csv")

        return df

    def plot_metrics(self, df, metric, context, thresholds, metric2=None, output_dir="figures"):
        """
        Creates grouped bar figures for a metric (or two metrics) and energy consumption
        for specified thresholds, grouped by baseline type, and saves them in a
        LaTeX-compatible format.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            metric (str): The primary metric to plot (e.g., 'rougeL', 'bleu', 'codebleu').
            context (str): The context to filter (column `ctx`).
            thresholds (list): List of thresholds to consider.
            metric2 (str, optional): A secondary metric to plot. Defaults to None.
            output_dir (str): Directory to save figures. Defaults to 'figures'.

        Returns:
            None
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        df_context = df[df['ctx'] == context]

        for thresh in thresholds:
            df_thresh = df_context[df_context['thresh'] == thresh]
            df_baseline = df_context[df_context['baseline_type'] == 'Base Model']
            df_all_layers = df_context[df_context['baseline_type'] == 'Full Model']

            data = [
                {'Group': f'Threshold {thresh}', 'Model': 'Base Model', metric: df_baseline[metric].mean()},
                {'Group': f'Threshold {thresh}', 'Model': 'FT w. All Layers', metric: df_all_layers[metric].mean()},
                {'Group': f'Threshold {thresh}', 'Model': 'With RL Agent', metric: df_thresh[metric].mean()}

            ]
            df_plot = pd.DataFrame(data)

            plt.figure(figsize=(8, 5))
            sns.barplot(
                data=df_plot,
                x='Group',
                y=metric,
                hue='Model',
                palette="viridis"
            )

            plt.title(f"{metric.capitalize()} Comparison for Context '{context}' and Threshold {thresh}")
            plt.xlabel("Group (Threshold / Baseline)", fontsize=12)
            plt.ylabel(f"{metric.capitalize()}", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='Model', fontsize=10)

            # Save the plot
            plt.tight_layout()
            plot_filename = f"{output_dir}/{metric}_thresh_{thresh}_context_{context}.pdf"
            plt.savefig(plot_filename)
            plt.close()

            print(f"Saved plot: {plot_filename}")

    def plot_multiple_metrics(self, df, metrics, context, thresholds, output_file="figures/multi_metric_plots.pdf"):
        """
        Creates grouped bar figures for multiple metrics and saves them as a single PDF file.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            metrics (list): List of metrics to plot (e.g., ['rougeL', 'bleu', 'codebleu']).
            context (str): The context to filter (column `ctx`).
            thresholds (list): List of thresholds to consider.
            output_file (str): Output file path for the multi-page PDF.

        Returns:
            None
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df_context = df[df['ctx'] == context]

        with PdfPages(output_file) as pdf:
            for metric in metrics:
                for thresh in thresholds:
                    df_thresh = df_context[df_context['thresh'] == thresh]
                    df_baseline = df_context[df_context['baseline_type'] == 'Base Model']
                    df_all_layers = df_context[df_context['baseline_type'] == 'Full Model']

                    data = [
                        {'Group': f'Threshold {thresh}', 'Model': 'Base Model', metric: df_baseline[metric].mean()},
                        {'Group': f'Threshold {thresh}', 'Model': 'FT w. All Layers',
                         metric: df_all_layers[metric].mean()},
                        {'Group': f'Threshold {thresh}', 'Model': 'With RL Agent', metric: df_thresh[metric].mean()}
                    ]
                    df_plot = pd.DataFrame(data)

                    plt.figure(figsize=(8, 5))
                    sns.barplot(
                        data=df_plot,
                        x='Group',
                        y=metric,
                        hue='Model',
                        palette="viridis"
                    )

                    plt.title(f"{metric.capitalize()} Comparison for Context '{context}' and Threshold {thresh}")
                    plt.xlabel("Group (Threshold / Baseline)", fontsize=12)
                    plt.ylabel(f"{metric.capitalize()}", fontsize=12)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.legend(title='Model', fontsize=10)

                    pdf.savefig()
                    plt.close()

                    print(f"Added plot for metric '{metric}' at threshold {thresh}")

        print(f"Saved all figures to {output_file}")

    def plot_metrics_combined(self, df, metrics, context, thresholds, output_file="figures/combined_metrics.pdf"):
        """
        Creates a grouped bar plot for each metric, comparing all thresholds, the full model,
        and the base model, and saves all figures as a single PDF file.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            metrics (list): List of metrics to plot (e.g., ['rougeL', 'bleu', 'codebleu']).
            context (str): The context to filter (column `ctx`).
            thresholds (list): List of thresholds to consider.
            output_file (str): Output file path for the multi-page PDF.

        Returns:
            None
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df_context = df[df['ctx'] == context]

        with PdfPages(output_file) as pdf:
            for metric in metrics:
                data = []

                for thresh in thresholds:
                    df_thresh = df_context[df_context['thresh'] == thresh]
                    data.append(
                        {'Group': f'Threshold {thresh}', 'Model': 'With RL Agent', metric: df_thresh[metric].mean()})

                df_baseline = df_context[df_context['baseline_type'] == 'Base Model']
                df_all_layers = df_context[df_context['baseline_type'] == 'Full Model']
                data.append({'Group': 'Base Model', 'Model': 'Base Model', metric: df_baseline[metric].mean()})
                data.append({'Group': 'Full Model', 'Model': 'FT w. All Layers', metric: df_all_layers[metric].mean()})

                df_plot = pd.DataFrame(data)

                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=df_plot,
                    x='Group',
                    y=metric,
                    hue='Model',
                    palette="viridis"
                )

                plt.title(f"{metric.capitalize()} Comparison for Context '{context}'")
                plt.xlabel("Thresholds and Baselines", fontsize=12)
                plt.ylabel(f"{metric.capitalize()}", fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend(title='Model', fontsize=10)

                pdf.savefig()
                plt.close()

                print(f"Added plot for metric '{metric}'")

        print(f"Saved all figures to {output_file}")

    def plot_all_metrics_single_plot(self, df, metrics, context, thresholds,
                                     output_file="figures/all_metrics_comparison.pdf", mode="paper"):
        """
        Creates a single grouped bar plot comparing thresholds, the full model, and the base model
        for multiple metrics, and saves it as a PDF.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            metrics (list): List of metrics to plot (e.g., ['rougeL', 'bleu', 'codebleu']).
            context (str): The context to filter (column `ctx`).
            thresholds (list): List of thresholds to consider.
            output_file (str): Output file path for the PDF.

        Returns:
            None
        """

        if mode == "paper":
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 10,
                "axes.titlesize": 22,
                "axes.labelsize": 32,
                "legend.fontsize": 26,
                "xtick.labelsize": 26,
                "ytick.labelsize": 26,
                "lines.linewidth": 1.0,
                "figure.dpi": 300,
            })

        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df_context = df[df['ctx'] == context]

        data = []
        for metric in metrics:

            df_baseline = df_context[df_context['baseline_type'] == 'Base Model']
            df_all_layers = df_context[df_context['baseline_type'] == 'Full Model']

            if metric == "syntax_match_score":
                data.append({'Metric': "syntax", 'Group': 'Base Model', 'Value': df_baseline[metric].mean()})
                data.append({'Metric': "syntax", 'Group': 'Full Model', 'Value': df_all_layers[metric].mean()})
            elif metric == "dataflow_match_score":
                data.append({'Metric': "dataflow", 'Group': 'Base Model', 'Value': df_baseline[metric].mean()})
                data.append({'Metric': "dataflow", 'Group': 'Full Model', 'Value': df_all_layers[metric].mean()})
            else:
                data.append({'Metric': metric, 'Group': 'Base Model', 'Value': df_baseline[metric].mean()})
                data.append({'Metric': metric, 'Group': 'Full Model', 'Value': df_all_layers[metric].mean()})
            # Add data for each threshold
            for thresh in thresholds:
                df_thresh = df_context[df_context['thresh'] == thresh]
                if metric == "syntax_match_score":
                    data.append({'Metric': "syntax", 'Group': f'GC(T={thresh})', 'Value': df_thresh[metric].mean()})
                if metric == "dataflow_match_score":
                    data.append({'Metric': "dataflow", 'Group': f'GC(T={thresh})', 'Value': df_thresh[metric].mean()})
                else:

                    data.append({'Metric': metric, 'Group': f'GC(T={thresh})', 'Value': df_thresh[metric].mean()})

        data = [d for d in data if d["Metric"] != "syntax_match_score"]
        df_plot = pd.DataFrame(data)

        threshold_colors = sns.color_palette("Blues", len(thresholds))
        baseline_colors = sns.color_palette("Reds", 2)  # Base Model and Full Model
        colors = {f'GC(T={t})': c for t, c in zip(thresholds, threshold_colors)}
        colors.update({'Base Model': baseline_colors[0], 'Full Model': baseline_colors[1]})

        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=df_plot,
            x='Metric',
            y='Value',
            hue='Group',
            palette=colors
        )

        # Customize plot aesthetics
        plt.ylabel("Score")
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tick_params(axis='y', )
        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Saved plot to {output_file}")

    def plot_energy_time_throughput(self, df, context, thresholds,
                                    output_file="figures/energy_time_throughput_comparison.pdf", mode="paper"):
        """
        Creates three grouped bar figures side by side:
        - First plot: total_energy comparison.
        - Second plot: total_time comparison.
        - Third plot: mean_through comparison.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            context (str): The context to filter (column `ctx`).
            thresholds (list): List of thresholds to consider.
            output_file (str): Output file path for the PDF.

        Returns:
            None
        """

        if mode == "paper":
            plt.rcParams.update({
                "font.family": "serif",
                "font.size": 14,
                "axes.titlesize": 22,
                "axes.labelsize": 28,
                "legend.fontsize": 28,
                "xtick.labelsize": 32,
                "ytick.labelsize": 32,
                "lines.linewidth": 1.0,
                "figure.dpi": 300,
            })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df_context = df[df['ctx'] == context]

        data = []
        metrics = ['total_energy', 'total_time', 'mean_through']
        for metric in metrics:

            df_baseline = df_context[df_context['baseline_type'] == 'Base Model']
            df_all_layers = df_context[df_context['baseline_type'] == 'Full Model']
            data.append(
                {'Metric': metric, 'Group': 'Base Model', 'Category': 'Baseline', 'Value': df_baseline[metric].mean()})
            data.append({'Metric': metric, 'Group': 'Full Model', 'Category': 'Baseline',
                         'Value': df_all_layers[metric].mean()})
            for thresh in thresholds:
                df_thresh = df_context[df_context['thresh'] == thresh]
                data.append({'Metric': metric, 'Group': f'GC(T={thresh})', 'Category': 'Threshold',
                             'Value': df_thresh[metric].mean()})

        df_plot = pd.DataFrame(data)

        threshold_colors = sns.color_palette("Blues", len(thresholds))
        baseline_colors = sns.color_palette("Reds", 2)
        group_colors = {f'GC(T={t})': c for t, c in zip(thresholds, threshold_colors)}
        group_colors.update({'Base Model': baseline_colors[0], 'Full Model': baseline_colors[1]})

        fig, axes = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1, 1, 1]})

        for i, metric in enumerate(metrics):
            metric_plot = df_plot[df_plot['Metric'] == metric]
            sns.barplot(
                data=metric_plot,
                x='Group',
                y='Value',
                hue='Group',
                palette=group_colors,
                ax=axes[i],
                dodge=False,
                width=0.99
            )
            axes[i].set_xlabel("", fontsize=18)

            if metric == "mean_through":
                axes[i].set_ylabel("Throughput [Token/s]")
            elif metric == "total_energy":
                axes[i].set_ylabel("Total Energy [Ws]")
            elif metric == "total_time":
                axes[i].set_ylabel("Total Time [s]")

            axes[i].tick_params(axis='x', labelrotation=60)
            axes[i].tick_params(axis='y')

            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            axes[i].legend().remove()  # Remove individual legends

        handles, labels = [], []
        for key, color in group_colors.items():
            handles.append(plt.Line2D([0], [0], color=color, lw=10))
            labels.append(key)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(output_file)
        plt.close()

        print(f"Saved plot to {output_file}")

    def plot_mean_layer_for_contexts_and_threshs(self, df, context_list, thresholds, highest_layer=32):
        """
        creates a plot for mean layers for each context and threshold
        (One plot, multiple lines)
        :param df:
        :param context_list:
        :param thresholds:
        :return:
        """
        df_context = df[df['ctx'].isin(context_list)]
        df_context = df_context[df_context['thresh'].isin(thresholds)]

        data = []
        for context in context_list:
            for thresh in thresholds:
                df_thresh = df_context[(df_context['ctx'] == context) & (df_context['thresh'] == thresh)]
                data.append({'Context': context, 'Threshold': thresh,
                             'Mean Layers': 100 * (1 - (df_thresh['mean_layers'].mean() / highest_layer))})

        df_plot = pd.DataFrame(data)

        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_plot,
            x='Threshold',
            y='Mean Layers',
            hue='Context',
            palette="viridis",
            marker='o'
        )

        plt.xlabel("Thresholds", fontsize=20)
        plt.xticks(thresholds, fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel("% of Layers saved", fontsize=20)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Context', fontsize=14)

        plt.tight_layout()
        plt.savefig("figures/mean_layers_comparison.pdf")
        plt.close()

    def plot_score_and_lev_dist_context(self, df1, df2, context, thresholds, label1, label2,
                                        output_file="figures/score_lev_dist_comparison_context.pdf"):
        """
        Creates a line plot with 4 lines for "score" and "lev_dist" from two dataframes
        and includes baselines as horizontal lines for both models.

        Args:
            df1 (pd.DataFrame): The first dataframe.
            df2 (pd.DataFrame): The second dataframe.
            context (str): The context to filter on (column `ctx`).
            thresholds (list): List of thresholds to include.
            label1 (str): Label for the first dataframe (used in legend).
            label2 (str): Label for the second dataframe (used in legend).
            output_file (str): Path to save the output plot. Default is "figures/score_lev_dist_comparison_context_with_baselines.pdf".

        Returns:
            None
        """
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df1_filtered = df1[(df1['ctx'] == context) & (df1['thresh'].isin(thresholds))]
        df2_filtered = df2[(df2['ctx'] == context) & (df2['thresh'].isin(thresholds))]

        df1_baseline = df1[(df1['ctx'] == context) & (df1['baseline_type'] == 'Full Model')]
        df2_baseline = df2[(df2['ctx'] == context) & (df2['baseline_type'] == 'Full Model')]

        baseline_score1 = df1_baseline['score'].mean()
        baseline_score2 = df2_baseline['score'].mean()
        baseline_lev_dist1 = df1_baseline['lev_dist'].mean()
        baseline_lev_dist2 = df2_baseline['lev_dist'].mean()

        df1_filtered = df1_filtered.sort_values('thresh')
        df2_filtered = df2_filtered.sort_values('thresh')

        thresholds1 = df1_filtered['thresh'].tolist()
        thresholds2 = df2_filtered['thresh'].tolist()
        score1 = df1_filtered['score'].tolist()
        score2 = df2_filtered['score'].tolist()
        lev_dist1 = df1_filtered['lev_dist'].tolist()
        lev_dist2 = df2_filtered['lev_dist'].tolist()

        plt.figure(figsize=(12, 7))

        plt.plot(thresholds1, score1, label=f"{label1} - ChrF", linestyle='-', marker='o', color='blue')
        plt.plot(thresholds2, score2, label=f"{label2} - ChrF", linestyle='-', marker='o', color='green')

        plt.plot(thresholds1, lev_dist1, label=f"{label1} - Lev Dist", linestyle='--', marker='s', color='red')
        plt.plot(thresholds2, lev_dist2, label=f"{label2} - Lev Dist", linestyle='--', marker='s', color='orange')

        plt.axhline(y=baseline_score1, color='blue', linestyle=':', label=f"{label1} Full Model - ChrF")
        plt.axhline(y=baseline_score2, color='green', linestyle=':', label=f"{label2} Full Model - ChrF")
        plt.axhline(y=baseline_lev_dist1, color='red', linestyle=':', label=f"{label1} Full Model - Lev Dist")
        plt.axhline(y=baseline_lev_dist2, color='orange', linestyle=':', label=f"{label2} Full Model - Lev Dist")

        plt.xlabel("Thresholds", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.xticks(thresholds, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save the plot
        plt.savefig(output_file)
        plt.close()

        print(f"Saved plot to {output_file}")
