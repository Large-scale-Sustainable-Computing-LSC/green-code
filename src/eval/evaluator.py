import datasets
import evaluate
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from codebleu import calc_codebleu
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from zeus.monitor import ZeusMonitor

from src.models.opt.modeling_opt_single_head_fixed_exit import OPTFixedEESingleHead


class Evaluator:
    def __init__(self, model, tokenizer, dataset, max_ctx_size=2048, device="cuda", ):
        self.model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.max_ctx_size = max_ctx_size
        self.dataset = dataset
        self.device = device
        self.prepare_dataset()
        self.monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def prepare_dataset(self):
        def dataset_array_to_string(example):
            example["code"] = ' '.join(example["code"][1:-1])
            return example

        self.dataset = self.dataset.map(dataset_array_to_string)

    def warmup_model(self):
        for i in range(0, 10):
            with torch.no_grad():
                code_tokens = self.tokenizer.encode("System.out.println(")
                output = self.model.generate(input_ids=torch.tensor(code_tokens).unsqueeze(0).to(self.device)
                                             , max_length=15, do_sample=False)

    def evaluate(self, max_new=25, num_samples=150, context_fraction=0.5, skip_codebleu=False, language="java",
                 measure_rl_energy=False, is_opt=False):

        curr = 0

        predictions = []
        references = []
        times = []
        energies = []
        througputs = []
        rl_agent_total_energy = []

        rl_agent_total_times = []

        for example in tqdm(self.dataset, desc="Evaluating", unit="example"):
            if curr > num_samples:
                break
            curr = curr + 1
            code = example['code']
            tokenized = self.tokenizer(code)
            code_tokens = tokenized['input_ids']

            code_tokens = torch.tensor(code_tokens).to(self.device)

            context_size = int(len(code_tokens) * context_fraction)

            if context_size > self.max_ctx_size:
                context_size = self.max_ctx_size - 1 - max_new

            context_tokens = code_tokens[:context_size]
            attention_mask = torch.tensor(tokenized['attention_mask'][:context_size]).to(self.device)
            attention_mask = attention_mask.clone().detach().unsqueeze(0).to(self.device)
            input_ids = context_tokens.clone().detach().unsqueeze(0).to(self.device)
            if input_ids.size() == [1, 0] or input_ids.shape == (1, 0):
                print(f"contex size: {context_size}")
                print(f"input ids: {input_ids}")
                continue

            with torch.no_grad():
                self.monitor.begin_window("gen", sync_execution=True)

                self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
                output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                             pad_token_id=self.tokenizer.eos_token_id,
                                             max_length=context_size + max_new, do_sample=False, num_beams=1)
                measurement = self.monitor.end_window("gen", sync_execution=True)
                if measure_rl_energy and not is_opt:
                    rl_energy, rl_time = self.model.model.get_rl_energy_time()
                    rl_agent_total_energy.append(np.sum(rl_energy))
                    rl_agent_total_times.append(np.sum(rl_time))
                    self.model.model.clear_rl_energy_time()
                elif measure_rl_energy and is_opt:
                    rl_energy, rl_time = self.model.model.decoder.get_rl_energy_time()
                    print(rl_energy)
                    print(rl_time)
                    rl_agent_total_energy.append(np.sum(rl_energy))
                    rl_agent_total_times.append(np.sum(rl_time))
                    self.model.model.decoder.clear_rl_energy_time()
                times.append(measurement.time)
                energies.append(measurement.gpu_energy[0])
            new_generated_tokens = output[0][len(context_tokens):len(context_tokens) + max_new]
            ground_truth_tokens = code_tokens[context_size:context_size + max_new]
            output_code = self.tokenizer.decode(new_generated_tokens, skip_special_tokens=False)
            ground_truth_code = self.tokenizer.decode(ground_truth_tokens,
                                                      skip_special_tokens=False)
            througput = len(new_generated_tokens) / measurement.time
            througputs.append(througput)
            predictions.append(output_code)
            references.append(ground_truth_code)

        if skip_codebleu:
            codebleu = None
        else:
            codebleu = calc_codebleu(references, predictions, lang=language, weights=(0.25, 0.25, 0.25, 0.25),
                                     tokenizer=None)

        if measure_rl_energy:
            total_time = float(np.sum(times))
            std_time = float(np.std(times))
            mean_time = float(np.mean(times))

            total_energy = float(np.sum(energies))
            std_energy = float(np.std(energies))
            mean_energy = float(np.mean(energies))

            total_rl_time = float(np.sum(rl_agent_total_times))
            mean_rl_time = float(np.mean(rl_agent_total_times))
            std_rl_time = float(np.std(rl_agent_total_times))

            total_rl_energy = float(np.sum(rl_agent_total_energy))
            mean_rl_energy = float(np.mean(rl_agent_total_energy))
            sdtd_rl_energy = float(np.std(rl_agent_total_energy))

            return {
                "mean_time": mean_time,
                "mean_energy": mean_energy,
                "std_time": std_time,
                "std_energy": std_energy,
                "total_energy": total_energy,
                "total_time": total_time,
                "total_rl_time": total_rl_time,
                "mean_rl_time": mean_rl_time,
                "std_rl_time": std_rl_time,
                "total_rl_energy": total_rl_energy,
                "mean_rl_energy": mean_rl_energy,
                "std_rl_energy": sdtd_rl_energy

            }

        chrf_metric = evaluate.load("chrf")
        chrf = chrf_metric.compute(predictions=predictions, references=references)

        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)

        bleu_metric = evaluate.load("bleu")
        bleu = bleu_metric.compute(predictions=predictions, references=references)

        exact_matches = [1 if pred == ref else 0 for pred, ref in zip(predictions, references)]
        exact_match = sum(exact_matches) / len(exact_matches)

        levenshtein_distances = [levenshtein_distance(pred, ref) for pred, ref in zip(predictions, references)]
        mean_levenshtein_distance = float(np.mean(levenshtein_distances))

        total_time = float(np.sum(times))
        std_time = float(np.std(times))
        mean_time = float(np.mean(times))

        total_energy = float(np.sum(energies))
        std_energy = float(np.std(energies))
        mean_energy = float(np.mean(energies))

        std_throug = float(np.std(througputs))
        mean_through = float(np.mean(througputs))

        return codebleu, chrf, rouge, bleu, {
            "exact_match": exact_match,
            "lev_dist": mean_levenshtein_distance,
            "mean_time": mean_time,
            "mean_energy": mean_energy,
            "std_time": std_time,
            "std_energy": std_energy,
            "total_energy": total_energy,
            "total_time": total_time,
            "std_through": std_throug,
            "mean_through": mean_through
        }

    def plot_all_layers(self, layers, codebleu, chrf, rouge, bleu, mean_time, mean_energy, plot_titles="",
                        filename="test.png"):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(layers, [c['codebleu'] for c in codebleu], label='CodeBLEU', marker='o')
        axes[0, 0].plot(layers, [r['rouge1'] for r in rouge], label='ROUGE-1', marker='o')
        axes[0, 0].plot(layers, [b['bleu'] for b in bleu], label='BLEU', marker='o')
        axes[0, 0].set_title('CodeBLEU, ROUGE-1, and BLEU Scores')
        axes[0, 0].set_xlabel('Number of Layers')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(layers, [c['score'] for c in chrf], label='CHRF', color='g', marker='o')
        axes[0, 1].set_title('CHRF Score')
        axes[0, 1].set_xlabel('Number of Layers')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(layers, mean_time, label='Mean Time', color='r', marker='o')
        axes[1, 0].set_title('Mean Time per Layer')
        axes[1, 0].set_xlabel('Number of Layers')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(layers, mean_energy, label='Mean Energy', color='m', marker='o')
        axes[1, 1].set_title('Mean Energy per Layer')
        axes[1, 1].set_xlabel('Number of Layers')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        if plot_titles:
            fig.suptitle(plot_titles)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_all_layers_multi_model(self, layers, codebleu, chrf, rouge, bleu, mean_time, mean_energy, plot_titles="",
                                    filename="plot.png", legend_line_styles=["model1", "model2", "model3"]):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        line_styles = ['-', '--', '-.']

        for i, model in enumerate(codebleu):
            axes[0, 0].plot(layers, [c['codebleu'] for c in model], label=f'{legend_line_styles[i]} CodeBLEU',
                            linestyle=line_styles[i], marker='o')
        for i, model in enumerate(rouge):
            axes[0, 0].plot(layers, [r['rouge1'] for r in model], label=f'{legend_line_styles[i]} ROUGE-1',
                            linestyle=line_styles[i], marker='o')
        for i, model in enumerate(bleu):
            axes[0, 0].plot(layers, [b['bleu'] for b in model], label=f'{legend_line_styles[i]} BLEU',
                            linestyle=line_styles[i], marker='o')
        axes[0, 0].set_title('CodeBLEU, ROUGE-1, and BLEU Scores')
        axes[0, 0].set_xlabel('Number of Layers')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        for i, model in enumerate(chrf):
            axes[0, 1].plot(layers, [c['score'] for c in model], label=f'{legend_line_styles[i]} CHRF',
                            linestyle=line_styles[i], marker='o')
        axes[0, 1].set_title('CHRF Score')
        axes[0, 1].set_xlabel('Number of Layers')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        for i, model in enumerate(mean_time):
            axes[1, 0].plot(layers, model, label=f'{legend_line_styles[i]} Mean Time', linestyle=line_styles[i],
                            marker='o')
        axes[1, 0].set_title('Mean Time per Layer')
        axes[1, 0].set_xlabel('Number of Layers')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        for i, model in enumerate(mean_energy):
            axes[1, 1].plot(layers, model, label=f'{legend_line_styles[i]} Mean Energy', linestyle=line_styles[i],
                            marker='o')
        axes[1, 1].set_title('Mean Energy per Layer')
        axes[1, 1].set_xlabel('Number of Layers')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        if plot_titles:
            fig.suptitle(plot_titles)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_final_scores(self, models, codebleu, chrf, rouge, bleu, mean_time, mean_energy, plot_title="",
                          filename="final_score_comp_opt125.png"):
        n = len(models)
        index = np.arange(n)
        bar_width = 0.1

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        axes[0, 0].bar(index, codebleu, bar_width, label='CodeBLEU')
        axes[0, 0].set_title('CodeBLEU Scores')
        axes[0, 0].set_xticks(index)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()

        axes[0, 1].bar(index, chrf, bar_width, label='CHRF', color='g')
        axes[0, 1].set_title('CHRF Scores')
        axes[0, 1].set_xticks(index)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()

        axes[0, 2].bar(index, rouge, bar_width, label='ROUGE-1', color='r')
        axes[0, 2].set_title('ROUGE-1 Scores')
        axes[0, 2].set_xticks(index)
        axes[0, 2].set_xticklabels(models)
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()

        axes[1, 0].bar(index, bleu, bar_width, label='BLEU', color='c')
        axes[1, 0].set_title('BLEU Scores')
        axes[1, 0].set_xticks(index)
        axes[1, 0].set_xticklabels(models)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()

        axes[1, 1].bar(index, mean_time, bar_width, label='Mean Time', color='m')
        axes[1, 1].set_title('Mean Time')
        axes[1, 1].set_xticks(index)
        axes[1, 1].set_xticklabels(models)
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].legend()

        axes[1, 2].bar(index, mean_energy, bar_width, label='Mean Energy', color='y')
        axes[1, 2].set_title('Mean Energy')
        axes[1, 2].set_xticks(index)
        axes[1, 2].set_xticklabels(models)
        axes[1, 2].set_ylabel('Energy (J)')
        axes[1, 2].legend()

        if plot_title:
            fig.suptitle(plot_title)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


def run_evaluation_opt_125m():
    model_id = "ANONYMOUS/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    num_layers = 12
    dataset = datasets.load_dataset("google/code_x_glue_cc_code_completion_token", "java")['test']
    codebleus_optee, chrfs_optee, rouges_optee, bleus_optee, mean_times_optee, mean_energies_optee = [], [], [], [], [], []
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=1)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_optee.append(codebleu)
        chrfs_optee.append(chrf)
        rouges_optee.append(rouge)
        bleus_optee.append(bleu)
        mean_times_optee.append(mean_time)
        mean_energies_optee.append(mean_energy)
    model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=3)
    evaluator = Evaluator(model, tokenizer, dataset)
    evaluator.plot_all_layers([1 + i for i in range(num_layers)], codebleus_optee, chrfs_optee, rouges_optee,
                              bleus_optee, mean_times_optee,
                              mean_energies_optee,
                              "Mean scores and efficency of different layers of opt-125m on 100 test samples, generating 10 tokens with 60% context")
    codebleus_opt, chrfs_opt, rouges_opt, bleus_opt, mean_times_opt, mean_energies_opt = [], [], [], [], [], []
    model_id = "facebook/opt-125m"
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_opt.append(codebleu)
        chrfs_opt.append(chrf)
        rouges_opt.append(rouge)
        bleus_opt.append(bleu)
        mean_times_opt.append(mean_time)
        mean_energies_opt.append(mean_energy)

    codebleus_optft, chrfs_optft, rouges_optft, bleus_optft, mean_times_optft, mean_energies_optft = [], [], [], [], [], []
    model_id = "ANONYM/opt-125m-code-java"
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_optft.append(codebleu)
        chrfs_optft.append(chrf)
        rouges_optft.append(rouge)
        bleus_optft.append(bleu)
        mean_times_optft.append(mean_time)
        mean_energies_optft.append(mean_energy)

    evaluator.plot_all_layers_multi_model(
        layers=[1 + i for i in range(num_layers)],
        codebleu=[codebleus_optee, codebleus_opt, codebleus_optft],
        chrf=[chrfs_optee, chrfs_opt, chrfs_optft],
        rouge=[rouges_optee, rouges_opt, rouges_optft],
        bleu=[bleus_optee, bleus_opt, bleus_optft],
        mean_time=[mean_times_optee, mean_times_opt, mean_times_optft],
        mean_energy=[mean_energies_optee, mean_energies_opt, mean_energies_optft],
        plot_titles="Mean scores and efficiency of different layers of opt-125m on 100 test samples, generating 10 tokens with 60% context",
        filename="XX",
        legend_line_styles=["OPT EE-FT", "OPT Base", "OPT FT"]
    )


def run_evaluation_opt_27bm(file_name="2_7B_comparison.png"):
    model_id = "ANONYM"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    num_layers = 32
    dataset = datasets.load_dataset("google/code_x_glue_cc_code_completion_token", "java")['test']
    codebleus_optee, chrfs_optee, rouges_optee, bleus_optee, mean_times_optee, mean_energies_optee = [], [], [], [], [], []
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_optee.append(codebleu)
        chrfs_optee.append(chrf)
        rouges_optee.append(rouge)
        bleus_optee.append(bleu)
        mean_times_optee.append(mean_time)
        mean_energies_optee.append(mean_energy)
    model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=3)
    evaluator = Evaluator(model, tokenizer, dataset)
    evaluator.plot_all_layers([1 + i for i in range(num_layers)], codebleus_optee, chrfs_optee, rouges_optee,
                              bleus_optee, mean_times_optee,
                              mean_energies_optee,
                              "Mean scores and efficency of different layers of opt-2.7B on 100 test samples, generating 10 tokens with 60% context")
    codebleus_opt, chrfs_opt, rouges_opt, bleus_opt, mean_times_opt, mean_energies_opt = [], [], [], [], [], []
    model_id = "facebook/opt-2.7B"
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_opt.append(codebleu)
        chrfs_opt.append(chrf)
        rouges_opt.append(rouge)
        bleus_opt.append(bleu)
        mean_times_opt.append(mean_time)
        mean_energies_opt.append(mean_energy)

    codebleus_optft, chrfs_optft, rouges_optft, bleus_optft, mean_times_optft, mean_energies_optft = [], [], [], [], [], []
    model_id = "XXX"
    for i in range(0, num_layers):
        model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=i)
        evaluator = Evaluator(model, tokenizer, dataset)
        codebleu, chrf, rouge, bleu, mean_time, mean_energy = evaluator.evaluate(max_new=10, num_samples=100,
                                                                                 context_fraction=0.6)
        codebleus_optft.append(codebleu)
        chrfs_optft.append(chrf)
        rouges_optft.append(rouge)
        bleus_optft.append(bleu)
        mean_times_optft.append(mean_time)
        mean_energies_optft.append(mean_energy)

    evaluator.plot_all_layers_multi_model(
        layers=[1 + i for i in range(num_layers)],
        codebleu=[codebleus_optee, codebleus_opt, codebleus_optft],
        chrf=[chrfs_optee, chrfs_opt, chrfs_optft],
        rouge=[rouges_optee, rouges_opt, rouges_optft],
        bleu=[bleus_optee, bleus_opt, bleus_optft],
        mean_time=[mean_times_optee, mean_times_opt, mean_times_optft],
        mean_energy=[mean_energies_optee, mean_energies_opt, mean_energies_optft],
        plot_titles="Mean scores and efficiency of different layers of opt-2.7B on 100 test samples, generating 10 tokens with 60% context",
        filename=file_name,
        legend_line_styles=["OPT EE-FT", "OPT Base", "OPT FT"]
    )


def run_evaluation_opt_comparison(model_id1, model_id2, model_id3, model_names, filename="final_layer_comp.png"):
    model_id = model_id1
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    dataset = datasets.load_dataset("google/code_x_glue_cc_code_completion_token", "java")['test']
    model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=-1)

    evaluator = Evaluator(model, tokenizer, dataset)
    codebleu_optft, chrf_optft, rouge_optft, bleu_optft, mean_time_optft, mean_energy_optft = evaluator.evaluate(
        max_new=10, num_samples=100,
        context_fraction=0.6)

    model_id = model_id2
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = OPTFixedEESingleHead.from_pretrained(model_id, device_map="cuda", exit_index=-1)
    evaluator = Evaluator(model, tokenizer, dataset)
    codebleu_optee, chrf_optee, rouge_optee, bleu_optee, mean_time_optee, mean_energy_optee = evaluator.evaluate(
        max_new=10, num_samples=100,
        context_fraction=0.6)
    model_id = model_id3
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    evaluator = Evaluator(model, tokenizer, dataset)
    codebleu_cg, chrf_cg, rouge_cg, bleu_cg, mean_time_cg, mean_energy_cg = evaluator.evaluate(max_new=10,
                                                                                               num_samples=100,
                                                                                               context_fraction=0.6)

    models = model_names
    evaluator.plot_final_scores(models,
                                [codebleu_optft['codebleu'], codebleu_optee['codebleu'], codebleu_cg['codebleu']],

                                [chrf_optft['score'], chrf_optee['score'], chrf_cg['score']],
                                [rouge_optft['rouge1'], rouge_optee['rouge1'], rouge_cg['rouge1']],

                                [bleu_optft['bleu'], bleu_optee['bleu'], bleu_cg['bleu']],
                                [mean_time_optft, mean_time_optee, mean_time_cg],
                                [mean_energy_optft, mean_energy_optee, mean_energy_cg],
                                plot_title="Comparison of Final Scores (Last Layer) Across Models",
                                filename=filename)


def compare_models(metrics_list):
    metrics = ['codebleu', 'bleu', 'rougeL']
    time_metric = ['time']
    energy_metric = ['energy']

    model1_metrics = [metrics_list[0][0]['codebleu'], metrics_list[0][3]['bleu'], metrics_list[0][2]['rougeL']]
    model2_metrics = [metrics_list[1][0]['codebleu'], metrics_list[1][3]['bleu'], metrics_list[1][2]['rougeL']]
    model3_metrics = [metrics_list[2][0]['codebleu'], metrics_list[2][3]['bleu'], metrics_list[2][2]['rougeL']]

    model1_time = [metrics_list[0][4]]
    model2_time = [metrics_list[1][4]]
    model3_time = [metrics_list[2][4]]

    model1_energy = [metrics_list[0][5]]
    model2_energy = [metrics_list[1][5]]
    model3_energy = [metrics_list[2][5]]

    x = np.arange(len(metrics))  #
    x_time = np.arange(len(time_metric))
    x_energy = np.arange(len(energy_metric))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, model1_metrics, width, label='with RL Agent')
    rects2 = ax.bar(x, model2_metrics, width, label='Base')
    rects3 = ax.bar(x + width, model3_metrics, width, label='all layers')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Code Metrics Across Three Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.bar(x_time - width, model1_time, width, label='with RL Agent')
    rects2 = ax.bar(x_time, model2_time, width, label='Base Model')
    rects3 = ax.bar(x_time + width, model3_time, width, label='all layers')

    ax.set_ylabel('Time (s)')
    ax.set_title('Comparison of Inference Time Across Three Models')
    ax.set_xticks(x_time)
    ax.set_xticklabels(time_metric)
    ax.legend()

    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    rects1 = ax.bar(x_energy - width, model1_energy, width, label='with RL agent')
    rects2 = ax.bar(x_energy, model2_energy, width, label='base model')
    rects3 = ax.bar(x_energy + width, model3_energy, width, label='all layers')

    ax.set_ylabel('Energy (mJ)')
    ax.set_title('Comparison of Energy Consumption Across Three Models')
    ax.set_xticks(x_energy)
    ax.set_xticklabels(energy_metric)
    ax.legend()

    fig.tight_layout()
    plt.show()


def compare_codebleu_energy(metrics_list, rl_agent_energy=None, base="Base Model"):
    metrics = ['codebleu']
    energy_metric = ['energy (Ws)']

    model1_codebleu = [metrics_list[0][0]['codebleu']]  # Base model
    model2_codebleu = [metrics_list[1][0]['codebleu']]  # All layers
    model3_codebleu = [metrics_list[2][0]['codebleu']]  # With RL Agent

    # Convert energy from mJ to Ws (1 mJ = 1e-3 J = 1e-3 Ws)
    model1_energy = [metrics_list[0][5] * 1e-3]
    model2_energy = [metrics_list[1][5] * 1e-3]
    model3_energy = [metrics_list[2][5] * 1e-3]

    x = np.arange(len(metrics))
    x_energy = np.arange(len(energy_metric))
    width = 0.2

    fig, ax = plt.subplots(figsize=(6, 4))
    rects1 = ax.bar(x - width, model1_codebleu, width, label=base)  # Base model leftmost
    rects2 = ax.bar(x, model2_codebleu, width, label='All layers (finetuned)')  # All layers in the middle
    rects3 = ax.bar(x + width, model3_codebleu, width, label='With RL Agent')  # RL Agent rightmost

    ax.set_ylabel('CodeBLEU Score')
    ax.set_title('Comparison of CodeBLEU Between Three Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    fig.tight_layout()
    plt.show()
    if rl_agent_energy:
        rl_agent_energy = np.sum(rl_agent_energy) * 1e-3
        model_with_rl_energy = [model3_energy[0] - rl_agent_energy]
        rl_agent_only_energy = rl_agent_energy

        fig, ax = plt.subplots(figsize=(6, 4))

        rects1 = ax.bar(x_energy - width, model1_energy, width, label=base)
        rects2 = ax.bar(x_energy, model2_energy, width, label=base)

        rects3a = ax.bar(x_energy + width, model_with_rl_energy, width, label='With RL Agent (Model Energy)',
                         color='lightblue')
        rects3b = ax.bar(x_energy + width, rl_agent_only_energy, width, bottom=model_with_rl_energy,
                         label='RL Agent Energy', color='orange')

        ax.set_ylabel('Energy (Ws)')
        ax.set_title('Comparison of Energy Consumption Between Three Models')
        ax.set_xticks(x_energy)
        ax.set_xticklabels(energy_metric)
        ax.legend()

        fig.tight_layout()
        plt.show()
    else:

        fig, ax = plt.subplots(figsize=(6, 4))
        rects1 = ax.bar(x_energy - width, model1_energy, width, label='Base Model')  # Base model leftmost
        rects2 = ax.bar(x_energy, model2_energy, width, label='All layers')  # All layers in the middle
        rects3 = ax.bar(x_energy + width, model3_energy, width, label='With RL Agent')  # RL Agent rightmost

        ax.set_ylabel('Energy (Ws)')
        ax.set_title('Comparison of Energy Consumption Between Three Models')
        ax.set_xticks(x_energy)
        ax.set_xticklabels(energy_metric)
        ax.legend()

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    pass
