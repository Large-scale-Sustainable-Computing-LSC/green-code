import gc
import json

import datasets
from huggingface_hub import login
from stable_baselines3 import PPO
from transformers import AutoTokenizer

from src.eval.evaluator import Evaluator
from src.models.opt.modeling_opt_cache import OPTEESingleHeadRLHiddenStateCache
from src.models.opt.modeling_opt_ee_rl import OPTEESingleHeadRLHiddenState

access_token = "XX"
login(token=access_token)
num_samples = 0
model_id = "ANONYMIZED"
origin_model_id = "facebook/opt-2.7b"
dataset = \
    datasets.load_dataset("google/code_x_glue_cc_code_completion_token", name="python")[
        'test']
eval_list = []
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_tokenizer = AutoTokenizer.from_pretrained(origin_model_id)

use_cache = True
language = "python"
if use_cache:
    model_clazz = OPTEESingleHeadRLHiddenStateCache

else:
    model_clazz = OPTEESingleHeadRLHiddenState

max_news = [15, ]
contexts = [0.2, 0.3, 0.5, 0.6]
thresholds = [0.6, 0.8, 0.9, 0.91, 0.92, 0.93]

path_to_results = "results/eval_opt"
include_baselines = True

for max_new in max_news:
    for ctx in contexts:
        if include_baselines or max_new == 50:
            model_all_layers = model_clazz.from_pretrained(model_id, mode="infer", device_map="cuda")

            evaluator = Evaluator(model_all_layers, tokenizer, dataset)
            results = evaluator.evaluate(num_samples=num_samples, max_new=max_new, context_fraction=ctx,
                                         language=language)
            with open(
                    f'{path_to_results}/eval_all_layers_opt27b_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_java.json',
                    'w') as f:
                json.dump(results, f)
            model_all_layers.cpu()
            del model_all_layers
            gc.collect()

            base_model = model_clazz.from_pretrained(origin_model_id, mode="infer", device_map="cuda")

            evaluator = Evaluator(base_model, base_tokenizer, dataset)
            results = evaluator.evaluate(num_samples=num_samples, max_new=max_new, context_fraction=ctx)
            with open(
                    f'{path_to_results}/eval_base_model_opt27b_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_java.json',
                    'w') as f:
                json.dump(results, f)

            base_model.cpu()
            del base_model
            gc.collect()
        for thr in thresholds:
            model = model_clazz.from_pretrained(model_id, exit_indices=[3, 5, 7, 9, 11, 13, 15, 19, 23, 27],
                                                threshold=thr, device_map="cuda",
                                                mode="infer_ee", measure_rl_energy=False)
            rl_model = PPO.load(
                "path_to_rl_model",
                device="cuda")
            model.model.decoder.set_RL_model(rl_model)

            evaluator = Evaluator(model, tokenizer, dataset)
            layers = model.get_exit_layers()
            results = evaluator.evaluate(num_samples=num_samples, max_new=max_new, context_fraction=ctx,
                                         language=language)

            with open(
                    f'{path_to_results}/eval_rl_model_opt27b_thresh_{thr}_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_{language}.json',
                    'w') as f:
                json.dump(results, f)

            with open(
                    f'{path_to_results}/eval_rl_model_opt27b_thresh_{thr}_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_{language}_exit_layers.json',
                    'w') as f:
                json.dump(layers, f)
            model.cpu()
            del model
            gc.collect()
