import gc
import json

import datasets
from huggingface_hub import login
from stable_baselines3 import PPO
from transformers import AutoTokenizer

from src.eval.evaluator import Evaluator
from src.models.llama.modeling_llama_rl_ee import LlamaEESingleHeadRL
from src.models.llama.modeling_llama_rl_with_cache import LlamaEESingleHeadRLCaching

access_token = "XX"
login(token=access_token)
num_samples = 1000
model_id = "ANONYMOUS"
origin_model_id = "meta-llama/Llama-3.2-3B"
dataset = \
    datasets.load_dataset("google/code_x_glue_cc_code_completion_token", name="java")[
        'test']
eval_list = []
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_tokenizer = AutoTokenizer.from_pretrained(origin_model_id)

output = True

max_news = [15]
contexts = [0.2, 0.3, 0.5, 0.6]
thresholds = [0.6, 0.8, 0.9, 0.91, 0.92, 0.94]
path_to_results = "PATH_TO_RESULTS"

use_cache = False
baselines = True
measure_rl_energy = True
language = "java"
if use_cache:
    model_clazz = LlamaEESingleHeadRLCaching

else:
    model_clazz = LlamaEESingleHeadRL

for max_new in max_news:
    for ctx in contexts:
        for thr in thresholds:
            model = model_clazz.from_pretrained(model_id, exit_indices=[3, 5, 7, 9, 11, 13, 17, 21, 25],
                                                threshold=thr, device_map="cuda",
                                                mode="infer_ee", measure_rl_energy=True)

            rl_model = PPO.load(
                "PATH_TO_RL_MODEL",
                env=None,
                device="cuda")
            model.model.set_RL_model(rl_model, mode="PPO")

            evaluator = Evaluator(model, tokenizer, dataset)
            layers = model.get_exit_layers()
            results = evaluator.evaluate(num_samples=num_samples, max_new=max_new, context_fraction=ctx,
                                         language=language, measure_rl_energy=True)

            if output:
                with open(
                        f'{path_to_results}/eval_overhead_thresh_{thr}_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_{language}.json',
                        'w') as f:
                    json.dump(results, f)
            else:
                print(results)
                print(layers)
            model.cpu()
            del model
            gc.collect()
