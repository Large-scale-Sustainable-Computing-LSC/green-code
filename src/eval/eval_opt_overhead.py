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
num_samples = 1000
model_id = "ANONYMIZED"
origin_model_id = "facebook/opt-2.7b"
dataset = \
    datasets.load_dataset("google/code_x_glue_cc_code_completion_token", name="python")[
        'test']
eval_list = []
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_tokenizer = AutoTokenizer.from_pretrained(origin_model_id)

use_cache = False
language = "python"
if use_cache:
    model_clazz = OPTEESingleHeadRLHiddenStateCache

else:
    model_clazz = OPTEESingleHeadRLHiddenState

max_news = [15, ]
contexts = [0.2, 0.3, 0.5, 0.6]
thresholds = [0.6, 0.8, 0.9, 0.91, 0.92, 0.94]

path_to_results = "results/eval_opt"
include_baselines = False

for max_new in max_news:
    print("MAXNEW: ", max_new)
    for ctx in contexts:

        for thr in thresholds:
            model = model_clazz.from_pretrained(model_id, exit_indices=[3, 5, 7, 9, 11, 13, 15, 19, 23, 27],
                                                threshold=thr, device_map="cuda",
                                                mode="infer_ee", measure_rl_energy=True)
            rl_model = PPO.load(
                "path_to_rl_model",
                device="cuda")
            model.model.decoder.set_RL_model(rl_model)

            evaluator = Evaluator(model, tokenizer, dataset)
            results = evaluator.evaluate(num_samples=num_samples, max_new=max_new, context_fraction=ctx,
                                         language="python", measure_rl_energy=True, is_opt=True)

            with open(
                    f'{path_to_results}/eval_overhead_thresh_{thr}_{ctx}ctx_{num_samples}_samples_{max_new}_maxnew_{language}.json',
                    'w') as f:
                json.dump(results, f)

            model.cpu()
            del model
            gc.collect()
