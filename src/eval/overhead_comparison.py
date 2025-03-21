import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from torch import nn
from tqdm import tqdm
from zeus.monitor import ZeusMonitor

from src.models.llama.modeling_llama_rl_ee import LlamaEESingleHeadRL
from src.models.rl.enviornments.sb3_torch_wrapper import SB3TorchWrapper

model = LlamaEESingleHeadRL.from_pretrained("XXX", device_map="cuda")

lm_head = model.lm_head

monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)

size = model.hidden_size

rl_model = SB3TorchWrapper(PPO.load("XXX",
                                    device="cuda"))

exit_classifier = nn.Linear(size, 2, bias=False, device="cuda")

energy_rl = []
energy_cls = []
energy_lmhead = []

time_rl = []
time_cls = []
time_lmhead = []

for i in tqdm(range(100_000), desc="Processing"):
    random_tensor = torch.randn(size, device="cuda")

    monitor.begin_window("test", sync_execution=True)
    result2 = exit_classifier(random_tensor)
    action_probabilities = F.softmax(result2 / 1, dim=0)
    action = torch.argmax(action_probabilities).item()
    measurement = monitor.end_window("test", sync_execution=True)
    time_cls.append(measurement.time)
    energy_cls.append(measurement.gpu_energy[0])

    monitor.begin_window("test", sync_execution=True)
    result3 = lm_head(model.model.norm(random_tensor))
    probabilities = F.softmax(result3, dim=-1)
    max_prob, max_index = probabilities.max(dim=-1)
    measurement = monitor.end_window("test", sync_execution=True)
    energy_lmhead.append(measurement.gpu_energy[0])
    time_lmhead.append(measurement.time)

    monitor.begin_window("test", sync_execution=True)
    result = rl_model(random_tensor)
    action_probabilities = F.softmax(result / 1, dim=0)
    action = torch.argmax(action_probabilities).item()

    measurement = monitor.end_window("test", sync_execution=True)
    time_rl.append(measurement.time)
    energy_rl.append(measurement.gpu_energy[0])

# save the data
np.save("energy_rl.npy", energy_rl)
np.save("energy_cls.npy", energy_cls)
np.save("energy_lmhead.npy", energy_lmhead)
