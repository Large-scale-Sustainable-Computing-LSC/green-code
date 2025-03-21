import torch
import torch.nn as nn


class SB3TorchWrapper(nn.Module):

    def __init__(self, sb3_model, device='cuda', mode="PPO"):
        super(SB3TorchWrapper, self).__init__()
        self.mode = mode
        if self.mode == "PPO":
            self.device = device
            self.extractor = sb3_model.policy.mlp_extractor
            self.policy_net = sb3_model.policy.mlp_extractor.policy_net
            self.action_net = sb3_model.policy.action_net

            self.extractor.to(self.device)
            self.policy_net.to(self.device)
            self.action_net.to(self.device)
            print(self)
        elif self.mode == "DQN":
            self.device = device
            self.q_net = sb3_model.q_net.q_net
            print(self.q_net)
            self.q_net.to(self.device)
        else:
            raise NotImplementedError("The mode is not supported")

    def forward(self, x):
        x = x.to(self.device)

        if self.mode == "PPO":
            x = self.policy_net(x)
            x = self.action_net(x)
            return x
        elif self.mode == "DQN":
            x = self.q_net(x)
            return x
        else:
            raise NotImplementedError("The mode is not supported")
