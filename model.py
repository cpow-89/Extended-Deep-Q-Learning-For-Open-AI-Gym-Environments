import torch
import torch.nn as nn
import torch.nn.functional as functional


class DuelingDQN(nn.Module):
    """Multilayer-Perceptron version of the Dueling Network Architecture"""
    def __init__(self, config):

        super(DuelingDQN, self).__init__()

        self.seed = torch.manual_seed(config["general"]["seed"])
        self.config = config

        self.fc1 = nn.Linear(config["general"]["state_size"], config["model"]["fc1_nodes"])

        self.fc2_adv = nn.Linear(in_features=config["model"]["fc1_nodes"], out_features=config["model"]["fc2_adv"])
        self.fc2_val = nn.Linear(in_features=config["model"]["fc1_nodes"], out_features=config["model"]["fc2_val"])

        self.fc3_adv = nn.Linear(in_features=config["model"]["fc2_adv"], out_features=config["general"]["action_size"])
        self.fc3_val = nn.Linear(in_features=config["model"]["fc2_val"], out_features=1)

    def forward(self, state):
        x = functional.relu(self.fc1(state))

        adv = functional.relu(self.fc2_adv(x))
        val = functional.relu(self.fc2_val(x))

        adv = self.fc3_adv(adv)
        val = self.fc3_val(val).expand(state.size(0), self.config["general"]["action_size"])

        x = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.config["general"]["action_size"])

        return x

