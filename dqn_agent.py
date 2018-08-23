import helper
import random
import os
import glob
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model import DuelingDQN
from experience_replay import ReplayBuffer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, config):
        """Initialize an Agent object"""
        self.seed = random.seed(config["general"]["seed"])
        self.config = config

        # Q-Network
        self.q = DuelingDQN(config).to(DEVICE)
        self.q_target = DuelingDQN(config).to(DEVICE)

        self.optimizer = optim.RMSprop(self.q.parameters(), lr=config["agent"]["learning_rate"])
        self.criterion = F.mse_loss

        self.memory = ReplayBuffer(config)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def save_experiences(self, state, action, reward, next_state, done):
        """Prepare and save experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done)

    def _current_step_is_a_learning_step(self):
        """Check if the current step is an update step"""
        self.t_step = (self.t_step + 1) % self.config["agent"]["update_rate"]
        return self.t_step == 0

    def _enough_samples_in_memory(self):
        """Check if minimum amount of samples are in memory"""
        return len(self.memory) > self.config["train"]["batch_size"] 

    def epsilon_greedy_action_selection(self, action_values, eps):
        """Epsilon-greedy action selection"""
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.config["general"]["action_size"]))

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.q.eval()
        with torch.no_grad():
            action_values = self.q(state)
        self.q.train()

        return self.epsilon_greedy_action_selection(action_values, eps)

    def _calc_loss(self, states, actions, rewards, next_states, dones):
        q_eval = self.q(states).gather(1, actions)
        q_eval_next = self.q(next_states)
        _, q_argmax = q_eval_next.detach().max(1)
        q_next = self.q_target(next_states)
        q_next = q_next.gather(1, q_argmax.unsqueeze(1))
        q_target = rewards + (self.config["agent"]["gamma"] * q_next * (1 - dones))
        loss = self.criterion(q_eval, q_target)
        return loss

    def _update_weights(self, loss):
        torch.nn.utils.clip_grad.clip_grad_value_(self.q.parameters(), 1.0)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        """Update network using one sample of experience from memory"""
        if self._current_step_is_a_learning_step() and self._enough_samples_in_memory():
            states, actions, rewards, next_states, dones = self.memory.sample(self.config["train"]["batch_size"])
            loss = self._calc_loss(states, actions, rewards, next_states, dones)
            self._update_weights(loss)
            self._soft_update(self.q, self.q_target)

    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.config["agent"]["tau"] * local_param.data +
                                    (1.0 - self.config["agent"]["tau"]) * target_param.data)

    def save(self):
        """Save the network weights"""
        helper.mkdir(os.path.join(".",
                                  *self.config["general"]["checkpoint_path"],
                                  self.config["general"]["env_name"]))
        current_date_time = helper.get_current_date_time()
        current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")

        torch.save(self.q.state_dict(), os.path.join(".",
                                                     *self.config["general"]["checkpoint_path"],
                                                     self.config["general"]["env_name"],
                                                     "ckpt_" + current_date_time))

    def load(self):
        """Load latest available network weights"""
        list_of_files = glob.glob(os.path.join(".",
                                               *self.config["general"]["checkpoint_path"],
                                               self.config["general"]["env_name"],
                                               "*"))
        latest_file = max(list_of_files, key=os.path.getctime)
        self.q.load_state_dict(torch.load(latest_file))
        self.q_target.load_state_dict(torch.load(latest_file))
