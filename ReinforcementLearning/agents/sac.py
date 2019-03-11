##
## This file is based on the Actor-Critic algorthim called SAC. The paper is referenced below.
## https://arxiv.org/abs/1801.01290
## The code is adapted from the one posted on this blogpost:
## https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665
## 
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import os

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample()
        action = torch.tanh(mean+ std*z.to(device))
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z      = normal.sample().to(device)
        action = torch.tanh(mean + std*z)
        
        action  = action.cpu()#.detach().cpu().numpy()
        return action[0]
    
    
class SACAgent():
    
    def __init__(self, task, target_pos, hidden_dim=256, replay_buffer_size=1000000):
        self.target_pos = target_pos # np.array([0., 0., 10.])
        self.task = task # Task(target_pos=target_pos, init_pose=np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0]))

        self.action_dim = task.action_size
        self.state_dim  = task.state_size
        self.hidden_dim = hidden_dim

        self.value_net        = ValueNetwork(self.state_dim, self.hidden_dim).to(device)
        self.target_value_net = ValueNetwork(self.state_dim, self.hidden_dim).to(device)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)


        self.replay_buffer_size = replay_buffer_size 
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
    def learn(self, batch_size,gamma=0.99,soft_tau=1e-2,):
        
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value    = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)
    
        # Training Q Function
        target_value = self.target_value_net(next_state)
        target_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()   
        
        # Training Value Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action),self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value - log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())


        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Training Policy Function
        policy_loss = (log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
    
    def normalize(self, x):
        return (x - self.task.action_high/2)/(self.task.action_high/2)
    
    def denormalize(self, x):
        return x*self.task.action_high/2 + self.task.action_high/2
    
    def get_actions(self, state):
        return self.denormalize(self.policy_net.get_action(state).detach())
    
    def save_experience(self, state, action, reward, next_state, done, normalize=True):
        naction = action
        if normalize:
            naction = self.normalize(action)
        self.replay_buffer.push(state, naction, reward, next_state, done)
        
    def save_models(self, path, policy='policy', soft_q='soft_q_net', value_net='value_net'):
        extension = '.pth'
        torch.save(self.policy_net.state_dict(), os.path.join(path, policy+extension))
        torch.save(self.soft_q_net1.state_dict(), os.path.join(path, soft_q+'1'+extension))
        torch.save(self.soft_q_net2.state_dict(), os.path.join(path, soft_q+'2'+extension))
        torch.save(self.value_net.state_dict(), os.path.join(path, value_net+extension))
        
        print("Models saved to {}".format(path))
        
    def load_models(self, path, policy='policy', soft_q='soft_q_net', value_net='value_net'):
        extension = '.pth'
        self.policy_net.load_state_dict(torch.load(os.path.join(path, policy+extension)))
        self.soft_q_net1.load_state_dict(torch.load(os.path.join(path, soft_q+'1'+extension)))
        self.soft_q_net1.load_state_dict(torch.load(os.path.join(path, soft_q+'2'+extension)))
        self.value_net.load_state_dict(torch.load(os.path.join(path, value_net+extension)))
        
        print("Models loaded from {}".format(path))
