import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
from networks import *


class Agent:
    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, visual=False):
        self.state_size = state_size#状态空间大小
        self.action_size = action_size#动作空间大小
        self.bs = bs#batch size
        self.lr = lr#learnning rate
        self.tau = tau#τ
        self.gamma = gamma#γ
        self.device = device#训练设备
        if visual:
            self.Q_local = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
        else:
            self.Q_local = Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Q_Network(self.state_size, self.action_size).to(self.device)

        self.soft_update(1)#同步两个网络的初始权重
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)#优化器针对local网络
        self.memory = deque(maxlen=100000)#创建记忆空间用作exp replay

    def act(self, state, eps=0):
        if random.random() > eps:#eps是ε 是变异的概率，会依据概率选择1.随机做动作，2.按模型取值，变异概率在训练中逐渐下降
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        experiences = random.sample(self.memory, self.bs)#随机从经验池中取出batch size个例子，用作训练

        #从经验池中取出状态、动作、下一状态、是否结束游戏
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        #从local网络中预估动作值
        Q_values = self.Q_local(states) #用local网络给过去的状态1就每种动作进行评分
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions) #取过去做的动作对应的分数
        with torch.no_grad():
            Q_targets = self.Q_target(next_states) #用target网络给过去的状态2就每种动作进行评分
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True) #返回动作评分中最高的分数以及对应的动作序号
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets #gamma是时间折算系数，目标系数表示现有的reward以及下一步可能的最好结果*时间折算系数
        loss = (Q_values - Q_targets).pow(2).mean()#标准差
        self.optimizer.zero_grad()#导数清零
        loss.backward()#反向传播
        self.optimizer.step()#进行单次优化

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
