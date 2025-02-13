import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
import matplotlib.pyplot as plt
import random


import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Normal, Categorical, Bernoulli
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from env.origin_env import ConstrainAction
from env.modified_env import ConstrainAction


# Parameters
# gamma = 1
gamma = 0.99
render = False
seed = 1
log_interval = 10

data_file = "/Data/liyan/ev_charging/HeurAgenix/src/problems/road_charging/config1_5EVs_1chargers.json"
env = ConstrainAction(data_file)
env.summarize_env()
env.seed(seed)

# env = gym.make('CartPole-v1').unwrapped
num_state = env.observation_space_dim
# num_action = pow(2, env.action_space.n)
num_action = env.action_space.n
num_charger = env.m


print("num_state:",num_state)
print("num_action:",num_action)

torch.manual_seed(seed)
env.action_space.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

def decimal_to_binary_array(action, num_bits):
    # 使用 bin() 转换为二进制字符串，去掉 '0b' 前缀，填充到 num_bits 位
    binary_string = bin(action)[2:].zfill(num_bits)
    # 将二进制字符串转换为数组
    binary_array = [int(bit) for bit in binary_string]
    return binary_array

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.fc2 = nn.Linear(256, 256)

        self.action_head = nn.Linear(256, num_action)

        self.fc_k_logits = nn.Linear(256, num_charger)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_prob = F.softmax(self.action_head(x), dim=1)
        # action_prob = torch.sigmoid(self.action_head(x))  # 得到 [0,1] 之间的概率
        
        # 输出 k 的“logits”，用于 Softmax 或其他分布
        k_logits = self.fc_k_logits(x)
        k_probs = torch.softmax(k_logits, dim=-1)  # Softmax 将 logits 转换为概率分布
        
        # 选择 k：将 logits 转换为概率分布
        k = torch.multinomial(k_probs, 1).squeeze(1) + 1  # 采样 k，+1 是因为 k 从 1 开始
        
        return action_prob, k



    def sample_action(self, x):
        action_prob = self.forward(x)
        m = Bernoulli(action_prob)
        action = m.sample()
        return action, m.log_prob(action)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 256)
        self.fc2 = nn.Linear(256, 256)

        self.state_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-4)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def process_state(self, state):
        if isinstance(state, np.ndarray):
            # If state is already a NumPy array, return it directly
            return state
        timestep = np.array(state['TimeStep'])
        ridetime = np.array(state['RideTime'])
        charging_status = np.array(state['ChargingStatus'])
        soc = np.array(state['SoC'])
        # state = np.concatenate([timestep, ridetime, charging_status, soc])
        state = np.concatenate([ridetime, charging_status, soc])

        return state

    def select_action(self, state, deterministic=False, activation="multi_joint_action"):
        # print("state:",state)
        # print("state.shape:",state.shape)

        # 生成mask
        mask = torch.ones_like(action_prob)
		
        for i in range(env.n):
            if self.obs["RideTime"][i] >= 1: # if on a ride, not charge
                # action[i] = 0
                mask[0][i] = 0
            elif self.obs["SoC"][i] > 1-self.c_rates[i]: # if full capacity, not charge
                # action[i] = 0
                mask[0][i] = 0

        print("mask:",mask)

        state = self.process_state(state)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0)
        with torch.no_grad():
            action_prob, k = self.actor_net(state)
            k = k.item()
        # print("action_prob:",action_prob)
        # print("k:",k)

        # 对 action_prob 进行掩码操作
        action_prob = action_prob * mask

        if activation == "discrete_action":
            if deterministic:
                # 选择具有最高概率的动作
                action = torch.argmax(action_prob, dim=1)
            else:
                # 采样动作
                c = Categorical(action_prob)
                action = c.sample()
            # 假设动作空间为 2^num_bits
            num_bits = int(np.ceil(np.log2(action_prob.size(1))))  # 动作编码的二进制位数
            binary_action = decimal_to_binary_array(action.item(), num_bits)
            return binary_action, action_prob[:, action.item()].item()
        
        elif activation == "multi_joint_action":
            if deterministic:
                # 选择具有最高概率的动作
                # action = torch.round(action_prob).view(-1).tolist() 
                action = self.select_topk_action(action_prob, k)
            else:
                # 采样动作
                sampled_action = torch.multinomial(action_prob, k, replacement=False)  # replacement=False 表示不允许重复

            action = [0 for _ in range(action_prob.size(1))]
            for i in sampled_action:
                action[i] = 1
            
            prob = 1
            for i in range(len(action)):
                if action[i] == 1:
                    prob *= action_prob[0][i]
                # prob *= action_prob[0][i] if action[i] == 1 else 1 - action_prob[0][i]
            # print("action:",action)
            # print("action_prob:",prob)
            return action, prob
        else:
            raise ValueError("Invalid activation function")
        return
    
    def select_topk_action(self, action_prob, k=1, threshold=0.5):
        # 选择概率最大的 k 个动作
        topk_action = torch.topk(action_prob, k, dim=1)
        topk_action_indices = topk_action.indices
        topk_action_prob = topk_action.values
        # # 过滤概率小于阈值的动作
        # topk_action_indices = topk_action_indices[topk_action_prob > threshold]
        # topk_action_prob = topk_action_prob[topk_action_prob > threshold]
        
        action = [0 for _ in range(action_prob.size(1))]
        for i in topk_action_indices:
            action[i] = 1
        return action
    
    def action_constraint(self, action):
        array_sum = sum(action)
        # array_sum = action.sum().item()  # 使用 PyTorch 的 sum 方法并转换为标量值
        # print("array_sum:",array_sum)
        # print("action:",action)
        # if array_sum > env.m:
        #     action = [0 for _ in range(len(action))]
        # return action

        if array_sum > env.m:
            # 找出数组中所有的1的索引
            ones_indices = [i for i, bit in enumerate(action) if bit == 1]
            # 随机选择k个1的索引
            selected_indices = random.sample(ones_indices, env.m)
            for i in range(len(action)):
                if i in selected_indices:
                    action[i] = 1
                else:
                    action[i] = 0
            return action
        else:
            return action


    def get_value(self, state):
        state = self.process_state(state)
        state = torch.from_numpy(np.array(state, dtype=np.float32)).float().unsqueeze(0)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10] + '.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10] + '.pkl')
    
    def load_param(self, filepath='../param/net_param/actor_net' + str(time.time())[:10] + '.pkl'):
        self.actor_net.load_state_dict(torch.load(filepath))
        print(f'Model parameters loaded from {filepath}')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        # for t in self.buffer:
        #     print("t:",t.state)

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!

                action_prob, k = self.actor_net(state[index])
                action_prob = action_prob.gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience

    
def train():
    agent = PPO()

    reward_list = []

    for i_epoch in range(500):
        state = env.reset()
        state = agent.process_state(state)
        # print("state:",state)

        if render: env.render()

        total_reward = 0

        for t in count():
            action, action_prob = agent.select_action(state, activation="multi_joint_action")

            action = agent.action_constraint(action)
            action = np.array(action)
            # print("action:",action)

            result = env.step(action)
            # print("step" ,result)
            # print("action_prob:",action_prob)

            next_state, reward, done, _ = result
            next_state = agent.process_state(next_state)
            total_reward += reward

            trans = Transition(state, action, action_prob, reward, next_state)
            if render: env.render()
            agent.store_transition(trans)
            state = next_state
            
            # print("reward:",reward)

            if done :
                print("epoch:",i_epoch)
                if len(agent.buffer) >= agent.batch_size:agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
                break
        
        print("total_reward:",total_reward)
        reward_list.append(total_reward)
        total_reward = 0

    plt.figure()
    plt.plot(reward_list)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Epoch')
    plt.tight_layout()
    plt.savefig('../param/img/total_reward.png')  # 保存图片
    plt.close()  # 关闭图表以释放内存


    agent.save_param()
    return

def test_network():
    agent = PPO()
    agent.load_param('/Data/liyan/ev_charging/HeurAgenix/src/problems/road_charging/marl/param/net_param/actor_net1739427875.pkl')  # 加载模型参数
    # 使用模型进行推理
    # state = torch.tensor([...])  # 输入状态
    state = env.reset()
    state = agent.process_state(state)

    soc_values = []
    actions = []
    total_reward = 0

    for t in count():
        
        action, action_prob = agent.select_action(state, activation="multi_joint_action")

        action = agent.action_constraint(action)
        action = np.array(action)
        # print("action:",action)
        # print("action_prob:",action_prob)
        actions.append(action)

        result = env.step(action)
        # print("step" ,result)

        next_state, reward, done, _ = result

        soc_list = [f"{x:.3f}" for x in next_state['SoC']]
        soc_list = [round(float(x), 3) for x in next_state['SoC']]  # 将字符串转换为浮点数并限制小数位数
        print(soc_list)
        print("action:",action)

        soc_values.append(soc_list)
        
        next_state = agent.process_state(next_state)

        if render: env.render()
        state = next_state
        print("reward:",reward)
        total_reward += reward

        if done :
            # print("epoch:",t)
            # if len(agent.buffer) >= agent.batch_size:agent.update(t)
            # agent.writer.add_scalar('liveTime/livestep', t, global_step=t)
            break

    # print("soc_values:",soc_values)
    for value in soc_values:
        print(value)

    print("total_reward:",total_reward)

    soc_values = soc_values[:100]
    # for i in range(len(soc_values)):
    #     print(soc_values[i][0])

    # Plotting the SoC values
    plt.figure()
    # for i in range(len(soc_values[0])):
    #     plt.plot([soc[i] for soc in soc_values], label=f'SoC {i}')
    # for i in range(len(soc_values[0])):
    #     plt.plot([soc[i] for soc in soc_values], label=f'SoC {i}')
    #     print([soc[i] for soc in soc_values])
    i = 3
    plt.plot([soc[i] for soc in soc_values], label=f'SoC {i}')
    print([soc[i] for soc in soc_values])

    plt.xlabel('Time Step')
    plt.ylabel('SoC')
    plt.legend()
    plt.title('State of Charge (SoC) over Time')

    plt.tight_layout()
    plt.savefig('soc_values-3.png')  # 保存图片
    plt.close()  # 关闭图表以释放内存
    # for action in actions:
    #     print(action)

    return

if __name__ == '__main__':
    train()
    print("end")

    # test_network()

    # action = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
    # agent = PPO()
    # agent.action_constraint(action, 2)