
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle


# 本地类
import Net



class DQN(object):
    def __init__(self,mlp_architecture):


        # mlp_architecture 是一个列表，描述了要求的神经网络的结构 每层几个神经元
        # N_STATES 是输入层神经元的个数
        # N_ACTIONS 是输出层神经元的个数
        
        self.MEMORY_CAPACITY = 2000
        self.N_STATES = mlp_architecture[0]
        self.N_ACTIONS = mlp_architecture[-1]
        lr = 0.01

        self.eval_net, self.target_net = Net.Net(), Net.Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        self.loss_func = nn.MSELoss()

        #pid
        self.integral = 0  # 位移的积分
        self.max_i = 2**6 # 位移积分的上界
        self.ki= -3 #-1

    def random_action(self):

        N_ACTIONS = self.N_ACTIONS
        action = np.random.randint(0, N_ACTIONS)
        return action
    

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        EPSILON = self.epsilon
        ENV_A_SHAPE = 0

        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)

            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = self.random_action()
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)


        return action

    def store_transition(self, state, a, r, s_):

        MEMORY_CAPACITY=2000

        transition = np.hstack((state, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print('dqn.learn()')
        TARGET_REPLACE_ITER = 100
        MEMORY_CAPACITY = 2000
        BATCH_SIZE = 32
        N_STATES = self.N_STATES


        GAMMA = 0.9


        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        # q_eval w.r.t the action in experience




        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)




        

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()