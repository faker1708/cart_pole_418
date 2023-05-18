

#我也来写个no qline的。

# 本地类
import Net
import DQN



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

import matplotlib.pyplot as plt

import pickle
import time


import copy





class cart_pole():

    def show(self):
        print('load')
        dqn = self.dqn
        # env = gym.make('CartPole-v1')
        
        env = gym.make('CartPole-v1',render_mode = 'human')
        env = env.unwrapped

        dqn.epsilon = 1

        enough = 2**10

        # while(1):
        load_time = 2**15    # 采样数量
        sum = 0
        for _ in range(0,load_time):
            
            state,_ = env.reset()
            step = 0
            reward = 0

            while(1):
                env.render()
                action = dqn.choose_action(state)


                next_state, _, done, _, _ = env.step(action)


                reward = step
                if (done) :
                    break
                else:
                    step +=1
                    state = next_state
                    if(step>enough):
                        break
                    else:
                        if(step%2**7==0):
                            print(step)

            print('reward',reward)    
            

        return 


    def load(self):
        print('load')
        dqn = self.dqn
        env = gym.make('CartPole-v1')
        
        # env = gym.make('CartPole-v1',render_mode = 'human')
        env = env.unwrapped

        dqn.epsilon = 1

        enough = 2**10

        # while(1):
        load_time = 2**5    # 采样数量
        sum = 0
        for _ in range(0,load_time):
            
            state,_ = env.reset()
            step = 0
            reward = 0

            while(1):
                env.render()
                action = dqn.choose_action(state)


                next_state, _, done, _, _ = env.step(action)


                reward = step
                if (done) :
                    break
                else:
                    step +=1
                    state = next_state
                    if(step>enough):
                        break
                    # else:
                    #     if(step%2**7):
                            

            # print('reward',reward)    
            sum+= reward
            # print()

        avg = sum/load_time

        return avg


    def train(self):

        # train
        print('train')
        dqn = self.dqn


        env = gym.make('CartPole-v1')
        env = env.unwrapped


        enough = 2**10


        patient = 2** 11

        for tt in range(0,patient):
            state,_ = env.reset()
            step = 0

            temp_list = list()



            while(1):
                env.render()
                action = dqn.choose_action(state)
                next_state, _, done, _, _ = env.step(action)
                temp_list.append(   [state,action,next_state]   )
                if done:
                    reward = step
                    # print('train reward',reward)

                    break
                else:
                    state = next_state
                    step +=1
                    if(step > enough):
                        break
            for _ , ele in enumerate(temp_list):
                state = ele[0]
                action = ele[1]
                next_state = ele[2]                
                value = 1 * 0.1 *reward
                dqn.store_transition(state, action, value, next_state)
            dqn.learn()
            temp_list = list()

            if(tt%2**9==0):
                print("reward",reward)


        return

    def __init__(self):

        # init


        N_ACTIONS = 2
        N_STATES =4

        mlp_architecture = [N_STATES,50,N_ACTIONS]


        while(1):   # 无限训练模型，检查算法 的收敛性

            dqn = DQN.DQN(mlp_architecture) # init 
            self.dqn = dqn
            # self.dqn.epsilon = 0.0
            self.dqn.epsilon = 0.9


            succ = 0
            for term in range(0,33):

                # self.dqn.epsilon = 0.9
                self.train()

                
                avg = self.load()
                print('avg',avg,'term',term)


                if(avg>0.9* 1024):
                    succ = 1
                    break


            print('成功与否',succ)


            self.show()
if __name__ == '__main__':
    cart_pole()