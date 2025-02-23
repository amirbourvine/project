# -*- coding: utf-8 -*-
"""
Created on Fri May 29 23:23:52 2020

@author: ChefLiutao
"""
import pandas as pd
import numpy as np
from DataPreprocessing import normalization
from DataPreprocessing import build_s_a
from DDPG_agent import DDPG
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config import N_FEATURES

#####################  hyper parameters  ####################

A_LOW = 0
A_HIGH = 1
LR_A = 0.001
LR_C = 0.003
N_ACTOR_HIDDEN = 30
N_CRITIC_HIDDEN = 30
MAX_EPISODES = 300
MAX_STEPS = 1000
NUM_TEST = 1000

GAMMA = 0.9                # 折扣因子
TAU = 0.1                 # 软更新因子
MEMORY_CAPACITY = 100000    #记忆库大小
BATCH_SIZE = 128            #批梯度下降的m
#############################################################

#Load data 
data_dir = os.path.join(os.getcwd(), 'V6.csv')
data = pd.read_csv(data_dir,encoding = 'gbk')
data = data.iloc[:,0]

#Build state matrix and best action
state,action = build_s_a(data,N_FEATURES,1)

#Data split
SPLIT_RATE = 0.75
split_index = round(len(state)*SPLIT_RATE)
train_s,train_a = state[:split_index],action[:split_index]
test_s,test_a = state[split_index:],action[split_index:]

#Normalization
train_s_scaled,test_s_scaled = normalization(train_s,test_s)
A,B = train_a.max(),train_a.min()
train_a_scaled,test_a_scaled = (train_a-B)/(A-B),(test_a-B)/(A-B)

# Training
ddpg = DDPG(N_FEATURES,A_LOW,A_HIGH,LR_A,LR_C,N_ACTOR_HIDDEN,N_CRITIC_HIDDEN)
for episode  in range(MAX_EPISODES):
    index = np.random.choice(range(len(train_s_scaled)))
    index = index - (index%16)
    s = train_s_scaled[index]
    ep_reward = 0
    
    for step in range(MAX_STEPS):
        a = ddpg.choose_action(s)
        r = -abs(a-train_a_scaled[index])
        ep_reward += r
        index += 1
        s_ = train_s_scaled[index]
        
        ddpg.store_transition(s,a,r,s_)
        ddpg.learn()
        
        if (index == len(train_s_scaled)-1) or (step == MAX_STEPS-1):
            print('Episode %d : %.2f'%(episode,ep_reward))
            break
        
        s = s_

# Testing
# pred = []
# for i in range(len(test_s_scaled)):
#     state = test_s_scaled[i]
#     action = ddpg.choose_action(state)
#     pred.append(action)

# pred = [pred[i][0] for i in range(len(test_s_scaled))]
# pred = pd.Series(pred)
# pred = pred*(A-B)+B
# actual = pd.Series(test_a)

# mean_error = (pred - actual).mean()
# print(mean_error)

# **** for 1 prediction ***

# action is an array with the prediction
# state is 6 numbers- need to find out what they are

pred = []
actual = []
for i in range(NUM_TEST):
    index = np.random.choice(range(len(test_s_scaled)))
    if index%16 > 9:
        index = index - 10
    
    state = test_s_scaled[index]
    action = ddpg.choose_action(state)
    pred.append(action)
    actual.append(test_a[index])

    # if i==0:
    #     print(f"{state=}")
    #     print(f"{action=}")

    #     exit(0)

pred = [pred[i][0] for i in range(NUM_TEST)]
pred = pd.Series(pred)
pred = pred*(A-B)+B
actual_df = pd.Series(actual)

mean_error = (pred - actual_df).mean()
print(f"FOR ONLY 1 PREDICTION: {mean_error}")

plt.scatter(pred,actual,marker = '.')
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')
plt.show()


# **** for 9 predictions ***


pred = []
actual = []
for i in range(NUM_TEST):
    index = np.random.choice(range(len(test_s_scaled)))
    index = index - (index%16)
    
    state = test_s_scaled[index]

    for _ in range(9):
        if index >= len(test_a):
            break

        action = ddpg.choose_action(state)
        pred.append(action[0])
        actual.append(test_a[index])
        index += 1
        state = np.roll(state, -1)
        state[-1] = action[0]

pred = pd.Series(pred)
pred = pred*(A-B)+B
actual_df = pd.Series(actual)

mean_error = (pred - actual_df).mean()
print(f"FOR 9 PREDICTIONS: {mean_error}")

plt.scatter(pred,actual,marker = '.')
plt.xlabel('Predicted Value')
plt.ylabel('Actual value')
plt.show()




