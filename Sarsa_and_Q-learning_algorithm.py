# -*- coding: utf-8 -*-
"""
@author: Quoc Thinh Vo - QTVo

"""

import numpy as np
import random


#%%  Initialize
alpha = 0.5
epsilon = 0.2
gamma = 1

States = ['LT','A','B','C','D','E','RT']

Actions = ['L','R']

State_Action_Pairs = {'(A, R)': 'B', '(B, R)': 'C', '(C, R)': 'D', '(D, R)': 'E', '(E, R)': 'RT', 
           '(A, L)': 'LT', '(B, L)': 'A', '(C, L)': 'B', '(D, L)': 'C', '(E, L)': 'D'}

Q_Values = {}
  
#%%  
def SARSA_learning(current_state, initialized_action, epsilon):
    steps = 0
       
    while True:
        steps += 1
        action = initialized_action
        state_action_pair = '({}, {})'.format(current_state, initialized_action)
        state_next = State_Action_Pairs[state_action_pair]
        
        if steps > 500:
            epsilon /= 2
            
        if state_next != 'LT' and state_next != 'RT':
            action_next = policy_define(state_next, epsilon)
            Q_Values[current_state, action] = Q_Values[current_state, action] + alpha*(reward(state_next) + gamma*Q_Values[state_next, action_next] - Q_Values[current_state, action])
            current_state = state_next
        else:
            Q_Values[current_state, action] = Q_Values[current_state, action] + alpha*(reward(state_next) + gamma*Q_Values[state_next, ] - Q_Values[current_state, action])
            #print("====== End of Espisode ========")
            break

def Q_learning(current_state, initialized_action, epsilon):

    while current_state != 'RT' and current_state != 'LT':
        action = initialized_action
        state_action_pair = '({}, {})'.format(current_state, initialized_action)
        state_next = State_Action_Pairs[state_action_pair]
       
        if state_next != 'LT' and state_next != 'RT':
            Q_Values[current_state, action] = Q_Values[current_state, action] + alpha*(reward(state_next) + gamma*max(Q_Values[state_next, 'L'], Q_Values[state_next, 'R']) - Q_Values[current_state, action])
            current_state = state_next
        else:
            Q_Values[current_state, action] = Q_Values[current_state, action] + alpha*(reward(state_next) + gamma*max(Q_Values[state_next, ]) - Q_Values[current_state, action])
            #print("====== End of Espisode ========")
            break

def policy_define(state, epsilon):
    action = ""
    # Explore
    if np.random.uniform(0, 1) < epsilon:
        action =  random.choice(['L', 'R'])
        #print("action is random:", action)
    # Exploit
    else: 
        next_left = State_Action_Pairs['({}, {})'.format(state, 'L')]
        if next_left == 'LT':
            value_left = Q_Values[state, 'L'] + alpha*(reward(next_left) + gamma*Q_Values[next_left, ] - Q_Values[state, 'L'])
        else:
            value_left1 = Q_Values[state, 'L'] + alpha*(reward(next_left) + gamma*Q_Values[next_left, 'L'] - Q_Values[state, 'L'])
            value_left2 = Q_Values[state, 'L'] + alpha*(reward(next_left) + gamma*Q_Values[next_left, 'R'] - Q_Values[state, 'L'])
            value_left = max(value_left1, value_left2)
        
        next_right = State_Action_Pairs['({}, {})'.format(state, 'R')]
        if next_right == 'RT':
            value_right = Q_Values[state, 'R'] + alpha*(reward(next_right) + gamma*Q_Values[next_right, ] - Q_Values[state, 'R'])
        else:
            value_right1 = Q_Values[state, 'R'] + alpha*(reward(next_right) + gamma*Q_Values[next_right, 'L'] - Q_Values[state, 'R'])
            value_right2 = Q_Values[state, 'R'] + alpha*(reward(next_right) + gamma*Q_Values[next_right, 'R'] - Q_Values[state, 'R'])
            value_right = max(value_right1, value_right2)
        
        action = 'L' if value_left > value_right else 'R'

        #print("action is greedy:", action)
        
    return action

def reward(state):
    if state == 'LT' :
        reward = 0
    elif state == 'RT':
        reward = 100
    else: 
        reward = -1
        
    return reward


def initialize_values(States):
    def reset_values(state, action):
        if state == 'LT' or state == 'RT':
            Q_Values[state, ] = 0
        else:
            Q_Values[state, action] = 0
            
    for state in States:
        for action in Actions:
            reset_values(state, action)

max_episodes = 100000
episodes = 0
initialize_values(States)
# Sarsa 'A'
while episodes < max_episodes:
    current_state = 'A'
    initialized_action = policy_define(current_state, epsilon)
    # SARSA learning method
    SARSA_learning(current_state, initialized_action, epsilon) 
    
    if episodes % 100 == 0:
        print("SARSA", Q_Values)
    episodes += 1

episodes = 0
initialize_values(States)
# Q-learning 'A'
while episodes < max_episodes:
    current_state = 'A'
    initialized_action = policy_define(current_state, epsilon)
    # Q_learning method
    Q_learning(current_state, initialized_action, epsilon) 
    
    if episodes % 100 == 0:
        print("Q Learning", Q_Values)
    episodes += 1
    