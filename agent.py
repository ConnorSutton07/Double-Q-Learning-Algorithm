""" 
Defines the Deep-Q Learning agent that will
learn to navigate the grid-space

"""
import random
import numpy as np
import argparse
from copy import deepcopy

random.seed(0)
np.random.seed(0)

class AgentQ:
    def __init__(self) -> None:
        self.gamma = 0.99
        self.alpha = 0.01
        self.exp_rate = 1
        self.decay = 0.99
        self.q_table = np.zeros((9, 8))
        self.q_target = np.zeros((9, 8))
        self.copy_steps = 10
        self.actions = [(a, b) for a in range(-1, 2)
                                for b in range(-1, 2)
                                 if (a, b) != (0, 0)]

    def act(self, state: tuple) -> tuple:
        if random.random() < self.exp_rate:
            action_id = random.randint(0, 7)
        else:
            state = state[0] * 3 + state[1]
            action_id = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))
        
        self.exp_rate *= self.decay 
        return action_id, self.actions[action_id]

    def copy(self) -> np.array:
        self.q_target = deepcopy(self.q_table)

    def q_update(self, 
                 state: tuple, 
                 action_id: int, 
                 reward: int, 
                 next_state: tuple, 
                 terminal: bool) -> None:
        state = state[0] * 3 + state[1]
        next_state = next_state[0] * 3 + next_state[1]
        if terminal:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_target[next_state])
        td_error = target - self.q_table[state, action_id]
        self.q_table[state, action_id] = self.q_table[state, action_id] + self.alpha * td_error

    def best_policy(self) -> np.array:
        policy = np.zeros((3, 3), dtype='<U5') # string data type
        states = [(i, j) for i in range(3) for j in range(3)]
        for s in states:
            state = s[0] * 3 + s[1]
            if s == (0, 2):
                policy[s] = '*'
            else:
                action_id = np.argmax(self.q_table[state])
                action = self.actions[action_id]
                policy[s] = translate_action(action)
        return policy



def translate_action(action: tuple) -> str:
    """ 
    Translates the action taken by the agent
    to the corresponding direction represented
    by an arrow.

    """
    if action == (-1, -1):
        return "↖"
    if action == (0, -1):
        return "←"
    if action == (1, -1):
        return "↙"
    if action == (1, 0):
        return "↓"
    if action == (1, 1):
        return "↘"
    if action == (0, 1):
        return "→"
    if action == (-1, 1):
        return "↗"
    if action == (-1, 0):
        return "↑"
    else:
        return "X"