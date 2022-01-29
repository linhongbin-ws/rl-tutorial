import enum
import numpy as np
import pprint
import sys
from rl_tutorial.lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """ estimate the value for policy

    refer to:
    Sutton RL book, Chapter 4.1 
    https://github.com/linhongbin-ws/reinforcement-learning
    """
    V = np.zeros(env.nS) # values
    while True:
        delta = 0
        for s in range(env.nS): # for each state, perform full backup
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]: # for each state and action, look at all the possible next state
                    v+=action_prob * prob * (reward + discount_factor * V[next_state])
                
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v

        if delta <= theta:
            break

    return np.array(V) 

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)
print(v)

# compare with the truth 
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)