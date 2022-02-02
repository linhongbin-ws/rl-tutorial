import enum
import numpy as np
import pprint
import sys
from rl_tutorial.lib.envs.gridworld import GridworldEnv
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """ Value Iteration Algorithm. A tuple (policy, V) of the optimal policy and the optimal value function.  
    
    refer to:
    Sutton RL book, Chapter 4.4
    https://github.com/linhongbin-ws/reinforcement-learning
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.  
    """
    # import ipdb
    # ipdb.set_trace()

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    cnt = 0
    while True:
        Delta = 0
        cnt+=1
        for s in range(env.nS):
            v_prv = V[s]
            
            qs = np.zeros(len(policy[s]))
            for a_idx, action in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a_idx]:
                    qs[a_idx] +=  prob*(reward+discount_factor*V[next_state])
            a_best_idx = np.argmax(qs)
            V[s] = qs[a_best_idx]

            policy[s] = 0 # set other prob to zero
            policy[s][a_best_idx] = 1

            Delta = max(Delta, np.abs(V[s]-v_prv))

        if Delta < theta:
            break
        print("loop {}".format(Delta))
    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

