import enum
import numpy as np
import pprint
from rl_tutorial.lib.envs.gridworld import GridworldEnv
from rl_tutorial.DP.policy_evaluation import policy_eval
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def policy_iteration(env, policy_eval_fn=policy_eval, discout_factor=1.0):
    """ Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    refer to:
    Sutton RL book, Chapter 4.2
    https://github.com/linhongbin-ws/reinforcement-learning
    """    

    policy = np.ones([env.nS, env.nA]) / env.nA  # Start with a random policy
    while True:
        V  = policy_eval_fn(policy, env, discount_factor=discout_factor)
        policy_old = policy.copy()
        for s in range(env.nS):
            q_s = np.zeros(len(policy[s])) # q in state s
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]: # for each state and action, look at all the possible next state
                    q_s[a]+=prob * (reward + discout_factor * V[next_state])
                
                policy[s] = 0 # set probability to zero
                policy[s][np.argmax(q_s)] = 1
            
        if np.array_equal(policy_old, policy): # stable
            break
    
    return policy, V

policy, v = policy_iteration(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")
policy_mat = np.reshape(np.argmax(policy, axis=1), env.shape)
direction_map = {0:'^', 1:'>', 2:'v', 3:'<'}
for i in range(policy_mat.shape[0]):
    for j in range(policy_mat.shape[1]):
        print(direction_map[policy_mat[i][j]], end="")
    print("")