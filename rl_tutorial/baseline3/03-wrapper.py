import gym
from stable_baselines3 import A2C, SAC, PPO, TD3
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np

""" refer to

https://github.com/openai/gym/tree/master/gym/wrappers
https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/2_gym_wrappers_saving_loading.ipynb
"""

#========= time limit
class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env, max_steps=100):
    # Call the parent constructor, so we can access self.env later
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    return obs, reward, done, info


print("=== test time limit")
# Here we create the environment directly because gym.make() already wrap the environement in a TimeLimit wrapper otherwise
env = PendulumEnv()
# Wrap the environment
env = TimeLimitWrapper(env, max_steps=100)
obs = env.reset()
done = False
n_steps = 0
while not done:
  # Take random actions
  random_action = env.action_space.sample()
  obs, reward, done, info = env.step(random_action)
  n_steps += 1

print(n_steps, info)


# action normalized

class NormalizeActionWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  """
  def __init__(self, env):
    # Retrieve the action space
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box), "This wrapper only works with continuous action space (spaces.Box)"
    # Retrieve the max/min values
    self.low, self.high = action_space.low, action_space.high

    # We modify the action space, so all actions will lie in [-1, 1]
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=action_space.shape, dtype=np.float32)

    # Call the parent constructor, so we can access self.env later
    super(NormalizeActionWrapper, self).__init__(env)
  
  def rescale_action(self, scaled_action):
      """
      Rescale the action from [-1, 1] to [low, high]
      (no need for symmetric action space)
      :param scaled_action: (np.ndarray)
      :return: (np.ndarray)
      """
      return self.low + (0.5 * (scaled_action + 1.0) * (self.high -  self.low))

  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    # Rescale action from [-1, 1] to original [low, high] interval
    rescaled_action = self.rescale_action(action)
    obs, reward, done, info = self.env.step(rescaled_action)
    return obs, reward, done, info

print("=== test action normalizer")
original_env = gym.make("Pendulum-v0")

print(original_env.action_space.low)
for _ in range(10):
  print(original_env.action_space.sample(), end="")
print("")

env = NormalizeActionWrapper(gym.make("Pendulum-v0"))

print(env.action_space.low)

for _ in range(10):
  print(env.action_space.sample() , end="")
print("")


#==== monitor wrapper
print("test moinitor wrapper")
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
env = Monitor(gym.make('Pendulum-v0'))
env = NormalizeActionWrapper(env)
env = DummyVecEnv([lambda: env])
model = A2C("MlpPolicy", env, verbose=1).learn(int(1000))


#===== time feature wrapper (changing observation space)
from gym.wrappers import TimeLimit

class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))
