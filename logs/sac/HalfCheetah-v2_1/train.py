import gym
import numpy as np
import stable_baselines
from stable_baselines.sac.policies import MlpPolicy
import sys
from stable_baselines import SAC,PPO2
#from rlzoo.utils.utils import ALGOS, create_test_env
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.common.buffers
stable_baselines.common.buffers.Memory = stable_baselines.common.buffers.ReplayBuffer
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
#from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

env = gym.make('HalfCheetah-v2')
#
# model = PPO2.load("baxter_lift")
model = SAC.load("best_model")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
