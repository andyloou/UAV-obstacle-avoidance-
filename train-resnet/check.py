import numpy as np
import gym
import gym_airsim

#thu vien de tao va dieu khien bien 
import argparse

#thu vien de tao model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import keras
from keras import regularizers 
from keras.layers import add

from PIL import Image

#thu vien mo phong
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

#thu vien luu lai model dqn
from rl.callbacks import FileLogger
from keras.callbacks import History

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='airsim')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n
print(env.actions_space)