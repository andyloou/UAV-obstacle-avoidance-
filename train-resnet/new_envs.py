import numpy as np
import gym


import controller.rl.gym_airsim


import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Concatenate, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import keras
from keras import regularizers 
from keras.layers.merge import add

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from controller.rl.callbacks import FileLogger

from keras.callbacks import History

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='airsim-v1')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

INPUT_SHAPE = (30, 100)
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 144, 256, 3
WINDOW_LENGTH = 1
#Creating lists to keep track of reward and epsilon values
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
print (input_shape)

model = Sequential()
model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape, data_format = "channels_first"))
model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
model.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


train = True

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
# memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)    #reduce memmory
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)
# number step ???

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=False, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


if train:
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=100)]
    
    # dqn.fit(env, nb_steps=1000, action_repetition=1, callbacks=callbacks, verbose=1, visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000, nb_max_episode_steps=None)
    dqn.fit(env, nb_steps=20000, visualize=True, verbose=2)
    
    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)


else:

    dqn.load_weights('dqn_{}_weights.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=10, visualize=False)
