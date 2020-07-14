import numpy as np

import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Dense, Flatten

from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback, ModelIntervalCheckpoint

#window length, state size
WINDOW_LENGTH = 1
INPUT_SHAPE = (WINDOW_LENGTH, 8) 

def Model(num_actions):
    return Sequential([
        Flatten(input_shape=INPUT_SHAPE),
        Dense(128, activation="relu"),
        Dense(62, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_actions, activation="linear")
    ])

def Memory():
    return SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)   

def Policy(num_steps=75000):
    return EpsGreedyQPolicy()

def Callbacks():
    return [ModelIntervalCheckpoint("./checkpoints/{step}_weights.h5f", interval=10000)]
            #VisualCallback()]


class VisualCallback(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1

        if self.total_steps % self.interval == 0:
            self.model.test(self.env, nb_episodes=1, nb_max_episode_steps=800, visualize=True)
            self.env.close()
            return
        return
        
