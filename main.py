import numpy as np
import gym

import warnings
warnings.filterwarnings('ignore')

from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent

from agent_deps import Model, Memory, Policy, Callbacks

def main():
    print("Lunar Lander DQN")

    agent = -1

    while True:
        print()
        print("1. Train\n2. Load\n3. Demo\n0. Exit")
        user = input()
        
        if user == "0":
            break
        elif user == "1":
            train()
        elif user == "2":
            agent = load()
        elif user == "3":
            if agent == -1: 
                print("[-] Please load or train model before Demo")
                continue

            demo(agent)


env = gym.make("LunarLander-v2")
np.random.seed(123)
env.seed(123)

num_actions = env.action_space.n
num_steps = 400000

def build_agent():
    model = Model(num_actions)
    memory = Memory()
    policy = Policy(num_steps=num_steps)

    print(model.summary())

    dqn = DQNAgent(model=model, nb_actions=num_actions, 
                    policy=policy, 
                    memory=memory,
                    nb_steps_warmup=1000,
                    gamma=.99, target_model_update=2e-1)

    dqn.compile(Adam(lr=.0001), metrics=['mae'])

    return dqn


def train():
    callbacks = Callbacks()
    agent = build_agent()

    agent.fit(env, callbacks=callbacks, nb_steps=num_steps, 
                nb_max_episode_steps=1000, log_interval=5000)
    agent.save_weights("./saved/weights.h5f")
    

def load():
    agent = build_agent()
    agent.load_weights("./saved/weights.h5f")

    return agent


def demo(agent):
    while True:
        try:
            agent.test(env, nb_episodes=1, nb_max_episode_steps=1000, visualize=True)
        except:
            return


if __name__ == "__main__":
    main()