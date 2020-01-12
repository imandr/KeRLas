import gym
from ac import ACAgent as Agent
import numpy as np
from monitor import Monitor
from smoothie import Smoothie
import getopt, sys

env_names = {
    "lander":   "LunarLander-v2",
    "cartpole": "CartPole-v1"
}

opts, args = getopt.getopt(sys.argv[1:], "vn:l:")
opts = dict(opts)
num_tests = int(opts.get("-n", 100))
do_render = "-v" in opts
load_from = opts.get("-l")
env_name = env_names[args[0]]

env = gym.make(env_name)
num_actions = env.action_space.n
observation_shape = env.observation_space.shape
assert len(observation_shape) == 1
observation_dim = observation_shape[0]

agent = Agent(observation_dim, num_actions, 0.00001, 0.00005)
if load_from:
    agent.load(load_from)
    print("\n\nAgent weights loaded from:", load_from,"\n\n")

for t in range(num_tests):
    score, _ = agent.run_episode(env, test=True, render=do_render)
    print(score)