import numpy as np
import getopt, sys, random
from envs import make_env
from multiagent import CollectiveMultiAgent
from AC import ACBrain

alpha, beta = 0.00001, 0.00005
gamma = 0.99

np.set_printoptions(precision=3)

opts, args = getopt.getopt(sys.argv[1:], "vn:l:")
opts = dict(opts)
n_agents = int(opts.get("-n", 2))
do_render = "-v" in opts
load_from = opts.get("-l")

env = make_env(args[0], n_agents)
actions_shape = env.ActionShape
num_actions = env.NumActions
obs_dim = env.ObservationShape[-1]

brains = [ACBrain(obs_dim, num_actions, alpha, beta, gamma=gamma)]
ma = CollectiveMultiAgent(brains, num_actions)

test_interval = 10
next_test = test_interval
n_tests = 5
eps = eps0 = 0.1
eps1 = 0.002
n_eps = 1000
d_eps = (eps0-eps1)/n_eps

for t in range(10000):
    scores, _ = ma.run_episode(env, learn=True, epsilon=eps, render=True)
    print(t, eps, scores)
    eps = max(eps-d_eps, eps1)
    
    if t+1 >= next_test:
        print("testing...")
        for _ in range(n_tests):
            ma.run_episode(env, test=True, learn=False, render=do_render)
        next_test += test_interval

