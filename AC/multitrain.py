import gym
from ac import Agent
import numpy as np
from monitor import Monitor
from multitrainer import MultiTrainer
from smoothie import Smoothie
from tqdm import tqdm, trange
import getopt, sys

env_names = {
    "lander":   "LunarLander-v2",
    "cartpole": "CartPole-v1"
}

opts, args = getopt.getopt(sys.argv[1:], "t:vn:g:")
opts = dict(opts)
test_interval = opts.get("-t")
num_tests = opts.get("-n", 10)
do_render = "-v" in opts
gamma = float(opts.get("-g", 0.99))

if test_interval is not None:   test_interval = int(test_interval)
if num_tests is not None:   num_tests = int(num_tests)
env_name = env_names[args[0]]


n_copies = 10

num_episodes = 10000
monitor = Monitor("monitor.csv")
monitor.start_server(8080)

#
# 1. Pre-train multiple agents and choose the best one
#
env = gym.make(env_name)
num_actions = env.action_space.n
observation_shape = env.observation_space.shape
assert len(observation_shape) == 1
observation_dim = observation_shape[0]


pretrain_episodes = 20
agents = [Agent(observation_dim, num_actions, 0.00001, 0.00005, gamma=gamma) for _ in range(n_copies)]
score_records = [[] for _ in range(n_copies)]

for i, agent in enumerate(agents):
    for t in range(pretrain_episodes):
        score, _, _, _ = agent.run_episode(env, learn=True)
        score_records[i].append(score)
        #monitor.add(t, data = {"score_%d" % (i,): score})
    print ("agent:", i, "  mean score:", np.mean(score_records[i]))

mean_scores = sorted([(np.mean(record), i) for i, record in enumerate(score_records)], reverse=True)
print("Pre-train scores:")
for s, i in mean_scores:
    print (s)

_, ibest = mean_scores[0]
agent = agents[ibest]

#
# 2. Multi-train the best agent
#

print("--- multi-training ---")

envs = [gym.make(env_name) for _ in range(n_copies)]

trainer = MultiTrainer(agent, envs)
score_smoother = Smoothie(0.01)

monitor.reset()
next_test = test_interval
for t, score in trainer.train(num_episodes, report_interval=10):
    min_score, score_ma, max_score = score_smoother.update(score)
    print(t, score, min_score, score_ma, max_score)
    monitor.add(t, train_score_ma = score_ma, train_score_min=min_score, train_score_max=max_score, train_score=score)
    
    if next_test is not None and t >= next_test:
        min_score, avg_score, max_score = trainer.test(num_tests, render=do_render)
        monitor.add(t, test_score_min = min_score, test_score_avg = avg_score, test_score_max = max_score)
        next_test += test_interval

