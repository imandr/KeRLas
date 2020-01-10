import gym
from ac import ACAgent
#from qac import QACAgent
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

opts, args = getopt.getopt(sys.argv[1:], "t:vn:g:m:l:s:w:r:T:P:a:p:c:b:")
opts = dict(opts)
report_interval = int(opts.get("-r", 10))
test_interval = opts.get("-t")
num_tests = opts.get("-n", 10)
do_render = "-v" in opts
gamma = float(opts.get("-g", 0.99))
n_copies = int(opts.get("-m", 30))
n_pretrain = int(opts.get("-p", 10))
load_from = opts.get("-l", opts.get("-w"))
save_to = opts.get("-s", opts.get("-w"))
title = opts.get("-T")
port = int(opts.get("-P", 8080))
agent_class = opts.get("-a", "ac")
comment = opts.get("-c", "")
batch_size = int(opts.get("-b", 10))

Agent = {
    "ac":   ACAgent,
    #"qac":   QACAgent
}[agent_class]

print ("Agent class:", Agent.__name__)


if test_interval is not None:   test_interval = int(test_interval)
if num_tests is not None:   num_tests = int(num_tests)
env_name = env_names[args[0]]



num_episodes = 10000
monitor = Monitor("monitor.csv", 
    title = title,
    metadata = dict(
        gamma=gamma,
        agent_class=Agent.__name__,
        pretrain=n_pretrain,
        copies=n_copies,
        test_interval=test_interval,
        report_interval=report_interval,
        comment = comment,
        environment = env_name,
        batch_size = batch_size
    ),
    plots=[
    [
        {
            "label":        "train score"
        },
        {
            "label":        "average test score"
        }
        {
            "label":        "min train score",
            "line_width":   1.0
        },
        {
            "label":        "max train score",
            "line_width":   1.0
        }
    ],
    [
        {
            "label":        "min test score",
            "line_width":   1.0
        },
        {
            "label":        "average test score"
        },
        {
            "label":        "max test score",
            "line_width":   1.0
        }
    ]
]
)
monitor.start_server(port)

#
# 1. Pre-train multiple agents and choose the best one
#
env = gym.make(env_name)
num_actions = env.action_space.n
observation_shape = env.observation_space.shape
assert len(observation_shape) == 1
observation_dim = observation_shape[0]

if load_from is None:
    pretrain_episodes = 20
    agents = [Agent(observation_dim, num_actions, 0.00001, 0.00005, gamma=gamma) for _ in range(n_pretrain)]
    score_records = [[] for _ in range(n_pretrain)]

    for i, agent in enumerate(agents):
        for t in range(pretrain_episodes):
            score, _ = agent.run_episode(env, learn=True)
            score_records[i].append(score)
            #monitor.add(t, data = {"score_%d" % (i,): score})
        print ("agent:", i, "  mean score:", np.mean(score_records[i]))

    mean_scores = sorted([(np.mean(record), i) for i, record in enumerate(score_records)], reverse=True)
    print("Pre-train scores:")
    for s, i in mean_scores:
        print (s)

    _, ibest = mean_scores[0]
    agent = agents[ibest]
else:
    agent = Agent(observation_dim, num_actions, 0.00001, 0.00005, gamma=gamma)
    agent.load(load_from)
    print("Agent weights loaded from:", load_from)

#
# 2. Multi-train the best agent
#

print("--- multi-training ---")

envs = [gym.make(env_name) for _ in range(n_copies)]

trainer = MultiTrainer(agent, envs)
score_smoother = Smoothie(0.01)

best_test_score = None

monitor.reset()
next_test = test_interval
for t, score in trainer.train(num_episodes, report_interval=report_interval, batch_size = batch_size):
    min_score, score_ma, max_score = score_smoother.update(score)
    print("Training: episodes=%4d score: %.3f" % (t, score))
    monitor.add(t, {
        "train score":      score, 
        "min train score":  min_score, 
        "max train score":  max_score
    })
    
    if next_test is not None and t >= next_test:
        min_score, avg_score, max_score = trainer.test(num_tests, render=do_render)
        print("Testing:                score: min:%.3f, average:%.3f, max:%.3f" % (min_score, avg_score, max_score))
        if (best_test_score is None or avg_score > best_test_score) and save_to is not None:
            best_test_score = avg_score
            agent.save(save_to)
            print("Agent weights saved to:", save_to)
        monitor.add(t, {
            "min test score":       min_score, 
            "average test score":   avg_score, 
            "max test score":       max_score
        })
        next_test += test_interval

