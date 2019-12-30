import gym
from ac import Agent
import numpy as np
from monitor import Monitor
from multitrainer import MultiTrainer
from smoothie import Smoothie
from tqdm import tqdm, trange

n_envs = 10

num_episodes = 10000
monitor = Monitor("monitor.csv")
monitor.start_server(8080)

#
# 1. Pre-train multiple agents and choose the best one
#
pretrain_episodes = 20
agents = [Agent(0.00001, 0.00005) for _ in range(n_envs)]
score_records = [[] for _ in range(n_envs)]
env = gym.make("LunarLander-v2")

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

envs = [gym.make("LunarLander-v2") for _ in range(n_envs)]

trainer = MultiTrainer(agent, envs)
score_smoother = Smoothie(0.01)

monitor.reset()

for t, score in trainer.train(num_episodes, report_interval=1):
    min_score, score_ma, max_score = score_smoother.update(score)
    print(t, score, min_score, score_ma, max_score)
    monitor.add(t, score_MA= score_ma, min_score=min_score, max_score=max_score, score=score)
