import gym
from ac import Agent
import numpy as np
from monitor import Monitor
from trainer import Trainer
from smoothie import Smoothie

agent = Agent(0.00001, 0.00005)
env = gym.make("LunarLander-v2")
score_history = []
num_episodes = 10000
monitor = Monitor("monitor.csv")
monitor.start_server(8080)
trainer = Trainer(agent, env, train_interval=1, mb_size=30, shuffle=False, reverse=False)
score_smoother = Smoothie(0.01)

if False:
    for t, actor_metrics, critic_metrics, score in trainer.train(num_episodes, report_interval=1):
        min_score, score_ma, max_score = score_smoother.update(score)
        print(t, score, min_score, score_ma, max_score)
        monitor.add(t, score_MA= score_ma, min_score=min_score, max_score=max_score, score=score)
else:
    for t in range(num_episodes):
        score, _, actor_metrics, critic_metrics = trainer.run_episode(learn=True)
        min_score, score_ma, max_score = score_smoother.update(score)
        print(t, score, min_score, score_ma, max_score, actor_metrics, critic_metrics)
        monitor.add(t, score_MA= score_ma, min_score=min_score, max_score=max_score)
