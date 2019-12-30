import gym, os
from actor_critic_keras import Agent
from utils import plotLearning
from gym import wrappers
import numpy as np
from monitor import Monitor
from smoothie import Smoothie

agent = Agent(alpha=0.00001, beta=0.00005)
monitor = Monitor("monitor.csv")
monitor.start_server(8080)
score_smoother = Smoothie(0.01)

env = gym.make('LunarLander-v2')
score_history = []
num_episodes = 3000

for t in range(num_episodes):
    done = False
    score = 0
    observation = env.reset()
    actor_metrics, critic_metrics = None, None
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        actor_metrics, critic_metrics = agent.learn(observation, action, reward, observation_, done)
        observation = observation_
        score += reward

    score_history.append(score)
    min_score, score_ma, max_score = score_smoother.update(score)
    print(t, score, min_score, score_ma, max_score, actor_metrics, critic_metrics)
    monitor.add(t, score_MA= score_ma, min_score=min_score, max_score=max_score, score=score)

filename = 'LunarLander.png'
plotLearning(score_history, filename=filename, window=100)
