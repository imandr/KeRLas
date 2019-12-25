import gym
from ac import Agent
import numpy as np
from monitor import Monitor

agent = Agent(0.00001, 0.00005)
env = gym.make("LunarLander-v2")
score_history = []
num_episodes = 2000
monitor = Monitor("monitor.csv")

for t in range(num_episodes):
    done = False
    observation = env.reset()
    score = 0.0
    
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.learn(observation, action, reward, observation_, done)
        observation = observation_
        score += reward
    
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    print("Episode:", t, "  score:", score, "  average score:", avg_score)
    monitor.add(t, average_score = avg_score)