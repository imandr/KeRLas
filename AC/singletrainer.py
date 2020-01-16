import numpy as np

class SingleTrainer(object):
    
    def __init__(self, agent, env):
        self.Agent = agent
        self.Env = env

    def train(self, num_episodes, report_interval=1, epsilon0 = 0.01, epsilon1 = 0.001):
        next_report = report_interval
        epsilon = epsilon0
        depsilon = (epsilon1-epsilon0)/(num_episodes*0.5)
        for episode in range(num_episodes):
            score, record = self.Agent.run_episode(self.Env, learn=True, epsilon=epsilon)
            if episode+1 >= next_report:
                yield episode+1, score
                next_report += report_interval

    def test(self, num_episodes, render=False):
        env = self.Env
        scores = np.zeros((num_episodes,))
        for t in range(num_episodes):
            score, record = self.Agent.run_episode(env, test=True, render=render)
            scores[t] = score
        return min(scores), np.mean(scores), max(scores)
