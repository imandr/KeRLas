import numpy as np
import math

class MultiTrainer(object):
    
    def __init__(self, agent, envs):
        self.Agent = agent
        self.Envs = envs

    def run_episode(self, epsilon=0.0, batch_size=10):
        for x0, a, r, x1, f, inx in self.Agent.play_many(self.Envs, epsilon=epsilon):
            self.Agent.learn_batch(x0, a, r, x1, f, batch_size=batch_size)
        return self.Agent.Scores, self.Agent.Records
        
    def train(self, num_episodes, report_interval=1, epsilon0 = 0.1, epsilon1 = 0.001, batch_size = 10):
        epsilon = epsilon0
        depsilon = (epsilon1-epsilon0)/(num_episodes*0.5)
        score_record = []
        next_report = report_interval
        avg_score = None
        t = 0
        while t < num_episodes:
            scores, records = self.run_episode(epsilon, batch_size = batch_size)
            epsilon = max(epsilon1, epsilon+depsilon)
            avg_score = np.mean(scores)
            score_record.append(avg_score)
            t += len(self.Envs)
            if t >= next_report:
                yield t, avg_score
                next_report += report_interval
        yield t, avg_score
        
    def test(self, num_episodes, render=False):
        env = self.Envs[0]
        scores = np.zeros((num_episodes,))
        for t in range(num_episodes):
            score, record = self.Agent.run_episode(env, test=True, render=render)
            scores[t] = score
        return min(scores), np.mean(scores), max(scores)
        
    
