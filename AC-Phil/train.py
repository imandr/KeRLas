import gym
from ac import Agent
import numpy as np

agent = Agent(0.00001, 0.00005)
env = gym.make("LunarLander-v2")
score_history = []
num_episodes = 2000

class Trainer(object):
    
    def __init__(self, agent, env, mb_size = 50, train_interval = 10, shuffle = True):
        self.Agent = agent
        self.Env = env
        self.BatchSize = mb_size
        self.TrainInterval = train_interval
        self.Shuffle = shuffle

    def run_episode(self):
        done = False
        observation = self.Env.reset()
        score = 0.0
        record = []
        while not done:
            action = self.Agent.choose_action(observation)
            observation_, reward, done, info = self.Env.step(action)
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            observation = observation_
            score += reward
        return score, record
        
    def train(self, num_episodes, tau=1.0):
        history = []
        next_train = self.TrainInterval
        num_trained = 0
        score_history = []
        actor_metrics, critic_metrics = None, None
        for t in range(num_episodes):
            score, record = self.run_episode()
            score_history.append(score)
            history += record
            t += 1
            if t >= next_train:
                next_train += self.TrainInterval
                x0, a, r, x1, f, infos = zip(*history)
                x0 = np.array(x0)
                a = np.array(a)
                r = np.array(r)
                x1 = np.array(x1)
                f = np.array(f)
                actor_metrics, critic_metrics = self.Agent.learn_batches(self.BatchSize, x0, a, r, x1, f, tau=tau, shuffle=self.Shuffle)
                num_trained += len(history)
                history = []
                yield t, actor_metrics, critic_metrics, score, np.mean(score_history[-100:])
        yield t, actor_metrics, critic_metrics, score, np.mean(score_history[-100:])
        
class Tester(object):
    def __init__(self, agent, env, tau=10.0):
        self.Env = env
        self.Agent = agent
        self.Tau = tau
        
    def run_episode(self):
        done = False
        observation = self.Env.reset()
        score = 0.0
        record = []
        while not done:
            action = self.Agent.choose_action(observation, tau=self.Tau)
            observation_, reward, done, info = self.Env.step(action)
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            observation = observation_
            score += reward
        return score, record
        
    def test(self, num_episodes=10):
        sum_score = 0.0
        for t in range(num_episodes):
            score, record = self.run_episode()
            sum_score += score
        return sum_score/num_episodes
            
    
    
                
trainer = Trainer(agent, env, mb_size=30)
tester = Tester(agent, env, tau=100.0)
best_average = None
best_average_t = t = 0
for num_trained, actor_metrics, critic_metrics, score, avg_score in trainer.train(num_episodes, tau=0.2):
    if best_average is None or best_average < avg_score:
        best_average = avg_score
        best_average_t = num_trained
    test_score = tester.test()
    print("Episodes: %6d   score: %+8.2f   average score: %+8.2f   best average: %+8.2f at %-6d --- test score: %+8.2f" % \
            (num_trained, score, avg_score, best_average, best_average_t, test_score))
                
