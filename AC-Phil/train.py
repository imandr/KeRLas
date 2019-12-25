import gym
from ac import Agent
from monitor import Monitor, http_server
import numpy as np

agent = Agent(0.0001, 0.0005)
env = gym.make("LunarLander-v2")
num_episodes = 2000

class Window(object):
    def __init__(self, alpha = 0.1):
        self.Low = self.High = None
        self.Alpha = alpha
        
    def update(self, x):
        if self.Low is None:
            self.Low = self.High = x
        if x < self.Low:
            self.Low = x
            self.High += self.Alpha*(x - self.High)
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += self.Alpha*(x - self.Low)
        else:
            self.Low += self.Alpha/10*(x - self.Low)
            self.High += self.Alpha/10*(x - self.High)
            
        return self
        
    __lshift__ = update
    
    def get(self):
        return self.Low, self.High
        
    __call__ = get

class Trainer(object):
    
    def __init__(self, agent, env, mb_size = 30, train_interval = 10, shuffle = True):
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
        actor_metrics, critic_metrics = None, None
        score_ma = None
        score_ma_alpha = 0.01
        score_record = []
        for t in range(num_episodes):
            score, record = self.run_episode()
            score_record.append(score)
            if len(score_record) < 100:
                score_ma = np.mean(score_record)
            else:
                score_ma += score_ma_alpha * (score - score_ma)
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
                history = []
                yield t, actor_metrics, critic_metrics, score, score_ma
        yield t, actor_metrics, critic_metrics, score, score_ma
        
class Tester(object):
    def __init__(self, agent, env, tau=10.0):
        self.Env = env
        self.Agent = agent
        self.Tau = tau
        
    def run_episode(self, render=False, tau=None):
        tau = self.Tau if tau is None else tau
        done = False
        observation = self.Env.reset()
        if render:
            self.Env.render()
        score = 0.0
        record = []
        while not done:
            action = self.Agent.choose_action(observation, tau=tau)
            observation_, reward, done, info = self.Env.step(action)
            if render:
                self.Env.render()
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            observation = observation_
            score += reward
        return score, record
        
    def test(self, num_episodes=10, tau=None):
        tau = self.Tau if tau is None else tau
        sum_score = 0.0
        for t in range(num_episodes):
            score, record = self.run_episode(tau=tau)
            sum_score += score
        return sum_score/num_episodes
        
    def display(self, num_episodes=1, tau=None):
        tau = self.Tau if tau is None else tau
        for t in range(num_episodes):
            score, record = self.run_episode(render=True, tau=tau)
            print("display score:", score)
        
            
    
    
                
trainer = Trainer(agent, env, train_interval=10, mb_size=30)
tester = Tester(agent, env, tau=10.0)
monitor = Monitor("monitor.csv")
mon_server = http_server(8080, monitor)
best_average = None
best_average_t = t = 0
next_display = display_interval = 100

test_window = Window()
avg_score_window = Window()

mon_server.start()

for num_trained, actor_metrics, critic_metrics, score, avg_score in trainer.train(num_episodes, tau=0.2):
    avg_score_window << avg_score
    test_score = tester.test()
    test_window << test_score
    print("Episodes: %6d   score: %+8.2f   average score: %+8.2f   average score_window: %+8.2f:%+8.2f   test score: %+8.2f" % \
            (num_trained, score, avg_score, avg_score_window.Low, avg_score_window.High, test_score))
            
    monitor.add(num_trained, score=score, avg_score=avg_score, 
        avg_low=avg_score_window.Low, avg_high=avg_score_window.High    
        #test_score=test_score
        )
    if num_trained >= next_display:
        tester.display(tau=1.0)
        next_display += display_interval
