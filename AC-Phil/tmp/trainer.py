import numpy as np

class Trainer(object):
    
    def __init__(self, agent, env, mb_size = 30, train_interval = 10, shuffle = True, reverse = False):
        self.Agent = agent
        self.Env = env
        self.BatchSize = mb_size
        self.TrainInterval = train_interval
        self.Shuffle = shuffle
        self.Reverse = reverse

    def run_episode(self, learn = False):
        done = False
        observation = self.Env.reset()
        score = 0.0
        record = []
        while not done:
            action = self.Agent.choose_action(observation)
            observation_, reward, done, info = self.Env.step(action)
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            if learn:
                self.Agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        return score, record
        
    def train(self, num_episodes, tau=1.0, report_interval=1):
        history = []
        next_train = self.TrainInterval
        actor_metrics, critic_metrics = None, None
        score_record = []
        next_report = report_interval
        for t in range(num_episodes):
            score, record = self.run_episode()
            score_record.append(score)
            history += record
            t += 1
            if t >= next_train:
                next_train += self.TrainInterval
                if self.Reverse:        history.reverse()
                x0, a, r, x1, f, infos = zip(*history)
                x0 = np.array(x0)
                a = np.array(a)
                r = np.array(r)
                x1 = np.array(x1)
                f = np.array(f)
                actor_metrics, critic_metrics = self.Agent.learn_batches(self.BatchSize, x0, a, r, x1, f, tau=tau, shuffle=self.Shuffle)
                history = []
                if t >= next_report:
                    yield t, actor_metrics, critic_metrics, score
                    next_report += report_interval
        yield t, actor_metrics, critic_metrics, score
        
