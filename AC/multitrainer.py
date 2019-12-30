import numpy as np

class MultiTrainer(object):
    
    def __init__(self, agent, envs):
        self.Agent = agent
        self.Envs = envs

    def run_episode(self):
        n = len(self.Envs)
        dones = [0.0]*n
        observations = [env.reset() for env in self.Envs]
        obs_dim = observations[0].shape[-1]
        scores = [0.0] * n
        records = [[] for _ in self.Envs]
        
        x0 = np.empty((n, obs_dim), dtype=np.float)
        x1 = np.empty((n, obs_dim), dtype=np.float)
        r = np.empty((n,), dtype=np.float)
        a = np.empty((n,), dtype=np.int)
        f = np.empty((n,), dtype=np.int)
        
        while not all(dones):
            j = 0
            for i, env in enumerate(self.Envs):
                if not dones[i]:
                    obs = observations[i]
                    action = self.Agent.choose_action(obs)
                    observation_, reward, done, info = env.step(action)
                    done = int(done)
                    x0[j,:] = obs
                    x1[j,:] = observation_
                    r[j] = reward
                    a[j] = action
                    f[j] = done
                    j += 1
                    records[i].append((obs, action, reward, observation_, done))
                    observations[i] = observation_
                    scores[i] += reward
                    dones[i] = done
            #print("run_episode: actions:", np.array(a), np.array(a).dtype)
            self.Agent.learn_batch(x0[:j], a[:j], r[:j], x1[:j], f[:j])
        return scores, records
        
    def train(self, num_episodes, report_interval=1):
        score_record = []
        next_report = report_interval
        avg_score = None
        for t in range(num_episodes):
            scores, records = self.run_episode()
            avg_score = np.mean(scores)
            score_record.append(avg_score)
            if t >= next_report:
                yield t, avg_score
                next_report += report_interval
        yield t, avg_score
        
