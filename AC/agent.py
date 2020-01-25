#from keras import backend as K
#from keras.layers import Dense, Input, Reshape, LSTM, concatenate
#from keras.models import Model
#from keras.optimizers import Adam
#from keras import regularizers
import numpy as np

class Agent(object):
    def __init__(self, brain, n_actions):
        self.Brain = brain
        self.ActionSpace = np.arange(n_actions)
        
    def reset(self):
        self.Brain.reset()

    def choose_action(self, observation, test = False, epsilon=0.01):
        probs = self.Brain.action_weights(observation)
        if test:
            epsilon = 0.0
        if False and test:
            #action = np.argmax(probs)
            probs = probs*probs
            probs = probs/np.sum(probs)
        else:
            probs = (probs+epsilon/len(probs))/(1.0+epsilon)
            
        return np.random.choice(self.ActionSpace, p=probs)
        return action
        
    def play(self, env, test=False, render=False, epsilon=0.01, learn=False):
        #if test:
        #    test = False
        #    epsilon = 0.0
        done = False
        observation = env.reset()
        self.reset()
        if render:
            env.render()
        self.Score = 0.0
        self.Record = []
        while not done:
            #print("run_episode: eps:", epsilon)
            action = self.choose_action(observation, test=test, epsilon=epsilon)
            #print("run_episode: obs:", observation, "  action:", action)
            observation_, reward, done, info = env.step(action)
            if learn:
                self.learn(observation, action, reward, observation_, done)
            yield observation, action, observation_, reward, done, info
            if render:
                env.render()
            self.Record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            observation = observation_
            self.Score += reward

    def run_episode(self, env, test=False, render=False, epsilon=0.01, learn=False):
        for _ in self.play(env, test, render, epsilon, learn=learn):
            pass
        return self.Score, self.Record

    def play_many(self, envs, test=False, render=False, epsilon=0.01):
        n = len(envs)
        dones = [0.0]*n
        observations = [env.reset() for env in envs]
        obs_dim = observations[0].shape[-1]
        self.Scores = np.zeros((n,))
        self.Records = [[] for _ in envs]
        
        x0 = np.empty((n, obs_dim), dtype=np.float)
        x1 = np.empty((n, obs_dim), dtype=np.float)
        r = np.empty((n,), dtype=np.float)
        a = np.empty((n,), dtype=np.int)
        f = np.empty((n,), dtype=np.int)
        inx = np.empty((2,n), dtype=np.int)
        
        while not all(dones):
            j = 0
            for i, env in enumerate(envs):
                inx[...] = -1
                if not dones[i]:
                    obs = observations[i]
                    action = self.choose_action(obs, epsilon=epsilon)
                    observation_, reward, done, info = env.step(action)
                    done = int(done)
                    x0[j,:] = obs
                    x1[j,:] = observation_
                    r[j] = reward
                    a[j] = action
                    f[j] = done
                    self.Records[i].append((obs, action, reward, observation_, done))
                    observations[i] = observation_
                    self.Scores[i] += reward
                    dones[i] = done
                    inx[0,j] = i
                    inx[1,i] = j
                    j += 1
                else:
                    self.Records[i].append((None, None, None, None, None))
            yield x0[:j], a[:j], r[:j], x1[:j], f[:j], inx
            
    #
    # Learn actions forwarded to Brain
    #
        
    def learn(self, *params, **args):
        return self.Brain.learn(*params, **args)

    def learn_batch(self, *params, **args):
        return self.Brain.learn_batch(*params, **args)

    def load(self, *params, **args):
        self.Brain.load(*params, **args)
        
    def save(self, *params, **args):
        self.Brain.save(*params, **args)
        
        
