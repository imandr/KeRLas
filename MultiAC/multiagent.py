#from keras import backend as K
#from keras.layers import Dense, Input, Reshape, LSTM, concatenate
#from keras.models import Model
#from keras.optimizers import Adam
#from keras import regularizers
import numpy as np

class MultiAgent(object):
    def __init__(self, brains, n_actions):
        self.Brains = brains
        self.ActionSpace = np.arange(n_actions)
        
    def observation_map(self, observtions):
        # map observation (i, observation) to a brain. Returns brain index
        # returns map = [] so that obs[i] -> self.Brains[map[i]]
        raise NotImplementedError
        
    def reset(self):
        [b.reset() for b in self.Brains]
        
    def choose_action(self, observation, brain, test = False, epsilon=0.01):
        if observation is None: return None
        probs = brain.action_weights(observation)
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
        
    def actions(self, obmap, observations, test, epsilon):
        return [self.choose_action(o, self.Brains[i], test, epsilon) 
                    for o, i in zip(observations, obmap)]
            
    def play(self, env, test=False, render=False, epsilon=0.01, learn=False):
        done = False
        observations = env.reset()
        self.reset()
        if render:
            env.render()
        self.Scores = [0.0 for _ in self.Brains]
        self.Records = [[] for _ in self.Brains]
        while not done:
            #print("run_episode: eps:", epsilon)
            obmap = self.observation_map(observations)
            actions = self.actions(obmap, observations, test, epsilon)
            observations_, rewards, dones, infos = env.step(actions)
            #print("step:",observations_, rewards, dones, infos)
            for ib, r in zip(obmap, rewards):
                self.Scores[ib] += r
            if learn:
                batches = {}        # ibrain -> list of (s,a,r,s1,f)
                for io, sarsf in enumerate(zip(observations, actions, rewards, 
                                                observations_, dones)):
                    ib = obmap[io]
                    batch = batches.setdefault(ib, [])
                    batch.append(sarsf)
                for ib, batch in batches.items():
                    states, actions, rewards, states_, dones = map(np.array, zip(*batch))
                    #print(type(states), type(actions), type(rewards), type(states_), type(dones))
                    self.Brains[ib].learn_batch(states, actions, rewards, states_, dones)
            yield observations, actions, observations_, rewards, dones, infos
            if render:
                env.render()
            for i, tup in enumerate(zip(observations, actions, observations_, rewards, dones)):
                ib = obmap[i]
                #print("appending tuple", tup, " to ", ib)
                self.Records[ib].append(tup)
            observations = [o_ for o_, d in zip(observations_, dones) if not d]
            done = all(dones)

    def run_episode(self, env, test=False, render=False, epsilon=0.01, learn=False):
        for _ in self.play(env, test, render, epsilon, learn=learn):
            pass
        return self.Scores, self.Records

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
        

class CollectiveMultiAgent(MultiAgent):
    
    def observation_map(self, observtions):
        return [0]*len(observtions)
