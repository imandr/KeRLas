import numpy as np
import random
import gym

class Env(gym.Env):
    NAgents = 1
    Delta = 0.1
    Width = 1.0
    NActions = 2
    StateDim = 1
    
    Moves = [-Delta*0.937, Delta*1.1]

    def __init__(self, tlimit=None):
        self.TLimit = self.T = tlimit

    def reset(self):
        self.T = self.TLimit
        self.state = random.random()
        return self.vector()
        
    def vector(self):
        return np.array([self.state])
        
    def step(self, action):
        x0 = self.state
        x1 = x0 + self.Moves[action]
        r = 0.0
        done = False
        if x1 < 0.0 or x1 > self.Width or (x1 > 0.35 and x1 < 0.4):
            done = True
            r = -1.0
        if self.T is not None:
            self.T -= 1
            if self.T <= 0:
                done = True
        self.state = x1
        #print("env.state: state:", self.state)
        return self.vector(), r, done, {}
        
    
        
        
    