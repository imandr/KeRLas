import numpy as np
import gym

class Env(gym.Env):
    NAgents = 1
    Delta = 0.1
    NActions = 4
    StateDim = 2

    Moves = np.array(
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0)
        ]
    ) * Delta
    
    def __init__(self, tlimit=None):
        self.TLimit = self.T = tlimit

    def randomStates(self, n):
        return np.random.random((n,2))
    
    def randomActions(self, n):
        return np.random.randint(0, self.NActions, n)
        
    def randomAction(self):
        return self.randomActions(1)[0]
        
    def reset(self, random=False):
        self.T = self.TLimit
        self.state = self.randomStates(1)[0]
        return self.state
        
    def step(self, action):
        s0 = self.state
        s1 = s0 + self.Moves[action]
        x, y = s1
        done = x > 1.0 or y > 1.0 or x < 0.0 or y < 0.0
        reward = -0.01
        if done:
            if x > 1:   reward = 1-2*y
            elif y > 1:   reward = 1-2*x
        self.state = s1
        if self.T is not None:
            self.T -= 1
            if self.T < 0: done = True
        return s1, reward, done, {}
        
    
        
        
    