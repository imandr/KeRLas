import numpy as np
import random
import gym
from gym import spaces, logger

class SimpleLander(gym.Env):

    X0 = 10.0
    G = 0.1
    VLand = 0.1
    Fire = 0.5
    Fuel = 100
    DT = 1.0
    
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(np.array([0.0, -100.0, 0.0]), np.array([X0*2, 100.0, Fuel]))
    
    
    def __init__(self):
        pass

    def reset(self):
        self.x = self.X0 + random.random()
        self.v = (random.random() - 0.5) * self.VLand
        self.f = self.Fuel
        return self.vector()
        
    def vector(self):
        return np.array([self.x, self.v, self.f])
        
    def step(self, action):
        if action == 1:
            if self.f > 0:
                self.v += self.Fire
                self.f -= 1
        
        self.v -= self.G*self.DT
        self.x += self.v*self.DT
        reward = 0.0
        done = False
        if self.x <= 0.0:
            if self.v > -self.VLand:    reward = 1.0
            else:   reward = -1.0
            done = True
        elif self.x > self.X0*2:
            reward = -1.0
            done = True
        #print("step: action:", action, "    vector->", self.vector())
        return self.vector(), reward, done, {}
    
    def render(self):
        print("H:%+7.3f V:%+7.3f F:%3d" % (self.x, self.v, self.f))
    
        
        
    