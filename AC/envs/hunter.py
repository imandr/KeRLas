import random
import numpy as np

class HunterEnv(object):
    
    SIZE = 20       # 20x20 field
    VR = 2      # visible range: [x-2, x-1, x, x+1, x+2]
    SPAWN_MIN = 2
    SPAWN_MAX = 10
    SPAWN_ATTEMPTS = 3
    
    MOVES = [(i,j) for i in (-1,0,1) for j in (-1,0,1)]
    
    def __init__(self):
        self.Field = np.ones((self.SIZE+self.VR*2, self.SIZE+self.VR*2), dtype=np.float) * -1
        self.NextSpawn = random.randint(self.SPAWN_MIN, self.SPAWN_MAX)
        self.Hunter = None
        self.Actions = np.arange(9)
        self.LastAction = 4
        self.ObservationDim = 9+1
        
    def scan(self, x, y):
        window = self.Field[x+self.VR-self.VR:x+self.VR+self.VR+1, y+self.VR-self.VR:y+self.VR+self.VR+1].copy()
        return window.reshape((-1,))
        
    def observation(self):
        obs = np.enmpty((self.ObservationDim,))
        obs[0:9] = self.scan(*self.Hunter)
        obs[9] = self.LastAction
        return obs
        
    def free_cell(self, n_attempts=None):
        i = 0
        while n_attempts is None or i < n_attempts:
            x = random.randint(0, self.SIZE-1)
            y = random.randint(0, self.SIZE-1)
            if self.Field[x+self.VR,y+self.VR] == 0:
                return x, y
            i += 1
        else:
            return None
            
    def spawn(self):
        location = self.free_cell(self.SPAWN_ATTEMPTS)
        if location is not None:
            x, y = location
            self.Field[x+self.VR, y+self.VR] = random.randint(self.SPAWN_MIN, self.SPAWN_MAX)
        
    def reset(self):
        self.Field[self.VR:self.SIZE+self.VR, self.VR:self.SIZE+self.VR] = 0.0
        self.spawn()
        self.NextSpawn = random.randint(self.SPAWN_MIN, self.SPAWN_MAX)
        self.Hunter = self.free_cell()
        return self.observation()
        
    def step(self, action):
        x, y = self.Hunter
        dx, dy = self.MOVES[action]
        x1, y1 = x+dx, y+dy
        if x - dx < 0 or x + dx > self.SIZE - 1 or y - dy < 0 or y + dy > self.SIZE - 1:
            x1, y1 = x, y
        reward = 0.0
        if self.Field[x+self.VR,y+self.VR] > 0.0:
            reward += 1.0
            self.Field[x+self.VR,y+self.VR] -= 1.0
        self.Hunter = x, y
        return self.observation()
        
    def render(self):
        pass
        
    
        
        