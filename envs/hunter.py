import random
import numpy as np
import math, time
from gym import spaces
from time_limit import TimeLimit

from draw2d import Viewer, Rectangle, Frame, Circle, Line, Polygon

class _HunterEnv(object):
    
    SIZE = 15       # 20x20 field
    VR = 2      # visible range: [x-2, x-1, x, x+1, x+2]
    SCAN_SIZE = (VR*2+1)**2+4
    SPAWN_MIN = 2
    SPAWN_MAX = 10
    SPAWN_ATTEMPTS = 3
    
    MOVES = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]
    ZERO_ACTION = 0
    
    HISTORY_SIZE = 1
    
    def __init__(self):
        self.Field = np.zeros((self.SIZE+self.VR*2, self.SIZE+self.VR*2), dtype=np.float)
        self.NextSpawn = random.random()*(self.SPAWN_MAX-self.SPAWN_MIN) + self.SPAWN_MIN
        self.Hunter = None
        self.Actions = np.arange(5)
        
        self.Fuel = 100.0
        
        self.Viewer = None

        #
        # scan, last_scan, borders, last_action
        self.ObservationDim = (self.SCAN_SIZE + 1)*self.HISTORY_SIZE
        high = np.array([np.inf]*self.ObservationDim)
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(5)
        
        self.LastAction = self.ZERO_ACTION
        self.History = np.zeros((self.HISTORY_SIZE, self.SCAN_SIZE+1))

    def scan(self, x, y):
        out = np.empty((self.SCAN_SIZE,))
        out[0] = -1.0 if x <= 0 else (1.0 if x >= self.SIZE-1 else 0.0)
        out[1] = -1.0 if y <= 0 else (1.0 if y >= self.SIZE-1 else 0.0)
        out[2] = float(x)/float(self.SIZE)
        out[3] = float(y)/float(self.SIZE)
        frame = self.Field[x+self.VR-self.VR:x+self.VR+self.VR+1, y+self.VR-self.VR:y+self.VR+self.VR+1].reshape((-1,))
        assert len(frame) + 4 == self.SCAN_SIZE
        out[4:] = frame
        return out
        
    def observation(self, roll=False):
        if roll:
            x, y = self.Hunter
            self.History = np.roll(self.History, 1, axis=0)
            self.History[0,0:self.SCAN_SIZE] = self.scan(x, y)
            #self.History[0,self.SCAN_SIZE] = self.LastAction
        return self.History.reshape((-1,))
        
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
            v = random.randint(self.SPAWN_MIN, self.SPAWN_MAX)
            if random.random() < 0.4:   v = -v
            self.Field[x+self.VR, y+self.VR] = v
        
    def reset(self):
        self.LastAction = self.ZERO_ACTION
        self.History[...] = 0.0
        self.History[:,self.SCAN_SIZE] = self.ZERO_ACTION
        self.Field[self.VR:self.SIZE+self.VR, self.VR:self.SIZE+self.VR] = 0.0
        self.spawn()
        self.NextSpawn = random.randint(self.SPAWN_MIN, self.SPAWN_MAX)
        self.Hunter = self.free_cell()
        self.Fuel = 100.0
        return self.observation(roll=False)
        
    def step(self, action):
        if 0.03 > random.random():
            self.spawn()
        x, y = self.Hunter
        self.LastAction = action
        dx, dy = self.MOVES[action]
        x1, y1 = x+dx, y+dy
        reward = 0.0
        if x1 < 0 or x1 > self.SIZE - 1 or y1 < 0 or y1 > self.SIZE - 1:
            reward -= 0.1
            x1, y1 = x, y
        if self.Field[x+self.VR,y+self.VR] > 0.0:   # and action != self.ZERO_ACTION:
            reward += 1.0
            self.Fuel += 10.0
            self.Field[x+self.VR,y+self.VR] -= 1.0
        elif self.Field[x+self.VR,y+self.VR] < 0.0:
            reward -= 1.0
            self.Field[x+self.VR,y+self.VR] += 1.0
        self.Fuel -= 0.1
        self.Hunter = x1, y1
        done = False
        if self.Fuel <= 0.0:
            done = True
            reward -= 10.0
        return self.observation(roll=True), reward, done, {}
        
    def render(self):
        if self.Viewer is None:
            self.Viewer = Viewer(800,800)
            self.Frame = self.Viewer.frame(-0.5, self.SIZE-0.5, -0.5, self.SIZE-0.5)
        self.Frame.remove_all()
        self.Frame.add(Rectangle(-0.5, self.SIZE+0.5, -0.5, self.SIZE+0.5).color(0,0,0))
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                if self.Field[self.VR+x, self.VR+y] != 0:
                    r = self.Field[self.VR+x, self.VR+y]/self.SPAWN_MAX*0.45
                    o = Circle(radius=r)
                    if self.Field[self.VR+x, self.VR+y] > 0:
                        o.color(0.1, 1.0, 0.1)
                    else:
                        o.color(1.0, 0.1, 0.1)
                    self.Frame.add(o, at=(x,y))
        h = Circle(radius=0.5, filled=False).color(1.0, 1.0, 0.0)
        self.Frame.add(h, at=self.Hunter)
        s = Rectangle(-self.VR-0.5, self.VR+0.5, -self.VR-0.5, self.VR+0.5, filled=False).color(0.2,0.4,0.4)
        self.Frame.add(s, at=self.Hunter)
        time.sleep(0.03)
        self.Viewer.render()
        #print("scan: position:", self.Hunter, "  out:", self.scan(*self.Hunter))
        #time.sleep(10)

class HunterEnv(TimeLimit):
    
    def __init__(self):
        TimeLimit.__init__(self, _HunterEnv(), time_limit=500)
        
    
        
        