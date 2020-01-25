import random
import numpy as np
import math, time
from gym import spaces
from .time_limit import TimeLimit
from draw2d import Viewer, Rectangle, Polygon, PolyLine, Frame

def gen_colors(B):
    colors = []
    B2 = B*B
    N = B**3-1
    for i in range(1,N):
        rgb = [i//B2, (i%B2)//B, i%B]
        colors.append(rgb)
    colors = np.array(colors, dtype=np.float)/(B-1)    
    colors = (0.1+colors)/1.1
    return colors    
    
def gen_colors(n, c1, c2, c3):
    c1 = np.array(c1, dtype=np.float)
    c2 = np.array(c2, dtype=np.float)
    c3 = np.array(c3, dtype=np.float)
    
    t = 0.0
    dt = 1.0/(n-1)
    
    colors = []
    
    for _ in range(n):
        if t < 0.5:
            s1 = 1.0 - 2.0*t
            s2 = 2.0*t
            s3 = 0.0
        else:
            s1 = 0.0
            s2 = 2.0-2*t
            s3 = 2.0*t-1.0
        colors.append(c1*s1 + c2*s2 + c3*s3)
        t += dt
    return colors
            
    

class _Elevator(object):
    
    NFloors = 8
    Capacity = 10
    ArrivalRate = 0.2
    
    def __init__(self):
        self.ObservationDim = self.NFloors*4
        high = np.array([np.inf]*self.ObservationDim)
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(3)
        self.Viewer = None
        
    def init_rendering(self):
        self.FloorColors = gen_colors(self.NFloors, (0.1, 0.1, 0.9), (0.1, 0.9, 0.1), (1.0, 1.0, 0.1))
        self.Viewer = Viewer(600, 600)
        self.Frame = self.Viewer.frame(0., 20., 0., self.NFloors)
        elevator_well = Frame()
        self.Frame.add(elevator_well, at=(3.0, 0.0))
        self.ElevatorFrame = Frame()
        self.ElevatorFrame.add(Rectangle(0.0, self.Capacity, 0.0, 1.0, filled=False).color(0.8, 0.8, 0.8),
            at=(0,0))
        elevator_well.add(self.ElevatorFrame, at=(0.0,0))
        self.FloorFrames = []
        self.QueueFrames = []
        for f in range(self.NFloors):
            frame = Frame()
            self.Frame.add(frame, at=(0.0, f))
            self.FloorFrames.append(frame)
            frame.add(Rectangle(0.0, 2.0, 0.0, 1.0).color(*self.FloorColors[f]), at=(0,0))
            qf = Frame()
            frame.add(qf, at=(3.0+self.Capacity, 0.0))
            self.QueueFrames.append(qf)
            
        
    def reset(self):
        self.Floor = 0
        self.Load = np.zeros((self.NFloors,))
        self.Queues = [[] for _ in range(self.NFloors)]
        return self.observation()
        

    def exchange(self):
        #
        # unload
        #
        unloaded = self.Load[self.Floor]
        self.Load[self.Floor] = 0        
        
        q = self.Queues[self.Floor]
        n = max(0, self.Capacity-int(sum(self.Load)))
        n = min(n, len(q))
        #print (type(n), n)
        for i in q[:n]:
            self.Load[i] += 1
        self.Queues[self.Floor] = q[n:]        
        return unloaded*10.0
        
    def observation(self):
        #
        # one-hot floor number
        # load
        # up buttons
        # down buttons
        #
        up_buttons = np.zeros((self.NFloors,))
        down_buttons = np.zeros((self.NFloors,))
        for f, q in enumerate(self.Queues):
            up = down = 0
            for d in q:
                if d > f:   up = 1
                elif d < f: down = 1
            up_buttons[f] = up
            down_buttons[f] = down
        floor = np.zeros((self.NFloors,), dtype=np.float)
        floor[self.Floor] = 1
        v = np.concatenate([floor, self.Load, up_buttons, down_buttons])
        return np.array(v, dtype=np.float)
        
    def step(self, action):
        reward = -0.1*(sum(len(q) for q in self.Queues) + sum(self.Load))
        if action == 0:
            reward += self.exchange()
        elif action == 1:
            if self.Floor > 0:
                self.Floor -= 1
            else:
                reward -= 10.0
            reward -= 0.01
        elif action == 2:
            if self.Floor < self.NFloors-1:
                self.Floor += 1
            else:
                reward -= 10.0
            reward -= 0.01
                
        if self.ArrivalRate > random.random():
            i = random.randint(0, self.NFloors-1)
            if len(self.Queues[i]) < self.Capacity:
                j = random.randint(0, self.NFloors-1)
                while j == i:
                    j = random.randint(0, self.NFloors-1)
                self.Queues[i].append(j)
                
        return self.observation(), reward, False, {}
        
    PersonUp =      [(0.0, 0.1), (0.9, 0.1), (0.45, 0.4)]
    PersonDown =    [(0.0, 0.4), (0.9, 0.4), (0.45, 0.1)]
    PersonSide =    [(0.0, 0.1), (0.0, 0.4), (0.9, 0.25)]
    
    def render(self):
        
        if self.Viewer is None:
            self.init_rendering()
        
        self.ElevatorFrame.remove_all()
        self.ElevatorFrame.move_to(0.0, self.Floor)
        self.ElevatorFrame.add(Rectangle(0.0, self.Capacity, 0.0, 1.0, filled=False).color(0.5, 0.5, 0.5), at=(0,0))
        
        x = 0
        for f, n in enumerate(self.Load):
            n = int(n)
            for _ in range(n):
                p = self.PersonUp if f > self.Floor else (
                        self.PersonDown if f < self.Floor else self.PersonSide
                )
                p = Polygon(p).color(*self.FloorColors[f])
                self.ElevatorFrame.add(p, at=(x, 0))
                x += 1
        
        for f, q in enumerate(self.Queues):
            qf = self.QueueFrames[f]
            qf.remove_all()
            x = 0
            for df in q:
                p = self.PersonUp if df > f else (
                        self.PersonDown if df < f else self.PersonSide
                )
                p = Polygon(p).color(*self.FloorColors[df])
                qf.add(p, at=(x,0))
                x += 1
        self.Viewer.render()
        time.sleep(0.1)

class Elevator(TimeLimit):
    
     def __init__(self, time_limit=100):
         TimeLimit.__init__(self, _Elevator(), time_limit=time_limit)
        
    