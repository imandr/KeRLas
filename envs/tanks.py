import random
import numpy as np
import math, time
from gym import spaces
from .time_limit import TimeLimit


class _TanksEnv(object):

    FireRange = 0.4
    Speed = 0.05
    RotSpeed = math.pi*2/50
    Width = 0.03
    
    X0 = 0.0
    X1 = 1.0
    Y0 = 0.0
    Y1 = 1.0
    
    IDLE = 0
    FIRE = 1
    FWD = 2
    #BACK = 3
    LEFT = 3
    RIGHT = 4
    NumActions = 5

    
    def __init__(self, ntanks):
        self.Viewer = None
        self.NPLAYERS = ntanks
        self.ObservationShape = (self.NPLAYERS, self.NPLAYERS*3)
        self.ActionShape = (self.NPLAYERS,)
        
    def reset(self):
        self.Pos = np.random.random((self.NPLAYERS, 2))
        self.Angle = np.random.random((self.NPLAYERS,))*2*math.pi
        self.Hit = np.zeros((self.NPLAYERS,))       # for rendering only
        self.Fire = np.zeros((self.NPLAYERS,))
        return self.observation()
        
    def round_angle(self, a):
        while a >= math.pi:   a -= math.pi*2
        while a < -math.pi:   a += math.pi*2
        return a
        
    def dist(self, i, j):
        return math.sqrt(np.sum(np.square(self.Pos[j]-self.Pos[i])))
        
    def bearing(self, i, j):
        d = self.Pos[j]-self.Pos[i]
        return math.atan2(d[1], d[0])
        
    def observation(self):
        obs = np.empty((self.NPLAYERS,self.NPLAYERS,3))
        for i in range(self.NPLAYERS):
            obs[i,0,:2] = self.Pos[i]
            obs[i,0,2] = self.Angle[i]
            others = []
            for j in range(self.NPLAYERS):
                if j != i:
                    others.append((self.dist(i,j), self.bearing(i, j), self.Angle[j]))
            others = sorted(others)
            obs[i,1:,:] = others
        return obs.reshape((self.NPLAYERS, -1))
        
    def step(self, actions):
        
        rewards = np.zeros((self.NPLAYERS,))
        dones = np.zeros((self.NPLAYERS,))
        self.Hit[:] = 0
        self.Fire[:] = 0
        
        #
        # move first, fire second
        #
        for i, a in enumerate(actions):
            if a != self.FIRE:
                pos = self.Pos[i]
                angle = self.Angle[i]
                if a == self.LEFT:
                    angle += self.RotSpeed
                elif a == self.RIGHT:
                    angle -= self.RotSpeed
                elif a in (self.FWD, self.FWD):
                    delta = self.Speed if a == self.FWD else -self.Speed*0.6
                    x0, y0 = pos
                    dx, dy = delta*math.cos(angle), delta*math.sin(angle)
                    x1, y1 = x0+dx, y0+dy
                    if x1 > self.X1 or x1 < self.X0 or y1 > self.Y1 or y1 < self.Y0:
                        x1, y1 = x0, y0
                        rewards[i] -= 1.0
                    pos = (x1, y1)
                self.Pos[i,:] = pos
                self.Angle[i] = self.round_angle(angle)
            
        for i, a in enumerate(actions):
            if a == self.FIRE:
                rewards[i] -= 0.1
                self.Fire[i] = 1
                posi = self.Pos[i]
                for j in range(self.NPLAYERS):
                    if j != i:
                        posj = self.Pos[j]
                        dist = self.dist(i,j)
                        if dist > 0.0 and dist < self.FireRange:
                            b = self.bearing(i, j)
                            da = abs(b-self.Angle[i])
                            if da < math.pi/4:
                                s = math.sin(da)*dist
                                if abs(s) < self.Width:
                                    self.Hit[j] = 1
                                    #print("hit:",i,j)
                                    rewards[j] -= 10.0
                                    rewards[i] = 10.0
        
        return self.observation(), rewards, dones, {}
        
    def init_rendering(self):
        from draw2d import Viewer, Rectangle, Frame, Circle, Line, Polygon
        self.Viewer = Viewer(800,800)
        self.Frame = self.Viewer.frame(self.X0, self.X1, self.Y0, self.Y1)
        
        class Tank(object):
            
            def __init__(self, parent):
                self.Frame = Frame(hidden=True)
                self.Body = Polygon([
                    (-0.01, -0.01), (0.01, -0.005), (0.01, 0.005), (-0.01, 0.01)
                ], filled=True)
                self.Frame.add(self.Body)
                self.FireLine = Line((0.005, 0.0), (_TanksEnv.FireRange, 0.0)).color(1.0,0.8,0.1)
                self.FireLine.hidden = True
                self.Frame.add(self.FireLine)
                
            def show(self, pos, angle, fire, hit):
                self.Frame.hidden = False
                self.Frame.move_to(*pos)
                self.Frame.rotate_to(angle)
                self.FireLine.hidden = fire == 0
                #print (self.FireLine.hidden)
                if hit:
                    self.Body.color(1,0,0)
                else:
                    self.Body.color(0,1,1)
                
        
        self.Tanks = [Tank(self.Frame) for _ in range(self.NPLAYERS)]
        for t in self.Tanks:
            self.Frame.add(t.Frame)
            
    def render(self):
        if self.Viewer is None:
            self.init_rendering()
        
        any_hit = False
        for tank, pos, angle, fire, hit in zip(self.Tanks, self.Pos, self.Angle, self.Fire, self.Hit):
            tank.show(pos, angle, fire, hit)
            if hit: any_hit = True
            
        self.Viewer.render()
        if any_hit: time.sleep(0.1)
        
class Tanks(TimeLimit):
    
    def __init__(self, *params):
        TimeLimit.__init__(self, _TanksEnv(*params), time_limit=500)
        
                        
                    
        
        