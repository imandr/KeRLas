import sys, math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v0
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER  = 20.0   # ivm, was 13
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14,+17), (-17,0), (-17,-10),
    (+17,-10), (+17,0), (+14,+17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY   = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

SPACE_W = 20.0
SPACE_H = float(VIEWPORT_H)/float(VIEWPORT_W)*SPACE_W

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class SimpleLander(gym.Env):

    VIEWPORT_W = 600
    VIEWPORT_H = 400

    SPACE_W = 20.0
    SPACE_H = float(VIEWPORT_H)/float(VIEWPORT_W)*SPACE_W
    
    MAX_V = 10.0
    MAX_W = math.pi     # rad/sec
    
    G = -1.0            # gravity
    MASS = 1.0
    SIZE = 1.0            

    FPS    = 50
    DT = 1.0/FPS
        


    def __init__(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
        
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        
    def iterate(self, state, dt, fx, fy, rot):
        x, y, vx, vy, phi, omega = state
        c, s = math.cos(phi), math.sin(phi)
        fx = c*fx + s*fy
        fy = c*fy - s*fx
        x += vx * dt
        y += vy * dt
        vy += (self.G + fy/self.MASS) * dt
        vx += fx/self.MASS * dt
        phi += omega * dt
        omega += (rot/self.MASS/self.SIZE**2) * dt
        return np.array([x, y, vx, vy, phi, omega])

    def reset(self, wide = False):
        if wide:
            x = (random.random()*2-1)*self.SPACE_W/2
            y = random.random()*self.SPACE_H
            vx = (random.random()*2-1)*self.MAX_V
            vy = (random.random()*2-1)*self.MAX_V
            phi = (random.random()*2-1)*math.pi
            omega = (random.random()*2-1)*self.MAX_W
        else:
            x = 0.0
            y = self.SPACE_H*0.9
            vx = (random.random()*2-1)*self.MAX_V/100.0
            vy = (random.random()*2-1)*self.MAX_V/100.0
            phi = (random.random()*2-1)*math.pi/100.0
            omega = (random.random()*2-1)*self.MAX_W/100.0
        return np.array([x,y,vx,vy,phi,omega])

    FX = [0.0, 0.0, -0.1, 0.1]
    FY = [0.0, 1.0, 0.0, 0.0]
    ROT = [0.0, 0.0, -0.1, 0.1]
    
    SOFT_LANDING_V = 0.01
    SOFT_LANDING_W = 0.01
    LANDING_TARGET_WIDTH = 1.0
    SOFT_LANDING_ANGLE  = 5.0*math.pi/180
    
    def step(self, action):
        state = self.state
        fx = self.FX[action]
        fy = self.FY[action]
        rot = self.ROT[action]
        new_state = self.iterate(state, self.DT, fx, fy, rot)

        x, y, vx, vy, phi, w = new_state
        
        done = False
        crash = False
        target = False
        if y <= 0.0:
            done = True
            crash = vy < -self.SOFT_LANDING_V \
                or abs(vx) > self.SOFT_LANDING_V \
                or abs(w) > self.SOFT_LANDING_W \
                or abs(phi) > self.SOFT_LANDING_ANGLE
            target = abs(x) < self.LANDING_TARGET_WIDTH
            distance = 0.0 if target else:
                abs(abs(x)-self.LANDING_TARGET_WIDTH)/self.LANDING_TARGET_WIDTH
        
        reward = 0.0
        if done:
            if crash:   reward = -10.0
            elif target:    reward = 10.0
            else:       reward = max(1.0-abs(distance), 0.0)
            
        self.state = new_state
        return new_state, reward, done, {}

