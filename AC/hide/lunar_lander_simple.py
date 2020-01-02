import sys, math, random, time
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

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

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
DT      = 1.0/FPS
RAD2DEG = 180.0/math.pi

MAIN_ENGINE_POWER  = 20.0   # ivm, was 13
SIDE_ENGINE_POWER  =  0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W/SCALE
H = VIEWPORT_H/SCALE
V_SCALE = W/2                   # to convert metters to observation relative units so that the box boundaries are at x= +/- 1

GROUND_Y = H/4
HELIPAD_W = W/11
HELIPAD_X = HELIPAD_W/2

#
# gym env params:
#
# action dvx dvy dangle domega
# 0 [ 0.00020729 -0.02566236  0.001715   -0.00040367]
# 1 [-0.01091307 -0.02545288 -0.0040379   0.04226915]
# 2 [0.00108743 0.01240228 0.00188269 0.00055057]
# 3 [ 1.22302026e-02 -2.60260999e-02 -1.13840215e-05 -4.84604426e-02]
#
# g: -0.026
# main engine: dvy = 0.012
# side: dvx: 0.012, domega = 0.045
#

G = -9.8
MAIN_ENGINE_DVDT = +40.0
SIDE_ENGINE_DVDT = 2.2
SIDE_ENGINE_DWDT = 4.5

class LunarLander(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        self.seed()
        self.viewer = None

        self.fired_main = self.fired_left = self.fired_right = False

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
        pass
        
    def __str__(self):
        return "Lander(x=%.2f, y=%.2f, a=%.2f, vx=%.2f, vy=%.2f, w=%.2f)" % (
            self.x, self.y, self.f*RAD2DEG, self.vx, self.vy, self.w*RAD2DEG
        )
        
    __repr__ = __str__

    def reset(self):
        self.game_over = False
        self.prev_shaping = None

        self.y = H*0.5
        self.x = 0.0
        self.vx = self.vy = self.w = 0.0
        #self.vy = self.np_random.uniform(-MAIN_ENGINE_DVDT*DT, MAIN_ENGINE_DVDT*DT)
        #self.vx = self.np_random.uniform(-SIDE_ENGINE_DVDT*DT, SIDE_ENGINE_DVDT*DT)
        #self.w = self.np_random.uniform(-SIDE_ENGINE_DWDT*DT, SIDE_ENGINE_DWDT*DT)
        self.f = 0.0

        self.prev_shaping = self.shaping()
        
        #print("reset:", self)
        #print("W:",W, "  H:",H, "  GROUND_Y:", GROUND_Y)
            
    def shaping(self):
        state = self.vector()
        return \
                - 10*np.sqrt(state[0]**2 + state[1]**2) \
                - 10*np.sqrt(state[2]**2 + state[3]**2) \
                - 10*abs(state[4])
        

    def vector(self):
        return np.array([self.x/V_SCALE, (self.y-GROUND_Y)/V_SCALE, self.vx/V_SCALE*DT, self.vy/V_SCALE*DT, self.f, self.w*DT])
        
    ACCELERATIONS = [  # dvx, dvy, dw
        np.array([0.0, 0.0, 0.0]),
        np.array([-SIDE_ENGINE_DVDT, 0.0, SIDE_ENGINE_DWDT]),
        np.array([0.0, MAIN_ENGINE_DVDT, 0.0]),
        np.array([SIDE_ENGINE_DVDT, 0.0, -SIDE_ENGINE_DWDT])
    ]

    def step(self, action):
        #print("step:", action, self)
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))
        ax = ay = aw = 0.0
        if action != 0:
            ax, ay, aw = self.ACCELERATIONS[action]
            vx_vy_w = np.array([self.vx, self.vy, self.w])
            f1 = (self.f + self.f * DT*self.w)*0.5
            c, s = math.cos(f1), math.sin(f1)
            ax, ay = ax*c - ay*s, ay*c + ax*s

        vx1 = self.vx + ax*DT
        vy1 = self.vy + (ay+G)*DT
        w1 = self.w + aw*DT
        
        self.x += (self.vx+vx1)/2*DT
        self.y += (self.vy+vy1)/2*DT
        self.f += (self.w+w1)/2*DT
        while self.f > math.pi:
            self.f -= math.pi*2
        while self.f < -math.pi:
            self.f += math.pi*2
        self.w = w1
        self.vx = vx1
        self.vy = vy1

        reward = 0.0
        shaping = self.shaping()
        reward = shaping_reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        
        #reward = 0.0

        #reward -= m_power*0.30  # less fuel spent is better, about -30 for heurisic landing
        #reward -= s_power*0.03

        done = False
        land = self.y <= GROUND_Y
        crash = land and (self.vy < -0.1 or self.w > 0.1 or abs(self.f) > 5.0*math.pi/180.0)
        out = abs(self.x) > W/2
        if crash or out:
            done = True
            reward -= 100.0
        elif land:
            done = True
            reward += 100.0
            
        #
        # for rendering
        #
        self.fired_main = action == 2
        self.fired_left = action == 1
        self.fired_right = action == 3
            
        return self.vector(), reward/100.0, done, {"shaping_reward":shaping_reward}

    LANDER_POLY =[
        (x/SCALE, y/SCALE) for x, y in [
            (-14,+27), (-17,10), (-17,-0),
            (+17,0), (+17,10), (+14,+27)
            ]
        ]
        
    LANDER_COLORS = [
        (0.5,0.4,0.9),
        (0.3,0.3,0.5)
    ]
    
    FLARE_LEFT = [
        (x/SCALE, y/SCALE) for x, y in [
            (15, 20), (25, 18), (25, 22)
        ]
    ]

    FLARE_RIGHT = [
        (x/SCALE, y/SCALE) for x, y in [
            (-15, 20), (-25, 18), (-25, 22)
        ]
    ]
    
    MAIN_FLARE = [
        (x/SCALE, y/SCALE) for x, y in [
            (0, -2), (5, -15), (-5, -15)
        ]
    ]
    
    FLARE_COLOR = (0.9, 0.5, 0.1)


    def render(self, mode='human'):
        
        #print(self.x, self.y, self.f, self.vy)
        
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)
            
        t = rendering.Transform(translation=(self.x+VIEWPORT_W/SCALE/2, self.y), rotation=self.f)
        
        self.viewer.draw_polygon(self.LANDER_POLY, color=self.LANDER_COLORS[0]).add_attr(t)
        self.viewer.draw_polyline(self.LANDER_POLY, color=self.LANDER_COLORS[1]).add_attr(t)
        
        if self.fired_main:
            self.viewer.draw_polygon(self.MAIN_FLARE, color=self.FLARE_COLOR).add_attr(t)
        if self.fired_left:
            self.viewer.draw_polygon(self.FLARE_LEFT, color=self.FLARE_COLOR).add_attr(t)
        if self.fired_right:
            self.viewer.draw_polygon(self.FLARE_RIGHT, color=self.FLARE_COLOR).add_attr(t)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

        
    
    