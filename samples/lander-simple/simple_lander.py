import sys, math, random
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

class SimpleLander(gym.Env):

    SCREEN_W = 600
    SCREEN_H = 400

    SPACE_W = 20.0
    SCALE = SCREEN_W/SPACE_W
    SPACE_H = float(SCREEN_H)/SCALE
    ZERO_Y = 30
    
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

        self.prev_reward = None
        self.action = None
        self.state = None
        self.crash = False

        high = np.array([np.inf]*6)  # useful range is -1 .. +1, but spikes can be higher
        
        self.observation_space = spaces.Box(-high, high)

        self.action_space = spaces.Discrete(4)
        
        self.Body = None

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def iterate(self, state, dt, fx, fy, rot):
        x, y, vx, vy, phi, omega = state
        c, s = math.cos(phi), math.sin(phi)
        fx, fy = c*fx - s*fy, c*fy + s*fx
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
            y = random.random()*self.SPACE_H*0.9
            vx = (random.random()*2-1)*self.MAX_V
            vy = (random.random()*2-1)*self.MAX_V
            phi = (random.random()*2-1)*math.pi
            omega = (random.random()*2-1)*self.MAX_W
        else:
            x = 0.0
            y = self.SPACE_H*0.5
            vx = (random.random()*2-1)*self.MAX_V/10.0
            vy = (random.random()*2-1)*self.MAX_V/10.0
            phi = (random.random()*2-1)*math.pi/20.0
            omega = (random.random()*2-1)*self.MAX_W/20.0
        self.state = np.array([x,y,vx,vy,phi,omega])
        return self.state
        
    FX = [0.0, 0.0, -0.1, 0.1]
    FY = [0.0, 4.0, 0.0, 0.0]
    ROT = [0.0, 0.0, -0.4, 0.4]
    
    SOFT_LANDING_V = 1.0
    SOFT_LANDING_W = 1.0
    SOFT_LANDING_ANGLE  = 15.0*math.pi/180
    LANDING_TARGET_WIDTH = 1.0
    
    def step(self, action):
        state = self.state
        x0, y0, vx0, vy0, phi0, w0 = state
        self.action = action
        fx = self.FX[action]
        fy = self.FY[action]
        rot = self.ROT[action]
        new_state = self.iterate(state, self.DT, fx, fy, rot)

        x1, y1, vx1, vy1, phi1, w1 = new_state
        
        reward = 0.1 * (
            abs(x0) - abs(x1) +
            y0 - y1 +
            abs(phi0) - abs(phi1) +
            abs(vx0) - abs(vx1) +
            abs(w0) - abs(w1)
        )

        
        
        done = False
        crash = False
        target = False
        
        land = y1 <= 0.0
        out =  y1 > self.SPACE_H or abs(x1) > self.SPACE_W/2
        
        done = land or out
        
        if land:
            crash = (
                vy1 < -self.SOFT_LANDING_V 
                or abs(vx1) > self.SOFT_LANDING_V 
                or abs(w1) > self.SOFT_LANDING_W 
                or abs(phi1) > self.SOFT_LANDING_ANGLE
            )
            target = abs(x1) < self.LANDING_TARGET_WIDTH
            distance = 0.0 if target else \
                abs(abs(x1)-self.LANDING_TARGET_WIDTH)/self.LANDING_TARGET_WIDTH
            if crash:   reward = -10.0
            elif target:    reward = 10.0
            else:       reward = max(1.0-abs(distance), 0.0)
        elif out:
            reward = -10.0
        self.crash = crash
        self.state = new_state
        
        #if land:
        #    print "Crashed" if crash else "Landed", new_state, "   reward:", reward
        
            
        
        return new_state, reward, done, {}
        
    LanderPoly = [
        (-20, 0),
        (-10, 30),
        (10, 30),
        (20, 0)
    ]

    MainEnginePoly = [
        (-3, -1),
        (-5, -20),
        (5,-20),
        (3, -1)
    ]

    LeftEnginePoly = [
        (-15, 25),
        (-25, 21),
        (-25, 29)
    ]
    
    RightEnginePoly = [(-x, y) for x, y in LeftEnginePoly]

    def render(self, mode='human'):
        scale = self.SCALE
        carty = 100 # TOP OF CART
        polewidth = 10.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(self.SCREEN_W, self.SCREEN_H)
            
            body = rendering.FilledPolygon(self.LanderPoly)
            body.set_color(0.5,0.4,0.9)
            
            main_engine_flare = rendering.FilledPolygon(self.MainEnginePoly)
            main_engine_flare.set_color(0.9, 0.9, 0.1)
            
            left_engine_flare = rendering.FilledPolygon(self.LeftEnginePoly)
            left_engine_flare.set_color(0.9, 0.9, 0.1)

            right_engine_flare = rendering.FilledPolygon(self.RightEnginePoly)
            right_engine_flare.set_color(0.9, 0.9, 0.1)
            
            self.Body = body
            self.MainEngine = main_engine_flare
            self.RightEngine = right_engine_flare
            self.LeftEngine = left_engine_flare
            
            sky = rendering.FilledPolygon([
                (0.0, 0.0),
                (self.SCREEN_W, 0.0),
                (self.SCREEN_W, self.SCREEN_H),
                (0.0, self.SCREEN_H)
                
            ])
            sky.set_color(0,0,0)
            self.viewer.add_geom(sky)
            base = rendering.FilledPolygon([
                (0.0, 0.0),
                (self.SCREEN_W, 0.0),
                (self.SCREEN_W, self.ZERO_Y),
                (0.0, self.ZERO_Y)
            ])
            base.set_color(0.9, 0.9, 0.9)
            self.viewer.add_geom(base)

            self.LanderTransform = rendering.Transform()
            for g in (self.Body, self.MainEngine, self.RightEngine, self.LeftEngine):
                g.add_attr(self.LanderTransform)
            #if self.crash:
            #    self.Body.set_color(1.0, 0.4, 0.1)
            self.viewer.add_geom(body)
            
        state = self.state
        if state is None:  return None
        
        x, y, _, _, phi, _ = state
        x0, y0 = self.SCREEN_W/2 + x*self.SCALE, self.ZERO_Y + y*self.SCALE

        self.LanderTransform.set_translation(x0, y0)
        self.LanderTransform.set_rotation(phi)
            
        if self.action == 1:
            self.viewer.add_onetime(self.MainEngine)
        elif self.action == 2:
            self.viewer.add_onetime(self.LeftEngine)
        elif self.action == 3:
            self.viewer.add_onetime(self.RightEngine)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
            
if __name__ == "__main__":
    
    env = SimpleLander()
    env.reset()
    done = False
    while not done:
        a = random.randint(1,3)
        new_state, reward, done, info = env.step(a)
        env.render()
        