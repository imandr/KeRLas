from lunar_lander_simple import LunarLander, FPS
from lunar_lander import LunarLander, FPS
import random, time
import numpy as np

np.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    
    env = LunarLander()
    dt = 1.0/FPS
    
    obs = env.reset()
    env.render()
    done = False
    t = 0
    t0 = time.time()
    while not done and t < 500:
        a = random.randint(0,3)
        s1, r, done, info = env.step(a)
        print (s1, r, info)
        #time.sleep(dt*10)
        env.render()
        t += 1
    print("rate:", t/(time.time()-t0))
