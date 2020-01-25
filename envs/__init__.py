import gym

def make_env(name):
    
    gym_aliases = {
        "lander":   "LunarLander-v2",
        "cartpole": "CartPole-v1"
    }

    if name == "hunter":
        from .hunter import HunterEnv
        return HunterEnv()
    
    elif name == "elevator":
        from .elevator import Elevator
        return Elevator()
    
    name = gym_aliases.get(name, name)
    return gym.make(name)

