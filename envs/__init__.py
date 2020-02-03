import gym

def make_env(name, *params):
    
    gym_aliases = {
        "lander":   "LunarLander-v2",
        "cartpole": "CartPole-v1"
    }

    if name == "hunter":
        from .hunter import HunterEnv
        return HunterEnv(*params)
    
    elif name == "elevator":
        from .elevator import Elevator
        return Elevator(*params)
    
    elif name == "tanks":
        from .tanks import Tanks
        return Tanks(*params)
    
    name = gym_aliases.get(name, name)
    return gym.make(name, *params)
