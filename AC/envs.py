import gym

def make_env(name):
    
    gym_aliases = {
        "lander":   "LunarLander-v2",
        "cartpole": "CartPole-v1"
    }

    if name == "hunter":
        from hunter import HunterEnv
        return HunterEnv()
    
    name = gym_aliases.get(name, name)
    return gym.make(name)

