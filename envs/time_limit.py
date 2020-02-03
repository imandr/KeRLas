import gym

class TimeLimit:
    def __init__(self, env, time_limit=None, timeout_reward=None):
        self.Env = env
        self.TimeLimit = time_limit
        self.TimeoutReward = timeout_reward
        self.T = None

    def reset(self, **kwargs):
        self.T = self.TimeLimit
        return self.Env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.Env.step(action)
        if self.T is not None:
            self.T -= 1
            if self.T <= 0:
                #info['TimeLimit.truncated'] = not done
                info["TimeLimit.original_reward"] = reward
                if self.TimeoutReward is not None:
                    reward = self.TimeoutReward
                if isinstance(done, bool):
                    done = True
                else:
                    done[:] = True
        return observation, reward, done, info
        
    def __getattr__(self, name):
        return getattr(self.Env, name)

