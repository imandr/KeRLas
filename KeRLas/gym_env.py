import gym, random, numpy as np

class GymEnv:
    #
    # Convert 1-agent Gym environment into a multi-agent environment
    #
    
    NAgents = 1
    
    def __init__(self, env, tlimit=None, random_observation_space=None):
        if isinstance(env, str):
            env = gym.make(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.TLimit = tlimit
        self.Env = env
        self.Obs = None
        self.Reward = None
        self.Done = False
        self.Info = None
        self.RandomObservationSpace = random_observation_space
        
    def __str__(self):
        return "GymEnv(%s)" % (self.Env,)
        
    def randomStates(self, n):
        space = self.RandomObservationSpace or self.Env.observation_space
        return np.array([space.sample() for _ in xrange(n)])
    
    def randomActions(self, n):
        return np.array([self.Env.action_space.sample() for _ in xrange(n)])
        
    def reset(self, agents, random = False):
        assert len(agents) == 1, "Converted Gym environments can not handle multiple agents"
        self.Agent = agents[0]
        obs = self.Env.reset()
        if random and self.RandomObservationSpace is not None:
            obs = self.RandomObservationSpace.sample()
            self.Env.state = obs
        self.Obs = obs
        self.Done = False
        self.T = self.TLimit
        return [self.Obs]

    def addAgent(self, agent, random = False):
        raise NotImplementedError
        
    def step(self, actions):
        assert len(actions) == 1, "Converted Gym environments can not handle multiple agents"
        self.Obs, self.Reward, self.Done, self.Info = self.Env.step(actions[0][1])
        if self.T is not None:
            self.T -= 1
            self.Done = self.Done or (self.T <= 0)
        
    def feedback(self):
        return False, [(self.Agent, self.Obs, self.Reward, self.Done, self.Info)]
        
    def __getattr__(self, name):
        return getattr(self.Env, name)
        
    