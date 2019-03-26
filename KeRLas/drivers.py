from .agent import Agent

class Driver(object):
    
    def __init__(self, env, brain, nagents):
        self.Env = env
        self.Brain = brain
        self.NAgents = nagents
    
    def sample(self, n):
        raise NotImplementedError

class RandomDriver(Driver):
    
    def __init__(self, env, brain, nagents = None):
        Driver.__init__(self, env, brain, nagents)
    
    def samples(self, size):
        agents = [Agent(self.Env, self.Brain) for _ in xrange(size)]
        observations = self.Env.reset(agents, random=True)
        actions = self.Env.randomActions(size)
        self.Env.step(zip(agents, actions))
        env_done, feedback = self.Env.feedback()
        return [(s0, a, s1, r, f) for s0, a, (_, s1, r, f, _) in zip(observations, actions, feedback)]
        
class GameDriver(Driver):
    
    def __init__(self, env, brain, nagents = None):
        Driver.__init__(self, env, brain, nagents)
    
    def samples(self, size):
        samples = []
        while len(samples) < size:
            active_agents = [Agent(self.Env, self.Brain) for _ in xrange(self.NAgents or self.Env.NAgents)]
            observations = self.Env.reset(active_agents)
            for agent, observation in zip(active_agents, observations):
                agent.init(observation)
            env_done = False
            while not env_done:
                agent_actions = [(a, a.action()) for a in active_agents]
                infos = self.Env.step(agent_actions)
                env_done, feedback = self.Env.feedback()
                active_agents = []
                for agent, new_observation, reward, done, info in feedback:
                    agent.step(new_observation, reward, done)
                    if done:
                        samples += agent.trajectory(clear=True)
                        agent.end()
                    else:
                        active_agents.append(agent)
        return samples
        
class MixedDriver(Driver):
    def __init__(self, env, brain, random_fraction = 0.0, chunk = 100):
        self.RandomFraction = random_fraction
        self.NGeneratedRandom = 0
        self.NGeneratedGame = 0
        self.Env = env
        self.Brain = brain
        self.GameDriver = GameDriver(env, brain, nagents = 1)       # nagents = 1 for now
        self.RandomDriver = RandomDriver(env, brain)
        self.ChunkSize = chunk
        
    def chunk(self):
        generate_random = self.RandomFraction > 0.0
        ntotal = self.NGeneratedRandom + self.NGeneratedGame
        if ntotal > 0 and generate_random:
            current_fraction = float(self.NGeneratedRandom)/float(ntotal)
            generate_random = current_fraction < self.RandomFraction
        if generate_random:
            samples = self.RandomDriver.samples(self.ChunkSize)
            self.NGeneratedRandom += len(samples)
        else:
            samples = self.GameDriver.samples(self.ChunkSize)
            self.NGeneratedGame += len(samples)
        return samples
        
    def samples(self, size):
        s = []
        while len(s) < size:
            s += self.chunk()
        return s
        
        
    
    