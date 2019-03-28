from .agent import Agent

class Driver(object):
    
    def __init__(self, env, brain, nagents):
        self.Env = env
        self.Brain = brain
        self.NAgents = nagents or env.NAgents
    
    def sample(self, n):
        raise NotImplementedError

class RandomDriver(Driver):
    
    def __init__(self, env, brain, nagents = None):
        Driver.__init__(self, env, brain, nagents)
    
    def samples(self, size):
        samples = []
        while len(samples) < size:
            agents = [Agent(self.Env, self.Brain) for _ in xrange(self.NAgents)]
            observations = self.Env.reset(agents, random=True)
            actions = self.Env.randomActions(size)
            self.Env.step(zip(agents, actions))
            env_done, feedback = self.Env.feedback()
            samples += [(s0, a, s1, r, f) for s0, a, (_, s1, r, f, _) in zip(observations, actions, feedback)]
        return samples
        
class Player(object):
    
    def __init__(self, env, brain, nagents = None, callback=None):
        self.NAgents = nagents or env.NAgents
        self.Brain = brain
        self.Env = env
        self.Callback = callback
        
    def runEpisode(self):
        record = []
        active_agents = [Agent(self.Env, self.Brain) for _ in xrange(self.NAgents)]
        env = self.Env
        observations = env.reset(active_agents)
        
        after_reset = env.Env.steps_beyond_done
        
        
        for agent, observation in zip(active_agents, observations):
            agent.init(observation)
            
        after_init = env.Env.steps_beyond_done

            
        #if self.Callback is not None:   self.Callback.onEpisodeBegin(env, active_agents, observations)
        after_callback = env.Env.steps_beyond_done
        env_done = False
        t = 0
        feedback = None
        after_last_step = -1
        after_feedback = -1
        end_of_loop = -1
        before_loop = env.Env.steps_beyond_done
        while not env_done and len(active_agents):
            begin_of_loop = env.Env.steps_beyond_done
            agent_actions = [(a, a.action()) for a in active_agents]
            before_step = env.Env.steps_beyond_done
            try:    
                infos = self.Env.step(agent_actions)
                after_last_step = env.Env.steps_beyond_done
            except Exception as e:
                print "Exception:", e
                print "t:", t
                print "agent_actions:", agent_actions
                print "active_agents:", active_agents
                for a in active_agents:
                    print "done:", a.Done
                print "last step feedback:", feedback
                print "env.steps_beyond_done:", after_reset, after_init, after_callback, before_loop, begin_of_loop, before_step, env.Env.steps_beyond_done, after_last_step, after_feedback, end_of_loop
                raise e
            env_done, feedback = self.Env.feedback()
            after_feedback = env.Env.steps_beyond_done
            t += 1
            new_active_agents = []
            for agent, new_observation, reward, done, info in feedback:
                agent.step(new_observation, reward, done)
                if done:
                    record += agent.trajectory(clear=True)
                    agent.end()
                else:
                    new_active_agents.append(agent)

            if self.Callback is not None:   self.Callback.onStep(env, env_done, active_agents, infos, feedback)

            active_agents = new_active_agents
            end_of_loop = env.Env.steps_beyond_done
            
        if self.Callback is not None:   self.Callback.onEpisodeEnd(env, record)

        return record
        
        
class GameDriver(Driver):
    
    def __init__(self, env, brain, nagents = None):
        Driver.__init__(self, env, brain, nagents)
        self.Player = Player(env, brain, nagents = nagents)
    
    def samples(self, size):
        samples = []
        while len(samples) < size:
            samples += self.Player.runEpisode()
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
        
        
    
    