from .agent import Agent
from .gym_env import TimedGymEnv

class Player(object):
    
    def __init__(self, env, brain, nagents = None, callback = None):
        self.Env = env
        self.Brain = brain
        self.NAgents = 1            # for now
        self.Callback = callback
    
    def gameSamples(self, size):
        samples = []
        while len(samples) < size:
            samples += self.runEpisode()
        return samples

    def randomSamples(self, size):
        raise NotImplementedError


class MultiPlayer(Player):
    
    def runEpisode(self):
        record = []
        active_agents = [Agent(self.Env, self.Brain) for _ in xrange(self.NAgents)]
        env = self.Env
        observations = env.reset(active_agents)
        
        after_reset = env.Env.steps_beyond_done
        
        
        for agent, observation in zip(active_agents, observations):
            agent.init(observation)
            
        after_init = env.Env.steps_beyond_done

            
        if self.Callback is not None:   self.Callback.onEpisodeBegin(env, active_agents, observations)
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
                step_infos = self.Env.step(agent_actions)
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

            if self.Callback is not None:   self.Callback.onStep(env, env_done, step_infos, feedback)

            active_agents = new_active_agents
            end_of_loop = env.Env.steps_beyond_done
            
        if self.Callback is not None:   self.Callback.onEpisodeEnd(env, record)

        return record
        
    def randomSamples(self, size):
        samples = []
        while len(samples) < size:
            agents = [Agent(self.Env, self.Brain) for _ in xrange(self.NAgents)]
            observations = self.Env.reset(agents, random=True)
            actions = self.Env.randomActions(size)
            self.Env.step(zip(agents, actions))
            env_done, feedback = self.Env.feedback()
            samples += [(s0, a, s1, r, f) for s0, a, (_, s1, r, f, _) in zip(observations, actions, feedback)]
        return samples


class GymPlayer(Player):
    """
        Player for Gym single agent environments
    """

    def runEpisode(self):
        record = []
        agent = Agent(self.Env, self.Brain)
        env = self.Env
        observation = env.reset()
        agent.init(observation)
        if self.Callback is not None:   self.Callback.onEpisodeBegin(env, [agent], [observation])
            
        done = False
        while not done:
            action = agent.action()
            new_observation, reward, done, info = env.step(action)
            agent.step(new_observation, reward, done)
            if self.Callback is not None:   self.Callback.onStep(env, done, [(agent, new_observation, reward, done, info)])
        record = agent.trajectory(clear=True)
        #print "gym episode done:", len(record)
        if self.Callback is not None:   self.Callback.onEpisodeEnd(env, record)

        return record

    def randomSamples(self, size):
        samples = []
        while len(samples) < size:
            o = self.Env.reset(random=True)
            a = self.Env.randomAction()
            o1, r, done, info = self.Env.step(a)
            samples.append((o, a, o1, r, done))
        return samples

        
class MixedDriver(object):
    def __init__(self, env, brain, random_fraction = 0.0, chunk = 100):
        self.RandomFraction = random_fraction
        self.NGeneratedRandom = 0
        self.NGeneratedGame = 0
        self.Env = env
        self.Brain = brain
        self.Player = GymPlayer(env, brain) if isinstance(env, TimedGymEnv) else MultiPlayer(env, brain)       # nagents = 1 for now
        self.ChunkSize = chunk
        
    def chunk(self):
        generate_random = self.RandomFraction > 0.0
        ntotal = self.NGeneratedRandom + self.NGeneratedGame
        if ntotal > 0 and generate_random:
            current_fraction = float(self.NGeneratedRandom)/float(ntotal)
            generate_random = current_fraction < self.RandomFraction
        if generate_random:
            samples = self.Player.randomSamples(self.ChunkSize)
            self.NGeneratedRandom += len(samples)
        else:
            samples = self.Player.gameSamples(self.ChunkSize)
            self.NGeneratedGame += len(samples)
        return samples
        
    def samples(self, size):
        s = []
        while len(s) < size:
            s += self.chunk()
        return s
        
        
    
    