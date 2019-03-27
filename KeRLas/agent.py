class Agent:
    
    def __init__(self, env, brain):
        self.Env = env
        self.Brain = brain
        self.Done = False
        self.Observation = None
        self.Action = None
        self.QVector = None
        self.Trajectory = []
        
        self._State = None
        
    def init(self, observation):
        self.Trajectory = []
        self.Observation = observation
        return self.Brain.episodeBegin()
    
    def action(self):
        #print "Agent.action: observation=", self.Observation
        a, qvector = self.Brain.action(self.Observation)
        self.Action = a
        self.QVector = qvector
        return a
                
    def step(self, new_observation, reward, done):
        #if len(self.Trajectory) < 100:
        #    print "Agent.step: ", self.Observation, self.Action, new_observation, reward, done
        self.Trajectory.append((self.Observation, self.Action, new_observation, reward, done))
        self.Observation = new_observation

    def end(self):
        info = self.Brain.episodeEnd()
        return self.Trajectory, info

    def info(self):
        return {
            "qvector":          self.QVector
        }

    def trajectory(self, clear=False):
        t = self.Trajectory
        if clear:
            self.Trajectory = []
        return t
            
