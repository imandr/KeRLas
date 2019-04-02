class Agent:
    
    def __init__(self, env, brain):
        self.Env = env
        self.Brain = brain
        self.Done = False
        self.Observation = None
        self.Action = None
        self.QVector = None
        self.Trajectory = []
        self.QVectors = []
        
        self._State = None
        
    def init(self, observation):
        self.Trajectory = []
        self.QVectors = []
        self.Observation = observation
        return {}
    
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
        self.QVectors.append(self.QVector)
        self.Observation = new_observation

    def end(self):
        return self.Trajectory, {"qvectors":self.QVectors}

    def info(self):
        return {
            "qvector":          self.QVector
        }

            
