import numpy as np

class Env:
    NAgents = 1
    Delta = 0.1
    NActions = 2
    StateDim = 2

    Moves = np.array(
        [
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, -1.0)
        ]
    ) * Delta
    
    def __init__(self, tlimit=100):
        self.TLimit = self.T = tlimit

    def randomStates(self, n):
        return np.random.random((n,2))
    
    def randomActions(self, n):
        return np.random.randint(0, self.NActions, n)
        
    def reset(self, agents, random=True):
        self.T = self.TLimit
        states = self.randomStates(len(agents))
        for a, s in zip(agents, states):
            a._State = s
        self.AllDone = False
        return states
        
    def step(self, agents_actions):
        n = len(agents_actions)
        feedback = []
        all_done = True
        for i, (agent, action) in enumerate(agents_actions):
            if not agent.Done:
                s0 = agent._State
                s1 = s0 + self.Moves[action]
                x, y = s1
                done = x > 1.0 or y > 1.0
                reward = 0.0
                if done:
                    z = x if y > 1.0 else y
                    reward = 1-2*z
                    agent.Done = True
                else:
                    all_done = False
                agent._State = s1
                feedback.append((agent, s1, reward, done, {}))
        self.AllDone = all_done
        self.Feedback = feedback
        
                
    def feedback(self):
        self.T -= 1
        return (self.T <= 0 or self.AllDone), self.Feedback
    
        
        
    