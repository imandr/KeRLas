class Env:
    
    NAgents = 2
    
    def reset(self, agents, random = False):
        self.Active = set()
        return observations
        
    def addAgents(self, agents, random = False):
        return observations
        
    def step(self, actions):
        # called once with actions for all active agents
        for agent, action in actions:
            if not agent.Done:
                agent.apply_action(agent, action)
            self.Active.add(agent)
        return [info]
        
    def feedback(self):
        # called once at the end of the episode
        # for all agents, set agent.Done = done
        feedback = [(agent, new_observation, reward, done, info) for agent in self.Active]
        self.Active = set()
        return stop_episode, feedback
        
    def randomActions(self, size):
        return [random_action]
    
    def randomTransitions(self, size):
        return 


