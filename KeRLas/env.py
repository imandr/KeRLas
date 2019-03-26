class Env:
    
    NAgents = 2
    
    def reset(self, agents, random = False):
        pass
        
    def addAgent(self, agent, random = False):
        pass
        
    def step(self, actions):
        # called once with actions for all active agents
        for agent, action in actions:
            if not agent.Done:
                agent.apply_action(agent, action)
        return [info]
        
    def feedback(self):
        # called once at the end of the step
        # for all agents, set agent.Done = done
        return env_done, [(agent, new_observation, reward, done, info) for agent in active_agents]
        
    def randomMoves(self, size):
        state0, action, state1, reward, done, info = self.Env.randomStep()
        if self.TLimit:
            done = done or random.random() * self.TLimit+1 < 1.0
        return (state0, action, state1, reward, done, info)
    