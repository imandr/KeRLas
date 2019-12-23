from .replay_buffer import ReplayBuffer

class MemoryFeeder(object):
    
    def __init__(self, brain, player):
        self.Brain = brain
        self.Player = player
        
    def dataChunk(self):
        trajectory, info = self.Player.runEpisode()
        data = self.Brain.formatTrajectory(trajectory)
        return data
        
class Trainer(object):
    
    def __init__(self, env, brain, memory_size, player, callbacks = None):
        self.Memory = ReplayBuffer(memory_size, MemoryFeeder(brain, player))
        self.Brain = brain
        self.Player = player
        self.Callbacks = callbacks
        self.NextPolicyInterval = 1000
        
    def train(self, mbsize, nsteps):
        with self.Brain.training(True):
            metrics = None
            steps_done = 0
            next_policy_change = self.NextPolicyInterval
            while steps_done < nsteps:
                sample = self.Memory.sample(mbsize)            # list of (s0,a,s1,r,f,...) tuples
                info = self.Brain.train_on_sample(sample)
                n = len(sample)
                
                if self.Callbacks is not None:
                    self.Callbacks.onTrainingBatch(self, steps_done, len(sample), info)
                    
                next_policy_change -= n
                if next_policy_change < 0:
                    self.Brain.nextTrainingPolicy()
                    next_policy_change = self.NextPolicyInterval
                    
                steps_done += n
            
            if self.Callbacks is not None:
                self.Callbacks.onTrainingEpoch(self, epoch, steps_this_epoch)
        return info
            
            
                