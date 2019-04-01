from .memory import ReplayMemory
from .player import MixedPlayer

class Trainer(object):
    
    def __init__(self, env, brain, random_mix, memory_size, callbacks = None):
        player = MixedPlayer(env, brain, random_mix)
        self.Memory = ReplayMemory(player, memory_size)
        self.Brain = brain
        self.Callbacks = callbacks
        self.NextPolicyInterval = 1000
        
    def train(self, mbsize, steps_per_epoch, epochs):
        with self.Brain.training(True):
            self.Memory.fill(mbsize)
            metrics = None
            next_policy_change = self.NextPolicyInterval
            for epoch in xrange(epochs):
                steps_this_epoch = 0
                while steps_this_epoch < steps_per_epoch:
                    sample = self.Memory.sample(mbsize)            # list of (s0,a,s1,r,f) tuples
                    info = self.Brain.train_on_sample(sample)
                    n = len(sample)
                    steps_this_epoch += n
                    
                    if self.Callbacks is not None:
                        self.Callbacks.onTrainingBatch(self, epoch, steps_this_epoch, len(sample), info)
                        
                    next_policy_change -= n
                    if next_policy_change < 0:
                        self.Brain.nextTrainingPolicy()
                        next_policy_change = self.NextPolicyInterval
                
                if self.Callbacks is not None:
                    self.Callbacks.onTrainingEpoch(self, epoch, steps_this_epoch)
        return info
            
            
                