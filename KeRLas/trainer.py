from .memory import ReplayMemory
from .player import MixedPlayer

class Trainer(object):
    
    def __init__(self, env, brain, random_mix, memory_size, callbacks = None):
        player = MixedPlayer(env, brain, random_mix)
        self.Memory = ReplayMemory(player, memory_size)
        self.Brain = brain
        self.Callbacks = callbacks
        
    def train(self, mbsize, steps_per_epoch, epochs):
        self.Memory.fill(mbsize)
        for epoch in xrange(epochs):
            steps_this_epoch = 0
            while steps_this_epoch < steps_per_epoch:
                samples = self.Memory.sample(mbsize)            # list of (s0,a,s1,r,f) tuples
                info = self.Brain.train_on_samples(samples)
                steps_this_epoch += len(samples)
                if self.Callbacks is not None:
                    self.Callbacks.onTrainingBatch(self, epoch, steps_this_epoch, len(samples), info)
            if self.Callbacks is not None:
                self.Callbacks.onTrainingEpoch(self, epoch, steps_this_epoch, len(sample))
            
            
                