from .memory import ReplayMemory
#from .player import MixedPlayer
from .drivers import MixedDriver

class Trainer(object):
    
    def __init__(self, env, brain, random_mix, memory_size, callbacks = None):
        player = MixedDriver(env, brain, random_mix)
        self.Memory = ReplayMemory(player, memory_size)
        self.Brain = brain
        self.Callbacks = callbacks
        
    def train(self, mbsize, steps_per_epoch, epochs):
        with self.Brain.training(True):
            self.Memory.fill(mbsize)
            metrics = None
            for epoch in xrange(epochs):
                steps_this_epoch = 0
                while steps_this_epoch < steps_per_epoch:
                    sample = self.Memory.sample(mbsize)            # list of (s0,a,s1,r,f) tuples
                    info = self.Brain.train_on_sample(sample)
                    steps_this_epoch += len(sample)
                    if self.Callbacks is not None:
                        self.Callbacks.onTrainingBatch(self, epoch, steps_this_epoch, len(sample), info)
                self.Brain.nextTrainingPolicy()
                if self.Callbacks is not None:
                    self.Callbacks.onTrainingEpoch(self, epoch, steps_this_epoch)
        return info
            
            
                