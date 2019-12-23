import random
import traceback
import numpy as np

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.Memory = []        # list of tuples (data, age)
        self.Buffer = []
        self.Capacity = capacity
        self.LowWater = int(capacity*0.75)
        self.HighWater = self.Capacity
        self.Age = 0
        self.RollingAge = 0.0
        
    def addSample(self, sample):
        self.Buffer += sample
        
    def addToMemory(self, sample):
        if len(self.Memory) + len(sample) > self.HighWater:
            self.Memory = random.sample(sample + self.Memory, self.LowWater)
        else:
            self.Memory += sample
        
    def sample(self, n):
        nn = n
        from_buffer = self.Buffer[:nn]
        n_from_buffer = len(from_buffer)
        out = from_buffer
        if n_from_buffer < n:
            nn = n - n_from_buffer
            from_mem = random.sample(self.Memory, nn)
            out = from_buffer + from_mem
        if n_from_buffer:
            self.addToMemory(from_buffer)
            self.Buffer = self.Buffer[n_from_buffer:]
        return out
        
