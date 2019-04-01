import random
import traceback
import numpy as np

class ReplayMemory(object):
    
    def __init__(self, source, capacity):
        self.Memory = []        # list of tuples ((s,a,s,r,f), tag)
        self.Source = source        
        self.Capacity = capacity
        self.LowWater = int(capacity * 0.9)
        self.HighWater = capacity * 2
        self.RefreshRate = 0.1
        self.Cursor = 0
        self.Age = 0
        self.RollingAge = 0.0
        
    FavorFinals = 0.5

    def fill(self, n):
        
        if len(self.Memory) < max(self.Cursor + n, self.LowWater):
            nflush = int(self.RefreshRate * self.Cursor)
            self.Memory = self.Memory[nflush:]
            
            limit = max(n, self.HighWater)
            
            while len(self.Memory) < limit:
                need = limit - len(self.Memory)
                sample = self.Source.sample(need)
                self.Memory += [(data, self.Age) for data in sample
                            if data[-1] or random.random() < self.FavorFinals
                        ]
                self.Age += 1

            random.shuffle(self.Memory)
            self.Cursor = 0      

    def sample(self, n):
        self.fill(n)
        s = self.Memory[self.Cursor:self.Cursor+n]
        for data, age in s:
            self.RollingAge += 0.01 * (age - self.Age - self.RollingAge)
        self.Cursor += n
        return [data for data, age in s]
        
