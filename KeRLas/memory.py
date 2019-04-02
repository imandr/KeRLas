import random
import traceback
import numpy as np

class ReplayMemory(object):
    
    def __init__(self, source, capacity):
        self.Memory = []        # list of tuples ((s,a,s,r,f), tag)
        self.Source = source        
        self.Capacity = capacity
        self.LowWater = int(capacity*0.75)
        self.HighWater = self.Capacity
        self.Cursor = 0
        self.Age = 0
        self.RollingAge = 0.0
        self.Buffered = 0
        
    FavorFinals = 0.5
    
    def addSample(self, sample):
        for data, tag in sample:
            tag["age"] = self.Age
            self.Memory.append((data, tag))
        self.Age += 1
        if len(self.Memory) > self.HighWater:
            self.Memory = self.Memory[-self.LowWater:]
            random.shuffle(self.Memory)
            self.Cursor = 0
            
    def fill(self, n):
        assert n < self.LowWater
        while self.Buffered < n:
            in_sample = self.Source.sample(n - self.Buffered)
            self.addSample(in_sample)
            self.Buffered += len(in_sample)
        assert self.Cursor + n <= len(self.Memory)

    def sample(self, n):
        self.fill(n)
        s = self.Memory[self.Cursor:self.Cursor+n]
        self.Buffered -= n
        self.Cursor += n
        tau_hist = {}
        nfinal = 0
        for data, tag in s:
            tau = tag.get("tau", -1)
            n = tau_hist.setdefault(tau, 0)
            tau_hist[tau] = n + 1
            age = tag["age"]
            self.RollingAge += 0.01 * (age - self.Age - self.RollingAge)
            if data[-1]:
                nfinal += 1
        #print "sample: %d of %d" % (nfinal, len(s))
        #if random.random() < 0.5:
        #    print "tau_hist:", tau_hist
        #    print [tag.get("tau", -1) for data, tag in s]
        return [data for data, tag in s]
        