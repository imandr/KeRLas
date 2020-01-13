import numpy as np

class Smoothie(object):
    def __init__(self, alpha = 0.1):
        self.Low = self.High = None
        self.Alpha = alpha
        self.Record = []
        
    def update(self, x):
        if self.Low is None:
            self.Low = self.High = x
        if x < self.Low:
            delta = x - self.Low
            self.Low = x
            self.High += (1.0-self.Alpha)*delta
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += (1.0-self.Alpha)*delta
        else:
            self.Low += self.Alpha*(x - self.Low)
            self.High += self.Alpha*(x - self.High)

        self.Record.append(x)
        
        ma10 = np.mean(self.Record[-10:])
        ma100 = np.mean(self.Record[-100:])
        self.MovingAverage = (ma10+10*ma100)/11.0
        
        """
        if len(self.Record) < self.Preload:
            self.MovingAverage = sum(self.Record)/len(self.Record)
        else:
            if self.MovingAverage is None:  self.MovingAverage = x
            self.MovingAverage += self.Alpha*(x-self.MovingAverage)
        """
        return self.Low, self.MovingAverage, self.High
        
    __lshift__ = update

    def get(self):
        return self.Low, self.MovingAverage, self.High
        
    __call__ = get

