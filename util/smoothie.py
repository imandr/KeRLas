import numpy as np

class Smoothie(object):
    def __init__(self, alpha = 0.1):
        self.Low = self.High = None
        self.Alpha = alpha
        self.Record = []
        
    def update_window_(self, x):
        
        if self.Low is None:
            self.Low = self.High = x
            return
            
        if x < self.Low:
            delta = x - self.Low
            self.Low = x
            self.High += (x-self.High)*self.Alpha*2
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += (x-self.Low)*self.Alpha*2
        else:
            self.Low += self.Alpha*(x - self.Low)
            self.High += self.Alpha*(x - self.High)
        
    def update_window(self, x):
        
        if self.Low is None:
            self.Low = self.High = x
            return
            
        if x < self.Low:
            delta = x - self.Low
            self.Low = x
            self.High += delta*0.5
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += delta*0.5
        else:
            self.Low += self.Alpha*(x - self.Low)
            self.High += self.Alpha*(x - self.High)
        
        
    def update(self, x):
        if self.Low is None:
            self.Low = self.High = x
        if x < self.Low:
            self.Low = x
            self.High += (x-self.High)*self.Alpha*2
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += (x-self.Low)*self.Alpha*2
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

class Smoother(object):
    def __init__(self, alpha = 0.1):
        self.Low = self.High = None
        self.Alpha = alpha
        self.Record = []
        self.LowRecord = []
        self.HighRecord = []
        self.TRecord = []
        
    def update(self, t, x):
        if self.Low is None:
            self.Low = self.High = x
        if x < self.Low:
            self.Low = x
            self.High += (x-self.High)*self.Alpha*2
        elif x > self.High:
            delta = x - self.High
            self.High = x
            self.Low += (x-self.Low)*self.Alpha*2
        else:
            self.Low += self.Alpha*(x - self.Low)
            self.High += self.Alpha*(x - self.High)
        self.LowRecord.append(self.Low)
        self.HighRecord.append(self.Low)
        self.TRecord.append(t)
        self.Record.append(x)
        ma10 = np.mean(self.Record[-10:])
        ma100 = np.mean(self.Record[-100:])
        ma = (ma10+10*ma100)/11.0
        self.MARecord.append(ma)
        
        if len(self.Record) < 3:
            return None
            
        med3 = np.median(self.Record[-3:])
        self.MedRecord.append(med3)
        
        return self.TRecord[-2], self.LowRecord[-2], self.MARecord[-2], self.HighRecord[-2],self.MedRecord[-1]
        
    __call__ = update

