from model import model
import numpy as np
import random
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


time_window = 100

def generate(length):
    a = random.random()
    da = random.random() * 0.2 - 0.1
    s = random.random()*0.1 + 0.5
    
    y_ = np.empty((length, 2))
    y_[:,0] = a + np.arange(length)*da
    y_[:,1] = s
    
    x = np.random.normal(0.0, s, (length,1)) + y_[:,0].reshape((-1,1))
    return x, y_

def train(model, time_window, num_episodes, mb_size):
    samples = [generate(time_window) for _ in range(num_episodes)]
    x, y_ = zip(*samples)
    x = np.array(x)
    y_ = np.array(y_)
    return model.fit(x, y_, batch_size=30)
    
m = model()
for _ in range(100):
    print(train(m, time_window, 1000, 30))
        
    
    