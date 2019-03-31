import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam

from .drivers import MixedDriver
from .memory import ReplayMemory

from models import DirectDiffModel

def defaultQModel(inp_width, out_width):
    
    inp = Input((inp_width,), name="qmodel_input")
    
    dense1 = Dense(inp_width*20, activation="tanh")(inp)
    dense2 = Dense(out_width*20, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
class Brain(object):
    
    def __init__(self, rlmodel, policy):
        self.RLModel = rlmodel
        self.Policy = policy
        
    def q(self, observation):
        return self.RLModel.predict_on_batch([np.array([observation])])[0]
        
    def train_on_samples(self, samples):
        return self.RLModel.train_on_samples(samples)
        
    def action(self, observation):
        q = self.q(observation)
        a = self.Policy(q)
        return a, q

    def episodeBegin(self):
        pass
        
    def episodeEnd(self):
        return {}
        
        
