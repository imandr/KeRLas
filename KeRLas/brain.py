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
    
    dense1 = Dense(inp_width*10, activation="softplus")(inp)
    dense2 = Dense(out_width*10, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="linear")(dense2)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model
    
class Brain(object):
    
    def __init__(self, env, rlmodel, policy, memory_size, random_mix, *params, **args):
        self.RLModel = rlmodel
        self.QModel = rlmodel.QModel
        self.TModel = rlmodel.TModel
        self.Policy = policy
        source = MixedDriver(env, self, random_mix)
        self.Memory = ReplayMemory(source, memory_size)
        
    def create_rl_model(self, qmodel, model_type, gamma, *params, **args):
        if model_type == "ddiff":
            return DirectDiffModel(qmodel, gamma, *params, **args)
        
    def q(self, observation):
        return self.QModel.predict_on_batch(np.array([observation]))[0]
        
    def action(self, observation):
        q = self.q(observation)
        a = self.Policy(q)
        return a, q

    def training_model(self):
        return self.TModel
        
    def trainig_data_generator(self, mbsize):
        print "Brain.trainig_data_generator"
        for data in self.Memory.generate_samples(mbsize):
            yield self.RLModel.training_data(*data)
        #return (
        #    self.RLModel.training_data(*data) for data in self.Memory.generate_samples(mbsize)
        #)
        
    def episodeBegin(self):
        pass
        
    def episodeEnd(self):
        return {}
        
        
