import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K


def BellmanDifferential(q0, mask, q1, final, gamma):
    def differential(args):
        q0, mask, q1, final = args
        q0 = K.sum(q0*mask, axis=-1)[:,None]
        q1max = K.max(q1, axis=-1)[:,None]
        reward = q0 - (1.0-final) * gamma * q1max
        return reward
    return Lambda(differential)([q0, mask, q1, final])


class RLModel(object):
    
    def __init__(self, qmodel, gamma, *params, **args):
        self.QModel = qmodel
        qmodel._make_predict_function()
        self.TModel = self.create_trainig_model(qmodel, gamma, *params, **args)
        
    def training_model(self):
        return self.TModel
        
    def fit_generator(self, generator, *params, **args):
        return self.TModel.fit_generator(
                (self.training_data(*data) for data in generator), 
                *params, **args
        )
        
    def predict_on_batch(self, *params, **args):
        return self.QModel.predict_on_batch(*params, **args)
        
    def __call__(self, *params, **kv):
        return self.QModel(*params, **kv)
        
    def training_data(self, *params):
        raise NotImplementedError
        
    def create_trainig_model(self, qmodel, gamma, *params, **args):
        raise NotImplementedError
