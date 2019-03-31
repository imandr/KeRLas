from keras.layers import Lambda
import keras.backend as K
import numpy as np


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

    def train_on_samples(self, samples):
        columns = zip(*samples)
        columns = map(np.array, columns[:5])
        x, y_ = self.training_data(*columns)
        #print type(x), x
        #print type(y_), y_
        self.TModel.train_on_batch(x, y_)
        
    def training_model(self):
        return self.TModel
        
    def predict_on_batch(self, *params, **args):
        return self.QModel.predict_on_batch(*params, **args)
        
    def __call__(self, *params, **kv):
        return self.QModel(*params, **kv)
        
    def training_data(self, *params):
        raise NotImplementedError
        
    def create_trainig_model(self, qmodel, gamma, *params, **args):
        raise NotImplementedError
