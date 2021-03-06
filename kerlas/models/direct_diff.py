import numpy as np, random

from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
#import keras.backend as K

from .model import RLModel, BellmanDifferential
        
class DirectDiffModel(RLModel):
    
    def __init__(self, qmodel, gamma, *params, **args):
        RLModel.__init__(self, qmodel, gamma, *params, **args)
        self.NActions = qmodel.outputs[0].shape[-1]
        
    def create_trainig_model(self, qmodel, gamma):
        x_shape = qmodel.inputs[0].shape[1:]
        #print "x_shape=", x_shape
        q_shape = qmodel.output.shape[1:]


        final = Input(shape=(1,), name="tmodel_input_final")
        mask = Input(shape=q_shape, name="tmodel_input_mask")
        s0 = Input(shape=x_shape, name="tmodel_input_s0")
        s1 = Input(shape=x_shape, name="tmodel_input_s1")
    
        q0 = qmodel(s0)
        q1 = qmodel(s1)
    
        out = BellmanDifferential(q0, mask, q1, final, gamma)
        model = Model(inputs=[s0, mask, s1, final], outputs=[out])
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def training_data(self, s0, action, s1, reward, final):
        n_actions = self.NActions
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        #print "DirectDiffModel.training_data: returning data"

        #for i, ri in enumerate(reward):
        #        if ri:
        #            print s0[i], action[i], mask[i], s1[i], ri


        return [s0, mask, s1, np.asarray(final, np.float32)], [reward[:,None]]

