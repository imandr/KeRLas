import numpy as np
from .tools import clone_model, AdditionalUpdatesOptimizer, get_soft_target_model_updates

from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K

from model import RLModel

class DQN(RLModel):
    
    #
    # Implements ideas from Mnih (2015):
    # Online model M fits to target model T:
    #
    #     M(x0,a) -> reward + gamma * max(T(x1)) * (1-final)
    #
    # and M is copied to T periodically
    #   
    
    def __init__(self, qmodel, gamma, *params, **args):
        RLModel.__init__(self, qmodel, gamma, *params, **args)
        self.Gamma = gamma

    def create_trainig_model(self, qmodel, gamma, soft_update=0.001, *params, **args):
        nactions = qmodel.outputs[0].shape[-1]
        xwidth = qmodel.inputs[0].shape[-1].value
        x_shape = (xwidth,)
        q_shape = (nactions,)

        #qmodel.compile(optimizer='sgd', loss='mse')
        target_model = clone_model(qmodel)
        target_model.compile(optimizer='sgd', loss='mse')
        
        
        # build trainable model

        x0 = Input(name="x0", shape=x_shape)
        mask = Input(name="mask", shape=q_shape)

        q0 = qmodel(x0)

        def masked_out(args):
            q0, mask = args
            return K.sum(q0*mask, axis=-1)[:,None]      # make it (n,1) shape
        
        out = Lambda(masked_out)([q0, mask])
        
        trainable = Model(inputs=[x0, mask], outputs=[out])
        updates = get_soft_target_model_updates(target_model, qmodel, soft_update)
        optimizer = AdditionalUpdatesOptimizer(Adam(lr=1e-3), updates)
        
        #print("--- trainable model summary ---")
        #print(trainable.summary())
        
        self.TargetModel = target_model
        trainable.compile(optimizer=optimizer, loss="mse")
        return trainable

    def training_data(self, s0, action, s1, reward, final):
        #print "training_data: s0:", type(s0), s0
        q1t = self.TargetModel.predict_on_batch(s0)
        q1 = np.max(q1t, axis=-1)
        q0_ = (reward + self.Gamma * (1.0-final) * q1)[:,None]
        mask = np.zeros_like(q1t)
        nactions = mask.shape[-1]
        for i in xrange(nactions):
            mask[action==i, i] = 1.0
        #print "training_data: s0:", type(s0), "   mask:", type(mask), "   q0_:", type(q0_)
        return [s0, mask], [q0_]
