import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
from model import RLModel, BellmanDifferential

class LateralDiffModel(RLModel):
    
    def __init__(self, qmodel, gamma, weight = 0.5):
        RLModel.__init__(self, qmodel, gamma, weight)
        self.NActions = qmodel.outputs[0].shape[-1]
        
    def create_trainig_model(self, qmodel, gamma, weight):

        x_shape = qmodel.inputs[0].shape[1:]
        #print "x_shape=", x_shape
        q_shape = qmodel.output.shape[1:]


        final = Input(shape=(1,))
        mask = Input(shape=q_shape)
        s0 = Input(shape=x_shape)
        q0i = Input(shape=q_shape)
        s1 = Input(shape=x_shape)
        q1i = Input(shape=q_shape)

        q0 = qmodel(s0)
        q1 = qmodel(s1)

        def combine(args):
            r0, r1, final = args
            return (r0 + weight * (1.0-final) * r1)/(1.0 + weight * (1.0-final))

        r0 = BellmanDifferential(q0, mask, q1i, final, gamma)
        r1 = BellmanDifferential(q0i, mask, q1, final, gamma)
        out = Lambda(combine)([r0, r1, final])
        model = Model(inputs=[s0, q0i, mask, s1, q1i, final], outputs=[out])
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def training_data(self, s0, action, s1, final, reward):
        n_actions = self.NActions
        q0i = self.QModel.predict_on_batch(s0)
        q1i = self.QModel.predict_on_batch(s1)
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        return [s0, q0i, mask, s1, q1i, final], [reward]    
