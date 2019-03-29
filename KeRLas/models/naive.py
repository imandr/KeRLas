import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
import keras.backend as K

from model import RLModel, BellmanDifferential
        
class NaiveQModel(RLModel):
    
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
        q1i = Input(shape=q_shape, name="tmodel_input_q1i")
        
        q0 = qmodel(s0)
    
        out = BellmanDifferential(q0, mask, q1i, final, gamma)
        model = Model(inputs=[s0, mask, q1i, final], outputs=[out])
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def training_data(self, s0, action, s1, final, reward):
        n_actions = self.NActions
        mask = np.zeros((len(s0), n_actions))
        for i in xrange(n_actions):
            mask[action==i, i] = 1.0
        q1i = self.QModel.predict_on_batch(s1)
        #print "DirectDiffModel.training_data: returning data"
        return [s0, mask, q1i, final], [reward]

