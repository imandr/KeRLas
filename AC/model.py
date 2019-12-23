import numpy as np, random

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta

import keras.backend as K
import numpy as np

def base_model(inp_width, out_width):
    
    inp = Input((inp_width,), name="base_model_input")
    
    dense1 = Dense(inp_width*20, activation="tanh")(inp)
    dense3 = Dense(out_width*20, activation="softplus")(dense1)
    
    out = Dense(out_width, activation="tanh")(dense3)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model

def top_model(inp_width, out_width):
    
    inp = Input((inp_width,), name="top_model_input")
    
    dense1 = Dense(inp_width*20, activation="softplus")(inp)
    dense2 = Dense(out_width*20, activation="softplus")(dense1)
    dense3 = Dense(out_width*20, activation="softplus")(dense2)
    
    out = Dense(out_width, activation="linear")(dense3)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adam(lr=1e-3), ["mse"])
    return model

def fc_model(inp_width, out_width):
    
    inp = Input((inp_width,), name="fc_model_input")
    
    dense1 = Dense(inp_width*10, activation="softplus")(inp)
    dense2 = Dense((inp_width+out_width)*10, activation="softplus")(dense1)
    dense3 = Dense((inp_width+out_width)*10, activation="softplus")(dense2)
    dense4 = Dense(out_width*10, activation="softplus")(dense3)
    out = Dense(out_width, activation="linear")(dense4)
    
    model=Model(inputs=[inp], output=out)
    model.compile(Adam(lr=1e-3), ["mse"])
    return model

class QVModel(object):
    
    def __init__(self, inp_width, q_width, gamma, v_width = 1):
        self.NQ = q_width
        self.XWidth = inp_width
        self.Gamma = gamma
        base_width = inp_width*10
        self.BaseModel = base_model(inp_width, base_width)
        self.VModel = self.v_model(inp_width, base_width, self.BaseModel, v_width)
        self.QModel = self.q_model(inp_width, base_width, self.BaseModel, q_width)
        self.QTModel = self.q_traning_model(inp_width, q_width, self.QModel)
        #self.VTModel = self.v_traning_model(inp_width, q_width, self.VModel)
        
    def v_model(self, inp_width, base_width, base_model, v_width):
        return fc_model(inp_width, v_width)
        
    def q_model(self, inp_width, base_width, base_model, v_width):
        return fc_model(inp_width, v_width)
        
    def q_traning_model(self, inp_width, q_width, qmodel):
        x = Input((inp_width,), name="qt_x")
        mask = Input((q_width,), name="qt_mask")
        
        def qmask(args):
            q, mask = args
            return K.sum(q*mask, axis=-1)[:,None]
            
        def sumx(q):
            return K.sum(q, axis=-1)[:, None]*0.01
            
        q = qmodel(x)
        q_masked = Lambda(qmask, name="qmasked")([q, mask])
        q_sum = Lambda(sumx)(q)
        model = Model(inputs=[x, mask], outputs=[q_masked, q_sum])
        model.compile(Adam(lr=1e-3), ["mse", "mse"])
        print("Q training model summary:---")
        #model.summary()
        return model
        
    def v_traning_model(self, inp_width, q_width, vmodel):
        x0 = Input((inp_width,), name="vt_x0")
        x1 = Input((inp_width,), name="vt_x1")
        f = Input((1,), name="vt_f")
        
        def v_differential(args):
            v0, v1, f = args
            return v0-self.Gamma*v1*(1.0-f)

        v0 = vmodel(x0)
        v1 = vmodel(x1)
        diff = Lambda(v_differential)([v0, v1, f])
        model = Model(inputs=[x0, x1, f], output=diff)
        model.compile(Adam(lr=1e-3), ["mse"])
        return model
        
    def q(self, x):
        return self.q_array(x.reshape((1,-1)))[0]
        
    def q_array(self, x):
        assert isinstance(x, np.ndarray) and len(x.shape) > 1
        return self.QModel.predict_on_batch(x)
        
    def v(self, x):
        return self.v_array(x.reshape((1,-1)))[0]
        
    def v_array(self, x):
        assert isinstance(x, np.ndarray) and len(x.shape) > 1
        return self.VModel.predict_on_batch(x)[:,0]
        
    def __train_q(self, x0, a, error):
        n = len(a)
        amask = np.zeros((n, self.NQ))
        for i in range(self.NQ):
            amask[a==i, i] = 1.0
        #print("a:", a,"  amask:",amask)
        q0 = self.q_array(x0)
        q0_masked = np.sum(q0*amask, axis=-1).reshape((-1,1))
        zeros = np.zeros((n,1))
        
        #print("train_q: a:", a, "    error:",error)
        
        metrics = self.QTModel.train_on_batch([x0, amask], [q0_masked + error.reshape((-1,1)), zeros])
        return metrics
        
    def train_q(self, x0, a, error):
        n = len(a)
        q = self.q_array(x0)
        for i, (aa, e) in enumerate(zip(a, error)):
            q[i, aa] += e
        metrics = self.QModel.train_on_batch(x0, q)
        return metrics
        
    def train_q(self, x0, a, error):
        n = len(a)
        q = self.q_array(x0)
        for i, (aa, e) in enumerate(zip(a, error)):
            q[i, aa] += e
        metrics = self.QModel.train_on_batch(x0, q)
        return metrics
        
    def train_v(self, x0, v1):
        #print ("train_v: x0:", x0,"  v1:", v1)
        metrics = self.VModel.train_on_batch(x0, v1.reshape((-1,1)))
        return metrics
        
    def train(self, x0, v0, a, r, x1, v1, f):
        v0_ = v1*self.Gamma + r
        vmetrics = self.VModel.train_on_batch(x0, v0_.reshape((-1,1)))
        
        q_ = self.q_array(x0)
        improvement = v1 + r - v0
        for i, (aa, d) in enumerate(zip(a, improvement)):
            q_[i, aa] = d
        qmetrics = self.QModel.train_on_batch(x0, q_)
        
        return vmetrics, qmetrics
        
        
        
        
