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
    model.compile(Adagrad(lr=1e-3), ["mse"])
    return model

def top_model(inp_width, out_width):
    
    inp = Input((inp_width,), name="top_model_input")
    
    dense1 = Dense(inp_width*20, activation="softplus")(inp)
    dense2 = Dense(out_width*20, activation="softplus")(dense1)
    dense3 = Dense(out_width*20, activation="softplus")(dense2)
    
    out = Dense(out_width, activation="linear")(dense3)
    
    model=Model(inputs=[inp], outputs=[out])
    model.compile(Adagrad(lr=1e-3), ["mse"])
    return model

def fc_model(inp_width, out_width):
    
    inp = Input((inp_width,), name="fc_model_input")
    
    dense1 = Dense(inp_width*10, activation="softplus")(inp)
    dense2 = Dense((inp_width+out_width)*10, activation="softplus")(dense1)
    dense3 = Dense((inp_width+out_width)*10, activation="softplus")(dense2)
    dense4 = Dense(out_width*10, activation="softplus")(dense3)
    out = Dense(out_width, activation="linear")(dense4)
    
    model=Model(inputs=[inp], output=out)
    model.compile(Adagrad(lr=1e-3), ["mse"])
    return model

class QVModel(object):
    
    NAhead = 5
    
    def __init__(self, inp_width, q_width, gamma, v_width = 1):
        self.NQ = q_width
        self.XWidth = inp_width
        self.Gamma = gamma
        base_width = inp_width*10
        #self.BaseModel = base_model(inp_width, base_width)
        self.VModel = self.v_model(inp_width, v_width)
        print("VModel: --------------")
        self.VModel.summary()
        self.QModel = self.q_model(inp_width, q_width)
        print("QModel: --------------")
        self.QModel.summary()
        #self.QTModel = self.q_traning_model(inp_width, q_width, self.QModel)
        #self.VTModel = self.v_traning_model(inp_width, q_width, self.VModel)
        
        
    SKEW = 0.0
    
    @staticmethod
    def v_loss(v_, v):
        diff = v_ - v
        skewed = (diff + QVModel.SKEW*K.abs(diff))/(1.0+QVModel.SKEW)
        return K.mean(K.square(skewed), axis=-1)
        
    def v_model(self, inp_width, out_width):
        inp = Input((inp_width,), name="v_model_input")
        dense1 = Dense(inp_width*10, activation="tanh", bias_initializer="zeros")(inp)
        dense2 = Dense((inp_width+out_width)*10, activation="softplus", bias_initializer="zeros")(dense1)
        dense3 = Dense((inp_width+out_width)*10, activation="softplus", bias_initializer="zeros")(dense2)
        dense4 = Dense(out_width*10, activation="softplus", bias_initializer="zeros")(dense3)
        out = Dense(out_width, activation="linear", bias_initializer="zeros")(dense4)
        model=Model(inputs=[inp], output=out)
        model.compile(Adagrad(lr=1e-3), [self.v_loss])
        return model
        
    def q_model(self, inp_width, out_width):
        inp = Input((inp_width,), name="q_model_input")
    
        dense1 = Dense(inp_width*10, activation="tanh", bias_initializer="zeros")(inp)
        dense2 = Dense((inp_width+out_width)*10, activation="softplus", bias_initializer="zeros")(dense1)
        dense3 = Dense((inp_width+out_width)*10, activation="softplus", bias_initializer="zeros")(dense2)
        dense4 = Dense(out_width*10, activation="softplus", bias_initializer="zeros")(dense3)
        out = Dense(out_width, activation="linear", bias_initializer="zeros")(dense4)
    
        model=Model(inputs=[inp], output=out)
        model.compile(Adagrad(lr=1e-3), ["mse"])
        return model
        
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
        model.compile(Adagrad(lr=1e-3), ["mse", "mse"])
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
        model.compile(Adagrad(lr=1e-3), ["mse"])
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
        
    def calc_v_estimates(self, v1, r, f):
        #
        # v(0) = r(0) + g*v1(0)
        # v(0) = r(0) + g*r(1) + g**2*v1(1)
        # v(0) = r(0) + g*r(1) + g**2*r(2) + g**3*v1(2)
        #
        v1 = v1 * (1-f)
        n = len(v1)
        v1_est = np.empty_like(v1)
        for j in range(n):
            k = min(self.NAhead, n-j)
            v1_est[j] = sum(r[j:j+self.NAhead]*self.GammaPowers[:k]) + v1[j+k-1]*self.Gamma**k
        print("calc_v_estimates:")
        print("  v1: ", v1[-9:])
        print("  r:  ", r[-9:])
        print("  v1_:", v1_est[-9:])
        return v1_est
        
    def train(self, x0, v0, a, r, x1, v1, f, v0_est, verbose):
        
        #v0_ = self.calc_v_estimates(v1, r, f)
        v0_ = v1*self.Gamma + r
        vmetrics = self.VModel.train_on_batch(x0, v0_.reshape((-1,1)))
        if verbose:
            v0_after = self.v_array(x0)
            v0_diff0 = np.mean(np.square(v0-v0_))
            v0_diff1 = np.mean(np.square(v0_after-v0_))
            print("train:")
            print("  f:  ", f[:5], f[-5:])
            print("  r:  ", r[:5], r[-5:])
            print("  v1: ", v1[:5], v1[-5:])
            print("  v0: ", v0[:5], v0[-5:], v0_diff0)
            print("  v0*:", v0_after[:5], v0_after[-5:], v0_diff1)
            print("  v0_:", v0_[:5], v0_[-5:])
        
        q = self.q_array(x0)
        q_ = q.copy()
        improvement = self.Gamma*v1 + r - v0
        for i, (aa, d) in enumerate(zip(a, improvement)):
            q_[i, aa] = q[i,aa] + d*0.1
        if verbose:
            n = len(a)
            q_a = q[np.arange(n), a]
            q__a = q_[np.arange(n), a]
            print("  a:     ", a[:5], a[-5:])
            print("  q[a]:  ", q_a[:5], q_a[-5:])
            print("  imp:   ", improvement[:5], improvement[-5:])
            print("  q_[a]: ", q__a[:5], q__a[-5:])
                
        
        qmetrics = self.QModel.train_on_batch(x0, q_)
        
        return vmetrics, qmetrics
        
        
        
        
