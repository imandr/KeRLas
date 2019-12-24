import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import Adam, Adagrad, Adadelta
#import keras.backend as K

from model import RLModel

class DQN(RLModel):
    
    #
    # Depending on kind, implements 3 different types of network:
    #
    # kind="naive": 
    # model fits to itself:
    #
    #    M(x0,a) -> reward + gamma * max(M(x1)) * (1-final)
    #
    # kind="dqn": 
    # Implements ideas from Mnih (2015):
    # Online model M fits to target model T:
    #
    #     M(x0,a) -> reward + gamma * max(T(x1)) * (1-final)
    #
    # and M is copied to T periodically
    #
    # kind="double"
    # Implememts Hasselt (2015):
    # Online model M fits to target model T:
    #
    #     M(x0,a) -> reward + gamma * (T(x1)[argmax(M(x1))]) * (1-final)
    # 
    # and M is copied to T periodically
    #
    # if "advantage" flag is set, replace top model layer with 2 new layers:
    #
    #   V = Dense(1)(top)
    #   A = Dense(n)(top)
    #   Q[a] = V + A[a] - mean_a(A[a])
    #   
    
    def __init__(self, model, kind="dqn", advantage=False, hard_update_samples = 100000, soft_update = None, gamma=0.99):
        self.Model = model
        self.TrainSamples = 0
        self.TrainSamplesBetweenUpdates = self.TrainSamplesToNextUpdate = hard_update_samples
        self.Gamma = gamma
        self.XWidth = self.Model.inputs[0].shape[-1].value
        self.NActions = self.Model.output.shape[-1].value
        self.Kind = kind
        self.SoftUpdate = soft_update  #0.01
        self.Advantage = advantage

    def compile(self, optimizer, metrics=[]):
        
        x_shape = (self.XWidth,)
        q_shape = (self.NActions,)

        if self.Advantage:
            second_layer = self.Model.layers[-2]
            advantage = Dense(self.NActions, activation="linear")(second_layer.output)
            value = Dense(1, activation="linear")(second_layer.output)
            def av_layer(args):
                a, v = args
                return v + a - K.mean(a, axis=-1, keepdims = True)
            q = Lambda(av_layer, output_shape=q_shape)([advantage, value])
            self.Model = Model(inputs=self.Model.inputs, output=q)
        
        self.TargetModel = clone_model(self.Model)
        self.TargetModel.compile(optimizer='sgd', loss='mse')
        self.Model.compile(optimizer='sgd', loss='mse')
        
        # build trainable model

        x0 = Input(name="x0", shape=x_shape)
        mask = Input(name='mask', shape=q_shape)

        q0 = self.Model(x0)

        def masked_out(args):
            q0, mask = args
            return K.sum(q0*mask, axis=-1)[:,None]      # make it (n,1) shape
        
        out = Lambda(masked_out)([q0, mask])
        
        trainable = Model(inputs=[x0, mask], output=out)

        if self.SoftUpdate is not None:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.TargetModel, self.Model, self.SoftUpdate)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
        
        print("--- trainable model summary ---")
        print(trainable.summary())
        
        trainable.compile(optimizer=optimizer, loss="mean_squared_error", metrics=metrics)
        self.TrainModel = trainable

    def get_weights(self):
        return (self.Model.get_weights(), self.TargetModel.get_weights())
        
    def blend_weights(self, alpha, weights):
        
        mw, tw = weights
        
        my_mw, my_tw = self.Model.get_weights(), self.TargetModel.get_weights()
        
        assert len(mw) == len(my_mw)
        for my_w, x_w in zip(my_mw, mv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        assert len(tw) == len(my_tw)
        for my_w, x_w in zip(my_tw, tv):
            my_w.flat[:] = my_w.flat + alpha*(x_w.flat-my_w.flat)

        self.set_weights((my_mw, my_tw))
        
    def set_weights(self, weights):
        mw, tw = weights
        self.Model.set_weights(mw)
        self.TargetModel.set_weights(tw)

        
    def compute(self, batch):
        return self.Model.predict_on_batch(batch)
        
    def train(self, sample, batch_size):
        # samples is list of tuples:
        # (last_observation, action, reward, new_observation, final, valid_actions, info)
        
        #print "samples:"
        #for s in samples:
        #    print s

        metrics = None
        
        for j in range(0, len(sample), batch_size):
            #print sample[j:j+batch_size]
            batches = zip(*sample[j:j+batch_size])
            batch_len = len(batches[0])
            if batch_len < batch_size:  break
        
            state0_batch = format_batch(batches[0])
            action_batch = np.array(batches[1])
            reward_batch = np.array(batches[2])
            state1_batch = format_batch(np.array(batches[3]))
            final_state1_batch = np.asarray(batches[4], dtype=np.float32)
            mask_batch = np.zeros((batch_len, self.NActions), dtype=np.float32)
            irange = np.arange(batch_len)
            mask_batch[irange, action_batch] = 1.0
            
            if self.Kind == "double":
                q1t_batch = self.TargetModel.predict_on_batch(state1_batch)
                q1m_batch = self.Model.predict_on_batch(state1_batch)
                inx = np.argmax(q1m_batch, axis=-1)
                q1_batch = q1t_batch[irange, inx]
            elif self.Kind == "dqn":
                q1t_batch = self.TargetModel.predict_on_batch(state1_batch)
                q1_batch = np.max(q1t_batch, axis=-1)
            else:   # naive
                q1m_batch = self.Model.predict_on_batch(state1_batch)
                q1_batch = np.max(q1m_batch, axis=-1)
                
            #print "q1:", q1_batch
            #print "reward:", reward_batch
            #print "final:", final_state1_batch
            #print "action:", action_batch
            #print "mask:", mask_batch
                
            q_ = (reward_batch + self.Gamma * (1.0-final_state1_batch) * q1_batch)[:,None]
            #print "q_:", q_
            
            #qvalues_calculated = self.Model.predict_on_batch(state0_batch)
        
            #print "batch (%d):" % (batch_len,)
            #for v0, v1, f, m, r, qc in zip(state0_batch, state1_batch, final_state1_batch, mask_batch, 
            #        reward_batch, qvalues_calculated):
            #    print v0[:5], v1[:5], f, m, r, qc

            t0 = time.time()
            #print state0_batch.shape, mask_batch.shape, q_.shape
            metrics = self.TrainModel.train_on_batch([state0_batch, mask_batch], q_)
            #print time.time() - t0
            
            if self.SoftUpdate is None:
                self.TrainSamples += batch_len
                self.TrainSamplesToNextUpdate -= batch_len
            
                if self.TrainSamplesToNextUpdate <= 0:
                    self.TargetModel.set_weights(self.Model.get_weights())
                    print "======== Target network updated ========"
                    self.TrainSamplesToNextUpdate = self.TrainSamplesBetweenUpdates
                    
        return metrics[0]
        
