from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,
                 layer1_size=1024, layer2_size=512, input_dims=8):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        tau = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        #v1 = Dense(self.fc2_dims, activation='softplus')(dense2)
        values = Dense(1, activation='linear')(dense2)

        #v2 = Dense(self.fc2_dims, activation='softplus')(dense2)
        pre_probs = Dense(self.n_actions, activation='linear')(dense2)
        
        def tau_softmax(args):
            pp, t = args
            return K.softmax(pp*t)
        
        probs = Lambda(tau_softmax)([pre_probs, tau])

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        policy = Model(inputs=[input, tau], outputs=[probs])
        
        actor = Model(inputs=[input, tau, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        
        critic = Model(inputs=[input], outputs=[values])
        
        def skewed_mse(y_, y):
            skew = 0.0
            d = y - y_
            skewed = (d+skew*K.abs(d))/(1.0+skew)
            return K.mean(K.square(skewed))
            
        critic.compile(optimizer=Adam(lr=self.beta), loss=skewed_mse)
        
        return actor, critic, policy

    def choose_action(self, observation, tau=1.0):
        state = observation[np.newaxis, :]
        tau = np.ones((1,1)) * tau
        probabilities = self.policy.predict([state, tau])[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done, tau=1.0):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        tau = np.ones((len(state),))*tau
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, tau, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)
        
    def learn_batches(self, mb_size, state, action, reward, state_, done, tau=1.0, shuffle=True):
        # make sure data is np arrays
        state = np.array(state)
        tau = np.ones((len(state),1))*tau
        action = np.array(action)
        reward = np.array(reward)
        state_ = np.array(state_)
        done = np.array(done, dtype=np.int)
        #print("learn_batches: inputs:", state.shape, action.shape, reward.shape, state_.shape, done.shape)
        
        n = len(state)
        
        critic_value = self.critic.predict(state)[:,0]
        critic_value_ = self.critic.predict(state_)[:,0]

        target = reward + self.gamma*critic_value_*(1-done)
        delta =  target - critic_value
        #print("learn_batches: delta:", delta.shape, delta)

        actions = np.zeros([n, self.n_actions])
        actions[np.arange(n), action] = 1

        target = target.reshape((-1,1))
        delta = delta.reshape((-1,1))
        #print("learn_batches: state:", state.shape, "  tau:", tau.shape, "  delta:",delta.shape)
        actor_metrics = self.actor.fit([state, tau, delta], actions, batch_size=mb_size, verbose=0, shuffle=shuffle)

        critic_metrics = self.critic.fit(state, target, batch_size=mb_size, verbose=0, shuffle=shuffle)
        
        return actor_metrics, critic_metrics
        
