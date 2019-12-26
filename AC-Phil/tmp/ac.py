from keras import backend as K
from keras.layers import Dense, Input
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
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        
        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss="mse")
        
        policy = Model(inputs=[input], outputs=[probs])
        
        return actor, critic, policy

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)

    def learn_batches(self, mb_size, state, action, reward, state_, done, tau=1.0, shuffle=True):
        # make sure data is np arrays
        state = np.array(state)
        #tau = np.ones((len(state),1))*tau
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
        #print("learn_batches: state:", state.shape, "  delta:", delta.shape, "  actions:", actions.shape, "  target:", target.shape)
        
        for i in range(0, n, mb_size):
            actor_metrics = self.actor.train_on_batch([state[i:i+mb_size], delta[i:i+mb_size]], actions[i:i+mb_size])
            critic_metrics = self.critic.train_on_batch(state[i:i+mb_size], target[i:i+mb_size])
        
        return actor_metrics, critic_metrics
