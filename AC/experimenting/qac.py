from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import math

class QACAgent(object):
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99,
                critic_skew = 0.0, qscale = 1.0,
                layer1_size=1024, layer2_size=512):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.critic_skew = critic_skew
        self.QScale = qscale

        self.value, self.advantage, self.policy = self.build_value_advantage_network()

        self.action_space = [i for i in range(n_actions)]
        self.all_layers = []
        
        self.q2_ma = None
        self.q2_ma_alpha = 0.01
        
    def save(self, path):
        weights = {}
        for i, l in enumerate(self.all_layers):
            for j, w in l.get_weights():
                weights["w_%d_%d" % (i, j)] = w
        np.savez(path, **weights)
        
    def load(self, path):
        data = np.load(path)
        for i, l in enumerate(self.all_layers):
            weights = [data["w_%d_%d" % (i, j)] for j, _ in enumerate(l.get_weights())]
            l.set_weights(weights)

    def build_value_advantage_network(self):
        input = Input(shape=(self.input_dims,))
        action_mask = Input(shape=(self.n_actions,))
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        
        dense3 = Dense(self.fc2_dims//10, activation='relu')(dense2)
        advantage = Dense(self.n_actions, activation='linear')(dense3)

        #
        # Policy model
        #

        policy_model = Model(input=[input], output=[advantage])
        
        #
        # advantage training
        #
        
        def masked_advantage(args):
            mask, adv = args
            return K.reshape(K.sum(mask*adv, axis = -1), (-1,1))

        masked_advantage = Lambda(masked_advantage)([action_mask, advantage])
        
        advatnage_model = Model(inputs=[input, action_mask], outputs=[masked_advantage])
        advatnage_model.compile(optimizer=Adam(lr=self.alpha), loss="mse")

        #
        # value model
        #

        dense4 = Dense(self.fc2_dims//10, activation='relu')(dense2)
        values = Dense(1, activation='linear')(dense4)
        values_model = Model(input=[input], output=[values])

        def skewed_loss(y_, y):
            delta = y_ - y
            delta = (delta + K.abs(delta)*self.critic_skew)/(1.0+self.critic_skew)
            return K.mean(delta*delta, axis=-1)

        values_model.compile(optimizer=Adam(lr=self.beta), loss=skewed_loss)
        
        
        self.all_layers = [dense1, dense2, dense3, dense4, advantage, values]

        
        return values_model, advatnage_model, policy_model

    def choose_action(self, observation, test = False, epsilon=0.0, tau=None):
        if tau is None: tau = self.QScale
        weights = self.policy.predict(observation[np.newaxis,:])[0]
        q2 = np.mean(weights*weights)
        q = np.mean(weights)
        if self.q2_ma is None:  
            self.q2_ma = q2
            self.q_ma = q
        self.q2_ma += self.q2_ma_alpha*(q2-self.q2_ma)
        self.q_ma += self.q2_ma_alpha*(q-self.q_ma)
        s = math.sqrt(self.q2_ma - self.q_ma**2)
        
        if test or tau < 0.0:
            action = np.argmax(weights)
        else:
            weights /= s
            weights = weights - np.max(weights)
            exp_values = np.exp(weights / tau)
            probs = exp_values / np.sum(exp_values)
            probs = (probs+epsilon)/(1.0+len(probs)*epsilon)
            action = np.random.choice(self.action_space, p=probs)
        return action

    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        action = np.array([action])
        reward = np.array([reward])
        done = np.array([done])
        
        return self.learn_batch(state, action, reward, state_, done)

    def learn_batch(self, states, actions, rewards, states_, dones):
        n = len(states)
        values1 = self.value.predict(states_)
        values0 = self.value.predict(states)
        rewards = rewards.reshape((-1, 1))
        dones = dones.reshape((-1, 1))

        #print("learn_batch: states:", states.shape, " actions:", actions.shape, " rewards:", rewards.shape,
        #            " states_:", states_.shape, " dones:", dones.shape)
                    
        #print("             critic_values_:", critic_values_.shape, " critic_values:", critic_values.shape)

        mask = np.zeros((n, self.n_actions))
        mask[np.arange(n), actions] = 1.0

        values_ = rewards + self.gamma*values1*(1.0-dones)
        advatages_ = values_ - values0
        advatages_ = advatages_.reshape((-1,1))
        

        #print("learn_batch: states:", states.shape, " deltas:", deltas.shape, " action_arrays:", action_array.shape,
        #            " tagets:", targets.shape)

        advantage_metrics = self.advantage.train_on_batch([states, mask], advatages_)
        value_metrics = self.value.train_on_batch(states, values_)
        
        return advantage_metrics, value_metrics

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
        
    def run_episode(self, env, learn = False, test = False, render=False):
        done = False
        observation = env.reset()
        if render:
            env.render()
        score = 0.0
        record = []
        actor_metrics, critic_metrics = None, None
        while not done:
            action = self.choose_action(observation, test)
            observation_, reward, done, info = env.step(action)
            if render:
                env.render()
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            if learn:
                self.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        return score, record

