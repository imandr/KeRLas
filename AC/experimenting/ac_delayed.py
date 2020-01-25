from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
import numpy as np

class ACDelayedAgent(object):
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99,
                rollup_window = 1,
                layer1_size=1024, layer2_size=512):
        #
        # rollup_winodw:
        # 1:   v(est)(0) = r(0) + g*v(1)
        # 2:   v(est)(0) = r(0) + g*(r(1) + g*v(2))
        # ...
        #
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.critic_skew = critic_skew

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]
        self.all_layers = []

        self.episode_record = []
        self.t = 0
        self.window = rollup_window         
        
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

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        dense1 = Dense(self.fc1_dims, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001),
            bias_regularizer=regularizers.l2(0.0001),
            )(input)
        dense2 = Dense(self.fc2_dims, activation='relu',
            kernel_regularizer=regularizers.l2(0.0001),
            bias_regularizer=regularizers.l2(0.0001),
            )(dense1)
        
        dense3 = Dense(self.fc2_dims//10, activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            )(dense2)
        probs = Dense(self.n_actions, activation='softmax',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            )(dense3)

        dense4 = Dense(self.fc2_dims//10, activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            )(dense2)
        values = Dense(1, activation='linear',
            kernel_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
            )(dense4)
        
        self.all_layers = [dense1, dense2, dense3, dense4, probs, values]

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        delta = Input(shape=[1])
        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        
        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss="mse")
        
        policy = Model(input=[input], output=[probs])
        
        return actor, critic, policy

    def choose_action(self, observation, test = False, epsilon=0.0):
        probs = self.policy.predict(observation[np.newaxis,:])[0]
        if test:
            action = np.argmax(probs)
        else:
            probs = (probs+epsilon)/(1.0+len(probs)*epsilon)
            action = np.random.choice(self.action_space, p=probs)
        return action
        
    def reset(self):
        self.episode_record = []
        self.t = 0

    def value_estimate(self, i, window):
        # min window = 1
        segment = self.episode_record[i:i+window]
        last_row = segment[-1]
        last_state = last_row[3]
        last_final = last_row[4]
        last_value = 0.0 if last_final else self.critic.predict(last_state[np.newaxis,:])[0,0]
        v_est = last_value
        for i in range(window):
            v_est = segment[-1-i][2] + v_est * self.gamma
        return v_est
        
    def learn_row(self, t, window):
        v_est = self.value_estimate(t, window)
        state, action, _, _, _ = self.episode_record[t]
        self.learn_with_estimate(state, v_est, action)

    def learn_with_estimate(self, state, value_estimate, action):
        state = state[np.newaxis,:]
        critic_value = self.critic.predict(state)
        
        delta = value_estimate - critic_value
        
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        actor_metrics = self.actor.fit([state, delta], actions, verbose=0)
        critic_metrics = self.critic.fit(state, critic_value, verbose=0)
        
        return actor_metrics, critic_metrics
        
    def learn(self, state, action, reward, state_, done):
        self.episode_record.append((state, action, reward, state_, done))
        self.t += 1
        if self.t >= self.window:
            self.learn_row(self.t-self.window, self.window)
        if done:
            t0 = self.t - self.window + 1
            for t in (t0, self.t):
                self.learn_row(t, self.window)
            
    def run_episode(self, env, learn = False, test = False, render=False):
        done = False
        observation = env.reset()
        if render:
            env.render()
        score = 0.0
        self.record = []
        self.t = 0
        actor_metrics, critic_metrics = None, None
        while not done:
            action = self.choose_action(observation, test)
            observation_, reward, done, info = env.step(action)
            if render:
                env.render()
            self.record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            if learn:
                self.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        return score, record

