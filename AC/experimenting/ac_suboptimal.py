from keras import backend as K
from keras.layers import Dense, Input, Reshape, LSTM, concatenate
from keras.models import Model
from keras.optimizers import Adam
#from keras import regularizers
import numpy as np

import os, random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ACSuboptimalAgent(object):
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99,
                layer1_size=1024, layer2_size=512):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.all_layers = []
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]
        
    def save(self, path):
        weights = {}
        for i, l in enumerate(self.all_layers):
            #print("save: layer:", l.name, type(l))
            for j, w in enumerate(l.get_weights()):
                weights["w_%d_%d" % (i, j)] = w
        np.savez(path, **weights)
        
    def load(self, path):
        data = np.load(path)
        for i, l in enumerate(self.all_layers):
            weights = [data["w_%d_%d" % (i, j)] for j, _ in enumerate(l.get_weights())]
            l.set_weights(weights)
    
    Activation = "relu"

    def build_actor_critic_network(self):
        input = Input((self.input_dims,))
        dense1 = Dense(self.fc1_dims, activation=self.Activation, name="dense1")(input)
        base = Dense(self.fc2_dims, activation=self.Activation, name="dense2")(dense1)
        
        dense3 = Dense(self.fc2_dims//5, activation=self.Activation, name="dense3")(base)
        probs = Dense(self.n_actions, activation='softmax', name="probs")(dense3)

        dense4 = Dense(self.fc2_dims//5, activation=self.Activation, name="dense4")(base)
        values = Dense(1, activation='linear', name="values")(dense4)
        
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        delta = Input(shape=[1])
        actor = Model(inputs=[input, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        self.all_layers = actor.layers[:]

        critic = Model(inputs=[input], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss="mse")
        
        for l in critic.layers:
            if not l in self.all_layers:
                self.all_layers.append(l)
                
        #for l in self.all_layers:
        #    print("layer", l.name)
        
        policy = Model(inputs=[input], outputs=[probs])
        
        return actor, critic, policy
        
    def reset(self):
        self.actor.reset_states()

    def choose_action(self, observation, test = False, epsilon=0.01):
        probs = self.policy.predict(observation[np.newaxis,:])[0]
        optimal_action = np.argmax(probs)
        if test:
            action = optimal_action
            optimal = True
        else:
            if epsilon > random.random():
                action = random.randint(0, self.n_actions-1)
                optimal = False
            else:
            #probs = (probs+epsilon/len(probs))/(1.0+epsilon)
                action = np.random.choice(self.action_space, p=probs)
                optimal = True
        return action, optimal
        
    def learn(self, state, action, optimal, reward, state_, done):
        #print ("learn: optimal:", optimal)
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        actor_metrics = self.actor.train_on_batch([state, delta], actions)
        critic_metrics = None
        if optimal: critic_metrics = self.critic.train_on_batch(state, target)
        
        return actor_metrics, critic_metrics

    def learn_batch(self, states, actions, optimal, rewards, states_, dones, batch_size = 10):
        n = len(states)
        critic_values_ = self.critic.predict(states_)
        critic_values = self.critic.predict(states)
        rewards = rewards.reshape((-1, 1))
        dones = dones.reshape((-1, 1))

        #print("learn_batch: states:", states.shape, " actions:", actions.shape, " rewards:", rewards.shape,
        #            " states_:", states_.shape, " dones:", dones.shape)
                    
        #print("             critic_values_:", critic_values_.shape, " critic_values:", critic_values.shape)


        targets = rewards + self.gamma*critic_values_*(1.0-dones)
        deltas =  targets - critic_values

        action_array = np.zeros((n, self.n_actions))
        action_array[np.arange(n), actions] = 1

        #print("learn_batch: states:", states.shape, " deltas:", deltas.shape, " action_arrays:", action_array.shape,
        #            " tagets:", targets.shape)

        actor_metrics = self.actor.fit([states, deltas], action_array, verbose = False, batch_size=batch_size, shuffle=True)
        critic_metrics = self.critic.fit(states, targets, verbose = False, batch_size=batch_size, shuffle=True)
        
        return actor_metrics, critic_metrics

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
        
    def run_episode(self, env, learn=False, test=False, render=False, epsilon=0.01):
        done = False
        observation = env.reset()
        self.reset()
        if render:
            env.render()
        score = 0.0
        record = []
        actor_metrics, critic_metrics = None, None
        while not done:
            action, optimal = self.choose_action(observation, test=test, epsilon=epsilon)
            #print("run_episode: obs:", observation, "  action:", action)
            observation_, reward, done, info = env.step(action)
            if render:
                env.render()
            record.append((observation, action, optimal, 
                                reward, observation_, 1.0 if done else 0.0, info))
            if learn:
                self.learn(observation, action, optimal, reward, observation_, done)
            observation = observation_
            score += reward
        return score, record


