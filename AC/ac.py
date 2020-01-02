from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

class Agent(object):
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99,
                 layer1_size=1024, layer2_size=512):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]
        
    def save(self, path):
        actor_weights = self.actor.get_weights()
        weights = {
            "actor_%05d" % (i,):    w for i, w in enumerate(actor_weights)
        }
        for i, w in enumerate(self.critic.layers[-1].get_weights()):
            weights["critic_top_%05d" % (i,)] = w
        np.savez(path, **weights)
        
    def load(self, path):
        actor_weights = []
        data = np.load(path)
        for i, _ in enumerate(self.actor.get_weights()):
            actor_weights.append(data["actor_%05d" % (i,)])
            
        critic_top_weights = []
        for i, _ in enumerate(self.critic.layers[-1].get_weights()):
            critic_top_weights.append(data["critic_top_%05d" % (i,)])
        
        self.actor.set_weights(actor_weights)
        self.critic.layers[-1].set_weights(critic_top_weights)

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu', bias_initializer='zeros')(input)
        dense2 = Dense(self.fc2_dims, activation='relu', bias_initializer='zeros')(dense1)
        probs = Dense(self.n_actions, activation='softmax', bias_initializer='zeros')(dense2)
        values = Dense(1, activation='linear', bias_initializer='zeros')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        actor = Model(input=[input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        
        critic = Model(input=[input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss="mse")
        
        policy = Model(input=[input], output=[probs])
        
        return actor, critic, policy

    def choose_action(self, observation, test = False, epsilon=0.0):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        if test:
            action = np.argmax(probabilities)
        else:
            action = np.random.choice(self.action_space, p=(probabilities+epsilon)/(1.0+len(probabilities)*epsilon))

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

        actor_metrics = self.actor.fit([state, delta], actions, verbose=0)
        critic_metrics = self.critic.fit(state, target, verbose=0)
        
        return actor_metrics, critic_metrics

    def learn_batch(self, states, actions, rewards, states_, dones):
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

        actor_metrics = self.actor.train_on_batch([states, deltas], action_array)
        critic_metrics = self.critic.train_on_batch(states, targets)
        
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
                actor_metrics, critic_metrics = self.learn(observation, action, reward, observation_, done)
                actor_metrics, critic_metrics = actor_metrics.history["loss"], critic_metrics.history["loss"]
            observation = observation_
            score += reward
        return score, record, actor_metrics, critic_metrics

