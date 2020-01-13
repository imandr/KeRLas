from keras import backend as K
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
#from keras import regularizers
import numpy as np

class ACRecurrentAgent(object):
    def __init__(self, input_dims, n_actions, alpha, beta, gamma=0.99,
                layer1_size=1024, layer2_size=512):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.base_dims = layer1_size
        self.recurrent_dims = layer2_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.rec_value = None
        self.prev_rec = None
        self.prev_state = None

        self.build_recurrent_network()
        
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

    def build_recurrent_network(self):
        obs = Input(shape=(self.input_dims,))
        rec_in = Input((self.recurrent_dims,))

        combined = concatenate([obs, rec_in])

        dense1 = Dense(self.fc1_dims, activation=self.Activation, name="dense1")(combined)
        base = Dense(self.base_dims, activation=self.Activation, name="base")(dense1)
        self.BModel = Model(inputs=[obs, rec_in], outputs=[base])

        p_in = Input((self.base_dims,))
        d = Dense(self.base_dims//5, activation=self.Activation, name="dense3")(p_in)
        probs = Dense(self.n_actions, activation='softmax', name="probs")(d)
        self.PModel = Model(inputs=[p_in], outputs=[probs])
        
        v_in = Input((self.base_dims,))
        d = Dense(self.fc2_dims//5, activation=self.Activation, name="dense4")(v_in)
        values = Dense(1, activation='linear', name="values")(d)
        self.VModel = Model(inputs=[v_in], outputs=[values])

        r_in = Input((self.base_dims,))
        d = Dense(self.recurrent_dims, activation=self.Activation, name="dense5")(r_in)
        rec_out = Dense(self.recurrent_dims, activation='softplus', name="rec_out")(d)
        self.RModel = Model(inputs=[r_in], outputs=[rec_out])
        
        #
        # Compute models
        #
        this_obs = Input(shape=(self.input_dims,))
        this_rec = Input((self.recurrent_dims,))
        base = self.BModel([this_obs, this_rec])

        self.ComputeVModel = Model(inputs=[this_obs, this_rec], outputs=[self.VModel(base)])
        self.ComputePModel = Model(inputs=[this_obs, this_rec], outputs=[self.PModel(base), self.RModel(base)])
        
        #
        # Trainable models
        #
        delta = Input(shape=[1])
        prev_obs = Input(shape=(self.input_dims,))
        this_obs = Input(shape=(self.input_dims,))
        prev_rec = Input((self.recurrent_dims,))
        
        prev_base = self.BModel([prev_obs, prev_rec])
        rec_in = self.RModel(prev_base)
        
        tbase = self.BModel([this_obs, rec_in])
        
        #
        # trainable actor
        #
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)
            
        self.TrainableActor = Model(
                    inputs=[prev_obs, prev_rec, this_obs, delta], 
                    outputs=[self.PModel(tbase)])
                    
        self.TrainableActor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        #
        #.trainable critic
        #
        self.TrainableCritic = Model(
            inputs=[prev_obs, prev_rec, this_obs], 
            outputs=[self.VModel(tbase)]
        )
        self.TrainableCritic.compile(optimizer=Adam(lr=self.beta), loss="mse")

        self.all_layers = self.BModel.layers + self.RModel.layers + self.VModel.layers + self.PModel.layers

    def reset(self):
        self.prev_state = None
        self.rec_value = np.zeros((1, self.recurrent_dims))

    def step(self, state, test = False, epsilon=0.0):
        probs, rec_out = self.ComputePModel.predict([state[np.newaxis,:], self.rec_value])
        self.prev_rec = self.rec_value
        self.rec_value = rec_out
        probs = probs[0]
        if test:
            action = np.argmax(probs)
        else:
            probs = (probs+epsilon)/(1.0+len(probs)*epsilon)
            action = np.random.choice(self.action_space, p=probs)
        return action
        
    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.ComputeVModel.predict([state_, self.rec_value])
        critic_value = self.ComputeVModel.predict([state, self.prev_rec])

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        prev_state = state if self.prev_state is None else self.prev_state

        actor_metrics = self.TrainableActor.train_on_batch(
            [prev_state, self.prev_rec, state, delta], 
            actions
        ) 
        critic_metrics = self.TrainableCritic.train_on_batch(
            [prev_state, self.prev_rec, state], 
            target
        )
        self.prev_state = state
        return actor_metrics, critic_metrics

    def learn_batch(self, states, actions, rewards, states_, dones, batch_size = 10):
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
        
    def run_episode(self, env, learn=False, test=False, render=False):
        done = False
        observation = env.reset()
        self.reset()
        if render:
            env.render()
        score = 0.0
        record = []
        actor_metrics, critic_metrics = None, None
        while not done:
            action = self.step(observation, test)
            observation_, reward, done, info = env.step(action)
            if render:
                env.render()
            record.append((observation, action, reward, observation_, 1.0 if done else 0.0, info))
            if learn:
                self.learn(observation, action, reward, observation_, done)
            observation = observation_
            score += reward
        return score, record

