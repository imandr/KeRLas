import numpy as np

np.set_printoptions(precision=4, suppress=True)

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Lambda
from keras.optimizers import Adam

class TrainingContext(object):
    
    def __init__(self, brain, in_traning):
        self.Brain = brain
        self.Flag = in_traning
        self.SavedTraining = None
        
    def __enter__(self):
        self.SavedTraining = self.Brain.Training
        self.Brain.setTraining(self.Flag)
        
    def __exit__(self, *params):
        self.Brain.setTraining(self.SavedTraining)

class Brain(object):
    
    def __init__(self, rlmodel, run_policy, training_policies = None):
        self.RLModel = rlmodel
        self.RunPolicy = run_policy
        self.Training = False
        self.TrainingPolicies = training_policies or [run_policy]
        self.TrainingPolicyIndex = 0
        self.Policy = self.RunPolicy
        
    def training(self, in_traning):
        return TrainingContext(self, in_traning)
        
    def setTraining(self, training):
        self.Training = training
        
    def episodeBegin(self):
        if self.Training:
            self.Policy = self.TrainingPolicies[self.TrainingPolicyIndex]
            #print "Brain.episodeBegin: index, tau=", self.TrainingPolicyIndex, self.Policy.tau
        else:
            self.Policy = self.RunPolicy
            
    def episodeEnd(self):
        pass
            
    def nextTrainingPolicy(self):
        self.TrainingPolicyIndex = (self.TrainingPolicyIndex + 1) % len(self.TrainingPolicies) 
        #print "Brain.nextTrainingPolicy:", self.TrainingPolicyIndex
            
    def q(self, observation):
        return self.RLModel.predict_on_batch([np.array([observation])])[0]
        
    def action(self, observation):
        q = self.q(observation)
        #print "Brain: q=", q
        a = self.Policy(q)
        return a, q

    def training_model(self):
        return self.RLModel.training_model()
        
    def train_on_sample(self, sample):
        return self.RLModel.train_on_sample(sample)
        
    def training_data(self, sample):
        return self.RLModel.training_data(sample)
        
    def formatTrajectory(self, trajectory):
        """ Format trajectory into a list of tuples before they are stored in memory.
            Trajectory is list of (s,a,r,s,d) tuples
        """
        return self.RLModel.formatTrajectory(trajectory)
        
        
        
